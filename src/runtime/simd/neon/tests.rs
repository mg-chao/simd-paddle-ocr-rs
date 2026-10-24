use super::*;

fn values(count: usize, factor: usize) -> Vec<f32> {
    (0..count)
        .map(|i| ((i * factor % 47) as f32 - 23.) * 0.01331)
        .collect()
}

fn assert_bits(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "element {i}: {actual} != {expected}"
        );
    }
}

#[test]
fn neon_dense_preserves_order_bias_packing_and_attention_tails() {
    for (rows, inner, columns) in [
        (1, 1, 1),
        (1, 3, 15),
        (4, 4, 16),
        (5, 65, 17),
        (7, 127, 47),
        (8, 17, 64),
        (9, 129, 67),
        (12, 128, 128),
    ] {
        let input = values(rows * inner, 13);
        let weights = values(inner * columns, 17);
        let bias = values(columns, 29);
        let packed = PackedMatrix::new(&weights, inner, columns).unwrap();
        for bias in [&[][..], bias.as_slice()] {
            let mut expected = vec![0.; rows * columns];
            for row in 0..rows {
                for column in 0..columns {
                    let mut sum = if bias.is_empty() { 0. } else { bias[column] };
                    for k in 0..inner {
                        sum = input[row * inner + k].mul_add(weights[k * columns + column], sum);
                    }
                    expected[row * columns + column] = sum;
                }
            }
            for packed in [None, Some(&packed)] {
                let mut actual = vec![f32::NAN; expected.len()];
                // SAFETY: NEON is part of the AArch64 target baseline and all
                // buffers are exactly the matrix dimensions declared above.
                unsafe {
                    matmul_bias_store(
                        &input,
                        if packed.is_some() { &[] } else { &weights },
                        packed,
                        bias,
                        &mut actual,
                        rows,
                        inner,
                        columns,
                        &IdentityStore,
                        0,
                    );
                }
                assert_bits(&actual, &expected);
                if bias.is_empty() {
                    unsafe {
                        matmul(&input, &weights, packed, &mut actual, rows, inner, columns);
                    }
                    assert_bits(&actual, &expected);
                }
            }
        }
    }
}

#[test]
fn neon_dense_fused_store_preserves_physical_offsets_and_partial_vectors() {
    struct OffsetStore;
    impl StoreTransform for OffsetStore {
        fn scalar(&self, value: f32, index: usize) -> f32 {
            value + index as f32 / 1024.
        }
        fn vector<const LANES: usize>(
            &self,
            value: Simd<f32, LANES>,
            index: usize,
        ) -> Simd<f32, LANES> {
            value + Simd::from_array(std::array::from_fn(|lane| (index + lane) as f32 / 1024.))
        }
    }
    let (rows, inner, columns, offset) = (5, 7, 19, 41);
    let input = values(rows * inner, 13);
    let weights = values(inner * columns, 17);
    let bias = values(columns, 29);
    let mut actual = vec![f32::NAN; rows * columns];
    unsafe {
        matmul_bias_store(
            &input,
            &weights,
            None,
            &bias,
            &mut actual,
            rows,
            inner,
            columns,
            &OffsetStore,
            offset,
        );
    }
    for row in 0..rows {
        for column in 0..columns {
            let mut sum = bias[column];
            for k in 0..inner {
                sum = input[row * inner + k].mul_add(weights[k * columns + column], sum);
            }
            let expected = OffsetStore.scalar(sum, offset + row * columns + column);
            assert_eq!(actual[row * columns + column].to_bits(), expected.to_bits());
        }
    }
}

#[test]
fn neon_dynamic_attention_retains_broadcast_batch_offsets() {
    let (_, kind) = crate::runtime::bound::bind_matmul(
        crate::tensor::Shape::new(&[2, 1, 5, 7]).unwrap(),
        crate::tensor::Shape::new(&[1, 3, 7, 33]).unwrap(),
    )
    .unwrap();
    let input = values(2 * 5 * 7, 13);
    let weights = values(3 * 7 * 33, 17);
    let mut output = vec![f32::NAN; 2 * 3 * 5 * 33];
    crate::runtime::kernels::execute_optimized(
        &kind,
        &[crate::model::TensorId(0), crate::model::TensorId(1)],
        |id| if id.0 == 0 { &input } else { &weights },
        &mut output,
        crate::runtime::simd::Backend::NeonFma,
        None,
        &crate::cpu::CpuContext::serial(),
        1,
    );
    for a in 0..2 {
        for b in 0..3 {
            for row in 0..5 {
                for column in 0..33 {
                    let mut expected = 0.;
                    for k in 0..7 {
                        expected = input[(a * 5 + row) * 7 + k]
                            .mul_add(weights[(b * 7 + k) * 33 + column], expected);
                    }
                    assert_eq!(
                        output[((a * 3 + b) * 5 + row) * 33 + column].to_bits(),
                        expected.to_bits()
                    );
                }
            }
        }
    }
}

#[test]
fn neon_compact_matches_order_across_cache_panels_row_groups_and_tails() {
    for (rows, inner, columns) in [(1, 3, 3), (5, 7, 17), (129, 65, 1041)] {
        let input = values(rows * inner, 13);
        let weights = values(inner * columns, 17);
        let bias = values(columns, 29);
        let packed = PackedMatrix::new(&weights, inner, columns).unwrap();
        for packed in [None, Some(&packed)] {
            for bias in [&[][..], bias.as_slice()] {
                let mut indices = vec![usize::MAX; rows];
                let mut scores = vec![f32::NAN; rows];
                unsafe {
                    compact_argmax(
                        &input,
                        &weights,
                        bias,
                        packed,
                        &mut indices,
                        &mut scores,
                        inner,
                        columns,
                    )
                    .unwrap();
                }
                for row in 0..rows {
                    let mut best = 0;
                    let mut maximum = f32::NEG_INFINITY;
                    for column in 0..columns {
                        let packed_tile = packed.is_some() && column < columns / 16 * 16;
                        let mut sum = if packed_tile || bias.is_empty() {
                            0.
                        } else {
                            bias[column]
                        };
                        for k in 0..inner {
                            let x = input[row * inner + k];
                            let w = weights[k * columns + column];
                            sum = if packed_tile {
                                x.mul_add(w, sum)
                            } else {
                                sum + x * w
                            };
                        }
                        if packed_tile && !bias.is_empty() {
                            sum += bias[column];
                        }
                        if sum > maximum {
                            best = column;
                            maximum = sum;
                        }
                    }
                    assert_eq!(
                        indices[row],
                        best,
                        "{rows}/{inner}/{columns} row={row} packed={}",
                        packed.is_some()
                    );
                    assert_eq!(scores[row].to_bits(), maximum.to_bits());
                }
            }
        }
    }
}

#[test]
fn neon_compact_bias_epilogue_and_unfused_tail_keep_first_ties() {
    let (rows, inner, columns) = (5, 65, 17);
    let input = vec![1.; rows * inner];
    let mut weights = vec![0.; inner * columns];
    let mut bias = vec![-2.; columns];
    for column in [0, 15, 16] {
        weights[column] = 1e8;
        weights[columns + column] = -1e8;
        bias[column] = 1.;
    }
    bias[1] = 0.5;
    let packed = PackedMatrix::new(&weights, inner, columns).unwrap();
    let mut indices = vec![usize::MAX; rows];
    let mut scores = vec![f32::NAN; rows];
    unsafe {
        compact_argmax(
            &input,
            &weights,
            &bias,
            Some(&packed),
            &mut indices,
            &mut scores,
            inner,
            columns,
        )
        .unwrap();
    }
    assert_eq!(indices, vec![0; rows]);
    assert_eq!(scores, vec![1.; rows]);

    let mut input = vec![0.; rows * inner];
    for row in 0..rows {
        input[row * inner] = 1. + f32::EPSILON;
    }
    let mut weights = vec![0.; inner * columns];
    let mut bias = vec![-2.; columns];
    bias[0] = 0.;
    weights[16] = 1. + f32::EPSILON;
    bias[16] = -(1. + 2. * f32::EPSILON);
    assert!(input[0].mul_add(weights[16], bias[16]) > 0.);
    let packed = PackedMatrix::new(&weights, inner, columns).unwrap();
    unsafe {
        compact_argmax(
            &input,
            &weights,
            &bias,
            Some(&packed),
            &mut indices,
            &mut scores,
            inner,
            columns,
        )
        .unwrap();
    }
    assert_eq!(indices, vec![0; rows]);
    assert_eq!(
        scores.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        vec![0f32.to_bits(); rows]
    );
}

#[test]
fn neon_compact_keeps_signed_zero_ties_and_rejects_nonwinning_nonfinite() {
    let (rows, inner, columns) = (5, 65, 1041);
    let input = vec![-0.; rows * inner];
    let weights = vec![0.; inner * columns];
    let packed = PackedMatrix::new(&weights, inner, columns).unwrap();
    let mut bias = vec![-2.; columns];
    bias[4] = -0.;
    bias[1024] = 0.;
    bias[1040] = 0.;
    let mut indices = vec![usize::MAX; rows];
    let mut scores = vec![f32::NAN; rows];
    unsafe {
        compact_argmax(
            &input,
            &weights,
            &bias,
            Some(&packed),
            &mut indices,
            &mut scores,
            inner,
            columns,
        )
        .unwrap();
    }
    assert_eq!(indices, vec![4; rows]);
    assert_eq!(
        scores.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        vec![0f32.to_bits(); rows]
    );
    for column in [3, 1025, 1040] {
        for invalid in [f32::NAN, f32::NEG_INFINITY, f32::INFINITY] {
            let mut bias = bias.clone();
            bias[column] = invalid;
            assert!(
                unsafe {
                    compact_argmax(
                        &input,
                        &weights,
                        &bias,
                        Some(&packed),
                        &mut indices,
                        &mut scores,
                        inner,
                        columns,
                    )
                }
                .is_err()
            );
        }
    }
}

#[test]
fn neon_compact_rejects_mismatched_buffers() {
    let mut indices = [0; 4];
    let mut scores = [0.; 4];
    assert!(
        unsafe {
            compact_argmax(
                &[0.; 8],
                &[0.; 32],
                &[],
                None,
                &mut indices,
                &mut scores,
                3,
                16,
            )
        }
        .is_err()
    );
    let panel = PackedMatrix::new(&[0.; 16], 1, 16).unwrap();
    assert!(
        unsafe {
            compact_argmax(
                &[0.; 8],
                &[0.; 32],
                &[],
                Some(&panel),
                &mut indices,
                &mut scores,
                2,
                16,
            )
        }
        .is_err()
    );
}

/// Fixed release diagnostic: control and candidate each receive two shapes,
/// one warm-up and three single calls. Use an external five-second timeout.
#[test]
#[ignore = "local ARM64 release diagnostic: two unpacked attention tails, one warm-up and three single calls; external timeout 5s"]
#[allow(clippy::assertions_on_constants)] // Keep the ignored runtime guard buildable in debug.
fn arm64_unpacked_attention_fixed_diagnostic() {
    use std::{
        hint::black_box,
        time::{Duration, Instant},
    };
    assert!(
        !cfg!(debug_assertions),
        "run this diagnostic in release mode"
    );
    let started = Instant::now();
    let check_budget = || {
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "fixed ARM64 attention diagnostic exceeded five seconds"
        )
    };
    for (shape_index, (rows, inner, columns)) in [(32, 64, 8), (64, 64, 63)].into_iter().enumerate()
    {
        check_budget();
        let input = values(rows * inner, 13);
        let weights = values(inner * columns, 17);
        let mut output = vec![f32::NAN; rows * columns];
        let mut spots = Vec::new();
        for row in [0, rows / 2, rows - 1] {
            for column in [0, columns / 2, columns - 1] {
                let mut expected = 0.;
                for k in 0..inner {
                    expected =
                        input[row * inner + k].mul_add(weights[k * columns + column], expected);
                }
                spots.push((row * columns + column, expected.to_bits()));
            }
        }
        let check_spots = |output: &[f32]| {
            for &(index, expected) in &spots {
                assert_eq!(
                    output[index].to_bits(),
                    expected,
                    "shape={rows}x{inner}x{columns} index={index}"
                );
            }
        };
        let call = |output: &mut [f32]| {
            // SAFETY: the module exists only on ARM64. Every buffer spans the
            // complete declared matrix, and the native entry validates it.
            unsafe {
                matmul(
                    black_box(&input),
                    black_box(&weights),
                    None,
                    black_box(output),
                    rows,
                    inner,
                    columns,
                );
            }
        };
        check_budget();
        call(&mut output);
        check_spots(&output);
        check_budget();
        let mut samples_ms = Vec::with_capacity(3);
        for _ in 0..3 {
            output.fill(f32::NAN);
            check_budget();
            let begin = Instant::now();
            call(&mut output);
            samples_ms.push(begin.elapsed().as_secs_f64() * 1000.);
            check_spots(&output);
            black_box(&output);
            check_budget();
        }
        println!(
            "{}",
            serde_json::json!({
                "diagnostic":"arm64_unpacked_attention_fixed", "backend":"neon_fma",
                "shape_index":shape_index, "rows":rows, "inner":inner, "columns":columns,
                "packed":false, "threads":1, "warmups":1, "calls_per_sample":1,
                "samples_ms":samples_ms, "exact_oracle_spots":spots.len(), "oracle_fma":true,
                "budget_seconds":5,
            })
        );
    }
}

#[test]
fn neon_partial_load_preserves_payload_bits_and_zero_padding_for_all_remaining_lanes() {
    let source = [
        -0.,
        f32::from_bits(0x7fc0_1234),
        f32::from_bits(1),
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.,
        f32::from_bits(0x7fa0_5678),
        -1.,
    ];
    for end in 0..=source.len() {
        let values = &source[..end];
        for column in (0..=source.len() + 2).chain([usize::MAX]) {
            // SAFETY: NEON is part of the ARM64 baseline. The helper must
            // avoid loading inactive lanes, including a wholly absent vector.
            let actual = as_simd(unsafe { partial_load(values, column) }).to_array();
            let expected: [f32; 4] = std::array::from_fn(|lane| {
                column
                    .checked_add(lane)
                    .and_then(|index| values.get(index))
                    .copied()
                    .unwrap_or(0.)
            });
            assert_eq!(
                actual.map(f32::to_bits),
                expected.map(f32::to_bits),
                "length={end} column={column}"
            );
        }
    }
}

#[test]
fn neon_dense_preserves_every_column_tail_across_selected_lane_and_scalar_k() {
    for rows in [1, 4, 5] {
        for inner in [3, 4, 7] {
            let input = values(rows * inner, 13);
            for columns in 1..=33 {
                let weights = values(inner * columns, 17);
                let bias = values(columns, 29);
                let packed = PackedMatrix::new(&weights, inner, columns).unwrap();
                for bias in [&[][..], bias.as_slice()] {
                    let mut expected = vec![0.; rows * columns];
                    for row in 0..rows {
                        for column in 0..columns {
                            let mut sum = bias.get(column).copied().unwrap_or(0.);
                            for k in 0..inner {
                                sum = input[row * inner + k]
                                    .mul_add(weights[k * columns + column], sum);
                            }
                            expected[row * columns + column] = sum;
                        }
                    }
                    for packed in [None, Some(&packed)] {
                        let mut actual = vec![f32::NAN; expected.len()];
                        // SAFETY: all native buffers span the exact declared
                        // dimensions; packed panels retain their padded lanes.
                        unsafe {
                            matmul_bias_store(
                                &input,
                                &weights,
                                packed,
                                bias,
                                &mut actual,
                                rows,
                                inner,
                                columns,
                                &IdentityStore,
                                0,
                            );
                        }
                        assert_bits(&actual, &expected);
                    }
                }
            }
        }
    }
}
