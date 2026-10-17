use super::*;
#[cfg(target_arch = "aarch64")]
#[path = "neon_bench.rs"]
mod neon_bench;

fn backends() -> Vec<Backend> {
    let mut available = vec![Backend::Scalar, Backend::Portable4];
    #[cfg(target_arch = "aarch64")]
    available.push(Backend::NeonFma);
    for (enabled, backend) in [
        (simd::avx_available(), Backend::Avx),
        (simd::avx_fma_available(), Backend::AvxFma),
        (simd::avx2_fma_available(), Backend::Avx2Fma),
        (simd::avx512_available(), Backend::Avx512),
    ] {
        if enabled {
            available.push(backend);
        }
    }
    available
}

pub(super) fn values(length: usize, seed: usize) -> Vec<f32> {
    (0..length)
        .map(|i| ((i * 37 + seed * 19) % 113) as f32 / 71. - 0.75)
        .collect()
}

#[test]
fn sme_crossover_preserves_platform_backend_and_small_shape_fallbacks() {
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    let available = std::arch::is_aarch64_feature_detected!("sme");
    #[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
    let available = false;
    for (rows, inner, columns, expected) in [
        (31, 384, 384, false),
        (32, 384, 384, true),
        (31, 96, 192, false),
        (32, 96, 192, true),
        (384, 95, 192, false),
        (384, 96, 191, false),
        (384, 96, 192, true),
        (384, 192, 96, true),
        (192, 191, 384, true),
        (192, 192, 383, true),
        (192, 192, 384, true),
        (192, 384, 192, true),
        (1920, 256, 512, true),
        (1920, 512, 256, true),
        (1920, 128, 256, true),
        (1920, 256, 128, true),
        (1920, 64, 128, false),
    ] {
        for backend in backends() {
            let selected = available && backend == Backend::NeonFma && expected;
            assert_eq!(
                sme_scratch_elements(backend, rows, inner, columns),
                if selected { 64 * inner } else { 0 },
                "{backend:?} {rows}x{inner}x{columns}"
            );
        }
    }
}

#[test]
fn inline_avx2_selection_keeps_platform_isa_and_shape_boundaries() {
    let enabled = cfg!(all(target_arch = "x86_64", target_os = "windows"));
    for (inner, columns, expected) in [
        (95, 128, false),
        (96, 127, false),
        (96, 128, true),
        (192, 384, true),
        (193, 384, false),
        (256, 512, false),
        (257, 512, false),
    ] {
        assert_eq!(
            inline_avx2_eligible(Backend::Avx2Fma, inner, columns),
            enabled && expected
        );
        for backend in [
            Backend::Scalar,
            Backend::Portable4,
            Backend::NeonFma,
            Backend::Avx,
            Backend::AvxFma,
            Backend::Avx512,
        ] {
            assert!(!inline_avx2_eligible(backend, inner, columns));
        }
    }
}

pub(super) fn assert_bits(actual: &[f32], expected: &[f32], context: &str) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{context}: index {i}, actual={a} expected={b}"
        );
    }
}

pub(super) fn oracle(
    backend: Backend,
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    inner: usize,
    columns: usize,
) -> Vec<f32> {
    let mut output = vec![0.; input.len() / inner * columns];
    for (row, out) in input
        .chunks_exact(inner)
        .zip(output.chunks_exact_mut(columns))
    {
        for (column, out) in out.iter_mut().enumerate() {
            let mut sum = bias.get(column).copied().unwrap_or(0.);
            for (k, &value) in row.iter().enumerate() {
                sum = if backend.uses_fma() {
                    value.mul_add(weights[k * columns + column], sum)
                } else {
                    value * weights[k * columns + column] + sum
                };
            }
            *out = sum;
        }
    }
    output
}

#[test]
fn direct_six_rows_match_ordered_oracle_for_row_channel_and_reduction_tails() {
    for (rows, inner, columns) in [
        (1, 1, 1),
        (2, 7, 8),
        (3, 17, 9),
        (4, 7, 16),
        (5, 17, 17),
        (6, 17, 32),
        (7, 19, 33),
        (11, 65, 17),
        (47, 7, 17),
        (48, 17, 16),
        (49, 17, 33),
        (13, 513, 17),
        (7, 1024, 32),
    ] {
        let input = values(rows * inner, 3);
        let weights = values(inner * columns, 5);
        let packed = PackedMatrix::new(&weights, inner, columns).unwrap();
        let bias = values(columns, 7);
        for bias in [bias.as_slice(), &[]] {
            for backend in backends() {
                let expected = oracle(backend, &input, &weights, bias, inner, columns);
                let mut actual = vec![f32::NAN; rows * columns];
                execute(
                    backend,
                    &input,
                    bias,
                    &packed,
                    &mut actual,
                    0,
                    inner,
                    columns,
                    &IdentityStore,
                );
                assert_bits(
                    &actual,
                    &expected,
                    &format!(
                        "{backend:?} rows={rows} inner={inner} columns={columns} bias={}",
                        !bias.is_empty()
                    ),
                );
            }
        }
    }
}

struct IndexStore;
impl StoreTransform for IndexStore {
    fn scalar(&self, value: f32, index: usize) -> f32 {
        value + (index % 97) as f32 / 127.
    }
    fn vector<const LANES: usize>(
        &self,
        value: Simd<f32, LANES>,
        index: usize,
    ) -> Simd<f32, LANES> {
        value
            + Simd::from_array(std::array::from_fn(|lane| {
                ((index + lane) % 97) as f32 / 127.
            }))
    }
}

#[test]
fn large_avx_cache_groups_preserve_reduction_order_and_offset_store_indexes() {
    let (rows, inner, columns, first) = (101, 384, 385, 3);
    let input = values(rows * inner, 41);
    let weights = values(inner * columns, 43);
    let bias = values(columns, 47);
    let packed = PackedMatrix::new(&weights, inner, columns).unwrap();
    assert_eq!(avx_group_rows(inner, columns), 96);
    assert_eq!(avx_group_rows(383, columns), GROUP_ROWS);
    assert_eq!(avx_group_rows(inner, 383), GROUP_ROWS);
    for backend in backends()
        .into_iter()
        .filter(|backend| matches!(backend, Backend::Avx | Backend::AvxFma | Backend::Avx2Fma))
    {
        let mut expected = oracle(backend, &input, &weights, &bias, inner, columns);
        for (index, value) in expected.iter_mut().enumerate() {
            *value = IndexStore.scalar(*value, index);
        }
        for band_rows in [95, 96, 97] {
            let sentinel = f32::from_bits(0x7fc0_1234);
            let mut guarded = vec![sentinel; band_rows * columns + 6];
            execute(
                backend,
                &input,
                &bias,
                &packed,
                &mut guarded[3..3 + band_rows * columns],
                first,
                inner,
                columns,
                &IndexStore,
            );
            assert_bits(
                &guarded[3..3 + band_rows * columns],
                &expected[first * columns..(first + band_rows) * columns],
                &format!("{backend:?} large group rows={band_rows}"),
            );
            assert!(
                guarded[..3]
                    .iter()
                    .chain(&guarded[3 + band_rows * columns..])
                    .all(|value| value.to_bits() == sentinel.to_bits())
            );
        }
    }
}

#[test]
fn direct_store_indexes_and_thread_bands_preserve_bits_across_groups() {
    let (rows, inner, columns) = (103, 65, 33);
    let input = values(rows * inner, 13);
    let weights = values(inner * columns, 17);
    let bias = values(columns, 19);
    let packed = PackedMatrix::new(&weights, inner, columns).unwrap();
    for backend in backends() {
        let mut expected = oracle(backend, &input, &weights, &bias, inner, columns);
        for (index, value) in expected.iter_mut().enumerate() {
            *value = IndexStore.scalar(*value, index);
        }
        for threads in [1usize, 2, 4] {
            let mut actual = vec![f32::NAN; rows * columns];
            std::thread::scope(|scope| {
                for (band, output) in actual
                    .chunks_mut(rows.div_ceil(threads) * columns)
                    .enumerate()
                {
                    let first = band * rows.div_ceil(threads);
                    let (input, bias, packed) = (&input, &bias, &packed);
                    scope.spawn(move || {
                        execute(
                            backend,
                            input,
                            bias,
                            packed,
                            output,
                            first,
                            inner,
                            columns,
                            &IndexStore,
                        )
                    });
                }
            });
            assert_bits(
                &actual,
                &expected,
                &format!("{backend:?} threads={threads}"),
            );
        }
        // Short bands exercise every six-row tail and start in the middle of
        // the logical input; transformed indexes remain global output indexes.
        let mut actual = vec![f32::NAN; 17 * columns];
        execute(
            backend,
            &input,
            &bias,
            &packed,
            &mut actual,
            43,
            inner,
            columns,
            &IndexStore,
        );
        assert_bits(
            &actual,
            &expected[43 * columns..60 * columns],
            "offset band",
        );
    }
}

#[test]
fn direct_pointwise_retains_bias_first_and_fma_edge_order() {
    let (rows, inner, columns) = (7, 3, 17);
    let input: Vec<_> = (0..rows).flat_map(|_| [1.0000001, 1., -1.]).collect();
    let weights: Vec<_> = (0..inner)
        .flat_map(|k| std::iter::repeat_n([1.0000001, -1., 1.][k], columns))
        .collect();
    let bias = vec![-1.; columns];
    let packed = PackedMatrix::new(&weights, inner, columns).unwrap();
    for backend in backends() {
        let expected = oracle(backend, &input, &weights, &bias, inner, columns);
        let mut actual = vec![0.; rows * columns];
        execute(
            backend,
            &input,
            &bias,
            &packed,
            &mut actual,
            0,
            inner,
            columns,
            &IdentityStore,
        );
        assert_bits(&actual, &expected, &format!("{backend:?} cancellation"));
    }
}

#[test]
fn full_vector_stores_and_channel_tails_preserve_worker_slice_guards() {
    let (rows, inner, first, band_rows) = (13, 17, 3, 7);
    for columns in [1, 7, 8, 9, 15, 16, 17, 31, 32, 33] {
        let input = values(rows * inner, 23);
        let weights = values(inner * columns, 29);
        let bias = values(columns, 31);
        let packed = PackedMatrix::new(&weights, inner, columns).unwrap();
        for backend in backends() {
            let expected = oracle(backend, &input, &weights, &bias, inner, columns);
            let sentinel = f32::from_bits(0x7fc0_1234);
            let mut guarded = vec![sentinel; band_rows * columns + 6];
            execute(
                backend,
                &input,
                &bias,
                &packed,
                &mut guarded[3..3 + band_rows * columns],
                first,
                inner,
                columns,
                &IdentityStore,
            );
            assert!(
                guarded[..3]
                    .iter()
                    .all(|v| v.to_bits() == sentinel.to_bits())
            );
            assert!(
                guarded[3 + band_rows * columns..]
                    .iter()
                    .all(|v| v.to_bits() == sentinel.to_bits())
            );
            assert_bits(
                &guarded[3..3 + band_rows * columns],
                &expected[first * columns..(first + band_rows) * columns],
                &format!("{backend:?} guarded columns={columns}"),
            );
        }
    }
}

#[test]
fn portable_paired_halves_preserve_nonfused_edges_and_every_column_tail() {
    for (rows, inner) in [(1, 3), (5, 7), (6, 17), (7, 513), (49, 65)] {
        for columns in [1, 3, 4, 5, 7, 8, 9, 12, 15, 16, 17, 31, 33] {
            let input: Vec<_> = (0..rows * inner)
                .map(|i| [1.0000001, 1., -1., -0.0, 0.375][i % 5])
                .collect();
            let weights: Vec<_> = (0..inner * columns)
                .map(|i| [1.0000001, -1., 1., 0.5, -0.25, 0.][i % 6])
                .collect();
            let bias: Vec<_> = (0..columns).map(|i| [-1., 1., -0.0][i % 3]).collect();
            let packed = PackedMatrix::new(&weights, inner, columns).unwrap();
            let expected = oracle(Backend::Portable4, &input, &weights, &bias, inner, columns);
            let mut actual = vec![f32::NAN; rows * columns];
            execute(
                Backend::Portable4,
                &input,
                &bias,
                &packed,
                &mut actual,
                0,
                inner,
                columns,
                &IdentityStore,
            );
            assert_bits(
                &actual,
                &expected,
                &format!("paired SSE rows={rows} inner={inner} columns={columns}"),
            );
        }
    }
}

#[test]
fn portable_paired_subtiles_preserve_bits_and_global_indexes_at_eight_bands() {
    let (rows, inner, columns) = (103usize, 513, 33);
    let input = values(rows * inner, 101);
    let weights = values(inner * columns, 103);
    let bias = values(columns, 107);
    let packed = PackedMatrix::new(&weights, inner, columns).unwrap();
    let mut expected = oracle(Backend::Portable4, &input, &weights, &bias, inner, columns);
    for (index, value) in expected.iter_mut().enumerate() {
        *value = IndexStore.scalar(*value, index);
    }
    for threads in [1usize, 2, 4, 8] {
        let band_rows = rows.div_ceil(threads);
        let mut actual = vec![f32::NAN; rows * columns];
        std::thread::scope(|scope| {
            for (band, output) in actual.chunks_mut(band_rows * columns).enumerate() {
                let (input, bias, packed) = (&input, &bias, &packed);
                scope.spawn(move || {
                    execute(
                        Backend::Portable4,
                        input,
                        bias,
                        packed,
                        output,
                        band * band_rows,
                        inner,
                        columns,
                        &IndexStore,
                    )
                });
            }
        });
        assert_bits(&actual, &expected, &format!("paired SSE bands={threads}"));
    }
}
