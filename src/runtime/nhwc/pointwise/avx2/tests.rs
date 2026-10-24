use super::super::tests::{assert_bits, oracle, values};
use super::*;

struct ResidualRelu<'a>(&'a [f32]);
impl StoreTransform for ResidualRelu<'_> {
    fn scalar(&self, value: f32, index: usize) -> f32 {
        (value + self.0[index]).max(0.)
    }
    fn vector<const N: usize>(&self, value: Simd<f32, N>, index: usize) -> Simd<f32, N> {
        Simd::from_array(std::array::from_fn(|lane| {
            self.scalar(value[lane], index + lane)
        }))
    }
}

#[test]
fn inline_pointwise_preserves_order_tails_and_final_epilogue() {
    if !simd::avx2_fma_available() {
        return;
    }
    for (rows, inner, columns) in [
        (7, 1, 17),
        (7, 2, 17),
        (7, 3, 17),
        (1, 255, 1),
        (5, 256, 17),
        (7, 257, 33),
        (49, 511, 16),
        (97, 512, 129),
        (101, 769, 17),
        (13, 1024, 32),
        (7, 1537, 33),
    ] {
        let first = 3;
        let input = values((first + rows) * inner, 23);
        let weights = values(inner * columns, 29);
        let packed = PackedMatrix::new(&weights, inner, columns).unwrap();
        let bias = values(columns, 31);
        // Include enough padding for the epilogue's final partial SIMD vector.
        let residual = values((first + rows) * columns + 8, 37);
        let store = ResidualRelu(&residual);
        for bias in [bias.as_slice(), &[]] {
            let expected = oracle(Backend::Avx2Fma, &input, &weights, bias, inner, columns);
            let expected: Vec<_> = expected
                .iter()
                .enumerate()
                .map(|(i, &v)| store.scalar(v, i))
                .collect();
            let sentinel = f32::from_bits(0x7fc0_1234);
            let mut output = vec![sentinel; rows * columns + 6];
            // SAFETY: runtime guard and owned, shape-checked buffers above.
            unsafe {
                execute(
                    &input[first * inner..],
                    bias,
                    &packed,
                    &mut output[3..3 + rows * columns],
                    first,
                    inner,
                    columns,
                    &store,
                )
            };
            assert_bits(
                &output[3..3 + rows * columns],
                &expected[first * columns..],
                &format!("inline rows={rows} inner={inner} columns={columns}"),
            );
            assert!(
                output[..3]
                    .iter()
                    .chain(&output[3 + rows * columns..])
                    .all(|value| value.to_bits() == sentinel.to_bits())
            );
        }
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn full_k(
    input: &[f32],
    bias: &[f32],
    packed: &PackedMatrix,
    output: &mut [f32],
    inner: usize,
    columns: usize,
) {
    wide_grouped::<8, true, 0>(
        input,
        bias,
        packed,
        output,
        0,
        inner,
        columns,
        &IdentityStore,
    );
}

#[test]
#[ignore = "physical AVX2 diagnostic: three shapes, one warmup and three alternating pairs"]
#[allow(clippy::needless_range_loop, clippy::assertions_on_constants)]
fn inline_pointwise_comparison() {
    assert!(simd::avx2_fma_available());
    #[cfg(windows)]
    // SAFETY: process pseudo-handle is valid; both masks point to writable words.
    unsafe {
        unsafe extern "system" {
            fn GetCurrentProcess() -> *mut std::ffi::c_void;
            fn GetProcessAffinityMask(
                p: *mut std::ffi::c_void,
                a: *mut usize,
                s: *mut usize,
            ) -> i32;
            fn SetProcessAffinityMask(p: *mut std::ffi::c_void, mask: usize) -> i32;
        }
        let p = GetCurrentProcess();
        let (mut a, mut s) = (0, 0);
        assert_ne!(GetProcessAffinityMask(p, &mut a, &mut s), 0);
        assert_ne!(a, 0);
        assert_ne!(SetProcessAffinityMask(p, a.isolate_lowest_one()), 0);
    }
    let deadline = std::time::Instant::now();
    for (rows, inner, columns) in [(384, 96, 192), (1920, 128, 256), (192, 192, 384)] {
        let input = values(rows * inner, 41);
        let weights = values(inner * columns, 43);
        let bias = values(columns, 47);
        let packed = PackedMatrix::new(&weights, inner, columns).unwrap();
        let mut original = vec![0.; rows * columns];
        let mut candidate = original.clone();
        let run = |variant, output: &mut [f32]| {
            // SAFETY: runtime ISA guard above; all buffers match their shapes.
            unsafe {
                if variant == 0 {
                    full_k(&input, &bias, &packed, output, inner, columns)
                } else {
                    super::super::execute(
                        Backend::Avx2Fma,
                        &input,
                        &bias,
                        &packed,
                        output,
                        0,
                        inner,
                        columns,
                        &IdentityStore,
                    )
                }
            }
        };
        run(0, &mut original);
        run(1, &mut candidate);
        assert_bits(&candidate, &original, "inline warmup");
        let mut samples = [[0.; 3]; 2];
        for sample in 0..3 {
            for variant in if sample == 1 { [1, 0] } else { [0, 1] } {
                let output = if variant == 0 {
                    &mut original
                } else {
                    &mut candidate
                };
                let start = std::time::Instant::now();
                run(variant, std::hint::black_box(output));
                samples[variant][sample] = start.elapsed().as_secs_f64() * 1000.;
            }
            assert_bits(&candidate, &original, "inline measured");
        }
        println!(
            "pointwise_inline_reduction {}",
            serde_json::json!({
                "rows": rows, "inner": inner, "columns": columns, "full_k": true,
                "warmups_per_variant": 1, "calls_per_sample": 1,
                "full_k_ms": samples[0], "inline_ms": samples[1], "equal_bits": true
            })
        );
        assert!(
            deadline.elapsed().as_secs() < 10,
            "bounded pointwise diagnostic exceeded ten seconds"
        );
    }
}

#[test]
fn inline_pointwise_keeps_bias_first_fma_cancellation() {
    if !simd::avx2_fma_available() {
        return;
    }
    let (rows, inner, columns) = (7, 191, 129);
    let input: Vec<_> = (0..rows * inner)
        .map(|i| [1.0000001, 1., -1., -0.0, 0.375][i % 5])
        .collect();
    let weights: Vec<_> = (0..inner * columns)
        .map(|i| [1.0000001, -1., 1., 0.5, -0.25, 0.][i % 6])
        .collect();
    let bias: Vec<_> = (0..columns).map(|i| [-1., 1., -0.0][i % 3]).collect();
    let packed = PackedMatrix::new(&weights, inner, columns).unwrap();
    let expected = oracle(Backend::Avx2Fma, &input, &weights, &bias, inner, columns);
    let mut output = vec![0.; rows * columns];
    super::super::execute(
        Backend::Avx2Fma,
        &input,
        &bias,
        &packed,
        &mut output,
        0,
        inner,
        columns,
        &IdentityStore,
    );
    assert_bits(&output, &expected, "inline bias-first cancellation");
}
