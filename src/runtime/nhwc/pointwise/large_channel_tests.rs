//! Fixed-count grouping diagnosis; this does not change production dispatch.
use super::tests::{assert_bits, oracle, values};
use super::*;

#[test]
fn larger_group_preserves_order_and_row_channel_tails() {
    if !simd::avx2_fma_available() {
        return;
    }
    let (rows, inner, columns) = (197, 385, 129);
    let input = values(rows * inner, 11);
    let weights = values(inner * columns, 13);
    let bias = values(columns, 17);
    let packed = PackedMatrix::new(&weights, inner, columns).unwrap();
    let sentinel = f32::from_bits(0x7fc0_1234);
    for bias in [bias.as_slice(), &[]] {
        let expected = oracle(Backend::Avx2Fma, &input, &weights, bias, inner, columns);
        let mut output = vec![sentinel; rows * columns + 6];
        execute_group_control::<192>(
            &input,
            bias,
            &packed,
            &mut output[3..3 + rows * columns],
            inner,
            columns,
        );
        assert_bits(&output[3..3 + rows * columns], &expected, "group 192 tails");
        assert!(
            output[..3]
                .iter()
                .chain(&output[3 + rows * columns..])
                .all(|value| value.to_bits() == sentinel.to_bits())
        );
    }
}

#[test]
#[ignore = "local release AVX2 grouping probe: two actual large shapes, one warmup and three single-call pairs"]
fn large_channel_group_comparison() {
    assert!(!cfg!(debug_assertions), "use a release build");
    assert!(simd::avx2_fma_available());
    #[cfg(windows)]
    // SAFETY: valid pseudo-handle and writable affinity masks.
    unsafe {
        unsafe extern "system" {
            fn GetCurrentProcess() -> *mut std::ffi::c_void;
            fn GetProcessAffinityMask(
                process: *mut std::ffi::c_void,
                allowed: *mut usize,
                system: *mut usize,
            ) -> i32;
            fn SetProcessAffinityMask(process: *mut std::ffi::c_void, mask: usize) -> i32;
        }
        let process = GetCurrentProcess();
        let (mut allowed, mut system) = (0, 0);
        assert_ne!(
            GetProcessAffinityMask(process, &mut allowed, &mut system),
            0
        );
        assert_ne!(allowed, 0);
        assert_ne!(
            SetProcessAffinityMask(process, allowed.isolate_lowest_one()),
            0
        );
    }
    let deadline = std::time::Instant::now();
    for (rows, inner, columns) in [(960, 512, 1024), (480, 1536, 768)] {
        let input = values(rows * inner, 41);
        let weights = values(inner * columns, 43);
        let bias = values(columns, 47);
        let packed = PackedMatrix::new(&weights, inner, columns).unwrap();
        let mut production = vec![0.; rows * columns];
        let mut larger = production.clone();
        let run = |variant, output: &mut [f32]| match variant {
            0 => execute_group_control::<0>(&input, &bias, &packed, output, inner, columns),
            1 => execute_group_control::<192>(&input, &bias, &packed, output, inner, columns),
            _ => unreachable!(),
        };
        run(0, &mut production);
        run(1, &mut larger);
        assert_bits(&larger, &production, "group 192 warmup");
        let mut samples = [[0.; 3]; 2];
        for sample in 0..3 {
            for variant in if sample == 1 { [1, 0] } else { [0, 1] } {
                let output = if variant == 0 {
                    &mut production
                } else {
                    &mut larger
                };
                let start = std::time::Instant::now();
                run(variant, std::hint::black_box(output));
                samples[variant][sample] = start.elapsed().as_secs_f64() * 1000.;
            }
            assert_bits(&larger, &production, "group 192 measured");
        }
        println!(
            "large_channel_groups {}",
            serde_json::json!({
                "rows":rows,"inner":inner,"columns":columns,
                "production_group":avx_group_rows(inner, columns),"diagnostic_group":192,
                "warmups_per_variant":1,"calls_per_sample":1,"samples_per_variant":3,
                "production_ms":samples[0],"group_192_ms":samples[1],
                "packed_address_mod_64":packed.data().as_ptr() as usize % 64,
                "input_address_mod_64":input.as_ptr() as usize % 64,
                "order":[[0,1],[1,0],[0,1]],"outputs_equal_bits":true,
                "scope":"one logical CPU, cached synthetic inputs and packed weights, identity final store"
            })
        );
        assert!(
            deadline.elapsed().as_secs() < 10,
            "group probe exceeded ten seconds"
        );
    }
}
