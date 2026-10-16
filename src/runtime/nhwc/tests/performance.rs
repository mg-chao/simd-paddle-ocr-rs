//! Explicit, bounded physical-machine diagnostic; excluded from normal tests.
use super::*;

#[test]
#[ignore = "local release diagnostic: one warm-up and three fixed kernel calls"]
fn sparse_detector_depthwise_7x7() {
    assert!(
        simd::avx2_fma_available(),
        "this diagnostic requires AVX2/FMA"
    );
    #[cfg(windows)]
    unsafe {
        unsafe extern "system" {
            fn GetCurrentProcess() -> *mut std::ffi::c_void;
            fn GetProcessAffinityMask(
                process: *mut std::ffi::c_void,
                process_mask: *mut usize,
                system_mask: *mut usize,
            ) -> i32;
            fn SetProcessAffinityMask(process: *mut std::ffi::c_void, mask: usize) -> i32;
        }
        let (mut allowed, mut system) = (0, 0);
        let process = GetCurrentProcess();
        assert_ne!(
            GetProcessAffinityMask(process, &mut allowed, &mut system),
            0
        );
        assert_ne!(allowed, 0);
        assert_ne!(
            SetProcessAffinityMask(process, allowed & allowed.wrapping_neg()),
            0
        );
    }
    let p = plan([1, 96, 96, 152, 240], [7, 7], [1, 1], [1, 1], [3; 4], 96);
    let input = values(p.cin * p.height * p.width, 53);
    let weights = values(p.cout * p.kh * p.kw, 59);
    let bias = values(p.cout, 61);
    let packed = pack(&p, &weights).unwrap();
    let mut output = vec![0.; p.oh * p.ow * p.cout];
    let run = |output: &mut [f32]| {
        conv_dispatch(
            Backend::Avx2Fma,
            &p,
            &input,
            &weights,
            &bias,
            Some(&packed),
            output,
            0,
            &mut [],
        );
    };
    run(&mut output);
    let expected = output.clone();
    let deadline = std::time::Instant::now();
    for sample in 0..3 {
        let started = std::time::Instant::now();
        run(std::hint::black_box(&mut output));
        println!(
            "nhwc_depthwise_strip 96x152x240 k7 sample={sample} ms={:.6}",
            started.elapsed().as_secs_f64() * 1000.
        );
        assert!(
            output
                .iter()
                .zip(&expected)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
        assert!(
            deadline.elapsed().as_secs() < 10,
            "fixed diagnostic exceeded ten seconds"
        );
    }
}
