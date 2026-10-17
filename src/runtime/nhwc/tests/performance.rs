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
            SetProcessAffinityMask(process, allowed.isolate_lowest_one()),
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

#[test]
#[ignore = "local release diagnostic: two recognition pointwise shapes, one warm-up and three calls"]
fn recognition_pointwise() {
    assert!(simd::avx2_fma_available());
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
            SetProcessAffinityMask(process, allowed.isolate_lowest_one()),
            0
        );
    }
    let deadline = std::time::Instant::now();
    for (rows, inner, columns) in [(96, 384, 768), (960, 512, 1024)] {
        let input: Vec<_> = (0..rows * inner)
            .map(|i| ((i * 37 % 43) as f32 - 21.) / 17.)
            .collect();
        let matrix: Vec<_> = (0..inner * columns)
            .map(|i| ((i * 19 % 29) as f32 - 14.) / 13.)
            .collect();
        let bias: Vec<_> = (0..columns).map(|i| i as f32 / 19. - 0.3).collect();
        let packed = PackedMatrix::new(&matrix, inner, columns).unwrap();
        let mut output = vec![0.; rows * columns];
        let call = |out: &mut [f32]| {
            pointwise::execute(
                Backend::Avx2Fma,
                &input,
                &bias,
                &packed,
                out,
                0,
                inner,
                columns,
                &IdentityStore,
            )
        };
        call(&mut output);
        let mut samples = [0.; 3];
        for sample in &mut samples {
            let timer = std::time::Instant::now();
            call(std::hint::black_box(&mut output));
            *sample = timer.elapsed().as_secs_f64() * 1000.;
        }
        for row in [0, 5, rows - 1] {
            for c in [0, 7, columns - 1] {
                let mut expected = bias[c];
                for k in 0..inner {
                    expected = input[row * inner + k].mul_add(matrix[k * columns + c], expected);
                }
                assert_eq!(output[row * columns + c].to_bits(), expected.to_bits());
            }
        }
        println!(
            "nhwc_recognition_pointwise rows={rows} inner={inner} columns={columns} calls_per_sample=1 ms={samples:?}"
        );
        assert!(deadline.elapsed().as_secs() < 10);
    }
}

#[test]
fn paired_controls_preserve_borders_channel_tails_and_partial_row_groups() {
    if !simd::avx2_fma_available() {
        return;
    }
    let p = plan([2, 40, 40, 9, 35], [7, 7], [2, 2], [1, 2], [3, 7, 3, 5], 40);
    let input = values(p.batch * p.cin * p.height * p.width, 53);
    let weights = values(p.cout * p.kh * p.kw, 59);
    let bias = values(p.cout, 61);
    let packed = pack(&p, &weights).unwrap();
    let mut baseline = vec![0.; p.batch * p.oh * p.ow * p.cout];
    let mut candidate = baseline.clone();
    strip::execute_control::<false>(&p, &input, &weights, &bias, &packed, &mut baseline);
    strip::execute_control::<true>(&p, &input, &weights, &bias, &packed, &mut candidate);
    assert!(
        baseline
            .iter()
            .zip(&candidate)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
    for (rows, inner, columns) in [(103, 384, 400), (17, 19, 25)] {
        let input = values(rows * inner, 67);
        let weights = values(inner * columns, 71);
        let bias = values(columns, 73);
        let packed = PackedMatrix::new(&weights, inner, columns).unwrap();
        let mut baseline = vec![0.; rows * columns];
        let mut candidate = baseline.clone();
        pointwise::execute_group_control::<48>(
            &input,
            &bias,
            &packed,
            &mut baseline,
            inner,
            columns,
        );
        pointwise::execute_group_control::<0>(
            &input,
            &bias,
            &packed,
            &mut candidate,
            inner,
            columns,
        );
        assert!(
            baseline
                .iter()
                .zip(&candidate)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "rows={rows} inner={inner} columns={columns}"
        );
    }
}

#[test]
#[ignore = "local paired diagnostic: three shapes, one warm-up per policy, three AB/BA/AB pairs"]
fn paired_spatial_kernel_controls() {
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
            SetProcessAffinityMask(process, allowed.isolate_lowest_one()),
            0
        );
    }
    let deadline = std::time::Instant::now();
    fn paired(
        name: &str,
        output_len: usize,
        deadline: &std::time::Instant,
        mut baseline: impl FnMut(&mut [f32]),
        mut candidate: impl FnMut(&mut [f32]),
    ) {
        // All output allocation and both fixed warm-ups precede measurement.
        // Input generation and packing also happen before this helper is called.
        let mut baseline_output = vec![0.; output_len];
        let mut candidate_output = vec![0.; output_len];
        baseline(&mut baseline_output);
        candidate(&mut candidate_output);
        let equal_bits = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b)
                .all(|(left, right)| left.to_bits() == right.to_bits())
        };
        assert!(
            equal_bits(&baseline_output, &candidate_output),
            "{name} warm-up"
        );
        assert!(deadline.elapsed() < std::time::Duration::from_secs(10));
        let mut baseline_ms = [0.; 3];
        let mut candidate_ms = [0.; 3];
        for sample in 0..3 {
            // A is baseline, B is candidate. Counterbalance fixed pairs so a
            // host-frequency/time-period shift is not confused with the delta.
            let order = if sample == 1 {
                [true, false]
            } else {
                [false, true]
            };
            for is_candidate in order {
                if is_candidate {
                    let timer = std::time::Instant::now();
                    candidate(std::hint::black_box(candidate_output.as_mut_slice()));
                    candidate_ms[sample] = timer.elapsed().as_secs_f64() * 1000.;
                } else {
                    let timer = std::time::Instant::now();
                    baseline(std::hint::black_box(baseline_output.as_mut_slice()));
                    baseline_ms[sample] = timer.elapsed().as_secs_f64() * 1000.;
                }
                // Validate outside every measured call, against the other
                // policy's most recent complete output for these same inputs.
                assert!(
                    equal_bits(&baseline_output, &candidate_output),
                    "{name} sample={sample} candidate={is_candidate}"
                );
                assert!(deadline.elapsed() < std::time::Duration::from_secs(10));
            }
        }
        println!(
            "paired_spatial kernel={name} warmups_per_policy=1 calls_per_sample=1 order=AB,BA,AB baseline_ms={baseline_ms:?} candidate_ms={candidate_ms:?} exact_output_bits=true"
        );
    }

    let p = plan([1, 96, 96, 152, 240], [7, 7], [1, 1], [1, 1], [3; 4], 96);
    let input = values(p.cin * p.height * p.width, 53);
    let weights = values(p.cout * p.kh * p.kw, 59);
    let bias = values(p.cout, 61);
    let packed = pack(&p, &weights).unwrap();
    paired(
        "depthwise_96x152x240_k7_unsegmented_vs_segmented",
        p.oh * p.ow * p.cout,
        &deadline,
        |output| {
            strip::execute_control::<false>(&p, &input, &weights, &bias, &packed, output);
        },
        |output| {
            strip::execute_control::<true>(&p, &input, &weights, &bias, &packed, output);
        },
    );
    for (rows, inner, columns) in [(96, 384, 768), (960, 512, 1024)] {
        let input: Vec<_> = (0..rows * inner)
            .map(|i| ((i * 37 % 43) as f32 - 21.) / 17.)
            .collect();
        let matrix: Vec<_> = (0..inner * columns)
            .map(|i| ((i * 19 % 29) as f32 - 14.) / 13.)
            .collect();
        let bias: Vec<_> = (0..columns).map(|i| i as f32 / 19. - 0.3).collect();
        let packed = PackedMatrix::new(&matrix, inner, columns).unwrap();
        paired(
            &format!("pointwise_{rows}x{inner}x{columns}_group48_vs_adaptive"),
            rows * columns,
            &deadline,
            |output| {
                pointwise::execute_group_control::<48>(
                    &input, &bias, &packed, output, inner, columns,
                );
            },
            |output| {
                pointwise::execute_group_control::<0>(
                    &input, &bias, &packed, output, inner, columns,
                );
            },
        );
    }
}
