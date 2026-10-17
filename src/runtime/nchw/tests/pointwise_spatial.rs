//! Packed 1x1 arithmetic and bounded, opt-in production-kernel measurements.
use super::*;

#[test]
fn packed_pointwise_spatial_tiles_keep_bias_reduction_bands_and_epilogues() {
    for channels in [5, 96] {
        for outputs in [24, 28] {
            for width in [3, 7, 15, 16, 17, 49, 513, 2737] {
                let p = parameters(channels, outputs, 1, width, [1, 1], [1, 1], [0, 0], 1);
                let input: Vec<_> = (0..p.batch * channels * width)
                    .map(|i| ((i * 37 % 43) as f32 - 21.) / 17.)
                    .collect();
                let weights: Vec<_> = (0..outputs * channels)
                    .map(|i| ((i * 19 % 29) as f32 - 14.) / 13.)
                    .collect();
                let packed = panels(&weights, &p);
                for backend in backends() {
                    for bias in [vec![], (0..outputs).map(|i| i as f32 / 19. - 0.3).collect()] {
                        let expected =
                            reference(&p, backend, Kind::PointwisePacked, &input, &weights, &bias);
                        // Nonzero bands cross both packed OC16 panels. A short
                        // final OC4 band and odd spatial tails stay in bounds.
                        for (batch, first, count) in [(0, 0, outputs), (1, 4, 20), (1, 20, 4)] {
                            let begin = (batch * outputs + first) * width;
                            let end = begin + count * width;
                            let store = IndexedStore::new(expected.len());
                            let guard = f32::from_bits(0x7fc0_55aa);
                            let mut actual = vec![guard; count * width + 8];
                            run(
                                backend,
                                Kind::PointwisePacked,
                                &p,
                                &input,
                                &weights,
                                &bias,
                                Some(&packed),
                                &mut actual[4..4 + count * width],
                                batch,
                                first,
                                count,
                                &store,
                            );
                            assert!(
                                actual[..4]
                                    .iter()
                                    .chain(&actual[4 + count * width..])
                                    .all(|v| v.to_bits() == guard.to_bits())
                            );
                            for (index, &value) in actual[4..4 + count * width].iter().enumerate() {
                                let index = begin + index;
                                assert_eq!(
                                    value.to_bits(),
                                    IndexedStore::expected(expected[index], index).to_bits(),
                                    "{backend:?} cin={channels} cout={outputs} width={width} band={first}/{count} index={index}"
                                );
                            }
                            for (index, visits) in store.visits.iter().enumerate() {
                                assert_eq!(
                                    visits.load(std::sync::atomic::Ordering::Relaxed),
                                    usize::from((begin..end).contains(&index))
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Exactly one warm-up plus three fixed samples of four calls per shape.
/// Compilation, allocation and packing are outside timing; no adaptive growth.
/// Run this test alone, after correctness, on the physical Windows host.
#[test]
#[ignore = "bounded physical-machine packed 1x1 measurement; run alone with --nocapture"]
fn measure_packed_pointwise_96_to_24() {
    use std::{
        hint::black_box,
        time::{Duration, Instant},
    };
    assert!(
        simd::avx2_fma_available(),
        "requires the qualified AVX2/FMA host"
    );
    let _affinity = SingleCpu::new();
    let cpu = CpuContext::new(1).unwrap();
    let budget = Duration::from_secs(10);
    let mut measured = Duration::ZERO;
    for (height, width) in [(19, 30), (76, 120), (152, 240)] {
        let mut p = parameters(96, 24, height, width, [1, 1], [1, 1], [0, 0], 1);
        p.batch = 1;
        assert_eq!(
            select(Backend::Avx2Fma, &p, true, 1),
            Some(Kind::PointwisePacked)
        );
        let input: Vec<_> = (0..96 * height * width)
            .map(|i| ((i * 37 % 43) as f32 - 21.) / 17.)
            .collect();
        let weights: Vec<_> = (0..96 * 24)
            .map(|i| ((i * 19 % 29) as f32 - 14.) / 13.)
            .collect();
        let bias: Vec<_> = (0..24).map(|i| i as f32 / 19. - 0.3).collect();
        let packed = panels(&weights, &p);
        let mut output = vec![0.; 24 * height * width];
        let mut execute = || {
            assert!(try_conv(
                Backend::Avx2Fma,
                &p,
                black_box(&input),
                &weights,
                &bias,
                Some(&packed),
                black_box(&mut output),
                &cpu,
                1
            ));
            black_box(&output);
        };
        let start = Instant::now();
        execute();
        measured += start.elapsed();
        let mut samples = [0.; 3];
        for sample in &mut samples {
            assert!(
                measured < budget,
                "fixed 10-second measurement budget exhausted"
            );
            let start = Instant::now();
            for _ in 0..4 {
                execute();
            }
            let elapsed = start.elapsed();
            measured += elapsed;
            *sample = elapsed.as_secs_f64() * 1000. / 4.;
        }
        println!(
            "packed_pointwise cin=96 cout=24 h={height} w={width} calls_per_sample=4 ms={samples:?}"
        );
        assert!(
            measured <= budget,
            "fixed 10-second measurement budget exhausted"
        );
    }
    println!(
        "packed_pointwise total_warmup_and_measured_ms={:.3}",
        measured.as_secs_f64() * 1000.
    );
}

#[cfg(windows)]
struct SingleCpu(usize);
#[cfg(windows)]
impl SingleCpu {
    fn new() -> Self {
        unsafe {
            let process = GetCurrentProcess();
            let mut allowed = 0;
            let mut system = 0;
            assert_ne!(
                GetProcessAffinityMask(process, &mut allowed, &mut system),
                0
            );
            let single = allowed.isolate_lowest_one();
            assert_ne!(single, 0);
            assert_ne!(SetProcessAffinityMask(process, single), 0);
            println!("packed_pointwise affinity={single:x}");
            Self(allowed)
        }
    }
}
#[cfg(windows)]
impl Drop for SingleCpu {
    fn drop(&mut self) {
        unsafe {
            SetProcessAffinityMask(GetCurrentProcess(), self.0);
        }
    }
}
#[cfg(windows)]
#[link(name = "kernel32")]
unsafe extern "system" {
    fn GetCurrentProcess() -> *mut std::ffi::c_void;
    fn GetProcessAffinityMask(
        process: *mut std::ffi::c_void,
        process_mask: *mut usize,
        system_mask: *mut usize,
    ) -> i32;
    fn SetProcessAffinityMask(process: *mut std::ffi::c_void, mask: usize) -> i32;
}
#[cfg(not(windows))]
struct SingleCpu;
#[cfg(not(windows))]
impl SingleCpu {
    fn new() -> Self {
        panic!("this focused measurement is qualified on Windows only")
    }
}
