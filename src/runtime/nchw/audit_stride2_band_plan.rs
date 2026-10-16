//! POST-FIX only: literal reference eligibility and floor OC8 child selection.
//! Simulated ISA policies do not execute their native instructions.
use super::*;

#[test]
fn audit_stride2_band_plan_and_reporting_match_independent_reference() {
    for backend in [
        Backend::Scalar,
        Backend::Portable4,
        Backend::Avx,
        Backend::AvxFma,
        Backend::Avx2Fma,
        Backend::Avx512,
    ] {
        let native_pair = matches!(backend, Backend::Avx | Backend::AvxFma | Backend::Avx2Fma);
        for outputs in [8usize, 16, 24, 32, 40, 64, 72, 80, 88, 128, 272] {
            let mut p = parameters(3, outputs, 4, 18, [3, 3], [2, 2], [1, 1], 1);
            p.batch = 1;
            assert!(
                outputs * 3 * 2 * 9 * 9 < 8_000_000,
                "all reference packed stride2 bands must remain below the generic work gate"
            );
            for threads in [1, 2, 3, 4, 8] {
                for capacity in [0, 1, 2, 4, 8] {
                    let workers = threads.min(capacity).min(outputs / 8);
                    let plan = packed_stride2_band_plan(backend, &p, true, threads, capacity);
                    let eligible = workers > 1
                        && backend != Backend::Scalar
                        && !(backend == Backend::Portable4 && outputs == 16 && threads > 2);
                    if !eligible {
                        assert!(
                            plan.is_none(),
                            "ineligible stride2 {backend:?}, cout={outputs}, threads={threads}, capacity={capacity}"
                        );
                        continue;
                    }
                    let plan = plan.expect("tiny packed stride2 has no work threshold");
                    assert_eq!(plan.workers(), workers);
                    let mut paired = false;
                    let mut eight = false;
                    for worker in 0..workers {
                        let blocks = outputs / 8;
                        let first = blocks * worker / workers * 8;
                        let end = blocks * (worker + 1) / workers * 8;
                        let count = end - first;
                        let expected = if native_pair && count % 16 == 0 {
                            paired = true;
                            "nchw_packed_3x3_stride2_pair16"
                        } else {
                            eight = true;
                            "nchw_packed_3x3_stride2"
                        };
                        let band = plan.band(worker).unwrap();
                        assert_eq!(
                            (band.first, band.count, band.family()),
                            (first, count, expected),
                            "source floor child {backend:?}, cout={outputs}, threads={threads}, capacity={capacity}, worker={worker}"
                        );
                    }
                    assert!(plan.band(workers).is_none());
                    #[cfg(feature = "parallel")]
                    if capacity == threads {
                        let expected = if paired && eight {
                            "nchw_packed_3x3_stride2_parallel_mixed16_8"
                        } else if paired {
                            "nchw_packed_3x3_stride2_parallel_pair16"
                        } else {
                            "nchw_packed_3x3_stride2"
                        };
                        assert_eq!(
                            family_for_threads(backend, &p, true, threads),
                            Some(expected)
                        );
                    }
                    #[cfg(not(feature = "parallel"))]
                    let _ = (paired, eight);
                }
            }
            if backend != Backend::Scalar {
                let expected = if native_pair && outputs % 16 == 0 {
                    "nchw_packed_3x3_stride2_pair16"
                } else {
                    "nchw_packed_3x3_stride2"
                };
                assert_eq!(family_for_threads(backend, &p, true, 1), Some(expected));
                assert_eq!(selected_backend_for_threads(backend, &p, true, 1), backend);
                // A multi-batch call cannot enter the OC8 worker shard, even
                // when the requested budget is greater than one.
                p.batch = 2;
                for threads in [1, 2, 4, 8] {
                    assert!(packed_stride2_band_plan(backend, &p, true, threads, 4).is_none());
                    // Preserve the existing Portable4 cout16/>2 unpacked gate.
                    if backend != Backend::Portable4 || outputs != 16 || threads <= 2 {
                        assert_eq!(
                            family_for_threads(backend, &p, true, threads),
                            Some(expected)
                        );
                        assert_eq!(
                            selected_backend_for_threads(backend, &p, true, threads),
                            backend
                        );
                    }
                }
            }
        }
        let mut p = parameters(3, 80, 4, 18, [3, 3], [2, 2], [1, 1], 1);
        p.batch = 1;
        assert!(packed_stride2_band_plan(backend, &p, false, 4, 4).is_none());
        let mut invalid = p;
        invalid.batch = 2;
        assert!(packed_stride2_band_plan(backend, &invalid, true, 4, 4).is_none());
        invalid = p;
        invalid.stride_width = 1;
        assert!(packed_stride2_band_plan(backend, &invalid, true, 4, 4).is_none());
        invalid = p;
        invalid.pad_left = 0;
        assert!(packed_stride2_band_plan(backend, &invalid, true, 4, 4).is_none());
        invalid = p;
        invalid.dilation_width = 2;
        assert!(packed_stride2_band_plan(backend, &invalid, true, 4, 4).is_none());
    }
}
