//! Packed 3x3 output-channel band planning and recursive child eligibility.
use super::*;

#[test]
fn packed3_recursive_band_plan_matches_independent_reference() {
    for backend in [
        Backend::Scalar,
        Backend::Portable4,
        Backend::Avx,
        Backend::AvxFma,
        Backend::Avx2Fma,
        Backend::Avx512,
    ] {
        for outputs in [8usize, 16, 56, 64, 72, 120, 128, 136, 264, 272] {
            // Two or more OC8 blocks exceed the 8M work threshold.
            let mut p = parameters(512, outputs, 8, 18, [3, 3], [1, 1], [1, 1], 1);
            p.batch = 1;
            for threads in [2, 3, 4, 8] {
                for capacity in [0, 1, 2, 4, 8] {
                    let workers = threads.min(capacity).min(outputs / 8);
                    let plan = packed3_band_plan(backend, &p, true, threads, capacity);
                    if workers <= 1
                        || backend == Backend::Scalar
                        || backend == Backend::Portable4 && outputs == 16 && threads > 2
                    {
                        assert!(plan.is_none());
                        continue;
                    }
                    let plan = plan.expect("eligible native/portable packed3 worker plan");
                    assert_eq!(plan.workers(), workers);
                    let mut any_paired = false;
                    let mut any_eight = false;
                    for worker in 0..workers {
                        let blocks = outputs / 8;
                        let first = blocks * worker / workers * 8;
                        let end = blocks * (worker + 1) / workers * 8;
                        let count = end - first;
                        let recursive_native =
                            matches!(backend, Backend::Avx | Backend::AvxFma | Backend::Avx2Fma);
                        let expected_family = if recursive_native && count >= 64 && count % 16 == 0
                        {
                            any_paired = true;
                            "nchw_packed_3x3_pair16"
                        } else {
                            any_eight = true;
                            "nchw_packed_3x3"
                        };
                        let band = plan.band(worker).unwrap();
                        assert_eq!(
                            (band.first, band.count, band.family()),
                            (first, count, expected_family),
                            "literal floor/recursive plan {backend:?}, cout={outputs}, threads={threads}, capacity={capacity}, worker={worker}"
                        );
                    }
                    assert!(plan.band(workers).is_none());
                    #[cfg(feature = "parallel")]
                    if capacity == threads {
                        let expected = if any_paired && any_eight {
                            "nchw_packed_3x3_parallel_mixed16_8"
                        } else if any_paired {
                            "nchw_packed_3x3_parallel_pair16"
                        } else {
                            "nchw_packed_3x3"
                        };
                        assert_eq!(
                            family_for_threads(backend, &p, true, threads),
                            Some(expected)
                        );
                    }
                    #[cfg(not(feature = "parallel"))]
                    let _ = (any_paired, any_eight);
                }
            }
        }
    }
    // These are the adjacent integral cin values around the pinned8M work
    // threshold for128x8x8x3x3. No production work/eligibility helper supplies
    // this expected decision. Public Session rejects budgets above context
    // capacity; this pure planner additionally clamps its supplied capacity.
    for inputs in [108usize, 109] {
        let mut p = parameters(inputs, 128, 8, 8, [3, 3], [1, 1], [1, 1], 1);
        p.batch = 1;
        let eligible = 128 * inputs * 8 * 8 * 9 >= 8_000_000;
        for backend in [
            Backend::Portable4,
            Backend::Avx,
            Backend::AvxFma,
            Backend::Avx2Fma,
            Backend::Avx512,
        ] {
            let plan = packed3_band_plan(backend, &p, true, 8, 2);
            assert_eq!(
                plan.is_some(),
                eligible,
                "8M threshold {backend:?}, cin={inputs}"
            );
            if let Some(plan) = plan {
                assert_eq!(
                    plan.workers(),
                    2,
                    "context capacity must bound the requested8 workers"
                );
            }
            assert!(packed3_band_plan(backend, &p, false, 8, 2).is_none());
            assert!(packed3_band_plan(backend, &p, true, 1, 2).is_none());
            assert!(packed3_band_plan(backend, &p, true, 8, 0).is_none());
            assert!(packed3_band_plan(backend, &p, true, 8, 1).is_none());
            let mut invalid = p;
            invalid.batch = 2;
            assert!(packed3_band_plan(backend, &invalid, true, 8, 2).is_none());
            invalid = p;
            invalid.pad_left = 0;
            assert!(packed3_band_plan(backend, &invalid, true, 8, 2).is_none());
            invalid = p;
            invalid.dilation_width = 2;
            assert!(packed3_band_plan(backend, &invalid, true, 8, 2).is_none());
        }
    }
}
