//! Focused threshold correctness and an explicitly bounded local diagnostic.
use super::*;

#[test]
fn bitmap_masks_expand_every_lane_without_touching_the_tail() {
    for mask in 0..=u16::MAX {
        let mut four = [197; 7];
        let mut eight = [197; 11];
        let mut sixteen = [197; 19];
        store_bitmap_mask::<4>(&mut four, u64::from(mask));
        store_bitmap_mask::<8>(&mut eight, u64::from(mask));
        store_bitmap_mask::<16>(&mut sixteen, u64::from(mask));
        for (bytes, lanes) in [(&four[..], 4), (&eight[..], 8), (&sixteen[..], 16)] {
            for (lane, &byte) in bytes[..lanes].iter().enumerate() {
                assert_eq!(byte, ((mask >> lane) & 1) as u8);
            }
            assert_eq!(&bytes[lanes..], [197; 3]);
        }
    }
}

#[test]
fn threshold_special_thresholds_and_sparse_masks_match_scalar() {
    let choices = [
        f32::MIN,
        -1.,
        -f32::MIN_POSITIVE,
        -f32::from_bits(1),
        -0.,
        0.,
        f32::from_bits(1),
        f32::MIN_POSITIVE,
        0.3,
        1.,
        f32::MAX,
    ];
    for kernel in PixelKernel::test_kernels() {
        for length in 0..=65 {
            let mut values: Vec<_> = (0..length).map(|i| choices[i % choices.len()]).collect();
            for threshold in [f32::NEG_INFINITY, -0., 0.3, f32::INFINITY, f32::NAN] {
                let expected: Vec<_> = values.iter().map(|&v| u8::from(v > threshold)).collect();
                let mut actual = vec![197; length + 3];
                assert_eq!(
                    binarize(kernel, &values, &mut actual, threshold).unwrap(),
                    expected.contains(&1),
                    "{kernel:?}/{length}/{threshold}"
                );
                assert_eq!(&actual[..length], expected);
                assert_eq!(&actual[length..], [197; 3]);
            }
            values.fill(0.);
            for index in 0..length {
                values[index] = 1.;
                let mut actual = vec![197; length + 3];
                assert!(binarize(kernel, &values, &mut actual, 0.3).unwrap());
                assert_eq!(
                    actual[..length].iter().filter(|&&byte| byte == 1).count(),
                    1
                );
                assert_eq!(actual[index], 1, "{kernel:?}/{length}/{index}");
                assert_eq!(&actual[length..], [197; 3]);
                values[index] = 0.;
            }
            let mut actual = vec![197; length + 3];
            assert!(!binarize(kernel, &values, &mut actual, 0.3).unwrap());
            assert!(actual[..length].iter().all(|&byte| byte == 0));
            assert_eq!(&actual[length..], [197; 3]);
        }
    }
}

// Keep the pre-optimization vector implementation as a paired performance
// control. DEFERRED separates validity reduction from bitmap-store changes.
#[inline(always)]
fn binarize_control<const N: usize, const DEFERRED: bool>(
    prediction: &[f32],
    bitmap: &mut [u8],
    threshold: f32,
) -> ImageResult<bool> {
    let end = prediction.len() / N * N;
    let mut any = false;
    let mut invalid = false;
    let mut invalid_mask = std::simd::Mask::<i32, N>::splat(false);
    for offset in (0..end).step_by(N) {
        let values = Simd::<f32, N>::from_slice(&prediction[offset..offset + N]);
        if DEFERRED {
            invalid_mask |= !values.is_finite();
        } else {
            invalid |= (!values.is_finite()).any();
        }
        let mask = values.simd_gt(Simd::splat(threshold)).to_bitmask();
        any |= mask != 0;
        for lane in 0..N {
            bitmap[offset + lane] = ((mask >> lane) & 1) as u8;
        }
    }
    if invalid || (DEFERRED && invalid_mask.any()) {
        return Err(ImageError::new(
            "detector output contains a nonfinite value",
        ));
    }
    Ok(binarize_scalar(&prediction[end..], &mut bitmap[end..], threshold)? || any)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn control_avx<const DEFERRED: bool>(p: &[f32], b: &mut [u8], t: f32) -> ImageResult<bool> {
    binarize_control::<8, DEFERRED>(p, b, t)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn control_avx512<const DEFERRED: bool>(
    p: &[f32],
    b: &mut [u8],
    t: f32,
) -> ImageResult<bool> {
    binarize_control::<16, DEFERRED>(p, b, t)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn packed_avx(p: &[f32], b: &mut [u8], t: f32) -> ImageResult<bool> {
    binarize_packed_vector::<8>(p, b, t)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn packed_avx512(p: &[f32], b: &mut [u8], t: f32) -> ImageResult<bool> {
    binarize_packed_vector::<16>(p, b, t)
}

#[derive(Clone, Copy, Debug)]
enum Variant {
    Original,
    Deferred,
    Packed,
    Selected,
}

#[inline(never)]
fn run_variant(
    variant: Variant,
    kernel: PixelKernel,
    prediction: &[f32],
    bitmap: &mut [u8],
    threshold: f32,
) -> ImageResult<bool> {
    kernel.require();
    if matches!(variant, Variant::Selected) {
        return binarize(kernel, prediction, bitmap, threshold);
    }
    #[cfg(target_arch = "x86_64")]
    // SAFETY: require() has checked the ISA before any of these calls.
    unsafe {
        if kernel == PixelKernel::Avx512 {
            return match variant {
                Variant::Original => control_avx512::<false>(prediction, bitmap, threshold),
                Variant::Deferred => control_avx512::<true>(prediction, bitmap, threshold),
                Variant::Packed => packed_avx512(prediction, bitmap, threshold),
                Variant::Selected => unreachable!(),
            };
        }
        if matches!(kernel, PixelKernel::Avx | PixelKernel::Avx2) {
            return match variant {
                Variant::Original => control_avx::<false>(prediction, bitmap, threshold),
                Variant::Deferred => control_avx::<true>(prediction, bitmap, threshold),
                Variant::Packed => packed_avx(prediction, bitmap, threshold),
                Variant::Selected => unreachable!(),
            };
        }
    }
    match variant {
        Variant::Original => binarize_control::<4, false>(prediction, bitmap, threshold),
        Variant::Deferred => binarize_control::<4, true>(prediction, bitmap, threshold),
        Variant::Packed => binarize_packed_vector::<4>(prediction, bitmap, threshold),
        Variant::Selected => unreachable!(),
    }
}

#[test]
#[ignore = "local release diagnostic: three maps, one warm-up and three single calls per policy; external timeout 10s"]
fn threshold_fixed_diagnostic() -> Result<(), &'static str> {
    if cfg!(debug_assertions) {
        return Err("run this diagnostic in release mode");
    }
    use std::{hint::black_box, time::Instant};
    let started = Instant::now();
    let mut kernels = vec![PixelKernel::Portable];
    let selected = PixelKernel::selected();
    if selected != PixelKernel::Portable {
        kernels.push(selected);
    }
    for kernel in kernels {
        for (width, height) in [(320, 320), (960, 960), (1536, 1537)] {
            let values: Vec<_> = (0..width * height)
                .map(|i| ((i * 101 + i / width * 37) % 1009) as f32 / 1008.)
                .collect();
            let expected: Vec<_> = values.iter().map(|&value| u8::from(value > 0.3)).collect();
            let mut bitmap = vec![197; values.len()];
            let variants = [
                Variant::Original,
                Variant::Deferred,
                Variant::Packed,
                Variant::Selected,
            ];
            let mut samples = [[0.; 4]; 3];
            for variant in variants {
                assert!(run_variant(variant, kernel, &values, &mut bitmap, 0.3).unwrap());
                assert_eq!(bitmap, expected, "{kernel:?}/{variant:?}");
            }
            for (sample, timings) in samples.iter_mut().enumerate() {
                // Rotate order to reduce systematic first/last-policy bias.
                for order in 0..variants.len() {
                    let policy = (order + sample) % variants.len();
                    let start = Instant::now();
                    let any = run_variant(
                        variants[policy],
                        kernel,
                        black_box(&values),
                        black_box(&mut bitmap),
                        black_box(0.3),
                    )
                    .unwrap();
                    timings[policy] = start.elapsed().as_secs_f64() * 1e6;
                    black_box(&bitmap);
                    assert!(any);
                    assert!(
                        started.elapsed().as_secs() < 10,
                        "threshold diagnostic exceeded ten seconds"
                    );
                }
            }
            for (policy, variant) in variants.into_iter().enumerate() {
                let samples = samples.map(|sample| sample[policy]);
                let mut sorted = samples;
                sorted.sort_by(f64::total_cmp);
                println!(
                    "{}",
                    serde_json::json!({
                        "diagnostic": "threshold_fixed",
                        "kernel": format!("{kernel:?}"),
                        "policy": format!("{variant:?}"),
                        "width": width,
                        "height": height,
                        "warmups": 1,
                        "single_call_samples_us": samples,
                        "median_us": sorted[1],
                    })
                );
            }
        }
    }
    Ok(())
}
