//! Semantic accounting contracts; no throughput or timing thresholds.
use super::{StageProfile, Timer};
use crate::{
    cpu::CpuContext,
    image::{ImageView, PixelFormat},
};

#[test]
fn profile_merge_keeps_reshape_work_separate_and_serializable() {
    let mut profile = StageProfile {
        total_ns: 100,
        binding_ns: u64::MAX - 1,
        reshape_ns: 700,
        recognition_wall_ns: 19,
        recognized_lines: 1,
        ..Default::default()
    };
    profile.merge(StageProfile {
        binding_ns: 9,
        reshape_ns: 800,
        recognition_wall_ns: 23,
        recognized_lines: 2,
        ..Default::default()
    });
    let json = serde_json::to_value(profile).unwrap();
    assert_eq!(json["total_ns"], 100);
    assert_eq!(json["binding_ns"], u64::MAX);
    assert_eq!(json["reshape_ns"], 1500);
    assert_eq!(json["recognition_wall_ns"], 42);
    assert_eq!(json["recognized_lines"], 3);
    // Work deliberately exceeds wall time; merging must not subtract it from
    // either wall counter or relabel it as binding.
    let mut saturated = StageProfile {
        reshape_ns: u64::MAX,
        ..Default::default()
    };
    saturated.merge(profile);
    assert_eq!(saturated.reshape_ns, u64::MAX);
}

#[test]
fn disabled_timers_have_no_clock_state() {
    let timer = Timer::<false>::start();
    assert!(timer.0.is_none());
    assert_eq!(timer.elapsed(), 0);
    #[cfg(not(feature = "profiling"))]
    {
        let timer = Timer::<true>::start();
        assert!(timer.0.is_none());
        assert_eq!(timer.elapsed(), 0);
    }
}

#[test]
fn unprofiled_det_single_rec_and_batch_leave_counters_unchanged() {
    let cpu = CpuContext::serial();
    let mut pipeline = crate::ocr::tests::synthetic_ocr(cpu.clone());
    let pixels = crate::ocr::tests::two_lines();
    let image = ImageView::new(&pixels, 96, 96, 0, PixelFormat::Bgr24).unwrap();
    let mut profile = StageProfile {
        total_ns: 7,
        binding_ns: 11,
        reshape_ns: 13,
        recognized_lines: 17,
        ..Default::default()
    };
    let before = serde_json::to_value(profile).unwrap();
    cpu.install(|| {
        let detection = pipeline
            .extractor_mut()
            .detector_mut()
            .detect_profiled::<false>(image, &mut profile)
            .unwrap();
        assert_eq!(detection.boxes.len(), 2);
        let recognizer = pipeline.recognizer_mut();
        let single = recognizer
            .recognize_impl::<false>(image, true, &mut profile)
            .unwrap()
            .try_to_owned()
            .unwrap();
        let batch = recognizer
            .recognize_batch_impl::<false, _>(2, |_| Ok(image), true, &mut profile)
            .unwrap()
            .try_to_owned()
            .unwrap();
        assert_eq!(batch, vec![single.clone(), single]);
    });
    assert_eq!(serde_json::to_value(profile).unwrap(), before);
}

#[cfg(feature = "profiling")]
#[test]
fn profiled_ocr_and_batch_preserve_results_and_publish_separate_fields() {
    let cpu = CpuContext::serial();
    let mut pipeline = crate::ocr::tests::synthetic_ocr(cpu.clone());
    let pixels = crate::ocr::tests::two_lines();
    let image = ImageView::new(&pixels, 96, 96, 0, PixelFormat::Bgr24).unwrap();
    let expected = pipeline.run(image, true).unwrap().try_to_owned().unwrap();
    let (result, profile) = pipeline.run_profiled(image, true).unwrap();
    assert_eq!(result.try_to_owned().unwrap(), expected);
    assert_eq!(profile.recognized_lines, 2);
    let json = serde_json::to_value(profile).unwrap();
    assert!(json["binding_ns"].is_u64());
    assert!(json["reshape_ns"].is_u64());
    // A direct same-width batch exercises its distinct reshape site; public
    // OCR above follows the single-line recognition sites for unequal widths.
    let recognizer = pipeline.recognizer_mut();
    let expected = recognizer
        .recognize(image, true)
        .unwrap()
        .try_to_owned()
        .unwrap();
    let mut profile = StageProfile::default();
    cpu.install(|| {
        let actual = recognizer
            .recognize_batch_impl::<true, _>(2, |_| Ok(image), true, &mut profile)
            .unwrap()
            .try_to_owned()
            .unwrap();
        assert_eq!(actual, vec![expected.clone(), expected]);
    });
    assert_eq!(profile.recognized_lines, 2);
    let json = serde_json::to_value(profile).unwrap();
    assert!(json["binding_ns"].is_u64());
    assert!(json["reshape_ns"].is_u64());
}
