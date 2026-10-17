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
        cls_preprocess_ns: 5,
        cls_backbone_ns: 7,
        cls_head_ns: 11,
        cls_rotation_ns: 13,
        recognized_lines: 1,
        ..Default::default()
    };
    profile.merge(StageProfile {
        binding_ns: 9,
        reshape_ns: 800,
        recognition_wall_ns: 23,
        cls_preprocess_ns: 17,
        cls_backbone_ns: 19,
        cls_head_ns: 23,
        cls_rotation_ns: 29,
        recognized_lines: 2,
        ..Default::default()
    });
    let json = serde_json::to_value(profile).unwrap();
    assert_eq!(json["total_ns"], 100);
    assert_eq!(json["binding_ns"], u64::MAX);
    assert_eq!(json["reshape_ns"], 1500);
    assert_eq!(json["recognition_wall_ns"], 42);
    assert_eq!(json["cls_preprocess_ns"], 22);
    assert_eq!(json["cls_backbone_ns"], 26);
    assert_eq!(json["cls_head_ns"], 34);
    assert_eq!(json["cls_rotation_ns"], 42);
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
    let mut classifier = crate::orientation::Classifier::new(
        crate::ocr::tests::synthetic_classifier(),
        crate::runtime::ExecutionLimits::default(),
    )
    .unwrap();
    classifier.set_execution_context(cpu.clone(), 1).unwrap();
    let expected_classification = classifier.classify(image).unwrap();
    assert_eq!(
        classifier
            .classify_profiled::<false>(image, &mut profile)
            .unwrap(),
        expected_classification
    );
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
fn profiled_classifier_and_oriented_ocr_preserve_complete_results() {
    use crate::{
        detector::{Detector, DetectorOptions},
        lines::{LineOptions, OrientedLineExtractor},
        orientation::Classifier,
        runtime::ExecutionLimits,
    };
    use std::sync::Arc;

    let pixels = crate::ocr::tests::two_lines();
    let image = ImageView::new(&pixels, 96, 96, 0, PixelFormat::Bgr24).unwrap();
    #[cfg(feature = "parallel")]
    let capacities = [1, 2];
    #[cfg(not(feature = "parallel"))]
    let capacities = [1];
    for capacity in capacities {
        let cpu = Arc::new(CpuContext::new(capacity).unwrap());
        let mut classifier = Classifier::new(
            crate::ocr::tests::synthetic_classifier(),
            ExecutionLimits::default(),
        )
        .unwrap();
        classifier.set_execution_context(cpu.clone(), 1).unwrap();
        let expected = classifier.classify(image).unwrap();
        let mut profile = StageProfile::default();
        assert_eq!(
            classifier
                .classify_profiled::<true>(image, &mut profile)
                .unwrap(),
            expected
        );
        assert_eq!(profile.cls_orientation_ns, 0);
        assert_eq!(profile.cls_rotation_ns, 0);
        let json = serde_json::to_value(profile).unwrap();
        for field in ["cls_preprocess_ns", "cls_backbone_ns", "cls_head_ns"] {
            assert!(json[field].is_u64());
        }

        let mut pipeline = crate::ocr::tests::synthetic_ocr(cpu.clone());
        let source_detector = pipeline.extractor_mut().detector();
        // Tight white crops give the synthetic classifier a positive signal,
        // so both serial and worker aggregation exercise applied rotations.
        let detector = Detector::new(
            source_detector.compiled().clone(),
            DetectorOptions {
                unclip_ratio: 0.05,
                ..source_detector.resolved_options()
            },
            source_detector.execution_limits(),
        )
        .unwrap();
        *pipeline.extractor_mut() = OrientedLineExtractor::new(
            detector,
            Some(crate::ocr::tests::synthetic_classifier()),
            LineOptions::default(),
        )
        .unwrap();
        pipeline.set_cpu_context(cpu).unwrap();
        let expected = pipeline.run(image, true).unwrap().try_to_owned().unwrap();
        assert!(expected.lines.iter().all(|line| {
            line.classification_label == 1 && line.applied_rotation_degrees == 180
        }));
        let (actual, profile) = pipeline.run_profiled(image, true).unwrap();
        assert_eq!(actual.try_to_owned().unwrap(), expected);
        assert_eq!(profile.recognized_lines, 2);
        assert!(profile.cls_rotation_ns <= profile.cls_orientation_ns);
        let cls_parts = profile
            .cls_preprocess_ns
            .saturating_add(profile.cls_backbone_ns)
            .saturating_add(profile.cls_head_ns)
            .saturating_add(profile.cls_rotation_ns);
        assert!(cls_parts <= profile.cls_orientation_ns);
    }
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
