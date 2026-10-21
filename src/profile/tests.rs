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
        gpu_admission_wait_ns: u64::MAX - 1,
        gpu_submission_ns: 31,
        gpu_completion_wait_ns: 37,
        gpu_readback_ns: 41,
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
        gpu_admission_wait_ns: 2,
        gpu_submission_ns: 43,
        gpu_completion_wait_ns: 47,
        gpu_readback_ns: 53,
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
    assert_eq!(json["gpu_admission_wait_ns"], u64::MAX);
    assert_eq!(json["gpu_submission_ns"], 74);
    assert_eq!(json["gpu_completion_wait_ns"], 84);
    assert_eq!(json["gpu_readback_ns"], 94);
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
    #[cfg(feature = "profiling")]
    assert_eq!(super::thread_cpu_ns::<false>(), None);
    #[cfg(not(feature = "profiling"))]
    {
        let timer = Timer::<true>::start();
        assert!(timer.0.is_none());
        assert_eq!(timer.elapsed(), 0);
    }
}

#[cfg(feature = "profiling")]
#[test]
fn recognition_owner_merge_keeps_one_snapshot_without_summing_intervals() {
    use super::RecognitionOwnerProfile;
    let record = RecognitionOwnerProfile {
        owner_index: 0,
        dispatch_start_offset_ns: 11,
        dispatch_end_offset_ns: 31,
        wall_ns: 20,
        job_count: 2,
        line_count: 3,
        scheduled_width_cost: 640,
        maximum_tensor_width: 320,
        kernel_threads: 1,
        ..Default::default()
    };
    let mut profile = StageProfile {
        recognition_owner_count: 1,
        ..Default::default()
    };
    profile.recognition_owners[0] = record;
    // Ordinary worker merges add stage work but never duplicate owner records.
    profile.merge(StageProfile {
        rec_backbone_ns: 100,
        recognized_lines: 3,
        ..Default::default()
    });
    assert_eq!(profile.recognition_owner_count, 1);
    assert_eq!(profile.recognition_owners[0], record);
    let mut next = StageProfile {
        recognition_owner_count: 1,
        recognized_lines: 2,
        ..Default::default()
    };
    next.recognition_owners[0] = RecognitionOwnerProfile {
        line_count: 2,
        ..record
    };
    profile.merge(next);
    assert_eq!(profile.recognized_lines, 5);
    assert_eq!(profile.recognition_owner_count, 1);
    assert_eq!(profile.recognition_owners[0].line_count, 2);
    assert_eq!(profile.recognition_owners[0].wall_ns, 20);
    assert!(
        profile.recognition_owners[1..]
            .iter()
            .all(|record| *record == RecognitionOwnerProfile::default())
    );
    let json = serde_json::to_value(profile).unwrap();
    assert_eq!(json["recognition_owner_count"], 1);
    assert_eq!(json["recognition_owners"][0]["scheduled_width_cost"], 640);
    assert!(json["recognition_owners"][0]["thread_cpu_ns"].is_null());
}

#[cfg(feature = "profiling")]
#[test]
fn recognition_owner_records_account_for_serial_parallel_and_same_width_batches() {
    use super::RecognitionOwnerProfile;
    use std::sync::Arc;
    let mut pixels = vec![0; 96 * 96 * 3];
    for top in [10, 58] {
        for y in top..top + 13 {
            for x in 12..52 {
                pixels[(y * 96 + x) * 3..(y * 96 + x) * 3 + 3].fill(255);
            }
        }
    }
    let image = ImageView::new(&pixels, 96, 96, 0, PixelFormat::Bgr24).unwrap();
    #[cfg(feature = "parallel")]
    let capacities = [1, 2];
    #[cfg(not(feature = "parallel"))]
    let capacities = [1];
    for capacity in capacities {
        for batch_size in [1, 2] {
            let cpu = Arc::new(CpuContext::new(capacity).unwrap());
            let mut pipeline = crate::ocr::tests::synthetic_ocr(cpu);
            pipeline.set_recognition_batch_size(batch_size).unwrap();
            let expected = pipeline.run(image, true).unwrap().try_to_owned().unwrap();
            let (actual, profile) = pipeline.run_profiled(image, true).unwrap();
            assert_eq!(actual.try_to_owned().unwrap(), expected);
            let expected_count = if batch_size == 2 {
                1
            } else if capacity > 1 {
                2
            } else {
                0
            };
            assert_eq!(profile.recognition_owner_count, expected_count);
            assert_eq!(profile.recognized_lines, 2);
            let records = &profile.recognition_owners[..expected_count];
            if expected_count != 0 {
                assert_eq!(records.iter().map(|r| r.line_count).sum::<usize>(), 2);
                assert_eq!(
                    records.iter().map(|r| r.job_count).sum::<usize>(),
                    if batch_size == 2 { 1 } else { 2 }
                );
                assert_eq!(
                    records
                        .iter()
                        .map(|r| r.scheduled_width_cost)
                        .sum::<usize>(),
                    expected
                        .lines
                        .iter()
                        .map(|line| line.recognition_tensor_width)
                        .sum::<usize>()
                );
            }
            for (index, record) in records.iter().enumerate() {
                assert_eq!(record.owner_index, index);
                assert_eq!(
                    record.maximum_tensor_width,
                    expected.lines[0].recognition_tensor_width
                );
                assert_eq!(
                    record.scheduled_width_cost,
                    record.maximum_tensor_width * record.line_count
                );
                assert!((1..=capacity).contains(&record.kernel_threads));
                assert_eq!(record.rayon_worker_index.is_some(), capacity > 1);
                assert!(record.dispatch_end_offset_ns >= record.dispatch_start_offset_ns);
                assert_eq!(
                    record.wall_ns,
                    record.dispatch_end_offset_ns - record.dispatch_start_offset_ns
                );
                assert!(record.dispatch_end_offset_ns <= profile.recognition_wall_ns);
                #[cfg(not(target_os = "macos"))]
                assert_eq!(record.thread_cpu_ns, None);
            }
            assert!(
                profile.recognition_owners[expected_count..]
                    .iter()
                    .all(|record| *record == RecognitionOwnerProfile::default())
            );
        }
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
            Some(crate::ocr::tests::synthetic_classifier().into()),
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
