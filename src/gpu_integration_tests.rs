//! Physical-device integration tests. Explicit invocation never skips missing assets/devices.
use crate::{
    OcrModelSet,
    cpu::CpuContext,
    detector::{DetectionTrace, Detector, DetectorOptions},
    engine::{EngineLimits, OcrEngine},
    image::{ImageView, PixelFormat},
    model::Model,
    ocr::{Ocr, OwnedOcr},
    preprocess,
    recognizer::{Dictionary, RecognitionScoreMode, RecognizerOptions},
    runtime::{
        CpuCompiledModel, ExecutionLimits, ExecutionTarget,
        gpu::{GpuBackendPreference, GpuCompiledModel, GpuContext, GpuContextOptions, GpuSession},
    },
};
use std::sync::{Arc, OnceLock};

fn hardware() -> Arc<GpuContext> {
    static CONTEXT: OnceLock<Arc<GpuContext>> = OnceLock::new();
    CONTEXT
        .get_or_init(|| {
            let backend = match std::env::var("SIMD_OCR_GPU_BACKEND")
                .as_deref()
                .unwrap_or("vulkan")
            {
                "auto" => GpuBackendPreference::Auto,
                "vulkan" => GpuBackendPreference::Vulkan,
                "dx12" => GpuBackendPreference::Dx12,
                "metal" => GpuBackendPreference::Metal,
                other => panic!("invalid SIMD_OCR_GPU_BACKEND: {other}"),
            };
            let context = GpuContext::new_with_options(GpuContextOptions {
                backend,
                ..Default::default()
            })
            .expect("explicit GPU integration tests require a physical GPU");
            eprintln!("OCR GPU adapter: {:?}", context.adapter_info());
            assert!(matches!(
                context.adapter_info().device_type,
                wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::IntegratedGpu
            ));
            context
        })
        .clone()
}

fn synthetic(target: ExecutionTarget, mode: RecognitionScoreMode) -> Ocr {
    let (det, rec) = crate::ocr::tests::synthetic_sources();
    let models = OcrModelSet::new(
        Arc::new(Model::load_bytes(&det).unwrap()),
        Arc::new(Model::load_bytes(&rec).unwrap()),
        Arc::new(Dictionary::from_utf8(b"A\n").unwrap()),
        None,
    );
    let capacity = if cfg!(feature = "parallel") { 2 } else { 1 };
    models
        .builder()
        .execution_target(target)
        .gpu_context(hardware())
        .cpu_context(Arc::new(CpuContext::new(capacity).unwrap()))
        .detector_options(DetectorOptions {
            limit_side_length: 96,
            ..Default::default()
        })
        .recognizer_options(RecognizerOptions {
            score_mode: mode,
            ..Default::default()
        })
        .recognition_batch_size(2)
        .build()
        .unwrap()
}

fn assert_ocr_close(actual: &OwnedOcr, expected: &OwnedOcr, label: &str) {
    assert_eq!(actual.text, expected.text, "{label}: text");
    assert_eq!(
        actual.detected_count, expected.detected_count,
        "{label}: detected count"
    );
    assert_eq!(
        actual.lines.len(),
        expected.lines.len(),
        "{label}: line count"
    );
    for (a, b) in actual.lines.iter().zip(&expected.lines) {
        assert_eq!(a.text, b.text, "{label}: reading order");
        assert_eq!(
            a.classification_label, b.classification_label,
            "{label}: orientation"
        );
        assert_eq!(a.applied_rotation_degrees, b.applied_rotation_degrees);
        assert_eq!(a.recognition_tensor_width, b.recognition_tensor_width);
        assert_eq!(a.recognition_time_steps, b.recognition_time_steps);
        assert!(
            (a.recognition_score - b.recognition_score).abs()
                <= 1e-3 + 1e-3 * b.recognition_score.abs(),
            "{label}: recognition score {} vs {}",
            a.recognition_score,
            b.recognition_score
        );
        assert!(
            (a.classification_score - b.classification_score).abs() <= 1e-3,
            "{label}: classification confidence"
        );
        assert!(
            (a.detection.score - b.detection.score).abs() <= 1e-3,
            "{label}: detection confidence"
        );
        for (pa, pb) in a.detection.points.iter().zip(&b.detection.points) {
            for axis in 0..2 {
                assert!((pa[axis] - pb[axis]).abs() <= 1., "{label}: box vertices");
            }
        }
        match (&a.ctc_spans, &b.ctc_spans) {
            (Some(a), Some(b)) => {
                assert_eq!(a.len(), b.len(), "{label}: alignment length");
                for (a, b) in a.iter().zip(b) {
                    assert_eq!(
                        (&a.text, a.start_column, a.end_column),
                        (&b.text, b.start_column, b.end_column)
                    );
                    assert!(
                        (a.score - b.score).abs() <= 1e-3 + 1e-3 * b.score.abs(),
                        "{label}: span score"
                    );
                }
            }
            (None, None) => {}
            _ => panic!("{label}: capture mismatch"),
        }
    }
}

#[test]
#[ignore = "requires a physical GPU; run explicitly"]
fn gpu_detector_bounded_resize_and_padding_preserve_trace_and_replay() {
    let (source, _) = crate::ocr::tests::synthetic_sources();
    let model = Arc::new(Model::load_bytes(&source).unwrap());
    let options = DetectorOptions {
        resize_policy: crate::preprocess::DetResizePolicy::Min,
        limit_side_length: 736,
        ..Default::default()
    };
    let mut reference = Detector::new(
        Arc::new(CpuCompiledModel::new(model.clone()).unwrap()),
        options,
        ExecutionLimits::default(),
    )
    .unwrap();
    let mut detector = Detector::new(
        Arc::new(GpuCompiledModel::compile(model, hardware()).unwrap()),
        options,
        ExecutionLimits::default(),
    )
    .unwrap();
    for (width, height) in [(192, 32), (46, 17), (17, 46)] {
        let (padded_width, padded_height) = if width + height < 64 {
            (width.max(32), height.max(32))
        } else {
            (width, height)
        };
        let mut pixels = vec![0; width * height * 3];
        let mut padded = vec![0; padded_width * padded_height * 3];
        for y in 3..height - 3 {
            pixels[(y * width + 3) * 3..(y * width + width - 3) * 3].fill(255);
        }
        for y in 0..height {
            padded[y * padded_width * 3..y * padded_width * 3 + width * 3]
                .copy_from_slice(&pixels[y * width * 3..(y + 1) * width * 3]);
        }
        let image = ImageView::new(&pixels, width, height, 0, PixelFormat::Bgr24).unwrap();
        let control =
            ImageView::new(&padded, padded_width, padded_height, 0, PixelFormat::Bgr24).unwrap();
        let mut expected = reference.detect(control).unwrap().try_to_owned().unwrap();
        assert_eq!(expected.boxes.len(), 1);
        for point in &mut expected.boxes[0].points {
            point[0] = point[0].clamp(0., width as f32);
            point[1] = point[1].clamp(0., height as f32);
        }
        let actual = detector.detect(image).unwrap().try_to_owned().unwrap();
        assert_eq!(actual.boxes.len(), 1, "{width}x{height}");
        assert_eq!(actual.boxes[0].points, expected.boxes[0].points);
        assert!((actual.boxes[0].score - expected.boxes[0].score).abs() <= 1e-5);
        assert_eq!(actual.width_ratio, expected.width_ratio);
        assert_eq!(actual.height_ratio, expected.height_ratio);
        assert!(actual.resized_width <= preprocess::DET_MAX_SIDE);
        assert!(actual.resized_height <= preprocess::DET_MAX_SIDE);
        if width == 192 {
            assert_eq!(actual.resized_width, preprocess::DET_MAX_SIDE);
        }

        let mut probabilities = Vec::new();
        let traced = detector
            .detect_with_trace(image, |event| {
                if let DetectionTrace::Probabilities { values, .. } = event {
                    probabilities.extend_from_slice(values);
                }
                Ok(())
            })
            .unwrap()
            .try_to_owned()
            .unwrap();
        assert_eq!(traced.boxes, actual.boxes);
        let replayed = detector
            .postprocess(
                &probabilities,
                traced.resized_width,
                traced.resized_height,
                image,
                traced.width_ratio,
                traced.height_ratio,
            )
            .unwrap();
        assert_eq!(replayed.boxes, actual.boxes);
    }
}

#[test]
#[ignore = "requires a physical GPU; run explicitly"]
fn gpu_ocr_synthetic_batches_trace_fork_factory_engine_and_concurrency() {
    let context = hardware();
    let initial_bytes = context.memory_statistics().used_bytes;
    let pixels = crate::ocr::tests::two_lines();
    let image = ImageView::new(&pixels, 96, 96, 0, PixelFormat::Bgr24).unwrap();
    for mode in [
        RecognitionScoreMode::Logit,
        RecognitionScoreMode::Probability,
    ] {
        let mut reference = synthetic(ExecutionTarget::Cpu, mode);
        let expected = reference.run(image, true).unwrap().try_to_owned().unwrap();
        let mut request = synthetic(ExecutionTarget::Gpu, mode);
        assert!(request.uses_gpu());
        let actual = request.run(image, true).unwrap().try_to_owned().unwrap();
        assert_ocr_close(&actual, &expected, "synthetic batch");
        assert_eq!(request.workers_len_for_gpu_test(), 0);
        assert!(request.retained_device_bytes() > 0);
        #[cfg(feature = "profiling")]
        {
            let retained = request.retained_device_bytes();
            let (actual, profile) = request.run_profiled(image, true).unwrap();
            let actual = actual.try_to_owned().unwrap();
            assert_ocr_close(&actual, &expected, "profiled synthetic batch");
            assert_eq!(profile.recognized_lines, expected.lines.len());
            assert!(profile.gpu_submission_ns > 0 && profile.gpu_readback_ns > 0);
            if mode == RecognitionScoreMode::Logit {
                assert!(profile.projection_ns > 0);
            }
            assert!(request.retained_device_bytes() >= retained);
        }
        let thread = std::thread::current().id();
        let cpu = request.cpu_context().clone();
        let mut sibling = synthetic(ExecutionTarget::Gpu, mode);
        let mut observations = 0;
        let traced = request
            .run_with_trace(image, true, |_| {
                assert_eq!(std::thread::current().id(), thread);
                assert!(!cpu.is_worker_thread());
                observations += 1;
                if observations == 1 {
                    assert_ocr_close(
                        &sibling.run(image, true).unwrap().try_to_owned().unwrap(),
                        &expected,
                        "sibling execution from an observer",
                    );
                }
                Ok(())
            })
            .unwrap()
            .try_to_owned()
            .unwrap();
        assert!(observations > 0);
        assert_ocr_close(&traced, &expected, "GPU trace");
        let mut cancelled_after_node = false;
        assert!(
            request
                .run_with_trace(image, true, |event| {
                    if matches!(
                        event,
                        crate::ocr::OcrTrace::Detection(crate::detector::DetectionTrace::Node(_))
                    ) {
                        cancelled_after_node = true;
                        Err(crate::image::ImageError::new(
                            "cancel after device execution",
                        ))
                    } else {
                        Ok(())
                    }
                })
                .is_err()
        );
        assert!(cancelled_after_node);
        assert_ocr_close(
            &request.run(image, true).unwrap().try_to_owned().unwrap(),
            &expected,
            "after cancel",
        );
        let mut fork = request.fork().unwrap();
        assert_ocr_close(
            &fork.run(image, true).unwrap().try_to_owned().unwrap(),
            &expected,
            "fork",
        );
        let factory = request.into_factory();
        let mut fresh = factory().unwrap();
        assert_ocr_close(
            &fresh.run(image, true).unwrap().try_to_owned().unwrap(),
            &expected,
            "factory",
        );
        drop((factory, fresh, fork, sibling));

        let engine = OcrEngine::new(
            synthetic(ExecutionTarget::Gpu, mode),
            EngineLimits {
                max_requests: 2,
                max_idle_bytes: 32 << 20,
            },
        )
        .unwrap();
        std::thread::scope(|scope| {
            let jobs = (0..2)
                .map(|_| scope.spawn(|| engine.run_owned(image, true).unwrap()))
                .collect::<Vec<_>>();
            for job in jobs {
                assert_ocr_close(&job.join().unwrap(), &expected, "concurrent leases");
            }
        });
        let stats = engine.stats();
        assert!(stats.idle_device_bytes > 0);
        assert_eq!(
            stats.idle_host_bytes + stats.idle_device_bytes,
            stats.idle_bytes
        );
        drop(engine);
        let uncached = OcrEngine::new(
            synthetic(ExecutionTarget::Gpu, mode),
            EngineLimits {
                max_requests: 1,
                max_idle_bytes: 1,
            },
        )
        .unwrap();
        let weights_only = context.memory_statistics().used_bytes;
        {
            let mut lease = uncached.acquire().unwrap();
            lease
                .run_with_trace(image, true, |_| {
                    assert!(
                        uncached.acquire().is_err(),
                        "blocking acquisition from a GPU trace must reject reentrancy"
                    );
                    assert!(uncached.try_acquire().unwrap().is_none());
                    Ok(())
                })
                .unwrap();
        }
        assert_eq!(uncached.stats(), crate::engine::EngineStats::default());
        assert_eq!(
            context.memory_statistics().used_bytes,
            weights_only,
            "idle eviction must release device buffers"
        );
    }
    assert_eq!(
        context.memory_statistics().used_bytes,
        initial_bytes,
        "request buffers and weights must release reservations"
    );
}

#[test]
#[ignore = "requires a physical GPU; run explicitly"]
fn gpu_ocr_context_quota_rolls_back_failed_compilation_and_session_allocation() {
    let context = GpuContext::new_with_options(GpuContextOptions {
        backend: hardware().options().backend,
        max_device_bytes: Some(4),
        ..Default::default()
    })
    .unwrap();
    let (det, rec) = crate::ocr::tests::synthetic_sources();
    let error =
        GpuCompiledModel::compile(Arc::new(Model::load_bytes(&rec).unwrap()), context.clone())
            .err()
            .expect("recognition weights must exceed the four-byte context quota");
    assert_eq!(error.kind(), crate::error::ErrorKind::ResourceLimit);
    assert_eq!(context.memory_statistics().used_bytes, 0);
    let compiled = Arc::new(
        GpuCompiledModel::compile(Arc::new(Model::load_bytes(&det).unwrap()), context.clone())
            .unwrap(),
    );
    assert_eq!(compiled.weight_bytes(), 4);
    assert_eq!(context.memory_statistics().used_bytes, 4);
    let error = GpuSession::new(
        compiled.clone(),
        &[1, 3, 32, 32],
        ExecutionLimits::default(),
    )
    .err()
    .expect("session buffers must exceed the remaining quota");
    assert_eq!(error.kind(), crate::error::ErrorKind::ResourceLimit);
    assert_eq!(context.memory_statistics().used_bytes, 4);
    drop(compiled);
    assert_eq!(context.memory_statistics().used_bytes, 0);
}
