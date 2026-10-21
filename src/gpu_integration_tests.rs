//! Physical-device integration tests. Explicit invocation never skips missing assets/devices.
use crate::{
    ModelPreset, OcrModelSet,
    cpu::CpuContext,
    detector::DetectorOptions,
    engine::{EngineLimits, OcrEngine},
    image::{ImageView, PixelFormat},
    model::Model,
    ocr::{Ocr, OwnedOcr},
    preprocess::{self, ResizeWorkspace},
    recognizer::{Dictionary, RecognitionScoreMode, RecognizerOptions},
    runtime::{
        CompiledModel, ExecutionLimits, ExecutionTarget, Session,
        gpu::{GpuBackendPreference, GpuCompiledModel, GpuContext, GpuContextOptions, GpuSession},
    },
};
use std::{
    path::PathBuf,
    sync::{Arc, OnceLock},
};

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
        let mut request = synthetic(ExecutionTarget::Gpu(context.clone()), mode);
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
            assert_gpu_profile(&profile, expected.lines.len());
            assert_eq!(request.retained_device_bytes(), retained);
        }
        let thread = std::thread::current().id();
        let cpu = request.cpu_context().clone();
        let mut sibling = synthetic(ExecutionTarget::Gpu(context.clone()), mode);
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
            synthetic(ExecutionTarget::Gpu(context.clone()), mode),
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
            synthetic(ExecutionTarget::Gpu(context.clone()), mode),
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

fn models_dir() -> PathBuf {
    std::env::var_os("SIMD_OCR_MODELS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .join("SimdPaddleOCR/models")
        })
}
fn screenshot() -> image::RgbImage {
    let path = std::env::var_os("SIMD_OCR_TEST_IMAGE")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("reports/ocr-comparison-20261021/screenshot.png")
        });
    image::ImageReader::open(&path)
        .unwrap_or_else(|e| panic!("missing local fixture {}: {e}", path.display()))
        .with_guessed_format()
        .unwrap()
        .decode()
        .unwrap()
        .to_rgb8()
}

#[cfg(feature = "profiling")]
fn assert_gpu_profile(profile: &crate::StageProfile, line_count: usize) {
    assert_eq!(profile.recognized_lines, line_count);
    assert_eq!(profile.recognition_owner_count, 0);
    assert_eq!(profile.projection_ns, 0);
    assert_eq!(profile.cls_head_ns, 0);
    assert!(profile.gpu_submission_ns > 0);
    assert!(profile.gpu_completion_wait_ns > 0);
    assert!(profile.gpu_readback_ns > 0);
    let gpu_host_ns = profile
        .gpu_admission_wait_ns
        .saturating_add(profile.gpu_submission_ns)
        .saturating_add(profile.gpu_completion_wait_ns)
        .saturating_add(profile.gpu_readback_ns);
    assert!(gpu_host_ns <= profile.total_ns);
}

#[test]
#[ignore = "requires a physical GPU and the seven local reference models"]
fn gpu_ocr_real_models_match_cpu_at_fixed_shapes() {
    let context = hardware();
    let pixels = screenshot();
    let image = ImageView::new(
        pixels.as_raw(),
        pixels.width() as usize,
        pixels.height() as usize,
        0,
        PixelFormat::Rgb24,
    )
    .unwrap();
    for file in [
        "cls.onnx",
        "det.onnx",
        "small_det.onnx",
        "medium_det.onnx",
        "rec.onnx",
        "small_rec.onnx",
        "medium_rec.onnx",
    ] {
        let model = Arc::new(Model::load_path(models_dir().join(file)).unwrap());
        let shape = if file == "cls.onnx" {
            [1, 3, 80, 160]
        } else if file.contains("det") {
            [1, 3, 128, 128]
        } else {
            [2, 3, 48, 128]
        };
        let mut input = vec![0.; shape.iter().product()];
        let mut workspace = ResizeWorkspace::default();
        if file == "cls.onnx" {
            preprocess::cls(image, &mut input, &mut workspace).unwrap();
        } else if file.contains("det") {
            preprocess::det(image, shape[3], shape[2], &mut input, &mut workspace).unwrap();
        } else {
            for sample in input.chunks_exact_mut(shape[1] * shape[2] * shape[3]) {
                preprocess::rec(image, shape[3], sample, &mut workspace).unwrap();
            }
        }
        let cpu = Arc::new(CompiledModel::new(model.clone()).unwrap());
        let mut reference = Session::new_serial(cpu, &shape, ExecutionLimits::default()).unwrap();
        let expected = reference.run(&input).unwrap();
        let gpu = Arc::new(GpuCompiledModel::compile(model, context.clone()).unwrap());
        let mut session = GpuSession::new(gpu, &shape, ExecutionLimits::default()).unwrap();
        let actual = session.run(&input).unwrap();
        assert_eq!(actual.shape(), expected.shape(), "{file}: shape");
        let mut maximum = 0f32;
        let mut sum = 0f64;
        for (index, (&a, &b)) in actual.data().iter().zip(expected.data()).enumerate() {
            let error = (a - b).abs();
            maximum = maximum.max(error);
            sum += f64::from(error);
            assert!(
                a.is_finite() && b.is_finite() && error <= 1e-3 + 1e-3 * b.abs(),
                "{file}[{index}]: GPU={a}, CPU={b}, error={error}"
            );
        }
        eprintln!(
            "{file}: {:?}, max_abs={maximum:.7}, mean_abs={:.7}",
            actual.shape().dimensions(),
            sum / actual.data().len() as f64
        );
    }
}

#[test]
#[ignore = "requires a physical GPU, local models, and the local OCR screenshot"]
fn gpu_ocr_public_all_presets_and_score_modes_match_cpu() {
    let context = hardware();
    let pixels = screenshot();
    let image = ImageView::new(
        pixels.as_raw(),
        pixels.width() as usize,
        pixels.height() as usize,
        0,
        PixelFormat::Rgb24,
    )
    .unwrap();
    for preset in [
        ModelPreset::ChineseV6Tiny,
        ModelPreset::ChineseV6Small,
        ModelPreset::ChineseV6Medium,
    ] {
        let models = OcrModelSet::from_directory(models_dir(), preset, true).unwrap();
        for mode in [
            RecognitionScoreMode::Logit,
            RecognitionScoreMode::Probability,
        ] {
            let options = RecognizerOptions {
                score_mode: mode,
                ..Default::default()
            };
            let mut cpu = models
                .clone()
                .builder()
                .recognizer_options(options)
                .build()
                .unwrap();
            let expected = cpu.run(image, true).unwrap().try_to_owned().unwrap();
            let mut gpu = models
                .clone()
                .builder()
                .execution_target(ExecutionTarget::Gpu(context.clone()))
                .recognizer_options(options)
                .build()
                .unwrap();
            #[cfg(feature = "profiling")]
            let actual = {
                let (actual, profile) = gpu.run_profiled(image, true).unwrap();
                let actual = actual.try_to_owned().unwrap();
                assert_gpu_profile(&profile, expected.lines.len());
                actual
            };
            #[cfg(not(feature = "profiling"))]
            let actual = gpu.run(image, true).unwrap().try_to_owned().unwrap();
            assert_ocr_close(&actual, &expected, &format!("{preset:?}/{mode:?}"));
            if mode == RecognitionScoreMode::Probability {
                for (actual, expected) in actual.lines.iter().zip(&expected.lines) {
                    assert!(
                        (actual.recognition_score - expected.recognition_score).abs() <= 1e-3,
                        "{preset:?}: probability confidence"
                    );
                }
            }
            for line in &actual.lines {
                line.as_view().estimate_character_boxes().unwrap();
            }
            eprintln!("{preset:?}/{mode:?}: {:?}", actual.text);
        }
    }
}
