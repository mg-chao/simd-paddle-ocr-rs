//! Bounded physical-device OCR comparison. Build release; one cold run and three warm measurements.
#![recursion_limit = "256"]
#[cfg(not(target_os = "macos"))]
use simd_paddle_ocr_rs::ocr::OwnedOcr;
#[cfg(not(target_os = "macos"))]
use simd_paddle_ocr_rs::{
    ExecutionTarget, ModelPreset, OcrModelSet,
    cpu::{CpuContext, CpuOptions},
    image::{ImageView, PixelFormat},
    runtime::{CompileOptions, GpuMatrixPrecision, gpu::*},
};
#[cfg(not(target_os = "macos"))]
use std::{error::Error, time::Instant};
#[path = "support/ocr_compare.rs"]
#[cfg(not(target_os = "macos"))]
mod ocr_compare;

#[cfg(not(target_os = "macos"))]
fn process_cpu_seconds() -> Option<f64> {
    None
}

#[cfg(not(target_os = "macos"))]
fn compare_output(
    actual: &OwnedOcr,
    expected: &serde_json::Value,
    capture: bool,
    mixed_matrix: bool,
) -> Result<(), Box<dyn Error>> {
    if mixed_matrix {
        ocr_compare::compare_fast(actual, expected, capture)
    } else {
        ocr_compare::compare(actual, expected, capture)
    }
}

#[cfg(not(target_os = "macos"))]
fn main() -> Result<(), Box<dyn Error>> {
    if cfg!(debug_assertions) {
        return Err("benchmark requires --release".into());
    }
    let args: Vec<String> = std::env::args().collect();
    if args
        .get(1)
        .is_some_and(|arg| arg == "--validate-fast-output")
    {
        if !(4..=5).contains(&args.len()) || args.get(4).is_some_and(|arg| arg != "--alignment") {
            return Err("Usage: gpu_ocr_bench --validate-fast-output CAPTURE_JSON CPU_REFERENCE_JSON [--alignment]".into());
        }
        let saved: serde_json::Value = serde_json::from_slice(&std::fs::read(&args[2])?)?;
        let expected = serde_json::from_slice(&std::fs::read(&args[3])?)?;
        let outputs = saved["outputs"].as_array().ok_or("missing saved outputs")?;
        if outputs.is_empty() {
            return Err("saved capture contains no outputs".into());
        }
        let capture = args.len() == 5;
        let mut errors = Vec::new();
        let mut differences = Vec::new();
        for actual in outputs {
            if let Err(error) = ocr_compare::compare_fast_values(actual, &expected, capture) {
                errors.push(error.to_string());
            }
            differences.push(ocr_compare::fast_differences_values(
                actual, &expected, capture,
            )?);
        }
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "validation_only":true,"qualified":errors.is_empty(),"errors":errors,
                "saved_output":args[2],"reference":args[3],"capture":capture,
                "accuracy_differences":differences
            }))?
        );
        return if errors.is_empty() {
            Ok(())
        } else {
            Err("saved fast output failed validation".into())
        };
    }
    if args.len() < 3 {
        return Err(
            "Usage: gpu_ocr_bench cpu|gpu IMAGE [REQUESTS=1] [SLOTS=1] [BATCH=1] [PRESET=tiny]"
                .into(),
        );
    }
    let target = match args[1].as_str() {
        "cpu" => ExecutionTarget::Cpu,
        "gpu" => ExecutionTarget::Gpu,
        _ => return Err("execution target must be cpu or gpu".into()),
    };
    let requests = args
        .get(3)
        .map(|x| x.parse())
        .transpose()?
        .unwrap_or(1usize);
    let slots = args
        .get(4)
        .map(|x| x.parse())
        .transpose()?
        .unwrap_or(1usize);
    let batch = args
        .get(5)
        .map(|x| x.parse())
        .transpose()?
        .unwrap_or(1usize);
    if requests == 0 || requests > 16 || slots == 0 || slots > 16 || batch == 0 || batch > 64 {
        return Err("invalid requests/slots/batch bound".into());
    }
    let preset = match args.get(6).map(String::as_str).unwrap_or("tiny") {
        "tiny" => ModelPreset::ChineseV6Tiny,
        "small" => ModelPreset::ChineseV6Small,
        "medium" => ModelPreset::ChineseV6Medium,
        _ => return Err("invalid preset".into()),
    };
    let capture = std::env::var("SIMD_OCR_BENCH_CAPTURE").as_deref() == Ok("1");
    let profile = std::env::var("SIMD_OCR_BENCH_PROFILE").as_deref() == Ok("1");
    let matrix_precision = match std::env::var("SIMD_OCR_BENCH_MATRIX") {
        Ok(value) => match value.as_str() {
            "f32" => GpuMatrixPrecision::F32,
            "f16" => GpuMatrixPrecision::F16,
            _ => return Err("SIMD_OCR_BENCH_MATRIX must be f32 or f16".into()),
        },
        Err(std::env::VarError::NotPresent) => GpuMatrixPrecision::F32,
        Err(error) => return Err(error.into()),
    };
    let storage_precision = match std::env::var("SIMD_OCR_BENCH_STORAGE").as_deref() {
        Ok("f16") => GpuPrecision::MixedF16,
        Ok("f32") | Err(_) => GpuPrecision::F32,
        Ok(_) => return Err("SIMD_OCR_BENCH_STORAGE must be f32 or f16".into()),
    };
    let mixed_matrix = matrix_precision == GpuMatrixPrecision::F16;
    let measurements: usize = std::env::var("SIMD_OCR_BENCH_MEASUREMENTS")
        .ok()
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(3);
    if measurements > 3 {
        return Err("at most three measurements".into());
    }
    let pixels = if let Ok(path) = std::env::var("SIMD_OCR_BENCH_RGB") {
        image::RgbImage::from_raw(
            std::env::var("SIMD_OCR_BENCH_WIDTH")?.parse()?,
            std::env::var("SIMD_OCR_BENCH_HEIGHT")?.parse()?,
            std::fs::read(path)?,
        )
        .ok_or("raw RGB size mismatch")?
    } else {
        image::ImageReader::open(&args[2])?
            .with_guessed_format()?
            .decode()?
            .to_rgb8()
    };
    let image = ImageView::new(
        pixels.as_raw(),
        pixels.width() as usize,
        pixels.height() as usize,
        0,
        PixelFormat::Rgb24,
    )?;
    let wall = Instant::now();
    let context = if target == ExecutionTarget::Gpu {
        Some(GpuContext::new_with_options(GpuContextOptions {
            max_in_flight: slots,
            recognition_wave_size: std::env::var("SIMD_OCR_BENCH_WAVE")
                .ok()
                .map(|s| s.parse())
                .transpose()?
                .unwrap_or(1),
            classification_batch_size: std::env::var("SIMD_OCR_BENCH_CLASSIFICATION_BATCH")
                .ok()
                .map(|s| s.parse())
                .transpose()?
                .unwrap_or(if mixed_matrix { 16 } else { 4 }),
            ..Default::default()
        })?)
    } else {
        None
    };
    if let Some(context) = &context {
        assert!(matches!(
            context.adapter_info().device_type,
            simd_paddle_ocr_rs::runtime::gpu::GpuDeviceType::DiscreteGpu
                | simd_paddle_ocr_rs::runtime::gpu::GpuDeviceType::IntegratedGpu
        ));
    }
    let models_dir = std::env::var_os("SIMD_OCR_BENCH_MODELS")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| "../SimdPaddleOCR/models".into());
    let models = OcrModelSet::from_directory(models_dir, preset, true)?;
    let mut builder = models
        .builder()
        .execution_target(target)
        .compile_options(CompileOptions {
            gpu: GpuCompileOptions {
                precision: storage_precision,
                matrix_precision,
            },
            ..Default::default()
        })
        .cpu_context(CpuContext::shared_default()?)
        .cpu_options(CpuOptions {
            line_workers: 4,
            detector_threads: 8,
            preprocess_workers: 16,
        })
        .recognition_batch_size(batch);
    if let Some(context) = &context {
        builder = builder.gpu_context(context.clone());
    }
    let template = builder.build()?.request()?;
    let plan = template.plan().clone();
    drop(template);
    let factory = move || plan.request();
    let mut workers = (0..requests)
        .map(|_| factory())
        .collect::<Result<Vec<_>, _>>()?;
    let startup_ms = wall.elapsed().as_secs_f64() * 1000.;
    let execute=|workers:&mut Vec<simd_paddle_ocr_rs::ocr::OcrRequest>|->Result<(f64,Vec<OwnedOcr>),Box<dyn Error>>{
        let started=Instant::now();
        let values=std::thread::scope(|scope|{
            let jobs=workers.iter_mut().map(|worker|scope.spawn(move|| {
                let result = worker.run(image, simd_paddle_ocr_rs::RunOptions { capture_spans: capture })?;
                let elapsed = started.elapsed().as_secs_f64()*1000.;
                Ok::<_,simd_paddle_ocr_rs::image::ImageError>((elapsed,result.try_to_owned()?))
            })).collect::<Vec<_>>();
            jobs.into_iter().map(|job|job.join().expect("OCR worker panic")).collect::<Result<Vec<_>,_>>()
        })?;
        let elapsed = values.iter().map(|(ms,_)| *ms).fold(0.,f64::max);
        Ok((elapsed,values.into_iter().map(|(_,result)|result).collect()))
    };
    let (cold_ms, reference) = execute(&mut workers)?;
    // Correctness-only invocations use zero measured repetitions. Preserve
    // their actual placement and residency evidence outside the timed call.
    let cold_statistics = workers
        .iter()
        .map(|worker| worker.gpu_pipeline_statistics())
        .collect::<Vec<_>>();
    let cold_run_statistics = workers
        .iter()
        .map(|worker| worker.latest_run().cloned())
        .collect::<Vec<_>>();
    let mut errors = Vec::new();
    let mut failed_outputs = Vec::new();
    let mut accuracy_differences = Vec::new();
    if let Ok(path) = std::env::var("SIMD_OCR_BENCH_REFERENCE") {
        let expected: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
        for result in &reference {
            if mixed_matrix {
                accuracy_differences
                    .push(ocr_compare::fast_differences(result, &expected, capture)?);
            }
            if let Err(error) = compare_output(result, &expected, capture, mixed_matrix) {
                errors.push(error.to_string());
            }
        }
    }
    let mut measured = Vec::new();
    let mut process_cpu_ms = Vec::new();
    let mut statistics = Vec::new();
    let mut run_statistics = Vec::new();
    for _ in 0..measurements {
        if !errors.is_empty() {
            break;
        }
        if wall.elapsed().as_secs() >= 60 {
            return Err("60 second measurement budget exhausted".into());
        }
        let cpu_start = process_cpu_seconds();
        let (time, actual) = execute(&mut workers)?;
        let cpu_elapsed = cpu_start
            .zip(process_cpu_seconds())
            .map(|(start, end)| (end - start) * 1000.0);
        if actual != reference {
            errors.push("nondeterministic OCR outputs".into());
            failed_outputs.push(serde_json::json!({"phase":"warm","outputs":actual}));
            break;
        }
        measured.push(time);
        process_cpu_ms.push(cpu_elapsed);
        statistics.push(
            workers
                .iter()
                .map(|w| w.gpu_pipeline_statistics())
                .collect::<Vec<_>>(),
        );
        // Capture per-invocation deltas outside the timed inference interval.
        // These expose shape-cache churn without instrumenting GPU execution.
        run_statistics.push(
            workers
                .iter()
                .map(|w| w.latest_run().cloned())
                .collect::<Vec<_>>(),
        );
    }
    let mut sorted = measured.clone();
    sorted.sort_by(f64::total_cmp);
    let median = sorted.get(sorted.len() / 2).copied();
    let device = context.as_ref().map(|c| c.memory_statistics());
    #[cfg(feature = "profiling")]
    let mut profiles = Vec::<serde_json::Value>::new();
    #[cfg(not(feature = "profiling"))]
    let profiles = Vec::<serde_json::Value>::new();
    if profile && errors.is_empty() {
        #[cfg(feature = "profiling")]
        for (worker, expected) in workers.iter_mut().zip(&reference) {
            let (actual, stages) = worker.run_profiled(image, capture)?;
            let actual = actual.try_to_owned()?;
            let expected = serde_json::to_value(expected)?;
            if let Err(error) = compare_output(&actual, &expected, capture, mixed_matrix) {
                errors.push(format!("profiling changes OCR output: {error}"));
                failed_outputs.push(serde_json::json!({"phase":"profile","outputs":actual}));
            }
            profiles.push(
                serde_json::json!({"stages":stages,"pipeline":worker.gpu_pipeline_statistics()}),
            );
        }
        #[cfg(not(feature = "profiling"))]
        return Err("profiling feature required".into());
    }
    let median = if errors.is_empty() { median } else { None };
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "adapter":context.as_ref().map(|c|format!("{:?}",c.adapter_info())),"pipeline":args[1],"image":args[2],"requests":requests,"slots":slots,"batch":batch,"capture":capture,
            "cpu_options":{"line_workers":4,"detector_threads":8,"preprocess_workers":16},
            "recognition_wave_size":context.as_ref().map(|c|c.options().recognition_wave_size),
            "classification_batch_size":context.as_ref().map(|c|c.options().classification_batch_size),
            "matrix_precision":if mixed_matrix { "f16" } else { "f32" },
            "storage_precision":format!("{storage_precision:?}"),
            "accuracy_contract":{
                "name":if mixed_matrix { "f16_text_geometry" } else { "fp32" },
                "text":"exact",
                "metadata":if mixed_matrix { "exact count, order, orientation and emitted count; internally valid derived shape" } else { "exact" },
                "ctc_columns":if mixed_matrix { "independently valid; differences reported when capture is enabled" } else { "exact when capture is enabled" },
                "bbox_absolute_tolerance":if mixed_matrix { 2.0 } else { 0.5 },
                "bbox_tolerance_metric":"per coordinate",
                "bbox_min_polygon_iou":mixed_matrix.then_some(0.8),
                "score_absolute_tolerance":if mixed_matrix { 0.05 } else { 1e-3 },
                "score_relative_tolerance":if mixed_matrix { 0.01 } else { 1e-3 },
                "score_comparison":if mixed_matrix { "mixed numeric gate for equivalent geometry and dimensions; finite values, probability bounds and raw differences for changed geometry" } else { "numeric gate" }
            },
            "timing_contract":"group start through last inference return; excludes decode, owned copy, validation, serialization and profile calls",
            "host_cpu_contract":"process CPU clock is unavailable on the supported GPU platforms; values are null",
            "warm_process_cpu_ms":process_cpu_ms,
            "qualified":errors.is_empty(),"errors":errors,"failed_outputs":failed_outputs,"accuracy_differences":accuracy_differences,
            "startup_ms":startup_ms,"cold_run_ms":cold_ms,"warm_group_ms":measured,"median_group_ms":median,"images_per_second":median.map(|ms|requests as f64*1000./ms),
            "retained_device_bytes":device.map(|d|d.used_bytes),"peak_device_bytes":device.map(|d|d.peak_bytes),
            "statistics":statistics.last().unwrap_or(&cold_statistics),"cold_statistics":cold_statistics,"cold_run_statistics":cold_run_statistics,
            "sample_statistics":statistics,"sample_run_statistics":run_statistics,"profiles":profiles,"outputs":reference
        }))?
    );
    Ok(())
}

#[cfg(test)]
#[cfg(not(target_os = "macos"))]
mod tests {
    use super::*;
    use simd_paddle_ocr_rs::{detector::DetectionBox, ocr::OwnedOcrLine, recognizer::OwnedCtcSpan};

    fn fixture() -> OwnedOcr {
        OwnedOcr {
            text: "test".into(),
            detected_count: 1,
            detector_resized_width: 64,
            detector_resized_height: 64,
            width_ratio: 1.0,
            height_ratio: 1.0,
            lines: vec![OwnedOcrLine {
                detection: DetectionBox {
                    points: [[0., 0.], [32., 0.], [32., 16.], [0., 16.]],
                    score: 0.9,
                },
                text: "test".into(),
                recognition_score: 20.0,
                classification_score: 0.95,
                classification_label: 0,
                applied_rotation_degrees: 0,
                emitted_count: 1,
                recognition_content_width: 32,
                recognition_tensor_width: 32,
                recognition_time_steps: 4,
                ctc_spans: Some(vec![OwnedCtcSpan {
                    text: "test".into(),
                    score: 20.0,
                    start_column: 0,
                    end_column: 2,
                }]),
            }],
        }
    }

    #[test]
    fn profiling_comparison_uses_the_existing_score_contract() {
        let expected = fixture();
        let reference = serde_json::to_value(&expected).unwrap();
        let mut actual = expected.clone();
        actual.lines[0].recognition_score += 0.0005;
        assert!(compare_output(&actual, &reference, true, false).is_ok());

        actual.lines[0].recognition_score = 20.1;
        actual.lines[0].detection.score += 0.01;
        assert!(compare_output(&actual, &reference, true, false).is_err());
        assert!(compare_output(&actual, &reference, true, true).is_ok());

        actual.lines[0].recognition_score = 21.0;
        assert!(compare_output(&actual, &reference, true, true).is_err());
    }

    #[test]
    fn profiling_comparison_preserves_strict_contract_and_fast_internal_validity() {
        let expected = fixture();
        let reference = serde_json::to_value(&expected).unwrap();
        let mut actual = expected.clone();
        actual.lines[0].recognition_content_width += 1;
        assert!(compare_output(&actual, &reference, true, false).is_err());
        assert!(compare_output(&actual, &reference, true, true).is_err());

        let mut actual = expected.clone();
        actual.lines[0].text.push('!');
        assert!(compare_output(&actual, &reference, true, true).is_err());

        let mut actual = expected;
        actual.lines[0].ctc_spans.as_mut().unwrap()[0].end_column += 1;
        assert!(compare_output(&actual, &reference, true, false).is_err());
        assert!(compare_output(&actual, &reference, true, true).is_ok());
    }
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err("GPU execution is not supported on macOS".into())
}
