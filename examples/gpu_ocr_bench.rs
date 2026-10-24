//! Bounded physical-device OCR comparison. Build release; one cold run and three warm measurements.
#![recursion_limit = "256"]
use simd_paddle_ocr_rs::ocr::OwnedOcr;
use simd_paddle_ocr_rs::{
    ExecutionTarget, ModelPreset, OcrModelSet,
    cpu::{CpuContext, CpuOptions},
    image::{ImageView, PixelFormat},
    runtime::gpu::*,
};
use std::{error::Error, time::Instant};
#[path = "support/ocr_compare.rs"]
mod ocr_compare;

fn main() -> Result<(), Box<dyn Error>> {
    if cfg!(debug_assertions) {
        return Err("benchmark requires --release".into());
    }
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        return Err("Usage: gpu_ocr_bench BACKEND PIPELINE IMAGE [REQUESTS=1] [SLOTS=1] [BATCH=1] [PRESET=tiny]".into());
    }
    let backend = match args[1].as_str() {
        "vulkan" => GpuBackendPreference::Vulkan,
        "dx12" => GpuBackendPreference::Dx12,
        "metal" => GpuBackendPreference::Metal,
        _ => return Err("invalid backend".into()),
    };
    let (target, mode) = match args[2].as_str() {
        "cpu" => (ExecutionTarget::Cpu, GpuPipelineMode::Full),
        "full" => (ExecutionTarget::Gpu, GpuPipelineMode::Full),
        "inference-only" => (ExecutionTarget::Gpu, GpuPipelineMode::InferenceOnly),
        _ => return Err("invalid pipeline".into()),
    };
    let requests = args
        .get(4)
        .map(|x| x.parse())
        .transpose()?
        .unwrap_or(1usize);
    let slots = args
        .get(5)
        .map(|x| x.parse())
        .transpose()?
        .unwrap_or(1usize);
    let batch = args
        .get(6)
        .map(|x| x.parse())
        .transpose()?
        .unwrap_or(1usize);
    if requests == 0 || requests > 16 || slots == 0 || slots > 16 || batch == 0 || batch > 64 {
        return Err("invalid requests/slots/batch bound".into());
    }
    let preset = match args.get(7).map(String::as_str).unwrap_or("tiny") {
        "tiny" => ModelPreset::ChineseV6Tiny,
        "small" => ModelPreset::ChineseV6Small,
        "medium" => ModelPreset::ChineseV6Medium,
        _ => return Err("invalid preset".into()),
    };
    let capture = std::env::var("SIMD_OCR_BENCH_CAPTURE").as_deref() == Ok("1");
    let profile = std::env::var("SIMD_OCR_BENCH_PROFILE").as_deref() == Ok("1");
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
        image::ImageReader::open(&args[3])?
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
            backend,
            max_in_flight: slots,
            ..Default::default()
        })?)
    } else {
        None
    };
    if let Some(context) = &context {
        assert!(matches!(
            context.adapter_info().device_type,
            wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::IntegratedGpu
        ));
    }
    let models = OcrModelSet::from_directory("../SimdPaddleOCR/models", preset, true)?;
    let mut builder = models
        .builder()
        .execution_target(target)
        .gpu_pipeline_mode(mode)
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
    let mut errors = Vec::new();
    let mut failed_outputs = Vec::new();
    if let Ok(path) = std::env::var("SIMD_OCR_BENCH_REFERENCE") {
        let expected: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
        for result in &reference {
            if let Err(error) = ocr_compare::compare(result, &expected, capture) {
                errors.push(error.to_string());
            }
        }
    }
    let mut measured = Vec::new();
    let mut statistics = Vec::new();
    for _ in 0..measurements {
        if !errors.is_empty() {
            break;
        }
        if wall.elapsed().as_secs() >= 60 {
            return Err("60 second measurement budget exhausted".into());
        }
        let (time, actual) = execute(&mut workers)?;
        if actual != reference {
            errors.push("nondeterministic OCR outputs".into());
            failed_outputs.push(serde_json::json!({"phase":"warm","outputs":actual}));
            break;
        }
        measured.push(time);
        statistics.push(
            workers
                .iter()
                .map(|w| w.gpu_pipeline_statistics())
                .collect::<Vec<_>>(),
        );
    }
    let mut sorted = measured.clone();
    sorted.sort_by(f64::total_cmp);
    let median = sorted.get(sorted.len() / 2).copied();
    let device = context.as_ref().map(|c| c.memory_statistics());
    let mut profiles = Vec::<serde_json::Value>::new();
    if profile && errors.is_empty() {
        #[cfg(feature = "profiling")]
        for (worker, expected) in workers.iter_mut().zip(&reference) {
            let (actual, stages) = worker.run_profiled(image, capture)?;
            let actual = actual.try_to_owned()?;
            if actual != *expected {
                errors.push("profiling changes OCR output".into());
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
            "adapter":context.as_ref().map(|c|format!("{:?}",c.adapter_info())),"pipeline":args[2],"image":args[3],"requests":requests,"slots":slots,"batch":batch,"capture":capture,
            "cpu_options":{"line_workers":4,"detector_threads":8,"preprocess_workers":16},
            "timing_contract":"group start through last inference return; excludes decode, owned copy, validation, serialization and profile calls",
            "qualified":errors.is_empty(),"errors":errors,"failed_outputs":failed_outputs,
            "startup_ms":startup_ms,"cold_run_ms":cold_ms,"warm_group_ms":measured,"median_group_ms":median,"images_per_second":median.map(|ms|requests as f64*1000./ms),
            "retained_device_bytes":device.map(|d|d.used_bytes),"peak_device_bytes":device.map(|d|d.peak_bytes),"statistics":statistics.last(),"sample_statistics":statistics,"profiles":profiles,"outputs":reference
        }))?
    );
    Ok(())
}
