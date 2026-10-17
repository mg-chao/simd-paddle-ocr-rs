//! Fixed-count production scheduling comparison; not a platform qualification.
#[path = "../tools/baseline/mod.rs"]
mod baseline;
#[path = "../tools/benchmark/mod.rs"]
mod benchmark;

use baseline::{Manifest, Result, read_json};
use simd_paddle_ocr_rs::{
    OcrModelSet,
    cpu::{CpuContext, CpuOptions},
    detector::DetectorOptions,
    image::{ImageView, PixelFormat},
    lines::LineOptions,
    recognizer::RecognizerOptions,
};
use std::{path::PathBuf, sync::Arc, time::Instant};

fn run() -> Result<()> {
    if cfg!(debug_assertions) || cfg!(feature = "profiling") || !cfg!(feature = "parallel") {
        return Err("use --release with default features and without profiling".into());
    }
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args
        .first()
        .is_some_and(|arg| arg == "--help" || arg == "-h")
    {
        println!(
            "cargo run --release --locked --example cpu_scheduling_probe -- [tiny|small|medium] [FIXTURE]\nEight logical CPUs; automatic then eight line workers; one warm-up and three measured owned-result calls each. Exact output equality required. Single process order, no automatic retries."
        );
        return Ok(());
    }
    if args.len() > 2 {
        return Err("expected MODEL FIXTURE; use --help".into());
    }
    let model = args.first().map_or("tiny", String::as_str);
    let fixture = args.get(1).map_or("imgs_00018069", String::as_str);
    let started = Instant::now();
    let affinity = benchmark::platform::affinity(8, None)?;
    benchmark::platform::configure(&affinity)?;
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let manifest: Manifest = read_json(&root.join("tests/baseline/manifest.json"))?;
    manifest.validate()?;
    let directory = root.join("../SimdPaddleOCR/models");
    manifest.check_assets(&directory)?;
    let frames = baseline::decode_frames(&manifest, &root.join("tests/baseline"), true)?;
    let frame = frames
        .iter()
        .find(|f| f.id == fixture)
        .ok_or("unknown fixture")?;
    let spec = manifest
        .models
        .iter()
        .find(|m| m.id == model)
        .ok_or("unknown model")?;
    let settings = &manifest.settings;
    let models = OcrModelSet::from_paths(
        directory.join(&spec.det),
        directory.join(&spec.rec),
        directory.join(&spec.dictionary),
        Some(&directory.join(&spec.cls)),
    )?;
    let cpu = Arc::new(CpuContext::new(8)?);
    let image = ImageView::new(
        &frame.pixels,
        frame.width as usize,
        frame.height as usize,
        0,
        PixelFormat::Bgr24,
    )?;
    let mut control = None;
    let mut cases = Vec::new();
    for line_workers in [0, 8] {
        let mut request = models
            .clone()
            .builder()
            .cpu_context(cpu.clone())
            .cpu_options(CpuOptions {
                line_workers,
                preprocess_workers: 8,
                detector_threads: 8,
            })
            .detector_options(DetectorOptions {
                limit_side_length: settings.detector_limit,
                max_candidates: settings.max_candidates,
                bitmap_threshold: settings.bitmap_threshold,
                box_threshold: Some(spec.box_threshold),
                unclip_ratio: settings.unclip_ratio,
                use_dilation: settings.use_dilation,
                ..Default::default()
            })
            .recognizer_options(RecognizerOptions {
                target_width: settings.target_width,
                adaptive_width: settings.adaptive_width,
                ..Default::default()
            })
            .line_options(LineOptions {
                classifier_threshold: settings.classifier_threshold,
                ..Default::default()
            })
            .direction_classification(settings.direction_classification)
            .recognition_batch_size(1)
            .build()?;
        let warmup_start = Instant::now();
        let warmup = request.run(image, false)?.try_to_owned()?;
        let warmup_ms = warmup_start.elapsed().as_secs_f64() * 1000.;
        if let Some(expected) = &control {
            if &warmup != expected {
                return Err(
                    "scheduling configurations produced different owned OCR results".into(),
                );
            }
        } else {
            control = Some(warmup);
        }
        let mut wall_ms = Vec::new();
        let mut cpu_ms = Vec::new();
        for _ in 0..3 {
            if started.elapsed().as_secs() >= 10 {
                return Err("fixed scheduling probe exceeded ten seconds".into());
            }
            let cpu_start = benchmark::platform::process_cpu_time()?;
            let start = Instant::now();
            let output = request.run(image, false)?.try_to_owned()?;
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.;
            let elapsed_cpu_ms = benchmark::platform::process_cpu_time()?
                .checked_sub(cpu_start)
                .ok_or("process CPU time moved backwards")?
                .as_secs_f64()
                * 1000.;
            if Some(&output) != control.as_ref() {
                return Err("measured output differs from automatic warm-up control".into());
            }
            std::hint::black_box(&output);
            wall_ms.push(elapsed_ms);
            cpu_ms.push(elapsed_cpu_ms);
        }
        cases.push(serde_json::json!({
            "line_workers": line_workers,
            "budgets": format!("{:?}", request.cpu_budgets()),
            "warmup_ms": warmup_ms,
            "samples_ms": wall_ms,
            "samples_cpu_ms": cpu_ms,
            "retained_activation_bytes": request.retained_activation_bytes(),
            "retained_workspace_bytes": request.retained_workspace_bytes(),
        }));
    }
    if started.elapsed().as_secs() >= 10 {
        return Err("fixed scheduling probe exceeded ten seconds".into());
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "instrumented": false,
            "model": model,
            "fixture": fixture,
            "affinity": affinity,
            "pixel_sha256": baseline::hash(&frame.pixels),
            "manifest_sha256": baseline::hash(&std::fs::read(root.join("tests/baseline/manifest.json"))?),
            "order": ["automatic", "eight_line_workers"],
            "warmups_per_configuration": 1,
            "measured_runs_per_configuration": 3,
            "complete_owned_outputs_exactly_equal": true,
            "recognized_lines": control.as_ref().unwrap().lines.len(),
            "cases": cases,
            "execution_seconds": started.elapsed().as_secs_f64(),
            "limitation": "One process and one configuration order; shared immutable models and CPU pool; fresh mutable request per configuration. No steady-state or full qualification claim."
        }))?
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("CPU scheduling probe: {error}");
        std::process::exit(1);
    }
}
