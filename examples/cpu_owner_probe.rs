//! Fixed-count recognition ownership comparison on one 24-thread CPU context.
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

const CAPACITY: usize = 24;

fn check_deadline(started: Instant) -> Result<()> {
    if started.elapsed().as_secs_f64() >= 15. {
        return Err("fixed ownership probe exceeded fifteen seconds".into());
    }
    Ok(())
}

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
            "cargo run --release --locked --example cpu_owner_probe -- [tiny|small|medium] [FIXTURE] [OWNERS]\nOWNERS is a comma-separated list of up to three budgets in 0..=16; default 0,12,16 (zero is automatic). One shared 24-thread CPU context; first 24 allowed Windows CPUs. One warm-up and three measured owned-result calls each. Exact complete output equality required. Fifteen-second internal deadline; also enforce an external timeout."
        );
        return Ok(());
    }
    if args.len() > 3 {
        return Err("expected MODEL FIXTURE OWNERS; use --help".into());
    }
    let owners = args
        .get(2)
        .map_or("0,12,16", String::as_str)
        .split(',')
        .map(str::parse::<usize>)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if owners.is_empty() || owners.len() > 3 || owners.iter().any(|&count| count > 16) {
        return Err("expected one to three owner budgets in 0..=16".into());
    }
    let started = Instant::now();
    let model = args.first().map_or("medium", String::as_str);
    let fixture = args.get(1).map_or("imgs_00018069", String::as_str);
    let affinity = benchmark::platform::affinity(CAPACITY, None)?;
    benchmark::platform::configure(&affinity)?;
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let manifest_path = root.join("tests/baseline/manifest.json");
    let manifest: Manifest = read_json(&manifest_path)?;
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
    let cpu = Arc::new(CpuContext::new(CAPACITY)?);
    let image = ImageView::new(
        &frame.pixels,
        frame.width as usize,
        frame.height as usize,
        0,
        PixelFormat::Bgr24,
    )?;
    let mut control = None;
    let mut cases = Vec::new();
    for &line_workers in &owners {
        check_deadline(started)?;
        let mut request = models
            .clone()
            .builder()
            .cpu_context(cpu.clone())
            .cpu_options(CpuOptions {
                line_workers,
                preprocess_workers: 0,
                detector_threads: 0,
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
        let initial_budgets = format!("{:?}", request.cpu_budgets());
        check_deadline(started)?;
        let warmup_start = Instant::now();
        let warmup = request.run(image, false)?.try_to_owned()?;
        let warmup_ms = warmup_start.elapsed().as_secs_f64() * 1000.;
        check_deadline(started)?;
        if let Some(expected) = &control {
            if &warmup != expected {
                return Err(
                    "ownership configurations produced different complete owned OCR results".into(),
                );
            }
        } else {
            control = Some(warmup);
        }
        let line_count = control.as_ref().unwrap().lines.len();
        let classifier_worker_budget = if line_workers == 0 {
            CAPACITY.min(16)
        } else {
            line_workers.min(CAPACITY).min(16)
        };
        let mut wall_ms = Vec::new();
        let mut cpu_ms = Vec::new();
        for _ in 0..3 {
            check_deadline(started)?;
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
            check_deadline(started)?;
            std::hint::black_box(&output);
            wall_ms.push(elapsed_ms);
            cpu_ms.push(elapsed_cpu_ms);
        }
        let (peak_private_bytes, peak_resident_bytes) = benchmark::platform::peak_memory()?;
        cases.push(serde_json::json!({
            "line_workers": line_workers,
            "initial_budgets": initial_budgets,
            "post_run_budgets": format!("{:?}", request.cpu_budgets()),
            "classifier_worker_budget_derived_from_policy": classifier_worker_budget,
            "active_classifier_workers_derived_from_policy": if settings.direction_classification { classifier_worker_budget.min(line_count) } else { 0 },
            "classifier_threads_per_worker": 1,
            "warmup_ms": warmup_ms,
            "samples_ms": wall_ms,
            "samples_cpu_ms": cpu_ms,
            "retained_activation_bytes": request.retained_activation_bytes(),
            "retained_workspace_bytes": request.retained_workspace_bytes(),
            "retained_kernel_scratch_bytes_subset_of_workspace": request.retained_kernel_scratch_bytes(),
            "process_peak_private_bytes": peak_private_bytes,
            "process_peak_resident_bytes": peak_resident_bytes,
        }));
    }
    check_deadline(started)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "instrumented": false,
            "model": model,
            "fixture": fixture,
            "cpu_context_capacity": CAPACITY,
            "affinity": affinity,
            "pixel_sha256": baseline::hash(&frame.pixels),
            "manifest_sha256": baseline::hash(&std::fs::read(manifest_path)?),
            "order": owners,
            "warmups_per_configuration": 1,
            "measured_runs_per_configuration": 3,
            "capture_ctc_spans": false,
            "complete_owned_outputs_exactly_equal": true,
            "control_output_sha256": baseline::hash(&serde_json::to_vec(control.as_ref().unwrap())?),
            "recognized_lines": control.as_ref().unwrap().lines.len(),
            "cases": cases,
            "execution_seconds": started.elapsed().as_secs_f64(),
            "limitation": "One process and one configuration order; immutable models and a 24-thread CPU pool shared, fresh mutable request per configuration. The automatic setting uses 16 classifiers while explicit twelve owners uses twelve; classifier figures are derived from current source policy for one nonconcurrent request. Process memory peaks are cumulative high-water counters, not isolated request memory. Internal deadline checks cannot interrupt an in-flight OCR call. No steady-state or full qualification claim."
        }))?
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("CPU owner probe: {error}");
        std::process::exit(1);
    }
}
