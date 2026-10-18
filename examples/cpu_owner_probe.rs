//! Fixed-count recognition ownership comparison on one 24-thread CPU context.
#[path = "../tools/baseline/mod.rs"]
mod baseline;
#[path = "../tools/benchmark/mod.rs"]
mod benchmark;

use baseline::{read_json, Manifest, Result};
use simd_paddle_ocr_rs::{
    cpu::{CpuContext, CpuOptions},
    detector::DetectorOptions,
    image::{ImageView, PixelFormat},
    lines::LineOptions,
    recognizer::RecognizerOptions,
    OcrModelSet,
};
use std::{io::Write, path::PathBuf, sync::Arc, time::Instant};

const CAPACITY: usize = 24;

fn event(value: &serde_json::Value) -> Result<()> {
    println!("{}", serde_json::to_string(value)?);
    std::io::stdout().flush()?;
    Ok(())
}

fn command(expected: &str) -> Result<()> {
    let mut line = String::new();
    if std::io::stdin().read_line(&mut line)? == 0 || line.trim() != expected {
        return Err(format!("paired probe expected {expected:?} on stdin").into());
    }
    Ok(())
}

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
    let mut args: Vec<_> = std::env::args().skip(1).collect();
    let paired = args.first().is_some_and(|arg| arg == "--paired");
    if paired {
        args.remove(0);
    }
    if args
        .first()
        .is_some_and(|arg| arg == "--help" || arg == "-h")
    {
        println!(
            "cargo run --release --locked --example cpu_owner_probe -- [tiny|small|medium] [FIXTURE] [OWNERS] [BATCH_SIZES]\nOWNERS is a comma-separated list of up to three budgets in 0..=16; default 0,12,16 (zero is automatic). BATCH_SIZES is one to three comma-separated sizes in 1..=4; default 1. One shared 24-thread CPU context; first 24 allowed Windows CPUs. One warm-up and three measured owned-result calls each. Exact complete output equality required. Fifteen-second internal deadline; also enforce an external timeout."
        );
        println!(
            "Optional --paired prefix enables a JSON-lines protocol for one owner/batch configuration: ready -> stdin warmup -> warmup result -> three stdin run commands/sample results -> complete report. No inference runs while waiting for commands. SIMD_OCR_PROBE_ROOT can select the common repository data root when comparing source snapshots. Enforce an external deadline including waits."
        );
        return Ok(());
    }
    if args.len() > 4 {
        return Err("expected MODEL FIXTURE OWNERS [BATCH_SIZES]; use --help".into());
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
    let batch_sizes = args
        .get(3)
        .map_or("1", String::as_str)
        .split(',')
        .map(str::parse::<usize>)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if batch_sizes.is_empty()
        || batch_sizes.len() > 3
        || batch_sizes.iter().any(|size| !(1..=4).contains(size))
    {
        return Err("expected one to three batch sizes in 1..=4".into());
    }
    if paired && (owners.len() != 1 || batch_sizes.len() != 1) {
        return Err("paired mode requires exactly one owner and batch configuration".into());
    }
    let started = Instant::now();
    let model = args.first().map_or("medium", String::as_str);
    let fixture = args.get(1).map_or("imgs_00018069", String::as_str);
    let affinity = benchmark::platform::affinity(CAPACITY, None)?;
    benchmark::platform::configure(&affinity)?;
    let root = std::env::var_os("SIMD_OCR_PROBE_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")));
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
    let import_start = Instant::now();
    let models = OcrModelSet::from_paths(
        directory.join(&spec.det),
        directory.join(&spec.rec),
        directory.join(&spec.dictionary),
        Some(&directory.join(&spec.cls)),
    )?;
    let import_ms = import_start.elapsed().as_secs_f64() * 1000.;
    let context_start = Instant::now();
    let cpu = Arc::new(CpuContext::new(CAPACITY)?);
    let context_ms = context_start.elapsed().as_secs_f64() * 1000.;
    let image = ImageView::new(
        &frame.pixels,
        frame.width as usize,
        frame.height as usize,
        0,
        PixelFormat::Bgr24,
    )?;
    let mut control = None;
    let mut cases = Vec::new();
    for (line_workers, batch_size) in owners
        .iter()
        .flat_map(|&owner| batch_sizes.iter().map(move |&batch| (owner, batch)))
    {
        check_deadline(started)?;
        let setup_start = Instant::now();
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
            .recognition_batch_size(batch_size)
            .build()?;
        let compile_request_ms = setup_start.elapsed().as_secs_f64() * 1000.;
        let initial_budgets = format!("{:?}", request.cpu_budgets());
        check_deadline(started)?;
        if paired {
            event(&serde_json::json!({
                "event": "ready", "model": model, "fixture": fixture,
                "affinity": affinity, "pixel_sha256": baseline::hash(&frame.pixels),
                "manifest_sha256": baseline::hash(&std::fs::read(&manifest_path)?),
            }))?;
            command("warmup")?;
            check_deadline(started)?;
        }
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
        if paired {
            event(&serde_json::json!({
                "event": "warmup", "milliseconds": warmup_ms,
                "output_sha256": baseline::hash(&serde_json::to_vec(control.as_ref().unwrap())?),
                "recognized_lines": control.as_ref().unwrap().lines.len(),
            }))?;
        }
        let line_count = control.as_ref().unwrap().lines.len();
        let classifier_worker_budget = if line_workers == 0 {
            CAPACITY.min(16)
        } else {
            line_workers.min(CAPACITY).min(16)
        };
        let mut wall_ms = Vec::new();
        let mut cpu_ms = Vec::new();
        for sample_index in 0..3 {
            if paired {
                command("run")?;
            }
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
            if paired {
                event(&serde_json::json!({
                    "event": "sample", "sample_index": sample_index,
                    "milliseconds": elapsed_ms, "cpu_ms": elapsed_cpu_ms,
                    "output_sha256": baseline::hash(&serde_json::to_vec(&output)?),
                }))?;
            }
        }
        let (peak_private_bytes, peak_resident_bytes) = benchmark::platform::peak_memory()?;
        cases.push(serde_json::json!({
            "line_workers": line_workers,
            "recognition_batch_size": batch_size,
            "model_import_ms": import_ms,
            "cpu_context_ms": context_ms,
            "compile_request_ms": compile_request_ms,
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
    let report = serde_json::json!({
        "event": "complete",
        "paired": paired,
        "instrumented": false,
        "model": model,
        "fixture": fixture,
        "cpu_context_capacity": CAPACITY,
        "affinity": affinity,
        "pixel_sha256": baseline::hash(&frame.pixels),
        "manifest_sha256": baseline::hash(&std::fs::read(manifest_path)?),
        "order": owners,
        "batch_size_order": batch_sizes,
        "warmups_per_configuration": 1,
        "measured_runs_per_configuration": 3,
        "capture_ctc_spans": false,
        "complete_owned_outputs_exactly_equal": true,
        "control_output_sha256": baseline::hash(&serde_json::to_vec(control.as_ref().unwrap())?),
        "recognized_lines": control.as_ref().unwrap().lines.len(),
        "cases": cases,
        "execution_seconds": started.elapsed().as_secs_f64(),
        "limitation": "One process and one configuration order; immutable models and a 24-thread CPU pool shared, fresh mutable request per configuration. The automatic setting uses 16 classifiers while explicit twelve owners uses twelve; classifier figures are derived from current source policy for one nonconcurrent request. Process memory peaks are cumulative high-water counters, not isolated request memory. Internal deadline checks cannot interrupt an in-flight OCR call. No steady-state or full qualification claim."
    });
    if paired {
        event(&report)?;
    } else {
        println!("{}", serde_json::to_string_pretty(&report)?);
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("CPU owner probe: {error}");
        std::process::exit(1);
    }
}
