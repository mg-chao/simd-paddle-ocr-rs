//! Bounded recognition-only ownership comparison on fixed real oriented crops.
//! This diagnoses worker contention; adoption requires complete-page validation.
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
    lines::{LineOptions, OwnedOrientedLines},
    recognizer::{OwnedRecognition, Recognizer, RecognizerOptions},
};
use std::{path::PathBuf, sync::Arc, time::Instant};

struct Worker {
    recognizer: Recognizer,
    jobs: Vec<usize>,
    width_cost: usize,
    outputs: Vec<(usize, OwnedRecognition)>,
    error: Option<String>,
}

fn run_wave(
    cpu: &CpuContext,
    workers: &mut [Worker],
    crops: &OwnedOrientedLines,
) -> Result<(f64, f64, Vec<OwnedRecognition>)> {
    let cpu_start = benchmark::platform::process_cpu_time()?;
    let start = Instant::now();
    cpu.for_each_mut(workers, workers.len(), |_, worker| {
        worker.outputs.clear();
        worker.error = None;
        let view = crops.as_view();
        for &index in &worker.jobs {
            let result = worker
                .recognizer
                .recognize(view.get(index).unwrap().image, false)
                .and_then(|value| value.try_to_owned());
            match result {
                Ok(value) => worker.outputs.push((index, value)),
                Err(error) => {
                    worker.error = Some(error.to_string());
                    break;
                }
            }
        }
    });
    let wall_ms = start.elapsed().as_secs_f64() * 1000.;
    let cpu_ms = benchmark::platform::process_cpu_time()?
        .checked_sub(cpu_start)
        .ok_or("process CPU clock moved backwards")?
        .as_secs_f64()
        * 1000.;
    let mut outputs = Vec::new();
    for worker in workers {
        if let Some(error) = &worker.error {
            return Err(error.clone().into());
        }
        outputs.extend(worker.outputs.iter().cloned());
    }
    outputs.sort_unstable_by_key(|(index, _)| *index);
    Ok((
        wall_ms,
        cpu_ms,
        outputs.into_iter().map(|(_, r)| r).collect(),
    ))
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
            "cpu_recognition_probe [tiny|small|medium] [FIXTURE] [OWNERS=16,12]\nFixed real crops; 24 shared threads, one intra-op thread per owner, one warm-up and three measured waves per configuration, alternating order. At most three configurations. Fifteen-second internal deadline; enforce an external timeout. Not complete OCR latency."
        );
        return Ok(());
    }
    if args.len() > 3 {
        return Err("expected MODEL FIXTURE [OWNERS]; use --help".into());
    }
    let model = args.first().map_or("medium", String::as_str);
    let fixture = args.get(1).map_or("imgs_00018069", String::as_str);
    let counts = args
        .get(2)
        .map_or("16,12", String::as_str)
        .split(',')
        .map(str::parse::<usize>)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if counts.is_empty() || counts.len() > 3 || counts.iter().any(|&n| !(1..=24).contains(&n)) {
        return Err("use one to three owner counts within 1..=24".into());
    }
    let started = Instant::now();
    let check = || -> Result<()> {
        if started.elapsed().as_secs_f64() >= 15. {
            Err("recognition probe exceeded fifteen seconds".into())
        } else {
            Ok(())
        }
    };
    let affinity = benchmark::platform::affinity(24, None)?;
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
    let cpu = Arc::new(CpuContext::new(24)?);
    let mut pipeline = OcrModelSet::from_paths(
        directory.join(&spec.det),
        directory.join(&spec.rec),
        directory.join(&spec.dictionary),
        Some(&directory.join(&spec.cls)),
    )?
    .builder()
    .cpu_context(cpu.clone())
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
    .build()?;
    pipeline.extractor_mut().set_execution_context(
        cpu.clone(),
        CpuOptions {
            line_workers: 16,
            preprocess_workers: 16,
            detector_threads: 16,
        },
    )?;
    let image = ImageView::new(
        &frame.pixels,
        frame.width as usize,
        frame.height as usize,
        0,
        PixelFormat::Bgr24,
    )?;
    let crops = pipeline.extractor_mut().extract(image)?.try_to_owned()?;
    let template = pipeline.recognizer_mut();
    let widths = crops
        .metadata()
        .iter()
        .map(|m| {
            template
                .select_width(m.width, m.height)
                .map(|s| s.tensor_width)
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let mut order: Vec<_> = (0..widths.len()).collect();
    order.sort_unstable_by(|&a, &b| widths[b].cmp(&widths[a]).then_with(|| a.cmp(&b)));
    let mut configurations = Vec::new();
    for &count in &counts {
        let active = count.min(widths.len());
        if active == 0 {
            return Err("fixture contains no recognized lines".into());
        }
        let mut workers = Vec::new();
        for _ in 0..active {
            let mut recognizer = template.fork()?;
            recognizer.set_execution_context(cpu.clone(), 1)?;
            workers.push(Worker {
                recognizer,
                jobs: Vec::new(),
                width_cost: 0,
                outputs: Vec::new(),
                error: None,
            });
        }
        for &index in &order {
            let worker = workers.iter_mut().min_by_key(|w| w.width_cost).unwrap();
            worker.jobs.push(index);
            worker.width_cost += widths[index];
        }
        configurations.push(workers);
    }
    let mut control = None;
    let mut warmups = Vec::new();
    for workers in &mut configurations {
        check()?;
        let (ms, _, result) = run_wave(&cpu, workers, &crops)?;
        if let Some(expected) = &control {
            if &result != expected {
                return Err("owner configurations change recognition output".into());
            }
        } else {
            control = Some(result);
        }
        warmups.push(ms);
    }
    let mut samples = vec![Vec::new(); counts.len()];
    let mut cpu_samples = samples.clone();
    let mut run_order = Vec::new();
    for sample in 0..3 {
        let mut order: Vec<_> = (0..counts.len()).collect();
        if sample % 2 == 1 {
            order.reverse();
        }
        for &variant in &order {
            check()?;
            let (ms, cpu_ms, result) = run_wave(&cpu, &mut configurations[variant], &crops)?;
            if Some(&result) != control.as_ref() {
                return Err("measured output differs from warm-up control".into());
            }
            samples[variant].push(ms);
            cpu_samples[variant].push(cpu_ms);
        }
        run_order.push(order.into_iter().map(|i| counts[i]).collect::<Vec<_>>());
    }
    check()?;
    let cases: Vec<_> = configurations.iter().enumerate().map(|(i,workers)|serde_json::json!({
        "requested_owners":counts[i],"active_owners":workers.len(),"kernel_threads":1,
        "warmup_ms":warmups[i],"samples_ms":samples[i],"samples_cpu_ms":cpu_samples[i],
        "retained_activation_bytes":workers.iter().map(|w|w.recognizer.retained_activation_bytes()).sum::<usize>(),
        "retained_workspace_bytes":workers.iter().map(|w|w.recognizer.retained_workspace_bytes()).sum::<usize>(),
        "width_costs":workers.iter().map(|w|w.width_cost).collect::<Vec<_>>(),
        "line_counts":workers.iter().map(|w|w.jobs.len()).collect::<Vec<_>>(),
    })).collect();
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "model":model,"fixture":fixture,"affinity":affinity,"cpu_capacity":24,
            "source_pixel_sha256":baseline::hash(&frame.pixels),"crop_pixels_sha256":baseline::hash(crops.pixels()),
            "control_output_sha256":baseline::hash(&serde_json::to_vec(control.as_ref().unwrap())?),
            "complete_recognition_outputs_equal":true,"line_count":widths.len(),"widths":widths,
            "warmups_per_configuration":1,"measured_runs_per_configuration":3,"order":run_order,"cases":cases,
            "execution_seconds":started.elapsed().as_secs_f64(),
            "scope":"Standalone recognition on identical oriented real crops. Includes per-line owned conversion, excludes final sorting/copies and full-page pipeline orchestration. Two or more worker sets retained together. CPU pool shared, single request, no other inference. This is a selection diagnostic, not an adoption page benchmark."
        }))?
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("Recognition probe: {error}");
        std::process::exit(1);
    }
}
