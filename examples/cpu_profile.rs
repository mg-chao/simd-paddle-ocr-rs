//! Instrumented production-stage diagnosis; never an official latency benchmark.
#[path = "../tools/baseline/mod.rs"]
mod baseline;
#[path = "../tools/benchmark/mod.rs"]
mod benchmark;
use baseline::Result;
#[cfg(feature = "profiling")]
use baseline::{Manifest, read_json};
#[cfg(feature = "profiling")]
fn run() -> Result<()> {
    use simd_paddle_ocr_rs::{
        OcrModelSet,
        cpu::{CpuContext, CpuOptions},
        detector::DetectorOptions,
        image::{ImageView, PixelFormat},
        lines::LineOptions,
        recognizer::RecognizerOptions,
    };
    use std::{path::PathBuf, sync::Arc};
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.first().is_some_and(|s| s == "--help" || s == "-h") {
        println!(
            "cargo run --release --locked --features profiling --example cpu_profile -- [tiny|small|medium] [FIXTURE] [1|8] [LINE_WORKERS=0]\nOne warm-up and one instrumented production run; JSON stage wall/work counters and retained capacities. No official performance claim."
        );
        return Ok(());
    }
    if args.len() > 4 {
        return Err("expected MODEL FIXTURE THREADS [LINE_WORKERS]; use --help".into());
    }
    let model = args.first().map_or("tiny", String::as_str);
    let fixture = args.get(1).map_or("imgs_1", String::as_str);
    let threads: usize = args.get(2).map_or("8", String::as_str).parse()?;
    let line_workers: usize = args.get(3).map_or("0", String::as_str).parse()?;
    if line_workers > 16 {
        return Err("line workers must be within 0..=16".into());
    }
    if ![1, 8].contains(&threads) || !["tiny", "small", "medium"].contains(&model) {
        return Err("use a shipped model and 1 or 8 threads".into());
    }
    #[cfg(windows)]
    {
        let affinity =
            benchmark::select_affinity(benchmark::windows::allowed_affinity()?, threads)?;
        benchmark::windows::set_affinity(affinity)?;
    }
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
    let s = &manifest.settings;
    let models = OcrModelSet::from_paths(
        directory.join(&spec.det),
        directory.join(&spec.rec),
        directory.join(&spec.dictionary),
        Some(&directory.join(&spec.cls)),
    )?;
    let mut ocr = models
        .builder()
        .cpu_context(Arc::new(CpuContext::new(threads)?))
        .cpu_options(CpuOptions {
            line_workers,
            preprocess_workers: threads,
            detector_threads: threads,
        })
        .detector_options(DetectorOptions {
            limit_side_length: s.detector_limit,
            max_candidates: s.max_candidates,
            bitmap_threshold: s.bitmap_threshold,
            box_threshold: Some(spec.box_threshold),
            unclip_ratio: s.unclip_ratio,
            use_dilation: s.use_dilation,
            ..Default::default()
        })
        .recognizer_options(RecognizerOptions {
            target_width: s.target_width,
            adaptive_width: s.adaptive_width,
            ..Default::default()
        })
        .line_options(LineOptions {
            classifier_threshold: s.classifier_threshold,
            ..Default::default()
        })
        .direction_classification(s.direction_classification)
        .build()?;
    let image = ImageView::new(
        &frame.pixels,
        frame.width as usize,
        frame.height as usize,
        0,
        PixelFormat::Bgr24,
    )?;
    let expected = ocr.run(image, false)?.try_to_owned()?;
    let (view, profile) = ocr.run_profiled(image, false)?;
    let actual = view.try_to_owned()?;
    if actual != expected {
        return Err("profiled output differs from warm-up".into());
    }
    // recognizer_mut() invalidates cached worker requests, so snapshot aggregate
    // capacities before using mutable component accessors for the breakdown.
    let total = ocr.retained_activation_bytes();
    let workspace = ocr.retained_workspace_bytes();
    let det = ocr.extractor_mut().detector().retained_activation_bytes();
    let cls = ocr.extractor_mut().classifier_retained_activation_bytes();
    let rec = ocr.recognizer_mut().retained_activation_bytes();
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "instrumented":true,"model":model,"fixture":fixture,"threads":threads,"requested_line_workers":line_workers,
            "budgets":format!("{:?}",ocr.cpu_budgets()),"recognized_lines":actual.lines.len(),"profile":profile,
            "retained_activation_bytes":{"total":total,"detector":det,"classifiers":cls,
                "root_recognizer":rec,"recognition_workers":total-det-cls-rec},
        "retained_workspace_bytes":workspace
        }))?
    );
    Ok(())
}
#[cfg(not(feature = "profiling"))]
fn run() -> Result<()> {
    Err("enable --features profiling; this is diagnostic instrumentation".into())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("CPU profile: {e}");
        std::process::exit(1);
    }
}
