//! Matched, bounded physical-machine comparisons; see docs/cpu-performance.md.
#[path = "../tools/baseline/mod.rs"]
mod baseline;
#[path = "../tools/benchmark/mod.rs"]
mod benchmark;
#[path = "../tools/harness/mod.rs"]
mod harness;
use baseline::{CaseResult, Line, Manifest, hash, read_json, write_json};
use benchmark::{Comparison, Job, Measurement, OrderStrategy, ProcessCpuTiming, Report, Result};
use simd_paddle_ocr_rs::{
    OcrModelSet,
    cpu::{CpuContext, CpuOptions},
    detector::DetectorOptions,
    image::{ImageView, PixelFormat},
    lines::LineOptions,
    recognizer::RecognizerOptions,
};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::Arc,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
fn main() {
    if let Err(error) = run() {
        eprintln!("CPU benchmark: {error}");
        std::process::exit(1);
    }
}
fn usage() -> &'static str {
    "cargo run --release --locked --example cpu_benchmark -- [--model tiny|small|medium|all] [--threads 1|8|all] [--fixture ID[,ID]] [--workload latency|throughput|all] [--order balanced|rust-first|csharp-first] [--runs 1..3] [--budget-seconds N] [--case-timeout-seconds N] [--affinity HEX] [--models-dir PATH] [--reference-root PATH]\nDefaults: all models, 1 and 8 threads, three representative fixtures, both workloads, balanced canonical model/thread/fixture order, 3 samples, one warm-up, 60 seconds total per implementation (build excluded). Balanced order is independent of workload and filtering; a single case has one recorded order, not repeated processes. Full qualification requires an explicit larger bounded budget. Windows x64, standard release/default features, no profiling."
}
fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.first().is_some_and(|a| a == "--help" || a == "-h") {
        println!("{}", usage());
        return Ok(());
    }
    if args.first().is_some_and(|a| a == "_worker") {
        let path = args.get(1).ok_or("missing worker job")?;
        return rust_worker(&read_json(Path::new(path))?);
    }
    if !cfg!(all(windows, target_arch = "x86_64")) {
        return Err("benchmark qualification requires physical Windows x64".into());
    }
    if cfg!(debug_assertions) || !cfg!(feature = "parallel") || cfg!(feature = "profiling") {
        return Err("use --release, default features, and no profiling".into());
    }
    harness::check_standard_environment()?;
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut reference = root.join("../SimdPaddleOCR");
    let mut models = None;
    let (mut selected_model, mut selected_threads, mut workload) =
        ("all".to_owned(), "all".to_owned(), "all".to_owned());
    let mut fixtures = benchmark::FIXTURES
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    let (mut runs, mut budget, mut timeout) = (3, 60u64, 120u64);
    let mut explicit_affinity = None;
    let mut order_strategy = OrderStrategy::Balanced;
    let mut i = 0;
    while i < args.len() {
        let flag = &args[i];
        i += 1;
        let value = args.get(i).ok_or("missing option value")?;
        match flag.as_str() {
            "--model" => selected_model = value.clone(),
            "--threads" => selected_threads = value.clone(),
            "--fixture" => fixtures = value.split(',').map(str::to_owned).collect(),
            "--workload" => workload = value.clone(),
            "--order" => order_strategy = OrderStrategy::parse(value)?,
            "--runs" => runs = value.parse()?,
            "--budget-seconds" => budget = value.parse()?,
            "--case-timeout-seconds" => timeout = value.parse()?,
            "--affinity" => {
                explicit_affinity = Some(u64::from_str_radix(value.trim_start_matches("0x"), 16)?)
            }
            "--models-dir" => models = Some(PathBuf::from(value)),
            "--reference-root" => reference = PathBuf::from(value),
            _ => return Err(format!("unknown option {flag}").into()),
        }
        i += 1;
    }
    if !(1..=3).contains(&runs)
        || budget == 0
        || timeout == 0
        || !fixtures
            .iter()
            .all(|s| benchmark::FIXTURES.contains(&s.as_str()))
        || fixtures.is_empty()
        || fixtures
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != fixtures.len()
    {
        return Err("invalid runs, budget, timeout or representative fixture selection".into());
    }
    let thread_counts = match selected_threads.as_str() {
        "1" => vec![1],
        "8" => vec![8],
        "all" => vec![1, 8],
        _ => return Err("--threads must be 1, 8 or all".into()),
    };
    let concurrencies = match workload.as_str() {
        "latency" => vec![1],
        "throughput" => vec![2],
        "all" => vec![1, 2],
        _ => return Err("--workload must be latency, throughput or all".into()),
    };
    if !["tiny", "small", "medium", "all"].contains(&selected_model.as_str()) {
        return Err("unknown model".into());
    }
    let manifest_path = root.join("tests/baseline/manifest.json");
    let manifest: Manifest = read_json(&manifest_path)?;
    manifest.validate()?;
    reference = harness::canonical_path(&reference)?;
    harness::check_reference(&reference, &manifest.reference_commit)?;
    let models = harness::canonical_path(&models.unwrap_or_else(|| reference.join("models")))?;
    manifest.check_assets(&models)?;
    let frames = baseline::decode_frames(&manifest, &root.join("tests/baseline"), true)?;
    let directory = root.join(format!(
        "target/cpu-benchmark/{}-{}",
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis(),
        std::process::id()
    ));
    fs::create_dir_all(&directory)?;
    println!("Artifacts: {}", directory.display());
    let mut provenance = BTreeMap::new();
    let source_hash = source_tree_hash(&root)?;
    provenance.insert("rust_source_sha256".into(), source_hash.clone());
    provenance.insert(
        "rust_executable_sha256".into(),
        hash(&fs::read(std::env::current_exe()?)?),
    );
    provenance.insert("manifest_sha256".into(), hash(&fs::read(&manifest_path)?));
    provenance.insert("csharp_head".into(), manifest.reference_commit.clone());
    provenance.insert(
        "cpu".into(),
        std::env::var("PROCESSOR_IDENTIFIER").unwrap_or_default(),
    );
    provenance.insert("measurement".into(),"standard release/default features; CPU automatic SIMD/layout; fusion; owned results; predecoded BGR24; no profiling; one first inference + one warmup + fixed samples; peak process private commit before diagnostics".into());
    provenance.insert("case_timeout_seconds".into(), timeout.to_string());
    provenance.insert(
        "execution_order_strategy".into(),
        order_strategy.name().into(),
    );
    provenance.insert(
        "balanced_order_rule".into(),
        "v1: parity(model_index + thread_index + fixture_index); even Rust-first, odd C#-first; canonical indices; concurrency and selection order excluded".into(),
    );
    provenance.insert(
        "process_cpu_measurement".into(),
        "GetProcessTimes kernel+user across all process threads for each first/warm/measured batch; diagnostic only; one warm-up does not establish steady state".into(),
    );
    for (name, program, arguments) in [
        ("rustc", "rustc", vec!["-Vv"]),
        ("rust_head", "git", vec!["rev-parse", "HEAD"]),
        ("rust_worktree", "git", vec!["status", "--porcelain"]),
    ] {
        provenance.insert(
            name.into(),
            harness::command_text(Command::new(program).args(arguments).current_dir(&root))?,
        );
    }
    provenance.insert(
        "dotnet".into(),
        harness::command_text(
            Command::new("dotnet")
                .arg("--info")
                .current_dir(root.join("tools/csharp-benchmark")),
        )?,
    );
    #[cfg(windows)]
    let available = benchmark::windows::allowed_affinity()?;
    #[cfg(not(windows))]
    let available = 0;
    let mut cases = vec![];
    for model in manifest
        .models
        .iter()
        .filter(|m| selected_model == "all" || m.id == selected_model)
    {
        for &threads in &thread_counts {
            let affinity =
                explicit_affinity.unwrap_or(benchmark::select_affinity(available, threads)?);
            if affinity & !available != 0 {
                return Err("explicit affinity is outside the parent process affinity".into());
            }
            for fixture in &fixtures {
                let frame = frames
                    .iter()
                    .find(|f| &f.id == fixture)
                    .ok_or("missing fixture")?;
                let pixels = directory.join(format!("{fixture}.bgr"));
                fs::write(&pixels, &frame.pixels)?;
                for &concurrency in &concurrencies {
                    let job = Job {
                        manifest: manifest_path.to_string_lossy().into(),
                        models: models.to_string_lossy().into(),
                        model: model.id.clone(),
                        fixture: fixture.clone(),
                        pixels: pixels.to_string_lossy().into(),
                        pixel_sha256: hash(&frame.pixels),
                        width: frame.width as usize,
                        height: frame.height as usize,
                        threads,
                        concurrency,
                        runs,
                        affinity: format!("{affinity:x}"),
                    };
                    job.validate()?;
                    let languages = order_strategy.languages(&job)?;
                    let mut comparison = Comparison::new(job);
                    comparison.planned_execution_order =
                        languages.into_iter().map(str::to_owned).collect();
                    cases.push(comparison);
                }
            }
        }
    }
    let mut report = Report {
        schema_version: 1,
        status: "inconclusive".into(),
        provenance,
        budget_seconds_per_language: budget,
        elapsed_seconds_per_language: BTreeMap::new(),
        cases,
    };
    benchmark::write_report(&directory, &report)?;
    let dll = harness::build_csharp(
        &root,
        &reference,
        &root.join("tools/csharp-benchmark/Benchmark.csproj"),
        &directory,
        Duration::from_secs(180),
    )?;
    report
        .provenance
        .insert("csharp_adapter_sha256".into(), hash(&fs::read(&dll)?));
    let mut totals = BTreeMap::from([
        ("rust".to_string(), Duration::ZERO),
        ("csharp".to_string(), Duration::ZERO),
    ]);
    for index in 0..report.cases.len() {
        let case = &mut report.cases[index];
        let stem = format!(
            "{}-{}-t{}-c{}",
            case.job.model, case.job.fixture, case.job.threads, case.job.concurrency
        );
        let job_path = directory.join(format!("{stem}.job.json"));
        write_json(&job_path, &case.job)?;
        let languages = order_strategy.languages(&case.job)?;
        for language in languages {
            let remaining = Duration::from_secs(budget)
                .checked_sub(totals[language])
                .filter(|v| !v.is_zero());
            let Some(remaining) = remaining else {
                case.error = Some(format!(
                    "{language} total budget exhausted; remaining jobs were not run"
                ));
                continue;
            };
            let mut command = if language == "rust" {
                let mut c = Command::new(std::env::current_exe()?);
                c.arg("_worker").arg(&job_path);
                c
            } else {
                let mut c = Command::new("dotnet");
                c.arg(&dll).arg(&job_path);
                c
            };
            command.env("DOTNET_PROCESSOR_COUNT", case.job.threads.to_string());
            let output = directory.join(format!("{stem}-{language}.json"));
            let log = directory.join(format!("{stem}-{language}.log"));
            let start = Instant::now();
            let result = run_worker(
                &mut command,
                &output,
                &log,
                remaining.min(Duration::from_secs(timeout)),
                || case.execution_order.push(language.to_owned()),
            );
            *totals.get_mut(language).unwrap() += start.elapsed();
            match result {
                Ok(measurement) => {
                    if language == "rust" {
                        case.rust = Some(measurement)
                    } else {
                        case.csharp = Some(measurement)
                    }
                }
                Err(error) => {
                    case.error = Some(format!("{language}: {error}"));
                    eprintln!("{stem}/{language}: {error}");
                }
            }
        }
        if let Err(error) = case.evaluate() {
            case.error = Some(error.to_string());
            case.status = "inconclusive".into();
        }
        println!("{stem}: {}", case.status);
        report.elapsed_seconds_per_language = totals
            .iter()
            .map(|(k, v)| (k.clone(), v.as_secs_f64()))
            .collect();
        benchmark::write_report(&directory, &report)?;
    }
    report.status = if report.cases.iter().any(|c| c.status == "failed") {
        "failed"
    } else if report.cases.iter().all(|c| c.status == "passed") {
        "passed"
    } else {
        "inconclusive"
    }
    .into();
    if source_tree_hash(&root)? != source_hash {
        report.status = "inconclusive".into();
        report.provenance.insert(
            "version_error".into(),
            "Rust source changed during measurement; results do not qualify one code version"
                .into(),
        );
        for case in &mut report.cases {
            case.status = "inconclusive".into();
            case.error = Some("source changed during measurement".into());
        }
    }
    benchmark::write_report(&directory, &report)?;
    if report.status != "passed" {
        return Err(format!("{}; see {}/report.md", report.status, directory.display()).into());
    }
    Ok(())
}
fn source_tree_hash(root: &Path) -> Result<String> {
    harness::source_tree_hash(
        root,
        &[
            "src",
            "tools/baseline",
            "tools/harness",
            "tools/benchmark",
            "tools/csharp-benchmark",
            "examples/cpu_benchmark.rs",
            "Cargo.toml",
            "Cargo.lock",
            "rust-toolchain.toml",
            ".cargo",
            "build.rs",
        ],
    )
}
fn run_worker(
    command: &mut Command,
    output: &Path,
    log: &Path,
    timeout: Duration,
    on_started: impl FnOnce(),
) -> Result<Measurement> {
    struct Running(std::process::Child);
    impl Drop for Running {
        fn drop(&mut self) {
            if !matches!(self.0.try_wait(), Ok(Some(_))) {
                harness::kill_tree(&mut self.0);
            }
        }
    }
    let mut child = Running(
        command
            .stdout(fs::File::create(output)?)
            .stderr(fs::File::create(log)?)
            .stdin(Stdio::null())
            .spawn()?,
    );
    on_started();
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(status) = child.0.try_wait()? {
            if !status.success() {
                return Err(format!("worker exited {status}; see {}", log.display()).into());
            }
            return read_json(output);
        }
        if Instant::now() >= deadline {
            harness::kill_tree(&mut child.0);
            return Err(format!(
                "worker exceeded {:.3}s deadline; terminated; see {}",
                timeout.as_secs_f64(),
                log.display()
            )
            .into());
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}
fn rust_worker(job: &Job) -> Result<()> {
    job.validate()?;
    #[cfg(windows)]
    benchmark::windows::set_affinity(u64::from_str_radix(&job.affinity, 16)?)?;
    #[cfg(not(windows))]
    return Err("Windows worker required".into());
    let pixels = fs::read(&job.pixels)?;
    if hash(&pixels) != job.pixel_sha256 || pixels.len() != job.width * job.height * 3 {
        return Err("predecoded BGR24 integrity failure".into());
    }
    let manifest: Manifest = read_json(Path::new(&job.manifest))?;
    let spec = manifest
        .models
        .iter()
        .find(|m| m.id == job.model)
        .ok_or("model absent")?;
    let s = &manifest.settings;
    let models = Path::new(&job.models);
    let start = Instant::now();
    let models = OcrModelSet::from_paths(
        models.join(&spec.det),
        models.join(&spec.rec),
        models.join(&spec.dictionary),
        Some(models.join(&spec.cls)),
    )?;
    let counters = models.clone();
    let capacity = job.threads;
    let first = models
        .builder()
        .cpu_context(Arc::new(CpuContext::new(capacity)?))
        .cpu_options(CpuOptions {
            line_workers: if capacity == 1 { 1 } else { 4 },
            preprocess_workers: capacity,
            detector_threads: capacity,
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
        .recognition_batch_size(1)
        .build()?;
    let mut requests = Vec::new();
    for _ in 1..job.concurrency {
        requests.push(first.fork()?);
    }
    requests.insert(0, first);
    let load_compile_ms = start.elapsed().as_secs_f64() * 1000.;
    eprintln!(
        "CPU={:?}; affinity={}; uninstrumented owned results",
        requests[0].cpu_budgets(),
        job.affinity
    );
    let image = ImageView::new(
        &pixels,
        job.width,
        job.height,
        job.width * 3,
        PixelFormat::Bgr24,
    )?;
    let batch = |requests: &mut Vec<simd_paddle_ocr_rs::ocr::Ocr>| -> Result<_> {
        #[cfg(windows)]
        let cpu_start = benchmark::windows::process_cpu_time()?;
        let start = Instant::now();
        let results = if requests.len() == 1 {
            vec![requests[0].run(image, false)?.try_to_owned()?]
        } else {
            let barrier = std::sync::Barrier::new(requests.len());
            std::thread::scope(|scope| {
                let handles = requests
                    .iter_mut()
                    .map(|request| {
                        let barrier = &barrier;
                        scope.spawn(move || {
                            barrier.wait();
                            request.run(image, false)?.try_to_owned()
                        })
                    })
                    .collect::<Vec<_>>();
                handles
                    .into_iter()
                    .map(|h| {
                        h.join()
                            .map_err(|_| "request worker panicked")?
                            .map_err(Into::into)
                    })
                    .collect::<Result<Vec<_>>>()
            })?
        };
        let elapsed = start.elapsed().as_secs_f64() * 1000.;
        #[cfg(windows)]
        let cpu_elapsed = benchmark::windows::process_cpu_time()?
            .checked_sub(cpu_start)
            .ok_or("process CPU time moved backwards")?
            .as_secs_f64()
            * 1000.;
        #[cfg(not(windows))]
        let cpu_elapsed = 0.;
        for result in &results {
            std::hint::black_box(result);
        }
        Ok((elapsed, cpu_elapsed, results))
    };
    let (first_inference_ms, first_cpu_ms, initial) = batch(&mut requests)?;
    if initial.iter().any(|result| result != &initial[0]) {
        return Err("concurrent cold Rust results differ".into());
    }
    let control = initial.into_iter().next().unwrap();
    let (warmup_ms, warmup_cpu_ms, warm) = batch(&mut requests)?;
    if warm.iter().any(|result| result != &control) {
        return Err("Rust warm-up result differs from cold control".into());
    }
    drop(warm);
    let mut samples_ms = vec![];
    let mut samples_cpu_ms = vec![];
    let mut final_results = vec![];
    for _ in 0..job.runs {
        drop(std::mem::take(&mut final_results));
        let (elapsed, cpu_elapsed, results) = batch(&mut requests)?;
        if results.iter().any(|result| result != &control) {
            return Err("Rust measured result differs from cold control".into());
        }
        samples_ms.push(elapsed);
        samples_cpu_ms.push(cpu_elapsed);
        final_results = results;
    }
    #[cfg(windows)]
    let peak_private_bytes = benchmark::windows::peak_private()?;
    #[cfg(not(windows))]
    let peak_private_bytes = 0;
    let mut diagnostics = BTreeMap::new();
    let models = std::iter::once(counters.detection.as_ref())
        .chain(std::iter::once(counters.recognition.as_ref()))
        .chain(counters.classification.as_deref());
    let mut weights = 0;
    let mut packs = 0;
    for m in models {
        weights += m.normalized_weight_bytes();
        packs += m.cached_packed_weight_bytes() as u64;
    }
    diagnostics.insert("normalized_weight_bytes".into(), weights);
    diagnostics.insert("shared_cached_weight_bytes".into(), packs);
    diagnostics.insert("retained_requests".into(), requests.len() as u64);
    diagnostics.insert(
        "retained_activation_bytes".into(),
        requests
            .iter()
            .map(|r| r.retained_activation_bytes() as u64)
            .sum(),
    );
    diagnostics.insert(
        "retained_workspace_bytes".into(),
        requests
            .iter()
            .map(|r| r.retained_workspace_bytes() as u64)
            .sum(),
    );
    diagnostics.insert(
        "retained_kernel_scratch_bytes_subset_of_workspace".into(),
        requests
            .iter()
            .map(|r| r.retained_kernel_scratch_bytes() as u64)
            .sum(),
    );
    let result = &final_results[0];
    let result = CaseResult {
        id: job.fixture.clone(),
        text: result.text.clone(),
        detected_count: result.detected_count,
        detector_resized_width: result.detector_resized_width,
        detector_resized_height: result.detector_resized_height,
        lines: result
            .as_view()
            .iter()
            .map(|l| Line {
                points: l.detection.points.map(|p| p.map(f64::from)),
                detection_score: l.detection.score.into(),
                text: l.text.into(),
                recognition_score: l.recognition_score.into(),
                classification_score: l.classification_score.into(),
                classification_label: l.classification_label,
                applied_rotation_degrees: l.applied_rotation_degrees,
                emitted_count: l.emitted_count,
                recognition_content_width: l.recognition_content_width,
                recognition_tensor_width: l.recognition_tensor_width,
                recognition_time_steps: l.recognition_time_steps,
            })
            .collect(),
    };
    let measurement = Measurement {
        language: "rust".into(),
        load_compile_ms,
        first_inference_ms,
        warmup_ms,
        samples_ms,
        process_cpu: Some(ProcessCpuTiming {
            first_inference_ms: first_cpu_ms,
            warmup_ms: warmup_cpu_ms,
            samples_ms: samples_cpu_ms,
        }),
        peak_private_bytes,
        diagnostics,
        result,
    };
    measurement.validate(job)?;
    serde_json::to_writer(std::io::stdout(), &measurement)?;
    Ok(())
}
