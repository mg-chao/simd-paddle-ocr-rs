//! See docs/cpu-baseline.md. All dependencies beyond the library are dev-only.
#[path = "../tools/baseline/mod.rs"]
mod baseline;
#[path = "../tools/harness/mod.rs"]
mod harness;
use baseline::*;
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
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, Command, Stdio},
    sync::mpsc,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const CASE_TIMEOUT: Duration = Duration::from_secs(120);
const SUITE_TIMEOUT: Duration = Duration::from_secs(30 * 60);
fn main() {
    if let Err(e) = run() {
        eprintln!("baseline FAILED: {e}");
        std::process::exit(1);
    }
}
fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if args.first().is_some_and(|s| s == "_worker") {
        if args.len() != 4 {
            return Err("invalid worker arguments".into());
        }
        return rust_worker(Path::new(&args[1]), Path::new(&args[2]), &args[3]);
    }
    let Some(mode) = args.first() else {
        return Err(usage().into());
    };
    if mode == "--help" || mode == "-h" {
        println!("{}", usage());
        return Ok(());
    }
    if !["verify", "record", "pin-pixels"].contains(&mode.as_str()) {
        return Err(usage().into());
    }
    if cfg!(debug_assertions) {
        return Err("use the standard --release build".into());
    }
    if !cfg!(feature = "parallel") || cfg!(feature = "profiling") {
        return Err("baseline requires the default Cargo features".into());
    }
    harness::check_standard_environment()?;
    let mut reference = root.join("../SimdPaddleOCR");
    let mut models = None;
    let mut rust_only = false;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--rust-only" => rust_only = true,
            "--reference-root" | "--models-dir" => {
                let flag = &args[i];
                i += 1;
                let value = args.get(i).ok_or("missing option value")?;
                if flag == "--reference-root" {
                    reference = PathBuf::from(value);
                } else {
                    models = Some(PathBuf::from(value));
                }
            }
            _ => return Err(format!("unknown option {}", args[i]).into()),
        }
        i += 1;
    }
    if rust_only && mode != "verify" {
        return Err("--rust-only is valid only for verify".into());
    }
    let data = root.join("tests/baseline");
    let manifest_path = data.join("manifest.json");
    let mut manifest: Manifest = read_json(&manifest_path)?;
    manifest.validate()?;
    let output = root.join("target/baseline");
    fs::create_dir_all(&output)?;
    if mode == "pin-pixels" {
        let frames = decode_frames(&manifest, &data, false)?;
        for (fixture, frame) in manifest.fixtures.iter_mut().zip(&frames) {
            fixture.width = frame.width;
            fixture.height = frame.height;
            fixture.pixel_sha256 = hash(&frame.pixels);
        }
        write_json(&output.join("manifest.candidate.json"), &manifest)?;
        println!(
            "Pixel manifest candidate: {}",
            output.join("manifest.candidate.json").display()
        );
        return Ok(());
    }
    // Validate every input and every golden before starting any inference.
    let models = canonical_path(&models.unwrap_or_else(|| reference.join("models")))?;
    manifest.check_assets(&models)?;
    let frames = decode_frames(&manifest, &data, true)?;
    let manifest_hash = hash(&fs::read(&manifest_path)?);
    let goldens: Vec<Golden> = if mode == "verify" {
        manifest
            .models
            .iter()
            .map(|m| {
                let g: Golden = read_json(&data.join(format!("expected/{}.json", m.id)))?;
                g.validate(&m.id, &manifest, &manifest_hash)?;
                Ok(g)
            })
            .collect::<Result<_>>()?
    } else {
        Vec::new()
    };
    if !rust_only {
        check_reference(&reference, &manifest.reference_commit)?;
    }
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();
    let run_dir = output.join(format!("{stamp}-{mode}-{}", std::process::id()));
    fs::create_dir_all(&run_dir)?;
    let inputs = prepare_inputs(&frames, &run_dir)?;
    let mut metadata = provenance(&root);
    metadata.insert("mode".into(), mode.clone());
    metadata.insert("manifest_sha256".into(), manifest_hash.clone());
    metadata.insert(
        "case_timeout_seconds".into(),
        CASE_TIMEOUT.as_secs().to_string(),
    );
    metadata.insert(
        "suite_timeout_seconds".into(),
        SUITE_TIMEOUT.as_secs().to_string(),
    );
    metadata.insert("status".into(), "incomplete".into());
    if !rust_only {
        reference = canonical_path(&reference)?;
        metadata.insert(
            "dotnet_info".into(),
            command_text(
                Command::new("dotnet")
                    .arg("--info")
                    .current_dir(root.join("tools/csharp-reference")),
            )?,
        );
    }
    write_json(&run_dir.join("run.json"), &metadata)?;
    println!("Artifacts: {}", run_dir.display());
    let start = Instant::now();
    let outcome = execute(
        mode,
        rust_only,
        &root,
        &reference,
        &models,
        &manifest_path,
        &manifest,
        &manifest_hash,
        &goldens,
        &inputs,
        &run_dir,
        &metadata,
        start,
    );
    metadata.insert(
        "elapsed_seconds".into(),
        format!("{:.3}", start.elapsed().as_secs_f64()),
    );
    metadata.insert(
        "status".into(),
        if outcome.is_ok() { "passed" } else { "failed" }.into(),
    );
    if let Err(e) = &outcome {
        metadata.insert("error".into(), e.to_string());
    }
    write_json(&run_dir.join("run.json"), &metadata)?;
    outcome
}
fn usage() -> &'static str {
    "cargo run --release --locked --example cpu_baseline -- <verify|record|pin-pixels> [--reference-root PATH] [--models-dir PATH] [--rust-only]\nverify: Rust vs saved expectations, plus live C# unless --rust-only\nrecord: C# candidates under target/baseline (never overwrites checked-in expectations)\npin-pixels: candidate manifest after decoding hash-verified originals"
}
fn canonical_path(path: &Path) -> Result<PathBuf> {
    let path = fs::canonicalize(path)?;
    // MSBuild appends forward-slash project paths. Windows verbatim paths
    // returned by canonicalize disallow those separators; use normal absolute
    // paths at the .NET boundary, including ordinary UNC spelling.
    #[cfg(windows)]
    {
        let text = path.to_str().ok_or("path is not UTF-8")?;
        if let Some(rest) = text.strip_prefix(r"\\?\UNC\") {
            return Ok(PathBuf::from(format!(r"\\{rest}")));
        }
        if let Some(rest) = text.strip_prefix(r"\\?\") {
            return Ok(PathBuf::from(rest));
        }
    }
    Ok(path)
}
fn command_text(command: &mut Command) -> Result<String> {
    let output = command.output()?;
    if !output.status.success() {
        return Err(format!(
            "{command:?} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    Ok(String::from_utf8(output.stdout)?.trim().to_owned())
}
fn check_reference(reference: &Path, expected: &str) -> Result<()> {
    let head = command_text(
        Command::new("git")
            .arg("-C")
            .arg(reference)
            .args(["rev-parse", "HEAD"]),
    )?;
    if head != expected {
        return Err(format!("C# HEAD mismatch: expected {expected}, got {head}").into());
    }
    let dirty = command_text(Command::new("git").arg("-C").arg(reference).args([
        "status",
        "--porcelain",
        "--untracked-files=normal",
    ]))?;
    if !dirty.is_empty() {
        return Err(format!("C# reference has modified or untracked files:\n{dirty}").into());
    }
    Ok(())
}
fn provenance(root: &Path) -> BTreeMap<String, String> {
    let mut info = BTreeMap::new();
    info.insert("os".into(), std::env::consts::OS.into());
    info.insert("architecture".into(), std::env::consts::ARCH.into());
    info.insert(
        "cpu".into(),
        std::env::var("PROCESSOR_IDENTIFIER").unwrap_or_else(|_| "see host CPU inventory".into()),
    );
    info.insert("cpu_threads".into(), "1".into());
    info.insert(
        "build".into(),
        "standard release; default features; automatic SIMD; no custom RUSTFLAGS".into(),
    );
    for (name, program, arguments) in [
        ("rustc", "rustc", vec!["-Vv"]),
        ("cargo", "cargo", vec!["-V"]),
        ("rust_head", "git", vec!["rev-parse", "HEAD"]),
        ("rust_worktree", "git", vec!["status", "--porcelain"]),
    ] {
        info.insert(
            name.into(),
            command_text(Command::new(program).args(arguments).current_dir(root))
                .unwrap_or_else(|e| e.to_string()),
        );
    }
    info
}
fn prepare_inputs(frames: &[Frame], directory: &Path) -> Result<Vec<CaseInput>> {
    let directory = directory.join("inputs");
    fs::create_dir_all(&directory)?;
    frames
        .iter()
        .map(|f| {
            let path = directory.join(format!("{}.bgr", f.id));
            fs::write(&path, &f.pixels)?;
            Ok(CaseInput {
                id: f.id.clone(),
                path: path.to_str().ok_or("input path is not UTF-8")?.into(),
                width: f.width as usize,
                height: f.height as usize,
                stride: f.width as usize * 3,
            })
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn execute(
    mode: &str,
    rust_only: bool,
    root: &Path,
    reference: &Path,
    models: &Path,
    manifest_path: &Path,
    manifest: &Manifest,
    manifest_hash: &str,
    goldens: &[Golden],
    inputs: &[CaseInput],
    directory: &Path,
    provenance: &BTreeMap<String, String>,
    start: Instant,
) -> Result<()> {
    let adapter = root.join("tools/csharp-reference");
    let reference_dll = if rust_only {
        None
    } else {
        Some(harness::build_csharp(
            root,
            reference,
            &adapter.join("Reference.csproj"),
            directory,
            remaining(start)?,
        )?)
    };
    let mut failures = Vec::new();
    for (model_index, model) in manifest.models.iter().enumerate() {
        let mut rust = if mode == "verify" {
            let mut command = Command::new(std::env::current_exe()?);
            command
                .arg("_worker")
                .arg(manifest_path)
                .arg(models)
                .arg(&model.id);
            Some(Worker::spawn(
                &mut command,
                &directory.join(format!("{}-rust.log", model.id)),
            )?)
        } else {
            None
        };
        let mut csharp = if !rust_only {
            let mut command = Command::new("dotnet");
            command
                .current_dir(&adapter)
                .arg(reference_dll.as_ref().ok_or("C# adapter was not built")?)
                .arg(manifest_path)
                .arg(models)
                .arg(&model.id)
                .env("DOTNET_PROCESSOR_COUNT", "1");
            Some(Worker::spawn(
                &mut command,
                &directory.join(format!("{}-csharp.log", model.id)),
            )?)
        } else {
            None
        };
        let mut candidates = Vec::new();
        for (index, input) in inputs.iter().enumerate() {
            let expected = goldens.get(model_index).map(|g| &g.cases[index]);
            let receive = |worker: &mut Worker, language: &str| -> Result<CaseResult> {
                let result = worker
                    .case(input, start)
                    .map_err(|e| format!("{}/{}/{language}: {e}", model.id, input.id))?;
                // Persist the received evidence before validation or comparison.
                write_json(
                    &directory.join(format!("{}-{}-{language}.json", model.id, input.id)),
                    &result,
                )?;
                result.validate()?;
                if result.id != input.id {
                    return Err("worker returned the wrong fixture id".into());
                }
                Ok(result)
            };
            let rust_result = rust.as_mut().map(|w| receive(w, "rust")).transpose()?;
            let csharp_result = csharp.as_mut().map(|w| receive(w, "csharp")).transpose()?;
            for (label, e, a) in [
                ("golden -> Rust", expected, rust_result.as_ref()),
                ("golden -> C#", expected, csharp_result.as_ref()),
                ("C# -> Rust", csharp_result.as_ref(), rust_result.as_ref()),
            ] {
                if let (Some(e), Some(a)) = (e, a) {
                    failures.extend(
                        compare(&model.id, e, a)
                            .into_iter()
                            .map(|s| format!("{label}: {s}")),
                    );
                }
            }
            if let Some(result) = csharp_result {
                candidates.push(result);
            }
            write_json(&directory.join("differences.json"), &failures)?;
            println!(
                "{}/{} completed ({}/{})",
                model.id,
                input.id,
                index + 1,
                inputs.len()
            );
        }
        if let Some(worker) = &mut rust {
            worker.finish(start)?;
        }
        if let Some(worker) = &mut csharp {
            worker.finish(start)?;
        }
        if mode == "record" {
            let mut provenance = provenance.clone();
            provenance.remove("status");
            provenance.insert(
                "csharp_runtime".into(),
                fs::read_to_string(directory.join(format!("{}-csharp.log", model.id)))?,
            );
            let candidate = Golden {
                schema_version: SCHEMA,
                model: model.id.clone(),
                reference_commit: manifest.reference_commit.clone(),
                manifest_sha256: manifest_hash.into(),
                settings: manifest.settings.clone(),
                provenance,
                cases: candidates,
            };
            candidate.validate(&model.id, manifest, manifest_hash)?;
            write_json(&directory.join(format!("{}.json", model.id)), &candidate)?;
        }
    }
    write_json(&directory.join("differences.json"), &failures)?;
    if !failures.is_empty() {
        for difference in failures.iter().take(30) {
            eprintln!("{difference}");
        }
        return Err(format!("{} differences; see differences.json", failures.len()).into());
    }
    println!(
        "{}: all {} model/case combinations completed",
        mode,
        manifest.models.len() * inputs.len()
    );
    Ok(())
}
struct Worker {
    child: Child,
    input: Option<ChildStdin>,
    lines: mpsc::Receiver<std::io::Result<String>>,
}
impl Worker {
    fn spawn(command: &mut Command, log: &Path) -> Result<Self> {
        let mut transcript = fs::File::create(log.with_extension("jsonl"))?;
        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(fs::File::create(log)?)
            .spawn()?;
        let input = child.stdin.take();
        let stdout = child.stdout.take().unwrap();
        let (sender, lines) = mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                if let Ok(text) = &line {
                    let _ = writeln!(transcript, "{text}");
                }
                if sender.send(line).is_err() {
                    break;
                }
            }
        });
        Ok(Self {
            child,
            input,
            lines,
        })
    }
    fn case(&mut self, input: &CaseInput, start: Instant) -> Result<CaseResult> {
        let timeout = remaining(start)?.min(CASE_TIMEOUT);
        let stdin = self.input.as_mut().ok_or("closed worker input")?;
        serde_json::to_writer(&mut *stdin, input)?;
        stdin.write_all(b"\n")?;
        stdin.flush()?;
        let line = self
            .lines
            .recv_timeout(timeout)
            .map_err(|e| format!("worker stopped or exceeded case/suite deadline: {e}"))??;
        Ok(serde_json::from_str(&line)?)
    }
    fn finish(&mut self, start: Instant) -> Result<()> {
        self.input.take();
        let deadline = Instant::now() + remaining(start)?.min(CASE_TIMEOUT);
        loop {
            if let Some(status) = self.child.try_wait()? {
                return if status.success() {
                    Ok(())
                } else {
                    Err(format!("worker exited with {status}").into())
                };
            }
            if Instant::now() >= deadline {
                return Err("worker shutdown deadline exceeded".into());
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}
impl Drop for Worker {
    fn drop(&mut self) {
        self.input.take();
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
fn remaining(start: Instant) -> Result<Duration> {
    SUITE_TIMEOUT
        .checked_sub(start.elapsed())
        .filter(|d| !d.is_zero())
        .ok_or_else(|| "30-minute suite deadline exceeded".into())
}
fn rust_worker(manifest_path: &Path, models: &Path, model_id: &str) -> Result<()> {
    let manifest: Manifest = read_json(manifest_path)?;
    let model = manifest
        .models
        .iter()
        .find(|m| m.id == model_id)
        .ok_or("unknown model")?;
    let s = &manifest.settings;
    let models = OcrModelSet::from_paths(
        models.join(&model.det),
        models.join(&model.rec),
        models.join(&model.dictionary),
        Some(&models.join(&model.cls)),
    )?;
    let mut ocr = models
        .builder()
        .cpu_context(CpuContext::serial())
        .cpu_options(CpuOptions {
            line_workers: 1,
            preprocess_workers: 1,
            detector_threads: 1,
        })
        .detector_options(DetectorOptions {
            limit_side_length: s.detector_limit,
            max_candidates: s.max_candidates,
            bitmap_threshold: s.bitmap_threshold,
            box_threshold: Some(model.box_threshold),
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
        .recognition_batch_size(s.recognition_batch_size)
        .build()?;
    eprintln!(
        "CPU budgets: {:?}; compile: {:?}; backend: {:?}",
        ocr.cpu_budgets(),
        ocr.recognizer_mut().compiled().options(),
        ocr.recognizer_mut().compiled().backend()
    );
    for line in std::io::stdin().lock().lines() {
        let input: CaseInput = serde_json::from_str(&line?)?;
        let pixels = fs::read(&input.path)?;
        if input.stride != input.width.checked_mul(3).ok_or("stride overflow")?
            || pixels.len()
                != input
                    .stride
                    .checked_mul(input.height)
                    .ok_or("size overflow")?
        {
            return Err("BGR24 dimensions mismatch".into());
        }
        let image = ImageView::new(
            &pixels,
            input.width,
            input.height,
            input.stride,
            PixelFormat::Bgr24,
        )?;
        let view = ocr.run(image, false)?;
        let result = CaseResult {
            id: input.id,
            text: view.text.into(),
            detected_count: view.detected_count,
            detector_resized_width: view.detector_resized_width,
            detector_resized_height: view.detector_resized_height,
            lines: view
                .iter()
                .map(|l| Line {
                    points: l.detection.points.map(|p| p.map(f64::from)),
                    detection_score: f64::from(l.detection.score),
                    text: l.text.into(),
                    recognition_score: f64::from(l.recognition_score),
                    classification_score: f64::from(l.classification_score),
                    classification_label: l.classification_label,
                    applied_rotation_degrees: l.applied_rotation_degrees,
                    emitted_count: l.emitted_count,
                    recognition_content_width: l.recognition_content_width,
                    recognition_tensor_width: l.recognition_tensor_width,
                    recognition_time_steps: l.recognition_time_steps,
                })
                .collect(),
        };
        result.validate()?;
        serde_json::to_writer(std::io::stdout().lock(), &result)?;
        println!();
        std::io::stdout().flush()?;
    }
    Ok(())
}
