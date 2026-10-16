//! Bounded, matched real-model configuration suite. See docs/cpu-qualification.md.
#[path = "../tools/baseline/mod.rs"]
mod baseline;
#[path = "../tools/harness/mod.rs"]
mod harness;
#[path = "../tools/qualification/mod.rs"]
mod qualification;
use baseline::{CaseResult, Frame, Golden, Line, Manifest, Result, hash, read_json, write_json};
use qualification::{Alignment, CharacterBox, Input, Profile, Response, Span};
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
    sync::{Arc, mpsc},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
const CASE_TIMEOUT: Duration = Duration::from_secs(120);
const SUITE_TIMEOUT: Duration = Duration::from_secs(60 * 60);
const SOURCE_PATHS: &[&str] = &[
    "src",
    "tools/baseline",
    "tools/harness",
    "tools/qualification",
    "tools/csharp-qualification",
    "examples/cpu_qualify.rs",
    "Cargo.toml",
    "Cargo.lock",
    "rust-toolchain.toml",
    ".cargo",
    "build.rs",
];
fn main() {
    if let Err(error) = run() {
        eprintln!("qualification FAILED: {error}");
        std::process::exit(1);
    }
}
fn usage() -> &'static str {
    "cargo run --release --locked --example cpu_qualify -- [--model tiny|small|medium] [--profile NAME] [--fixture ID] [--reference-root PATH] [--models-dir PATH]\nDefault: all three bundles; default-parallel on all 22 fixtures plus bounded focused configuration/reuse/recovery profiles.\nRepeat --model, --profile, or --fixture to select more than one. Fixture filtering is unavailable for reuse/recovery.\nProfiles: default-parallel, serial, workers-2, workers-4, batch-2, batch-4, alignment, rgb-padded, bgra-padded, rgba-padded, cls-off, fixed-width, default-thresholds, dilation, reuse, recovery.\n--list prints the selected plan without building workers or running inference.\nEvidence: target/qualification/<timestamp>-<pid>; case deadline 120 seconds; whole suite 60 minutes including adapter build."
}
fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().is_some_and(|s| s == "_worker") {
        if args.len() != 5 {
            return Err("invalid worker arguments".into());
        }
        return rust_worker(
            Path::new(&args[1]),
            Path::new(&args[2]),
            &args[3],
            Path::new(&args[4]),
        );
    }
    if args.iter().any(|s| s == "--help" || s == "-h") {
        println!("{}", usage());
        return Ok(());
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut reference = root.join("../SimdPaddleOCR");
    let mut model_dir = None;
    let mut selected_models = Vec::new();
    let mut selected_profiles = Vec::new();
    let mut selected_fixtures = Vec::new();
    let mut list = false;
    let mut i = 0;
    while i < args.len() {
        let flag = &args[i];
        if flag == "--list" {
            list = true;
            i += 1;
            continue;
        }
        i += 1;
        let value = args.get(i).ok_or("missing option value; use --help")?;
        match flag.as_str() {
            "--model" => selected_models.push(value.clone()),
            "--profile" => selected_profiles.push(value.clone()),
            "--fixture" => selected_fixtures.push(value.clone()),
            "--reference-root" => reference = PathBuf::from(value),
            "--models-dir" => model_dir = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown option {flag}").into()),
        }
        i += 1;
    }
    let manifest_path = root.join("tests/baseline/manifest.json");
    let manifest: Manifest = read_json(&manifest_path)?;
    manifest.validate()?;
    let manifest_hash = hash(&fs::read(&manifest_path)?);
    let fixture_ids: Vec<_> = manifest.fixtures.iter().map(|f| f.id.clone()).collect();
    let mut profiles = qualification::profiles(&fixture_ids);
    for id in &selected_profiles {
        if !profiles.iter().any(|p| &p.id == id) {
            return Err(format!("unknown profile {id}").into());
        }
    }
    for id in &selected_models {
        if !manifest.models.iter().any(|m| &m.id == id) {
            return Err(format!("unknown model {id}").into());
        }
    }
    for id in &selected_fixtures {
        if !fixture_ids.contains(id) {
            return Err(format!("unknown fixture {id}").into());
        }
    }
    profiles.retain(|p| selected_profiles.is_empty() || selected_profiles.contains(&p.id));
    if !selected_fixtures.is_empty() {
        if profiles
            .iter()
            .any(|p| p.id == "reuse" || p.id == "recovery")
        {
            return Err(
                "select explicit profiles excluding reuse/recovery before filtering fixtures"
                    .into(),
            );
        }
        for profile in &mut profiles {
            profile.fixtures.retain(|f| selected_fixtures.contains(f));
        }
        profiles.retain(|p| !p.fixtures.is_empty());
    }
    if profiles.is_empty() {
        return Err("selection contains no cases".into());
    }
    let models: Vec<_> = manifest
        .models
        .iter()
        .filter(|m| selected_models.is_empty() || selected_models.contains(&m.id))
        .collect();
    let total: usize = models.len() * profiles.iter().map(|p| p.fixtures.len()).sum::<usize>();
    if list {
        println!(
            "{}",
            serde_json::to_string_pretty(
                &serde_json::json!({"models": models.iter().map(|m| &m.id).collect::<Vec<_>>(), "profiles": profiles, "case_count":total, "case_timeout_seconds":CASE_TIMEOUT.as_secs(), "suite_timeout_seconds":SUITE_TIMEOUT.as_secs()})
            )?
        );
        return Ok(());
    }
    if cfg!(debug_assertions) || !cfg!(feature = "parallel") || cfg!(feature = "profiling") {
        return Err("qualification requires a standard --release build with default features and no profiling".into());
    }
    harness::check_standard_environment()?;
    let reference = harness::canonical_path(&reference)?;
    harness::check_reference(&reference, &manifest.reference_commit)?;
    let model_dir =
        harness::canonical_path(&model_dir.unwrap_or_else(|| reference.join("models")))?;
    manifest.check_assets(&model_dir)?;
    // Verify every checked-in golden and pixel hash before any inference.
    let frames = baseline::decode_frames(&manifest, &root.join("tests/baseline"), true)?;
    let goldens: BTreeMap<String, Golden> = manifest
        .models
        .iter()
        .map(|m| {
            let golden: Golden =
                read_json(&root.join(format!("tests/baseline/expected/{}.json", m.id)))?;
            golden.validate(&m.id, &manifest, &manifest_hash)?;
            Ok((m.id.clone(), golden))
        })
        .collect::<Result<_>>()?;
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();
    let run_dir = root.join(format!(
        "target/qualification/{stamp}-{}",
        std::process::id()
    ));
    fs::create_dir_all(&run_dir)?;
    let coverage = if selected_models.is_empty()
        && selected_profiles.is_empty()
        && selected_fixtures.is_empty()
    {
        "full"
    } else {
        "focused"
    };
    let source_hash = harness::source_tree_hash(&root, SOURCE_PATHS)?;
    let mut metadata = serde_json::json!({
        "schema_version":1,"status":"incomplete","manifest_sha256":manifest_hash,
        "reference_commit":manifest.reference_commit,"profiles":profiles,
        "models":models.iter().map(|m| &m.id).collect::<Vec<_>>(),"case_count":total,
        "coverage":coverage,"completed_cases":0,
        "case_timeout_seconds":CASE_TIMEOUT.as_secs(),"suite_timeout_seconds":SUITE_TIMEOUT.as_secs(),
        "host_available_parallelism":std::thread::available_parallelism().map_or(1,usize::from),
        "build":"standard release; default features; automatic SIMD/layout; fusion enabled",
        "rust_source_sha256":source_hash,
        "rust_executable_sha256":hash(&fs::read(std::env::current_exe()?)?),
        "source_hash_paths":SOURCE_PATHS,
        "rustc":harness::command_text(Command::new("rustc").arg("-Vv"))?,
        "rust_head":harness::command_text(Command::new("git").args(["rev-parse","HEAD"]).current_dir(&root))?,
        "rust_worktree":harness::command_text(Command::new("git").args(["status","--porcelain"]).current_dir(&root))?,
        "dotnet_info":harness::command_text(Command::new("dotnet").arg("--info").current_dir(root.join("tools/csharp-qualification")))?,
    });
    write_json(&run_dir.join("run.json"), &metadata)?;
    println!(
        "Artifacts: {} ({total} cases, one run per case)",
        run_dir.display()
    );
    let start = Instant::now();
    let mut completed = 0;
    let mut outcome = (|| -> Result<()> {
        let dll = harness::build_csharp(
            &root,
            &reference,
            &root.join("tools/csharp-qualification/Qualification.csproj"),
            &run_dir,
            remaining(start)?,
        )?;
        metadata["csharp_adapter_sha256"] = hash(&fs::read(&dll)?).into();
        write_json(&run_dir.join("run.json"), &metadata)?;
        let mut failures = Vec::new();
        for profile in &profiles {
            let profile_dir = run_dir.join(&profile.id);
            fs::create_dir_all(&profile_dir)?;
            let profile_path = profile_dir.join("profile.json");
            write_json(&profile_path, profile)?;
            let inputs = prepare_inputs(&frames, profile, &profile_dir)?;
            write_json(&profile_dir.join("inputs.json"), &inputs)?;
            for model in &models {
                let capacity = if profile.capacity == 0 {
                    std::thread::available_parallelism().map_or(1, usize::from)
                } else {
                    profile.capacity
                };
                let mut rust_cmd = Command::new(std::env::current_exe()?);
                rust_cmd
                    .arg("_worker")
                    .arg(&manifest_path)
                    .arg(&model_dir)
                    .arg(&model.id)
                    .arg(&profile_path);
                let mut csharp_cmd = Command::new("dotnet");
                csharp_cmd
                    .arg(&dll)
                    .arg(&manifest_path)
                    .arg(&model_dir)
                    .arg(&model.id)
                    .arg(&profile_path)
                    .env("DOTNET_PROCESSOR_COUNT", capacity.to_string());
                let mut rust = Worker::spawn(
                    &mut rust_cmd,
                    &profile_dir.join(format!("{}-rust", model.id)),
                )?;
                let mut csharp = Worker::spawn(
                    &mut csharp_cmd,
                    &profile_dir.join(format!("{}-csharp", model.id)),
                )?;
                for input in &inputs {
                    let expected = csharp.case(input, start)?;
                    let actual = rust.case(input, start)?;
                    let differences =
                        qualification::compare(&model.id, &expected, &actual, input, profile);
                    failures.extend(differences);
                    // The default parallel configuration must also preserve the serial golden contract.
                    if profile.id == "default-parallel" {
                        let golden = goldens[&model.id]
                            .cases
                            .iter()
                            .find(|c| c.id == input.fixture)
                            .ok_or("missing golden case")?;
                        if let Some(result) = &actual.result {
                            failures.extend(baseline::compare(&model.id, golden, result));
                        }
                        if let Some(result) = &expected.result {
                            failures.extend(baseline::compare(
                                &format!("{} C#", model.id),
                                golden,
                                result,
                            ));
                        }
                    }
                    completed += 1;
                    println!(
                        "[{completed}/{total}] {}/{} {}",
                        model.id, profile.id, input.id
                    );
                    write_json(&run_dir.join("differences.json"), &failures)?;
                    metadata["completed_cases"] = completed.into();
                    write_json(&run_dir.join("run.json"), &metadata)?;
                }
                rust.finish(start)?;
                csharp.finish(start)?;
            }
        }
        if !failures.is_empty() {
            for failure in failures.iter().take(25) {
                eprintln!("{failure}");
            }
            return Err(format!("{} differences; see differences.json", failures.len()).into());
        }
        Ok(())
    })();
    match harness::source_tree_hash(&root, SOURCE_PATHS) {
        Ok(current) if current == source_hash => (),
        current => {
            let error = match current {
                Ok(_) => "Rust source changed during qualification; results do not qualify one code version".to_owned(),
                Err(error) => format!("could not recheck qualification source version: {error}"),
            };
            metadata["version_error"] = error.clone().into();
            outcome = Err(error.into());
        }
    }
    metadata["elapsed_seconds"] = start.elapsed().as_secs_f64().into();
    metadata["status"] = if outcome.is_ok() { "passed" } else { "failed" }.into();
    if let Err(error) = &outcome {
        metadata["error"] = error.to_string().into();
    }
    write_json(&run_dir.join("run.json"), &metadata)?;
    if outcome.is_ok() {
        println!("{coverage} qualification passed: {completed}/{total} selected requests");
    }
    outcome
}
fn prepare_inputs(frames: &[Frame], profile: &Profile, directory: &Path) -> Result<Vec<Input>> {
    profile
        .fixtures
        .iter()
        .enumerate()
        .map(|(index, id)| {
            let frame = frames
                .iter()
                .find(|f| &f.id == id)
                .ok_or("profile refers to missing fixture")?;
            let (bytes, mut stride) = qualification::pixels(frame, profile)?;
            let path = directory.join(format!("{index}-{id}.pixels"));
            fs::write(&path, bytes)?;
            let expected_error = if profile.id == "recovery" {
                match index {
                    1 => {
                        stride = 1;
                        Some("invalid_stride".into())
                    }
                    3 => Some("pixel_limit".into()),
                    _ => None,
                }
            } else {
                None
            };
            Ok(Input {
                id: if profile.id == "default-parallel" {
                    id.clone()
                } else {
                    format!("{index}-{id}")
                },
                fixture: id.clone(),
                path: path.to_str().ok_or("non-UTF8 input path")?.into(),
                width: frame.width as usize,
                height: frame.height as usize,
                stride,
                format: profile.format.clone(),
                expected_error,
            })
        })
        .collect()
}
fn remaining(start: Instant) -> Result<Duration> {
    SUITE_TIMEOUT
        .checked_sub(start.elapsed())
        .filter(|d| !d.is_zero())
        .ok_or_else(|| "60-minute qualification suite deadline exceeded".into())
}
struct Worker {
    child: Child,
    input: Option<ChildStdin>,
    lines: mpsc::Receiver<std::io::Result<String>>,
}
impl Worker {
    fn spawn(command: &mut Command, path: &Path) -> Result<Self> {
        let mut transcript = fs::File::create(path.with_extension("jsonl"))?;
        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(fs::File::create(path.with_extension("log"))?)
            .spawn()?;
        let input = child.stdin.take();
        let stdout = child.stdout.take().ok_or("worker stdout unavailable")?;
        let (sender, lines) = mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                if let Ok(text) = &line {
                    let _ = writeln!(transcript, "{text}");
                    let _ = transcript.flush();
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
    fn case(&mut self, input: &Input, start: Instant) -> Result<Response> {
        let timeout = remaining(start)?.min(CASE_TIMEOUT);
        let stdin = self.input.as_mut().ok_or("worker input closed")?;
        serde_json::to_writer(&mut *stdin, input)?;
        stdin.write_all(b"\n")?;
        stdin.flush()?;
        let line = self.lines.recv_timeout(timeout).map_err(|e| {
            format!(
                "{}: worker stopped or exceeded case/suite deadline: {e}",
                input.id
            )
        })??;
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
        if !matches!(self.child.try_wait(), Ok(Some(_))) {
            harness::kill_tree(&mut self.child);
        }
    }
}
fn pixel_format(value: &str) -> Result<PixelFormat> {
    match value {
        "Bgr24" => Ok(PixelFormat::Bgr24),
        "Rgb24" => Ok(PixelFormat::Rgb24),
        "Bgra32" => Ok(PixelFormat::Bgra32),
        "Rgba32" => Ok(PixelFormat::Rgba32),
        _ => Err("unknown pixel format".into()),
    }
}
fn rust_worker(
    manifest_path: &Path,
    directory: &Path,
    model_id: &str,
    profile_path: &Path,
) -> Result<()> {
    let manifest: Manifest = read_json(manifest_path)?;
    let profile: Profile = read_json(profile_path)?;
    let model = manifest
        .models
        .iter()
        .find(|m| m.id == model_id)
        .ok_or("unknown model")?;
    let s = &manifest.settings;
    let models = OcrModelSet::from_paths(
        directory.join(&model.det),
        directory.join(&model.rec),
        directory.join(&model.dictionary),
        Some(directory.join(&model.cls)),
    )?;
    let cpu = if profile.capacity == 0 {
        Arc::new(CpuContext::automatic()?)
    } else {
        Arc::new(CpuContext::new(profile.capacity)?)
    };
    let mut ocr = models
        .builder()
        .cpu_context(cpu)
        .cpu_options(CpuOptions {
            line_workers: profile.line_workers,
            preprocess_workers: profile.preprocess_workers,
            detector_threads: profile.detector_threads,
        })
        .detector_options(DetectorOptions {
            limit_side_length: s.detector_limit,
            max_candidates: s.max_candidates,
            bitmap_threshold: s.bitmap_threshold,
            box_threshold: (!profile.default_thresholds).then_some(model.box_threshold),
            unclip_ratio: s.unclip_ratio,
            use_dilation: profile.dilation,
            max_image_pixels: profile.max_image_pixels,
        })
        .recognizer_options(RecognizerOptions {
            target_width: profile.target_width,
            adaptive_width: profile.adaptive_width,
            ..Default::default()
        })
        .line_options(LineOptions {
            classifier_threshold: s.classifier_threshold,
            ..Default::default()
        })
        .direction_classification(profile.classification)
        .recognition_batch_size(profile.batch_size)
        .build()?;
    eprintln!(
        "CPU budgets: {:?}; compile {:?}; backend {:?}",
        ocr.cpu_budgets(),
        ocr.recognizer_mut().compiled().options(),
        ocr.recognizer_mut().compiled().backend()
    );
    for line in std::io::stdin().lock().lines() {
        let input: Input = serde_json::from_str(&line?)?;
        let outcome = (|| -> Result<Response> {
            let pixels = fs::read(&input.path)?;
            let image = ImageView::new(
                &pixels,
                input.width,
                input.height,
                input.stride,
                pixel_format(&input.format)?,
            )?;
            let view = ocr.run(image, profile.capture)?;
            let result = CaseResult {
                id: input.id.clone(),
                text: view.text.into(),
                detected_count: view.detected_count,
                detector_resized_width: view.detector_resized_width,
                detector_resized_height: view.detector_resized_height,
                lines: view
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
            let alignments = view
                .iter()
                .map(|l| -> Result<Alignment> {
                    Ok(Alignment {
                        spans: l.ctc_spans.map(|spans| {
                            spans
                                .iter()
                                .map(|s| Span {
                                    text: s.text.into(),
                                    score: s.score.into(),
                                    start_column: s.start_column,
                                    end_column: s.end_column,
                                })
                                .collect()
                        }),
                        character_boxes: if profile.capture {
                            Some(
                                l.estimate_character_boxes()?
                                    .iter()
                                    .map(|b| CharacterBox {
                                        text: b.text.into(),
                                        score: b.score.into(),
                                        points: b.points.map(|p| p.map(f64::from)),
                                    })
                                    .collect(),
                            )
                        } else {
                            None
                        },
                    })
                })
                .collect::<Result<_>>()?;
            Ok(Response {
                id: input.id.clone(),
                result: Some(result),
                alignments,
                error: None,
            })
        })();
        let response = outcome.unwrap_or_else(|e| Response {
            id: input.id,
            result: None,
            alignments: Vec::new(),
            error: Some(e.to_string()),
        });
        serde_json::to_writer(std::io::stdout().lock(), &response)?;
        println!();
        std::io::stdout().flush()?;
    }
    Ok(())
}
