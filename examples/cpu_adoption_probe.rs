//! Bounded production-session probe for comparing two source snapshots.
//!
//! Run each snapshot in its own process, alternating snapshot order by case.
//! Keep all other inference idle and impose an external process deadline. This
//! measures a graph on deterministic raw tensors, not end-to-end OCR latency.
use serde::Serialize;
use sha2::{Digest, Sha256};
use simd_paddle_ocr_rs::{
    cpu::CpuContext,
    model::Model,
    runtime::{CompileOptions, CompiledModel, ExecutionOptions, ExecutionStatistics, Session},
    tensor::Shape,
};
use std::{
    error::Error,
    io::Write,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Clone)]
struct Case {
    id: String,
    path: PathBuf,
    shape: Vec<usize>,
}

#[derive(Serialize)]
struct Sample {
    milliseconds: f64,
    output_sha256: String,
}

#[derive(Serialize)]
struct CaseReport {
    case: String,
    model_path: PathBuf,
    model_sha256: String,
    input_shape: Vec<usize>,
    input_sha256: String,
    input_pattern: &'static str,
    output_shape: Vec<usize>,
    output_f32le_file: Option<PathBuf>,
    load_compile_bind_ms: f64,
    warmup: Sample,
    samples: Vec<Sample>,
    median_ms: f64,
    statistics: ExecutionStatistics,
    retained_planning_bytes: usize,
    naive_activation_bytes: usize,
}

fn sha_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn sha_floats(values: &[f32]) -> Result<String> {
    let mut digest = Sha256::new();
    for &value in values {
        if !value.is_finite() {
            return Err("non-finite input or output in adoption probe".into());
        }
        digest.update(value.to_bits().to_le_bytes());
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn dump_floats(path: &Path, values: &[f32]) -> Result<()> {
    // Separate snapshot directories prevent accidental baseline replacement.
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    let mut writer = std::io::BufWriter::new(file);
    for &value in values {
        writer.write_all(&value.to_bits().to_le_bytes())?;
    }
    writer.flush()?;
    Ok(())
}

fn budget(started: Instant, limit: Duration) -> Result<()> {
    if started.elapsed() >= limit {
        return Err("adoption probe exceeded its fixed execution budget".into());
    }
    Ok(())
}

fn run_case(
    case: Case,
    cpu: Arc<CpuContext>,
    threads: usize,
    started: Instant,
    limit: Duration,
    dump_dir: Option<&Path>,
) -> Result<CaseReport> {
    budget(started, limit)?;
    let setup = Instant::now();
    let bytes = std::fs::read(&case.path)?;
    let model_sha256 = sha_bytes(&bytes);
    let model = Arc::new(Model::load_bytes(&bytes)?);
    let compiled = Arc::new(CompiledModel::compile(model, CompileOptions::default())?);
    let mut session = Session::new(
        compiled,
        cpu,
        &case.shape,
        ExecutionOptions {
            intra_op_threads: threads,
            ..Default::default()
        },
    )?;
    let load_compile_bind_ms = setup.elapsed().as_secs_f64() * 1000.;
    let elements = Shape::new(&case.shape)?.element_count();
    // Fixed exactly representable F32 values; no decoder or preprocessing
    // implementation can change the input between source snapshots.
    let input: Vec<f32> = (0..elements)
        .map(|i| (((i % 101) * 37 % 101) as i32 - 50) as f32 / 64.)
        .collect();
    let input_sha256 = sha_floats(&input)?;
    budget(started, limit)?;
    let warmup_start = Instant::now();
    let output = session.run(&input)?;
    let warmup_ms = warmup_start.elapsed().as_secs_f64() * 1000.;
    let output_shape = output.shape().dimensions().to_vec();
    let output_sha256 = sha_floats(output.data())?;
    let warmup = Sample {
        milliseconds: warmup_ms,
        output_sha256: output_sha256.clone(),
    };
    let output_f32le_file = dump_dir.map(|dir| dir.join(format!("{}.f32le", case.id)));
    budget(started, limit)?;
    let mut samples = Vec::with_capacity(3);
    for sample_index in 0..3 {
        budget(started, limit)?;
        let sample_start = Instant::now();
        let output = session.run(&input)?;
        let milliseconds = sample_start.elapsed().as_secs_f64() * 1000.;
        // Hashing and correctness checks are outside the timed graph call.
        let digest = sha_floats(output.data())?;
        if output.shape().dimensions() != output_shape || digest != output_sha256 {
            return Err(format!("{} output changed after warm-up", case.id).into());
        }
        samples.push(Sample {
            milliseconds,
            output_sha256: digest,
        });
        if sample_index == 2
            && let Some(path) = &output_f32le_file
        {
            dump_floats(path, output.data())?;
        }
        budget(started, limit)?;
    }
    let mut times: Vec<_> = samples.iter().map(|s| s.milliseconds).collect();
    times.sort_by(f64::total_cmp);
    Ok(CaseReport {
        case: case.id,
        model_path: case.path,
        model_sha256,
        input_shape: case.shape,
        input_sha256,
        input_pattern: "F32((((i % 101) * 37 % 101) - 50) / 64), logical NCHW",
        output_shape,
        output_f32le_file,
        load_compile_bind_ms,
        warmup,
        samples,
        median_ms: times[1],
        statistics: session.statistics(),
        retained_planning_bytes: session.retained_planning_bytes(),
        naive_activation_bytes: session.naive_activation_bytes(),
    })
}

#[cfg(windows)]
fn pin_cpu(threads: usize) -> Result<Option<String>> {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> *mut std::ffi::c_void;
        fn GetProcessAffinityMask(
            process: *mut std::ffi::c_void,
            process_mask: *mut usize,
            system_mask: *mut usize,
        ) -> i32;
        fn SetProcessAffinityMask(process: *mut std::ffi::c_void, mask: usize) -> i32;
    }
    let (mut allowed, mut system) = (0usize, 0usize);
    // SAFETY: pseudo process handle is valid; masks point to writable locals.
    let process = unsafe { GetCurrentProcess() };
    if unsafe { GetProcessAffinityMask(process, &mut allowed, &mut system) } == 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    if (allowed.count_ones() as usize) < threads {
        return Err("requested threads exceed allowed affinity CPUs".into());
    }
    let mut selected = 0usize;
    for _ in 0..threads {
        let bit = 1usize << allowed.trailing_zeros();
        selected |= bit;
        allowed &= !bit;
    }
    // SAFETY: selected is a nonzero subset of this process's allowed CPUs.
    if unsafe { SetProcessAffinityMask(process, selected) } == 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    Ok(Some(format!("{selected:X}")))
}

#[cfg(not(windows))]
fn pin_cpu(_threads: usize) -> Result<Option<String>> {
    Ok(None)
}

fn main() {
    if let Err(error) = run() {
        eprintln!("CPU adoption probe: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    if cfg!(debug_assertions) || cfg!(feature = "profiling") {
        return Err("use --release with the default features and no profiling".into());
    }
    let mut model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../SimdPaddleOCR/models");
    let mut selected = "all".to_owned();
    let mut label = "unlabelled".to_owned();
    let mut threads = 1usize;
    let mut seconds = 60u64;
    let mut output = None::<PathBuf>;
    let mut dump_dir = None::<PathBuf>;
    let mut custom_model = None::<PathBuf>;
    let mut custom_shape = None::<Vec<usize>>;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--help" || arg == "-h" {
            println!(
                "cpu_adoption_probe [--case all|tiny-det|small-rec|medium-rec] [--models-dir DIR] [--threads 1..16] [--label LABEL] [--budget-seconds 1..60] [--output FILE] [--dump-dir DIR]\nAlternative custom case: --model ONNX --shape 1,3,H,W\nExactly one warm-up and three timed production graph runs per case. Hashes/checks occur outside timers. Optional dumps store the last sample's logical F32 little-endian output bits after its timer and refuse existing files. Pin to the first allowed CPUs on Windows. Enforce an external deadline; compare baseline/candidate output hashes and all input/model/shape/configuration fields before interpreting medians. These are raw-tensor graph timings, not full OCR timings."
            );
            return Ok(());
        }
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {arg}"))?;
        match arg.as_str() {
            "--case" => selected = value,
            "--models-dir" => model_dir = value.into(),
            "--threads" => threads = value.parse()?,
            "--label" => label = value,
            "--budget-seconds" => seconds = value.parse()?,
            "--output" => output = Some(value.into()),
            "--dump-dir" => dump_dir = Some(value.into()),
            "--model" => custom_model = Some(value.into()),
            "--shape" => {
                custom_shape = Some(
                    value
                        .split(',')
                        .map(str::parse)
                        .collect::<std::result::Result<_, _>>()?,
                );
            }
            _ => return Err(format!("unknown argument {arg}").into()),
        }
    }
    if !(1..=16).contains(&threads) || !(1..=60).contains(&seconds) {
        return Err("threads must be 1..16 and budget-seconds 1..60".into());
    }
    let cases = match (custom_model, custom_shape) {
        (Some(path), Some(shape)) => {
            if selected != "all" || shape.len() != 4 || shape.contains(&0) {
                return Err("custom model/shape requires no --case and positive N,C,H,W".into());
            }
            vec![Case {
                id: "custom".into(),
                path,
                shape,
            }]
        }
        (None, None) => {
            let cases: Vec<_> = [
                ("tiny-det", "det.onnx", vec![1, 3, 320, 320]),
                ("small-rec", "small_rec.onnx", vec![1, 3, 48, 128]),
                ("medium-rec", "medium_rec.onnx", vec![1, 3, 48, 320]),
            ]
            .into_iter()
            .filter(|(id, _, _)| selected == "all" || selected == *id)
            .map(|(id, file, shape)| Case {
                id: id.into(),
                path: model_dir.join(file),
                shape,
            })
            .collect();
            if cases.is_empty() {
                return Err("unknown --case; use --help".into());
            }
            cases
        }
        _ => return Err("--model and --shape must be supplied together".into()),
    };
    let started = Instant::now();
    let limit = Duration::from_secs(seconds);
    if let Some(path) = &dump_dir {
        std::fs::create_dir_all(path)?;
    }
    let affinity = pin_cpu(threads)?;
    let cpu = Arc::new(CpuContext::new(threads)?);
    let mut reports = Vec::new();
    for case in cases {
        reports.push(run_case(
            case,
            cpu.clone(),
            threads,
            started,
            limit,
            dump_dir.as_deref(),
        )?);
    }
    budget(started, limit)?;
    let report = serde_json::json!({
        "schema_version": 1,
        "label": label,
        "purpose": "fixed-sample production Session graph comparison on deterministic raw tensors",
        "end_to_end_ocr": false,
        "instrumented": false,
        "compile_options": CompileOptions::default(),
        "threads": threads,
        "affinity_hex": affinity,
        "warmup_runs_per_case": 1,
        "measured_runs_per_case": 3,
        "budget_seconds": seconds,
        "total_process_execution_ms": started.elapsed().as_secs_f64() * 1000.,
        "cases": reports,
    });
    let json = serde_json::to_string_pretty(&report)?;
    if let Some(path) = output {
        std::fs::write(path, &json)?;
    }
    println!("{json}");
    Ok(())
}
