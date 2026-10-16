//! Interleaved, raw-pixel CPU benchmark. Run with --release; see CPU validation docs.
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use simd_paddle_ocr_rs::{
    cpu::CpuOptions,
    detector::{Detector, DetectorOptions},
    image::{ImageView, PixelFormat},
    lines::{LineOptions, OrientedLineExtractor},
    model::Model,
    ocr::{Ocr, OcrOptions},
    recognizer::{Dictionary, Recognizer, RecognizerOptions},
    runtime::{
        CompileOptions, CompiledModel, CpuContext, ExecutionLimits, ExecutionOptions, KernelPolicy,
        LayoutPolicy, Session,
    },
};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fs,
    hint::black_box,
    path::{Path, PathBuf},
    process::Command,
    sync::Arc,
    time::Instant,
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const MODELS: [&str; 7] = [
    "cls",
    "det",
    "small_det",
    "medium_det",
    "rec",
    "small_rec",
    "medium_rec",
];

struct Image {
    id: String,
    width: usize,
    height: usize,
    bytes: Vec<u8>,
}
impl Image {
    fn view(&self) -> Result<ImageView<'_>> {
        Ok(ImageView::new(
            &self.bytes,
            self.width,
            self.height,
            self.width * 3,
            PixelFormat::Bgr24,
        )?)
    }
}

struct Variant {
    name: String,
    options: CompileOptions,
    threads: usize,
    cpu: Arc<CpuContext>,
    models: BTreeMap<String, Arc<CompiledModel>>,
    compile: Vec<Value>,
}
struct VariantConfig {
    name: String,
    options: CompileOptions,
    threads: usize,
}

enum Runner {
    Graph(Box<Session>),
    Ocr(Box<Ocr>),
}
enum Input<'a> {
    Graph(&'a [f32]),
    Image(&'a Image),
}
impl Runner {
    fn execute(&mut self, input: &Input<'_>) -> Result<()> {
        match (self, input) {
            (Self::Graph(session), Input::Graph(values)) => {
                black_box(session.run(values)?.data());
            }
            (Self::Ocr(ocr), Input::Image(image)) => {
                black_box(ocr.run(image.view()?, true)?);
            }
            _ => return Err("benchmark runner/input mismatch".into()),
        }
        Ok(())
    }
    fn memory(&self) -> Value {
        match self {
            Self::Graph(session) => json!({
                "retained_activation_bytes":session.retained_activation_bytes(),
                "retained_planning_bytes":session.retained_planning_bytes(),
                "retained_scratch_bytes":session.retained_scratch_bytes(),
                "effective_intra_op_threads":session.intra_op_threads(),
                "statistics":session.statistics(),
            }),
            Self::Ocr(ocr) => json!({
                "retained_activation_bytes":ocr.retained_activation_bytes(),
                "retained_workspace_bytes":ocr.retained_workspace_bytes(),
                "diagnostic_retained_activation_bytes":ocr.diagnostic_retained_activation_bytes(),
                "effective_cpu_budgets":{"capacity":ocr.cpu_budgets().capacity,
                    "line_workers":ocr.cpu_budgets().line_workers,"preprocess_workers":ocr.cpu_budgets().preprocess_workers,
                    "detector_threads":ocr.cpu_budgets().detector_threads,"recognizer_base_threads":ocr.cpu_budgets().recognizer_base_threads,
                    "classifier_threads":1},
            }),
        }
    }
    fn reshape(&mut self, shape: &[usize]) -> Result<u64> {
        let start = Instant::now();
        match self {
            Self::Graph(session) => session.reshape(shape)?,
            _ => return Err("reshape requires a graph runner".into()),
        }
        Ok(ns(start))
    }
}

fn ns(start: Instant) -> u64 {
    start.elapsed().as_nanos().try_into().unwrap_or(u64::MAX)
}
fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn kernel(name: &str) -> Result<KernelPolicy> {
    Ok(match name {
        "auto" => KernelPolicy::Auto,
        "scalar" => KernelPolicy::Scalar,
        "portable4" => KernelPolicy::Portable4,
        "avx" => KernelPolicy::Avx,
        "avx-fma" => KernelPolicy::AvxFma,
        "avx2-fma" => KernelPolicy::Avx2Fma,
        "avx512" => KernelPolicy::Avx512,
        _ => return Err(format!("unknown kernel policy: {name}").into()),
    })
}
fn variant_configs(names: &str, backends: bool, threads: usize) -> Result<Vec<VariantConfig>> {
    let mut seen = BTreeSet::new();
    names
        .split(',')
        .map(|name| {
            if !seen.insert(name) {
                return Err(format!("duplicate variant: {name}").into());
            }
            let options = if backends {
                CompileOptions {
                    kernels: kernel(name)?,
                    ..Default::default()
                }
            } else {
                match name {
                    "reference" => CompileOptions::reference(),
                    "simd" => CompileOptions {
                        layout: LayoutPolicy::Nchw,
                        fusion: false,
                        ..Default::default()
                    },
                    "nhwc" => CompileOptions {
                        fusion: false,
                        ..Default::default()
                    },
                    "auto" => CompileOptions::default(),
                    _ => return Err(format!("unknown stage: {name}").into()),
                }
            };
            Ok(VariantConfig {
                name: name.into(),
                options,
                threads,
            })
        })
        .collect()
}

fn thread_configs(names: &str) -> Result<Vec<VariantConfig>> {
    let mut seen = BTreeSet::new();
    let configs = names
        .split(',')
        .map(|name| {
            let threads: usize = name.parse()?;
            if !(1..=16).contains(&threads) || !seen.insert(threads) {
                return Err("thread-counts must be distinct values within 1..=16".into());
            }
            Ok(VariantConfig {
                name: format!("auto-c{threads}"),
                options: CompileOptions::default(),
                threads,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    if configs.len() < 2 || configs[0].threads != 1 {
        return Err(
            "thread-counts must start with the one-thread baseline and include a candidate".into(),
        );
    }
    Ok(configs)
}

fn corpus(path: &Path, selected: &[String]) -> Result<Vec<Image>> {
    let manifest: Value = serde_json::from_slice(&fs::read(path)?)?;
    if manifest["schema_version"] != 1 || manifest["pixel_format"] != "bgr24" {
        return Err("unsupported raw corpus manifest".into());
    }
    let root = path
        .parent()
        .ok_or("corpus manifest has no parent")?
        .canonicalize()?;
    let mut images = Vec::new();
    for id in selected {
        let entry = manifest["images"]
            .as_array()
            .ok_or("missing corpus images")?
            .iter()
            .find(|image| image["id"].as_str() == Some(id))
            .ok_or_else(|| format!("required benchmark image is missing: {id}"))?;
        let file = root
            .join(entry["file"].as_str().ok_or("missing raw path")?)
            .canonicalize()?;
        if !file.starts_with(&root) {
            return Err("raw corpus path escapes its directory".into());
        }
        let bytes = fs::read(file)?;
        let width = usize::try_from(entry["width"].as_u64().ok_or("missing width")?)?;
        let height = usize::try_from(entry["height"].as_u64().ok_or("missing height")?)?;
        if width == 0
            || height == 0
            || width.checked_mul(height).and_then(|n| n.checked_mul(3)) != Some(bytes.len())
            || entry["sha256"].as_str() != Some(digest(&bytes).as_str())
        {
            return Err(format!("invalid raw corpus bytes: {id}").into());
        }
        images.push(Image {
            id: id.clone(),
            width,
            height,
            bytes,
        });
    }
    Ok(images)
}

#[derive(Clone, Copy)]
struct Protocol {
    warmups: usize,
    batches: usize,
    iterations: usize,
    min_batch_ns: u64,
}
struct Progress<'a> {
    path: &'a Path,
    case_id: &'a str,
    names: &'a [String],
}
impl Progress<'_> {
    fn note(&self, phase: &str, index: usize, batch: Option<usize>, data: Value) -> Result<()> {
        println!(
            "{} / {} / {} / batch {:?}",
            self.case_id,
            self.names[index],
            phase,
            batch.map(|n| n + 1)
        );
        write_json(
            self.path,
            &json!({"schema_version":1,"complete":false,"case_id":self.case_id,
            "variant":self.names[index],"phase":phase,"batch":batch,"data":data,
            "timing_contract":"Progress and checkpoint serialization occurs outside timed regions."}),
        )
    }
}
struct Measurements {
    variants: Vec<Value>,
    order: Vec<Vec<usize>>,
    iterations: usize,
}
fn write_json(path: &Path, value: &Value) -> Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}
fn batch_iterations(requested: usize, minimum_ns: u64, fastest_warm_ns: u64) -> usize {
    let automatic = minimum_ns.div_ceil(fastest_warm_ns.max(1)).min(10_000) as usize;
    requested.max(automatic)
}

/// Repeated batches rotate the order and use one common iteration count per
/// case, estimated from existing warmups. Progress is outside timed regions.
fn measure(
    runners: &mut [Runner],
    input: &Input<'_>,
    protocol: Protocol,
    progress: &Progress<'_>,
    offset: usize,
) -> Result<Measurements> {
    let mut first = Vec::new();
    let mut warm = Vec::new();
    for (index, runner) in runners.iter_mut().enumerate() {
        progress.note("first-and-warmup-start", index, None, Value::Null)?;
        let start = Instant::now();
        runner.execute(input)?;
        first.push(ns(start));
        let mut times = Vec::new();
        for _ in 0..protocol.warmups {
            let start = Instant::now();
            runner.execute(input)?;
            times.push(ns(start));
        }
        progress.note(
            "warmup-complete",
            index,
            None,
            json!({"first_case_run_ns":first[index],"warmup_ns":times}),
        )?;
        warm.push(times);
    }
    let fastest = warm
        .iter()
        .flatten()
        .copied()
        .min()
        .ok_or("missing warmup samples")?;
    let iterations = batch_iterations(protocol.iterations, protocol.min_batch_ns, fastest);
    let mut samples = vec![Vec::with_capacity(protocol.batches); runners.len()];
    let mut order_log = Vec::new();
    for batch in 0..protocol.batches {
        let order: Vec<_> = (0..runners.len())
            .map(|index| (index + batch + offset) % runners.len())
            .collect();
        for &index in &order {
            progress.note(
                "timed-batch-start",
                index,
                Some(batch),
                json!({"iterations":iterations}),
            )?;
            let start = Instant::now();
            for _ in 0..iterations {
                runners[index].execute(input)?;
            }
            let elapsed = ns(start);
            samples[index].push(elapsed as f64 / iterations as f64);
            progress.note("timed-batch-complete", index, Some(batch), json!({"iterations":iterations,"batch_ns":elapsed,"samples_ns_per_inference":samples}))?;
        }
        order_log.push(order);
    }
    let rows = samples.into_iter().enumerate().map(|(index, samples)| {
        let mut sorted = samples.clone();
        sorted.sort_by(f64::total_cmp);
        let batches = protocol.batches;
        let median = if batches.is_multiple_of(2) { (sorted[batches / 2 - 1] + sorted[batches / 2]) * 0.5 } else { sorted[batches / 2] };
        let p95 = sorted[(batches * 95).div_ceil(100).saturating_sub(1)];
        json!({"first_case_run_ns":first[index],"batch_ns_per_inference":samples,"median_ns":median,"p95_ns":p95,
            "warmup_ns":warm[index],"iterations_per_batch":iterations,"fastest_warm_ns":fastest,
            "throughput_per_second":1e9/median,"memory":runners[index].memory()})
    }).collect();
    Ok(Measurements {
        variants: rows,
        order: order_log,
        iterations,
    })
}

fn source_files(root: &Path, directory: &Path, files: &mut BTreeMap<String, String>) -> Result<()> {
    for entry in fs::read_dir(directory)? {
        let path = entry?.path();
        if path.is_dir() {
            source_files(root, &path, files)?;
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            files.insert(
                path.strip_prefix(root)?
                    .to_string_lossy()
                    .replace('\\', "/"),
                digest(&fs::read(path)?),
            );
        }
    }
    Ok(())
}
fn command(program: &str, args: &[&str]) -> Result<String> {
    let output = Command::new(program).args(args).output()?;
    if !output.status.success() {
        return Err(format!("metadata command failed: {program}").into());
    }
    Ok(String::from_utf8(output.stdout)?.trim().into())
}
fn fingerprint(manifest: &Path, models: &Path, threads: usize) -> Result<Value> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = BTreeMap::new();
    source_files(root, &root.join("src"), &mut files)?;
    for name in [
        "Cargo.toml",
        "Cargo.lock",
        "rust-toolchain.toml",
        "examples/cpu_bench.rs",
        "tools/reference-lock.json",
    ] {
        files.insert(name.into(), digest(&fs::read(root.join(name))?));
    }
    let mut model_hashes = BTreeMap::new();
    for name in MODELS
        .map(|name| format!("{name}.onnx"))
        .into_iter()
        .chain(["ppocr_keys.txt".into(), "rec_keys.txt".into()])
    {
        model_hashes.insert(name.clone(), digest(&fs::read(models.join(name))?));
    }
    let lock: Value = serde_json::from_slice(&fs::read(root.join("tools/reference-lock.json"))?)?;
    if serde_json::to_value(&model_hashes)? != lock["models"] {
        return Err("benchmark model/dictionary bytes differ from the reference lock".into());
    }
    let build_flags: BTreeMap<_, _> = std::env::vars()
        .filter(|(name, _)| {
            name.contains("RUSTFLAGS")
                || name.starts_with("CARGO_PROFILE_")
                || name == "CARGO_BUILD_TARGET"
        })
        .collect();
    let operating_system = if cfg!(windows) {
        command("cmd", &["/C", "ver"])?
    } else {
        command("uname", &["-a"])?
    };
    let cpu_identity = std::env::var("PROCESSOR_IDENTIFIER").unwrap_or_else(|_| {
        fs::read_to_string("/proc/cpuinfo")
            .unwrap_or_default()
            .lines()
            .find_map(|line| {
                line.strip_prefix("model name")
                    .and_then(|rest| rest.split_once(':').map(|(_, name)| name.trim().to_owned()))
            })
            .unwrap_or_default()
    });
    Ok(
        json!({"host_name":std::env::var("COMPUTERNAME").or_else(|_|std::env::var("HOSTNAME")).unwrap_or_default(),
        "cpu_identity":cpu_identity,"operating_system":operating_system,"architecture":std::env::consts::ARCH,"threads":threads,
        "available_processors":std::thread::available_parallelism().map_or(1,usize::from),
        "features":{"parallel":cfg!(feature="parallel"),"profiling":cfg!(feature="profiling")},
        "rustc":command("rustc", &["--version","--verbose"])?,"rust_revision":command("git", &["-C",env!("CARGO_MANIFEST_DIR"),"rev-parse","HEAD"])?,
        "build_flags":build_flags,"source_snapshot_at_execution":{"sha256":digest(&serde_json::to_vec(&files)?),"files":files},
        "harness_source_sha256":digest(include_bytes!("cpu_bench.rs")),"harness_source_utf8":include_str!("cpu_bench.rs"),
        "binary_sha256":digest(&fs::read(std::env::current_exe()?)?),"corpus_sha256":digest(&fs::read(manifest)?),"model_hashes":model_hashes,
        "images_decoded_inside_timing":false,"release":!cfg!(debug_assertions)}),
    )
}

fn make_ocr(variant: &Variant, models: &Path, det: &str, rec: &str, keys: &str) -> Result<Ocr> {
    let extractor = OrientedLineExtractor::new(
        Detector::new(
            variant.models[det].clone(),
            DetectorOptions {
                limit_side_length: 320,
                ..Default::default()
            },
            ExecutionLimits::default(),
        )?,
        Some(variant.models["cls"].clone()),
        LineOptions::default(),
    )?;
    Ok(Ocr::new_with_execution(
        extractor,
        Recognizer::new(
            variant.models[rec].clone(),
            Arc::new(Dictionary::from_utf8(&fs::read(models.join(keys))?)?),
            RecognizerOptions::default(),
            ExecutionLimits::default(),
        )?,
        OcrOptions::default(),
        variant.cpu.clone(),
        CpuOptions {
            line_workers: variant.threads.min(4),
            preprocess_workers: variant.threads,
            detector_threads: variant.threads,
        },
    )?)
}

fn pair_result(cases: &[Value], from: &str, to: &str, minimum: f64) -> Value {
    let mut log_sum = 0.0;
    let mut regressions = Vec::new();
    let mut measurements = Vec::new();
    for case in cases {
        let before = case["variants"][from]["median_ns"].as_f64().unwrap();
        let after = case["variants"][to]["median_ns"].as_f64().unwrap();
        let ratio = before / after.max(1.0);
        log_sum += ratio.ln();
        measurements
            .push(json!({"id":case["id"],"baseline_ns":before,"current_ns":after,"speedup":ratio}));
        if case["id"].as_str().unwrap().starts_with("ocr/") && after > before * 1.03 {
            regressions
                .push(json!({"id":case["id"],"regression_percent":(after/before-1.0)*100.0}));
        }
    }
    let geometric = (log_sum / cases.len() as f64).exp();
    json!({"from":from,"to":to,"geometric_mean_speedup":geometric,"minimum":minimum,
        "passed":geometric>=minimum&&regressions.is_empty(),"maximum_e2e_regression_percent":3,"e2e_regressions":regressions,"cases":measurements})
}
fn comparisons(cases: &[Value], variants: &[Variant], minimum: f64, thread_counts: bool) -> Value {
    let mut rows = Vec::new();
    if thread_counts {
        for variant in &variants[1..] {
            rows.push(pair_result(cases, "auto-c1", &variant.name, minimum));
        }
        return Value::Array(rows);
    }
    for pair in variants.windows(2) {
        rows.push(pair_result(cases, &pair[0].name, &pair[1].name, minimum));
    }
    Value::Array(rows)
}

fn gate_pairs(names: &[String], backends: bool, thread_counts: bool) -> Vec<(&str, &str, &str)> {
    if thread_counts {
        return names
            .iter()
            .skip(1)
            .map(|name| ("M3", "auto-c1", name.as_str()))
            .collect();
    }
    if backends {
        return names
            .windows(2)
            .map(|pair| ("backend", pair[0].as_str(), pair[1].as_str()))
            .collect();
    }
    let contains = |name: &str| names.iter().any(|value| value == name);
    let mut pairs = Vec::new();
    if contains("reference") && contains("simd") {
        pairs.push(("M1", "reference", "simd"));
    }
    if contains("simd") && contains("auto") {
        pairs.push(("M2", "simd", "auto"));
    }
    if pairs.is_empty() && contains("reference") && contains("auto") {
        pairs.push(("combined", "reference", "auto"));
    }
    pairs
}
struct Report<'a> {
    protocol: Protocol,
    metadata: &'a Value,
    loads: &'a [Value],
    variants: &'a [Variant],
    names: &'a [String],
    minimum: f64,
    backends: bool,
    thread_counts: bool,
    gate: bool,
}
impl Report<'_> {
    fn save(&self, path: &Path, cases: &[Value], complete: bool) -> Result<Value> {
        let acceptance: Vec<_> = gate_pairs(self.names, self.backends, self.thread_counts)
            .into_iter()
            .map(|(milestone, from, to)| {
                let mut result = pair_result(cases, from, to, self.minimum);
                result["milestone"] = json!(milestone);
                result
            })
            .collect();
        let gate_passed = complete && acceptance.iter().all(|value| value["passed"] == true);
        let result = json!({"schema_version":1,"complete":complete,"passed":complete&&(!self.gate||gate_passed),
            "gate_enabled":self.gate,"gate_passed":gate_passed,
            "protocol":{"warmups":self.protocol.warmups,"batches":self.protocol.batches,"requested_iterations_per_batch":self.protocol.iterations,
                "minimum_batch_ms":self.protocol.min_batch_ns as f64/1e6,"maximum_automatic_iterations":10000,"order":"rotating balanced Latin square",
                "timed_region":"inference including CTC decoding; cached raw input; no image-file decoding, tracing, progress writes or serialization",
                "first_case_run":"first run after case binding/reshape; later cases may reuse caches"},
            "metadata":self.metadata,"model_loads":self.loads,
            "variants":self.variants.iter().map(|v|json!({"name":v.name,"options":v.options,"compile":v.compile,
                "requested_threads":v.threads,"context_capacity":v.cpu.capacity(),"classifier_threads":1,
                "requested_stage_budgets":{"line_workers":v.threads.min(4),"preprocess_workers":v.threads,"detector_threads":v.threads}})).collect::<Vec<_>>(),
            "cases":cases,"comparisons":comparisons(cases,self.variants,self.minimum,self.thread_counts),"milestone_acceptance":acceptance});
        write_json(path, &result)?;
        Ok(result)
    }
}

fn main() -> Result<()> {
    if cfg!(debug_assertions) {
        return Err("cpu_bench requires a release build".into());
    }
    let mut args = std::env::args().skip(1);
    let mut options = BTreeMap::new();
    let mut gate = false;
    while let Some(key) = args.next() {
        if key == "--gate" {
            gate = true;
        } else if [
            "--models-dir",
            "--corpus",
            "--output",
            "--variants",
            "--backends",
            "--threads",
            "--thread-counts",
            "--warmups",
            "--batches",
            "--iterations",
            "--images",
            "--min-speedup",
            "--min-batch-ms",
        ]
        .contains(&key.as_str())
        {
            options.insert(key, args.next().ok_or("missing option value")?);
        } else {
            return Err(format!("unknown argument: {key}").into());
        }
    }
    let required = |key: &str| -> Result<PathBuf> {
        Ok(PathBuf::from(
            options.get(key).ok_or_else(|| format!("missing {key}"))?,
        ))
    };
    let number = |key: &str, default: usize| -> Result<usize> {
        Ok(options
            .get(key)
            .map_or(Ok(default), |value| value.parse())?)
    };
    let threads = number("--threads", 1)?;
    let warmups = number("--warmups", 2)?;
    let batches = number("--batches", 12)?;
    let iterations = number("--iterations", 1)?;
    let minimum_batch_ms: f64 = options
        .get("--min-batch-ms")
        .map_or(Ok(10.0), |value| value.parse())?;
    let minimum: f64 = options
        .get("--min-speedup")
        .map_or(Ok(1.0), |value| value.parse())?;
    if !(1..=16).contains(&threads)
        || batches < 2
        || warmups == 0
        || iterations == 0
        || !minimum.is_finite()
        || minimum <= 0.0
        || !minimum_batch_ms.is_finite()
        || !(0.0..=1000.0).contains(&minimum_batch_ms)
    {
        return Err(
            "threads must be 1..=16, batches >=2, warmups/iterations/min-speedup positive, min-batch-ms within 0..=1000".into(),
        );
    }
    let thread_counts = options.contains_key("--thread-counts");
    if ["--variants", "--backends", "--thread-counts"]
        .iter()
        .filter(|key| options.contains_key(**key))
        .count()
        > 1
        || (thread_counts && options.contains_key("--threads"))
    {
        return Err("choose one of --variants, --backends, or --thread-counts; thread-counts supplies its own budgets".into());
    }
    let backends = options.contains_key("--backends");
    let configs = if thread_counts {
        thread_configs(&options["--thread-counts"])?
    } else {
        variant_configs(
            options
                .get(if backends { "--backends" } else { "--variants" })
                .map_or("reference,simd,nhwc,auto", String::as_str),
            backends,
            threads,
        )?
    };
    let mut variants = configs
        .into_iter()
        .map(|config| {
            Ok(Variant {
                name: config.name,
                options: config.options,
                threads: config.threads,
                cpu: Arc::new(CpuContext::new(config.threads)?),
                models: BTreeMap::new(),
                compile: Vec::new(),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    if variants.len() < 2 || !batches.is_multiple_of(variants.len()) {
        return Err(
            "at least two variants are required; batches must be divisible by variant count".into(),
        );
    }
    let names: Vec<_> = variants
        .iter()
        .map(|variant| variant.name.clone())
        .collect();
    if gate && gate_pairs(&names, backends, thread_counts).is_empty() {
        return Err("unfused NHWC comparisons are diagnostic; a milestone gate needs reference+simd or simd+auto".into());
    }
    let protocol = Protocol {
        warmups,
        batches,
        iterations,
        min_batch_ns: (minimum_batch_ms * 1e6) as u64,
    };
    let output = required("--output")?;
    let progress_path = output.with_extension("progress.json");
    let selected: Vec<String> = options
        .get("--images")
        .map_or("sample,text-line,blank", String::as_str)
        .split(',')
        .map(str::to_owned)
        .collect();
    let models = required("--models-dir")?.canonicalize()?;
    let manifest = required("--corpus")?.canonicalize()?;
    let images = corpus(&manifest, &selected)?;
    let mut metadata = fingerprint(
        &manifest,
        &models,
        variants
            .iter()
            .map(|variant| variant.threads)
            .max()
            .unwrap(),
    )?;
    metadata["experiment"] = json!(if thread_counts {
        "interleaved thread scaling"
    } else {
        "interleaved compile policies"
    });
    metadata["compiled_model_sharing"] = json!(if thread_counts {
        "one Arc<CompiledModel> per model shared across all contexts; packed weights shared"
    } else {
        "each compile policy owns its compiled model and packing cache"
    });
    let mut loads = Vec::new();
    for name in MODELS {
        let start = Instant::now();
        let model = Arc::new(Model::load_path(models.join(format!("{name}.onnx")))?);
        loads.push(json!({"model":name,"load_ns":ns(start)}));
        if thread_counts {
            let start = Instant::now();
            let compiled = Arc::new(CompiledModel::compile(model, CompileOptions::default())?);
            let compile_ns = ns(start);
            let owner = variants[0].name.clone();
            for (index, variant) in variants.iter_mut().enumerate() {
                variant.compile.push(json!({"model":name,"compile_ns":if index==0 {compile_ns} else {0},
                    "selected_backend":compiled.backend(),"shared_compiled_from":owner,
                    "shared_across_all_thread_variants":true,"packed_weight_bytes":compiled.packed_weight_bytes()}));
                variant.models.insert(name.into(), compiled.clone());
            }
            continue;
        }
        for variant in &mut variants {
            let start = Instant::now();
            let compiled = Arc::new(CompiledModel::compile(model.clone(), variant.options)?);
            variant.compile.push(
                json!({"model":name,"compile_ns":ns(start),"selected_backend":compiled.backend(),
                    "shared_across_all_thread_variants":false,"packed_weight_bytes":compiled.packed_weight_bytes()}),
            );
            variant.models.insert(name.into(), compiled);
        }
    }
    let checkpoint = Report {
        protocol,
        metadata: &metadata,
        loads: &loads,
        variants: &variants,
        names: &names,
        minimum,
        backends,
        thread_counts,
        gate,
    };
    let mut cases = Vec::new();
    for name in MODELS {
        let initial = if name == "cls" {
            vec![1, 3, 80, 160]
        } else if name.ends_with("det") {
            vec![1, 3, 32, 32]
        } else {
            vec![1, 3, 48, 32]
        };
        let mut binding = Vec::new();
        let mut runners = Vec::new();
        for variant in &variants {
            let start = Instant::now();
            runners.push(Runner::Graph(Box::new(Session::new(
                variant.models[name].clone(),
                variant.cpu.clone(),
                &initial,
                ExecutionOptions {
                    intra_op_threads: if name == "cls" { 1 } else { variant.threads },
                    ..Default::default()
                },
            )?)));
            binding.push(ns(start));
        }
        for index in 0..3 {
            let mut shape = initial.clone();
            if index == 1 && name != "cls" {
                shape[3] = 96;
            }
            let reshape: Vec<_> = runners
                .iter_mut()
                .map(|runner| runner.reshape(&shape))
                .collect::<Result<_>>()?;
            let input: Vec<_> = (0..shape.iter().product())
                .map(|i| ((i % 97) as f32 - 48.0) / 64.0)
                .collect();
            let id = format!("graph/{name}/{index}");
            let measurements = measure(
                &mut runners,
                &Input::Graph(&input),
                protocol,
                &Progress {
                    path: &progress_path,
                    case_id: &id,
                    names: &names,
                },
                cases.len(),
            )?;
            let values: BTreeMap<_, _> = measurements
                .variants
                .into_iter()
                .enumerate()
                .map(|(i, mut value)| {
                    value["binding_ns"] = json!(binding[i]);
                    value["reshape_ns"] = json!(reshape[i]);
                    (variants[i].name.clone(), value)
                })
                .collect();
            cases.push(json!({"id":id,"input_shape":shape,"variant_order":measurements.order,"iterations_per_batch":measurements.iterations,"variants":values}));
            checkpoint.save(&output, &cases, false)?;
            println!("interleaved graph {name}/{index}");
        }
    }
    for (det, rec, keys) in [
        ("det", "rec", "ppocr_keys.txt"),
        ("small_det", "small_rec", "rec_keys.txt"),
        ("medium_det", "medium_rec", "rec_keys.txt"),
    ] {
        let mut binding = Vec::new();
        let mut runners = Vec::new();
        for variant in &variants {
            let start = Instant::now();
            runners.push(Runner::Ocr(Box::new(make_ocr(
                variant, &models, det, rec, keys,
            )?)));
            binding.push(ns(start));
        }
        for image in &images {
            let id = format!("ocr/{rec}/{}", image.id);
            let measurements = measure(
                &mut runners,
                &Input::Image(image),
                protocol,
                &Progress {
                    path: &progress_path,
                    case_id: &id,
                    names: &names,
                },
                cases.len(),
            )?;
            let values: BTreeMap<_, _> = measurements
                .variants
                .into_iter()
                .enumerate()
                .map(|(i, mut value)| {
                    value["binding_ns"] = json!(binding[i]);
                    (variants[i].name.clone(), value)
                })
                .collect();
            cases.push(json!({"id":id,"input_size":[image.width,image.height],"variant_order":measurements.order,"iterations_per_batch":measurements.iterations,"variants":values}));
            checkpoint.save(&output, &cases, false)?;
            println!("interleaved OCR {rec}/{}", image.id);
        }
    }
    let report = checkpoint.save(&output, &cases, true)?;
    write_json(
        &progress_path,
        &json!({"schema_version":1,"complete":true,"passed":report["passed"],"report":output}),
    )?;
    println!(
        "{}: {} cases; report {}",
        if report["passed"] == true {
            "PASS"
        } else {
            "FAIL"
        },
        report["cases"].as_array().unwrap().len(),
        output.display()
    );
    if gate && report["gate_passed"] != true {
        std::process::exit(1);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adaptive_batches_have_a_common_floor_and_bounded_automatic_growth() {
        assert_eq!(batch_iterations(1, 10_000_000, 1_000_000), 10);
        assert_eq!(batch_iterations(12, 10_000_000, 1_000_000), 12);
        assert_eq!(batch_iterations(1, 10_000_000, 25_000_000_000), 1);
        assert_eq!(batch_iterations(1, 10_000_000, 0), 10_000);
        assert_eq!(batch_iterations(5, 0, 1), 5);
    }

    #[test]
    fn layout_and_fusion_form_one_m2_acceptance_boundary() {
        let names = ["reference", "simd", "nhwc", "auto"].map(str::to_owned);
        assert_eq!(
            gate_pairs(&names, false, false),
            vec![("M1", "reference", "simd"), ("M2", "simd", "auto")]
        );
        let cases = [json!({"id":"ocr/page","variants":{
            "reference":{"median_ns":100},"simd":{"median_ns":90},
            "nhwc":{"median_ns":200},"auto":{"median_ns":60}}})];
        assert_eq!(pair_result(&cases, "simd", "nhwc", 1.0)["passed"], false);
        assert_eq!(pair_result(&cases, "simd", "auto", 1.0)["passed"], true);
    }

    #[test]
    fn a_fast_graph_cannot_hide_an_e2e_regression() {
        let cases = [
            json!({"id":"graph/fast","variants":{"a":{"median_ns":100},"b":{"median_ns":10}}}),
            json!({"id":"ocr/page","variants":{"a":{"median_ns":100},"b":{"median_ns":104}}}),
        ];
        let result = pair_result(&cases, "a", "b", 1.0);
        assert!(result["geometric_mean_speedup"].as_f64().unwrap() > 1.0);
        assert_eq!(result["passed"], false);
        assert_eq!(result["e2e_regressions"][0]["id"], "ocr/page");
    }

    #[test]
    fn thread_scaling_configuration_requires_one_thread_baseline_and_distinct_budgets() {
        let configs = thread_configs("1,2,4,8").unwrap();
        assert_eq!(
            configs
                .iter()
                .map(|config| config.threads)
                .collect::<Vec<_>>(),
            vec![1, 2, 4, 8]
        );
        assert_eq!(
            configs
                .iter()
                .map(|config| config.name.as_str())
                .collect::<Vec<_>>(),
            vec!["auto-c1", "auto-c2", "auto-c4", "auto-c8"]
        );
        assert!(configs.iter().all(
            |config| matches!(config.options.kernels, KernelPolicy::Auto)
                && matches!(config.options.layout, LayoutPolicy::Auto)
                && config.options.fusion
        ));
        for invalid in ["", "1", "2,4", "1,1", "1,0", "1,17", "1,invalid"] {
            assert!(thread_configs(invalid).is_err(), "{invalid}");
        }
    }

    #[test]
    fn every_thread_candidate_is_compared_with_c1_and_retains_its_e2e_regression() {
        let names = ["auto-c1", "auto-c2", "auto-c4", "auto-c8"].map(str::to_owned);
        assert_eq!(
            gate_pairs(&names, false, true),
            vec![
                ("M3", "auto-c1", "auto-c2"),
                ("M3", "auto-c1", "auto-c4"),
                ("M3", "auto-c1", "auto-c8")
            ]
        );
        let cases = [json!({"id":"ocr/page","variants":{
            "auto-c1":{"median_ns":100},"auto-c2":{"median_ns":80},
            "auto-c4":{"median_ns":70},"auto-c8":{"median_ns":104}}})];
        let result = pair_result(&cases, "auto-c1", "auto-c8", 1.0);
        assert_eq!(result["passed"], false);
        assert_eq!(result["e2e_regressions"][0]["id"], "ocr/page");
        assert_eq!(result["cases"][0]["baseline_ns"].as_f64(), Some(100.0));
    }
}
