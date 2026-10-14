//! Graph-only CLS timing. Allocator instrumentation lives in this executable;
//! the inference library continues to forbid unsafe code.
use serde::Serialize;
use simd_paddle_ocr_rs::{
    cls::load_cls_inputs,
    model::Model,
    runtime::{CompiledModel, ExecutionLimits, Session},
};
use std::{env, hint::black_box, path::PathBuf, process::ExitCode, sync::Arc, time::Instant};

#[cfg(feature = "heap-profiler")]
use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::atomic::{AtomicUsize, Ordering::Relaxed},
};

#[cfg(feature = "heap-profiler")]
struct CountingSystem;
#[cfg(feature = "heap-profiler")]
static LIVE: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "heap-profiler")]
static PEAK: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "heap-profiler")]
static CALLS: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "heap-profiler")]
static REQUESTED: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "heap-profiler")]
fn allocated(size: usize) {
    CALLS.fetch_add(1, Relaxed);
    REQUESTED.fetch_add(size, Relaxed);
    let live = LIVE.fetch_add(size, Relaxed) + size;
    PEAK.fetch_max(live, Relaxed);
}
// Counters do not allocate. System receives exactly the allocation, alignment,
// pointer and ownership arguments provided by each GlobalAlloc caller.
#[cfg(feature = "heap-profiler")]
unsafe impl GlobalAlloc for CountingSystem {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forward the caller's valid allocation request to System.
        let result = unsafe { System.alloc(layout) };
        if !result.is_null() {
            allocated(layout.size());
        }
        result
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forward the caller's valid zeroed-allocation request to System.
        let result = unsafe { System.alloc_zeroed(layout) };
        if !result.is_null() {
            allocated(layout.size());
        }
        result
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: pointer/layout came from this allocator and are forwarded intact.
        unsafe { System.dealloc(pointer, layout) };
        LIVE.fetch_sub(layout.size(), Relaxed);
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        // SAFETY: forward the caller's valid original allocation and requested size.
        let result = unsafe { System.realloc(pointer, layout, size) };
        if !result.is_null() {
            LIVE.fetch_sub(layout.size(), Relaxed);
            allocated(size);
        }
        result
    }
}
#[cfg(feature = "heap-profiler")]
#[global_allocator]
static ALLOCATOR: CountingSystem = CountingSystem;

#[derive(Serialize)]
struct InputReport {
    case_id: String,
    artifact_id: String,
    shape: Vec<usize>,
    sha256: String,
    first_call_ms: f64,
    samples_ms: Vec<f64>,
    p50_ms: f64,
    p95_ms: f64,
    mean_ms: f64,
    inputs_per_second: f64,
}
#[derive(Serialize)]
struct CorpusReport {
    inputs: usize,
    samples_ms: Vec<f64>,
    sweep_wall_samples_ms: Vec<f64>,
    p50_ms: f64,
    p95_ms: f64,
    mean_ms: f64,
    inputs_per_second: f64,
    wall_inputs_per_second: f64,
}
#[derive(Serialize)]
struct Report {
    schema_version: u32,
    runtime: &'static str,
    profile: &'static str,
    reference: PathBuf,
    model_sha256: String,
    manifest_sha256: String,
    warmup: usize,
    samples: usize,
    load_ms: f64,
    prepare_ms: f64,
    first_sweep_ms: f64,
    distinct_shapes: usize,
    activation_bytes: usize,
    naive_activation_bytes: usize,
    instrumentation: &'static str,
    warm_allocation_count: Option<usize>,
    warm_allocation_bytes: Option<usize>,
    warm_retained_heap_bytes: Option<usize>,
    warm_peak_heap_growth_bytes: Option<usize>,
    measurement: &'static str,
    checksum: f64,
    inputs: Vec<InputReport>,
    corpus: CorpusReport,
}

fn milliseconds(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}
fn mean(samples: &[f64]) -> f64 {
    samples.iter().sum::<f64>() / samples.len() as f64
}
fn percentile(samples: &[f64], fraction: f64) -> f64 {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let position = (sorted.len() - 1) as f64 * fraction;
    let left = position.floor() as usize;
    let right = position.ceil() as usize;
    sorted[left] + (sorted[right] - sorted[left]) * (position - left as f64)
}
fn run() -> Result<(), String> {
    let mut reference = None;
    let mut warmup = 30usize;
    let mut samples = 100usize;
    let mut seen = std::collections::HashSet::new();
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        if !seen.insert(flag.clone()) {
            return Err(format!("duplicate argument {flag}"));
        }
        let value = args.next().ok_or_else(|| format!("{flag} needs a value"))?;
        match flag.as_str() {
            "--reference" => reference = Some(PathBuf::from(value)),
            "--warmup" | "--samples" => {
                let count = value.parse().map_err(|_| format!("invalid {flag} count"))?;
                if flag == "--warmup" {
                    warmup = count;
                } else {
                    samples = count;
                }
            }
            _ => return Err(format!("unknown flag {flag}")),
        }
    }
    let reference = reference.ok_or_else(|| {
        "usage: cls-benchmark --reference <canonical-bundle> [--warmup 30] [--samples 100]"
            .to_string()
    })?;
    if samples == 0 {
        return Err("--samples must be positive".into());
    }
    let corpus = load_cls_inputs(&reference).map_err(|error| error.to_string())?;
    let started = Instant::now();
    let model = Arc::new(Model::load_path(&corpus.model_path).map_err(|error| error.to_string())?);
    let load_ms = milliseconds(started);
    let started = Instant::now();
    let compiled = Arc::new(CompiledModel::new(model).map_err(|error| error.to_string())?);
    let mut sessions = Vec::new();
    let mut shapes: Vec<Vec<usize>> = Vec::new();
    let mut session_indexes = Vec::with_capacity(corpus.inputs.len());
    for input in &corpus.inputs {
        let index = match shapes.iter().position(|shape| shape == &input.shape) {
            Some(index) => index,
            None => {
                let index = sessions.len();
                sessions.push(
                    Session::new(compiled.clone(), &input.shape, ExecutionLimits::default())
                        .map_err(|error| error.to_string())?,
                );
                shapes.push(input.shape.clone());
                index
            }
        };
        session_indexes.push(index);
    }
    let prepare_ms = milliseconds(started);
    let activation_bytes = sessions.iter().map(Session::activation_bytes).sum();
    let naive_activation_bytes = sessions.iter().map(Session::naive_activation_bytes).sum();
    let mut checksum = 0f64;
    let mut first_calls = vec![0.; corpus.inputs.len()];
    let first_sweep = Instant::now();
    for (index, input) in corpus.inputs.iter().enumerate() {
        let started = Instant::now();
        let prediction = sessions[session_indexes[index]]
            .run(&input.values)
            .map_err(|error| error.to_string())?;
        first_calls[index] = milliseconds(started);
        checksum += f64::from(black_box(prediction.data())[0]);
    }
    let first_sweep_ms = milliseconds(first_sweep);
    for _ in 0..warmup {
        for (index, input) in corpus.inputs.iter().enumerate() {
            let prediction = sessions[session_indexes[index]]
                .run(&input.values)
                .map_err(|error| error.to_string())?;
            checksum += f64::from(black_box(prediction.data())[0]);
        }
    }
    let mut times = vec![vec![0.; samples]; corpus.inputs.len()];
    let mut sweep_times = vec![0.; samples];
    let mut wall_times = vec![0.; samples];
    // All bookkeeping and inputs are allocated before this counter baseline.
    #[cfg(feature = "heap-profiler")]
    let allocation_baseline = {
        let live = LIVE.load(Relaxed);
        PEAK.store(live, Relaxed);
        CALLS.store(0, Relaxed);
        REQUESTED.store(0, Relaxed);
        live
    };
    for sweep in 0..samples {
        let wall_started = Instant::now();
        let mut graph_ms = 0.;
        for (index, input) in corpus.inputs.iter().enumerate() {
            let started = Instant::now();
            let prediction = sessions[session_indexes[index]]
                .run(&input.values)
                .map_err(|error| error.to_string())?;
            let elapsed = milliseconds(started);
            times[index][sweep] = elapsed;
            graph_ms += elapsed;
            checksum += f64::from(black_box(prediction.data())[0]);
        }
        wall_times[sweep] = milliseconds(wall_started);
        sweep_times[sweep] = graph_ms;
    }
    #[cfg(feature = "heap-profiler")]
    let (
        warm_allocation_count,
        warm_allocation_bytes,
        warm_retained_heap_bytes,
        warm_peak_heap_growth_bytes,
    ) = (
        Some(CALLS.load(Relaxed)),
        Some(REQUESTED.load(Relaxed)),
        Some(LIVE.load(Relaxed).saturating_sub(allocation_baseline)),
        Some(PEAK.load(Relaxed).saturating_sub(allocation_baseline)),
    );
    #[cfg(not(feature = "heap-profiler"))]
    let (
        warm_allocation_count,
        warm_allocation_bytes,
        warm_retained_heap_bytes,
        warm_peak_heap_growth_bytes,
    ) = (None, None, None, None);
    #[cfg(feature = "heap-profiler")]
    if warm_allocation_count != Some(0) {
        return Err(format!(
            "steady CLS inference allocated: {} calls, {} requested bytes",
            warm_allocation_count.unwrap(),
            warm_allocation_bytes.unwrap()
        ));
    }
    let inputs = corpus
        .inputs
        .into_iter()
        .zip(times)
        .enumerate()
        .map(|(index, (input, times))| {
            let mean_ms = mean(&times);
            InputReport {
                case_id: input.case_id,
                artifact_id: input.artifact_id,
                shape: input.shape,
                sha256: input.sha256,
                first_call_ms: first_calls[index],
                p50_ms: percentile(&times, 0.5),
                p95_ms: percentile(&times, 0.95),
                mean_ms,
                inputs_per_second: 1000. / mean_ms,
                samples_ms: times,
            }
        })
        .collect::<Vec<_>>();
    let mean_ms = mean(&sweep_times);
    let corpus_report = CorpusReport {
        inputs: inputs.len(),
        p50_ms: percentile(&sweep_times, 0.5),
        p95_ms: percentile(&sweep_times, 0.95),
        mean_ms,
        inputs_per_second: 1000. * inputs.len() as f64 / mean_ms,
        wall_inputs_per_second: 1000. * inputs.len() as f64 / mean(&wall_times),
        samples_ms: sweep_times,
        sweep_wall_samples_ms: wall_times,
    };
    #[cfg(feature = "heap-profiler")]
    let instrumentation = "System allocator forwarding counts successful alloc, alloc_zeroed and realloc calls and requested bytes. Atomic bookkeeping is included; latency must come from the separate uninstrumented executable.";
    #[cfg(not(feature = "heap-profiler"))]
    let instrumentation = "Standard allocator; no allocation instrumentation. Use the separate heap-profiler build to verify zero steady allocations.";
    let report = Report {
        schema_version: 1,
        runtime: "rust",
        profile: "canonical",
        reference,
        model_sha256: corpus.model_sha256,
        manifest_sha256: corpus.manifest_sha256,
        warmup,
        samples,
        load_ms,
        prepare_ms,
        first_sweep_ms,
        distinct_shapes: sessions.len(),
        activation_bytes,
        naive_activation_bytes,
        instrumentation,
        warm_allocation_count,
        warm_allocation_bytes,
        warm_retained_heap_bytes,
        warm_peak_heap_growth_bytes,
        measurement: "Complete manifest-order corpus sweeps. Graph call includes session input copy/finite validation and borrowed output; tracing, hashing, file I/O and output copying are excluded. Corpus samples sum individual graph times; sweep wall samples include timestamp/bookkeeping overhead. First calls share prepared sessions; only the first call is process-cold.",
        checksum,
        inputs,
        corpus: corpus_report,
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&report).map_err(|error| error.to_string())?
    );
    Ok(())
}
fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("CLS benchmark failed: {error}");
            ExitCode::from(2)
        }
    }
}
