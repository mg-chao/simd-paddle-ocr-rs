//! Isolated allocation instrumentation. Unsafe code is confined to forwarding
//! the standard System allocator; the imported library forbids unsafe code.
use serde::Serialize;
use simd_paddle_ocr_rs::model::{LoadOptions, Model};
use std::{env, path::PathBuf, process::ExitCode, time::Instant};

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
static VOLUME: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "heap-profiler")]
fn added(bytes: usize) {
    let live = LIVE.fetch_add(bytes, Relaxed) + bytes;
    PEAK.fetch_max(live, Relaxed);
    VOLUME.fetch_add(bytes, Relaxed);
}
// All calls preserve exactly System's allocation, alignment and ownership
// requirements. Atomic counters never allocate or invoke the allocator.
#[cfg(feature = "heap-profiler")]
unsafe impl GlobalAlloc for CountingSystem {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: GlobalAlloc forwards the caller's valid layout to System.
        let result = unsafe { System.alloc(layout) };
        if !result.is_null() {
            added(layout.size());
        }
        result
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: GlobalAlloc forwards the caller's valid layout to System.
        let result = unsafe { System.alloc_zeroed(layout) };
        if !result.is_null() {
            added(layout.size());
        }
        result
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: pointer and layout are those supplied by the allocator caller.
        unsafe { System.dealloc(pointer, layout) };
        LIVE.fetch_sub(layout.size(), Relaxed);
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: the original pointer/layout and requested size are forwarded.
        let result = unsafe { System.realloc(pointer, layout, new_size) };
        if !result.is_null() {
            if new_size >= layout.size() {
                added(new_size - layout.size());
            } else {
                LIVE.fetch_sub(layout.size() - new_size, Relaxed);
            }
        }
        result
    }
}
#[cfg(feature = "heap-profiler")]
#[global_allocator]
static ALLOCATOR: CountingSystem = CountingSystem;

#[derive(Serialize)]
struct Sample {
    elapsed_ms: f64,
    parse_ms: f64,
    normalize_ms: f64,
    retained_heap_bytes: Option<usize>,
    peak_heap_bytes: Option<usize>,
    allocation_growth_bytes: Option<usize>,
    source_bytes: u64,
    normalized_weight_bytes: u64,
}
#[derive(Serialize)]
struct Report {
    schema_version: u32,
    paths: Vec<PathBuf>,
    warmup: usize,
    repeat: usize,
    instrumentation: &'static str,
    file_cache: &'static str,
    median_ms: f64,
    p95_ms: f64,
    samples: Vec<Sample>,
}
fn load(paths: &[PathBuf]) -> Result<Sample, String> {
    let mut models = Vec::with_capacity(paths.len());
    #[cfg(feature = "heap-profiler")]
    let baseline = {
        let baseline = LIVE.load(Relaxed);
        PEAK.store(baseline, Relaxed);
        VOLUME.store(0, Relaxed);
        baseline
    };
    let options = LoadOptions {
        measure_timings: true,
        ..Default::default()
    };
    let started = Instant::now();
    for path in paths {
        models.push(
            Model::load_path_with_options(path, &options)
                .map_err(|error| format!("{}: {error}", path.display()))?,
        );
    }
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    #[cfg(feature = "heap-profiler")]
    let (retained_heap_bytes, peak_heap_bytes, allocation_growth_bytes) = (
        Some(LIVE.load(Relaxed).saturating_sub(baseline)),
        Some(PEAK.load(Relaxed).saturating_sub(baseline)),
        Some(VOLUME.load(Relaxed)),
    );
    #[cfg(not(feature = "heap-profiler"))]
    let (retained_heap_bytes, peak_heap_bytes, allocation_growth_bytes) = (None, None, None);
    let sample = Sample {
        elapsed_ms,
        parse_ms: models
            .iter()
            .filter_map(Model::import_timings)
            .map(|t| t.parse_ms)
            .sum(),
        normalize_ms: models
            .iter()
            .filter_map(Model::import_timings)
            .map(|t| t.normalize_ms)
            .sum(),
        retained_heap_bytes,
        peak_heap_bytes,
        allocation_growth_bytes,
        source_bytes: models.iter().map(|m| m.source_bytes()).sum(),
        normalized_weight_bytes: models.iter().map(Model::normalized_weight_bytes).sum(),
    };
    drop(models);
    Ok(sample)
}
fn run() -> Result<(), String> {
    let mut paths = Vec::new();
    let mut warmup = 3;
    let mut repeat = 30;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--warmup" | "--repeat" => {
                let count: usize = args
                    .next()
                    .ok_or_else(|| format!("{arg} needs a count"))?
                    .parse()
                    .map_err(|_| format!("invalid {arg} count"))?;
                if arg == "--warmup" {
                    warmup = count;
                } else {
                    repeat = count;
                }
            }
            flag if flag.starts_with('-') => return Err(format!("unknown flag {flag}")),
            path => paths.push(PathBuf::from(path)),
        }
    }
    if paths.is_empty() || repeat == 0 {
        return Err(
            "usage: import-benchmark <one-or-more-model-paths> [--warmup 3] [--repeat 30]".into(),
        );
    }
    for _ in 0..warmup {
        load(&paths)?;
    }
    let mut samples = Vec::with_capacity(repeat);
    for _ in 0..repeat {
        samples.push(load(&paths)?);
    }
    let mut times: Vec<f64> = samples.iter().map(|sample| sample.elapsed_ms).collect();
    times.sort_by(f64::total_cmp);
    let percentile = |fraction: f64| {
        let index = (times.len() - 1) as f64 * fraction;
        let left = index.floor() as usize;
        let right = index.ceil() as usize;
        times[left] + (times[right] - times[left]) * (index - left as f64)
    };
    let median_ms = percentile(0.5);
    let p95_ms = percentile(0.95);
    #[cfg(feature = "heap-profiler")]
    let instrumentation = "System forwarding records requested live/peak heap bytes; atomic bookkeeping included; use default build for latency";
    #[cfg(not(feature = "heap-profiler"))]
    let instrumentation = "standard allocator; heap counters absent; latency measurements include no allocation instrumentation";
    let report = Report {
        schema_version: 1,
        paths,
        warmup,
        repeat,
        instrumentation,
        file_cache: "warm filesystem; one fresh process per model set",
        median_ms,
        p95_ms,
        samples,
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
            eprintln!("{error}");
            ExitCode::from(2)
        }
    }
}
