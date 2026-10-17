//! Bounded, separate-process CPU benchmark contract.
#![allow(dead_code)]
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::Path};
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
pub const FIXTURES: [&str; 3] = ["imgs_1", "imgs_00018069", "imgs_00006737"];
pub mod platform;
/// Explicitly unpinned scheduling, used on macOS instead of an invented CPU mask.
pub const UNBOUND_AFFINITY: &str = "unbound";
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrderStrategy {
    Balanced,
    RustFirst,
    CsharpFirst,
}
impl OrderStrategy {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "balanced" => Ok(Self::Balanced),
            "rust-first" => Ok(Self::RustFirst),
            "csharp-first" => Ok(Self::CsharpFirst),
            _ => Err("--order must be balanced, rust-first or csharp-first".into()),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Balanced => "balanced",
            Self::RustFirst => "rust-first",
            Self::CsharpFirst => "csharp-first",
        }
    }
    pub fn languages(self, job: &Job) -> Result<[&'static str; 2]> {
        job.validate()?;
        // Use the canonical group rather than the selected case's index. Both
        // workloads receive the same order, and filtering cannot reverse it.
        let model = ["tiny", "small", "medium"]
            .iter()
            .position(|&model| model == job.model)
            .ok_or("unknown model in execution order")?;
        let threads = usize::from(job.threads == 8);
        let fixture = FIXTURES
            .iter()
            .position(|&fixture| fixture == job.fixture)
            .ok_or("unknown fixture in execution order")?;
        let parity = (model + threads + fixture) % 2;
        let rust_first = match self {
            Self::Balanced => parity == 0,
            Self::RustFirst => true,
            Self::CsharpFirst => false,
        };
        Ok(if rust_first {
            ["rust", "csharp"]
        } else {
            ["csharp", "rust"]
        })
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Job {
    pub manifest: String,
    pub models: String,
    pub model: String,
    pub fixture: String,
    pub pixels: String,
    pub pixel_sha256: String,
    pub width: usize,
    pub height: usize,
    pub threads: usize,
    pub concurrency: usize,
    pub runs: usize,
    pub affinity: String,
}
impl Job {
    pub fn validate(&self) -> Result<()> {
        if !["tiny", "small", "medium"].contains(&self.model.as_str())
            || !FIXTURES.contains(&self.fixture.as_str())
            || ![1, 8].contains(&self.threads)
            || ![1, 2].contains(&self.concurrency)
            || !(1..=3).contains(&self.runs)
            || self.width == 0
            || self.height == 0
            || self
                .width
                .checked_mul(self.height)
                .and_then(|v| v.checked_mul(3))
                .is_none()
        {
            return Err("invalid benchmark job".into());
        }
        if self.affinity != UNBOUND_AFFINITY {
            let mask = u64::from_str_radix(self.affinity.trim_start_matches("0x"), 16)?;
            if mask.count_ones() != self.threads as u32 {
                return Err("affinity CPU count must equal thread budget".into());
            }
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Measurement {
    pub language: String,
    pub load_compile_ms: f64,
    pub first_inference_ms: f64,
    pub warmup_ms: f64,
    pub samples_ms: Vec<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub process_cpu: Option<ProcessCpuTiming>,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub peak_private_bytes: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peak_resident_bytes: Option<u64>,
    pub diagnostics: BTreeMap<String, u64>,
    pub result: super::baseline::CaseResult,
}
/// Process-wide kernel plus user time, summed over all threads in each batch.
/// Zero is valid because the operating-system counter can have coarse ticks.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProcessCpuTiming {
    pub first_inference_ms: f64,
    pub warmup_ms: f64,
    pub samples_ms: Vec<f64>,
}
impl Measurement {
    /// Keep legacy Windows commitment and macOS resident high-water marks distinct.
    pub fn peak_memory(&self) -> Result<(&'static str, u64)> {
        match (self.peak_private_bytes, self.peak_resident_bytes) {
            (private, None) if private > 0 => Ok(("peak_private_commit", private)),
            (0, Some(resident)) if resident > 0 => Ok(("peak_resident", resident)),
            _ => Err("exactly one positive process memory metric is required".into()),
        }
    }
    pub fn validate(&self, job: &Job) -> Result<()> {
        let (metric, _) = self.peak_memory()?;
        if (job.affinity == UNBOUND_AFFINITY) != (metric == "peak_resident") {
            return Err("process memory metric does not match job scheduling platform".into());
        }
        if self.samples_ms.len() != job.runs
            || self
                .samples_ms
                .iter()
                .chain([
                    &self.load_compile_ms,
                    &self.first_inference_ms,
                    &self.warmup_ms,
                ])
                .any(|v| !v.is_finite() || *v <= 0.)
            || self.result.id != job.fixture
        {
            return Err("incomplete or invalid benchmark measurement".into());
        }
        if let Some(cpu) = &self.process_cpu
            && (cpu.samples_ms.len() != job.runs
                || cpu
                    .samples_ms
                    .iter()
                    .chain([&cpu.first_inference_ms, &cpu.warmup_ms])
                    .any(|v| !v.is_finite() || *v < 0.))
        {
            return Err("incomplete or invalid process CPU measurement".into());
        }
        self.result.validate()?;
        Ok(())
    }
    pub fn median_ms(&self) -> f64 {
        median(&self.samples_ms)
    }
}
pub fn median(values: &[f64]) -> f64 {
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    let n = values.len();
    if n == 0 {
        return f64::NAN;
    }
    if n.is_multiple_of(2) {
        (values[n / 2 - 1] + values[n / 2]) / 2.
    } else {
        values[n / 2]
    }
}
#[derive(Debug, Serialize, Deserialize)]
pub struct Comparison {
    pub job: Job,
    #[serde(default)]
    pub planned_execution_order: Vec<String>,
    /// Successfully spawned workers, including workers that later fail.
    #[serde(default)]
    pub execution_order: Vec<String>,
    pub rust: Option<Measurement>,
    pub csharp: Option<Measurement>,
    pub latency_ratio: Option<f64>,
    pub throughput_ratio: Option<f64>,
    pub memory_ratio: Option<f64>,
    pub status: String,
    pub differences: Vec<String>,
    pub error: Option<String>,
}
impl Comparison {
    pub fn new(job: Job) -> Self {
        Self {
            job,
            planned_execution_order: vec![],
            execution_order: vec![],
            rust: None,
            csharp: None,
            latency_ratio: None,
            throughput_ratio: None,
            memory_ratio: None,
            status: "inconclusive".into(),
            differences: vec![],
            error: None,
        }
    }
    pub fn evaluate(&mut self) -> Result<()> {
        let (Some(rust), Some(csharp)) = (&self.rust, &self.csharp) else {
            return Ok(());
        };
        rust.validate(&self.job)?;
        csharp.validate(&self.job)?;
        self.differences = super::baseline::compare(&self.job.model, &csharp.result, &rust.result);
        let latency = rust.median_ms() / csharp.median_ms();
        let throughput = csharp.median_ms() / rust.median_ms();
        let (rust_metric, rust_memory) = rust.peak_memory()?;
        let (csharp_metric, csharp_memory) = csharp.peak_memory()?;
        if rust_metric != csharp_metric {
            return Err("cannot compare different process memory metrics".into());
        }
        let memory = rust_memory as f64 / csharp_memory as f64;
        self.latency_ratio = Some(latency);
        self.throughput_ratio = Some(throughput);
        self.memory_ratio = Some(memory);
        let min = |m: &Measurement| m.samples_ms.iter().copied().fold(f64::INFINITY, f64::min);
        let max = |m: &Measurement| m.samples_ms.iter().copied().fold(0., f64::max);
        // Observed extrema bound the ratio across these small fixed samples.
        // Crossing a gate is unresolved evidence, never an excuse for retries.
        let (clear_pass, clear_fail) = if self.job.concurrency == 1 {
            (
                max(rust) / min(csharp) <= 1.1,
                min(rust) / max(csharp) > 1.1,
            )
        } else {
            (
                min(csharp) / max(rust) >= 0.9,
                max(csharp) / min(rust) < 0.9,
            )
        };
        self.status = if !self.differences.is_empty() || clear_fail || memory > 1.1 {
            "failed"
        } else if self.job.runs < 3 || !clear_pass {
            "inconclusive"
        } else {
            "passed"
        }
        .into();
        Ok(())
    }
}
#[derive(Debug, Serialize, Deserialize)]
pub struct Report {
    pub schema_version: u32,
    pub status: String,
    pub provenance: BTreeMap<String, String>,
    pub budget_seconds_per_language: u64,
    pub elapsed_seconds_per_language: BTreeMap<String, f64>,
    pub cases: Vec<Comparison>,
}
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Coverage {
    pub scope: String,
    pub required_case_count: usize,
    pub selected_unique_case_count: usize,
    pub duplicate_case_count: usize,
    pub full_qualification_passed: bool,
}
pub fn coverage(cases: &[Comparison]) -> Coverage {
    let keys = cases
        .iter()
        .map(|case| {
            (
                case.job.model.as_str(),
                case.job.fixture.as_str(),
                case.job.threads,
                case.job.concurrency,
            )
        })
        .collect::<std::collections::BTreeSet<_>>();
    let mut required = std::collections::BTreeSet::new();
    for model in ["tiny", "small", "medium"] {
        for fixture in FIXTURES {
            for threads in [1, 8] {
                for concurrency in [1, 2] {
                    required.insert((model, fixture, threads, concurrency));
                }
            }
        }
    }
    let unbound = cases
        .first()
        .is_some_and(|c| c.job.affinity == UNBOUND_AFFINITY);
    let full = keys == required
        && keys.len() == cases.len()
        && cases
            .iter()
            .all(|c| (c.job.affinity == UNBOUND_AFFINITY) == unbound);
    Coverage {
        scope: if full {
            if unbound {
                "full macOS matrix"
            } else {
                "full Windows x64 matrix"
            }
        } else {
            "focused comparison"
        }
        .into(),
        required_case_count: required.len(),
        selected_unique_case_count: keys.len(),
        duplicate_case_count: cases.len() - keys.len(),
        full_qualification_passed: full
            && cases
                .iter()
                .all(|case| case.status == "passed" && case.job.runs == 3),
    }
}
pub fn write_report(path: &Path, report: &Report) -> Result<()> {
    let coverage = coverage(&report.cases);
    let mut json = serde_json::to_value(report)?;
    json.as_object_mut()
        .unwrap()
        .insert("coverage".into(), serde_json::to_value(&coverage)?);
    super::baseline::write_json(&path.join("report.json"), &json)?;
    let mut md = format!(
        "# {} CPU comparison\n\nStatus: **{}**. One first inference, one warm-up, and 1–3 fixed measured batches per case. Throughput batches contain two concurrent owned results. Build time excluded; worker startup, model load, first inference and warm-up count toward the per-language budget.\n\n| Model | Fixture | Threads | Requests | Rust wall ms | C# wall ms | Rust CPU ms | C# CPU ms | Worker order | Rust/C# memory | Status |\n|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|\n",
        report
            .provenance
            .get("platform")
            .map(String::as_str)
            .unwrap_or("Windows x64"),
        report.status
    );
    for c in &report.cases {
        let metric = |m: &Option<Measurement>| {
            m.as_ref()
                .map(|m| format!("{:.3}", m.median_ms()))
                .unwrap_or_else(|| "—".into())
        };
        let cpu_metric = |m: &Option<Measurement>| {
            m.as_ref()
                .and_then(|m| m.process_cpu.as_ref())
                .map(|cpu| format!("{:.3}", median(&cpu.samples_ms)))
                .unwrap_or_else(|| "—".into())
        };
        md.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            c.job.model,
            c.job.fixture,
            c.job.threads,
            c.job.concurrency,
            metric(&c.rust),
            metric(&c.csharp),
            cpu_metric(&c.rust),
            cpu_metric(&c.csharp),
            if c.execution_order.is_empty() {
                "—".into()
            } else {
                c.execution_order.join(" → ")
            },
            c.memory_ratio
                .map(|v| format!("{v:.3}"))
                .unwrap_or_else(|| "—".into()),
            c.status
        ));
    }
    md.push_str(&format!(
        "\nScope: **{}**, {} of {} unique required cases. Full qualification passed: **{}**.\n",
        coverage.scope,
        coverage.selected_unique_case_count,
        coverage.required_case_count,
        coverage.full_qualification_passed
    ));
    let memory_description = if report
        .cases
        .first()
        .is_some_and(|c| c.job.affinity == UNBOUND_AFFINITY)
    {
        "macOS process peak resident bytes (`getrusage(RUSAGE_SELF).ru_maxrss`), including shared resident pages. This is not Windows private commit. Both workers use unpinned macOS scheduling with matched OCR worker budgets; no CPU affinity or performance-core placement is guaranteed"
    } else {
        "Windows process peak private commit (`GetProcessMemoryInfo.PeakPagefileUsage`)"
    };
    md.push_str(&format!("\nMemory gate: {memory_description}. Capture occurs before diagnostic inspection and includes runtime, loading, requests and retained allocations. Diagnostic counters are partial capacity/accounting data, not the memory gate. See report.json and per-worker logs for raw samples, output agreement, errors, affinity and provenance. One or two samples cannot establish a final pass; no automatic retries or sampling growth occur.\n"));
    md.push_str(&format!(
        "\nExecution strategy: **{}**. Planned order and successfully spawned workers are recorded per case. CPU columns are medians of process-wide kernel plus user milliseconds for the same batches; they can exceed wall time when multiple threads run. CPU time is diagnostic and does not change any gate. Missing CPU/order fields in older reports remain unavailable. One warm-up is a bounded warm measurement, not evidence of steady state.\n",
        report.provenance.get("execution_order_strategy").map(String::as_str).unwrap_or("unrecorded")
    ));
    std::fs::write(path.join("report.md"), md)?;
    Ok(())
}
fn is_zero(value: &u64) -> bool {
    *value == 0
}
#[cfg(windows)]
pub mod windows {
    use super::Result;
    use std::time::Duration;
    #[repr(C)]
    #[derive(Default)]
    struct FileTime {
        low: u32,
        high: u32,
    }
    impl FileTime {
        fn ticks(&self) -> u64 {
            u64::from(self.low) | (u64::from(self.high) << 32)
        }
    }
    #[repr(C)]
    #[derive(Default)]
    struct MemoryCounters {
        cb: u32,
        faults: u32,
        peak_ws: usize,
        ws: usize,
        peak_paged: usize,
        paged: usize,
        peak_nonpaged: usize,
        nonpaged: usize,
        pagefile: usize,
        peak_pagefile: usize,
        private: usize,
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> *mut std::ffi::c_void;
        fn GetProcessAffinityMask(
            process: *mut std::ffi::c_void,
            mask: *mut usize,
            system: *mut usize,
        ) -> i32;
        fn SetProcessAffinityMask(process: *mut std::ffi::c_void, mask: usize) -> i32;
        fn GetProcessTimes(
            process: *mut std::ffi::c_void,
            created: *mut FileTime,
            exited: *mut FileTime,
            kernel: *mut FileTime,
            user: *mut FileTime,
        ) -> i32;
    }
    #[link(name = "psapi")]
    unsafe extern "system" {
        fn GetProcessMemoryInfo(
            process: *mut std::ffi::c_void,
            counters: *mut MemoryCounters,
            size: u32,
        ) -> i32;
    }
    pub fn allowed_affinity() -> Result<u64> {
        let (mut mask, mut system) = (0, 0);
        if unsafe { GetProcessAffinityMask(GetCurrentProcess(), &mut mask, &mut system) } == 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        Ok(mask as u64)
    }
    pub fn set_affinity(mask: u64) -> Result<()> {
        if mask == 0 || unsafe { SetProcessAffinityMask(GetCurrentProcess(), mask as usize) } == 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        Ok(())
    }
    pub fn peak_private() -> Result<u64> {
        let mut counters = MemoryCounters::default();
        counters.cb = std::mem::size_of::<MemoryCounters>() as u32;
        if unsafe {
            GetProcessMemoryInfo(
                GetCurrentProcess(),
                &mut counters,
                std::mem::size_of::<MemoryCounters>() as u32,
            )
        } == 0
        {
            return Err(std::io::Error::last_os_error().into());
        }
        Ok(counters.peak_pagefile as u64)
    }
    pub fn process_cpu_time() -> Result<Duration> {
        let (mut created, mut exited, mut kernel, mut user) = (
            FileTime::default(),
            FileTime::default(),
            FileTime::default(),
            FileTime::default(),
        );
        if unsafe {
            GetProcessTimes(
                GetCurrentProcess(),
                &mut created,
                &mut exited,
                &mut kernel,
                &mut user,
            )
        } == 0
        {
            return Err(std::io::Error::last_os_error().into());
        }
        let ticks = kernel
            .ticks()
            .checked_add(user.ticks())
            .ok_or("process CPU time overflow")?;
        Ok(Duration::new(
            ticks / 10_000_000,
            ((ticks % 10_000_000) * 100) as u32,
        ))
    }
}
pub fn select_affinity(available: u64, count: usize) -> Result<u64> {
    let mut mask = 0u64;
    for bit in 0..64 {
        if available & (1 << bit) != 0 && mask.count_ones() < count as u32 {
            mask |= 1 << bit;
        }
    }
    if mask.count_ones() != count as u32 {
        return Err(
            format!("need {count} allowed logical CPUs in the current processor group").into(),
        );
    }
    Ok(mask)
}
