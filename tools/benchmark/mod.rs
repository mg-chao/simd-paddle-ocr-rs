//! Bounded, separate-process Windows CPU benchmark contract.
#![allow(dead_code)]
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::Path};
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
pub const FIXTURES: [&str; 3] = ["imgs_1", "imgs_00018069", "imgs_00006737"];
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
        let mask = u64::from_str_radix(self.affinity.trim_start_matches("0x"), 16)?;
        if mask.count_ones() != self.threads as u32 {
            return Err("affinity CPU count must equal thread budget".into());
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
    pub peak_private_bytes: u64,
    pub diagnostics: BTreeMap<String, u64>,
    pub result: super::baseline::CaseResult,
}
impl Measurement {
    pub fn validate(&self, job: &Job) -> Result<()> {
        if self.samples_ms.len() != job.runs
            || self.peak_private_bytes == 0
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
    if n % 2 == 0 {
        (values[n / 2 - 1] + values[n / 2]) / 2.
    } else {
        values[n / 2]
    }
}
#[derive(Debug, Serialize, Deserialize)]
pub struct Comparison {
    pub job: Job,
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
        let memory = rust.peak_private_bytes as f64 / csharp.peak_private_bytes as f64;
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
    let full = keys == required && keys.len() == cases.len();
    Coverage {
        scope: if full {
            "full Windows x64 matrix"
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
        "# Windows x64 CPU comparison\n\nStatus: **{}**. One first inference, one warm-up, and 1–3 fixed measured batches per case. Throughput batches contain two concurrent owned results. Build time excluded; worker startup, model load, first inference and warm-up count toward the per-language budget.\n\n| Model | Fixture | Threads | Requests | Rust median ms | C# median ms | Rust/C# memory | Status |\n|---|---|---:|---:|---:|---:|---:|---|\n",
        report.status
    );
    for c in &report.cases {
        let metric = |m: &Option<Measurement>| {
            m.as_ref()
                .map(|m| format!("{:.3}", m.median_ms()))
                .unwrap_or_else(|| "—".into())
        };
        md.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} |\n",
            c.job.model,
            c.job.fixture,
            c.job.threads,
            c.job.concurrency,
            metric(&c.rust),
            metric(&c.csharp),
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
    md.push_str("\nPeak private memory is Windows process peak commit (`GetProcessMemoryInfo.PeakPagefileUsage`), captured before diagnostic inspection. It includes runtime, loading, requests and retained allocations. Diagnostic counters are partial capacity/accounting data, not the memory gate. See report.json and per-worker logs for raw samples, output agreement, errors, affinity and provenance. One or two samples cannot establish a final pass; no automatic retries or sampling growth occur.\n");
    std::fs::write(path.join("report.md"), md)?;
    Ok(())
}
#[cfg(windows)]
pub mod windows {
    use super::Result;
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
