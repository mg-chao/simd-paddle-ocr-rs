#[path = "../tools/baseline/mod.rs"]
mod baseline;
#[path = "../tools/benchmark/mod.rs"]
mod benchmark;
use benchmark::*;
use std::collections::BTreeMap;
fn job() -> Job {
    Job {
        manifest: "manifest".into(),
        models: "models".into(),
        model: "tiny".into(),
        fixture: "imgs_1".into(),
        pixels: "input.bgr".into(),
        pixel_sha256: "0".repeat(64),
        width: 100,
        height: 50,
        threads: 1,
        concurrency: 1,
        runs: 3,
        affinity: "1".into(),
    }
}
fn measurement(samples: &[f64]) -> Measurement {
    Measurement {
        language: "rust".into(),
        load_compile_ms: 1.,
        first_inference_ms: 1.,
        warmup_ms: 1.,
        samples_ms: samples.to_vec(),
        peak_private_bytes: 100,
        diagnostics: BTreeMap::new(),
        result: baseline::CaseResult {
            id: "imgs_1".into(),
            text: "".into(),
            detected_count: 0,
            detector_resized_width: 100,
            detector_resized_height: 50,
            lines: vec![],
        },
    }
}
fn comparison(rust: &[f64], csharp: &[f64]) -> Comparison {
    let mut c = Comparison::new(job());
    c.rust = Some(measurement(rust));
    c.csharp = Some(measurement(csharp));
    c
}
#[test]
fn fixed_samples_and_observed_noise_are_not_hidden() {
    let mut c = comparison(&[99., 100., 101.], &[100., 100., 100.]);
    c.evaluate().unwrap();
    assert_eq!(c.status, "passed");
    let mut c = comparison(&[95., 105., 115.], &[100., 100., 100.]);
    c.evaluate().unwrap();
    assert_eq!(c.status, "inconclusive");
    let mut c = comparison(&[112., 115., 118.], &[100., 100., 100.]);
    c.evaluate().unwrap();
    assert_eq!(c.status, "failed");
    let mut c = comparison(&[100., 100.], &[100., 100.]);
    c.job.runs = 2;
    c.evaluate().unwrap();
    assert_eq!(c.status, "inconclusive");
}
#[test]
fn throughput_uses_two_owned_requests_and_its_own_gate() {
    let mut c = comparison(&[110.5, 110.5, 110.5], &[100., 100., 100.]);
    c.job.concurrency = 2;
    c.evaluate().unwrap();
    assert_eq!(c.status, "passed");
    assert!(c.throughput_ratio.unwrap() >= 0.9);
    let mut c = comparison(&[110., 111., 112.], &[100., 100., 100.]);
    c.job.concurrency = 2;
    c.evaluate().unwrap();
    assert_eq!(c.status, "inconclusive");
    let mut c = comparison(&[113., 114., 115.], &[100., 100., 100.]);
    c.job.concurrency = 2;
    c.evaluate().unwrap();
    assert_eq!(c.status, "failed");
}
#[test]
fn memory_and_correctness_are_independent_gates() {
    let mut c = comparison(&[50., 50., 50.], &[100., 100., 100.]);
    c.rust.as_mut().unwrap().peak_private_bytes = 111;
    c.evaluate().unwrap();
    assert_eq!(c.status, "failed");
    let mut c = comparison(&[50., 50., 50.], &[100., 100., 100.]);
    c.rust.as_mut().unwrap().result.detector_resized_width = 101;
    c.evaluate().unwrap();
    assert_eq!(c.status, "failed");
    assert!(!c.differences.is_empty());
}
#[test]
fn missing_or_invalid_measurements_cannot_pass() {
    let mut c = Comparison::new(job());
    c.evaluate().unwrap();
    assert_eq!(c.status, "inconclusive");
    let mut c = comparison(&[1., f64::NAN, 1.], &[1., 1., 1.]);
    assert!(c.evaluate().is_err());
    let mut c = comparison(&[1., 1., 1.], &[1., 1., 1.]);
    c.csharp.as_mut().unwrap().peak_private_bytes = 0;
    assert!(c.evaluate().is_err());
}
#[test]
fn jobs_are_bounded_and_affinity_is_matched() {
    let j = job();
    j.validate().unwrap();
    let mut invalid = j.clone();
    invalid.runs = 4;
    assert!(invalid.validate().is_err());
    invalid = j.clone();
    invalid.affinity = "3".into();
    assert!(invalid.validate().is_err());
    assert_eq!(select_affinity(0b11110100, 3).unwrap(), 0b00110100);
    assert!(select_affinity(1, 8).is_err());
    assert_eq!(median(&[3., 1., 2.]), 2.);
    assert_eq!(median(&[4., 2.]), 3.);
}
#[test]
fn focused_and_duplicate_jobs_cannot_complete_the_full_matrix() {
    let mut one = comparison(&[50., 50., 50.], &[100., 100., 100.]);
    one.evaluate().unwrap();
    assert_eq!(coverage(&[one]).scope, "focused comparison");
    let mut cases = vec![];
    for model in ["tiny", "small", "medium"] {
        for fixture in FIXTURES {
            for threads in [1, 8] {
                for concurrency in [1, 2] {
                    let mut c = comparison(&[50., 50., 50.], &[100., 100., 100.]);
                    c.job.model = model.into();
                    c.job.fixture = fixture.into();
                    c.job.threads = threads;
                    c.job.concurrency = concurrency;
                    c.status = "passed".into();
                    cases.push(c);
                }
            }
        }
    }
    assert!(coverage(&cases).full_qualification_passed);
    let missing = cases.pop().unwrap();
    assert!(!coverage(&cases).full_qualification_passed);
    cases.push(missing);
    cases[35].job = cases[0].job.clone();
    let cov = coverage(&cases);
    assert_eq!(cov.duplicate_case_count, 1);
    assert!(!cov.full_qualification_passed);
}
