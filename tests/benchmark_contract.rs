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
        process_cpu: None,
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

#[test]
fn balanced_order_is_independent_of_workload_and_selection() {
    let mut first_language_counts = BTreeMap::new();
    let mut model_counts = BTreeMap::new();
    let mut fixture_counts = BTreeMap::new();
    let mut thread_counts = BTreeMap::new();
    for model in ["tiny", "small", "medium"] {
        for threads in [1, 8] {
            for fixture in FIXTURES {
                let mut j = job();
                j.model = model.into();
                j.threads = threads;
                j.affinity = if threads == 1 { "1" } else { "ff" }.into();
                j.fixture = fixture.into();
                let expected = OrderStrategy::Balanced.languages(&j).unwrap();
                for concurrency in [1, 2] {
                    j.concurrency = concurrency;
                    let order = OrderStrategy::Balanced.languages(&j).unwrap();
                    assert_eq!(order, expected);
                    *first_language_counts
                        .entry((concurrency, order[0]))
                        .or_insert(0) += 1;
                    *model_counts
                        .entry((model, concurrency, order[0]))
                        .or_insert(0) += 1;
                    *fixture_counts
                        .entry((fixture, concurrency, order[0]))
                        .or_insert(0) += 1;
                    *thread_counts
                        .entry((threads, concurrency, order[0]))
                        .or_insert(0) += 1;
                }
            }
        }
    }
    for concurrency in [1, 2] {
        assert_eq!(first_language_counts[&(concurrency, "rust")], 9);
        assert_eq!(first_language_counts[&(concurrency, "csharp")], 9);
        for model in ["tiny", "small", "medium"] {
            assert_eq!(model_counts[&(model, concurrency, "rust")], 3);
            assert_eq!(model_counts[&(model, concurrency, "csharp")], 3);
        }
        for fixture in FIXTURES {
            assert_eq!(fixture_counts[&(fixture, concurrency, "rust")], 3);
            assert_eq!(fixture_counts[&(fixture, concurrency, "csharp")], 3);
        }
        for threads in [1, 8] {
            let rust_count: i32 = thread_counts[&(threads, concurrency, "rust")];
            let csharp_count: i32 = thread_counts[&(threads, concurrency, "csharp")];
            assert_eq!((rust_count - csharp_count).abs(), 1);
        }
    }
    // A focused selection or reversed fixture traversal retains the full
    // matrix's canonical order rather than restarting a selected-index parity.
    let orders = FIXTURES.map(|fixture| {
        let mut j = job();
        j.fixture = fixture.into();
        OrderStrategy::Balanced.languages(&j).unwrap()
    });
    for (index, fixture) in FIXTURES.iter().enumerate().rev() {
        let mut focused = job();
        focused.fixture = (*fixture).into();
        assert_eq!(
            OrderStrategy::Balanced.languages(&focused).unwrap(),
            orders[index]
        );
    }
    assert_ne!(orders[0], orders[1]);
}

#[test]
fn explicit_orders_are_available_without_changing_the_workload() {
    for (value, expected) in [
        ("rust-first", ["rust", "csharp"]),
        ("csharp-first", ["csharp", "rust"]),
    ] {
        let strategy = OrderStrategy::parse(value).unwrap();
        assert_eq!(strategy.name(), value);
        for concurrency in [1, 2] {
            let mut j = job();
            j.concurrency = concurrency;
            assert_eq!(strategy.languages(&j).unwrap(), expected);
        }
    }
    assert_eq!(
        OrderStrategy::parse("balanced").unwrap(),
        OrderStrategy::Balanced
    );
    assert!(OrderStrategy::parse("random").is_err());
}

#[test]
fn legacy_measurements_and_cases_keep_unavailable_diagnostics() {
    let mut raw = serde_json::to_value(comparison(&[1., 1., 1.], &[1., 1., 1.])).unwrap();
    let object = raw.as_object_mut().unwrap();
    object.remove("planned_execution_order");
    object.remove("execution_order");
    for language in ["rust", "csharp"] {
        object
            .get_mut(language)
            .unwrap()
            .as_object_mut()
            .unwrap()
            .remove("process_cpu");
    }
    let mut legacy: Comparison = serde_json::from_value(raw).unwrap();
    assert!(legacy.planned_execution_order.is_empty());
    assert!(legacy.execution_order.is_empty());
    assert!(legacy.rust.as_ref().unwrap().process_cpu.is_none());
    legacy.evaluate().unwrap();
    assert_eq!(legacy.status, "passed");
    legacy.planned_execution_order = vec!["csharp".into(), "rust".into()];
    legacy.execution_order = vec!["csharp".into()];
    legacy.rust.as_mut().unwrap().process_cpu = Some(ProcessCpuTiming {
        first_inference_ms: 3.,
        warmup_ms: 2.,
        samples_ms: vec![1., 0., 2.],
    });
    let restored: Comparison =
        serde_json::from_value(serde_json::to_value(&legacy).unwrap()).unwrap();
    assert_eq!(
        restored.planned_execution_order,
        legacy.planned_execution_order
    );
    assert_eq!(restored.execution_order, legacy.execution_order);
    assert_eq!(
        restored.rust.unwrap().process_cpu.unwrap().samples_ms,
        [1., 0., 2.]
    );
}

#[test]
fn process_cpu_samples_are_validated_without_changing_wall_gates() {
    let cpu = ProcessCpuTiming {
        first_inference_ms: 0.,
        warmup_ms: 1.,
        samples_ms: vec![0., 250., 500.],
    };
    let mut c = comparison(&[100., 100., 100.], &[100., 100., 100.]);
    c.rust.as_mut().unwrap().process_cpu = Some(cpu.clone());
    c.evaluate().unwrap();
    assert_eq!(c.status, "passed");
    for invalid_samples in [vec![1., 2.], vec![1., -1., 1.], vec![1., f64::NAN, 1.]] {
        let mut invalid = cpu.clone();
        invalid.samples_ms = invalid_samples;
        c.rust.as_mut().unwrap().process_cpu = Some(invalid);
        assert!(c.evaluate().is_err());
    }
    for invalid_first in [-1., f64::INFINITY] {
        let mut invalid = cpu.clone();
        invalid.first_inference_ms = invalid_first;
        c.rust.as_mut().unwrap().process_cpu = Some(invalid);
        assert!(c.evaluate().is_err());
    }
    let mut invalid_warmup = cpu.clone();
    invalid_warmup.warmup_ms = -1.;
    c.rust.as_mut().unwrap().process_cpu = Some(invalid_warmup);
    assert!(c.evaluate().is_err());
    c.rust.as_mut().unwrap().process_cpu = Some(cpu);
    c.rust.as_mut().unwrap().samples_ms = vec![120.; 3];
    c.evaluate().unwrap();
    assert_eq!(c.status, "failed");
    let mut partial = serde_json::to_value(measurement(&[1.; 3])).unwrap();
    partial["process_cpu"] = serde_json::json!({"samples_ms": [1., 1., 1.]});
    assert!(serde_json::from_value::<Measurement>(partial).is_err());
}
