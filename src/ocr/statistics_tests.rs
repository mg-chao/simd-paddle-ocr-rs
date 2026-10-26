//! Compare counter-only collection with the prior full-statistics contract.
use super::*;
use crate::{
    error::{ErrorKind, ErrorOrigin, FailureProvenance, OcrStage},
    image::PixelFormat,
    runtime::{
        ExecutionBackend, ExecutionStatistics, KernelSelection, KernelSpecialization,
        StatisticsCounters,
    },
};
use std::collections::BTreeMap;

type StatisticsMap = BTreeMap<OcrStage, Vec<ExecutionStatistics>>;

fn reference_stage_statistics(
    request: &OcrRequest,
    resident: bool,
) -> std::collections::BTreeMap<crate::error::OcrStage, Vec<crate::runtime::ExecutionStatistics>> {
    #[cfg(not(all(feature = "gpu", not(target_os = "macos"))))]
    let _ = resident;
    let mut result = std::collections::BTreeMap::new();
    result.insert(
        crate::error::OcrStage::Detector,
        vec![request.extractor.detector().execution_statistics()],
    );
    result.insert(
        crate::error::OcrStage::Recognizer,
        std::iter::once(request.recognizer.execution_statistics())
            .chain(
                request
                    .workers
                    .iter()
                    .map(|w| w.recognizer.execution_statistics()),
            )
            .collect(),
    );
    let classifiers = request.extractor.classifier_statistics();
    if !classifiers.is_empty() {
        result.insert(crate::error::OcrStage::Classifier, classifiers);
    }
    #[cfg(all(feature = "gpu", not(target_os = "macos")))]
    if resident && let Some(pipeline) = request.execution.resident() {
        let diagnostics = result;
        result = pipeline.execution_statistics();
        for (stage, sessions) in diagnostics {
            if let Some(active) = result.get_mut(&stage).and_then(|s| s.first_mut()) {
                active.diagnostic_readbacks +=
                    sessions.iter().map(|s| s.diagnostic_readbacks).sum::<u64>();
                active.diagnostic_readback_bytes += sessions
                    .iter()
                    .map(|s| s.diagnostic_readback_bytes)
                    .sum::<u64>();
            }
        }
    }
    result
}

fn reference_since(now: &mut ExecutionStatistics, previous: &ExecutionStatistics) {
    let before = [
        previous.production_submissions,
        previous.production_readbacks,
        previous.diagnostic_readbacks,
        previous.uploaded_bytes,
        previous.output_readback_bytes,
        previous.diagnostic_readback_bytes,
        previous.plan_cache_hits,
        previous.plan_cache_misses,
    ];
    let after = [
        &mut now.production_submissions,
        &mut now.production_readbacks,
        &mut now.diagnostic_readbacks,
        &mut now.uploaded_bytes,
        &mut now.output_readback_bytes,
        &mut now.diagnostic_readback_bytes,
        &mut now.plan_cache_hits,
        &mut now.plan_cache_misses,
    ];
    for (value, previous) in after.into_iter().zip(before) {
        if *value >= previous {
            *value -= previous;
        }
    }
}

fn reference_run_stages(
    request: &OcrRequest,
    before: &StatisticsMap,
    resident: bool,
) -> StatisticsMap {
    let mut after = reference_stage_statistics(request, resident);
    for (stage, sessions) in &mut after {
        for (i, session) in sessions.iter_mut().enumerate() {
            if let Some(previous) = before.get(stage).and_then(|s| s.get(i)) {
                reference_since(session, previous);
            }
        }
    }
    if request.latest_run().is_some_and(|run| run.succeeded) && request.result.detected_count == 0 {
        for stage in [OcrStage::Recognizer, OcrStage::Classifier] {
            if let Some(sessions) = after.get_mut(&stage) {
                for session in sessions {
                    session.execution_backend = None;
                }
            }
        }
    }
    after
}

/// Also used by existing ignored GPU integration tests. Collection must not bind
/// lazy sessions, allocate activations, increment counters or change latest_run.
pub(crate) fn assert_snapshot_equivalence(request: &OcrRequest) {
    let activation_bytes = request.retained_activation_bytes();
    let latest = serde_json::to_value(request.latest_run()).unwrap();
    let cumulative = serde_json::to_value(request.counters()).unwrap();
    for resident in [false, true] {
        let full = reference_stage_statistics(request, resident);
        let expected: BTreeMap<_, Vec<_>> = full
            .iter()
            .map(|(stage, sessions)| {
                (
                    *stage,
                    sessions
                        .iter()
                        .map(ExecutionStatistics::counter_snapshot)
                        .collect(),
                )
            })
            .collect();
        assert_eq!(request.stage_counter_snapshots(resident), expected);
        assert_eq!(
            serde_json::to_value(request.stage_statistics(resident)).unwrap(),
            serde_json::to_value(full).unwrap(),
            "the final public kernel, precision, fallback and traffic metadata must be unchanged"
        );
    }
    assert_eq!(request.retained_activation_bytes(), activation_bytes);
    assert_eq!(serde_json::to_value(request.latest_run()).unwrap(), latest);
    assert_eq!(
        serde_json::to_value(request.counters()).unwrap(),
        cumulative
    );
}

fn rich_statistics(counters: [u64; 8]) -> ExecutionStatistics {
    ExecutionStatistics {
        execution_backend: Some(ExecutionBackend::Cpu(crate::runtime::Backend::Scalar)),
        fallback: Some(FailureProvenance {
            kind: ErrorKind::Unsupported,
            origin: ErrorOrigin::Operation,
            stage: Some(OcrStage::Recognizer),
            message: "retained fallback provenance".into(),
        }),
        effective_precision: "reference precision".into(),
        adapter: Some("reference adapter".into()),
        production_submissions: counters[0],
        production_readbacks: counters[1],
        diagnostic_readbacks: counters[2],
        uploaded_bytes: counters[3],
        output_readback_bytes: counters[4],
        diagnostic_readback_bytes: counters[5],
        plan_cache_hits: counters[6],
        plan_cache_misses: counters[7],
        selected_kernels: vec![KernelSelection {
            operator: "MatMul".into(),
            layout: "canonical".into(),
            packed: true,
            steps: 3,
        }],
        specialized_kernels: vec![KernelSpecialization {
            operator: "MatMul".into(),
            family: "reference".into(),
            isa: "scalar".into(),
            store_fused: true,
            steps: 2,
        }],
        normalized_nodes: 11,
        physical_steps: 9,
        activation_bytes: 512,
        retained_activation_bytes: 1024,
        scratch_bytes: 64,
        backend: "reference backend".into(),
        ..Default::default()
    }
}

#[test]
fn counter_snapshots_preserve_delta_resets_and_all_final_metadata() {
    assert_eq!(size_of::<StatisticsCounters>(), size_of::<[u64; 8]>());
    for (before, after) in [
        ([0; 8], [0; 8]),
        ([3, 5, 7, 11, 13, 17, 19, 23], [3, 5, 7, 11, 13, 17, 19, 23]),
        (
            [3, 5, 7, 11, 13, 17, 19, 23],
            [8, 13, 18, 23, 28, 33, 38, 43],
        ),
        (
            [30, 50, 70, 110, 130, 170, 190, 230],
            [3, 5, 7, 11, 13, 17, 19, 23],
        ),
        (
            [u64::MAX, 0, u64::MAX, 1, 2, 3, 4, 5],
            [1, u64::MAX, u64::MAX, 0, 7, 3, 9, 2],
        ),
    ] {
        let before = rich_statistics(before);
        let mut expected = rich_statistics(after);
        let mut actual = expected.clone();
        reference_since(&mut expected, &before);
        actual.since(&before.counter_snapshot());
        assert_eq!(
            serde_json::to_value(actual).unwrap(),
            serde_json::to_value(expected).unwrap()
        );
    }
}

#[test]
fn counter_snapshots_preserve_missing_stages_and_new_session_indices() {
    let before = BTreeMap::from([
        (OcrStage::Detector, vec![rich_statistics([10; 8])]),
        (
            OcrStage::Recognizer,
            vec![rich_statistics([20; 8]), rich_statistics([40; 8])],
        ),
    ]);
    let mut actual = BTreeMap::from([
        (OcrStage::Detector, vec![rich_statistics([2; 8])]), // Replaced session.
        (OcrStage::Classifier, vec![rich_statistics([7; 8])]), // New stage.
        (
            OcrStage::Recognizer,
            vec![
                rich_statistics([31; 8]),
                rich_statistics([42; 8]),
                rich_statistics([9; 8]),
            ],
        ),
    ]);
    let mut expected = actual.clone();
    for (stage, sessions) in &mut expected {
        for (i, session) in sessions.iter_mut().enumerate() {
            if let Some(previous) = before.get(stage).and_then(|s| s.get(i)) {
                reference_since(session, previous);
            }
        }
    }
    let snapshots: BTreeMap<_, Vec<_>> = before
        .iter()
        .map(|(stage, sessions)| {
            (
                *stage,
                sessions
                    .iter()
                    .map(ExecutionStatistics::counter_snapshot)
                    .collect(),
            )
        })
        .collect();
    for (stage, sessions) in &mut actual {
        for (i, session) in sessions.iter_mut().enumerate() {
            if let Some(previous) = snapshots.get(stage).and_then(|s| s.get(i)) {
                session.since(previous);
            }
        }
    }
    assert_eq!(
        serde_json::to_value(actual).unwrap(),
        serde_json::to_value(expected).unwrap()
    );
}

#[test]
fn counter_snapshots_match_cpu_pending_reuse_trace_failure_and_no_work() {
    let mut request = tests::synthetic_ocr(CpuContext::serial());
    assert_eq!(request.retained_activation_bytes(), 0);
    assert_snapshot_equivalence(&request);
    let pixels = tests::two_lines();
    let image = ImageView::new(&pixels, 96, 96, 0, PixelFormat::Bgr24).unwrap();
    let mut expected_result = None;
    for run in 1..=2 {
        let before = reference_stage_statistics(&request, true);
        let actual = request
            .run(
                image,
                RunOptions {
                    capture_spans: true,
                },
            )
            .unwrap()
            .try_to_owned()
            .unwrap();
        assert_eq!(actual.detected_count, 2);
        if let Some(expected) = &expected_result {
            assert_eq!(&actual, expected);
        } else {
            expected_result = Some(actual);
        }
        assert_eq!(
            serde_json::to_value(&request.latest_run().unwrap().stages).unwrap(),
            serde_json::to_value(reference_run_stages(&request, &before, true)).unwrap()
        );
        assert_eq!(request.counters().runs, run);
        assert_snapshot_equivalence(&request);
    }
    let before = reference_stage_statistics(&request, false);
    let mut callbacks = 0;
    let error = request
        .run_with_trace(image, true, |_| {
            callbacks += 1;
            Err(ImageError::with_kind(
                ErrorKind::DeviceLost,
                "counter-snapshot observer failure",
            ))
        })
        .err()
        .unwrap();
    assert_eq!(callbacks, 1);
    assert_eq!(error.origin(), ErrorOrigin::Callback);
    assert!(!request.latest_run().unwrap().succeeded);
    assert_eq!(request.counters().failures, 1);
    assert_eq!(
        serde_json::to_value(&request.latest_run().unwrap().stages).unwrap(),
        serde_json::to_value(reference_run_stages(&request, &before, false)).unwrap()
    );
    assert_snapshot_equivalence(&request);

    let black = vec![0; pixels.len()];
    let image = ImageView::new(&black, 96, 96, 0, PixelFormat::Bgr24).unwrap();
    let before = reference_stage_statistics(&request, true);
    assert_eq!(
        request
            .run(image, RunOptions::default())
            .unwrap()
            .detected_count,
        0
    );
    assert_eq!(
        serde_json::to_value(&request.latest_run().unwrap().stages).unwrap(),
        serde_json::to_value(reference_run_stages(&request, &before, true)).unwrap()
    );
    assert_snapshot_equivalence(&request);
    assert!(
        request.latest_run().unwrap().stages[&OcrStage::Recognizer]
            .iter()
            .all(|s| s.execution_backend.is_none())
    );
}

#[test]
fn counter_snapshots_match_classifier_and_worker_session_selection() {
    #[cfg(feature = "parallel")]
    let capacity = 2;
    #[cfg(not(feature = "parallel"))]
    let capacity = 1;
    let cpu = Arc::new(CpuContext::new(capacity).unwrap());
    let base = tests::synthetic_ocr(cpu.clone());
    let extractor = OrientedLineExtractor::new(
        base.extractor.detector().fork().unwrap(),
        Some(tests::synthetic_classifier().into()),
        crate::lines::LineOptions::default(),
    )
    .unwrap();
    let mut request = OcrRequest::new_with_execution(
        extractor,
        base.recognizer.fork().unwrap(),
        OcrOptions::default(),
        cpu,
        CpuOptions {
            line_workers: capacity,
            ..Default::default()
        },
    )
    .unwrap();
    assert_snapshot_equivalence(&request);
    let pixels = tests::two_lines();
    let image = ImageView::new(&pixels, 96, 96, 0, PixelFormat::Bgr24).unwrap();
    let before = reference_stage_statistics(&request, true);
    assert_eq!(
        request
            .run(image, RunOptions::default())
            .unwrap()
            .detected_count,
        2
    );
    assert_snapshot_equivalence(&request);
    assert_eq!(
        serde_json::to_value(&request.latest_run().unwrap().stages).unwrap(),
        serde_json::to_value(reference_run_stages(&request, &before, true)).unwrap()
    );
    assert!(
        request
            .execution_statistics()
            .contains_key(&OcrStage::Classifier)
    );
    #[cfg(feature = "parallel")]
    assert!(
        !request.workers.is_empty(),
        "the counter collector must cover live worker sessions"
    );
}

#[cfg(all(feature = "gpu", not(target_os = "macos")))]
#[test]
#[ignore = "requires a physical GPU; run explicitly"]
fn gpu_hardware_counter_snapshots_match_resident_hybrid_and_trace() {
    use crate::{
        Compiler, CompilerOptions, OcrModelSet,
        detector::DetectorOptions,
        model::Model,
        recognizer::Dictionary,
        runtime::{ExecutionTarget, gpu::GpuContext},
    };
    let context = GpuContext::new().unwrap();
    assert!(matches!(
        context.adapter_info().device_type,
        crate::runtime::gpu::GpuDeviceType::DiscreteGpu
            | crate::runtime::gpu::GpuDeviceType::IntegratedGpu
    ));
    let compiler = Arc::new(Compiler::new(CompilerOptions {
        gpu: Some(context),
        ..Default::default()
    }));
    let (detector, recognizer) = tests::synthetic_sources();
    let models = OcrModelSet::new(
        Arc::new(Model::load_bytes(&detector).unwrap()),
        Arc::new(Model::load_bytes(&recognizer).unwrap()),
        Arc::new(Dictionary::from_utf8(b"A\n").unwrap()),
        None,
    );
    let pixels = tests::two_lines();
    let image = ImageView::new(&pixels, 96, 96, 0, PixelFormat::Bgr24).unwrap();
    let black = vec![0; pixels.len()];
    let empty = ImageView::new(&black, 96, 96, 0, PixelFormat::Bgr24).unwrap();
    for recognizer_target in [ExecutionTarget::Gpu, ExecutionTarget::Cpu] {
        let plan = models
            .clone()
            .builder()
            .compiler(compiler.clone())
            .execution_target(ExecutionTarget::Gpu)
            .recognizer_target(recognizer_target)
            .cpu_context(CpuContext::serial())
            .detector_options(DetectorOptions {
                limit_side_length: 96,
                ..Default::default()
            })
            .build()
            .unwrap();
        let mut request = plan.request().unwrap();
        assert_eq!(request.retained_activation_bytes(), 0);
        assert_snapshot_equivalence(&request);
        let mut submissions = 0;
        for _ in 0..2 {
            let before = reference_stage_statistics(&request, true);
            assert_eq!(
                request
                    .run(image, RunOptions::default())
                    .unwrap()
                    .detected_count,
                2
            );
            assert_eq!(
                serde_json::to_value(&request.latest_run().unwrap().stages).unwrap(),
                serde_json::to_value(reference_run_stages(&request, &before, true)).unwrap()
            );
            submissions += request.gpu_pipeline_statistics().unwrap().submissions;
            assert_eq!(request.counters().submissions, submissions);
            assert_snapshot_equivalence(&request);
        }
        let before = reference_stage_statistics(&request, false);
        request.run_with_trace(image, false, |_| Ok(())).unwrap();
        assert_eq!(
            request.latest_run().unwrap().execution,
            OcrExecution::Hybrid
        );
        assert_eq!(
            serde_json::to_value(&request.latest_run().unwrap().stages).unwrap(),
            serde_json::to_value(reference_run_stages(&request, &before, false)).unwrap()
        );
        assert!(
            request
                .latest_run()
                .unwrap()
                .stages
                .values()
                .flatten()
                .any(|s| s.diagnostic_readbacks > 0),
            "trace stage counters ({recognizer_target:?}): {:?}",
            request.latest_run().unwrap().stages
        );
        // A retained resident pipeline now coexists with primary trace sessions.
        // The comparison exercises the diagnostic-only first-session merge.
        assert_snapshot_equivalence(&request);
        let before = reference_stage_statistics(&request, false);
        assert!(
            request
                .run_with_trace(image, false, |_| Err(ImageError::new(
                    "snapshot observer failure"
                )))
                .is_err()
        );
        assert!(!request.latest_run().unwrap().succeeded);
        assert_eq!(
            serde_json::to_value(&request.latest_run().unwrap().stages).unwrap(),
            serde_json::to_value(reference_run_stages(&request, &before, false)).unwrap()
        );
        assert_snapshot_equivalence(&request);
        let before = reference_stage_statistics(&request, true);
        assert_eq!(
            request
                .run(empty, RunOptions::default())
                .unwrap()
                .detected_count,
            0
        );
        assert_eq!(
            serde_json::to_value(&request.latest_run().unwrap().stages).unwrap(),
            serde_json::to_value(reference_run_stages(&request, &before, true)).unwrap()
        );
        assert_snapshot_equivalence(&request);
    }
}
