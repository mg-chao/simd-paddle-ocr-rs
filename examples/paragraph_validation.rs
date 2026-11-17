//! Local JSON harness for paragraph parity checks and benchmarks.
use serde::{Deserialize, Serialize};
use simd_paddle_ocr_rs::{
    ExecutionBackend, ExecutionConfig, ExecutionTarget, ModelPreset, OcrExecution, OcrModelSet,
    OcrStage, RunOptions,
    cpu::CpuContext,
    image::{ImageView, PixelFormat},
    paragraph::{ParagraphPlan, ParagraphProfile},
    runtime::ExecutionStatistics,
};
use std::{
    error::Error,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

type Quad = [[f32; 2]; 4];

#[derive(Deserialize)]
struct Fixture {
    id: usize,
    width: usize,
    height: usize,
    bgr: PathBuf,
    quads: Vec<Quad>,
    prepared: Option<Prepared>,
}

#[derive(Deserialize, Serialize)]
struct Prepared {
    nodes: Vec<f32>,
    edges: Vec<f32>,
    pairs: Vec<[usize; 2]>,
}

#[derive(Serialize)]
struct CaseCheck {
    id: usize,
    groups: Vec<Vec<usize>>,
    prepared: Prepared,
    logits: Vec<f32>,
    same_feed_logits: Vec<f32>,
    backend: Option<ExecutionBackend>,
    adapter: Option<String>,
}

#[derive(Serialize)]
struct Sample {
    total_ns: u64,
    preprocess_ns: u64,
    inference_ns: u64,
    decode_ns: u64,
}

#[derive(Serialize)]
struct CaseBenchmark {
    id: usize,
    groups: Vec<Vec<usize>>,
    candidates: usize,
    samples: Vec<Sample>,
    backend: Option<ExecutionBackend>,
    adapter: Option<String>,
}

#[derive(Serialize)]
struct IntegratedCheck {
    id: usize,
    quads: Vec<Quad>,
    groups: Vec<Vec<usize>>,
    execution: OcrExecution,
    backend: Option<ExecutionBackend>,
    adapter: Option<String>,
}

fn selected_backend(
    statistics: &ExecutionStatistics,
    target: ExecutionTarget,
) -> Result<(), Box<dyn Error>> {
    if !matches!(
        (target, statistics.execution_backend),
        (ExecutionTarget::Cpu, Some(ExecutionBackend::Cpu(_)))
            | (ExecutionTarget::Gpu, Some(ExecutionBackend::Gpu))
    ) || statistics.fallback.is_some()
    {
        return Err(format!(
            "requested {target:?}, got {:?}, fallback {:?}",
            statistics.execution_backend, statistics.fallback
        )
        .into());
    }
    if target == ExecutionTarget::Gpu
        && (statistics.production_submissions == 0 || statistics.production_readbacks == 0)
    {
        return Err("the GPU paragraph request recorded no submissions or readbacks".into());
    }
    Ok(())
}

fn integrated_ocr(
    model: &Path,
    models: OcrModelSet,
    fixture_id: usize,
    image: ImageView<'_>,
    paragraph_target: ExecutionTarget,
    expected: OcrExecution,
) -> Result<IntegratedCheck, Box<dyn Error>> {
    let ocr_target = if expected == OcrExecution::Resident {
        ExecutionTarget::Gpu
    } else {
        ExecutionTarget::Cpu
    };
    let plan = models
        .builder()
        .cpu_context(Arc::new(CpuContext::new(1)?))
        .execution_target(ocr_target)
        .paragraph_model(model)
        .paragraph_target(paragraph_target)
        .build()?;
    let mut request = plan.request()?;
    let ocr = request.run(image, RunOptions::default())?;
    let quads = ocr.iter().map(|line| line.detection.points).collect();
    let groups = ocr.paragraphs().ok_or("OCR omitted paragraphs")?.to_vec();
    let run = request.latest_run().ok_or("OCR omitted run statistics")?;
    if run.execution != expected {
        return Err(format!("expected {expected:?} OCR, got {:?}", run.execution).into());
    }
    let statistics = run
        .stages
        .get(&OcrStage::Paragraph)
        .and_then(|stage| stage.first())
        .ok_or("OCR omitted paragraph stage statistics")?;
    selected_backend(statistics, paragraph_target)?;
    Ok(IntegratedCheck {
        id: fixture_id,
        quads,
        groups,
        execution: run.execution,
        backend: statistics.execution_backend,
        adapter: statistics.adapter.clone(),
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let usage = "usage: paragraph_validation ONNX FIXTURES_JSON cpu|gpu check|bench [OCR_MODELS]";
    let model = PathBuf::from(args.next().ok_or(usage)?);
    let fixtures = PathBuf::from(args.next().ok_or(usage)?);
    let target = match args.next().as_deref().and_then(|value| value.to_str()) {
        Some("cpu") => ExecutionTarget::Cpu,
        Some("gpu") => ExecutionTarget::Gpu,
        _ => return Err(usage.into()),
    };
    let benchmark = match args.next().as_deref().and_then(|value| value.to_str()) {
        Some("check") => false,
        Some("bench") => true,
        _ => return Err(usage.into()),
    };
    let ocr_models = args.next().map(PathBuf::from);
    if args.next().is_some() || (benchmark && ocr_models.is_some()) {
        return Err(usage.into());
    }

    let fixtures: Vec<Fixture> = serde_json::from_slice(&std::fs::read(fixtures)?)?;
    let plan = ParagraphPlan::load_path(
        &model,
        ExecutionConfig {
            target,
            ..Default::default()
        },
    )?;
    let mut request = plan.request()?;
    let mut checks = Vec::new();
    let mut benchmarks = Vec::new();
    let mut integrated = None;
    let mut resident = None;
    for fixture in fixtures {
        let pixels = std::fs::read(&fixture.bgr)?;
        let image = ImageView::new(
            &pixels,
            fixture.width,
            fixture.height,
            0,
            PixelFormat::Bgr24,
        )?;
        if benchmark {
            let groups = request.group(image, &fixture.quads)?.to_vec();
            let mut samples = Vec::with_capacity(3);
            let mut candidates = 0;
            for _ in 0..3 {
                let start = Instant::now();
                let (measured_groups, profile) = request.group_profiled(image, &fixture.quads)?;
                let total_ns = start.elapsed().as_nanos().try_into()?;
                if measured_groups != groups {
                    return Err(format!(
                        "case {} produced different groups between runs",
                        fixture.id
                    )
                    .into());
                }
                let ParagraphProfile {
                    preprocess_ns,
                    inference_ns,
                    decode_ns,
                    candidate_count,
                    ..
                } = profile;
                candidates = candidate_count;
                samples.push(Sample {
                    total_ns,
                    preprocess_ns,
                    inference_ns,
                    decode_ns,
                });
            }
            let statistics = request.statistics();
            selected_backend(&statistics, target)?;
            benchmarks.push(CaseBenchmark {
                id: fixture.id,
                groups,
                candidates,
                samples,
                backend: statistics.execution_backend,
                adapter: statistics.adapter,
            });
            continue;
        }

        let groups = request.group(image, &fixture.quads)?.to_vec();
        let diagnostics = request.diagnostics();
        let prepared = Prepared {
            nodes: diagnostics.nodes.to_vec(),
            edges: diagnostics.edges.to_vec(),
            pairs: diagnostics.pairs.to_vec(),
        };
        let logits = diagnostics.logits.to_vec();
        let statistics = request.statistics();
        selected_backend(&statistics, target)?;
        let python = fixture
            .prepared
            .ok_or("check fixtures require Python model inputs")?;
        let same_feed_logits = request
            .score_prepared(&python.nodes, &python.edges, &python.pairs)?
            .to_vec();
        selected_backend(&request.statistics(), target)?;
        checks.push(CaseCheck {
            id: fixture.id,
            groups,
            prepared,
            logits,
            same_feed_logits,
            backend: statistics.execution_backend,
            adapter: statistics.adapter,
        });

        if fixture.id == 5
            && let Some(models_dir) = &ocr_models
        {
            let models =
                OcrModelSet::from_directory(models_dir, ModelPreset::ChineseV6Small, false)?;
            let expected = if target == ExecutionTarget::Gpu {
                OcrExecution::Hybrid
            } else {
                OcrExecution::Cpu
            };
            integrated = Some(integrated_ocr(
                &model,
                models.clone(),
                fixture.id,
                image,
                target,
                expected,
            )?);
            if target == ExecutionTarget::Gpu {
                resident = Some(integrated_ocr(
                    &model,
                    models,
                    fixture.id,
                    image,
                    target,
                    OcrExecution::Resident,
                )?);
            }
        }
    }

    let stdout = std::io::stdout().lock();
    if benchmark {
        serde_json::to_writer(stdout, &benchmarks)?;
    } else {
        if ocr_models.is_some() && integrated.is_none() {
            return Err("integrated OCR validation requires fixture 5".into());
        }
        serde_json::to_writer(
            stdout,
            &serde_json::json!({"cases": checks, "integrated": integrated, "resident": resident}),
        )?;
    }
    Ok(())
}
