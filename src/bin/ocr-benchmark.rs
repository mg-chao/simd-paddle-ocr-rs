//! Independently scoped scalar REC/OCR timing, production profiling and allocation measurements.
#[path = "support/allocator.rs"]
mod allocator;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use simd_paddle_ocr_rs::{
    detector::{Detector, DetectorOptions},
    image::{ImageView, PixelFormat},
    lines::{LineOptions, OrientedLineExtractor},
    model::Model,
    ocr::{Ocr, OcrOptions},
    recognizer::{Dictionary, Recognizer, RecognizerOptions},
    runtime::{CompiledModel, ExecutionLimits, Session},
    validate_bundle,
};
use std::{
    env, fs,
    hint::black_box,
    path::{Path, PathBuf},
    process::ExitCode,
    sync::Arc,
    time::Instant,
};
type Result<T> = std::result::Result<T, String>;
#[derive(Clone)]
struct Input {
    case: String,
    id: String,
    shape: Vec<usize>,
    sha256: String,
    data: Vec<u8>,
    width: usize,
    height: usize,
    stride: usize,
}
impl Input {
    fn image(&self) -> Result<ImageView<'_>> {
        ImageView::new(
            &self.data,
            self.width,
            self.height,
            self.stride,
            PixelFormat::Bgr24,
        )
        .map_err(|e| e.to_string())
    }
}
#[derive(Serialize)]
struct Row {
    case_id: String,
    artifact_id: String,
    shape: Vec<usize>,
    sha256: String,
    stage: String,
    first_call_ms: f64,
    first_call_allocations: allocator::Counts,
    samples_ms: Vec<f64>,
    p50_ms: f64,
    p95_ms: f64,
    mean_ms: f64,
    #[serde(flatten)]
    counts: allocator::Counts,
}
fn percentile(values: &[f64], q: f64) -> f64 {
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    let p = (v.len() - 1) as f64 * q;
    let a = p.floor() as usize;
    let b = p.ceil() as usize;
    v[a] + (v[b] - v[a]) * (p - a as f64)
}
fn measure(
    input: &Input,
    stage: &str,
    warmup: usize,
    samples: usize,
    zero: bool,
    mut call: impl FnMut() -> Result<f64>,
) -> Result<(Row, f64)> {
    let baseline = allocator::begin();
    let start = Instant::now();
    let mut checksum = black_box(call()?);
    let first_call_ms = start.elapsed().as_secs_f64() * 1000.;
    let first_call_allocations = allocator::end(baseline);
    for _ in 0..warmup {
        checksum += black_box(call()?);
    }
    let mut times = Vec::with_capacity(samples);
    let baseline = allocator::begin();
    for _ in 0..samples {
        let start = Instant::now();
        checksum += black_box(call()?);
        times.push(start.elapsed().as_secs_f64() * 1000.);
    }
    let counts = allocator::end(baseline);
    if zero && counts.warm_allocation_count.is_some_and(|n| n != 0) {
        return Err(format!(
            "{stage}/{} warmed allocation gate failed: {} calls, {} bytes",
            input.id,
            counts.warm_allocation_count.unwrap(),
            counts.warm_allocation_bytes.unwrap()
        ));
    }
    if zero && counts.warm_retained_heap_bytes.is_some_and(|n| n != 0) {
        return Err(format!("{stage}/{} warmed retained heap grew", input.id));
    }
    Ok((
        Row {
            case_id: input.case.clone(),
            artifact_id: input.id.clone(),
            shape: input.shape.clone(),
            sha256: input.sha256.clone(),
            stage: stage.into(),
            first_call_ms,
            first_call_allocations,
            p50_ms: percentile(&times, 0.5),
            p95_ms: percentile(&times, 0.95),
            mean_ms: times.iter().sum::<f64>() / samples as f64,
            samples_ms: times,
            counts,
        },
        checksum,
    ))
}
fn read_json(path: impl AsRef<Path>) -> Result<Value> {
    serde_json::from_slice(&fs::read(path).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}
fn number(v: &Value, k: &str) -> Result<usize> {
    v[k].as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| format!("missing {k}"))
}
fn artifact<'a>(case: &'a Value, id: &str) -> Result<&'a Value> {
    case["artifacts"]
        .as_array()
        .ok_or("missing artifacts")?
        .iter()
        .find(|a| a["id"] == id)
        .ok_or_else(|| format!("missing {id}"))
}
fn inputs(
    root: &Path,
    manifest: &Value,
    stage: &str,
    selected: Option<&str>,
) -> Result<Vec<Input>> {
    let mut result = Vec::new();
    for case in manifest["cases"].as_array().ok_or("missing cases")? {
        let cid = case["id"].as_str().ok_or("case id")?;
        if selected.is_some_and(|s| s != cid) {
            continue;
        }
        for a in case["artifacts"].as_array().ok_or("artifacts")? {
            let id = a["id"].as_str().ok_or("artifact id")?;
            if !match stage {
                "rec_graph" | "node_profile" => id.ends_with(".rec.input"),
                "rec_public_owned" | "rec_public_borrowed" | "owned_copies" => {
                    id.ends_with(".rec.pixels")
                }
                _ => id == "input.pixels",
            } {
                continue;
            }
            let shape = a["shape"]
                .as_array()
                .ok_or("shape")?
                .iter()
                .map(|v| v.as_u64().map(|n| n as usize).ok_or("dimension".into()))
                .collect::<Result<Vec<_>>>()?;
            let (width, height, stride) = if id == "input.pixels" {
                let m = read_json(
                    root.join(artifact(case, "input.metadata")?["path"].as_str().unwrap()),
                )?;
                (
                    number(&m, "width")?,
                    number(&m, "height")?,
                    number(&m, "stride")?,
                )
            } else if id.ends_with(".pixels") {
                (shape[1], shape[0], shape[1] * 3)
            } else {
                (0, 0, 0)
            };
            result.push(Input {
                case: cid.into(),
                id: id.into(),
                shape,
                sha256: a["sha256"].as_str().ok_or("sha")?.into(),
                data: fs::read(root.join(a["path"].as_str().ok_or("path")?))
                    .map_err(|e| e.to_string())?,
                width,
                height,
                stride,
            });
        }
    }
    if result.is_empty() {
        return Err("selected benchmark corpus is empty".into());
    }
    Ok(result)
}
#[derive(Deserialize)]
struct DetConfig {
    limit_side_length: usize,
    max_candidates: usize,
    use_dilation: bool,
    bitmap_threshold: f32,
    box_threshold: f32,
    unclip_ratio: f32,
    max_image_pixels: usize,
}
fn rec_options(m: &Value) -> Result<RecognizerOptions> {
    let options = &m["contract"]["options"]["recognizer"];
    Ok(RecognizerOptions {
        target_width: number(options, "target_width")?,
        adaptive_width: options["adaptive_width"]
            .as_bool()
            .ok_or("adaptive_width")?,
        max_image_pixels: number(options, "max_image_pixels")?,
        ..Default::default()
    })
}
fn make_ocr(root: &Path, m: &Value, rec: Arc<CompiledModel>, dict: Arc<Dictionary>) -> Result<Ocr> {
    let asset = |id: &str| {
        m["models"]
            .as_array()
            .unwrap()
            .iter()
            .find(|v| v["id"] == id)
            .ok_or("missing model")
    };
    let compile = |id: &str| -> Result<Arc<CompiledModel>> {
        Ok(Arc::new(
            CompiledModel::new(Arc::new(
                Model::load_path(root.join(asset(id)?["path"].as_str().unwrap()))
                    .map_err(|e| e.to_string())?,
            ))
            .map_err(|e| e.to_string())?,
        ))
    };
    let c: DetConfig = serde_json::from_value(m["contract"]["options"]["detector"].clone())
        .map_err(|e| e.to_string())?;
    let detector = Detector::new(
        compile("det")?,
        DetectorOptions {
            limit_side_length: c.limit_side_length,
            max_candidates: c.max_candidates,
            use_dilation: c.use_dilation,
            bitmap_threshold: c.bitmap_threshold,
            box_threshold: Some(c.box_threshold),
            unclip_ratio: c.unclip_ratio,
            max_image_pixels: c.max_image_pixels,
        },
        ExecutionLimits::default(),
    )
    .map_err(|e| e.to_string())?;
    let extractor = OrientedLineExtractor::new(
        detector,
        if m["contract"]["options"]["use_direction_classification"]
            .as_bool()
            .ok_or("use_direction_classification")?
        {
            Some(compile("cls")?)
        } else {
            None
        },
        LineOptions {
            classifier_threshold: m["contract"]["options"]["classifier_threshold"]
                .as_f64()
                .ok_or("classifier threshold")? as f32,
            max_crop_pixels: number(&m["contract"]["options"], "max_crop_pixels")?,
            max_classifier_image_pixels: number(
                &m["contract"]["options"]["classifier"],
                "max_image_pixels",
            )?,
            ..Default::default()
        },
    )
    .map_err(|e| e.to_string())?;
    Ocr::new(
        extractor,
        Recognizer::new(rec, dict, rec_options(m)?, ExecutionLimits::default())
            .map_err(|e| e.to_string())?,
        OcrOptions::default(),
    )
    .map_err(|e| e.to_string())
}
fn run() -> Result<()> {
    let mut options = std::collections::HashMap::new();
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        if !matches!(
            flag.as_str(),
            "--reference" | "--mode" | "--warmup" | "--samples" | "--case"
        ) {
            return Err(format!("unknown {flag}"));
        }
        let value = args.next().ok_or("missing value")?;
        if options.insert(flag, value).is_some() {
            return Err("duplicate option".into());
        }
    }
    let root = PathBuf::from(options.get("--reference").ok_or("--reference required")?);
    let mode = options.get("--mode").map_or("rec_graph", String::as_str);
    if !matches!(
        mode,
        "rec_graph"
            | "rec_public_owned"
            | "rec_public_borrowed"
            | "ocr_owned"
            | "ocr_borrowed"
            | "mixed_widths"
            | "node_profile"
            | "owned_copies"
            | "public_stage_profile"
    ) {
        return Err("invalid mode".into());
    }
    let warmup = options
        .get("--warmup")
        .map_or(Ok(30), |v| v.parse::<usize>().map_err(|_| "warmup"))?;
    let samples = options
        .get("--samples")
        .map_or(Ok(100), |v| v.parse::<usize>().map_err(|_| "samples"))?;
    if samples == 0 {
        return Err("samples must be positive".into());
    }
    validate_bundle(&root).map_err(|e| e.to_string())?;
    let manifest_bytes = fs::read(root.join("manifest.json")).map_err(|e| e.to_string())?;
    let m: Value = serde_json::from_slice(&manifest_bytes).map_err(|e| e.to_string())?;
    let family = m["contract"]["model"]
        .as_str()
        .ok_or("missing model family")?;
    if m["contract"]["profile"] != "canonical" || !matches!(family, "tiny" | "small" | "medium") {
        return Err("requires a shipped-model canonical reference".into());
    }
    if mode == "public_stage_profile" && !cfg!(feature = "profiling") {
        return Err("public_stage_profile requires --features profiling".into());
    }
    let model_asset = |id: &str| {
        m["models"]
            .as_array()
            .unwrap()
            .iter()
            .find(|v| v["id"] == id)
            .ok_or("missing model")
    };
    let rec_asset = model_asset("rec")?;
    let dict_asset = model_asset("dictionary")?;
    let load_baseline = allocator::begin();
    let start = Instant::now();
    let model = Arc::new(
        Model::load_path(root.join(rec_asset["path"].as_str().unwrap()))
            .map_err(|e| e.to_string())?,
    );
    let dictionary = Arc::new(
        Dictionary::from_utf8(
            &fs::read(root.join(dict_asset["path"].as_str().unwrap()))
                .map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?,
    );
    let load_ms = start.elapsed().as_secs_f64() * 1000.;
    let load_allocations = allocator::end(load_baseline);
    let prepare_baseline = allocator::begin();
    let start = Instant::now();
    let compiled = Arc::new(CompiledModel::new(model).map_err(|e| e.to_string())?);
    let mut recognizer = if matches!(
        mode,
        "rec_public_owned" | "rec_public_borrowed" | "mixed_widths" | "owned_copies"
    ) {
        Some(
            Recognizer::new(
                compiled.clone(),
                dictionary.clone(),
                rec_options(&m)?,
                ExecutionLimits::default(),
            )
            .map_err(|e| e.to_string())?,
        )
    } else {
        None
    };
    let mut ocr = if mode.starts_with("ocr_") || mode == "public_stage_profile" {
        Some(make_ocr(&root, &m, compiled.clone(), dictionary)?)
    } else {
        None
    };
    let prepare_ms = start.elapsed().as_secs_f64() * 1000.;
    let prepare_allocations = allocator::end(prepare_baseline);
    let mut rows = Vec::new();
    let mut session_preparations = Vec::new();
    let mut checksum = 0.;
    let mut node_profile = Vec::new();
    #[cfg(feature = "profiling")]
    let mut stage_profile: Vec<Value> = Vec::new();
    #[cfg(not(feature = "profiling"))]
    let stage_profile: Vec<Value> = Vec::new();
    let mut shape_sequence = Vec::new();
    let mut graph_activation_peak = 0;
    let mut graph_retained_peak = 0;
    let mut graph_planning_peak = 0;
    if mode == "mixed_widths" {
        let recognizer = recognizer.as_mut().unwrap();
        let widths = [32, 64, 96, 160, 224, 320, 480, 576, 992, 32, 992];
        let mut sequence = Vec::new();
        let mut hasher = Sha256::new();
        for width in widths {
            let data = (0..width * 48 * 3)
                .map(|i| ((i * 17 + width) % 251) as u8)
                .collect::<Vec<_>>();
            hasher.update(&data);
            shape_sequence.push(vec![1, 3, 48, width]);
            sequence.push(Input {
                case: "mixed-widths".into(),
                id: "rec.pixels".into(),
                shape: vec![48, width, 3],
                sha256: String::new(),
                data,
                width,
                height: 48,
                stride: width * 3,
            });
        }
        // Warm both planning buffers and every high-water output capacity before the measured sequence.
        for _ in 0..2 {
            for input in &sequence {
                let r = recognizer
                    .recognize(input.image()?, true)
                    .map_err(|e| e.to_string())?;
                black_box(r.score);
            }
        }
        let identity = Input {
            case: "mixed-widths".into(),
            id: "rec.pixels".into(),
            shape: vec![],
            sha256: format!("{:x}", hasher.finalize()),
            data: vec![],
            width: 0,
            height: 0,
            stride: 0,
        };
        let (row, sum) = measure(&identity, mode, warmup, samples, true, || {
            let mut sum = 0.;
            for input in &sequence {
                let r = recognizer
                    .recognize(input.image()?, true)
                    .map_err(|e| e.to_string())?;
                sum += r.score as f64 + r.text.len() as f64;
            }
            Ok(sum)
        })?;
        rows.push(row);
        checksum += sum;
    } else {
        let corpus = inputs(&root, &m, mode, options.get("--case").map(String::as_str))?;
        for input in &corpus {
            let (row, sum) = if mode == "rec_graph" || mode == "node_profile" {
                let data = input
                    .data
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect::<Vec<_>>();
                let start = Instant::now();
                let mut session =
                    Session::new(compiled.clone(), &input.shape, ExecutionLimits::default())
                        .map_err(|e| e.to_string())?;
                graph_activation_peak = graph_activation_peak.max(session.activation_bytes());
                graph_retained_peak = graph_retained_peak.max(session.retained_activation_bytes());
                graph_planning_peak = graph_planning_peak.max(session.retained_planning_bytes());
                session_preparations.push(json!({"case_id":input.case,"artifact_id":input.id,"elapsed_ms":start.elapsed().as_secs_f64()*1000.,"activation_bytes":session.activation_bytes(),"retained_activation_bytes":session.retained_activation_bytes()}));
                if mode == "node_profile" {
                    let mut intervals = vec![0f64; compiled.node_count()];
                    let (row, sum) = measure(input, mode, warmup, samples, true, || {
                        let mut previous = Instant::now();
                        let r = session
                            .run_with_trace(&data, |node| {
                                let now = Instant::now();
                                intervals[node.index] += (now - previous).as_secs_f64() * 1000.;
                                previous = now;
                                Ok(())
                            })
                            .map_err(|e| e.to_string())?;
                        Ok(r.data()[0] as f64)
                    })?;
                    let divisor = (1 + warmup + samples) as f64;
                    node_profile.push(json!({"case_id":input.case,"artifact_id":input.id,"nodes":compiled.model().graph().nodes.iter().enumerate().map(|(i,n)|json!({"index":i,"key":n.key,"op_type":n.op_type,"mean_ms":intervals[i]/divisor})).collect::<Vec<_>>()}));
                    (row, sum)
                } else {
                    measure(input, mode, warmup, samples, true, || {
                        Ok(session.run(&data).map_err(|e| e.to_string())?.data()[0] as f64)
                    })?
                }
            } else if mode == "public_stage_profile" {
                #[cfg(feature = "profiling")]
                {
                    let engine = ocr.as_mut().unwrap();
                    let mut totals = simd_paddle_ocr_rs::StageProfile::default();
                    let mut calls = 0usize;
                    let measured = measure(input, mode, warmup, samples, false, || {
                        let (result, profile) = engine
                            .run_profiled(input.image()?, true)
                            .map_err(|e| e.to_string())?;
                        if calls > warmup {
                            totals += profile;
                        }
                        calls += 1;
                        Ok(result.len() as f64 + result.text.len() as f64)
                    })?;
                    stage_profile.push(json!({"case_id":input.case,"artifact_id":input.id,"samples":samples,"totals":totals}));
                    measured
                }
                #[cfg(not(feature = "profiling"))]
                {
                    return Err("profiling feature is required".into());
                }
            } else if mode.starts_with("ocr_") {
                let engine = ocr.as_mut().unwrap();
                measure(input, mode, warmup, samples, mode == "ocr_borrowed", || {
                    let r = engine
                        .run(input.image()?, true)
                        .map_err(|e| e.to_string())?;
                    if mode == "ocr_owned" {
                        let owned = r.try_to_owned().map_err(|e| e.to_string())?;
                        let v = owned.as_view();
                        Ok(v.len() as f64 + v.text.len() as f64)
                    } else {
                        Ok(r.len() as f64 + r.text.len() as f64)
                    }
                })?
            } else if mode == "owned_copies" {
                let r = recognizer
                    .as_mut()
                    .unwrap()
                    .recognize(input.image()?, true)
                    .map_err(|e| e.to_string())?;
                measure(input, mode, warmup, samples, false, || {
                    let owned = r.try_to_owned().map_err(|e| e.to_string())?;
                    Ok(owned.as_view().text.len() as f64)
                })?
            } else {
                measure(
                    input,
                    mode,
                    warmup,
                    samples,
                    mode == "rec_public_borrowed",
                    || {
                        let r = recognizer
                            .as_mut()
                            .unwrap()
                            .recognize(input.image()?, true)
                            .map_err(|e| e.to_string())?;
                        if mode == "rec_public_owned" {
                            let owned = r.try_to_owned().map_err(|e| e.to_string())?;
                            Ok(owned.as_view().score as f64 + owned.as_view().text.len() as f64)
                        } else {
                            Ok(r.score as f64 + r.text.len() as f64)
                        }
                    },
                )?
            };
            rows.push(row);
            checksum += sum;
        }
    }
    let activation_bytes = ocr.as_ref().map_or_else(
        || {
            recognizer
                .as_ref()
                .map_or(graph_activation_peak, Recognizer::activation_bytes)
        },
        Ocr::activation_bytes,
    );
    let retained_activation_bytes = ocr.as_ref().map_or_else(
        || {
            recognizer
                .as_ref()
                .map_or(graph_retained_peak, Recognizer::retained_activation_bytes)
        },
        Ocr::retained_activation_bytes,
    );
    let workspace_bytes = ocr.as_ref().map_or_else(
        || {
            recognizer
                .as_ref()
                .map_or(graph_planning_peak, Recognizer::retained_workspace_bytes)
        },
        Ocr::retained_workspace_bytes,
    );
    let corpus_samples = (0..samples)
        .map(|i| rows.iter().map(|r| r.samples_ms[i]).sum::<f64>())
        .collect::<Vec<_>>();
    let executable_sha256 = format!(
        "{:x}",
        Sha256::digest(
            fs::read(env::current_exe().map_err(|e| e.to_string())?).map_err(|e| e.to_string())?
        )
    );
    println!("{}",serde_json::to_string(&json!({"schema_version":2,"kind":"ocr_benchmark","model_family":family,"runtime":"rust","profile":"canonical","mode":mode,"model_sha256":rec_asset["sha256"],"det_model_sha256":if mode.starts_with("ocr_") || mode == "public_stage_profile" {Some(model_asset("det")?["sha256"].clone())}else{None},"cls_model_sha256":if mode.starts_with("ocr_") || mode == "public_stage_profile" {Some(model_asset("cls")?["sha256"].clone())}else{None},"dictionary_sha256":dict_asset["sha256"],"manifest_sha256":format!("{:x}",Sha256::digest(&manifest_bytes)),"executable_sha256":executable_sha256,"warmup":warmup,"samples":samples,"load_ms":load_ms,"load_allocations":load_allocations,"prepare_ms":prepare_ms,"prepare_allocations":prepare_allocations,"activation_bytes":activation_bytes,"retained_activation_bytes":retained_activation_bytes,"workspace_bytes":workspace_bytes,"session_preparations":session_preparations,"inputs":rows,"shape_sequence":shape_sequence,"node_profile":node_profile,"stage_profile":stage_profile,"checksum":checksum,"instrumented":cfg!(feature="heap-profiler"),"measurement":"Per-row contiguous blocks, sum matching sample indexes; graphs include input copy. Public owned includes owned conversion and drop; borrowed and diagnostic modes are separate. Mixed first-row call follows two prewarm cycles. Graph memory counters are peaks of one sequentially live full session, not sums; C# retains all graph sessions and its RSS is not a paired memory scope. Cold loading/preparation boundaries differ by engine.","corpus":{"p50_ms":percentile(&corpus_samples,0.5),"p95_ms":percentile(&corpus_samples,0.95),"mean_ms":corpus_samples.iter().sum::<f64>()/samples as f64,"samples_ms":corpus_samples}})).map_err(|e|e.to_string())?);
    Ok(())
}
fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("OCR benchmark failed: {e}");
            ExitCode::from(2)
        }
    }
}
