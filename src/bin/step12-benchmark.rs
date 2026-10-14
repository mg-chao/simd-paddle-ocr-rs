//! Matched scalar graph, stage, pipeline and shape-change measurements.
#[path = "support/allocator.rs"]
mod allocator;
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use simd_paddle_ocr_rs::{
    crop::{CropTransform, rotate180},
    detector::{DetectionBox, Detector, DetectorOptions},
    image::{ImageView, PixelFormat},
    lines::{LineOptions, OrientedLineExtractor},
    model::Model,
    preprocess::{self, ResizeWorkspace},
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

const MIXED_SHAPES: [[usize; 2]; 14] = [
    [32, 64],
    [64, 32],
    [64, 128],
    [128, 64],
    [32, 64],
    [64, 128],
    [32, 32],
    [32, 96],
    [96, 32],
    [64, 64],
    [64, 96],
    [96, 64],
    [96, 96],
    [128, 32],
];
#[derive(Clone)]
struct Frozen {
    id: String,
    shape: Vec<usize>,
    sha: String,
    values: Vec<f32>,
}
struct Frame {
    id: String,
    pixels: Vec<u8>,
    width: usize,
    height: usize,
    stride: usize,
    sha: String,
    det: Frozen,
    probabilities: Vec<f32>,
    boxes: Vec<DetectionBox>,
    width_ratio: f32,
    height_ratio: f32,
    crops: Vec<(String, Vec<u8>, usize, usize, String)>,
    cls: Vec<Frozen>,
    artifact_metadata: std::collections::HashMap<String, (Vec<usize>, String)>,
}
impl Frame {
    fn image(&self) -> Result<ImageView<'_>, String> {
        ImageView::new(
            &self.pixels,
            self.width,
            self.height,
            self.stride,
            PixelFormat::Bgr24,
        )
        .map_err(|e| e.to_string())
    }
    fn measurement(&self, id: &str, stage: &str) -> Result<Measurement, String> {
        let (shape, sha) = self
            .artifact_metadata
            .get(id)
            .ok_or_else(|| format!("missing stage identity {id}"))?;
        Ok(row(&self.id, id, shape, sha, stage))
    }
}
#[derive(Serialize)]
struct Measurement {
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
#[derive(Serialize)]
struct SessionPreparation {
    case_id: String,
    stage: String,
    elapsed_ms: f64,
    allocations: allocator::Counts,
    activation_bytes: usize,
    retained_activation_bytes: usize,
}
fn prepare_session(
    compiled: Arc<CompiledModel>,
    shape: &[usize],
    case: &str,
    stage: &str,
    preparations: &mut Vec<SessionPreparation>,
) -> Result<Session, String> {
    let baseline = allocator::begin();
    let start = Instant::now();
    let session =
        Session::new(compiled, shape, ExecutionLimits::default()).map_err(|e| e.to_string())?;
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.;
    let allocations = allocator::end(baseline);
    preparations.push(SessionPreparation {
        case_id: case.into(),
        stage: stage.into(),
        elapsed_ms,
        allocations,
        activation_bytes: session.activation_bytes(),
        retained_activation_bytes: session.retained_activation_bytes(),
    });
    Ok(session)
}
fn percentile(v: &[f64], q: f64) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let p = (s.len() - 1) as f64 * q;
    let a = p.floor() as usize;
    let b = p.ceil() as usize;
    s[a] + (s[b] - s[a]) * (p - a as f64)
}
fn measure<F>(
    mut row: Measurement,
    warmup: usize,
    samples: usize,
    zero_alloc: bool,
    mut call: F,
) -> Result<(Measurement, f64), String>
where
    F: FnMut() -> Result<f64, String>,
{
    let first_baseline = allocator::begin();
    let start = Instant::now();
    let mut checksum = call()?;
    row.first_call_ms = start.elapsed().as_secs_f64() * 1000.;
    row.first_call_allocations = allocator::end(first_baseline);
    for _ in 0..warmup {
        checksum += call()?;
    }
    row.samples_ms.resize(samples, 0.);
    let baseline = allocator::begin();
    for elapsed in &mut row.samples_ms {
        let start = Instant::now();
        checksum += black_box(call()?);
        *elapsed = start.elapsed().as_secs_f64() * 1000.;
    }
    row.counts = allocator::end(baseline);
    if zero_alloc && row.counts.warm_allocation_count.is_some_and(|n| n != 0) {
        return Err(format!(
            "{} {} warm calls allocated {} times / {} bytes",
            row.case_id,
            row.stage,
            row.counts.warm_allocation_count.unwrap(),
            row.counts.warm_allocation_bytes.unwrap()
        ));
    }
    row.p50_ms = percentile(&row.samples_ms, 0.5);
    row.p95_ms = percentile(&row.samples_ms, 0.95);
    row.mean_ms = row.samples_ms.iter().sum::<f64>() / samples as f64;
    Ok((row, checksum))
}
fn row(case: &str, id: &str, shape: &[usize], sha: &str, stage: &str) -> Measurement {
    Measurement {
        case_id: case.into(),
        artifact_id: id.into(),
        shape: shape.into(),
        sha256: sha.into(),
        stage: stage.into(),
        first_call_ms: 0.,
        first_call_allocations: allocator::Counts::default(),
        samples_ms: vec![],
        p50_ms: 0.,
        p95_ms: 0.,
        mean_ms: 0.,
        counts: allocator::Counts::default(),
    }
}
fn json(path: &Path) -> Result<Value, String> {
    serde_json::from_slice(&fs::read(path).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}
fn number(v: &Value, key: &str) -> Result<usize, String> {
    v[key]
        .as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| format!("missing {key}"))
}
fn artifact<'a>(case: &'a Value, id: &str) -> Result<&'a Value, String> {
    case["artifacts"]
        .as_array()
        .ok_or("missing artifacts")?
        .iter()
        .find(|a| a["id"] == id)
        .ok_or_else(|| format!("missing {id}"))
}
fn bytes(root: &Path, a: &Value) -> Result<Vec<u8>, String> {
    fs::read(root.join(a["path"].as_str().ok_or("missing path")?)).map_err(|e| e.to_string())
}
fn floats(root: &Path, a: &Value) -> Result<Frozen, String> {
    let raw = bytes(root, a)?;
    if raw.len() % 4 != 0 {
        return Err("float payload length".into());
    }
    Ok(Frozen {
        id: a["id"].as_str().ok_or("missing id")?.into(),
        shape: a["shape"]
            .as_array()
            .ok_or("missing shape")?
            .iter()
            .map(|n| {
                n.as_u64()
                    .map(|n| n as usize)
                    .ok_or_else(|| "bad shape".into())
            })
            .collect::<Result<_, String>>()?,
        sha: a["sha256"].as_str().ok_or("missing hash")?.into(),
        values: raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect(),
    })
}
fn frames(root: &Path, manifest: &Value, selected: Option<&str>) -> Result<Vec<Frame>, String> {
    let mut result = vec![];
    for case in manifest["cases"].as_array().ok_or("missing cases")? {
        let id = case["id"].as_str().ok_or("missing case id")?;
        if selected.is_some_and(|s| s != id) {
            continue;
        }
        let meta = json(&root.join(artifact(case, "input.metadata")?["path"].as_str().unwrap()))?;
        let input = artifact(case, "input.pixels")?;
        let det = floats(root, artifact(case, "det.input")?)?;
        let probabilities = floats(root, artifact(case, "det.probabilities")?)?.values;
        let box_json = json(&root.join(artifact(case, "det.boxes")?["path"].as_str().unwrap()))?;
        let mut boxes = vec![];
        for b in box_json["boxes"].as_array().ok_or("missing boxes")? {
            let mut points = [[0f32; 2]; 4];
            for (i, p) in points.iter_mut().enumerate() {
                for (c, x) in p.iter_mut().enumerate() {
                    *x = b["points"][i][c].as_f64().ok_or("bad box")? as f32;
                }
            }
            boxes.push(DetectionBox {
                points,
                score: b["score"].as_f64().ok_or("bad box score")? as f32,
            });
        }
        let mut artifact_metadata = std::collections::HashMap::new();
        let mut crops = vec![];
        let mut cls = vec![];
        for a in case["artifacts"].as_array().unwrap() {
            let aid = a["id"].as_str().unwrap();
            if !aid.contains(".node.") {
                artifact_metadata.insert(
                    aid.to_string(),
                    (
                        a["shape"]
                            .as_array()
                            .ok_or("missing shape")?
                            .iter()
                            .map(|v| v.as_u64().map(|n| n as usize).ok_or("bad shape"))
                            .collect::<Result<_, _>>()?,
                        a["sha256"].as_str().ok_or("missing hash")?.to_string(),
                    ),
                );
            }
            if aid.starts_with("line.") && aid.ends_with(".crop") {
                let h = a["shape"][0].as_u64().unwrap() as usize;
                let w = a["shape"][1].as_u64().unwrap() as usize;
                crops.push((
                    aid.into(),
                    bytes(root, a)?,
                    w,
                    h,
                    a["sha256"].as_str().unwrap().into(),
                ));
            }
            if aid.starts_with("line.") && aid.ends_with(".cls.input") {
                cls.push(floats(root, a)?);
            }
        }
        result.push(Frame {
            id: id.into(),
            pixels: bytes(root, input)?,
            width: number(&meta, "width")?,
            height: number(&meta, "height")?,
            stride: number(&meta, "stride")?,
            sha: input["sha256"].as_str().unwrap().into(),
            det,
            probabilities,
            boxes,
            width_ratio: box_json["width_ratio"]
                .as_f64()
                .ok_or("missing width_ratio")? as f32,
            height_ratio: box_json["height_ratio"]
                .as_f64()
                .ok_or("missing height_ratio")? as f32,
            crops,
            cls,
            artifact_metadata,
        });
    }
    if result.is_empty() {
        return Err("selected benchmark corpus is empty".into());
    }
    Ok(result)
}
fn run() -> Result<(), String> {
    let mut root = None;
    let mut mode = "det_graph".to_string();
    let mut warmup = 30usize;
    let mut samples = 100usize;
    let mut selected = None;
    let mut args = env::args().skip(1);
    let mut seen = std::collections::HashSet::new();
    while let Some(flag) = args.next() {
        if !seen.insert(flag.clone()) {
            return Err(format!("duplicate {flag}"));
        }
        let value = args.next().ok_or("option needs a value")?;
        match flag.as_str() {
            "--reference" => root = Some(PathBuf::from(value)),
            "--mode" => mode = value,
            "--warmup" => warmup = value.parse().map_err(|_| "bad warmup")?,
            "--samples" => samples = value.parse().map_err(|_| "bad samples")?,
            "--case" => selected = Some(value),
            _ => return Err(format!("unknown {flag}")),
        }
    }
    if samples == 0 {
        return Err("samples must be positive".into());
    }
    let root = root.ok_or("--reference required")?;
    validate_bundle(&root).map_err(|e| e.to_string())?;
    let manifest = json(&root.join("manifest.json"))?;
    if manifest["contract"]["profile"] != "canonical" {
        return Err("requires canonical profile".into());
    }
    let frames = frames(&root, &manifest, selected.as_deref())?;
    let asset = |id: &str| {
        manifest["models"]
            .as_array()
            .unwrap()
            .iter()
            .find(|a| a["id"] == id)
            .ok_or_else(|| format!("missing model {id}"))
    };
    let det_asset = asset("det")?;
    let cls_asset = asset("cls")?;
    let load_baseline = allocator::begin();
    let start = Instant::now();
    let model = Arc::new(
        Model::load_path(root.join(det_asset["path"].as_str().unwrap()))
            .map_err(|e| e.to_string())?,
    );
    let cls_model = Arc::new(
        Model::load_path(root.join(cls_asset["path"].as_str().unwrap()))
            .map_err(|e| e.to_string())?,
    );
    let load_ms = start.elapsed().as_secs_f64() * 1000.;
    let load_allocations = allocator::end(load_baseline);
    let prepare_baseline = allocator::begin();
    let start = Instant::now();
    let compiled = Arc::new(CompiledModel::new(model).map_err(|e| e.to_string())?);
    let cls_compiled = Arc::new(CompiledModel::new(cls_model).map_err(|e| e.to_string())?);
    let configured = &manifest["contract"]["options"];
    let configured_det = &configured["detector"];
    let detector_options = DetectorOptions {
        limit_side_length: number(configured_det, "limit_side_length")?,
        max_candidates: number(configured_det, "max_candidates")?,
        bitmap_threshold: configured_det["bitmap_threshold"]
            .as_f64()
            .ok_or("bitmap threshold")? as f32,
        box_threshold: Some(
            configured_det["box_threshold"]
                .as_f64()
                .ok_or("box threshold")? as f32,
        ),
        unclip_ratio: configured_det["unclip_ratio"]
            .as_f64()
            .ok_or("unclip ratio")? as f32,
        use_dilation: configured_det["use_dilation"].as_bool().ok_or("dilation")?,
        max_image_pixels: number(configured_det, "max_image_pixels")?,
    };
    let line_options = LineOptions {
        classifier_threshold: configured["classifier_threshold"]
            .as_f64()
            .ok_or("classifier threshold")? as f32,
        max_crop_pixels: configured["max_crop_pixels"].as_u64().unwrap_or(16_000_000) as usize,
        ..Default::default()
    };
    let detector = Detector::new(
        compiled.clone(),
        detector_options,
        ExecutionLimits::default(),
    )
    .map_err(|e| e.to_string())?;
    let mut extractor =
        OrientedLineExtractor::new(detector, Some(cls_compiled.clone()), line_options)
            .map_err(|e| e.to_string())?;
    let prepare_ms = start.elapsed().as_secs_f64() * 1000.;
    let prepare_allocations = allocator::end(prepare_baseline);
    let mut measurements = vec![];
    let mut session_preparations = vec![];
    let mut checksum = 0.;
    let mut activation_bytes = 0;
    let mut retained_activation_bytes = 0;
    let mut reshape_samples_ms = Vec::new();
    if mode == "det_graph" || mode == "node_profile" {
        for frame in &frames {
            let mut session = prepare_session(
                compiled.clone(),
                &frame.det.shape,
                &frame.id,
                &mode,
                &mut session_preparations,
            )
            .map_err(|e| e.to_string())?;
            activation_bytes = activation_bytes.max(session.activation_bytes());
            retained_activation_bytes =
                retained_activation_bytes.max(session.retained_activation_bytes());
            if mode == "node_profile" {
                for _ in 0..warmup {
                    session.run(&frame.det.values).map_err(|e| e.to_string())?;
                }
                let mut traces = vec![vec![0.; compiled.node_count()]; samples];
                for trace in &mut traces {
                    let mut last = Instant::now();
                    session
                        .run_with_trace(&frame.det.values, |node| {
                            trace[node.index] = last.elapsed().as_secs_f64() * 1000.;
                            last = Instant::now();
                            Ok(())
                        })
                        .map_err(|e| e.to_string())?;
                }
                for (index, node) in compiled.model().graph().nodes.iter().enumerate() {
                    let times: Vec<_> = traces.iter().map(|trace| trace[index]).collect();
                    let mut r = row(
                        &frame.id,
                        &format!("det.node.{index:04}"),
                        &[],
                        &frame.det.sha,
                        &node.op_type,
                    );
                    r.p50_ms = percentile(&times, 0.5);
                    r.p95_ms = percentile(&times, 0.95);
                    r.mean_ms = times.iter().sum::<f64>() / samples as f64;
                    r.samples_ms = times;
                    measurements.push(r);
                }
                continue;
            }
            let (r, c) = measure(
                row(
                    &frame.id,
                    &frame.det.id,
                    &frame.det.shape,
                    &frame.det.sha,
                    "det_graph",
                ),
                warmup,
                samples,
                true,
                || {
                    let output = session.run(&frame.det.values).map_err(|e| e.to_string())?;
                    Ok(f64::from(black_box(output.data())[0]))
                },
            )?;
            checksum += c;
            activation_bytes = activation_bytes.max(session.activation_bytes());
            retained_activation_bytes =
                retained_activation_bytes.max(session.retained_activation_bytes());
            measurements.push(r);
        }
    } else if mode == "pipeline" {
        for frame in &frames {
            let image = frame.image()?;
            let (r, c) = measure(
                row(
                    &frame.id,
                    "input.pixels",
                    &[frame.height, frame.stride],
                    &frame.sha,
                    "pipeline",
                ),
                warmup,
                samples,
                true,
                || {
                    let result = extractor.extract(image).map_err(|e| e.to_string())?;
                    Ok(black_box(result.lines.len()) as f64
                        + f64::from(black_box(result.pixels).first().copied().unwrap_or(0))
                        + result
                            .lines
                            .iter()
                            .filter_map(|m| m.classification)
                            .map(|r| f64::from(r.score))
                            .sum::<f64>())
                },
            )?;
            checksum += c;
            measurements.push(r);
        }
    } else if mode == "owned_copies" {
        for frame in &frames {
            // Extract once outside the measured interval; only persistent-result
            // copying belongs to this independent diagnostic mode.
            let snapshot = extractor
                .extract(frame.image()?)
                .map_err(|e| e.to_string())?
                .try_to_owned()
                .map_err(|e| e.to_string())?;
            let (r, c) = measure(
                row(
                    &frame.id,
                    "input.pixels",
                    &[frame.height, frame.stride],
                    &frame.sha,
                    "owned_result_copy",
                ),
                warmup,
                samples,
                false,
                || {
                    let copy = snapshot
                        .as_view()
                        .try_to_owned()
                        .map_err(|e| e.to_string())?;
                    Ok(black_box(copy.lines.len()) as f64
                        + f64::from(black_box(&copy.pixels).first().copied().unwrap_or(0)))
                },
            )?;
            checksum += c;
            measurements.push(r);
        }
    } else if mode == "mixed_shapes" {
        let shapes = MIXED_SHAPES;
        let inputs: Vec<_> = shapes
            .iter()
            .map(|&[h, w]| {
                vec![0f32; 3 * h * w]
                    .into_iter()
                    .enumerate()
                    .map(|(i, _)| ((i * 17 % 251) as f32 - 125.) / 125.)
                    .collect::<Vec<_>>()
            })
            .collect();
        let mut digest = Sha256::new();
        for v in &inputs {
            for f in v {
                digest.update(f.to_le_bytes());
            }
        }
        let sha = format!("{:x}", digest.finalize());
        let mut session = prepare_session(
            compiled.clone(),
            &[1, 3, 32, 64],
            "mixed-shapes",
            "mixed_shapes",
            &mut session_preparations,
        )
        .map_err(|e| e.to_string())?;
        reshape_samples_ms.resize(samples, 0.);
        let mut call_index = 0usize;
        let (r, c) = measure(
            row("mixed-shapes", "det.input", &[], &sha, "mixed_shapes"),
            warmup,
            samples,
            false,
            || {
                let mut c = 0.;
                let mut reshape_ns = 0u128;
                for (&[h, w], values) in shapes.iter().zip(&inputs) {
                    let rebind_start = Instant::now();
                    session.reshape(&[1, 3, h, w]).map_err(|e| e.to_string())?;
                    reshape_ns += rebind_start.elapsed().as_nanos();
                    c += f64::from(
                        black_box(session.run(values).map_err(|e| e.to_string())?.data())[0],
                    );
                }
                if call_index > warmup {
                    reshape_samples_ms[call_index - warmup - 1] = reshape_ns as f64 / 1e6;
                }
                call_index += 1;
                Ok(c)
            },
        )?;
        checksum += c;
        activation_bytes = session.activation_bytes();
        retained_activation_bytes = session.retained_activation_bytes();
        measurements.push(r);
    } else if mode == "stages" {
        let mut detector = Detector::new(
            compiled.clone(),
            detector_options,
            ExecutionLimits::default(),
        )
        .map_err(|e| e.to_string())?;
        activation_bytes = detector.activation_bytes();
        retained_activation_bytes = detector.retained_activation_bytes();
        for frame in &frames {
            let image = frame.image()?;
            let mut padded = vec![];
            let source = if frame.width + frame.height < 64 {
                let w = frame.width.max(32);
                let h = frame.height.max(32);
                padded.resize(w * h * 3, 0);
                for y in 0..frame.height {
                    padded[y * w * 3..y * w * 3 + frame.width * 3].copy_from_slice(
                        &frame.pixels[y * frame.stride..y * frame.stride + frame.width * 3],
                    );
                }
                ImageView::new(&padded, w, h, 0, PixelFormat::Bgr24).map_err(|e| e.to_string())?
            } else {
                image
            };
            let h = frame.det.shape[2];
            let w = frame.det.shape[3];
            let mut tensor = vec![0.; 3 * w * h];
            let mut resize = ResizeWorkspace::default();
            let (r, c) = measure(
                frame.measurement("det.input", "det_preprocess")?,
                warmup,
                samples,
                true,
                || {
                    preprocess::det(source, w, h, &mut tensor, &mut resize)
                        .map_err(|e| e.to_string())?;
                    Ok(f64::from(black_box(&tensor)[0]))
                },
            )?;
            checksum += c;
            measurements.push(r);
            let (r, c) = measure(
                frame.measurement("det.probabilities", "db_postprocess")?,
                warmup,
                samples,
                true,
                || {
                    let v = detector
                        .postprocess(
                            &frame.probabilities,
                            w,
                            h,
                            image,
                            frame.width_ratio,
                            frame.height_ratio,
                        )
                        .map_err(|e| e.to_string())?;
                    Ok(black_box(v.boxes.len()) as f64)
                },
            )?;
            checksum += c;
            measurements.push(r);
            for (index, (id, pixels, cw, ch, sha)) in frame.crops.iter().enumerate() {
                let mut crop = vec![0; pixels.len()];
                let (r, c) = measure(
                    row(&frame.id, id, &[*ch, *cw, 3], sha, "crop"),
                    warmup,
                    samples,
                    true,
                    || {
                        CropTransform::new(&frame.boxes[index])
                            .map_err(|e| e.to_string())?
                            .extract_into(image, &mut crop)
                            .map_err(|e| e.to_string())?;
                        Ok(f64::from(black_box(&crop)[0]))
                    },
                )?;
                checksum += c;
                measurements.push(r);
                let ci = ImageView::new(pixels, *cw, *ch, 0, PixelFormat::Bgr24)
                    .map_err(|e| e.to_string())?;
                let mut input = vec![0.; 3 * 80 * 160];
                let (r, c) = measure(
                    frame.measurement(&id.replace(".crop", ".cls.input"), "cls_preprocess")?,
                    warmup,
                    samples,
                    true,
                    || {
                        let width = preprocess::cls(ci, &mut input, &mut resize)
                            .map_err(|e| e.to_string())?;
                        Ok(width as f64 + f64::from(black_box(&input)[0]))
                    },
                )?;
                checksum += c;
                measurements.push(r);
                let frozen_cls = frame
                    .cls
                    .get(index)
                    .ok_or("missing captured classifier input for crop")?;
                let mut cls_session = prepare_session(
                    cls_compiled.clone(),
                    &frozen_cls.shape,
                    &frame.id,
                    "cls_graph",
                    &mut session_preparations,
                )
                .map_err(|e| e.to_string())?;
                // The public head retains the same full binding as the dense
                // session; both are simultaneously live in this stage block.
                activation_bytes = activation_bytes
                    .max(detector.activation_bytes() + 2 * cls_session.activation_bytes());
                retained_activation_bytes = retained_activation_bytes.max(
                    detector.retained_activation_bytes()
                        + 2 * cls_session.retained_activation_bytes(),
                );
                let (r, c) = measure(
                    frame.measurement(
                        &frozen_cls.id.replace(".input", ".probabilities"),
                        "cls_graph",
                    )?,
                    warmup,
                    samples,
                    true,
                    || {
                        Ok(f64::from(
                            black_box(
                                cls_session
                                    .run(&frozen_cls.values)
                                    .map_err(|e| e.to_string())?
                                    .data(),
                            )[0],
                        ))
                    },
                )?;
                checksum += c;
                measurements.push(r);
                let public_prepare_baseline = allocator::begin();
                let public_prepare_start = Instant::now();
                let mut public_cls = simd_paddle_ocr_rs::step12::public_classifier_stage(
                    cls_compiled.clone(),
                    ExecutionLimits::default(),
                )
                .map_err(|e| e.to_string())?;
                let elapsed_ms = public_prepare_start.elapsed().as_secs_f64() * 1000.;
                let allocations = allocator::end(public_prepare_baseline);
                session_preparations.push(SessionPreparation {
                    case_id: frame.id.clone(),
                    stage: "cls_public".into(),
                    elapsed_ms,
                    allocations,
                    activation_bytes: cls_session.activation_bytes(),
                    retained_activation_bytes: cls_session.retained_activation_bytes(),
                });
                let (r, c) = measure(
                    frame.measurement(&id.replace(".crop", ".cls.public"), "cls_public")?,
                    warmup,
                    samples,
                    true,
                    || {
                        let result = public_cls(ci).map_err(|e| e.to_string())?;
                        Ok(black_box(result.label) as f64 + f64::from(result.score))
                    },
                )?;
                checksum += c;
                measurements.push(r);
                let (r, c) = measure(
                    frame.measurement(&id.replace(".crop", ".rec.pixels"), "rotate180")?,
                    warmup,
                    samples,
                    true,
                    || {
                        rotate180(&mut crop, *cw, *ch).map_err(|e| e.to_string())?;
                        Ok(f64::from(black_box(&crop)[0]))
                    },
                )?;
                checksum += c;
                measurements.push(r);
            }
        }
    } else {
        return Err(
            "mode must be det_graph, pipeline, stages, mixed_shapes, owned_copies or node_profile"
                .into(),
        );
    }
    activation_bytes +=
        extractor.detector_mut().activation_bytes() + extractor.classifier_activation_bytes();
    retained_activation_bytes += extractor.detector_mut().retained_activation_bytes()
        + extractor.classifier_retained_activation_bytes();
    let mut corpus = vec![0.; samples];
    for r in &measurements {
        for (i, &v) in r.samples_ms.iter().enumerate() {
            corpus[i] += v;
        }
    }
    let allocation_count = measurements
        .iter()
        .filter_map(|r| r.counts.warm_allocation_count)
        .sum::<usize>();
    let allocation_bytes = measurements
        .iter()
        .filter_map(|r| r.counts.warm_allocation_bytes)
        .sum::<usize>();
    let report = serde_json::json!({
        "schema_version": 1, "runtime": "rust", "profile": "canonical", "mode": mode,
        "shape_sequence": if mode == "mixed_shapes" { Some(MIXED_SHAPES) } else { None },
        "reshape_samples_ms": reshape_samples_ms,
        "reference": root,
        "manifest_sha256": format!("{:x}", Sha256::digest(fs::read(root.join("manifest.json")).map_err(|e| e.to_string())?)),
        "model_sha256": det_asset["sha256"], "cls_model_sha256": cls_asset["sha256"],
        "load_ms": load_ms, "prepare_ms": prepare_ms,
        "load_allocations": load_allocations, "prepare_allocations": prepare_allocations,
        "session_preparations": session_preparations,
        "preparation_semantics": "load covers model loading/compilation. prepare covers persistent extractor construction. session_preparations separately covers each additional measured graph or stage binding; source corpus decoding/validation is outside these phases.",
        "activation_memory_semantics": "Persistent extractor plus maximum simultaneously live additional benchmark sessions; excludes shared weights, source corpus, image scratch and timing metadata. Mixed-shape additional session counters describe the final binding and retained capacities. Public CLS retains the same full binding as its dense graph session.",
        "retained_workspace_bytes": extractor.retained_workspace_bytes(),
        "warmup": warmup, "samples": samples, "checksum": checksum,
        "activation_bytes": activation_bytes, "retained_activation_bytes": retained_activation_bytes,
        "crop_capacity_bytes": extractor.crop_capacity_bytes(), "line_capacity_bytes": extractor.line_capacity_bytes(),
        "warm_allocation_count": if cfg!(feature = "heap-profiler") { Some(allocation_count) } else { None },
        "warm_allocation_bytes": if cfg!(feature = "heap-profiler") { Some(allocation_bytes) } else { None },
        "instrumentation": if cfg!(feature = "heap-profiler") { "separate allocator-instrumented executable; exclude these timings" } else { "standard allocator; uninstrumented latency" },
        "measurement": "Per-case timed blocks; corpus samples sum matching sample indexes, not manifest-order wall sweeps. Input copying included in graph and mixed-shape calls. Pipeline preprocesses directly into session input; no tracing/hashing/file I/O in timed calls. Node profiling is a separate observer-instrumented mode. Owned-result copy mode includes dropping the copied result.",
        "inputs": measurements,
        "corpus": { "samples_ms": corpus, "p50_ms": percentile(&corpus, 0.5), "p95_ms": percentile(&corpus, 0.95), "mean_ms": corpus.iter().sum::<f64>() / samples as f64 }
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?
    );
    Ok(())
}
fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("Step12 benchmark failed: {e}");
            ExitCode::from(2)
        }
    }
}
