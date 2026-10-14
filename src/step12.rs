//! Independently execute frozen image pixels through complete oriented crops.
use crate::{
    Bundle, Result, ValidationReport,
    bundle::{ArtifactDescriptor, BundleWriter, Dtype},
    det::{self, ValidatedReference},
    detector::{DetectionBox, DetectionTrace, Detector, DetectorOptions},
    image::{ImageError, ImageView, PixelFormat},
    invalid,
    lines::{LineOptions, LineTrace, OrientedLineExtractor},
    model::Model,
    read_bounded,
    runtime::CompiledModel,
    safe_file,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    fmt,
    fs::File,
    io::Read,
    path::{Path, PathBuf},
    sync::Arc,
};

#[derive(Debug, Serialize)]
pub struct Step12RunReport {
    pub output: PathBuf,
    pub processed_cases: usize,
    pub processed_lines: usize,
    pub written_artifacts: usize,
    pub reference_manifest_sha256: String,
    pub reference_validation: ValidationReport,
}
impl fmt::Display for Step12RunReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "STEP12 CANDIDATE: {} cases, {} lines, {} artifacts; saved {}; scope frozen image pixels through oriented crops; no recognition",
            self.processed_cases,
            self.processed_lines,
            self.written_artifacts,
            self.output.display()
        )
    }
}
pub(crate) struct Selection {
    pub artifacts: HashSet<(usize, usize)>,
    pub lines: usize,
}
pub(crate) fn is_step12_artifact(id: &str) -> Result<bool> {
    if matches!(
        id,
        "input.metadata" | "input.pixels" | "det.boxes" | "step12.public"
    ) || det::is_detector_artifact(id)?
    {
        return Ok(true);
    }
    let fields: Vec<_> = id.split('.').collect();
    if fields.first() != Some(&"line") {
        return Ok(false);
    }
    if fields.len() < 3 || fields[1].is_empty() || !fields[1].bytes().all(|b| b.is_ascii_digit()) {
        return Err(invalid(format!("malformed line artifact {id:?}")));
    }
    if let [_, _, "cls", "node", ordinal] = fields.as_slice()
        && (ordinal.is_empty() || !ordinal.bytes().all(|byte| byte.is_ascii_digit()))
    {
        return Err(invalid(format!(
            "malformed classifier node artifact {id:?}"
        )));
    }
    Ok(matches!(
        fields.as_slice(),
        [_, _, "crop"]
            | [_, _, "rec", "pixels"]
            | [_, _, "cls", "input"]
            | [_, _, "cls", "probabilities"]
            | [_, _, "cls", "public"]
            | [_, _, "cls", "node", _]
    ))
}
pub(crate) fn selection(bundle: &Bundle) -> Result<Selection> {
    selection_impl(bundle, true)
}
fn selection_impl(bundle: &Bundle, verify_models: bool) -> Result<Selection> {
    det::require_canonical(&bundle.manifest)?;
    let contract = &bundle.manifest.contract;
    if contract["capture_scope"] != "step12"
        || contract["max_traced_lines"] != 0
        || contract["input_origin"] != "frozen_bgr_pixels"
    {
        return Err(invalid(
            "complete Step12 reference requires capture_scope=step12, max_traced_lines=0 and frozen_bgr_pixels origin",
        ));
    }
    let detector = det::detector_selection(&bundle.manifest)?;
    if detector.cases.len() != bundle.manifest.cases.len() {
        return Err(invalid("every Step12 case requires complete DET capture"));
    }
    if verify_models {
        det::validate_reference_coverage(bundle, &detector)?;
    }
    validate_declared_structures(bundle)?;
    let mut artifacts = HashSet::new();
    let mut lines = 0usize;
    for (ci, case) in bundle.manifest.cases.iter().enumerate() {
        let by_id: HashMap<_, _> = case.artifacts.iter().map(|a| (a.id.as_str(), a)).collect();
        for required in [
            "input.metadata",
            "input.pixels",
            "det.boxes",
            "step12.public",
        ] {
            if !by_id.contains_key(required) {
                return Err(invalid(format!("case {} missing {required}", case.id)));
            }
        }
        let public = read_json(bundle, by_id["step12.public"])?;
        let count = public["detected_count"]
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| invalid("invalid Step12 detected_count"))?;
        if public["lines"]
            .as_array()
            .is_none_or(|array| array.len() != count)
        {
            return Err(invalid("Step12 summary line count differs"));
        }
        for index in 0..count {
            let prefix = format!("line.{index:03}");
            for suffix in [
                "crop",
                "cls.input",
                "cls.probabilities",
                "cls.public",
                "rec.pixels",
            ] {
                if !by_id.contains_key(format!("{prefix}.{suffix}").as_str()) {
                    return Err(invalid(format!(
                        "case {} missing {prefix}.{suffix}",
                        case.id
                    )));
                }
            }
            if !by_id
                .keys()
                .any(|id| id.starts_with(&format!("{prefix}.cls.node.")))
            {
                return Err(invalid(format!(
                    "case {} missing CLS node capture",
                    case.id
                )));
            }
        }
        for (ai, artifact) in case.artifacts.iter().enumerate() {
            if is_step12_artifact(&artifact.id)? {
                if artifact.id.starts_with("line.") {
                    let index: usize = artifact
                        .id
                        .split('.')
                        .nth(1)
                        .unwrap()
                        .parse()
                        .map_err(|_| invalid("invalid line ordinal"))?;
                    if index >= count {
                        return Err(invalid("Step12 artifact line outside public count"));
                    }
                }
                artifacts.insert((ci, ai));
            }
        }
        lines = lines
            .checked_add(count)
            .ok_or_else(|| invalid("line count overflow"))?;
    }
    if lines != 0 {
        let classifier = crate::cls::classifier_selection(&bundle.manifest)?;
        if classifier.lines.len() != lines {
            return Err(invalid("Step12 classifier line coverage differs"));
        }
        let asset = bundle
            .manifest
            .models
            .iter()
            .find(|model| model.id == "cls")
            .ok_or_else(|| invalid("missing cls model"))?;
        let compiled = if verify_models {
            Some(
                CompiledModel::new(Arc::new(
                    Model::load_path(safe_file(&bundle.root, &asset.path)?)
                        .map_err(|e| invalid(e.to_string()))?,
                ))
                .map_err(|e| invalid(e.to_string()))?,
            )
        } else {
            None
        };
        for line in classifier.lines {
            let case = &bundle.manifest.cases[line.case_index];
            if case.artifacts[line.input_index].shape != [1, 3, 80, 160] {
                return Err(invalid("Step12 CLS input must be [1,3,80,160]"));
            }
            let probability = &case.artifacts[line.probabilities_index];
            if probability.shape != [1, 2]
                || probability.layout != "row_major"
                || probability.dtype != Dtype::F32
            {
                return Err(invalid(
                    "Step12 CLS probabilities must be [1,2] row_major f32",
                ));
            }
            if let Some(compiled) = &compiled {
                det::verify_inventory_bundle(bundle, compiled.model(), "cls", &line.nodes)?;
                det::validate_bound_metadata(
                    case,
                    &line.nodes,
                    compiled,
                    &case.artifacts[line.input_index].shape,
                )?;
            }
        }
    }
    Ok(Selection { artifacts, lines })
}
fn integer(value: &Value, field: &str, positive: bool) -> Result<u64> {
    let number = value[field].as_u64().ok_or_else(|| {
        invalid(format!(
            "Step12 {field} must be an exact nonnegative JSON integer"
        ))
    })?;
    if positive && number == 0 {
        return Err(invalid(format!("Step12 {field} must be positive")));
    }
    Ok(number)
}
fn classification(value: &Value, label_field: &str, rotation_field: &str) -> Result<()> {
    let label = integer(value, label_field, false)?;
    let rotation = integer(value, rotation_field, false)?;
    if label > 1 || !matches!(rotation, 0 | 180) {
        return Err(invalid("invalid Step12 classification label/rotation"));
    }
    if rotation_field == "orientation_degrees" && rotation != label * 180 {
        return Err(invalid("classification orientation differs from label"));
    }
    if label == 0 && rotation != 0 {
        return Err(invalid("class zero cannot apply a 180-degree rotation"));
    }
    Ok(())
}
fn score(value: &Value, field: &str) -> Result<()> {
    if value[field]
        .as_f64()
        .is_none_or(|n| !n.is_finite() || !(0.0..=1.0).contains(&n))
    {
        return Err(invalid(format!(
            "Step12 {field} must be a finite probability"
        )));
    }
    Ok(())
}
fn ratio(value: &Value, field: &str) -> Result<()> {
    if value[field]
        .as_f64()
        .is_none_or(|n| !n.is_finite() || n <= 0. || !(n as f32).is_finite())
    {
        return Err(invalid(format!(
            "Step12 {field} must be a finite positive ratio"
        )));
    }
    Ok(())
}
fn validate_box(value: &Value) -> Result<()> {
    let points = value["points"]
        .as_array()
        .ok_or_else(|| invalid("box requires points array"))?;
    if points.len() != 4
        || points.iter().any(|point| {
            point.as_array().is_none_or(|pair| {
                pair.len() != 2
                    || pair
                        .iter()
                        .any(|v| v.as_f64().is_none_or(|n| !n.is_finite()))
            })
        })
    {
        return Err(invalid("box requires four finite XY points"));
    }
    if value["score"]
        .as_f64()
        .is_none_or(|score| !score.is_finite())
    {
        return Err(invalid("box score must be finite"));
    }
    Ok(())
}
pub(crate) fn validate_declared_structures(bundle: &Bundle) -> Result<()> {
    for case in &bundle.manifest.cases {
        for artifact in &case.artifacts {
            if artifact.dtype != Dtype::Json {
                continue;
            }
            match artifact.id.as_str() {
                "input.metadata" => {
                    let value = read_json(bundle, artifact)?;
                    let width = integer(&value, "width", true)?;
                    integer(&value, "height", true)?;
                    let stride = integer(&value, "stride", true)?;
                    if width.checked_mul(3).is_none_or(|bytes| stride < bytes)
                        || value["pixel_format"] != "bgr24"
                    {
                        return Err(invalid("invalid Step12 source pixels"));
                    }
                }
                "det.boxes" => {
                    let value = read_json(bundle, artifact)?;
                    integer(&value, "resized_width", true)?;
                    integer(&value, "resized_height", true)?;
                    ratio(&value, "width_ratio")?;
                    ratio(&value, "height_ratio")?;
                    let boxes = value["boxes"]
                        .as_array()
                        .ok_or_else(|| invalid("det.boxes requires boxes array"))?;
                    for detection in boxes {
                        validate_box(detection)?;
                    }
                }
                "step12.public" => {
                    let value = read_json(bundle, artifact)?;
                    let count = integer(&value, "detected_count", false)?;
                    integer(&value, "detector_resized_width", true)?;
                    integer(&value, "detector_resized_height", true)?;
                    ratio(&value, "width_ratio")?;
                    ratio(&value, "height_ratio")?;
                    let lines = value["lines"]
                        .as_array()
                        .ok_or_else(|| invalid("Step12 lines must be an array"))?;
                    if lines.len() as u64 != count {
                        return Err(invalid("Step12 structural line count differs"));
                    }
                    for line in lines {
                        classification(line, "classification_label", "applied_rotation_degrees")?;
                        score(line, "classification_score")?;
                        integer(line, "width", true)?;
                        integer(line, "height", true)?;
                        validate_box(&line["box"])?;
                    }
                }
                id if id.starts_with("line.") && id.ends_with(".cls.public") => {
                    let value = read_json(bundle, artifact)?;
                    classification(&value, "label", "orientation_degrees")?;
                    score(&value, "score")?;
                    integer(&value, "resized_width", true)?;
                    integer(&value, "dense_content_width", true)?;
                }
                _ => {}
            }
        }
    }
    Ok(())
}
pub(crate) fn read_json(bundle: &Bundle, artifact: &ArtifactDescriptor) -> Result<Value> {
    if artifact.dtype != Dtype::Json {
        return Err(invalid(format!("{} must be JSON", artifact.id)));
    }
    let bytes = read_bounded(
        File::open(safe_file(&bundle.root, &artifact.path)?).map_err(|e| invalid(e.to_string()))?,
        &artifact.id,
    )?;
    serde_json::from_slice(&bytes).map_err(|e| invalid(e.to_string()))
}
#[derive(Deserialize)]
pub struct InputMetadata {
    pub width: usize,
    pub height: usize,
    pub stride: usize,
    pub pixel_format: String,
}
#[derive(Debug)]
pub struct FrozenImage {
    pub case_id: String,
    pub artifact_id: String,
    pub sha256: String,
    pub width: usize,
    pub height: usize,
    pub stride: usize,
    pub pixels: Vec<u8>,
}
impl FrozenImage {
    pub fn image(&self) -> crate::image::ImageResult<ImageView<'_>> {
        ImageView::new(
            &self.pixels,
            self.width,
            self.height,
            self.stride,
            PixelFormat::Bgr24,
        )
    }
}
pub struct FrozenImages {
    pub detector_model_path: PathBuf,
    pub classifier_model_path: PathBuf,
    pub model_sha256: String,
    pub cls_model_sha256: String,
    pub manifest_sha256: String,
    pub contract: Value,
    pub images: Vec<FrozenImage>,
}
pub fn load_images(directory: &Path) -> Result<FrozenImages> {
    let reference = ValidatedReference::load(directory)?;
    selection(&reference.bundle)?;
    load_images_from_reference(&reference)
}
fn load_images_from_reference(reference: &ValidatedReference) -> Result<FrozenImages> {
    let mut images = Vec::new();
    for case in &reference.bundle.manifest.cases {
        let metadata = case
            .artifacts
            .iter()
            .find(|a| a.id == "input.metadata")
            .unwrap();
        let input: InputMetadata = serde_json::from_value(read_json(&reference.bundle, metadata)?)
            .map_err(|e| invalid(e.to_string()))?;
        if input.pixel_format != "bgr24" {
            return Err(invalid("Step12 input must be BGR24"));
        }
        let limit = reference.bundle.manifest.contract["options"]["detector"]["max_image_pixels"]
            .as_u64()
            .unwrap_or(40_000_000);
        if input.width > i32::MAX as usize
            || input.height > i32::MAX as usize
            || (input.width as u64)
                .checked_mul(input.height as u64)
                .is_none_or(|pixels| pixels > limit)
        {
            return Err(invalid(
                "Step12 image exceeds configured dimensions/pixel limit",
            ));
        }
        if input.stride
            < input
                .width
                .checked_mul(3)
                .ok_or_else(|| invalid("source row byte overflow"))?
        {
            return Err(invalid("Step12 source stride is too small"));
        }
        let artifact = case
            .artifacts
            .iter()
            .find(|a| a.id == "input.pixels")
            .unwrap();
        if artifact.dtype != Dtype::U8
            || artifact.layout != "bgr24_rows"
            || artifact.shape != [input.height, input.stride]
        {
            return Err(invalid("Step12 input pixel metadata differs"));
        }
        let length = usize::try_from(artifact.bytes)
            .map_err(|_| invalid("source pixels exceed addressable buffer size"))?;
        if length > i32::MAX as usize {
            return Err(invalid(
                "frozen source pixel buffer exceeds the C# I32 buffer contract",
            ));
        }
        let mut pixels = Vec::new();
        pixels
            .try_reserve_exact(length)
            .map_err(|e| invalid(format!("source pixel allocation failed: {e}")))?;
        pixels.resize(length, 0);
        File::open(safe_file(&reference.bundle.root, &artifact.path)?)
            .map_err(|e| invalid(e.to_string()))?
            .read_exact(&mut pixels)
            .map_err(|e| invalid(e.to_string()))?;
        let image = FrozenImage {
            case_id: case.id.clone(),
            artifact_id: artifact.id.clone(),
            sha256: artifact.sha256.clone(),
            width: input.width,
            height: input.height,
            stride: input.stride,
            pixels,
        };
        image.image().map_err(|e| invalid(e.to_string()))?;
        images.push(image);
    }
    let asset = |id: &str| {
        reference
            .bundle
            .manifest
            .models
            .iter()
            .find(|a| a.id == id)
            .ok_or_else(|| invalid(format!("missing model {id}")))
    };
    Ok(FrozenImages {
        detector_model_path: reference.model_path("det")?,
        classifier_model_path: reference.model_path("cls")?,
        model_sha256: asset("det")?.sha256.clone(),
        cls_model_sha256: asset("cls")?.sha256.clone(),
        manifest_sha256: reference.manifest_sha256.clone(),
        contract: reference.bundle.manifest.contract.clone(),
        images,
    })
}
pub fn box_json(detection: &DetectionBox) -> Value {
    json!({"points":detection.points,"score":detection.score})
}
/// Benchmark adapter retaining one private classifier and its reusable workspace.
#[doc(hidden)]
pub fn public_classifier_stage(
    compiled: Arc<CompiledModel>,
    limits: crate::runtime::ExecutionLimits,
) -> crate::image::ImageResult<
    impl for<'a> FnMut(
        ImageView<'a>,
    ) -> crate::image::ImageResult<crate::orientation::ClassificationResult>,
> {
    let mut classifier = crate::orientation::Classifier::new(compiled, limits)?;
    Ok(move |image: ImageView<'_>| classifier.classify(image))
}
pub fn build_extractor(corpus: &FrozenImages) -> Result<OrientedLineExtractor> {
    let det = Arc::new(
        CompiledModel::new(Arc::new(
            Model::load_path(&corpus.detector_model_path).map_err(|e| invalid(e.to_string()))?,
        ))
        .map_err(|e| invalid(e.to_string()))?,
    );
    let cls = Arc::new(
        CompiledModel::new(Arc::new(
            Model::load_path(&corpus.classifier_model_path).map_err(|e| invalid(e.to_string()))?,
        ))
        .map_err(|e| invalid(e.to_string()))?,
    );
    extractor_from_models(corpus, det, cls)
}
fn extractor_from_models(
    corpus: &FrozenImages,
    det: Arc<CompiledModel>,
    cls: Arc<CompiledModel>,
) -> Result<OrientedLineExtractor> {
    let value = &corpus.contract["options"]["detector"];
    let options = DetectorOptions {
        limit_side_length: value["limit_side_length"].as_u64().unwrap_or(960) as usize,
        max_candidates: value["max_candidates"].as_u64().unwrap_or(3000) as usize,
        bitmap_threshold: value["bitmap_threshold"].as_f64().unwrap_or(0.2) as f32,
        box_threshold: Some(value["box_threshold"].as_f64().unwrap_or(
            if corpus.contract["model"] == "tiny" {
                0.4
            } else {
                0.45
            },
        ) as f32),
        unclip_ratio: value["unclip_ratio"].as_f64().unwrap_or(1.4) as f32,
        use_dilation: value["use_dilation"].as_bool().unwrap_or(false),
        max_image_pixels: value["max_image_pixels"].as_u64().unwrap_or(40_000_000) as usize,
    };
    let detector = Detector::new(det, options, crate::runtime::ExecutionLimits::default())
        .map_err(|e| invalid(e.to_string()))?;
    let line_options = LineOptions {
        classifier_threshold: corpus.contract["options"]["classifier_threshold"]
            .as_f64()
            .unwrap_or(0.) as f32,
        max_crop_pixels: corpus.contract["options"]["max_crop_pixels"]
            .as_u64()
            .unwrap_or(16_000_000) as usize,
        ..Default::default()
    };
    OrientedLineExtractor::new(detector, Some(cls), line_options)
        .map_err(|e| invalid(e.to_string()))
}
pub fn run_step12(reference_directory: &Path, output_directory: &Path) -> Result<Step12RunReport> {
    let reference = ValidatedReference::load(reference_directory)?;
    selection_impl(&reference.bundle, false)?;
    let corpus = load_images_from_reference(&reference)?;
    let det_model = Arc::new(
        Model::load_path(reference.model_path("det")?).map_err(|e| invalid(e.to_string()))?,
    );
    let cls_model = Arc::new(
        Model::load_path(reference.model_path("cls")?).map_err(|e| invalid(e.to_string()))?,
    );
    let compiled_det = Arc::new(CompiledModel::new(det_model).map_err(|e| invalid(e.to_string()))?);
    let compiled_cls = Arc::new(CompiledModel::new(cls_model).map_err(|e| invalid(e.to_string()))?);
    for (id, compiled) in [("det", &compiled_det), ("cls", &compiled_cls)] {
        for case in &reference.bundle.manifest.cases {
            let mut groups: HashMap<String, Vec<(usize, usize)>> = HashMap::new();
            for (ai, artifact) in case.artifacts.iter().enumerate() {
                let prefix = if id == "det" {
                    "det.node.".to_string()
                } else {
                    artifact
                        .id
                        .split(".node.")
                        .next()
                        .filter(|s| s.ends_with(".cls"))
                        .map(|s| format!("{s}.node."))
                        .unwrap_or_default()
                };
                if !prefix.is_empty()
                    && let Some(ordinal) = artifact.id.strip_prefix(&prefix)
                {
                    groups.entry(prefix).or_default().push((
                        ai,
                        ordinal
                            .parse()
                            .map_err(|_| invalid("invalid node ordinal"))?,
                    ));
                }
            }
            for (prefix, nodes) in &groups {
                det::verify_inventory(&reference, compiled.model(), id, nodes)?;
                let input_id = format!("{}input", prefix.strip_suffix("node.").unwrap());
                let input = case
                    .artifacts
                    .iter()
                    .find(|artifact| artifact.id == input_id)
                    .ok_or_else(|| invalid("trace group input missing"))?;
                det::validate_bound_metadata(case, nodes, compiled, &input.shape)?;
            }
        }
    }
    let mut extractor = extractor_from_models(&corpus, compiled_det, compiled_cls)?;
    let mut writer =
        BundleWriter::new(output_directory, reference.bundle.manifest.contract.clone())?;
    for asset in &reference.bundle.manifest.models {
        writer.copy_model(
            &asset.id,
            &safe_file(&reference.bundle.root, &asset.path)?,
            &asset.path,
        )?;
    }
    let mut written_artifacts = 0;
    let mut processed_lines = 0;
    for (ci, case) in reference.bundle.manifest.cases.iter().enumerate() {
        let image = &corpus.images[ci];
        let descriptors: HashMap<_, _> =
            case.artifacts.iter().map(|a| (a.id.as_str(), a)).collect();
        let mut artifacts = Vec::new();
        let input_meta = descriptors["input.metadata"];
        artifacts.push(writer.write_json(
            &input_meta.id,
            &input_meta.path,
            &read_json(&reference.bundle, input_meta)?,
        )?);
        let pixels = descriptors["input.pixels"];
        artifacts.push(writer.write_bytes(
            &pixels.id,
            &pixels.path,
            &image.pixels,
            &pixels.shape,
            &pixels.layout,
        )?);
        let cell = RefCell::new((&mut writer, &mut artifacts));
        let dense_widths = RefCell::new(HashMap::new());
        let write_float = |id: String,
                           values: &[f32],
                           shape: &[usize],
                           layout: &str|
         -> crate::image::ImageResult<()> {
            if let Some(descriptor) = descriptors.get(id.as_str()) {
                let mut pair = cell.borrow_mut();
                let artifact = pair
                    .0
                    .write_floats(&id, &descriptor.path, values, shape, layout)
                    .map_err(|e| ImageError::new(e.to_string()))?;
                pair.1.push(artifact);
            }
            Ok(())
        };
        let write_json = |id: String, value: Value| -> crate::image::ImageResult<()> {
            if let Some(descriptor) = descriptors.get(id.as_str()) {
                let mut pair = cell.borrow_mut();
                let artifact = pair
                    .0
                    .write_json(&id, &descriptor.path, &value)
                    .map_err(|e| ImageError::new(e.to_string()))?;
                pair.1.push(artifact);
            }
            Ok(())
        };
        let write_image = |id: String, image: ImageView<'_>| -> crate::image::ImageResult<()> {
            if let Some(descriptor) = descriptors.get(id.as_str()) {
                let mut pair = cell.borrow_mut();
                let artifact = pair
                    .0
                    .write_bytes(
                        &id,
                        &descriptor.path,
                        image.pixels(),
                        &[image.height(), image.width(), 3],
                        "bgr24",
                    )
                    .map_err(|e| ImageError::new(e.to_string()))?;
                pair.1.push(artifact);
            }
            Ok(())
        };
        let output=extractor.extract_with_trace(image.image().map_err(|e|invalid(e.to_string()))?,|trace|match trace {
            DetectionTrace::Input{shape,values}=>write_float("det.input".into(),values,shape.dimensions(),"nchw"),
            DetectionTrace::Node(node)=>write_float(format!("det.node.{:04}",node.index),node.values,node.shape.dimensions(),node.layout),
            DetectionTrace::Probabilities{shape,values}=>write_float("det.probabilities".into(),values,shape.dimensions(),"nchw"),
            DetectionTrace::Boxes(view)=>write_json("det.boxes".into(),json!({"resized_width":view.resized_width,"resized_height":view.resized_height,"width_ratio":view.width_ratio,"height_ratio":view.height_ratio,"boxes":view.boxes.iter().map(box_json).collect::<Vec<_>>()})),
        },|trace|match trace {
            LineTrace::Crop{index,image}=>write_image(format!("line.{index:03}.crop"),image),
            LineTrace::ClsInput{index,values,resized_width}=>{dense_widths.borrow_mut().insert(index,resized_width);write_float(format!("line.{index:03}.cls.input"),values,&[1,3,80,160],"nchw")},
            LineTrace::ClsNode{index,node}=>write_float(format!("line.{index:03}.cls.node.{:04}",node.index),node.values,node.shape.dimensions(),node.layout),
            LineTrace::ClsProbabilities{index,values}=>write_float(format!("line.{index:03}.cls.probabilities"),values,&[1,2],"row_major"),
            LineTrace::Classification{index,result}=>write_json(format!("line.{index:03}.cls.public"),json!({"label":result.label,"score":result.score,"resized_width":result.resized_width,"orientation_degrees":result.orientation_degrees(),"dense_content_width":dense_widths.borrow()[&index]})),
            LineTrace::OrientedCrop{index,image,..}=>write_image(format!("line.{index:03}.rec.pixels"),image),
        }).map_err(|e|invalid(e.to_string()))?;
        let public = descriptors["step12.public"];
        let lines:Vec<_>=output.lines.iter().map(|line|{let cls=line.classification.unwrap();json!({"box":box_json(&line.detection),"classification_score":cls.score,"classification_label":cls.label,"applied_rotation_degrees":line.applied_rotation_degrees,"width":line.width,"height":line.height})}).collect();
        artifacts.push(writer.write_json(&public.id,&public.path,&json!({"detected_count":output.lines.len(),"detector_resized_width":output.resized_width,"detector_resized_height":output.resized_height,"width_ratio":output.width_ratio,"height_ratio":output.height_ratio,"lines":lines}))?);
        processed_lines += output.lines.len();
        written_artifacts += artifacts.len();
        writer.add_case(&case.id, &case.kind, artifacts)?;
    }
    let output=writer.finish(json!({"provenance":{"implementation":"rust scalar canonical pixel-to-oriented-crop pipeline","reference_manifest_sha256":reference.manifest_sha256,"coverage":"complete Step12 independently from frozen BGR pixels; no recognition graph or text"}}))?;
    Ok(Step12RunReport {
        output,
        processed_cases: corpus.images.len(),
        processed_lines,
        written_artifacts,
        reference_manifest_sha256: reference.manifest_sha256,
        reference_validation: reference.validation,
    })
}
