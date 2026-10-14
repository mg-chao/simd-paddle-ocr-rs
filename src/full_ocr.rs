//! Complete, independent tiny OCR execution from frozen image pixels.
use crate::{
    Bundle, ComparisonReport, Result, Tolerance, ValidationReport,
    bundle::{BundleWriter, Dtype},
    cls,
    det::{self, ValidatedReference},
    detector::{DetectionTrace, Detector, DetectorOptions},
    image::{ImageError, ImageView, PixelFormat},
    invalid, inventory,
    lines::{LineOptions, LineTrace, OrientedLineExtractor},
    model::Model,
    ocr::{Ocr, OcrOptions, OcrTrace, OcrView},
    rec,
    recognizer::RecognitionTrace,
    runtime::{CompiledModel, ExecutionLimits},
    safe_file,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    fmt,
    path::{Path, PathBuf},
    sync::Arc,
};

#[derive(Debug, Serialize)]
pub struct OcrRunReport {
    pub output: PathBuf,
    pub processed_cases: usize,
    pub processed_lines: usize,
    pub written_artifacts: usize,
    pub reference_manifest_sha256: String,
    pub reference_validation: ValidationReport,
}
impl fmt::Display for OcrRunReport {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "OCR CANDIDATE: {} cases, {} lines, {} artifacts; saved {}; complete tiny OCR independently from frozen pixels",
            self.processed_cases,
            self.processed_lines,
            self.written_artifacts,
            self.output.display()
        )
    }
}
#[derive(Deserialize)]
struct InputMetadata {
    width: usize,
    height: usize,
    stride: usize,
    pixel_format: String,
}
struct Selection {
    lines: usize,
}

fn exact_integer(value: &Value, field: &str, positive: bool) -> Result<usize> {
    let number = value[field]
        .as_u64()
        .and_then(|number| usize::try_from(number).ok())
        .ok_or_else(|| invalid(format!("OCR {field} must be an exact nonnegative integer")))?;
    if positive && number == 0 {
        return Err(invalid(format!("OCR {field} must be positive")));
    }
    Ok(number)
}
fn finite(value: &Value, field: &str) -> Result<f64> {
    value[field]
        .as_f64()
        .filter(|number| number.is_finite())
        .ok_or_else(|| invalid(format!("OCR {field} must be finite")))
}
fn check_box(value: &Value) -> Result<()> {
    if !(0.0..=1.0).contains(&finite(value, "score")?) {
        return Err(invalid("OCR detection score must be a probability"));
    }
    let points = value["points"]
        .as_array()
        .filter(|points| points.len() == 4)
        .ok_or_else(|| invalid("OCR box must contain four points"))?;
    if points.iter().any(|point| {
        point.as_array().is_none_or(|coordinates| {
            coordinates.len() != 2
                || coordinates
                    .iter()
                    .any(|number| number.as_f64().is_none_or(|number| !number.is_finite()))
        })
    }) {
        return Err(invalid("OCR box coordinates must be finite pairs"));
    }
    Ok(())
}
fn selection(bundle: &Bundle) -> Result<Selection> {
    rec::require_canonical(bundle)?;
    let contract = &bundle.manifest.contract;
    if contract["capture_scope"] != "ocr"
        || contract["input_origin"] != "frozen_bgr_pixels"
        || contract["inventory_semantics"] != "lineage_keys_v1"
    {
        return Err(invalid(
            "complete tiny OCR requires capture_scope=ocr, frozen_bgr_pixels and lineage_keys_v1 inventory semantics",
        ));
    }
    if contract["options"]["use_direction_classification"].as_bool() != Some(true) {
        return Err(invalid(
            "canonical complete OCR requires direction classification",
        ));
    }
    if bundle.manifest.cases.is_empty() {
        return Err(invalid(
            "complete OCR requires at least one frozen image case",
        ));
    }
    let model_ids = bundle
        .manifest
        .models
        .iter()
        .map(|asset| asset.id.as_str())
        .collect::<HashSet<_>>();
    if model_ids != HashSet::from(["det", "cls", "rec", "dictionary"]) {
        return Err(invalid(
            "complete tiny OCR requires exactly DET, CLS, REC and dictionary assets",
        ));
    }
    let detector = det::detector_selection(&bundle.manifest)?;
    if detector.cases.len() != bundle.manifest.cases.len() {
        return Err(invalid("every OCR case requires complete DET capture"));
    }
    let compile = |id: &str| -> Result<CompiledModel> {
        let asset = bundle
            .manifest
            .models
            .iter()
            .find(|asset| asset.id == id)
            .ok_or_else(|| invalid(format!("missing OCR {id} model")))?;
        CompiledModel::new(Arc::new(
            Model::load_path(safe_file(&bundle.root, &asset.path)?)
                .map_err(|error| invalid(error.to_string()))?,
        ))
        .map_err(|error| invalid(error.to_string()))
    };
    let det_compiled = compile("det")?;
    let cls_compiled = compile("cls")?;
    let rec_compiled = compile("rec")?;
    let dictionary_asset = bundle
        .manifest
        .models
        .iter()
        .find(|asset| asset.id == "dictionary")
        .unwrap();
    let dictionary = crate::recognizer::Dictionary::from_utf8(
        &std::fs::read(safe_file(&bundle.root, &dictionary_asset.path)?)
            .map_err(|error| invalid(error.to_string()))?,
    )
    .map_err(|error| invalid(error.to_string()))?;
    let minimum_shapes = rec_compiled
        .node_output_shapes(&[1, 3, 48, 32])
        .map_err(|error| invalid(error.to_string()))?;
    if minimum_shapes.last().is_none_or(|shape| {
        let dimensions = shape.dimensions();
        dimensions.len() != 3 || dimensions[0] != 1 || dimensions[2] != dictionary.class_count()
    }) {
        return Err(invalid(
            "OCR REC output vocabulary differs from validated dictionary",
        ));
    }
    // Every model must have its semantic inventory even for a zero-line corpus.
    // Ordinal coverage is checked for every actual execution below.
    let model_nodes = |compiled: &CompiledModel| {
        (0..compiled.model().graph().nodes.len())
            .map(|index| (index, index))
            .collect::<Vec<_>>()
    };
    for (id, compiled) in [
        ("det", &det_compiled),
        ("cls", &cls_compiled),
        ("rec", &rec_compiled),
    ] {
        inventory::verify(bundle, compiled.model(), id, &model_nodes(compiled), true)?;
    }
    let mut lines = 0usize;
    for (ci, case) in bundle.manifest.cases.iter().enumerate() {
        if case.kind != "ocr" {
            return Err(invalid("complete OCR requires OCR case kinds"));
        }
        let by_id: HashMap<_, _> = case
            .artifacts
            .iter()
            .map(|artifact| (artifact.id.as_str(), artifact))
            .collect();
        let required = |id: &str| {
            by_id
                .get(id)
                .copied()
                .ok_or_else(|| invalid(format!("OCR case {} missing {id}", case.id)))
        };
        let metadata: InputMetadata =
            serde_json::from_value(rec::read_json(bundle, required("input.metadata")?)?)
                .map_err(|error| invalid(error.to_string()))?;
        if metadata.pixel_format != "bgr24"
            || metadata.width == 0
            || metadata.height == 0
            || metadata.stride
                < metadata
                    .width
                    .checked_mul(3)
                    .ok_or_else(|| invalid("OCR source row overflow"))?
        {
            return Err(invalid("invalid OCR source pixel metadata"));
        }
        let source = required("input.pixels")?;
        if source.dtype != Dtype::U8
            || source.layout != "bgr24_rows"
            || source.shape != [metadata.height, metadata.stride]
        {
            return Err(invalid("OCR source pixel shape/layout differs"));
        }
        let pixels = rec::read_bytes(bundle, source)?;
        ImageView::new(
            &pixels,
            metadata.width,
            metadata.height,
            metadata.stride,
            PixelFormat::Bgr24,
        )
        .map_err(|error| invalid(error.to_string()))?;
        let det_case = detector
            .cases
            .iter()
            .find(|selected| selected.case_index == ci)
            .ok_or_else(|| invalid("missing selected DET case"))?;
        let resized = crate::preprocess::compute_det_size(
            metadata.width,
            metadata.height,
            contract["options"]["detector"]["limit_side_length"]
                .as_u64()
                .unwrap_or(960) as usize,
        )
        .map_err(|error| invalid(error.to_string()))?;
        if case.artifacts[det_case.input_index].shape != [1, 3, resized.height, resized.width] {
            return Err(invalid(
                "OCR DET input dimensions differ from independent source/options metadata",
            ));
        }
        inventory::verify(bundle, det_compiled.model(), "det", &det_case.nodes, true)?;
        det::validate_bound_metadata(
            case,
            &det_case.nodes,
            &det_compiled,
            &case.artifacts[det_case.input_index].shape,
        )?;
        let boxes = rec::read_json(bundle, required("det.boxes")?)?;
        exact_integer(&boxes, "resized_width", true)?;
        exact_integer(&boxes, "resized_height", true)?;
        if exact_integer(&boxes, "resized_width", true)? != resized.width
            || exact_integer(&boxes, "resized_height", true)? != resized.height
            || finite(&boxes, "width_ratio")? as f32 != resized.width_ratio
            || finite(&boxes, "height_ratio")? as f32 != resized.height_ratio
        {
            return Err(invalid(
                "OCR detection resize metadata differs from source/options",
            ));
        }
        let box_array = boxes["boxes"]
            .as_array()
            .ok_or_else(|| invalid("OCR boxes must be an array"))?;
        for detection in box_array {
            check_box(detection)?;
        }
        let public = rec::read_json(bundle, required("result.public")?)?;
        let count = exact_integer(&public, "detected_count", false)?;
        let public_lines = public["lines"]
            .as_array()
            .filter(|array| array.len() == count)
            .ok_or_else(|| invalid("OCR result line count differs"))?;
        if box_array.len() != count {
            return Err(invalid("OCR public/detection line counts differ"));
        }
        exact_integer(&public, "detector_resized_width", true)?;
        exact_integer(&public, "detector_resized_height", true)?;
        if public["detector_resized_width"] != boxes["resized_width"]
            || public["detector_resized_height"] != boxes["resized_height"]
        {
            return Err(invalid("OCR public/detector dimensions differ"));
        }
        let mut combined = Vec::new();
        for (index, line) in public_lines.iter().enumerate() {
            check_box(&line["box"])?;
            if line["box"] != box_array[index] {
                return Err(invalid("OCR line boxes differ from detector order"));
            }
            let text = line["text"]
                .as_str()
                .ok_or_else(|| invalid("OCR line text must be string"))?;
            combined.push(text);
            finite(line, "recognition_score")?;
            let classification_score = finite(line, "classification_score")?;
            let classification_label = exact_integer(line, "classification_label", false)?;
            let applied_rotation = exact_integer(line, "applied_rotation_degrees", false)?;
            let threshold = contract["options"]["classifier_threshold"]
                .as_f64()
                .unwrap_or(0.0);
            let expected_rotation = if classification_label == 1 && classification_score > threshold
            {
                180
            } else {
                0
            };
            if classification_label > 1
                || !(0.0..=1.0).contains(&classification_score)
                || applied_rotation != expected_rotation
            {
                return Err(invalid("OCR classification label/rotation invalid"));
            }
            let prefix = format!("line.{index:03}");
            for suffix in [
                "crop",
                "cls.input",
                "cls.probabilities",
                "cls.public",
                "rec.pixels",
                "rec.input",
                "rec.probabilities",
                "rec.decoded_dense",
                "rec.public",
            ] {
                required(&format!("{prefix}.{suffix}"))?;
            }
            let cls_input = required(&format!("{prefix}.cls.input"))?;
            if cls_input.dtype != Dtype::F32
                || cls_input.layout != "nchw"
                || cls_input.shape != [1, 3, 80, 160]
            {
                return Err(invalid("OCR CLS input metadata differs"));
            }
            let cls_probability = required(&format!("{prefix}.cls.probabilities"))?;
            if cls_probability.dtype != Dtype::F32
                || cls_probability.layout != "row_major"
                || cls_probability.shape != [1, 2]
            {
                return Err(invalid("OCR CLS probability metadata differs"));
            }
            let cls_public = rec::read_json(bundle, required(&format!("{prefix}.cls.public"))?)?;
            if exact_integer(&cls_public, "label", false)? != classification_label
                || exact_integer(&cls_public, "orientation_degrees", false)?
                    != classification_label * 180
            {
                return Err(invalid(
                    "OCR classifier label/orientation integer structure invalid",
                ));
            }
            exact_integer(&cls_public, "resized_width", true)?;
            exact_integer(&cls_public, "dense_content_width", true)?;
            if cls_public["label"] != line["classification_label"]
                || cls_public["score"] != line["classification_score"]
            {
                return Err(invalid("OCR per-line/public classification differs"));
            }
            let cls_nodes = case
                .artifacts
                .iter()
                .enumerate()
                .filter_map(|(ai, artifact)| {
                    artifact
                        .id
                        .strip_prefix(&format!("{prefix}.cls.node."))
                        .map(|ordinal| (ai, ordinal))
                })
                .map(|(ai, ordinal)| {
                    ordinal
                        .parse()
                        .map(|ordinal| (ai, ordinal))
                        .map_err(|_| invalid("malformed OCR CLS node"))
                })
                .collect::<Result<Vec<_>>>()?;
            inventory::verify(bundle, cls_compiled.model(), "cls", &cls_nodes, true)?;
            det::validate_bound_metadata(case, &cls_nodes, &cls_compiled, &cls_input.shape)?;
            let rec_public = rec::read_json(bundle, required(&format!("{prefix}.rec.public"))?)?;
            rec::validate_decode(&rec_public)?;
            for (public_field, rec_field) in [
                ("text", "text"),
                ("recognition_score", "score"),
                ("emitted_count", "emitted_count"),
                ("recognition_content_width", "resized_width"),
                ("recognition_tensor_width", "tensor_width"),
                ("recognition_time_steps", "time_steps"),
                ("ctc_spans", "ctc_spans"),
            ] {
                if line[public_field] != rec_public[rec_field] {
                    return Err(invalid(format!(
                        "OCR public line {public_field} differs from recognizer"
                    )));
                }
            }
            for field in [
                "emitted_count",
                "recognition_content_width",
                "recognition_tensor_width",
                "recognition_time_steps",
            ] {
                exact_integer(line, field, field != "emitted_count")?;
            }
        }
        if public["text"].as_str() != Some(combined.join("\n").as_str()) {
            return Err(invalid("OCR public text differs from ordered line text"));
        }
        for artifact in &case.artifacts {
            if artifact.id.starts_with("line.") {
                let index = artifact
                    .id
                    .split('.')
                    .nth(1)
                    .and_then(|number| number.parse::<usize>().ok())
                    .ok_or_else(|| invalid("malformed OCR line ordinal"))?;
                if index >= count {
                    return Err(invalid("OCR artifact line outside public count"));
                }
                if !rec::is_rec_artifact(&artifact.id)?
                    && !matches!(
                        artifact.id.split('.').collect::<Vec<_>>().as_slice(),
                        [_, _, "crop"]
                            | [_, _, "cls", "input" | "probabilities" | "public"]
                            | [_, _, "cls", "node", _]
                    )
                {
                    return Err(invalid("unsupported OCR line artifact"));
                }
            } else if !matches!(
                artifact.id.as_str(),
                "input.metadata" | "input.pixels" | "input.encoded" | "result.public" | "det.boxes"
            ) && !det::is_detector_artifact(&artifact.id)?
            {
                return Err(invalid("unsupported complete OCR artifact"));
            }
        }
        lines = lines
            .checked_add(count)
            .ok_or_else(|| invalid("OCR line count overflow"))?;
    }
    if lines > 0 {
        let recognition = rec::selection(bundle)?;
        let classifier = cls::classifier_selection(&bundle.manifest)?;
        if recognition.lines.len() != lines || classifier.lines.len() != lines {
            return Err(invalid(
                "complete OCR classifier/recognizer line coverage differs",
            ));
        }
    }
    Ok(Selection { lines })
}

pub fn compare_ocr(
    reference: &Path,
    candidate: &Path,
    tolerance: Tolerance,
) -> Result<ComparisonReport> {
    let started = std::time::Instant::now();
    // Validate all declared files on both sides before semantic completeness so
    // invalid unselected data cannot be hidden by an earlier parity difference.
    let left = ValidatedReference::load(reference)?;
    let right = ValidatedReference::load(candidate)?;
    selection(&left.bundle)?;
    selection(&right.bundle)?;
    let mut report = crate::compare_bundles(reference, candidate, tolerance)?;
    report.elapsed_ms = started.elapsed().as_secs_f64() * 1000.;
    Ok(report)
}

pub fn public_json(view: OcrView<'_>) -> Value {
    let lines=view.iter().map(|line|json!({"box":crate::step12::box_json(&line.detection),"text":line.text,"recognition_score":line.recognition_score,"classification_score":line.classification_score,"classification_label":line.classification_label,"applied_rotation_degrees":line.applied_rotation_degrees,"emitted_count":line.emitted_count,"recognition_content_width":line.recognition_content_width,"recognition_tensor_width":line.recognition_tensor_width,"recognition_time_steps":line.recognition_time_steps,"ctc_spans":line.ctc_spans.map(|spans|spans.iter().map(|span|json!({"text":span.text,"score":span.score,"start_column":span.start_column,"end_column":span.end_column})).collect::<Vec<_>>())})).collect::<Vec<_>>();
    json!({"detected_count":view.len(),"detector_resized_width":view.detector_resized_width,"detector_resized_height":view.detector_resized_height,"text":view.text,"lines":lines})
}

pub(crate) fn build_ocr(reference: &ValidatedReference) -> Result<Ocr> {
    let compile = |id: &str| -> Result<Arc<CompiledModel>> {
        Ok(Arc::new(
            CompiledModel::new(Arc::new(
                Model::load_path(reference.model_path(id)?)
                    .map_err(|error| invalid(error.to_string()))?,
            ))
            .map_err(|error| invalid(error.to_string()))?,
        ))
    };
    let contract = &reference.bundle.manifest.contract;
    let value = &contract["options"]["detector"];
    let options = DetectorOptions {
        limit_side_length: value["limit_side_length"].as_u64().unwrap_or(960) as usize,
        max_candidates: value["max_candidates"].as_u64().unwrap_or(3000) as usize,
        bitmap_threshold: value["bitmap_threshold"].as_f64().unwrap_or(0.2) as f32,
        box_threshold: Some(value["box_threshold"].as_f64().unwrap_or(0.4) as f32),
        unclip_ratio: value["unclip_ratio"].as_f64().unwrap_or(1.4) as f32,
        use_dilation: value["use_dilation"].as_bool().unwrap_or(false),
        max_image_pixels: value["max_image_pixels"].as_u64().unwrap_or(40_000_000) as usize,
    };
    let detector = Detector::new(compile("det")?, options, ExecutionLimits::default())
        .map_err(|error| invalid(error.to_string()))?;
    let line_options = LineOptions {
        classifier_threshold: contract["options"]["classifier_threshold"]
            .as_f64()
            .unwrap_or(0.) as f32,
        max_crop_pixels: contract["options"]["max_crop_pixels"]
            .as_u64()
            .unwrap_or(16_000_000) as usize,
        max_classifier_image_pixels: contract["options"]["classifier"]["max_image_pixels"]
            .as_u64()
            .unwrap_or(40_000_000) as usize,
        ..Default::default()
    };
    let extractor = OrientedLineExtractor::new(detector, Some(compile("cls")?), line_options)
        .map_err(|error| invalid(error.to_string()))?;
    Ocr::new(
        extractor,
        rec::recognizer(reference)?,
        OcrOptions::default(),
    )
    .map_err(|error| invalid(error.to_string()))
}

pub fn run_ocr(reference_directory: &Path, output_directory: &Path) -> Result<OcrRunReport> {
    let reference = ValidatedReference::load(reference_directory)?;
    let selected = selection(&reference.bundle)?;
    let mut ocr = build_ocr(&reference)?;
    let mut writer =
        BundleWriter::new(output_directory, reference.bundle.manifest.contract.clone())?;
    rec::copy_assets(&reference, &mut writer)?;
    let mut processed_lines = 0usize;
    let mut written_artifacts = 0usize;
    for case in &reference.bundle.manifest.cases {
        let descriptors: HashMap<_, _> = case
            .artifacts
            .iter()
            .map(|artifact| (artifact.id.as_str(), artifact))
            .collect();
        let metadata: InputMetadata = serde_json::from_value(rec::read_json(
            &reference.bundle,
            descriptors["input.metadata"],
        )?)
        .map_err(|error| invalid(error.to_string()))?;
        let pixels = rec::read_bytes(&reference.bundle, descriptors["input.pixels"])?;
        let mut artifacts = Vec::new();
        for id in ["input.metadata", "input.pixels", "input.encoded"] {
            if let Some(descriptor) = descriptors.get(id) {
                let artifact = if descriptor.dtype == Dtype::Json {
                    writer.write_json(
                        id,
                        &descriptor.path,
                        &rec::read_json(&reference.bundle, descriptor)?,
                    )?
                } else {
                    writer.write_bytes(
                        id,
                        &descriptor.path,
                        &rec::read_bytes(&reference.bundle, descriptor)?,
                        &descriptor.shape,
                        &descriptor.layout,
                    )?
                };
                artifacts.push(artifact);
            }
        }
        let cell = RefCell::new((&mut writer, &mut artifacts));
        let dense_widths = RefCell::new(HashMap::new());
        let descriptor = |id: &str| {
            descriptors
                .get(id)
                .copied()
                .ok_or_else(|| ImageError::new(format!("unexpected OCR trace {id}")))
        };
        let write_float = |id: String,
                           values: &[f32],
                           shape: &[usize],
                           layout: &str|
         -> crate::image::ImageResult<()> {
            let saved = descriptor(&id)?;
            let mut pair = cell.borrow_mut();
            let artifact = pair
                .0
                .write_floats(&id, &saved.path, values, shape, layout)
                .map_err(|error| ImageError::new(error.to_string()))?;
            pair.1.push(artifact);
            Ok(())
        };
        let write_json = |id: String, value: Value| -> crate::image::ImageResult<()> {
            let saved = descriptor(&id)?;
            let mut pair = cell.borrow_mut();
            let artifact = pair
                .0
                .write_json(&id, &saved.path, &value)
                .map_err(|error| ImageError::new(error.to_string()))?;
            pair.1.push(artifact);
            Ok(())
        };
        let write_image = |id: String, image: ImageView<'_>| -> crate::image::ImageResult<()> {
            let saved = descriptor(&id)?;
            let mut pair = cell.borrow_mut();
            let artifact = pair
                .0
                .write_bytes(
                    &id,
                    &saved.path,
                    image.pixels(),
                    &[image.height(), image.width(), 3],
                    "bgr24",
                )
                .map_err(|error| ImageError::new(error.to_string()))?;
            pair.1.push(artifact);
            Ok(())
        };
        let output=ocr.run_with_trace(ImageView::new(&pixels,metadata.width,metadata.height,metadata.stride,PixelFormat::Bgr24).map_err(|error|invalid(error.to_string()))?,true,|event|match event{
            OcrTrace::Detection(event)=>match event{
                DetectionTrace::Input{shape,values}=>write_float("det.input".into(),values,shape.dimensions(),"nchw"),
                DetectionTrace::Node(node)=>write_float(format!("det.node.{:04}",node.index),node.values,node.shape.dimensions(),node.layout),
                DetectionTrace::Probabilities{shape,values}=>write_float("det.probabilities".into(),values,shape.dimensions(),"nchw"),
                DetectionTrace::Boxes(view)=>write_json("det.boxes".into(),json!({"resized_width":view.resized_width,"resized_height":view.resized_height,"width_ratio":view.width_ratio,"height_ratio":view.height_ratio,"boxes":view.boxes.iter().map(crate::step12::box_json).collect::<Vec<_>>()})),
            },
            OcrTrace::Line(event)=>match event{
                LineTrace::Crop{index,image}=>write_image(format!("line.{index:03}.crop"),image),
                LineTrace::ClsInput{index,values,resized_width}=>{dense_widths.borrow_mut().insert(index,resized_width);write_float(format!("line.{index:03}.cls.input"),values,&[1,3,80,160],"nchw")},
                LineTrace::ClsNode{index,node}=>write_float(format!("line.{index:03}.cls.node.{:04}",node.index),node.values,node.shape.dimensions(),node.layout),
                LineTrace::ClsProbabilities{index,values}=>write_float(format!("line.{index:03}.cls.probabilities"),values,&[1,2],"row_major"),
                LineTrace::Classification{index,result}=>write_json(format!("line.{index:03}.cls.public"),json!({"label":result.label,"score":result.score,"resized_width":result.resized_width,"orientation_degrees":result.orientation_degrees(),"dense_content_width":dense_widths.borrow()[&index]})),
                LineTrace::OrientedCrop{index,image,..}=>write_image(format!("line.{index:03}.rec.pixels"),image),
            },
            OcrTrace::Recognition{index,event}=>match event{
                RecognitionTrace::Input{shape,values,..}=>write_float(format!("line.{index:03}.rec.input"),values,shape.dimensions(),"nchw"),
                RecognitionTrace::Node(node)=>write_float(format!("line.{index:03}.rec.node.{:04}",node.index),node.values,node.shape.dimensions(),node.layout),
                RecognitionTrace::Probabilities{shape,values}=>write_float(format!("line.{index:03}.rec.probabilities"),values,shape.dimensions(),"row_major"),
                RecognitionTrace::DecodedDense(view)=>write_json(format!("line.{index:03}.rec.decoded_dense"),rec::recognition_json(view)),
                RecognitionTrace::Public(view)=>write_json(format!("line.{index:03}.rec.public"),rec::recognition_json(view)),
            },
            OcrTrace::Result(view)=>write_json("result.public".into(),public_json(view)),
        }).map_err(|error|invalid(error.to_string()))?;
        processed_lines += output.len();
        let generated: HashSet<_> = artifacts
            .iter()
            .map(|artifact| artifact.id.as_str())
            .collect();
        if generated.len() != case.artifacts.len()
            || case
                .artifacts
                .iter()
                .any(|artifact| !generated.contains(artifact.id.as_str()))
        {
            return Err(invalid(
                "OCR execution did not produce every required artifact",
            ));
        }
        artifacts.sort_by_key(|artifact| {
            case.artifacts
                .iter()
                .position(|reference| reference.id == artifact.id)
                .unwrap()
        });
        written_artifacts += artifacts.len();
        writer.add_case(&case.id, &case.kind, artifacts)?;
    }
    if processed_lines != selected.lines {
        return Err(invalid(
            "OCR execution line count differs from complete reference",
        ));
    }
    let output=writer.finish(json!({"provenance":{"implementation":"rust scalar canonical complete tiny OCR","reference_manifest_sha256":reference.manifest_sha256,"coverage":"all tiny OCR stages independently from frozen BGR source pixels; dense diagnostic and native public heads"}}))?;
    Ok(OcrRunReport {
        output,
        processed_cases: reference.bundle.manifest.cases.len(),
        processed_lines,
        written_artifacts,
        reference_manifest_sha256: reference.manifest_sha256,
        reference_validation: reference.validation,
    })
}
