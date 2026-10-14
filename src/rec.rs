//! Frozen recognizer graph execution and explicitly partial parity for shipped OCR families.
use crate::{
    Bundle, ComparisonReport, ComparisonScope, Result, Tolerance, ValidationReport,
    bundle::{ArtifactDescriptor, BundleWriter, Dtype},
    det::{self, ValidatedReference},
    invalid, inventory,
    model::Model,
    recognizer::{Dictionary, RecognitionTrace, RecognitionView, Recognizer, RecognizerOptions},
    runtime::{CompiledModel, ExecutionLimits},
    safe_file,
};
use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet},
    fmt, fs,
    path::{Path, PathBuf},
    sync::Arc,
};

#[derive(Debug, Serialize)]
pub struct RecRunReport {
    pub output: PathBuf,
    pub processed_cases: usize,
    pub processed_lines: usize,
    pub written_artifacts: usize,
    pub reference_manifest_sha256: String,
    pub reference_validation: ValidationReport,
}
impl fmt::Display for RecRunReport {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "REC CANDIDATE: {} lines, {} artifacts, {} cases; saved {}; scope frozen recognition tensors; no pixel preprocessing or full OCR parity",
            self.processed_lines,
            self.written_artifacts,
            self.processed_cases,
            self.output.display()
        )
    }
}
pub(crate) struct RecLine {
    pub case_index: usize,
    pub prefix: String,
    pub input_index: usize,
    pub pixels_index: usize,
    pub resized_width: usize,
}
pub(crate) struct Selection {
    pub lines: Vec<RecLine>,
    pub artifacts: HashSet<(usize, usize)>,
}

pub(crate) fn require_canonical(bundle: &Bundle) -> Result<()> {
    let contract = &bundle.manifest.contract;
    if !matches!(
        contract["model"].as_str(),
        Some("tiny" | "small" | "medium")
    ) {
        return Err(invalid(
            "recognition requires a shipped tiny, small or medium family",
        ));
    }
    for (field, expected) in [
        ("profile", "canonical"),
        ("backend", "cpu"),
        ("logical_input_layout", "nchw"),
        ("nodes", "all"),
    ] {
        if contract[field].as_str() != Some(expected) {
            return Err(invalid(format!(
                "recognition requires {field}={expected:?}"
            )));
        }
    }
    if contract["max_traced_lines"].as_u64() != Some(0)
        || contract["return_ctc_alignment"].as_bool() != Some(true)
    {
        return Err(invalid(
            "complete recognition reference requires all lines and CTC alignment",
        ));
    }
    Ok(())
}

pub(crate) fn positive_option(options: &Value, field: &str, fallback: usize) -> Result<usize> {
    match options.get(field) {
        None => Ok(fallback),
        Some(value) => value
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .filter(|value| *value > 0)
            .ok_or_else(|| {
                invalid(format!(
                    "OCR option {field} must be an exact positive integer"
                ))
            }),
    }
}
pub(crate) fn boolean_option(options: &Value, field: &str, fallback: bool) -> Result<bool> {
    match options.get(field) {
        None => Ok(fallback),
        Some(value) => value
            .as_bool()
            .ok_or_else(|| invalid(format!("OCR option {field} must be a boolean"))),
    }
}
pub(crate) fn number_option(options: &Value, field: &str, fallback: f32) -> Result<f32> {
    match options.get(field) {
        None => Ok(fallback),
        Some(value) => value
            .as_f64()
            .filter(|number| number.is_finite() && (*number as f32).is_finite())
            .map(|number| number as f32)
            .ok_or_else(|| invalid(format!("OCR option {field} must be a finite F32 number"))),
    }
}
pub(crate) fn recognizer_options(contract: &Value) -> Result<RecognizerOptions> {
    if contract
        .get("options")
        .is_some_and(|value| !value.is_object())
        || contract["options"]
            .get("recognizer")
            .is_some_and(|value| !value.is_object())
    {
        return Err(invalid("recognizer options must be JSON objects"));
    }
    let value = &contract["options"]["recognizer"];
    Ok(RecognizerOptions {
        target_width: positive_option(value, "target_width", 320)?,
        adaptive_width: boolean_option(value, "adaptive_width", true)?,
        max_image_pixels: positive_option(value, "max_image_pixels", 40_000_000)?,
        ..Default::default()
    })
}

pub(crate) fn is_rec_artifact(id: &str) -> Result<bool> {
    let fields: Vec<_> = id.split('.').collect();
    if fields.first() != Some(&"line") || fields.get(2) != Some(&"rec") {
        return Ok(false);
    }
    if fields
        .get(1)
        .is_none_or(|index| index.is_empty() || !index.bytes().all(|byte| byte.is_ascii_digit()))
    {
        return Err(invalid(format!("malformed recognition artifact {id:?}")));
    }
    match fields.as_slice() {
        [
            _,
            _,
            _,
            "pixels" | "input" | "probabilities" | "decoded_dense" | "public",
        ] => Ok(true),
        [_, _, _, "node", ordinal]
            if !ordinal.is_empty() && ordinal.bytes().all(|byte| byte.is_ascii_digit()) =>
        {
            Ok(true)
        }
        _ => Err(invalid(format!("unsupported recognition artifact {id:?}"))),
    }
}

pub(crate) fn read_bytes(bundle: &Bundle, artifact: &ArtifactDescriptor) -> Result<Vec<u8>> {
    let bytes = fs::read(safe_file(&bundle.root, &artifact.path)?)
        .map_err(|error| invalid(error.to_string()))?;
    if bytes.len() as u64 != artifact.bytes
        || !format!("{:x}", Sha256::digest(&bytes)).eq_ignore_ascii_case(&artifact.sha256)
    {
        return Err(invalid(format!(
            "{} payload integrity differs",
            artifact.id
        )));
    }
    Ok(bytes)
}
pub(crate) fn read_json(bundle: &Bundle, artifact: &ArtifactDescriptor) -> Result<Value> {
    if artifact.dtype != Dtype::Json {
        return Err(invalid(format!("{} must be JSON", artifact.id)));
    }
    serde_json::from_slice(&read_bytes(bundle, artifact)?)
        .map_err(|error| invalid(format!("{}: {error}", artifact.id)))
}
fn integer(value: &Value, field: &str) -> Result<usize> {
    value[field]
        .as_u64()
        .and_then(|number| usize::try_from(number).ok())
        .ok_or_else(|| {
            invalid(format!(
                "recognition {field} must be an exact nonnegative integer"
            ))
        })
}
pub(crate) fn validate_decode(value: &Value) -> Result<()> {
    let text = value["text"]
        .as_str()
        .ok_or_else(|| invalid("recognition text must be a string"))?;
    let score = value["score"]
        .as_f64()
        .filter(|score| score.is_finite())
        .ok_or_else(|| invalid("recognition score must be finite"))?;
    let emitted = integer(value, "emitted_count")?;
    let content = integer(value, "resized_width")?;
    let width = integer(value, "tensor_width")?;
    let steps = integer(value, "time_steps")?;
    if content == 0 || content > width || width == 0 || steps == 0 || emitted > steps {
        return Err(invalid("recognition dimensions/counts are inconsistent"));
    }
    let spans = value["ctc_spans"]
        .as_array()
        .ok_or_else(|| invalid("captured recognition requires CTC span array"))?;
    if spans.len() != emitted || (emitted == 0 && (!text.is_empty() || score != 0.0)) {
        return Err(invalid("recognition text/span/count structure differs"));
    }
    let mut previous_end = 0usize;
    let mut combined = String::new();
    for span in spans {
        let start = integer(span, "start_column")?;
        let end = integer(span, "end_column")?;
        let token = span["text"]
            .as_str()
            .ok_or_else(|| invalid("CTC token text must be a string"))?;
        if start < previous_end
            || start >= end
            || end > steps
            || span["score"]
                .as_f64()
                .is_none_or(|number| !number.is_finite())
        {
            return Err(invalid("CTC span is invalid"));
        }
        combined.push_str(token);
        previous_end = end;
    }
    if combined != text {
        return Err(invalid("CTC spans do not reconstruct recognition text"));
    }
    Ok(())
}
pub(crate) fn validate_declared_structures(bundle: &Bundle) -> Result<()> {
    for case in &bundle.manifest.cases {
        for artifact in &case.artifacts {
            if is_rec_artifact(&artifact.id)?
                && (artifact.id.ends_with(".public") || artifact.id.ends_with(".decoded_dense"))
            {
                validate_decode(&read_json(bundle, artifact)?)?;
            }
        }
    }
    Ok(())
}

pub(crate) fn selection(bundle: &Bundle) -> Result<Selection> {
    require_canonical(bundle)?;
    validate_declared_structures(bundle)?;
    let asset = |id: &str| {
        bundle
            .manifest
            .models
            .iter()
            .find(|asset| asset.id == id)
            .ok_or_else(|| invalid(format!("missing {id} asset")))
    };
    let model = Arc::new(
        Model::load_path(safe_file(&bundle.root, &asset("rec")?.path)?)
            .map_err(|error| invalid(error.to_string()))?,
    );
    let compiled = CompiledModel::new(model.clone()).map_err(|error| invalid(error.to_string()))?;
    let dictionary_bytes = fs::read(safe_file(&bundle.root, &asset("dictionary")?.path)?)
        .map_err(|error| invalid(error.to_string()))?;
    let dictionary =
        Dictionary::from_utf8(&dictionary_bytes).map_err(|error| invalid(error.to_string()))?;
    let classes = dictionary.class_count();
    let mut lines = Vec::new();
    let mut selected = HashSet::new();
    for (ci, case) in bundle.manifest.cases.iter().enumerate() {
        let by_id: HashMap<_, _> = case
            .artifacts
            .iter()
            .enumerate()
            .map(|(index, artifact)| (artifact.id.as_str(), index))
            .collect();
        let mut consumed = HashSet::new();
        for (input_index, input) in case
            .artifacts
            .iter()
            .enumerate()
            .filter(|(_, artifact)| artifact.id.ends_with(".rec.input"))
        {
            let prefix = input.id.strip_suffix(".input").unwrap().to_string();
            if !is_rec_artifact(&input.id)?
                || input.dtype != Dtype::F32
                || input.layout != "nchw"
                || input.shape.len() != 4
                || input.shape[..3] != [1, 3, 48]
                || input.shape[3] == 0
            {
                return Err(invalid(
                    "REC input must be [1,3,48,W] NCHW f32 with positive W",
                ));
            }
            let required = |suffix: &str| {
                by_id
                    .get(format!("{prefix}.{suffix}").as_str())
                    .copied()
                    .ok_or_else(|| invalid(format!("case {} missing {prefix}.{suffix}", case.id)))
            };
            let pixels_index = required("pixels")?;
            let pixels = &case.artifacts[pixels_index];
            if pixels.dtype != Dtype::U8
                || pixels.layout != "bgr24"
                || pixels.shape.len() != 3
                || pixels.shape[2] != 3
                || pixels.shape[..2].contains(&0)
            {
                return Err(invalid("REC pixels must be a nonempty packed BGR crop"));
            }
            let natural = 48usize
                .checked_mul(pixels.shape[1])
                .and_then(|value| value.checked_add(pixels.shape[0] - 1))
                .ok_or_else(|| invalid("REC content width overflow"))?
                / pixels.shape[0];
            let options = recognizer_options(&bundle.manifest.contract)?;
            let expected_width = if options.adaptive_width {
                natural
                    .checked_add(31)
                    .map(|width| (width / 32 * 32).max(32))
                    .ok_or_else(|| invalid("REC adaptive width overflow"))?
            } else {
                options.target_width
            };
            if input.shape[3] != expected_width {
                return Err(invalid(
                    "REC input width differs from independent crop/options metadata",
                ));
            }
            let resized_width = natural.min(input.shape[3]);
            let probabilities_index = required("probabilities")?;
            let probability = &case.artifacts[probabilities_index];
            let output = compiled
                .node_output_shapes(&input.shape)
                .map_err(|error| invalid(error.to_string()))?;
            let shape = output
                .last()
                .ok_or_else(|| invalid("REC graph is empty"))?
                .dimensions();
            if shape.len() != 3
                || shape[0] != 1
                || shape[2] != classes
                || probability.shape != shape
                || probability.dtype != Dtype::F32
                || probability.layout != "row_major"
            {
                return Err(invalid(
                    "REC probability shape must match the bound graph and dictionary",
                ));
            }
            for suffix in ["decoded_dense", "public"] {
                let index = required(suffix)?;
                let result = read_json(bundle, &case.artifacts[index])?;
                if integer(&result, "resized_width")? != resized_width
                    || integer(&result, "tensor_width")? != input.shape[3]
                    || integer(&result, "time_steps")? != shape[1]
                {
                    return Err(invalid(
                        "REC decoded dimensions differ from independent input/model metadata",
                    ));
                }
            }
            let nodes = case
                .artifacts
                .iter()
                .enumerate()
                .filter_map(|(index, artifact)| {
                    artifact
                        .id
                        .strip_prefix(&format!("{prefix}.node."))
                        .map(|ordinal| (index, ordinal))
                })
                .map(|(index, ordinal)| {
                    ordinal
                        .parse::<usize>()
                        .map(|ordinal| (index, ordinal))
                        .map_err(|_| invalid("invalid REC node ordinal"))
                })
                .collect::<Result<Vec<_>>>()?;
            inventory::verify(
                bundle,
                &model,
                "rec",
                &nodes,
                bundle.manifest.contract["inventory_semantics"] == "lineage_keys_v1",
            )?;
            det::validate_bound_metadata(case, &nodes, &compiled, &input.shape)?;
            for (index, artifact) in case.artifacts.iter().enumerate() {
                if artifact.id.starts_with(&format!("{prefix}.")) && is_rec_artifact(&artifact.id)?
                {
                    selected.insert((ci, index));
                    consumed.insert(index);
                }
            }
            lines.push(RecLine {
                case_index: ci,
                prefix,
                input_index,
                pixels_index,
                resized_width,
            });
        }
        for (index, artifact) in case.artifacts.iter().enumerate() {
            if is_rec_artifact(&artifact.id)? && !consumed.contains(&index) {
                return Err(invalid(format!(
                    "orphan recognition artifact {}",
                    artifact.id
                )));
            }
        }
    }
    if lines.is_empty() {
        return Err(invalid("reference contains no captured recognition lines"));
    }
    Ok(Selection {
        lines,
        artifacts: selected,
    })
}

pub(crate) fn recognition_json(view: RecognitionView<'_>) -> Value {
    json!({"text":view.text,"score":view.score,"emitted_count":view.emitted_count,"resized_width":view.resized_width,"tensor_width":view.tensor_width,"time_steps":view.time_steps,"ctc_spans":view.ctc_spans.map(|spans|spans.iter().map(|span|json!({"text":span.text,"score":span.score,"start_column":span.start_column,"end_column":span.end_column})).collect::<Vec<_>>())})
}
pub(crate) fn recognizer(reference: &ValidatedReference) -> Result<Recognizer> {
    let model = Arc::new(
        Model::load_path(reference.model_path("rec")?)
            .map_err(|error| invalid(error.to_string()))?,
    );
    let compiled = Arc::new(CompiledModel::new(model).map_err(|error| invalid(error.to_string()))?);
    let dictionary = Arc::new(
        Dictionary::from_utf8(
            &fs::read(reference.model_path("dictionary")?)
                .map_err(|error| invalid(error.to_string()))?,
        )
        .map_err(|error| invalid(error.to_string()))?,
    );
    let options = recognizer_options(&reference.bundle.manifest.contract)?;
    Recognizer::new(compiled, dictionary, options, ExecutionLimits::default())
        .map_err(|error| invalid(error.to_string()))
}
pub(crate) fn copy_assets(reference: &ValidatedReference, writer: &mut BundleWriter) -> Result<()> {
    for asset in &reference.bundle.manifest.models {
        writer.copy_model(
            &asset.id,
            &safe_file(&reference.bundle.root, &asset.path)?,
            &asset.path,
        )?;
    }
    for asset in &reference.bundle.manifest.auxiliary_files {
        writer.copy_auxiliary(
            &safe_file(&reference.bundle.root, &asset.path)?,
            &asset.path,
            &asset.role,
        )?;
    }
    Ok(())
}
pub fn compare_rec(
    reference: &Path,
    candidate: &Path,
    tolerance: Tolerance,
) -> Result<ComparisonReport> {
    if !tolerance.atol.is_finite()
        || tolerance.atol < 0.0
        || !tolerance.rtol.is_finite()
        || tolerance.rtol < 0.0
    {
        return Err(invalid("tolerances must be finite and nonnegative"));
    }
    let started = std::time::Instant::now();
    // Complete both-side file integrity before interpreting reference coverage
    // or candidate structures. A malformed semantic stage must not hide a bad
    // payload elsewhere in either declared bundle.
    let left = ValidatedReference::load(reference)?;
    let right = ValidatedReference::load(candidate)?;
    if crate::identity_difference(&left.bundle.manifest, &right.bundle.manifest).is_none() {
        let model = Model::load_path(right.model_path("rec")?)
            .map_err(|error| invalid(error.to_string()))?;
        // These pairs describe the model's complete inventory, not the
        // candidate's captured tensors. Missing/extra captures continue through
        // the existing scoped comparator as ordinary parity differences.
        let model_nodes = (0..model.graph().nodes.len())
            .map(|ordinal| (ordinal, ordinal))
            .collect::<Vec<_>>();
        inventory::verify(
            &right.bundle,
            &model,
            "rec",
            &model_nodes,
            right.bundle.manifest.contract["inventory_semantics"] == "lineage_keys_v1",
        )?;
    }
    let mut report = crate::compare(
        reference,
        candidate,
        tolerance,
        ComparisonScope::Rec {
            required_lines: 0,
            required_artifacts: 0,
        },
    )?;
    report.elapsed_ms = started.elapsed().as_secs_f64() * 1000.;
    Ok(report)
}
pub fn run_rec(reference_directory: &Path, output_directory: &Path) -> Result<RecRunReport> {
    let reference = ValidatedReference::load(reference_directory)?;
    let selected = selection(&reference.bundle)?;
    let mut recognizer = recognizer(&reference)?;
    let mut writer =
        BundleWriter::new(output_directory, reference.bundle.manifest.contract.clone())?;
    copy_assets(&reference, &mut writer)?;
    let mut written_artifacts = 0;
    let mut processed_cases = 0;
    for (ci, case) in reference.bundle.manifest.cases.iter().enumerate() {
        let mut artifacts = Vec::new();
        let descriptors: HashMap<_, _> = case
            .artifacts
            .iter()
            .map(|artifact| (artifact.id.as_str(), artifact))
            .collect();
        let lines: Vec<_> = selected
            .lines
            .iter()
            .filter(|line| line.case_index == ci)
            .collect();
        if !lines.is_empty() {
            processed_cases += 1;
        }
        for line in lines {
            let input = &case.artifacts[line.input_index];
            let pixels = &case.artifacts[line.pixels_index];
            artifacts.push(writer.write_bytes(
                &pixels.id,
                &pixels.path,
                &read_bytes(&reference.bundle, pixels)?,
                &pixels.shape,
                &pixels.layout,
            )?);
            let values = reference.read_f32(input)?;
            recognizer
                .recognize_tensor_with_trace(
                    &values,
                    &input.shape,
                    line.resized_width,
                    true,
                    |event| {
                        let (id, artifact) = match event {
                            RecognitionTrace::Input { shape, values, .. } => {
                                let id = format!("{}.input", line.prefix);
                                let descriptor = descriptors[id.as_str()];
                                let artifact = writer.write_floats(
                                    &id,
                                    &descriptor.path,
                                    values,
                                    shape.dimensions(),
                                    "nchw",
                                );
                                (id, artifact)
                            }
                            RecognitionTrace::Node(node) => {
                                let id = format!("{}.node.{:04}", line.prefix, node.index);
                                let descriptor = descriptors.get(id.as_str()).ok_or_else(|| {
                                    crate::image::ImageError::new("unexpected REC trace node")
                                })?;
                                let artifact = writer.write_floats(
                                    &id,
                                    &descriptor.path,
                                    node.values,
                                    node.shape.dimensions(),
                                    node.layout,
                                );
                                (id, artifact)
                            }
                            RecognitionTrace::Probabilities { shape, values } => {
                                let id = format!("{}.probabilities", line.prefix);
                                let descriptor = descriptors[id.as_str()];
                                let artifact = writer.write_floats(
                                    &id,
                                    &descriptor.path,
                                    values,
                                    shape.dimensions(),
                                    "row_major",
                                );
                                (id, artifact)
                            }
                            RecognitionTrace::DecodedDense(view) => {
                                let id = format!("{}.decoded_dense", line.prefix);
                                let artifact = writer.write_json(
                                    &id,
                                    &descriptors[id.as_str()].path,
                                    &recognition_json(view),
                                );
                                (id, artifact)
                            }
                            RecognitionTrace::Public(view) => {
                                let id = format!("{}.public", line.prefix);
                                let artifact = writer.write_json(
                                    &id,
                                    &descriptors[id.as_str()].path,
                                    &recognition_json(view),
                                );
                                (id, artifact)
                            }
                        };
                        let _ = id;
                        artifacts.push(
                            artifact.map_err(|error| {
                                crate::image::ImageError::new(error.to_string())
                            })?,
                        );
                        Ok(())
                    },
                )
                .map_err(|error| invalid(error.to_string()))?;
        }
        let expected: HashSet<_> = selected
            .artifacts
            .iter()
            .filter(|(case_index, _)| *case_index == ci)
            .map(|(_, index)| case.artifacts[*index].id.as_str())
            .collect();
        let observed: HashSet<_> = artifacts
            .iter()
            .map(|artifact| artifact.id.as_str())
            .collect();
        if observed != expected {
            return Err(invalid(
                "REC execution did not produce every required artifact",
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
    let output=writer.finish(json!({"provenance":{"implementation":"rust scalar canonical REC","reference_manifest_sha256":reference.manifest_sha256,"coverage":"frozen normalized REC tensors, dense diagnostic and native compatible public head; no image preprocessing or full OCR"}}))?;
    Ok(RecRunReport {
        output,
        processed_cases,
        processed_lines: selected.lines.len(),
        written_artifacts,
        reference_manifest_sha256: reference.manifest_sha256,
        reference_validation: reference.validation,
    })
}
