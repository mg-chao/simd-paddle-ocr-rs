//! Frozen-tensor classifier execution and explicitly partial parity.

use crate::{
    Bundle, Manifest, Result, ValidationReport,
    bundle::{ArtifactDescriptor, BundleWriter, Dtype},
    invalid,
    model::Model,
    read_bounded,
    runtime::{CompiledModel, ExecutionLimits, RuntimeError, Session},
    safe_file, validate_artifact, validate_assets,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet},
    fmt,
    fs::File,
    io::Read,
    path::{Path, PathBuf},
    sync::Arc,
};

#[derive(Debug)]
pub struct FrozenClsInput {
    pub case_id: String,
    pub artifact_id: String,
    pub shape: Vec<usize>,
    pub sha256: String,
    pub values: Vec<f32>,
}

/// Verified, preloaded inputs shared by the graph benchmark and CLS harness.
#[derive(Debug)]
pub struct FrozenClsInputs {
    pub model_path: PathBuf,
    pub model_sha256: String,
    pub manifest_sha256: String,
    pub inputs: Vec<FrozenClsInput>,
}

#[derive(Debug, Serialize)]
pub struct ClsRunReport {
    pub output: PathBuf,
    pub processed_cases: usize,
    pub processed_lines: usize,
    pub written_artifacts: usize,
    pub reference_manifest_sha256: String,
    pub reference_validation: ValidationReport,
}

impl fmt::Display for ClsRunReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CLS CANDIDATE: {} lines, {} artifacts, {} cases; saved {}; scope frozen classifier tensors",
            self.processed_lines,
            self.written_artifacts,
            self.processed_cases,
            self.output.display()
        )
    }
}

#[derive(Clone, Copy, Debug)]
enum ClsStage {
    Input,
    Node(usize),
    Probabilities,
}

fn parse_artifact_id(id: &str) -> Result<Option<(String, ClsStage)>> {
    let fields: Vec<_> = id.split('.').collect();
    if fields.len() < 4 || fields[0] != "line" || fields[2] != "cls" {
        return Ok(None);
    }
    if fields[1].is_empty() || !fields[1].bytes().all(|b| b.is_ascii_digit()) {
        return Err(invalid(format!("malformed classifier artifact ID {id:?}")));
    }
    let stage = match fields.as_slice() {
        [_, _, _, "input"] => ClsStage::Input,
        [_, _, _, "probabilities"] => ClsStage::Probabilities,
        [_, _, _, "node", ordinal]
            if !ordinal.is_empty() && ordinal.bytes().all(|b| b.is_ascii_digit()) =>
        {
            ClsStage::Node(
                ordinal
                    .parse()
                    .map_err(|_| invalid(format!("malformed classifier node ID {id:?}")))?,
            )
        }
        [_, _, _, "public"] => return Ok(None),
        _ => {
            return Err(invalid(format!(
                "unsupported classifier artifact ID {id:?}"
            )));
        }
    };
    Ok(Some((format!("line.{}.cls", fields[1]), stage)))
}

pub(crate) fn is_classifier_artifact(id: &str) -> Result<bool> {
    Ok(parse_artifact_id(id)?.is_some())
}

pub(crate) struct ClsLine {
    pub case_index: usize,
    pub input_index: usize,
    pub nodes: Vec<(usize, usize)>,
    pub probabilities_index: usize,
}

pub(crate) struct ClassifierSelection {
    pub lines: Vec<ClsLine>,
    pub artifacts: HashSet<(usize, usize)>,
}

fn require_canonical(manifest: &Manifest) -> Result<()> {
    for (field, expected) in [
        ("profile", "canonical"),
        ("backend", "cpu"),
        ("logical_input_layout", "nchw"),
    ] {
        if manifest.contract[field].as_str() != Some(expected) {
            return Err(invalid(format!(
                "CLS reference requires contract {field}={expected:?}"
            )));
        }
    }
    if !matches!(
        manifest.contract["nodes"].as_str(),
        Some("all" | "checkpoints")
    ) {
        return Err(invalid("CLS reference requires nodes=all or checkpoints"));
    }
    Ok(())
}

pub(crate) fn classifier_selection(manifest: &Manifest) -> Result<ClassifierSelection> {
    require_canonical(manifest)?;
    let mut lines = Vec::new();
    let mut artifacts = HashSet::new();
    for (case_index, case) in manifest.cases.iter().enumerate() {
        let mut grouped: HashMap<String, Vec<(usize, ClsStage)>> = HashMap::new();
        let mut input_order = Vec::new();
        for (artifact_index, artifact) in case.artifacts.iter().enumerate() {
            if let Some((prefix, stage)) = parse_artifact_id(&artifact.id)? {
                if artifact.dtype != Dtype::F32 {
                    return Err(invalid(format!(
                        "classifier artifact {} must be f32",
                        artifact.id
                    )));
                }
                if matches!(stage, ClsStage::Input) {
                    if artifact.shape.len() != 4
                        || artifact.shape[0] == 0
                        || artifact.shape[1..] != [3, 80, 160]
                        || artifact.layout != "nchw"
                    {
                        return Err(invalid(format!(
                            "classifier input {} must have shape [N,3,80,160] and NCHW layout",
                            artifact.id
                        )));
                    }
                    input_order.push(prefix.clone());
                }
                grouped
                    .entry(prefix)
                    .or_default()
                    .push((artifact_index, stage));
                artifacts.insert((case_index, artifact_index));
            }
        }
        for prefix in input_order {
            let stages = grouped
                .remove(&prefix)
                .ok_or_else(|| invalid(format!("duplicate classifier input {prefix}")))?;
            let mut input = None;
            let mut probabilities = None;
            let mut nodes = Vec::new();
            let mut ordinals = HashSet::new();
            for (index, stage) in stages {
                match stage {
                    ClsStage::Input => input = Some(index),
                    ClsStage::Probabilities => probabilities = Some(index),
                    ClsStage::Node(ordinal) => {
                        if !ordinals.insert(ordinal) {
                            return Err(invalid(format!(
                                "duplicate classifier node ordinal {prefix}/{ordinal}"
                            )));
                        }
                        nodes.push((index, ordinal));
                    }
                }
            }
            if nodes.is_empty() {
                return Err(invalid(format!(
                    "classifier line {prefix} has no captured nodes"
                )));
            }
            lines.push(ClsLine {
                case_index,
                input_index: input.unwrap(),
                nodes,
                probabilities_index: probabilities.ok_or_else(|| {
                    invalid(format!("classifier line {prefix} is missing probabilities"))
                })?,
            });
        }
        if let Some(prefix) = grouped.keys().next() {
            return Err(invalid(format!(
                "classifier line {prefix} is missing an input"
            )));
        }
    }
    if lines.is_empty() {
        return Err(invalid("reference contains no captured classifier inputs"));
    }
    Ok(ClassifierSelection { lines, artifacts })
}

struct ValidatedReference {
    bundle: Bundle,
    selection: ClassifierSelection,
    validation: ValidationReport,
    manifest_sha256: String,
}

impl ValidatedReference {
    fn load(directory: &Path) -> Result<Self> {
        let bundle = Bundle::load(directory)?;
        let mut validation = ValidationReport::default();
        validate_assets(&bundle, &mut validation)?;
        for case in &bundle.manifest.cases {
            for artifact in &case.artifacts {
                validate_artifact(&bundle, artifact, &mut validation)?;
            }
        }
        let selection = classifier_selection(&bundle.manifest)?;
        let bytes = read_bounded(
            File::open(safe_file(&bundle.root, "manifest.json")?)
                .map_err(|e| invalid(e.to_string()))?,
            "manifest.json",
        )?;
        let manifest_sha256 = format!("{:x}", Sha256::digest(bytes));
        Ok(Self {
            bundle,
            selection,
            validation,
            manifest_sha256,
        })
    }

    fn model_asset(&self) -> Result<&crate::bundle::ModelDescriptor> {
        self.bundle
            .manifest
            .models
            .iter()
            .find(|model| model.id == "cls")
            .ok_or_else(|| invalid("reference is missing the cls model"))
    }

    fn model_path(&self) -> Result<PathBuf> {
        safe_file(&self.bundle.root, &self.model_asset()?.path)
    }

    fn read_f32(&self, artifact: &ArtifactDescriptor) -> Result<Vec<f32>> {
        let path = safe_file(&self.bundle.root, &artifact.path)?;
        let count = usize::try_from(artifact.bytes / 4)
            .map_err(|_| invalid("classifier tensor does not fit usize"))?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|e| invalid(format!("cannot reserve classifier tensor: {e}")))?;
        let mut file = File::open(path).map_err(|e| invalid(e.to_string()))?;
        let mut hash = Sha256::new();
        let mut buffer = [0u8; crate::STREAM_BLOCK_BYTES];
        let mut remaining = artifact.bytes;
        while remaining != 0 {
            let length = remaining.min(buffer.len() as u64) as usize;
            file.read_exact(&mut buffer[..length])
                .map_err(|e| invalid(format!("{}: {e}", artifact.path)))?;
            hash.update(&buffer[..length]);
            for bytes in buffer[..length].chunks_exact(4) {
                let value = f32::from_le_bytes(bytes.try_into().unwrap());
                if !value.is_finite() {
                    return Err(invalid(format!(
                        "{}: nonfinite classifier input",
                        artifact.id
                    )));
                }
                values.push(value);
            }
            remaining -= length as u64;
        }
        let mut trailing = [0u8; 1];
        if file
            .read(&mut trailing)
            .map_err(|e| invalid(e.to_string()))?
            != 0
            || !format!("{:x}", hash.finalize()).eq_ignore_ascii_case(&artifact.sha256)
        {
            return Err(invalid(format!(
                "{}: classifier input integrity differs",
                artifact.path
            )));
        }
        Ok(values)
    }
}

/// Validate the entire bundle and load all captured inputs before measurement.
pub fn load_cls_inputs(directory: &Path) -> Result<FrozenClsInputs> {
    let reference = ValidatedReference::load(directory)?;
    let mut inputs = Vec::with_capacity(reference.selection.lines.len());
    for line in &reference.selection.lines {
        let case = &reference.bundle.manifest.cases[line.case_index];
        let artifact = &case.artifacts[line.input_index];
        inputs.push(FrozenClsInput {
            case_id: case.id.clone(),
            artifact_id: artifact.id.clone(),
            shape: artifact.shape.clone(),
            sha256: artifact.sha256.clone(),
            values: reference.read_f32(artifact)?,
        });
    }
    Ok(FrozenClsInputs {
        model_path: reference.model_path()?,
        model_sha256: reference.model_asset()?.sha256.clone(),
        manifest_sha256: reference.manifest_sha256,
        inputs,
    })
}

#[derive(Deserialize)]
struct Inventory {
    id: String,
    nodes: Vec<InventoryNode>,
}

#[derive(Deserialize)]
struct InventoryNode {
    index: usize,
    name: String,
    op_type: String,
    domain: String,
    opset: i64,
}

fn verify_inventory(reference: &ValidatedReference, model: &Model) -> Result<()> {
    let path = safe_file(&reference.bundle.root, "graph-inventory.json")?;
    let bytes = read_bounded(
        File::open(path).map_err(|e| invalid(e.to_string()))?,
        "graph-inventory.json",
    )?;
    let inventories: Vec<Inventory> = serde_json::from_slice(&bytes)
        .map_err(|e| invalid(format!("graph-inventory.json: {e}")))?;
    let classifiers: Vec<_> = inventories
        .iter()
        .filter(|inventory| inventory.id == "cls")
        .collect();
    if classifiers.len() != 1 || classifiers[0].nodes.len() != model.graph().nodes.len() {
        return Err(invalid(
            "classifier graph inventory count differs from the normalized Rust graph",
        ));
    }
    for (ordinal, (saved, node)) in classifiers[0]
        .nodes
        .iter()
        .zip(&model.graph().nodes)
        .enumerate()
    {
        if saved.index != ordinal
            || saved.name != node.name
            || saved.op_type != node.op_type
            || saved.domain != node.domain
            || saved.opset != node.opset
        {
            return Err(invalid(format!(
                "classifier graph inventory differs at normalized node {ordinal}"
            )));
        }
    }
    for line in &reference.selection.lines {
        if reference.bundle.manifest.contract["nodes"] == "all"
            && line.nodes.len() != model.graph().nodes.len()
        {
            return Err(invalid(
                "all-node classifier reference is missing a node capture",
            ));
        }
        if line
            .nodes
            .iter()
            .any(|(_, ordinal)| *ordinal >= model.graph().nodes.len())
        {
            return Err(invalid(
                "classifier reference node ordinal is outside the normalized graph",
            ));
        }
    }
    Ok(())
}

/// Run the canonical classifier graph on every frozen CLS input, without image preprocessing.
pub fn run_cls(reference_directory: &Path, output_directory: &Path) -> Result<ClsRunReport> {
    let reference = ValidatedReference::load(reference_directory)?;
    let model =
        Arc::new(Model::load_path(reference.model_path()?).map_err(|e| invalid(e.to_string()))?);
    verify_inventory(&reference, &model)?;
    let compiled = Arc::new(CompiledModel::new(model).map_err(|e| invalid(e.to_string()))?);
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
    let mut processed_cases = 0;
    let mut session: Option<Session> = None;
    for (case_index, case) in reference.bundle.manifest.cases.iter().enumerate() {
        let mut artifacts = Vec::new();
        let lines: Vec<_> = reference
            .selection
            .lines
            .iter()
            .filter(|line| line.case_index == case_index)
            .collect();
        if !lines.is_empty() {
            processed_cases += 1;
        }
        for line in lines {
            let input_descriptor = &case.artifacts[line.input_index];
            let values = reference.read_f32(input_descriptor)?;
            artifacts.push(writer.write_floats(
                &input_descriptor.id,
                &input_descriptor.path,
                &values,
                &input_descriptor.shape,
                &input_descriptor.layout,
            )?);
            if let Some(session) = &mut session {
                session
                    .reshape(&input_descriptor.shape)
                    .map_err(|e| invalid(e.to_string()))?;
            } else {
                session = Some(
                    Session::new(
                        compiled.clone(),
                        &input_descriptor.shape,
                        ExecutionLimits::default(),
                    )
                    .map_err(|e| invalid(e.to_string()))?,
                );
            }
            let selected: HashMap<_, _> = line
                .nodes
                .iter()
                .map(|(artifact_index, ordinal)| (*ordinal, &case.artifacts[*artifact_index]))
                .collect();
            let mut observed = HashSet::new();
            let output = session
                .as_mut()
                .unwrap()
                .run_with_trace(&values, |trace| {
                    if let Some(descriptor) = selected.get(&trace.index) {
                        artifacts.push(
                            writer
                                .write_floats(
                                    &descriptor.id,
                                    &descriptor.path,
                                    trace.values,
                                    trace.shape.dimensions(),
                                    trace.layout,
                                )
                                .map_err(|e| RuntimeError::new(e.to_string()))?,
                        );
                        observed.insert(trace.index);
                    }
                    Ok(())
                })
                .map_err(|e| invalid(e.to_string()))?;
            if observed.len() != selected.len() {
                return Err(invalid(
                    "classifier execution omitted a requested node trace",
                ));
            }
            let probabilities = &case.artifacts[line.probabilities_index];
            let layout = if output.shape().rank() == 4 {
                "nchw"
            } else {
                "row_major"
            };
            artifacts.push(writer.write_floats(
                &probabilities.id,
                &probabilities.path,
                output.data(),
                output.shape().dimensions(),
                layout,
            )?);
        }
        written_artifacts += artifacts.len();
        writer.add_case(&case.id, &case.kind, artifacts)?;
    }
    let output = writer.finish(json!({
        "provenance": {
            "implementation": "rust scalar canonical CLS",
            "reference_manifest_sha256": reference.manifest_sha256,
            "coverage": "classifier input, selected node outputs, and probabilities only; no image preprocessing or full OCR inference",
        }
    }))?;
    Ok(ClsRunReport {
        output,
        processed_cases,
        processed_lines: reference.selection.lines.len(),
        written_artifacts,
        reference_manifest_sha256: reference.manifest_sha256,
        reference_validation: reference.validation,
    })
}
