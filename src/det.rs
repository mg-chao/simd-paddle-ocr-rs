//! Frozen DET graph execution. Pixel preprocessing and geometry are outside this scope.
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
pub struct FrozenDetInput {
    pub case_id: String,
    pub artifact_id: String,
    pub shape: Vec<usize>,
    pub sha256: String,
    pub values: Vec<f32>,
}
#[derive(Debug)]
pub struct FrozenDetInputs {
    pub model_path: PathBuf,
    pub model_sha256: String,
    pub manifest_sha256: String,
    pub inputs: Vec<FrozenDetInput>,
}
#[derive(Debug, Serialize)]
pub struct DetRunReport {
    pub output: PathBuf,
    pub processed_cases: usize,
    pub written_artifacts: usize,
    pub reference_manifest_sha256: String,
    pub reference_validation: ValidationReport,
}
impl fmt::Display for DetRunReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DET CANDIDATE: {} cases, {} artifacts; saved {}; scope frozen normalized detector tensors",
            self.processed_cases,
            self.written_artifacts,
            self.output.display()
        )
    }
}
pub(crate) struct DetCase {
    pub case_index: usize,
    pub input_index: usize,
    pub nodes: Vec<(usize, usize)>,
    pub probabilities_index: usize,
}
pub(crate) struct DetSelection {
    pub cases: Vec<DetCase>,
    pub artifacts: HashSet<(usize, usize)>,
}
pub(crate) fn require_canonical(manifest: &Manifest) -> Result<()> {
    for (field, value) in [
        ("profile", "canonical"),
        ("backend", "cpu"),
        ("logical_input_layout", "nchw"),
    ] {
        if manifest.contract[field].as_str() != Some(value) {
            return Err(invalid(format!("DET reference requires {field}={value:?}")));
        }
    }
    if !matches!(
        manifest.contract["nodes"].as_str(),
        Some("all" | "checkpoints")
    ) {
        return Err(invalid("DET reference requires nodes=all or checkpoints"));
    }
    Ok(())
}
pub(crate) fn is_detector_artifact(id: &str) -> Result<bool> {
    if matches!(id, "det.input" | "det.probabilities") {
        return Ok(true);
    }
    if let Some(ordinal) = id.strip_prefix("det.node.") {
        if ordinal.is_empty() || !ordinal.bytes().all(|b| b.is_ascii_digit()) {
            return Err(invalid(format!("malformed detector artifact ID {id:?}")));
        }
        return Ok(true);
    }
    Ok(false)
}
pub(crate) fn detector_selection(manifest: &Manifest) -> Result<DetSelection> {
    require_canonical(manifest)?;
    let mut cases = Vec::new();
    let mut artifacts = HashSet::new();
    for (ci, case) in manifest.cases.iter().enumerate() {
        let mut input = None;
        let mut probabilities = None;
        let mut nodes = Vec::new();
        let mut ordinals = HashSet::new();
        for (ai, a) in case.artifacts.iter().enumerate() {
            if !is_detector_artifact(&a.id)? {
                continue;
            }
            if a.dtype != Dtype::F32 {
                return Err(invalid(format!("{} must be f32", a.id)));
            }
            artifacts.insert((ci, ai));
            match a.id.as_str() {
                "det.input" => {
                    if a.shape.len() != 4
                        || a.shape[0] != 1
                        || a.shape[1] != 3
                        || a.shape[2..].contains(&0)
                        || a.layout != "nchw"
                    {
                        return Err(invalid(
                            "det.input requires [1,3,H,W] NCHW positive dimensions",
                        ));
                    }
                    input = Some(ai);
                }
                "det.probabilities" => {
                    if a.shape.len() != 4
                        || a.shape[0..2] != [1, 1]
                        || a.shape[2..].contains(&0)
                        || a.layout != "nchw"
                    {
                        return Err(invalid("det.probabilities requires [1,1,H,W] NCHW"));
                    }
                    probabilities = Some(ai)
                }
                _ => {
                    let ordinal: usize = a.id[9..]
                        .parse()
                        .map_err(|_| invalid("invalid detector node ordinal"))?;
                    if !ordinals.insert(ordinal) {
                        return Err(invalid("duplicate detector node ordinal"));
                    }
                    nodes.push((ai, ordinal));
                }
            }
        }
        if input.is_none() && probabilities.is_none() && nodes.is_empty() {
            continue;
        }
        if nodes.is_empty() {
            return Err(invalid(format!(
                "case {} has no captured detector nodes",
                case.id
            )));
        }
        cases.push(DetCase {
            case_index: ci,
            input_index: input.ok_or_else(|| invalid("missing det.input"))?,
            nodes,
            probabilities_index: probabilities
                .ok_or_else(|| invalid("missing det.probabilities"))?,
        });
        let current = cases.last().unwrap();
        if case.artifacts[current.input_index].shape[2..]
            != case.artifacts[current.probabilities_index].shape[2..]
        {
            return Err(invalid("detector probability/input spatial shapes differ"));
        }
    }
    if cases.is_empty() {
        return Err(invalid("reference contains no captured detector inputs"));
    }
    Ok(DetSelection { cases, artifacts })
}
pub(crate) struct ValidatedReference {
    pub bundle: Bundle,
    pub validation: ValidationReport,
    pub manifest_sha256: String,
}
impl ValidatedReference {
    pub fn load(directory: &Path) -> Result<Self> {
        let bundle = Bundle::load(directory)?;
        let mut validation = ValidationReport::default();
        validate_assets(&bundle, &mut validation)?;
        for case in &bundle.manifest.cases {
            for artifact in &case.artifacts {
                validate_artifact(&bundle, artifact, &mut validation)?;
            }
        }
        let bytes = read_bounded(
            File::open(safe_file(&bundle.root, "manifest.json")?)
                .map_err(|e| invalid(e.to_string()))?,
            "manifest.json",
        )?;
        Ok(Self {
            bundle,
            validation,
            manifest_sha256: format!("{:x}", Sha256::digest(bytes)),
        })
    }
    pub fn model_path(&self, id: &str) -> Result<PathBuf> {
        let asset = self
            .bundle
            .manifest
            .models
            .iter()
            .find(|m| m.id == id)
            .ok_or_else(|| invalid(format!("missing model {id}")))?;
        safe_file(&self.bundle.root, &asset.path)
    }
    pub fn read_f32(&self, artifact: &ArtifactDescriptor) -> Result<Vec<f32>> {
        if artifact.dtype != Dtype::F32 {
            return Err(invalid("expected f32 artifact"));
        }
        let count = usize::try_from(artifact.bytes / 4)
            .map_err(|_| invalid("tensor does not fit usize"))?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|e| invalid(e.to_string()))?;
        let mut file = File::open(safe_file(&self.bundle.root, &artifact.path)?)
            .map_err(|e| invalid(e.to_string()))?;
        let mut buffer = [0u8; crate::STREAM_BLOCK_BYTES];
        let mut remaining = artifact.bytes;
        let mut hash = Sha256::new();
        while remaining != 0 {
            let n = remaining.min(buffer.len() as u64) as usize;
            file.read_exact(&mut buffer[..n])
                .map_err(|e| invalid(e.to_string()))?;
            hash.update(&buffer[..n]);
            for bytes in buffer[..n].chunks_exact(4) {
                let value = f32::from_le_bytes(bytes.try_into().unwrap());
                if !value.is_finite() {
                    return Err(invalid("nonfinite frozen input"));
                }
                values.push(value)
            }
            remaining -= n as u64;
        }
        let mut trailing = [0];
        if file
            .read(&mut trailing)
            .map_err(|e| invalid(e.to_string()))?
            != 0
            || !format!("{:x}", hash.finalize()).eq_ignore_ascii_case(&artifact.sha256)
        {
            return Err(invalid("frozen tensor integrity differs"));
        }
        Ok(values)
    }
}
#[derive(Deserialize)]
struct Inventory {
    id: String,
    #[serde(default)]
    inputs: Option<Vec<String>>,
    #[serde(default)]
    outputs: Option<Vec<String>>,
    nodes: Vec<InventoryNode>,
}
#[derive(Deserialize)]
struct InventoryNode {
    index: usize,
    name: String,
    op_type: String,
    domain: String,
    opset: i64,
    #[serde(default)]
    inputs: Option<serde_json::Value>,
    #[serde(default)]
    outputs: Option<serde_json::Value>,
}
pub(crate) fn verify_inventory(
    reference: &ValidatedReference,
    model: &Model,
    id: &str,
    nodes: &[(usize, usize)],
) -> Result<()> {
    verify_inventory_bundle(&reference.bundle, model, id, nodes)
}
pub(crate) fn verify_inventory_bundle(
    bundle: &Bundle,
    model: &Model,
    id: &str,
    nodes: &[(usize, usize)],
) -> Result<()> {
    let bytes = read_bounded(
        File::open(safe_file(&bundle.root, "graph-inventory.json")?)
            .map_err(|e| invalid(e.to_string()))?,
        "graph-inventory.json",
    )?;
    let all: Vec<Inventory> = serde_json::from_slice(&bytes).map_err(|e| invalid(e.to_string()))?;
    let selected: Vec<_> = all.iter().filter(|i| i.id == id).collect();
    let graph = model.graph();
    if selected.len() != 1 || selected[0].nodes.len() != graph.nodes.len() {
        return Err(invalid(format!("{id} graph inventory count differs")));
    }
    let saved = selected[0];
    let semantic = bundle.manifest.contract["capture_scope"] == "step12";
    if semantic && (saved.inputs.is_none() || saved.outputs.is_none()) {
        return Err(invalid(
            "Step12 graph inventory requires semantic input/output identity",
        ));
    }
    let tensor_keys = |ids: &[crate::model::TensorId]| {
        ids.iter()
            .map(|id| graph.tensors[id.0].key.clone())
            .collect::<Vec<_>>()
    };
    if saved
        .inputs
        .as_ref()
        .is_some_and(|x| *x != tensor_keys(&graph.inputs))
        || saved
            .outputs
            .as_ref()
            .is_some_and(|x| *x != tensor_keys(&graph.outputs))
    {
        return Err(invalid(format!("{id} graph input/output identity differs")));
    }
    for (ordinal, (saved, node)) in saved.nodes.iter().zip(&graph.nodes).enumerate() {
        if saved.index != ordinal
            || saved.name != node.name
            || saved.op_type != node.op_type
            || saved.domain != node.domain
            || saved.opset != node.opset
        {
            return Err(invalid(format!(
                "{id} inventory differs at normalized node {ordinal}"
            )));
        }
        let keys = |ids: &[Option<crate::model::TensorId>]| {
            json!(
                ids.iter()
                    .map(|id| id.map(|id| graph.tensors[id.0].key.as_str()))
                    .collect::<Vec<_>>()
            )
        };
        // Legacy inventories store numeric C# indexes; new ones preserve semantic lineage keys.
        for (actual, expected) in [
            (&saved.inputs, keys(&node.inputs)),
            (&saved.outputs, keys(&node.outputs)),
        ] {
            if semantic
                && actual.as_ref().is_none_or(|value| {
                    value
                        .as_array()
                        .is_none_or(|items| items.iter().any(|v| !v.is_string() && !v.is_null()))
                })
            {
                return Err(invalid("Step12 node inventory requires semantic edges"));
            }
            if actual.as_ref().is_some_and(|x| {
                x.as_array()
                    .is_some_and(|items| items.iter().all(|v| v.is_string() || v.is_null()))
                    && *x != expected
            }) {
                return Err(invalid(format!(
                    "{id} inventory edges differ at normalized node {ordinal}"
                )));
            }
        }
    }
    let ordinals: HashSet<_> = nodes.iter().map(|(_, ordinal)| *ordinal).collect();
    if ordinals.len() != nodes.len() {
        return Err(invalid(format!("duplicate {id} node ordinal")));
    }
    if bundle.manifest.contract["nodes"] == "all"
        && (ordinals.len() != graph.nodes.len()
            || (0..graph.nodes.len()).any(|ordinal| !ordinals.contains(&ordinal)))
    {
        return Err(invalid(format!("all-node {id} reference missing captures")));
    }
    if nodes.iter().any(|(_, n)| *n >= graph.nodes.len()) {
        return Err(invalid(format!(
            "{id} node ordinal outside normalized graph"
        )));
    }
    Ok(())
}
pub(crate) fn validate_reference_coverage(bundle: &Bundle, selection: &DetSelection) -> Result<()> {
    let asset = bundle
        .manifest
        .models
        .iter()
        .find(|model| model.id == "det")
        .ok_or_else(|| invalid("missing det model"))?;
    let model = Arc::new(
        Model::load_path(safe_file(&bundle.root, &asset.path)?)
            .map_err(|e| invalid(e.to_string()))?,
    );
    let compiled = CompiledModel::new(model.clone()).map_err(|e| invalid(e.to_string()))?;
    for selected in &selection.cases {
        verify_inventory_bundle(bundle, &model, "det", &selected.nodes)?;
        let case = &bundle.manifest.cases[selected.case_index];
        validate_bound_metadata(
            case,
            &selected.nodes,
            &compiled,
            &case.artifacts[selected.input_index].shape,
        )?;
    }
    Ok(())
}
pub(crate) fn validate_bound_metadata(
    case: &crate::bundle::CaseDescriptor,
    nodes: &[(usize, usize)],
    compiled: &CompiledModel,
    input_shape: &[usize],
) -> Result<()> {
    let shapes = compiled
        .node_output_shapes(input_shape)
        .map_err(|e| invalid(e.to_string()))?;
    for (artifact_index, ordinal) in nodes {
        let artifact = &case.artifacts[*artifact_index];
        validate_node_metadata(artifact, compiled.model(), *ordinal)?;
        let shape = shapes
            .get(*ordinal)
            .ok_or_else(|| invalid("trace node ordinal outside bound model"))?;
        if artifact.shape != shape.dimensions() {
            return Err(invalid(format!(
                "{} trace shape differs from bound model",
                artifact.id
            )));
        }
    }
    Ok(())
}
pub(crate) fn validate_node_metadata(
    artifact: &ArtifactDescriptor,
    model: &Model,
    ordinal: usize,
) -> Result<()> {
    let node = &model.graph().nodes[ordinal];
    let output = node
        .outputs
        .first()
        .copied()
        .flatten()
        .ok_or_else(|| invalid("trace node has no output"))?;
    let tensor = &model.graph().tensors[output.0];
    let layout = if artifact.shape.len() == 4 {
        "nchw"
    } else {
        "row_major"
    };
    if artifact.dtype != Dtype::F32 || artifact.layout != layout || artifact.shape.contains(&0) {
        return Err(invalid(format!(
            "{} trace tensor metadata invalid",
            artifact.id
        )));
    }
    if let crate::tensor::ShapeSpec::Ranked(template) = &tensor.shape
        && (template.len() != artifact.shape.len()
            || template.iter().zip(&artifact.shape).any(
                |(dim, value)| matches!(dim,crate::tensor::Dim::Known(expected) if expected!=value),
            ))
    {
        return Err(invalid(format!(
            "{} trace shape differs from normalized model",
            artifact.id
        )));
    }
    Ok(())
}
pub fn load_det_inputs(directory: &Path) -> Result<FrozenDetInputs> {
    let reference = ValidatedReference::load(directory)?;
    let selection = detector_selection(&reference.bundle.manifest)?;
    let model_path = reference.model_path("det")?;
    let model_sha256 = reference
        .bundle
        .manifest
        .models
        .iter()
        .find(|m| m.id == "det")
        .unwrap()
        .sha256
        .clone();
    let mut inputs = Vec::new();
    for selected in selection.cases {
        let case = &reference.bundle.manifest.cases[selected.case_index];
        let artifact = &case.artifacts[selected.input_index];
        inputs.push(FrozenDetInput {
            case_id: case.id.clone(),
            artifact_id: artifact.id.clone(),
            shape: artifact.shape.clone(),
            sha256: artifact.sha256.clone(),
            values: reference.read_f32(artifact)?,
        });
    }
    Ok(FrozenDetInputs {
        model_path,
        model_sha256,
        manifest_sha256: reference.manifest_sha256,
        inputs,
    })
}
pub fn run_det(reference_directory: &Path, output_directory: &Path) -> Result<DetRunReport> {
    let reference = ValidatedReference::load(reference_directory)?;
    let selection = detector_selection(&reference.bundle.manifest)?;
    let model = Arc::new(
        Model::load_path(reference.model_path("det")?).map_err(|e| invalid(e.to_string()))?,
    );
    let compiled = Arc::new(CompiledModel::new(model.clone()).map_err(|e| invalid(e.to_string()))?);
    for case in &selection.cases {
        verify_inventory(&reference, &model, "det", &case.nodes)?;
        let source_case = &reference.bundle.manifest.cases[case.case_index];
        validate_bound_metadata(
            source_case,
            &case.nodes,
            &compiled,
            &source_case.artifacts[case.input_index].shape,
        )?;
    }
    let mut writer =
        BundleWriter::new(output_directory, reference.bundle.manifest.contract.clone())?;
    for asset in &reference.bundle.manifest.models {
        writer.copy_model(
            &asset.id,
            &safe_file(&reference.bundle.root, &asset.path)?,
            &asset.path,
        )?;
    }
    let mut session: Option<Session> = None;
    let mut written_artifacts = 0;
    for (ci, case) in reference.bundle.manifest.cases.iter().enumerate() {
        let mut artifacts = Vec::new();
        if let Some(selected) = selection.cases.iter().find(|x| x.case_index == ci) {
            let input = &case.artifacts[selected.input_index];
            let values = reference.read_f32(input)?;
            artifacts.push(writer.write_floats(
                &input.id,
                &input.path,
                &values,
                &input.shape,
                &input.layout,
            )?);
            match &mut session {
                Some(session) => session
                    .reshape(&input.shape)
                    .map_err(|e| invalid(e.to_string()))?,
                None => {
                    session = Some(
                        Session::new(compiled.clone(), &input.shape, ExecutionLimits::default())
                            .map_err(|e| invalid(e.to_string()))?,
                    )
                }
            }
            let requested: HashMap<_, _> = selected
                .nodes
                .iter()
                .map(|(ai, n)| (*n, &case.artifacts[*ai]))
                .collect();
            let mut observed = HashSet::new();
            let output = session
                .as_mut()
                .unwrap()
                .run_with_trace(&values, |trace| {
                    if let Some(descriptor) = requested.get(&trace.index) {
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
            if observed.len() != requested.len() {
                return Err(invalid("DET omitted a requested trace"));
            }
            let probabilities = &case.artifacts[selected.probabilities_index];
            artifacts.push(writer.write_floats(
                &probabilities.id,
                &probabilities.path,
                output.data(),
                output.shape().dimensions(),
                "nchw",
            )?);
        }
        written_artifacts += artifacts.len();
        writer.add_case(&case.id, &case.kind, artifacts)?;
    }
    let output=writer.finish(json!({"provenance":{"implementation":"rust scalar canonical DET","reference_manifest_sha256":reference.manifest_sha256,"coverage":"frozen normalized detector tensors; no pixel preprocessing, boxes, crops or recognition"}}))?;
    Ok(DetRunReport {
        output,
        processed_cases: selection.cases.len(),
        written_artifacts,
        reference_manifest_sha256: reference.manifest_sha256,
        reference_validation: reference.validation,
    })
}
