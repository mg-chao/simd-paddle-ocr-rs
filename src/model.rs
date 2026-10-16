//! Structural ONNX import. This module does not execute graphs or allocate activations.
use crate::onnx::{
    self, Attribute, AttributeValue, ImportError, LoadLimits, RawGraph, RawModel, RawNode,
    RawTensor, ValueInfo,
};
use crate::tensor::{DType, Dim, Shape, ShapeSpec, Tensor, TensorData};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    fs::File,
    io::{BufRead, BufReader, Cursor},
    path::Path,
    time::Instant,
};

type Result<T> = std::result::Result<T, ImportError>;
fn error(message: impl Into<String>) -> ImportError {
    ImportError::new(message)
}

/// Limits apply before allocation; diagnostics are opt-in and never retain raw weights.
#[derive(Clone, Debug, Default)]
pub struct LoadOptions {
    pub limits: LoadLimits,
    pub capture_raw_snapshot: bool,
    pub measure_timings: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct ImportTimings {
    pub parse_ms: f64,
    pub normalize_ms: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct TensorId(pub usize);

#[derive(Debug)]
pub struct TensorDescriptor {
    pub key: String,
    pub name: String,
    pub dtype: DType,
    pub shape: ShapeSpec,
    pub constant: Option<TensorData>,
}

#[derive(Clone, Debug)]
pub struct ConvAttributes {
    pub kernel_shape: [i64; 2],
    pub strides: [i64; 2],
    pub dilations: [i64; 2],
    pub pads: [i64; 4],
    pub group: i64,
}
#[derive(Clone, Debug)]
pub struct PoolAttributes {
    pub kernel_shape: [i64; 2],
    pub strides: [i64; 2],
    pub pads: [i64; 4],
    pub ceil_mode: i64,
    pub count_include_pad: i64,
}
#[derive(Clone, Debug)]
pub struct ResizeAttributes {
    pub scales: Vec<f32>,
    pub mode: Vec<u8>,
    pub coordinate_transformation_mode: Vec<u8>,
    pub nearest_mode: Vec<u8>,
}

/// Typed effective parameters; source protobuf layout and C# parameter blocks are not runtime APIs.
#[derive(Clone, Debug)]
pub enum EffectiveAttributes {
    None,
    Conv(ConvAttributes),
    ConvTranspose(ConvAttributes),
    Pool(PoolAttributes),
    BatchNormalization {
        epsilon: f32,
        momentum: f32,
        training_mode: i64,
    },
    HardSigmoid {
        alpha: f32,
        beta: f32,
    },
    Axes(Vec<i64>),
    Transpose(Vec<i64>),
    Axis(i64),
    ReduceMean {
        axes: Vec<i64>,
        keepdims: i64,
        noop_with_empty_axes: i64,
    },
    Resize(ResizeAttributes),
    Reshape {
        allowzero: i64,
    },
    Slice(BTreeMap<String, CanonicalAttribute>),
}

#[derive(Clone, Debug)]
pub struct Node {
    pub key: String,
    pub name: String,
    pub op_type: String,
    pub domain: String,
    pub opset: i64,
    pub source_indices: Vec<usize>,
    pub inputs: Vec<Option<TensorId>>,
    pub outputs: Vec<Option<TensorId>>,
    pub attributes: EffectiveAttributes,
}

#[derive(Debug)]
pub struct Graph {
    pub tensors: Vec<TensorDescriptor>,
    pub nodes: Vec<Node>,
    pub inputs: Vec<TensorId>,
    pub outputs: Vec<TensorId>,
}

/// A stripped metadata relationship retained for runtime shape binding without executing it during import.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ShapeRecipe {
    pub node_key: String,
    pub data_input: String,
    pub shape_input: String,
    pub template: CanonicalShape,
    pub source_indices: Vec<usize>,
}

#[derive(Debug)]
pub struct Model {
    pub(crate) packed_weights: crate::runtime::packing::WeightCache,
    ir_version: i64,
    opsets: BTreeMap<String, i64>,
    source_bytes: u64,
    graph: Graph,
    shape_recipes: Vec<ShapeRecipe>,
    shape_metadata: ShapeMetadata,
    // Runtime bindings resolve original Shape sources without altering the
    // authoritative raw metadata or normalized inspection contract.
    shape_data_sources: HashMap<String, TensorId>,
    raw_snapshot: Option<ImportSnapshot>,
    timings: Option<ImportTimings>,
}

impl Model {
    /// Shared packed capacities retained by all compiled views of this model.
    /// These bytes are separate from normalized coefficients and request arenas.
    pub fn cached_packed_weight_bytes(&self) -> usize {
        self.packed_weights.bytes()
    }
    pub fn graph(&self) -> &Graph {
        &self.graph
    }
    pub fn ir_version(&self) -> i64 {
        self.ir_version
    }
    pub fn opsets(&self) -> &BTreeMap<String, i64> {
        &self.opsets
    }
    pub fn source_bytes(&self) -> u64 {
        self.source_bytes
    }
    pub fn shape_recipes(&self) -> &[ShapeRecipe] {
        &self.shape_recipes
    }
    pub fn shape_metadata(&self) -> &ShapeMetadata {
        &self.shape_metadata
    }
    pub(crate) fn shape_data_source(&self, name: &str) -> Option<TensorId> {
        self.shape_data_sources.get(name).copied()
    }
    pub fn load_path(path: impl AsRef<Path>) -> Result<Self> {
        Self::load_path_with_options(path, &LoadOptions::default())
    }
    pub fn load_reader<R: BufRead>(reader: R) -> Result<Self> {
        Self::load_reader_with_options(reader, &LoadOptions::default())
    }
    pub fn load_bytes(bytes: &[u8]) -> Result<Self> {
        Self::load_bytes_with_options(bytes, &LoadOptions::default())
    }
    pub fn load_path_with_options(path: impl AsRef<Path>, options: &LoadOptions) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path).map_err(|e| error(format!("{}: {e}", path.display())))?;
        if file.metadata().map_err(|e| error(e.to_string()))?.len() > options.limits.max_input_bytes
        {
            return Err(error("model exceeds max_input_bytes"));
        }
        Self::load_reader_with_options(BufReader::with_capacity(64 * 1024, file), options)
    }
    pub fn load_bytes_with_options(bytes: &[u8], options: &LoadOptions) -> Result<Self> {
        Self::load_reader_with_options(Cursor::new(bytes), options)
    }
    pub fn load_reader_with_options<R: BufRead>(reader: R, options: &LoadOptions) -> Result<Self> {
        let start = options.measure_timings.then(Instant::now);
        let raw = onnx::parse(reader, &options.limits)?;
        let parse_ms = start.map(|s| s.elapsed().as_secs_f64() * 1000.0);
        let snapshot = if options.capture_raw_snapshot {
            Some(raw_snapshot(&raw)?)
        } else {
            None
        };
        let start = options.measure_timings.then(Instant::now);
        let mut model = normalize(raw, snapshot, &options.limits)?;
        model.timings = parse_ms.zip(start).map(|(parse_ms, start)| ImportTimings {
            parse_ms,
            normalize_ms: start.elapsed().as_secs_f64() * 1000.0,
        });
        Ok(model)
    }
    pub fn import_timings(&self) -> Option<&ImportTimings> {
        self.timings.as_ref()
    }
    pub fn normalized_weight_bytes(&self) -> u64 {
        self.graph
            .tensors
            .iter()
            .filter_map(|t| t.constant.as_ref())
            .map(|d| {
                (d.len()
                    * match d.dtype() {
                        DType::F32 | DType::I32 => 4,
                        DType::I64 => 8,
                        DType::U8 => 1,
                    }) as u64
            })
            .sum()
    }
    pub fn raw_snapshot(&self) -> Option<&ImportSnapshot> {
        self.raw_snapshot.as_ref()
    }
    pub fn normalized_snapshot(&self) -> Result<ImportSnapshot> {
        let tensors = self
            .graph
            .tensors
            .iter()
            .map(|t| {
                let (bytes, hash) = t.constant.as_ref().map(data_summary).unwrap_or((0, None));
                SnapshotTensor {
                    key: t.key.clone(),
                    name: t.name.clone(),
                    dtype: dtype_name(t.dtype).into(),
                    shape: canonical_shape(&t.shape),
                    constant: t.constant.is_some(),
                    data_bytes: bytes,
                    data_sha256: hash,
                    data_path: None,
                }
            })
            .collect::<Vec<_>>();
        let nodes = self
            .graph
            .nodes
            .iter()
            .map(|n| SnapshotNode {
                key: n.key.clone(),
                name: n.name.clone(),
                op_type: n.op_type.clone(),
                domain: n.domain.clone(),
                opset: n.opset,
                source_indices: n.source_indices.clone(),
                inputs: n
                    .inputs
                    .iter()
                    .map(|t| t.map(|id| self.graph.tensors[id.0].key.clone()))
                    .collect(),
                outputs: n
                    .outputs
                    .iter()
                    .map(|t| t.map(|id| self.graph.tensors[id.0].key.clone()))
                    .collect(),
                attributes: n.attributes.canonical(&n.op_type),
            })
            .collect();
        let mut snapshot = ImportSnapshot {
            schema_version: 1,
            stage: "normalized".into(),
            ir_version: self.ir_version,
            opsets: snapshot_opsets(&self.opsets),
            inputs: self
                .graph
                .inputs
                .iter()
                .map(|id| self.graph.tensors[id.0].key.clone())
                .collect(),
            outputs: self
                .graph
                .outputs
                .iter()
                .map(|id| self.graph.tensors[id.0].key.clone())
                .collect(),
            tensors,
            nodes,
            recipes: self.shape_recipes.clone(),
            shape_metadata: metadata_snapshot(&self.shape_metadata, &self.opsets),
        };
        snapshot.tensors.sort_by(|a, b| a.key.cmp(&b.key));
        Ok(snapshot)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CanonicalShape {
    pub rank_known: bool,
    pub dims: Vec<CanonicalDim>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum CanonicalDim {
    Known { value: usize },
    Symbol { value: String },
    Unknown,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum CanonicalAttribute {
    Int { value: i64 },
    Ints { values: Vec<i64> },
    Float { bits: u32 },
    Floats { bits: Vec<u32> },
    Bytes { hex: String },
    Strings { hex: Vec<String> },
    Tensor { value: Box<SnapshotTensor> },
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SnapshotTensor {
    pub key: String,
    pub name: String,
    pub dtype: String,
    pub shape: CanonicalShape,
    pub constant: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data_path: Option<String>,
    pub data_bytes: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data_sha256: Option<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SnapshotNode {
    pub key: String,
    pub name: String,
    pub op_type: String,
    pub domain: String,
    pub opset: i64,
    pub source_indices: Vec<usize>,
    pub inputs: Vec<Option<String>>,
    pub outputs: Vec<Option<String>>,
    pub attributes: BTreeMap<String, CanonicalAttribute>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SnapshotOpset {
    pub domain: String,
    pub version: i64,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ImportSnapshot {
    pub schema_version: u32,
    pub stage: String,
    pub ir_version: i64,
    pub opsets: Vec<SnapshotOpset>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub tensors: Vec<SnapshotTensor>,
    pub nodes: Vec<SnapshotNode>,
    #[serde(default)]
    pub recipes: Vec<ShapeRecipe>,
    #[serde(default)]
    pub shape_metadata: SnapshotShapeMetadata,
}

/// Owned compact integer shape program. No activation or source model payload is retained.
#[derive(Debug, Default)]
pub struct ShapeMetadata {
    pub nodes: Vec<RawNode>,
    pub constants: Vec<RawTensor>,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct SnapshotShapeMetadata {
    pub nodes: Vec<SnapshotNode>,
    pub constants: Vec<SnapshotShapeConstant>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SnapshotShapeConstant {
    pub key: String,
    pub name: String,
    pub dtype: String,
    pub shape: CanonicalShape,
    pub values: Vec<i64>,
}

fn metadata_snapshot(
    metadata: &ShapeMetadata,
    opsets: &BTreeMap<String, i64>,
) -> SnapshotShapeMetadata {
    SnapshotShapeMetadata {
        nodes: metadata
            .nodes
            .iter()
            .map(|n| SnapshotNode {
                key: node_key(n),
                name: n.name.clone(),
                op_type: n.op_type.clone(),
                domain: n.domain.clone(),
                opset: *opsets.get(domain_key(&n.domain)).unwrap_or(&0),
                source_indices: n.source_indices.clone(),
                inputs: n
                    .inputs
                    .iter()
                    .map(|name| {
                        if name.is_empty() {
                            None
                        } else {
                            Some(original(name))
                        }
                    })
                    .collect(),
                outputs: n
                    .outputs
                    .iter()
                    .map(|name| {
                        if name.is_empty() {
                            None
                        } else {
                            Some(original(name))
                        }
                    })
                    .collect(),
                attributes: n
                    .attributes
                    .iter()
                    .map(|a| (a.name.clone(), canonical_attr(&a.value)))
                    .collect(),
            })
            .collect(),
        constants: metadata
            .constants
            .iter()
            .map(|c| SnapshotShapeConstant {
                key: original(&c.name),
                name: c.name.clone(),
                dtype: dtype_name(c.data.dtype()).into(),
                shape: canonical_shape(&constant_shape(&c.data)),
                values: match &c.data {
                    TensorData::I64(t) => t.data().to_vec(),
                    TensorData::I32(t) => t.data().iter().map(|&v| i64::from(v)).collect(),
                    _ => unreachable!("shape metadata constants are integer typed"),
                },
            })
            .collect(),
    }
}

fn dtype_name(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "f32",
        DType::I32 => "i32",
        DType::I64 => "i64",
        DType::U8 => "u8",
    }
}
fn original(name: &str) -> String {
    format!("original:{name}")
}
fn node_key(node: &RawNode) -> String {
    format!("node:{}", node.source_indices[0])
}
fn domain_key(domain: &str) -> &str {
    if domain == "ai.onnx" { "" } else { domain }
}
fn snapshot_opsets(opsets: &BTreeMap<String, i64>) -> Vec<SnapshotOpset> {
    opsets
        .iter()
        .map(|(d, &v)| SnapshotOpset {
            domain: d.clone(),
            version: v,
        })
        .collect()
}
fn canonical_shape(shape: &ShapeSpec) -> CanonicalShape {
    match shape {
        ShapeSpec::UnknownRank => CanonicalShape {
            rank_known: false,
            dims: vec![],
        },
        ShapeSpec::Ranked(dims) => CanonicalShape {
            rank_known: true,
            dims: dims
                .iter()
                .map(|d| match d {
                    Dim::Known(value) => CanonicalDim::Known { value: *value },
                    Dim::Symbol(value) => CanonicalDim::Symbol {
                        value: value.clone(),
                    },
                    Dim::Unknown => CanonicalDim::Unknown,
                })
                .collect(),
        },
    }
}
fn constant_shape(data: &TensorData) -> ShapeSpec {
    ShapeSpec::Ranked(data.dimensions().iter().map(|&d| Dim::Known(d)).collect())
}
fn normalized_shape(shape: &ShapeSpec) -> ShapeSpec {
    shape.clone()
}
fn data_summary(data: &TensorData) -> (u64, Option<String>) {
    let mut sha = Sha256::new();
    data_chunks(data, |bytes| {
        sha.update(bytes);
        Ok(())
    })
    .expect("hash sink is infallible");
    (
        (data.len()
            * match data.dtype() {
                DType::F32 | DType::I32 => 4,
                DType::I64 => 8,
                DType::U8 => 1,
            }) as u64,
        Some(format!("{:x}", sha.finalize())),
    )
}
fn data_chunks(data: &TensorData, mut sink: impl FnMut(&[u8]) -> Result<()>) -> Result<()> {
    let mut buffer = [0u8; 64 * 1024];
    match data {
        TensorData::F32(t) => {
            for values in t.data().chunks(buffer.len() / 4) {
                for (i, v) in values.iter().enumerate() {
                    buffer[i * 4..i * 4 + 4].copy_from_slice(&v.to_bits().to_le_bytes());
                }
                sink(&buffer[..values.len() * 4])?;
            }
        }
        TensorData::I32(t) => {
            for values in t.data().chunks(buffer.len() / 4) {
                for (i, v) in values.iter().enumerate() {
                    buffer[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
                }
                sink(&buffer[..values.len() * 4])?;
            }
        }
        TensorData::I64(t) => {
            for values in t.data().chunks(buffer.len() / 8) {
                for (i, v) in values.iter().enumerate() {
                    buffer[i * 8..i * 8 + 8].copy_from_slice(&v.to_le_bytes());
                }
                sink(&buffer[..values.len() * 8])?;
            }
        }
        TensorData::U8(t) => {
            for bytes in t.data().chunks(buffer.len()) {
                sink(bytes)?;
            }
        }
    }
    Ok(())
}
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
fn canonical_attr(attr: &AttributeValue) -> CanonicalAttribute {
    match attr {
        AttributeValue::Float(v) => CanonicalAttribute::Float { bits: v.to_bits() },
        AttributeValue::Int(v) => CanonicalAttribute::Int { value: *v },
        AttributeValue::String(v) => CanonicalAttribute::Bytes { hex: hex(v) },
        AttributeValue::Floats(v) => CanonicalAttribute::Floats {
            bits: v.iter().map(|v| v.to_bits()).collect(),
        },
        AttributeValue::Ints(v) => CanonicalAttribute::Ints { values: v.clone() },
        AttributeValue::Strings(v) => CanonicalAttribute::Strings {
            hex: v.iter().map(|v| hex(v)).collect(),
        },
        AttributeValue::Tensor(t) => {
            let (bytes, hash) = data_summary(&t.data);
            CanonicalAttribute::Tensor {
                value: Box::new(SnapshotTensor {
                    key: original(&t.name),
                    name: t.name.clone(),
                    dtype: dtype_name(t.data.dtype()).into(),
                    shape: canonical_shape(&constant_shape(&t.data)),
                    constant: true,
                    data_path: None,
                    data_bytes: bytes,
                    data_sha256: hash,
                }),
            }
        }
    }
}

fn raw_snapshot(raw: &RawModel) -> Result<ImportSnapshot> {
    let mut tensors = BTreeMap::<String, SnapshotTensor>::new();
    for value in raw
        .graph
        .inputs
        .iter()
        .chain(&raw.graph.value_info)
        .chain(&raw.graph.outputs)
    {
        tensors.insert(
            value.name.clone(),
            SnapshotTensor {
                key: original(&value.name),
                name: value.name.clone(),
                dtype: dtype_name(value.dtype).into(),
                shape: canonical_shape(&value.shape),
                constant: false,
                data_path: None,
                data_bytes: 0,
                data_sha256: None,
            },
        );
    }
    for node in &raw.graph.nodes {
        for name in node
            .inputs
            .iter()
            .chain(&node.outputs)
            .filter(|n| !n.is_empty())
        {
            tensors
                .entry(name.clone())
                .or_insert_with(|| SnapshotTensor {
                    key: original(name),
                    name: name.clone(),
                    dtype: "unknown".into(),
                    shape: CanonicalShape {
                        rank_known: false,
                        dims: vec![],
                    },
                    constant: false,
                    data_path: None,
                    data_bytes: 0,
                    data_sha256: None,
                });
        }
    }
    for tensor in &raw.graph.initializers {
        let (bytes, hash) = data_summary(&tensor.data);
        tensors.insert(
            tensor.name.clone(),
            SnapshotTensor {
                key: original(&tensor.name),
                name: tensor.name.clone(),
                dtype: dtype_name(tensor.data.dtype()).into(),
                shape: canonical_shape(&constant_shape(&tensor.data)),
                constant: true,
                data_path: None,
                data_bytes: bytes,
                data_sha256: hash,
            },
        );
    }
    let nodes = raw
        .graph
        .nodes
        .iter()
        .map(|node| SnapshotNode {
            key: node_key(node),
            name: node.name.clone(),
            op_type: node.op_type.clone(),
            domain: node.domain.clone(),
            opset: *raw.opsets.get(domain_key(&node.domain)).unwrap_or(&0),
            source_indices: node.source_indices.clone(),
            inputs: node
                .inputs
                .iter()
                .map(|n| {
                    if n.is_empty() {
                        None
                    } else {
                        Some(original(n))
                    }
                })
                .collect(),
            outputs: node
                .outputs
                .iter()
                .map(|n| {
                    if n.is_empty() {
                        None
                    } else {
                        Some(original(n))
                    }
                })
                .collect(),
            attributes: node
                .attributes
                .iter()
                .map(|a| (a.name.clone(), canonical_attr(&a.value)))
                .collect(),
        })
        .collect();
    Ok(ImportSnapshot {
        schema_version: 1,
        stage: "raw".into(),
        ir_version: raw.ir_version,
        opsets: snapshot_opsets(&raw.opsets),
        inputs: raw.graph.inputs.iter().map(|v| original(&v.name)).collect(),
        outputs: raw
            .graph
            .outputs
            .iter()
            .map(|v| original(&v.name))
            .collect(),
        tensors: tensors.into_values().collect(),
        nodes,
        recipes: vec![],
        shape_metadata: SnapshotShapeMetadata::default(),
    })
}

impl EffectiveAttributes {
    fn canonical(&self, op_type: &str) -> BTreeMap<String, CanonicalAttribute> {
        let mut a = BTreeMap::new();
        let int = |v| CanonicalAttribute::Int { value: v };
        let ints = |v: Vec<i64>| CanonicalAttribute::Ints { values: v };
        let float = |v: f32| CanonicalAttribute::Float { bits: v.to_bits() };
        match self {
            Self::None => {}
            Self::Conv(v) | Self::ConvTranspose(v) => {
                a.insert("kernel_shape".into(), ints(v.kernel_shape.to_vec()));
                a.insert("strides".into(), ints(v.strides.to_vec()));
                a.insert("dilations".into(), ints(v.dilations.to_vec()));
                a.insert("pads".into(), ints(v.pads.to_vec()));
                a.insert("group".into(), int(v.group));
            }
            Self::Pool(v) => {
                a.insert("kernel_shape".into(), ints(v.kernel_shape.to_vec()));
                a.insert("strides".into(), ints(v.strides.to_vec()));
                a.insert("pads".into(), ints(v.pads.to_vec()));
                a.insert("ceil_mode".into(), int(v.ceil_mode));
                a.insert("count_include_pad".into(), int(v.count_include_pad));
            }
            Self::BatchNormalization {
                epsilon,
                momentum,
                training_mode,
            } => {
                a.insert("epsilon".into(), float(*epsilon));
                a.insert("momentum".into(), float(*momentum));
                a.insert("training_mode".into(), int(*training_mode));
            }
            Self::HardSigmoid { alpha, beta } => {
                a.insert("alpha".into(), float(*alpha));
                a.insert("beta".into(), float(*beta));
            }
            Self::Axes(v) => {
                a.insert("axes".into(), ints(v.clone()));
            }
            Self::Transpose(v) => {
                a.insert("perm".into(), ints(v.clone()));
            }
            Self::Axis(v) => {
                a.insert("axis".into(), int(*v));
            }
            Self::ReduceMean {
                axes,
                keepdims,
                noop_with_empty_axes,
            } => {
                a.insert("axes".into(), ints(axes.clone()));
                a.insert("keepdims".into(), int(*keepdims));
                a.insert("noop_with_empty_axes".into(), int(*noop_with_empty_axes));
            }
            Self::Resize(v) => {
                a.insert(
                    "scales".into(),
                    CanonicalAttribute::Floats {
                        bits: v.scales.iter().map(|v| v.to_bits()).collect(),
                    },
                );
                a.insert(
                    "mode".into(),
                    CanonicalAttribute::Bytes { hex: hex(&v.mode) },
                );
                a.insert(
                    "coordinate_transformation_mode".into(),
                    CanonicalAttribute::Bytes {
                        hex: hex(&v.coordinate_transformation_mode),
                    },
                );
                a.insert(
                    "nearest_mode".into(),
                    CanonicalAttribute::Bytes {
                        hex: hex(&v.nearest_mode),
                    },
                );
            }
            Self::Reshape { allowzero } => {
                a.insert("allowzero".into(), int(*allowzero));
            }
            Self::Slice(attrs) => a = attrs.clone(),
        }
        let _ = op_type;
        a
    }
}

fn attribute<'a>(node: &'a RawNode, name: &str) -> Option<&'a AttributeValue> {
    node.attributes
        .iter()
        .find(|a| a.name == name)
        .map(|a| &a.value)
}
fn int_attr(node: &RawNode, name: &str, default: i64) -> Result<i64> {
    match attribute(node, name) {
        None => Ok(default),
        Some(AttributeValue::Int(v)) => Ok(*v),
        _ => Err(error(format!("{}: {name} must be an integer", node.name))),
    }
}
fn float_attr(node: &RawNode, name: &str, default: f32) -> Result<f32> {
    match attribute(node, name) {
        None => Ok(default),
        Some(AttributeValue::Float(v)) => Ok(*v),
        _ => Err(error(format!("{}: {name} must be a float", node.name))),
    }
}
fn ints_attr(node: &RawNode, name: &str, default: &[i64]) -> Result<Vec<i64>> {
    match attribute(node, name) {
        None => Ok(default.to_vec()),
        Some(AttributeValue::Ints(v)) => Ok(v.clone()),
        _ => Err(error(format!("{}: {name} must be integers", node.name))),
    }
}
fn bytes_attr(node: &RawNode, name: &str, default: &[u8]) -> Result<Vec<u8>> {
    match attribute(node, name) {
        None => Ok(default.to_vec()),
        Some(AttributeValue::String(v)) => Ok(v.clone()),
        _ => Err(error(format!("{}: {name} must be bytes", node.name))),
    }
}
fn array_attr<const N: usize>(node: &RawNode, name: &str, default: &[i64]) -> Result<[i64; N]> {
    ints_attr(node, name, default)?
        .try_into()
        .map_err(|_| error(format!("{}: {name} needs {N} values", node.name)))
}
fn set_attr(node: &mut RawNode, name: &str, value: AttributeValue) {
    if let Some(a) = node.attributes.iter_mut().find(|a| a.name == name) {
        a.value = value;
    } else {
        node.attributes.push(Attribute {
            name: name.into(),
            value,
        });
    }
}

fn unsupported_attribute(node: &RawNode, name: &str) -> ImportError {
    error(format!(
        "{} ({}): unsupported attribute {name} or value for the OCR CPU subset",
        node.name, node.op_type
    ))
}

/// Validate before transformations can erase source semantics. Attributes stored in
/// EffectiveAttributes are checked by their typed readers and the runtime; other
/// attributes are accepted only when they are documented neutral ONNX defaults.
/// This is an OCR operator subset, not a general ONNX schema validator.
fn validate_attributes(node: &RawNode) -> Result<()> {
    let retained: &[&str] = match node.op_type.as_str() {
        "Conv" | "ConvTranspose" => &[
            "kernel_shape",
            "strides",
            "dilations",
            "pads",
            "group",
            "auto_pad",
        ],
        "MaxPool" | "AveragePool" => &[
            "kernel_shape",
            "strides",
            "pads",
            "ceil_mode",
            "count_include_pad",
            "auto_pad",
        ],
        "BatchNormalization" => &["epsilon", "momentum", "training_mode"],
        "HardSigmoid" => &["alpha", "beta"],
        "Squeeze" | "Unsqueeze" => &["axes"],
        "Transpose" => &["perm"],
        "Softmax" | "Concat" => &["axis"],
        "ReduceMean" => &["axes", "keepdims", "noop_with_empty_axes"],
        "Resize" => &[
            "scales",
            "mode",
            "coordinate_transformation_mode",
            "nearest_mode",
        ],
        "Reshape" => &["allowzero"],
        "Slice" => &["starts", "ends", "axes", "steps"],
        "Constant" => &[
            "value",
            "value_float",
            "value_int",
            "value_floats",
            "value_ints",
        ],
        "Add" | "Mul" | "Div" | "Erf" | "Relu" | "MatMul" | "Sigmoid" | "Sub" | "Pow" | "Sqrt"
        | "Identity" | "GlobalAveragePool" => &[],
        // Shape programs and unsupported operators retain their attributes for
        // their own compiler to diagnose; they do not pass through effective().
        _ => return Ok(()),
    };
    for attr in &node.attributes {
        if retained.contains(&attr.name.as_str()) {
            continue;
        }
        let neutral = match (node.op_type.as_str(), attr.name.as_str(), &attr.value) {
            ("ConvTranspose", "output_padding", AttributeValue::Ints(v)) => v == &[0, 0],
            ("MaxPool" | "AveragePool", "dilations", AttributeValue::Ints(v)) => v == &[1, 1],
            ("MaxPool", "storage_order", AttributeValue::Int(v)) => *v == 0,
            ("BatchNormalization", "spatial" | "is_test", AttributeValue::Int(v)) => *v == 1,
            ("Resize", "antialias" | "exclude_outside", AttributeValue::Int(v)) => *v == 0,
            ("Resize", "cubic_coeff_a", AttributeValue::Float(v)) => *v == -0.75,
            ("Resize", "extrapolation_value", AttributeValue::Float(v)) => *v == 0.0,
            ("Resize", "keep_aspect_ratio_policy", AttributeValue::String(v)) => v == b"stretch",
            ("Resize", "axes", AttributeValue::Ints(v)) => v == &[0, 1, 2, 3],
            _ => false,
        };
        if !neutral {
            return Err(unsupported_attribute(node, &attr.name));
        }
    }
    if node.op_type == "Constant" && node.attributes.len() != 1 {
        return Err(error(format!(
            "{} (Constant): exactly one value attribute is required",
            node.name
        )));
    }
    if node.op_type == "BatchNormalization" {
        // Folding into Conv must never turn training-mode BN into inference BN.
        if int_attr(node, "training_mode", 0)? != 0 {
            return Err(unsupported_attribute(node, "training_mode"));
        }
        for (name, default) in [("epsilon", 1e-5), ("momentum", 0.9)] {
            if !float_attr(node, name, default)?.is_finite() {
                return Err(unsupported_attribute(node, name));
            }
        }
    }
    if matches!(
        node.op_type.as_str(),
        "Conv" | "ConvTranspose" | "MaxPool" | "AveragePool"
    ) {
        let auto_pad = bytes_attr(node, "auto_pad", b"NOTSET")?;
        if auto_pad != b"NOTSET" {
            // VALID is neutral only with zero pads; SAME_UPPER is normalized
            // below only for the supported, input-size-independent stride one.
            let supported_same = auto_pad == b"SAME_UPPER"
                && matches!(node.op_type.as_str(), "Conv" | "MaxPool")
                && ints_attr(node, "strides", &[1, 1])? == [1, 1]
                && ints_attr(node, "kernel_shape", &[])?.len() == 2;
            if (auto_pad != b"VALID" && !supported_same)
                || array_attr::<4>(node, "pads", &[0; 4])? != [0; 4]
            {
                return Err(unsupported_attribute(node, "auto_pad"));
            }
        }
    }
    Ok(())
}

fn effective(node: &RawNode, opset: i64) -> Result<EffectiveAttributes> {
    Ok(match node.op_type.as_str() {
        "Conv" | "ConvTranspose" => {
            let attrs = ConvAttributes {
                kernel_shape: array_attr(node, "kernel_shape", &[])?,
                strides: array_attr(node, "strides", &[1, 1])?,
                dilations: array_attr(node, "dilations", &[1, 1])?,
                pads: array_attr(node, "pads", &[0, 0, 0, 0])?,
                group: int_attr(node, "group", 1)?,
            };
            if attrs.group <= 0
                || attrs
                    .kernel_shape
                    .iter()
                    .chain(&attrs.strides)
                    .chain(&attrs.dilations)
                    .any(|&v| v <= 0)
                || attrs.pads.iter().any(|&v| v < 0)
            {
                return Err(error(format!(
                    "{}: invalid convolution attributes",
                    node.name
                )));
            }
            if bytes_attr(node, "auto_pad", b"NOTSET")? != b"NOTSET" {
                return Err(error(format!(
                    "{}: unsupported auto_pad after normalization",
                    node.name
                )));
            }
            if node.op_type == "Conv" {
                EffectiveAttributes::Conv(attrs)
            } else {
                EffectiveAttributes::ConvTranspose(attrs)
            }
        }
        "AveragePool" | "MaxPool" => EffectiveAttributes::Pool(PoolAttributes {
            kernel_shape: array_attr(node, "kernel_shape", &[])?,
            strides: array_attr(node, "strides", &[1, 1])?,
            pads: array_attr(node, "pads", &[0, 0, 0, 0])?,
            ceil_mode: int_attr(node, "ceil_mode", 0)?,
            count_include_pad: int_attr(node, "count_include_pad", 0)?,
        }),
        "BatchNormalization" => EffectiveAttributes::BatchNormalization {
            epsilon: float_attr(node, "epsilon", 1e-5)?,
            momentum: float_attr(node, "momentum", 0.9)?,
            training_mode: int_attr(node, "training_mode", 0)?,
        },
        "HardSigmoid" => EffectiveAttributes::HardSigmoid {
            alpha: float_attr(node, "alpha", 0.2)?,
            beta: float_attr(node, "beta", 0.5)?,
        },
        "Squeeze" | "Unsqueeze" => EffectiveAttributes::Axes(ints_attr(node, "axes", &[])?),
        "Transpose" => EffectiveAttributes::Transpose(ints_attr(node, "perm", &[])?),
        "Softmax" => {
            EffectiveAttributes::Axis(int_attr(node, "axis", if opset >= 13 { -1 } else { 1 })?)
        }
        "Concat" => EffectiveAttributes::Axis(int_attr(node, "axis", 1)?),
        "ReduceMean" => EffectiveAttributes::ReduceMean {
            axes: ints_attr(node, "axes", &[])?,
            keepdims: int_attr(node, "keepdims", 1)?,
            noop_with_empty_axes: int_attr(node, "noop_with_empty_axes", 0)?,
        },
        "Resize" => {
            let scales = match attribute(node, "scales") {
                Some(AttributeValue::Floats(v)) if v.len() == 4 => v.clone(),
                _ => {
                    return Err(error(format!(
                        "{}: Resize requires four constant scales",
                        node.name
                    )));
                }
            };
            EffectiveAttributes::Resize(ResizeAttributes {
                scales,
                mode: bytes_attr(node, "mode", b"nearest")?,
                coordinate_transformation_mode: bytes_attr(
                    node,
                    "coordinate_transformation_mode",
                    b"asymmetric",
                )?,
                nearest_mode: bytes_attr(node, "nearest_mode", b"floor")?,
            })
        }
        "Reshape" => EffectiveAttributes::Reshape {
            allowzero: int_attr(node, "allowzero", 0)?,
        },
        "Slice" => EffectiveAttributes::Slice(
            node.attributes
                .iter()
                .map(|a| (a.name.clone(), canonical_attr(&a.value)))
                .collect(),
        ),
        "Add" | "Mul" | "Div" | "Erf" | "Relu" | "MatMul" | "Sigmoid" | "Sub" | "Pow" | "Sqrt" => {
            EffectiveAttributes::None
        }
        _ => {
            return Err(error(format!(
                "unsupported retained ONNX operator {} at {}",
                node.op_type, node.name
            )));
        }
    })
}

#[cfg(test)]
mod tests;

struct Constant {
    name: String,
    key: String,
    data: TensorData,
}
struct Constants {
    items: Vec<Option<Constant>>,
    index: HashMap<String, usize>,
}
impl Constants {
    fn new(tensors: Vec<RawTensor>) -> Result<Self> {
        let mut s = Self {
            items: vec![],
            index: HashMap::new(),
        };
        for t in tensors {
            s.insert(t.name.clone(), original(&t.name), t.data)?;
        }
        Ok(s)
    }
    fn insert(&mut self, name: String, key: String, data: TensorData) -> Result<()> {
        if self.index.contains_key(&name) {
            return Err(error(format!("duplicate initializer {name}")));
        }
        self.index.insert(name.clone(), self.items.len());
        self.items.push(Some(Constant { name, key, data }));
        Ok(())
    }
    fn get(&self, name: &str) -> Option<&Constant> {
        self.index.get(name).and_then(|&i| self.items[i].as_ref())
    }
    fn unique(&self, base: &str) -> String {
        let mut name = base.to_string();
        let mut serial = 1;
        while self.index.contains_key(&name) {
            name = format!("{base}_{serial}");
            serial += 1;
        }
        name
    }
    fn f32(&self, name: &str) -> Option<&[f32]> {
        self.get(name).and_then(|c| c.data.as_f32().ok())
    }
    fn ints(&self, name: &str) -> Option<Vec<i64>> {
        self.get(name).and_then(|c| match &c.data {
            TensorData::I64(t) => Some(t.data().to_vec()),
            TensorData::I32(t) => Some(t.data().iter().map(|&v| i64::from(v)).collect()),
            _ => None,
        })
    }
    fn take(&mut self, name: &str) -> Option<Constant> {
        self.index.get(name).and_then(|&i| self.items[i].take())
    }
}
struct WorkNode {
    raw: RawNode,
    key: String,
    origin: usize,
}
fn origins(nodes: &[WorkNode]) -> (HashMap<String, usize>, HashMap<String, usize>) {
    let mut p = HashMap::new();
    let mut c = HashMap::new();
    for (i, node) in nodes.iter().enumerate() {
        for name in node.raw.outputs.iter().filter(|n| !n.is_empty()) {
            p.insert(name.clone(), i);
        }
        for name in node.raw.inputs.iter().filter(|n| !n.is_empty()) {
            *c.entry(name.clone()).or_default() += 1;
        }
    }
    (p, c)
}
fn lineage(parts: impl IntoIterator<Item = Vec<usize>>) -> Vec<usize> {
    parts
        .into_iter()
        .flatten()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}
fn f32_tensor(dims: &[usize], values: Vec<f32>) -> Result<TensorData> {
    Ok(TensorData::F32(Tensor::from_vec(
        Shape::new(dims)?,
        values,
    )?))
}
fn i64_tensor(dims: &[usize], values: Vec<i64>) -> Result<TensorData> {
    Ok(TensorData::I64(Tensor::from_vec(
        Shape::new(dims)?,
        values,
    )?))
}

fn resolve(name: &str, aliases: &HashMap<String, String>) -> Result<String> {
    let mut name = name.to_string();
    let mut seen = HashSet::new();
    while let Some(next) = aliases.get(&name) {
        if !seen.insert(name.clone()) {
            return Err(error(format!("Identity alias cycle involving {name}")));
        }
        name = next.clone();
    }
    Ok(name)
}
fn metadata_sources(
    name: &str,
    data_input: &str,
    nodes: &[RawNode],
    producer: &HashMap<String, usize>,
    seen: &mut BTreeSet<usize>,
) {
    if name == data_input {
        return;
    }
    let Some(&i) = producer.get(name) else {
        return;
    };
    let n = &nodes[i];
    if !seen.insert(i) {
        return;
    }
    if n.op_type != "Shape" {
        for input in &n.inputs {
            metadata_sources(input, data_input, nodes, producer, seen);
        }
    }
}
fn normalize(
    raw: RawModel,
    snapshot: Option<ImportSnapshot>,
    limits: &LoadLimits,
) -> Result<Model> {
    let RawModel {
        ir_version,
        opsets,
        graph,
        source_bytes,
    } = raw;
    let RawGraph {
        mut nodes,
        initializers,
        inputs,
        outputs,
        value_info,
    } = graph;
    let mut constants = Constants::new(initializers)?;
    let mut aliases = HashMap::new();
    let mut producer = HashMap::new();
    for (i, n) in nodes.iter().enumerate() {
        let mut attrs = HashSet::new();
        for a in &n.attributes {
            if !attrs.insert(&a.name) {
                return Err(error(format!(
                    "node {} has duplicate attribute {}",
                    n.name, a.name
                )));
            }
        }
        validate_attributes(n)?;
        for name in n.outputs.iter().filter(|n| !n.is_empty()) {
            if producer.insert(name.clone(), i).is_some() {
                return Err(error(format!("duplicate tensor producer {name}")));
            }
        }
        if n.op_type == "Identity"
            && n.inputs.len() == 1
            && n.outputs.len() == 1
            && !n.inputs[0].is_empty()
            && !n.outputs[0].is_empty()
        {
            aliases.insert(n.outputs[0].clone(), n.inputs[0].clone());
        }
    }
    for name in aliases.keys() {
        resolve(name, &aliases)?;
    }
    // Constant tensors are moved out of attributes before any metadata node copy.
    let mut constant_forms = HashMap::<usize, (String, Option<String>)>::new();
    for node in &mut nodes {
        if node.op_type != "Constant" {
            continue;
        }
        if node.outputs.len() != 1 || node.outputs[0].is_empty() {
            return Err(error(format!("invalid Constant {}", node.name)));
        }
        let name = node.outputs[0].clone();
        let position = [
            "value",
            "value_float",
            "value_int",
            "value_floats",
            "value_ints",
        ]
        .iter()
        .find_map(|name| node.attributes.iter().position(|a| a.name == *name))
        .ok_or_else(|| error(format!("unsupported Constant {}", node.name)))?;
        let attribute = node.attributes.remove(position);
        let field = attribute.name;
        let mut tensor_name = None;
        let data = match (field.as_str(), attribute.value) {
            ("value", AttributeValue::Tensor(t)) => {
                tensor_name = Some(t.name);
                t.data
            }
            ("value_float", AttributeValue::Float(v)) => f32_tensor(&[], vec![v])?,
            ("value_int", AttributeValue::Int(v)) => i64_tensor(&[], vec![v])?,
            ("value_floats", AttributeValue::Floats(v)) => f32_tensor(&[v.len()], v)?,
            ("value_ints", AttributeValue::Ints(v)) => i64_tensor(&[v.len()], v)?,
            _ => {
                return Err(error(format!(
                    "invalid Constant {field} form at {}",
                    node.name
                )));
            }
        };
        constant_forms.insert(node.source_indices[0], (field, tensor_name));
        constants.insert(name.clone(), original(&name), data)?;
    }
    let mut info = HashMap::<String, ValueInfo>::new();
    for value in inputs.iter().chain(&value_info).chain(&outputs) {
        info.insert(value.name.clone(), value.clone());
    }
    let input_names = inputs
        .iter()
        .filter(|v| constants.get(&v.name).is_none())
        .map(|v| resolve(&v.name, &aliases))
        .collect::<Result<Vec<_>>>()?;
    let output_names = outputs
        .iter()
        .map(|v| resolve(&v.name, &aliases))
        .collect::<Result<Vec<_>>>()?;
    if input_names.len() != 1 || output_names.len() != 1 {
        return Err(error("only one-input/one-output OCR graphs are supported"));
    }
    let mut recipes = vec![];
    for node in &nodes {
        if node.op_type == "Reshape" && node.inputs.len() > 1 {
            let shape_input = resolve(&node.inputs[1], &aliases)?;
            if constants.get(&shape_input).is_none() {
                let data_input = resolve(&node.inputs[0], &aliases)?;
                let mut sources = BTreeSet::new();
                metadata_sources(&shape_input, &data_input, &nodes, &producer, &mut sources);
                let template = node
                    .outputs
                    .first()
                    .and_then(|name| info.get(name))
                    .map(|v| canonical_shape(&v.shape))
                    .unwrap_or_else(|| canonical_shape(&ShapeSpec::UnknownRank));
                recipes.push(ShapeRecipe {
                    node_key: node_key(node),
                    data_input: original(&data_input),
                    shape_input: original(&shape_input),
                    template,
                    source_indices: sources.into_iter().collect(),
                });
            }
        }
    }
    let mut transformed = vec![];
    let metadata_indices = recipes
        .iter()
        .flat_map(|r| &r.source_indices)
        .copied()
        .collect::<HashSet<_>>();
    let mut metadata_nodes = nodes
        .iter()
        .filter(|n| metadata_indices.contains(&n.source_indices[0]))
        .cloned()
        .collect::<Vec<_>>();
    for node in &mut metadata_nodes {
        if let Some((field, tensor_name)) = constant_forms.get(&node.source_indices[0]) {
            let constant = constants
                .get(&node.outputs[0])
                .expect("promoted Constant exists");
            if !matches!(constant.data.dtype(), DType::I32 | DType::I64) {
                return Err(error("shape-program Constant must be integer typed"));
            }
            let value = match field.as_str() {
                "value" => AttributeValue::Tensor(Box::new(RawTensor {
                    name: tensor_name.clone().unwrap_or_default(),
                    data: constant.data.clone(),
                })),
                "value_int" => AttributeValue::Int(
                    constants.ints(&constant.name).expect("integer constant")[0],
                ),
                "value_ints" => {
                    AttributeValue::Ints(constants.ints(&constant.name).expect("integer constant"))
                }
                _ => return Err(error("shape-program Constant form is not integer typed")),
            };
            set_attr(node, field, value);
        }
    }
    let metadata_names = metadata_nodes
        .iter()
        .flat_map(|n| &n.inputs)
        .filter(|name| constants.get(name).is_some())
        .cloned()
        .collect::<BTreeSet<_>>();
    let mut metadata_constants = vec![];
    for name in metadata_names {
        let c = constants.get(&name).expect("collected existing constant");
        let data = match &c.data {
            TensorData::I64(t) => TensorData::I64(Tensor::from_vec(t.shape(), t.data().to_vec())?),
            TensorData::I32(t) => TensorData::I32(Tensor::from_vec(t.shape(), t.data().to_vec())?),
            _ => {
                return Err(error(format!(
                    "shape metadata constant {name} is not integer typed"
                )));
            }
        };
        metadata_constants.push(RawTensor { name, data });
    }
    let mut shape_metadata = ShapeMetadata {
        nodes: metadata_nodes,
        constants: metadata_constants,
    };
    for mut node in nodes {
        if matches!(node.op_type.as_str(), "Identity" | "Constant") {
            continue;
        }
        let origin = node.source_indices[0];
        if node.op_type == "GlobalAveragePool" {
            node.op_type = "ReduceMean".into();
            node.attributes.clear();
            set_attr(&mut node, "axes", AttributeValue::Ints(vec![2, 3]));
            set_attr(&mut node, "keepdims", AttributeValue::Int(1));
        } else if node.op_type == "Reshape" && node.inputs.len() > 1 {
            if constants
                .get(&resolve(&node.inputs[1], &aliases)?)
                .is_none()
            {
                node.inputs.truncate(1);
            }
        } else if node.op_type == "Resize" && node.inputs.len() >= 3 {
            // C# checks the direct initializer name here; aliases are resolved afterward.
            if let Some(scales) = constants.f32(&node.inputs[2])
                && scales.len() == 4
            {
                let scales = scales.to_vec();
                node.inputs.truncate(1);
                node.attributes
                    .retain(|a| a.name != "scales" && a.name != "sizes");
                set_attr(&mut node, "scales", AttributeValue::Floats(scales));
            }
        } else if matches!(node.op_type.as_str(), "Squeeze" | "Unsqueeze")
            && node.inputs.len() > 1
            && attribute(&node, "axes").is_none()
        {
            let name = resolve(&node.inputs[1], &aliases)?;
            let axes = constants.ints(&name).ok_or_else(|| {
                error(format!(
                    "{} axes input {name} is not constant integer",
                    node.name
                ))
            })?;
            if axes.is_empty() || axes.len() > limits.max_rank {
                return Err(error(format!("{} invalid axes input", node.name)));
            }
            set_attr(&mut node, "axes", AttributeValue::Ints(axes));
            node.inputs.truncate(1);
        }
        normalize_same(&mut node)?;
        node.inputs = node
            .inputs
            .iter()
            .map(|name| resolve(name, &aliases))
            .collect::<Result<Vec<_>>>()?;
        node.outputs = node
            .outputs
            .iter()
            .map(|name| resolve(name, &aliases))
            .collect::<Result<Vec<_>>>()?;
        transformed.push(WorkNode {
            key: format!("node:{origin}"),
            origin,
            raw: node,
        });
    }
    fold_bn(&mut transformed, &mut constants, &output_names)?;
    merge_branches(&mut transformed, &mut constants, &output_names)?;
    let mut needed = output_names.iter().cloned().collect::<HashSet<_>>();
    let mut keep = vec![false; transformed.len()];
    for (i, node) in transformed.iter().enumerate().rev() {
        if node.raw.outputs.iter().any(|name| needed.contains(name)) {
            keep[i] = true;
            for name in node.raw.inputs.iter().filter(|n| !n.is_empty()) {
                needed.insert(name.clone());
            }
        }
    }
    transformed = transformed
        .into_iter()
        .zip(keep)
        .filter_map(|(node, keep)| keep.then_some(node))
        .collect();
    let live_origins = transformed.iter().map(|n| n.origin).collect::<HashSet<_>>();
    recipes.retain(|r| {
        r.node_key
            .strip_prefix("node:")
            .and_then(|n| n.parse::<usize>().ok())
            .is_some_and(|n| live_origins.contains(&n))
    });
    recipes.sort_by(|a, b| a.node_key.cmp(&b.node_key));
    let live_metadata = recipes
        .iter()
        .flat_map(|r| &r.source_indices)
        .copied()
        .collect::<HashSet<_>>();
    shape_metadata
        .nodes
        .retain(|n| live_metadata.contains(&n.source_indices[0]));
    let live_metadata_constants = shape_metadata
        .nodes
        .iter()
        .flat_map(|n| &n.inputs)
        .collect::<HashSet<_>>();
    shape_metadata
        .constants
        .retain(|c| live_metadata_constants.contains(&c.name));
    let used_constants = transformed
        .iter()
        .flat_map(|n| &n.raw.inputs)
        .filter(|name| constants.get(name).is_some())
        .cloned()
        .chain(
            output_names
                .iter()
                .filter(|name| constants.get(name).is_some())
                .cloned(),
        )
        .collect::<HashSet<_>>();
    let mut names = vec![];
    let mut name_set = HashSet::new();
    let add_name = |name: &str, names: &mut Vec<String>, seen: &mut HashSet<String>| {
        if !name.is_empty() && seen.insert(name.into()) {
            names.push(name.into());
        }
    };
    for name in &input_names {
        add_name(name, &mut names, &mut name_set);
    }
    for c in constants.items.iter().flatten() {
        if used_constants.contains(&c.name) {
            add_name(&c.name, &mut names, &mut name_set);
        }
    }
    for n in &transformed {
        for name in &n.raw.outputs {
            add_name(name, &mut names, &mut name_set);
        }
    }
    for name in &output_names {
        add_name(name, &mut names, &mut name_set);
    }
    let index = names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.clone(), TensorId(i)))
        .collect::<HashMap<_, _>>();
    let mut tensors = Vec::with_capacity(names.len());
    for name in names {
        let (key, dtype, shape, constant) = if let Some(c) = constants.take(&name) {
            if c.data.dimensions().contains(&0) {
                return Err(error(format!(
                    "retained initializer {name} has nonpositive dimension"
                )));
            }
            (c.key, c.data.dtype(), constant_shape(&c.data), Some(c.data))
        } else {
            let metadata = info.get(&name);
            (
                original(&name),
                metadata.map_or(DType::F32, |v| v.dtype),
                metadata.map_or(ShapeSpec::UnknownRank, |v| normalized_shape(&v.shape)),
                None,
            )
        };
        if let ShapeSpec::Ranked(dims) = &shape
            && dims.len() > limits.max_rank
        {
            return Err(error(format!("tensor {name} rank exceeds limit")));
        }
        tensors.push(TensorDescriptor {
            key,
            name,
            dtype,
            shape,
            constant,
        });
    }
    let mut available = input_names
        .iter()
        .cloned()
        .chain(used_constants.iter().cloned())
        .collect::<HashSet<_>>();
    let mut frozen_nodes = vec![];
    for work in transformed {
        let n = work.raw;
        if !domain_key(&n.domain).is_empty() {
            return Err(error(format!(
                "unsupported operator domain {} at {}",
                n.domain, n.name
            )));
        }
        for input in n.inputs.iter().filter(|n| !n.is_empty()) {
            if !available.contains(input) {
                return Err(error(format!(
                    "{} has unresolved or non-topological input {input}",
                    n.name
                )));
            }
        }
        for output in n.outputs.iter().filter(|n| !n.is_empty()) {
            available.insert(output.clone());
        }
        let ids = |names: &[String]| {
            names
                .iter()
                .map(|name| {
                    if name.is_empty() {
                        Ok(None)
                    } else {
                        index
                            .get(name)
                            .copied()
                            .map(Some)
                            .ok_or_else(|| error(format!("unresolved tensor {name}")))
                    }
                })
                .collect::<Result<Vec<_>>>()
        };
        let attributes = effective(&n, *opsets.get(domain_key(&n.domain)).unwrap_or(&0))?;
        frozen_nodes.push(Node {
            key: work.key,
            name: n.name,
            op_type: n.op_type,
            domain: n.domain.clone(),
            opset: *opsets.get(domain_key(&n.domain)).unwrap_or(&0),
            source_indices: n.source_indices,
            inputs: ids(&n.inputs)?,
            outputs: ids(&n.outputs)?,
            attributes,
        });
    }
    for name in &output_names {
        if !available.contains(name) {
            return Err(error(format!("unresolved graph output {name}")));
        }
    }
    let mut shape_data_sources = HashMap::new();
    for node in &shape_metadata.nodes {
        if node.op_type == "Shape" && node.inputs.len() == 1 {
            let source = resolve(&node.inputs[0], &aliases)?;
            if let Some(&id) = index.get(&source) {
                shape_data_sources.insert(node.inputs[0].clone(), id);
            }
        }
    }
    let graph = Graph {
        tensors,
        nodes: frozen_nodes,
        inputs: input_names.iter().map(|n| index[n]).collect(),
        outputs: output_names.iter().map(|n| index[n]).collect(),
    };
    Ok(Model {
        packed_weights: crate::runtime::packing::WeightCache::default(),
        ir_version,
        opsets,
        source_bytes,
        graph,
        shape_recipes: recipes,
        shape_metadata,
        shape_data_sources,
        raw_snapshot: snapshot,
        timings: None,
    })
}

fn normalize_same(node: &mut RawNode) -> Result<()> {
    if matches!(
        node.op_type.as_str(),
        "Conv" | "ConvTranspose" | "MaxPool" | "AveragePool"
    ) && bytes_attr(node, "auto_pad", b"NOTSET")? == b"VALID"
    {
        node.attributes.retain(|a| a.name != "auto_pad");
        return Ok(());
    }
    if !matches!(node.op_type.as_str(), "Conv" | "MaxPool") {
        return Ok(());
    }
    if bytes_attr(node, "auto_pad", b"NOTSET")? != b"SAME_UPPER" {
        return Ok(());
    }
    let kernel = ints_attr(node, "kernel_shape", &[])?;
    let strides = ints_attr(node, "strides", &[1, 1])?;
    let dilation = ints_attr(node, "dilations", &[1, 1])?;
    if kernel.len() != 2 || strides != [1, 1] || dilation.len() != 2 {
        return Ok(());
    }
    let ph = kernel[0]
        .checked_sub(1)
        .and_then(|v| v.checked_mul(dilation[0]))
        .ok_or_else(|| error("padding overflow"))?;
    let pw = kernel[1]
        .checked_sub(1)
        .and_then(|v| v.checked_mul(dilation[1]))
        .ok_or_else(|| error("padding overflow"))?;
    node.attributes.retain(|a| a.name != "auto_pad");
    set_attr(
        node,
        "pads",
        AttributeValue::Ints(vec![ph / 2, pw / 2, ph - ph / 2, pw - pw / 2]),
    );
    Ok(())
}

fn fold_bn(nodes: &mut Vec<WorkNode>, constants: &mut Constants, outputs: &[String]) -> Result<()> {
    let (producer, consumers) = origins(nodes);
    let graph_outputs = outputs.iter().collect::<HashSet<_>>();
    let mut folded = HashSet::new();
    for bi in 0..nodes.len() {
        let bn = &nodes[bi].raw;
        if bn.op_type != "BatchNormalization" || bn.inputs.len() != 5 || bn.outputs.len() != 1 {
            continue;
        }
        let Some(&ci) = producer.get(&bn.inputs[0]) else {
            continue;
        };
        if consumers.get(&bn.inputs[0]) != Some(&1) || ci >= bi {
            continue;
        }
        let conv = &nodes[ci].raw;
        if conv.op_type != "Conv"
            || !(2..=3).contains(&conv.inputs.len())
            || conv.outputs.len() != 1
            || graph_outputs.contains(&conv.outputs[0])
        {
            continue;
        }
        let Some(weight) = constants.get(&conv.inputs[1]) else {
            continue;
        };
        let Some(weights) = constants.f32(&conv.inputs[1]) else {
            continue;
        };
        let Some(gamma) = constants.f32(&bn.inputs[1]) else {
            continue;
        };
        let Some(beta) = constants.f32(&bn.inputs[2]) else {
            continue;
        };
        let Some(mean) = constants.f32(&bn.inputs[3]) else {
            continue;
        };
        let Some(variance) = constants.f32(&bn.inputs[4]) else {
            continue;
        };
        if weight.data.dimensions().len() != 4
            || gamma.is_empty()
            || [beta.len(), mean.len(), variance.len()]
                .iter()
                .any(|&n| n != gamma.len())
            || weight.data.dimensions()[0] != gamma.len()
        {
            continue;
        }
        let epsilon = float_attr(bn, "epsilon", 1e-5)?;
        if !epsilon.is_finite()
            || epsilon < 0.0
            || variance.iter().any(|v| !v.is_finite() || *v < 0.0)
            || gamma.iter().chain(beta).chain(mean).any(|v| v.is_nan())
        {
            continue;
        }
        let bias = if conv.inputs.len() == 3 {
            let Some(bias) = constants.f32(&conv.inputs[2]) else {
                continue;
            };
            if bias.len() != gamma.len() {
                continue;
            }
            Some(bias)
        } else {
            None
        };
        let channels = gamma.len();
        let per_channel = weights.len() / channels;
        let mut folded_weights = vec![0f32; weights.len()];
        let mut folded_bias = vec![0f32; channels];
        for channel in 0..channels {
            // Keep the reference's individual F32 operations, with no fused multiply-add.
            let variance_epsilon = variance[channel] + epsilon;
            let scale = gamma[channel] / variance_epsilon.sqrt();
            let shifted = bias.map_or(0.0, |b| b[channel]) - mean[channel];
            let scaled = shifted * scale;
            folded_bias[channel] = scaled + beta[channel];
            for i in 0..per_channel {
                folded_weights[channel * per_channel + i] =
                    weights[channel * per_channel + i] * scale;
            }
        }
        let dims = weight.data.dimensions().to_vec();
        let weight_name = constants.unique(&format!("{}__managed_bn_weight", conv.inputs[1]));
        let bias_base = if conv.name.is_empty() {
            &conv.outputs[0]
        } else {
            &conv.name
        };
        let bias_name = constants.unique(&format!("{bias_base}__managed_bn_bias"));
        let weight_key = format!(
            "generated:bn_weight:{}:{}",
            nodes[ci].origin, nodes[bi].origin
        );
        let bias_key = format!(
            "generated:bn_bias:{}:{}",
            nodes[ci].origin, nodes[bi].origin
        );
        let output = bn.outputs[0].clone();
        let sources = lineage([
            nodes[ci].raw.source_indices.clone(),
            nodes[bi].raw.source_indices.clone(),
        ]);
        constants.insert(
            weight_name.clone(),
            weight_key,
            f32_tensor(&dims, folded_weights)?,
        )?;
        constants.insert(
            bias_name.clone(),
            bias_key,
            f32_tensor(&[channels], folded_bias)?,
        )?;
        let conv = &mut nodes[ci].raw;
        conv.inputs[1] = weight_name;
        if conv.inputs.len() == 2 {
            conv.inputs.push(bias_name);
        } else {
            conv.inputs[2] = bias_name;
        }
        conv.outputs[0] = output;
        conv.source_indices = sources;
        folded.insert(bi);
    }
    let mut i = 0;
    nodes.retain(|_| {
        let keep = !folded.contains(&i);
        i += 1;
        keep
    });
    prune_constants(nodes, constants, outputs);
    Ok(())
}

struct Branch<'a> {
    input: &'a str,
    output_channels: usize,
    input_channels: usize,
    kernel_h: usize,
    kernel_w: usize,
    weights: &'a [f32],
    bias: Option<&'a [f32]>,
}
fn branch<'a>(
    node: &'a RawNode,
    constants: &'a Constants,
    outputs: &HashSet<&String>,
) -> Result<Option<Branch<'a>>> {
    if node.op_type != "Conv"
        || !(2..=3).contains(&node.inputs.len())
        || node.outputs.len() != 1
        || outputs.contains(&node.outputs[0])
    {
        return Ok(None);
    }
    if ints_attr(node, "strides", &[1, 1])? != [1, 1]
        || ints_attr(node, "dilations", &[1, 1])? != [1, 1]
        || int_attr(node, "group", 1)? != 1
    {
        return Ok(None);
    }
    let auto = bytes_attr(node, "auto_pad", b"NOTSET")?;
    if !auto.is_empty() && auto != b"NOTSET" {
        return Ok(None);
    }
    let Some(weight) = constants.get(&node.inputs[1]) else {
        return Ok(None);
    };
    let dims = weight.data.dimensions();
    if dims.len() != 4 {
        return Ok(None);
    }
    let Some(weights) = constants.f32(&node.inputs[1]) else {
        return Ok(None);
    };
    let (kh, kw) = (dims[2], dims[3]);
    if kh % 2 == 0 || kw % 2 == 0 {
        return Ok(None);
    }
    if ints_attr(node, "kernel_shape", &[kh as i64, kw as i64])? != [kh as i64, kw as i64]
        || ints_attr(node, "pads", &[0, 0, 0, 0])?
            != [
                ((kh - 1) / 2) as i64,
                ((kw - 1) / 2) as i64,
                ((kh - 1) / 2) as i64,
                ((kw - 1) / 2) as i64,
            ]
    {
        return Ok(None);
    }
    let bias = if node.inputs.len() == 3 {
        let Some(bias) = constants.f32(&node.inputs[2]) else {
            return Ok(None);
        };
        if bias.len() != dims[0] {
            return Ok(None);
        }
        Some(bias)
    } else {
        None
    };
    Ok(Some(Branch {
        input: &node.inputs[0],
        output_channels: dims[0],
        input_channels: dims[1],
        kernel_h: kh,
        kernel_w: kw,
        weights,
        bias,
    }))
}
fn accumulate(target: &mut [f32], branch: &Branch<'_>, kh: usize, kw: usize) {
    let oy = (kh - branch.kernel_h) / 2;
    let ox = (kw - branch.kernel_w) / 2;
    for f in 0..branch.output_channels * branch.input_channels {
        for y in 0..branch.kernel_h {
            for x in 0..branch.kernel_w {
                target[(f * kh + y + oy) * kw + x + ox] +=
                    branch.weights[(f * branch.kernel_h + y) * branch.kernel_w + x];
            }
        }
    }
}
fn merge_branches(
    nodes: &mut Vec<WorkNode>,
    constants: &mut Constants,
    outputs: &[String],
) -> Result<()> {
    let graph_outputs = outputs.iter().collect::<HashSet<_>>();
    loop {
        let (producer, consumers) = origins(nodes);
        let mut merged = false;
        for ai in 0..nodes.len() {
            let add = &nodes[ai].raw;
            if add.op_type != "Add" || add.inputs.len() != 2 || add.outputs.len() != 1 {
                continue;
            }
            let (Some(&fi), Some(&si)) =
                (producer.get(&add.inputs[0]), producer.get(&add.inputs[1]))
            else {
                continue;
            };
            if fi == si {
                continue;
            }
            let Some(first) = branch(&nodes[fi].raw, constants, &graph_outputs)? else {
                continue;
            };
            let Some(second) = branch(&nodes[si].raw, constants, &graph_outputs)? else {
                continue;
            };
            if first.input != second.input
                || first.output_channels != second.output_channels
                || first.input_channels != second.input_channels
                || consumers.get(&nodes[fi].raw.outputs[0]) != Some(&1)
                || consumers.get(&nodes[si].raw.outputs[0]) != Some(&1)
            {
                continue;
            }
            let kh = first.kernel_h.max(second.kernel_h);
            let kw = first.kernel_w.max(second.kernel_w);
            let dims = [first.output_channels, first.input_channels, kh, kw];
            let len = Shape::new(&dims)?.element_count();
            let mut weights = vec![0f32; len];
            accumulate(&mut weights, &first, kh, kw);
            accumulate(&mut weights, &second, kh, kw);
            let bias = if first.bias.is_some() || second.bias.is_some() {
                Some(
                    (0..first.output_channels)
                        .map(|i| {
                            first.bias.map_or(0.0, |b| b[i]) + second.bias.map_or(0.0, |b| b[i])
                        })
                        .collect::<Vec<_>>(),
                )
            } else {
                None
            };
            let output = add.outputs[0].clone();
            let add_origin = nodes[ai].origin;
            let weight_name = constants.unique(&format!(
                "{}__managed_branch_weight",
                nodes[fi].raw.inputs[1]
            ));
            let bias_name = constants.unique(&format!("{weight_name}_bias"));
            let sources = lineage([
                nodes[fi].raw.source_indices.clone(),
                nodes[si].raw.source_indices.clone(),
                nodes[ai].raw.source_indices.clone(),
            ]);
            constants.insert(
                weight_name.clone(),
                format!("generated:branch_weight:{add_origin}"),
                f32_tensor(&dims, weights)?,
            )?;
            if let Some(bias) = bias {
                constants.insert(
                    bias_name.clone(),
                    format!("generated:branch_bias:{add_origin}"),
                    f32_tensor(&[dims[0]], bias)?,
                )?;
                let conv = &mut nodes[fi].raw;
                if conv.inputs.len() == 2 {
                    conv.inputs.push(bias_name);
                } else {
                    conv.inputs[2] = bias_name;
                }
            }
            let conv = &mut nodes[fi].raw;
            conv.inputs[1] = weight_name;
            conv.outputs[0] = output;
            conv.source_indices = sources;
            set_attr(
                conv,
                "kernel_shape",
                AttributeValue::Ints(vec![kh as i64, kw as i64]),
            );
            set_attr(
                conv,
                "pads",
                AttributeValue::Ints(vec![
                    ((kh - 1) / 2) as i64,
                    ((kw - 1) / 2) as i64,
                    ((kh - 1) / 2) as i64,
                    ((kw - 1) / 2) as i64,
                ]),
            );
            let high = ai.max(si);
            let low = ai.min(si);
            nodes.remove(high);
            nodes.remove(low);
            merged = true;
            prune_constants(nodes, constants, outputs);
            break;
        }
        if !merged {
            break;
        }
    }
    Ok(())
}
fn prune_constants(nodes: &[WorkNode], constants: &mut Constants, outputs: &[String]) {
    let used = nodes
        .iter()
        .flat_map(|n| &n.raw.inputs)
        .chain(outputs)
        .collect::<HashSet<_>>();
    for item in &mut constants.items {
        if item.as_ref().is_some_and(|c| !used.contains(&c.name)) {
            *item = None;
        }
    }
    // Keep the names reserved, preserving C# UniqueName collision behavior.
}
