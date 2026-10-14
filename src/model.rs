//! Structural ONNX import. This module does not execute graphs or allocate activations.
use crate::onnx::{
    self, Attribute, AttributeValue, ImportError, LoadLimits, RawGraph, RawModel, RawNode,
    RawTensor, ValueInfo,
};
use crate::tensor::{DType, Dim, Shape, ShapeSpec, Tensor, TensorData};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    fs::{self, File},
    io::{BufRead, BufReader, BufWriter, Cursor, Read, Write},
    path::{Path, PathBuf},
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
    ir_version: i64,
    opsets: BTreeMap<String, i64>,
    source_bytes: u64,
    graph: Graph,
    shape_recipes: Vec<ShapeRecipe>,
    shape_metadata: ShapeMetadata,
    raw_snapshot: Option<ImportSnapshot>,
    timings: Option<ImportTimings>,
}

impl Model {
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
    let graph = Graph {
        tensors,
        nodes: frozen_nodes,
        inputs: input_names.iter().map(|n| index[n]).collect(),
        outputs: output_names.iter().map(|n| index[n]).collect(),
    };
    Ok(Model {
        ir_version,
        opsets,
        source_bytes,
        graph,
        shape_recipes: recipes,
        shape_metadata,
        raw_snapshot: snapshot,
        timings: None,
    })
}

fn normalize_same(node: &mut RawNode) -> Result<()> {
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

#[derive(Debug, Serialize)]
pub struct ImportInspectionReport {
    pub source_bytes: u64,
    pub normalized_weight_bytes: u64,
    pub raw_node_count: usize,
    pub normalized_node_count: usize,
    pub normalized_tensor_count: usize,
    pub elapsed_ms: f64,
    pub output_dir: PathBuf,
}
impl std::fmt::Display for ImportInspectionReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "IMPORTED: {} -> {} nodes, {} tensors, {} normalized weight bytes from {} source bytes; {} ({:.2} ms)",
            self.raw_node_count,
            self.normalized_node_count,
            self.normalized_tensor_count,
            self.normalized_weight_bytes,
            self.source_bytes,
            self.output_dir.display(),
            self.elapsed_ms
        )
    }
}

/// Write one model's inspection as raw.json, normalized.json and normalized weight files.
pub fn inspect_model(
    path: impl AsRef<Path>,
    directory: impl AsRef<Path>,
    options: &LoadOptions,
) -> Result<ImportInspectionReport> {
    let started = Instant::now();
    let mut options = options.clone();
    options.capture_raw_snapshot = true;
    let model = Model::load_path_with_options(path, &options)?;
    let directory = directory.as_ref();
    fs::create_dir_all(directory.join("weights")).map_err(|e| error(e.to_string()))?;
    let raw = model.raw_snapshot().expect("capture was enabled");
    write_json(&directory.join("raw.json"), raw)?;
    let mut normalized = model.normalized_snapshot()?;
    let mut ordinal = 0;
    for tensor in &mut normalized.tensors {
        if !tensor.constant {
            continue;
        }
        let data = model
            .graph
            .tensors
            .iter()
            .find(|t| t.key == tensor.key)
            .and_then(|t| t.constant.as_ref())
            .expect("snapshot constant exists");
        let relative = format!("weights/{ordinal:04}.bin");
        ordinal += 1;
        let mut writer = BufWriter::with_capacity(
            64 * 1024,
            File::create(directory.join(&relative)).map_err(|e| error(e.to_string()))?,
        );
        write_data(&mut writer, data)?;
        writer.flush().map_err(|e| error(e.to_string()))?;
        tensor.data_path = Some(relative);
    }
    write_json(&directory.join("normalized.json"), &normalized)?;
    Ok(ImportInspectionReport {
        source_bytes: model.source_bytes,
        normalized_weight_bytes: model.normalized_weight_bytes(),
        raw_node_count: raw.nodes.len(),
        normalized_node_count: model.graph.nodes.len(),
        normalized_tensor_count: model.graph.tensors.len(),
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        output_dir: directory.to_path_buf(),
    })
}
fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let file = File::create(path).map_err(|e| error(format!("{}: {e}", path.display())))?;
    write_json_to(file, value).map_err(|e| error(format!("{}: {e}", path.display())))
}
fn write_json_to(writer: impl Write, value: &impl Serialize) -> Result<()> {
    let mut writer = BufWriter::new(writer);
    serde_json::to_writer_pretty(&mut writer, value).map_err(|e| error(e.to_string()))?;
    writer.flush().map_err(|e| error(e.to_string()))
}
fn write_data(writer: &mut impl Write, data: &TensorData) -> Result<()> {
    data_chunks(data, |bytes| {
        writer.write_all(bytes).map_err(|e| error(e.to_string()))
    })
}

#[derive(Clone, Debug, Deserialize)]
struct FileRecord {
    id: String,
    path: String,
    bytes: u64,
    sha256: String,
}
#[derive(Clone, Debug, Deserialize)]
struct ContractRecord {
    model_id: String,
    raw: FileRecord,
    normalized: FileRecord,
}
#[derive(Debug, Deserialize)]
struct ImportManifest {
    schema_version: u32,
    kind: String,
    models: Vec<FileRecord>,
    #[serde(default)]
    dictionaries: Vec<FileRecord>,
    contracts: Vec<ContractRecord>,
    #[serde(default)]
    auxiliary_files: Vec<FileRecord>,
}
struct ImportReference {
    root: PathBuf,
    manifest: ImportManifest,
    raw: BTreeMap<String, ImportSnapshot>,
    normalized: BTreeMap<String, ImportSnapshot>,
}
#[derive(Debug, Serialize)]
pub struct ImportValidationReport {
    pub checked_models: usize,
    pub checked_contracts: usize,
    pub checked_files: usize,
}
impl std::fmt::Display for ImportValidationReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "VALID: {} models, {} structural contracts, {} files verified",
            self.checked_models, self.checked_contracts, self.checked_files
        )
    }
}
#[derive(Debug, Serialize)]
pub struct ImportComparisonReport {
    pub model_id: String,
    pub checked_stages: usize,
    pub checked_weights: usize,
    pub first_difference: Option<String>,
}
impl ImportComparisonReport {
    pub fn matches(&self) -> bool {
        self.first_difference.is_none()
    }
}
impl std::fmt::Display for ImportComparisonReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(d) = &self.first_difference {
            write!(f, "DIFFERENCE: {d}")
        } else {
            write!(
                f,
                "MATCH: {} raw/normalized stages, {} normalized weights for {}",
                self.checked_stages, self.checked_weights, self.model_id
            )
        }
    }
}

fn safe_reference_file(root: &Path, path: &str) -> Result<PathBuf> {
    crate::safe_file(root, path).map_err(|e| error(e.to_string()))
}
fn verify_file(root: &Path, record: &FileRecord) -> Result<()> {
    if record.id.is_empty()
        || record.sha256.len() != 64
        || !record.sha256.bytes().all(|b| b.is_ascii_hexdigit())
    {
        return Err(error(format!("invalid file record {}", record.id)));
    }
    let path = safe_reference_file(root, &record.path)?;
    let mut file = File::open(&path).map_err(|e| error(e.to_string()))?;
    if file.metadata().map_err(|e| error(e.to_string()))?.len() != record.bytes {
        return Err(error(format!("{}: byte count mismatch", record.path)));
    }
    let mut sha = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let n = file.read(&mut buffer).map_err(|e| error(e.to_string()))?;
        if n == 0 {
            break;
        }
        sha.update(&buffer[..n]);
    }
    let digest = format!("{:x}", sha.finalize());
    if !digest.eq_ignore_ascii_case(&record.sha256) {
        return Err(error(format!("{}: SHA256 mismatch", record.path)));
    }
    Ok(())
}
fn load_reference(directory: &Path) -> Result<(ImportReference, ImportValidationReport)> {
    let root = directory
        .canonicalize()
        .map_err(|e| error(format!("{}: {e}", directory.display())))?;
    let manifest_file = safe_reference_file(&root, "manifest.json")?;
    let manifest: ImportManifest = serde_json::from_reader(BufReader::new(
        File::open(manifest_file).map_err(|e| error(e.to_string()))?,
    ))
    .map_err(|e| error(e.to_string()))?;
    if manifest.schema_version != 1 || manifest.kind != "model_import_contracts" {
        return Err(error("unsupported model import manifest kind/schema"));
    }
    let mut files = BTreeMap::<String, &FileRecord>::new();
    let mut ids = HashSet::new();
    for record in manifest
        .models
        .iter()
        .chain(&manifest.dictionaries)
        .chain(&manifest.auxiliary_files)
        .chain(
            manifest
                .contracts
                .iter()
                .flat_map(|c| [&c.raw, &c.normalized]),
        )
    {
        if !ids.insert(record.id.as_str()) {
            return Err(error(format!("duplicate archive file ID {}", record.id)));
        }
        if files.insert(record.path.clone(), record).is_some() {
            return Err(error(format!("duplicate archive path {}", record.path)));
        }
        verify_file(&root, record)?;
    }
    let mut raw = BTreeMap::new();
    let mut normalized = BTreeMap::new();
    let model_ids = manifest
        .models
        .iter()
        .map(|m| m.id.as_str())
        .collect::<HashSet<_>>();
    for c in &manifest.contracts {
        if !model_ids.contains(c.model_id.as_str()) || raw.contains_key(&c.model_id) {
            return Err(error(format!(
                "unknown/duplicate contract model ID {}",
                c.model_id
            )));
        }
        let read = |r: &FileRecord| -> Result<ImportSnapshot> {
            serde_json::from_reader(BufReader::new(
                File::open(safe_reference_file(&root, &r.path)?)
                    .map_err(|e| error(e.to_string()))?,
            ))
            .map_err(|e| error(format!("{}: {e}", r.path)))
        };
        let r = read(&c.raw)?;
        let n = read(&c.normalized)?;
        validate_snapshot(&r, "raw")?;
        validate_snapshot(&n, "normalized")?;
        validate_shape_metadata(&r, &n)?;
        for t in &n.tensors {
            if t.constant {
                let path = t.data_path.as_ref().ok_or_else(|| {
                    error(format!(
                        "{} normalized constant {} lacks archive path",
                        c.model_id, t.key
                    ))
                })?;
                let file = files
                    .get(path)
                    .ok_or_else(|| error(format!("weight {path} missing from auxiliary_files")))?;
                if t.data_bytes != file.bytes
                    || t.data_sha256
                        .as_ref()
                        .is_none_or(|h| !h.eq_ignore_ascii_case(&file.sha256))
                {
                    return Err(error(format!(
                        "tensor {} weight declaration mismatch",
                        t.key
                    )));
                }
            }
        }
        raw.insert(c.model_id.clone(), r);
        normalized.insert(c.model_id.clone(), n);
    }
    if raw.len() != manifest.models.len() {
        return Err(error(
            "every archived model requires raw and normalized contracts",
        ));
    }
    let report = ImportValidationReport {
        checked_models: manifest.models.len(),
        checked_contracts: manifest.contracts.len() * 2,
        checked_files: files.len(),
    };
    Ok((
        ImportReference {
            root,
            manifest,
            raw,
            normalized,
        },
        report,
    ))
}
fn validate_snapshot(snapshot: &ImportSnapshot, stage: &str) -> Result<()> {
    if snapshot.schema_version != 1 || snapshot.stage != stage {
        return Err(error(format!("invalid {stage} snapshot schema/stage")));
    }
    let mut tensors = HashSet::new();
    for t in &snapshot.tensors {
        if t.key.is_empty() || !tensors.insert(t.key.as_str()) {
            return Err(error(format!("duplicate/empty tensor key {}", t.key)));
        }
        let width = match t.dtype.as_str() {
            "f32" | "i32" => 4u64,
            "i64" => 8,
            "u8" => 1,
            "unknown" if !t.constant => 0,
            _ => return Err(error(format!("unsupported snapshot dtype {}", t.dtype))),
        };
        if !t.shape.rank_known && !t.shape.dims.is_empty() {
            return Err(error(format!("{} unknown rank has dimensions", t.key)));
        }
        if t.constant {
            if !t.shape.rank_known {
                return Err(error(format!("{} constant lacks concrete shape", t.key)));
            }
            let mut size = width;
            for dim in &t.shape.dims {
                let CanonicalDim::Known { value } = dim else {
                    return Err(error(format!("{} constant has symbolic dimension", t.key)));
                };
                size = size
                    .checked_mul(*value as u64)
                    .ok_or_else(|| error("snapshot shape size overflow"))?;
            }
            if size != t.data_bytes {
                return Err(error(format!("{} constant shape/bytes mismatch", t.key)));
            }
            let Some(hash) = &t.data_sha256 else {
                return Err(error(format!("{} missing data hash", t.key)));
            };
            if hash.len() != 64 || !hash.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err(error(format!("{} invalid data hash", t.key)));
            }
        } else if t.data_bytes != 0 || t.data_sha256.is_some() || t.data_path.is_some() {
            return Err(error(format!("{} activation has constant payload", t.key)));
        }
    }
    let mut nodes = HashSet::new();
    for n in &snapshot.nodes {
        if n.key.is_empty() || !nodes.insert(n.key.as_str()) {
            return Err(error(format!("duplicate/empty node key {}", n.key)));
        }
        for key in n.inputs.iter().chain(&n.outputs).flatten() {
            if !tensors.contains(key.as_str()) {
                return Err(error(format!("{} references missing tensor {key}", n.key)));
            }
        }
    }
    for key in snapshot.inputs.iter().chain(&snapshot.outputs) {
        if !tensors.contains(key.as_str()) {
            return Err(error(format!("missing graph tensor {key}")));
        }
    }
    Ok(())
}

fn validate_shape_metadata(raw: &ImportSnapshot, normalized: &ImportSnapshot) -> Result<()> {
    if raw.ir_version != normalized.ir_version || raw.opsets != normalized.opsets {
        return Err(error("raw/normalized IR version or opset metadata differ"));
    }
    if !raw.recipes.is_empty()
        || !raw.shape_metadata.nodes.is_empty()
        || !raw.shape_metadata.constants.is_empty()
    {
        return Err(error("raw snapshot contains normalized shape recipes"));
    }
    let source_nodes = raw
        .nodes
        .iter()
        .map(|n| (n.key.as_str(), n))
        .collect::<HashMap<_, _>>();
    let normalized_nodes = normalized
        .nodes
        .iter()
        .map(|n| (n.key.as_str(), n))
        .collect::<HashMap<_, _>>();
    let normalized_tensors = normalized
        .tensors
        .iter()
        .map(|t| (t.key.as_str(), t))
        .collect::<HashMap<_, _>>();
    let aliases = raw
        .nodes
        .iter()
        .filter(|n| n.op_type == "Identity" && n.inputs.len() == 1 && n.outputs.len() == 1)
        .filter_map(|n| {
            n.inputs[0]
                .as_ref()
                .zip(n.outputs[0].as_ref())
                .map(|(input, output)| (output.clone(), input.clone()))
        })
        .collect::<HashMap<_, _>>();
    let mut producers = HashMap::new();
    for node in &raw.nodes {
        for output in node.outputs.iter().flatten() {
            if producers.insert(output.as_str(), node).is_some() {
                return Err(error(format!(
                    "raw shape source has duplicate producer {output}"
                )));
            }
        }
    }
    let mut required = BTreeSet::new();
    let mut recipe_keys = HashSet::new();
    for recipe in &normalized.recipes {
        if !recipe_keys.insert(&recipe.node_key)
            || normalized_nodes
                .get(recipe.node_key.as_str())
                .is_none_or(|n| n.op_type != "Reshape")
        {
            return Err(error(format!(
                "invalid shape recipe node {}",
                recipe.node_key
            )));
        }
        let normalized_node = normalized_nodes[recipe.node_key.as_str()];
        let source = source_nodes
            .get(recipe.node_key.as_str())
            .ok_or_else(|| error(format!("recipe {} has no raw source node", recipe.node_key)))?;
        let source_input = |index: usize| {
            source
                .inputs
                .get(index)
                .and_then(Option::as_ref)
                .ok_or_else(|| {
                    error(format!(
                        "recipe {} lacks raw input {index}",
                        recipe.node_key
                    ))
                })
        };
        if source.op_type != "Reshape"
            || resolve(source_input(0)?, &aliases)? != recipe.data_input
            || resolve(source_input(1)?, &aliases)? != recipe.shape_input
            || normalized_node.inputs.first().and_then(Option::as_ref) != Some(&recipe.data_input)
        {
            return Err(error(format!(
                "recipe {} data/shape input binding differs from its graph nodes",
                recipe.node_key
            )));
        }
        let output = normalized_node
            .outputs
            .first()
            .and_then(Option::as_ref)
            .and_then(|key| normalized_tensors.get(key.as_str()))
            .ok_or_else(|| {
                error(format!(
                    "recipe {} lacks normalized output",
                    recipe.node_key
                ))
            })?;
        if output.shape != recipe.template {
            return Err(error(format!(
                "recipe {} template differs from normalized output shape",
                recipe.node_key
            )));
        }
        let mut expected_sources = BTreeSet::new();
        let mut visited = HashSet::new();
        recipe_sources(
            &recipe.shape_input,
            &recipe.data_input,
            &producers,
            &mut visited,
            &mut expected_sources,
        );
        if expected_sources.into_iter().collect::<Vec<_>>() != recipe.source_indices {
            return Err(error(format!(
                "recipe {} metadata ancestry differs from its shape input",
                recipe.node_key
            )));
        }
        required.extend(recipe.source_indices.iter().copied());
    }
    let mut actual = BTreeSet::new();
    let mut used_constants = HashSet::new();
    for node in &normalized.shape_metadata.nodes {
        let source = source_nodes.get(node.key.as_str()).ok_or_else(|| {
            error(format!(
                "shape metadata node {} missing from raw source",
                node.key
            ))
        })?;
        if *source != node {
            return Err(error(format!(
                "shape metadata node {} differs from raw source",
                node.key
            )));
        }
        if node.source_indices.len() != 1 || !actual.insert(node.source_indices[0]) {
            return Err(error(
                "shape metadata source index is repeated or not singular",
            ));
        }
        used_constants.extend(node.inputs.iter().flatten().map(|s| s.as_str()));
    }
    if actual != required {
        return Err(error(
            "shape recipe indices do not exactly identify retained metadata nodes",
        ));
    }
    let mut constant_keys = HashSet::new();
    for constant in &normalized.shape_metadata.constants {
        if !constant_keys.insert(&constant.key) || !used_constants.contains(constant.key.as_str()) {
            return Err(error(format!(
                "unused or duplicate shape metadata constant {}",
                constant.key
            )));
        }
        if !constant.shape.rank_known {
            return Err(error("shape constant rank is unknown"));
        }
        let mut count = 1usize;
        for dim in &constant.shape.dims {
            let CanonicalDim::Known { value } = dim else {
                return Err(error("shape constant has symbolic dimensions"));
            };
            count = count
                .checked_mul(*value)
                .ok_or_else(|| error("shape constant size overflow"))?;
        }
        if count != constant.values.len() {
            return Err(error(format!(
                "shape constant {} payload length mismatch",
                constant.key
            )));
        }
        let mut sha = Sha256::new();
        let width = match constant.dtype.as_str() {
            "i64" => {
                for value in &constant.values {
                    sha.update(value.to_le_bytes());
                }
                8
            }
            "i32" => {
                for value in &constant.values {
                    sha.update(
                        i32::try_from(*value)
                            .map_err(|_| error("shape int32 constant overflow"))?
                            .to_le_bytes(),
                    );
                }
                4
            }
            _ => return Err(error("shape metadata constants must be int32/int64")),
        };
        let hash = format!("{:x}", sha.finalize());
        if let Some(source) = raw
            .tensors
            .iter()
            .find(|t| t.key == constant.key && t.constant)
            && (source.dtype != constant.dtype
                || source.shape != constant.shape
                || source.data_bytes != count as u64 * width
                || source
                    .data_sha256
                    .as_ref()
                    .is_none_or(|h| !h.eq_ignore_ascii_case(&hash)))
        {
            return Err(error(format!(
                "shape constant {} differs from raw initializer",
                constant.key
            )));
        }
    }
    Ok(())
}
fn recipe_sources<'a>(
    key: &str,
    data_input: &str,
    producers: &HashMap<&'a str, &'a SnapshotNode>,
    visited: &mut HashSet<String>,
    sources: &mut BTreeSet<usize>,
) {
    if key == data_input || !visited.insert(key.into()) {
        return;
    }
    let Some(node) = producers.get(key) else {
        return;
    };
    sources.extend(node.source_indices.iter().copied());
    if node.op_type != "Shape" {
        for input in node.inputs.iter().flatten() {
            recipe_sources(input, data_input, producers, visited, sources);
        }
    }
}
/// Validate all sources, dictionaries, contracts, provenance and normalized weights in an archive.
pub fn validate_import_reference(directory: impl AsRef<Path>) -> Result<ImportValidationReport> {
    load_reference(directory.as_ref()).map(|(_, report)| report)
}

/// Compare one archived source model to this loader after validating the entire reference archive.
pub fn compare_import(
    directory: impl AsRef<Path>,
    model_id: &str,
    options: &LoadOptions,
) -> Result<ImportComparisonReport> {
    let (reference, _) = load_reference(directory.as_ref())?;
    let source = reference
        .manifest
        .models
        .iter()
        .find(|m| m.id == model_id)
        .ok_or_else(|| error(format!("unknown archived model ID {model_id}")))?;
    let mut options = options.clone();
    options.capture_raw_snapshot = true;
    let model = Model::load_path_with_options(
        safe_reference_file(&reference.root, &source.path)?,
        &options,
    )?;
    compare_import_model(&reference, model_id, &model)
}
fn compare_import_model(
    reference: &ImportReference,
    model_id: &str,
    model: &Model,
) -> Result<ImportComparisonReport> {
    let mut report = ImportComparisonReport {
        model_id: model_id.into(),
        checked_stages: 0,
        checked_weights: 0,
        first_difference: None,
    };
    let actual_raw = model
        .raw_snapshot()
        .ok_or_else(|| error("raw capture required for import comparison"))?;
    for (stage, expected, actual) in [
        ("raw", &reference.raw[model_id], actual_raw.clone()),
        (
            "normalized",
            &reference.normalized[model_id],
            model.normalized_snapshot()?,
        ),
    ] {
        let mut expected = expected.clone();
        let mut actual = actual;
        for t in &mut expected.tensors {
            t.data_path = None;
        }
        for t in &mut actual.tensors {
            t.data_path = None;
        }
        expected.tensors.sort_by(|a, b| a.key.cmp(&b.key));
        actual.tensors.sort_by(|a, b| a.key.cmp(&b.key));
        let expected = serde_json::to_value(expected).map_err(|e| error(e.to_string()))?;
        let actual = serde_json::to_value(actual).map_err(|e| error(e.to_string()))?;
        if let Some(difference) = first_json_difference(&expected, &actual, "$") {
            report.first_difference = Some(format!("{model_id} {stage}: {difference}"));
            return Ok(report);
        }
        report.checked_stages += 1;
    }
    report.checked_weights = model
        .graph
        .tensors
        .iter()
        .filter(|t| t.constant.is_some())
        .count();
    Ok(report)
}
fn first_json_difference(expected: &Value, actual: &Value, path: &str) -> Option<String> {
    match (expected, actual) {
        (Value::Object(a), Value::Object(b)) => {
            for (key, value) in a {
                let next = format!("{path}.{key}");
                let Some(other) = b.get(key) else {
                    return Some(format!("{next} missing"));
                };
                if let Some(d) = first_json_difference(value, other, &next) {
                    return Some(d);
                }
            }
            b.keys()
                .find(|k| !a.contains_key(*k))
                .map(|k| format!("{path}.{k} unexpected"))
        }
        (Value::Array(a), Value::Array(b)) => {
            if a.len() != b.len() {
                return Some(format!("{path} length {} != {}", a.len(), b.len()));
            }
            a.iter().zip(b).enumerate().find_map(|(i, (a, b))| {
                let label = a
                    .get("key")
                    .and_then(Value::as_str)
                    .map(|s| format!("{s:?}"))
                    .unwrap_or_else(|| i.to_string());
                first_json_difference(a, b, &format!("{path}[{label}]"))
            })
        }
        _ if expected == actual => None,
        _ => Some(format!("{path}: expected {expected}, actual {actual}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    struct FailingWrite;
    impl Write for FailingWrite {
        fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
            Err(std::io::Error::other("buffered write failed"))
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    struct FailingFlush;
    impl Write for FailingFlush {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Err(std::io::Error::other("flush failed"))
        }
    }
    #[test]
    fn inspection_json_reports_final_buffer_and_flush_errors() {
        let tiny = serde_json::json!({"small":"buffered"});
        assert!(
            write_json_to(FailingWrite, &tiny)
                .unwrap_err()
                .to_string()
                .contains("buffered write failed")
        );
        assert!(
            write_json_to(FailingFlush, &tiny)
                .unwrap_err()
                .to_string()
                .contains("flush failed")
        );
    }
}
