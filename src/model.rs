//! Structural ONNX import. This module does not execute graphs or allocate activations.
use crate::onnx::{
    self, Attribute, AttributeValue, ImportError, LoadLimits, RawGraph, RawModel, RawNode,
    RawTensor, ValueInfo,
};
use crate::tensor::{DType, Dim, Shape, ShapeSpec, Tensor, TensorData};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    fs::File,
    io::{BufRead, BufReader, Cursor},
    path::Path,
};

use crate::error::ErrorKind;

/// Arithmetic rewrites change floating-point rounding and are optional.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OptimizationPolicy {
    #[default]
    Optimized,
    PreserveArithmetic,
}

#[derive(Clone, Debug, Default)]
pub struct LoadOptions {
    pub limits: LoadLimits,
    pub optimization: OptimizationPolicy,
}

type Result<T> = std::result::Result<T, ImportError>;
fn error(message: impl Into<String>) -> ImportError {
    ImportError::new(message)
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

/// Typed operator parameters consumed by the CPU runtime.
#[derive(Clone, Debug)]
pub enum Operator {
    Add,
    Mul,
    Div,
    Sub,
    Pow,
    Sqrt,
    Erf,
    Relu,
    MatMul,
    Sigmoid,
    GlobalAveragePool,
    Conv(ConvAttributes),
    ConvTranspose(ConvAttributes),
    MaxPool(PoolAttributes),
    AveragePool(PoolAttributes),
    BatchNormalization {
        epsilon: f32,
        momentum: f32,
        training_mode: i64,
    },
    HardSigmoid {
        alpha: f32,
        beta: f32,
    },
    Squeeze(Vec<i64>),
    Unsqueeze(Vec<i64>),
    Transpose(Vec<i64>),
    Softmax(i64),
    Concat(i64),
    ReduceMean {
        axes: Vec<i64>,
        keepdims: i64,
        noop_with_empty_axes: i64,
    },
    Resize(ResizeAttributes),
    Reshape {
        allowzero: i64,
    },
    Slice,
}

#[derive(Clone, Debug)]
pub struct Node {
    pub key: String,
    pub name: String,
    pub domain: String,
    pub opset: i64,
    pub source_indices: Vec<usize>,
    pub inputs: Vec<Option<TensorId>>,
    pub outputs: Vec<Option<TensorId>>,
    pub operator: Operator,
}

impl Operator {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Add => "Add",
            Self::Mul => "Mul",
            Self::Div => "Div",
            Self::Sub => "Sub",
            Self::Pow => "Pow",
            Self::Sqrt => "Sqrt",
            Self::Erf => "Erf",
            Self::Relu => "Relu",
            Self::MatMul => "MatMul",
            Self::Sigmoid => "Sigmoid",
            Self::GlobalAveragePool => "GlobalAveragePool",
            Self::Slice => "Slice",
            Self::Conv(..) => "Conv",
            Self::ConvTranspose(..) => "ConvTranspose",
            Self::MaxPool(..) => "MaxPool",
            Self::AveragePool(..) => "AveragePool",
            Self::Squeeze(..) => "Squeeze",
            Self::Unsqueeze(..) => "Unsqueeze",
            Self::Transpose(..) => "Transpose",
            Self::Softmax(..) => "Softmax",
            Self::Concat(..) => "Concat",
            Self::Resize(..) => "Resize",
            Self::BatchNormalization { .. } => "BatchNormalization",
            Self::HardSigmoid { .. } => "HardSigmoid",
            Self::ReduceMean { .. } => "ReduceMean",
            Self::Reshape { .. } => "Reshape",
        }
    }
}
impl Node {
    pub fn op_type(&self) -> &'static str {
        self.operator.name()
    }
}

#[derive(Debug)]
pub struct Graph {
    pub tensors: Vec<TensorDescriptor>,
    pub nodes: Vec<Node>,
    pub inputs: Vec<TensorId>,
    pub outputs: Vec<TensorId>,
}

/// A stripped metadata relationship retained for runtime shape binding without executing it during import.
#[derive(Clone, Debug, PartialEq)]
pub struct ShapeRecipe {
    pub node_key: String,
    pub data_input: String,
    pub shape_input: String,
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
    // Resolve Shape inputs through aliases to retained runtime tensors.
    shape_data_sources: HashMap<String, TensorId>,
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
        Self::load_path_with_limits(path, &LoadLimits::default())
    }
    pub fn load_reader<R: BufRead>(reader: R) -> Result<Self> {
        Self::load_reader_with_limits(reader, &LoadLimits::default())
    }
    pub fn load_bytes(bytes: &[u8]) -> Result<Self> {
        Self::load_bytes_with_limits(bytes, &LoadLimits::default())
    }
    pub fn load_path_with_limits(path: impl AsRef<Path>, limits: &LoadLimits) -> Result<Self> {
        Self::load_path_with_options(
            path,
            &LoadOptions {
                limits: limits.clone(),
                ..Default::default()
            },
        )
    }
    pub fn load_bytes_with_limits(bytes: &[u8], limits: &LoadLimits) -> Result<Self> {
        Self::load_reader_with_limits(Cursor::new(bytes), limits)
    }
    pub fn load_reader_with_limits<R: BufRead>(reader: R, limits: &LoadLimits) -> Result<Self> {
        Self::load_reader_with_options(
            reader,
            &LoadOptions {
                limits: limits.clone(),
                ..Default::default()
            },
        )
    }
    pub fn load_path_with_options(path: impl AsRef<Path>, options: &LoadOptions) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path).map_err(|e| ImportError::from(e).context(path.display()))?;
        if file.metadata()?.len() > options.limits.max_input_bytes {
            return Err(ImportError::with_kind(
                ErrorKind::ResourceLimit,
                "model exceeds max_input_bytes",
            ));
        }
        Self::load_reader_with_options(BufReader::with_capacity(64 * 1024, file), options)
            .map_err(|e| e.context(path.display()))
    }
    pub fn load_bytes_with_options(bytes: &[u8], options: &LoadOptions) -> Result<Self> {
        Self::load_reader_with_options(Cursor::new(bytes), options)
    }
    pub fn load_reader_with_options<R: BufRead>(reader: R, options: &LoadOptions) -> Result<Self> {
        let raw = onnx::parse(reader, &options.limits)?;
        normalize_with_policy(raw, &options.limits, options.optimization)
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
}

/// Owned compact integer shape program. No activation or source model payload is retained.
#[derive(Debug, Default)]
pub struct ShapeMetadata {
    pub nodes: Vec<RawNode>,
    pub constants: Vec<RawTensor>,
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
fn constant_shape(data: &TensorData) -> ShapeSpec {
    ShapeSpec::Ranked(data.dimensions().iter().map(|&d| Dim::Known(d)).collect())
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
    ImportError::with_kind(
        ErrorKind::Unsupported,
        format!(
            "{} ({}): unsupported attribute {name} or value for the OCR CPU subset",
            node.name, node.op_type
        ),
    )
}

/// Validate before transformations can erase source semantics. Attributes stored in
/// Operator are checked by their typed readers and the runtime; other
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

fn effective(node: &RawNode, opset: i64) -> Result<Operator> {
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
                Operator::Conv(attrs)
            } else {
                Operator::ConvTranspose(attrs)
            }
        }
        "AveragePool" | "MaxPool" => {
            let attributes = PoolAttributes {
                kernel_shape: array_attr(node, "kernel_shape", &[])?,
                strides: array_attr(node, "strides", &[1, 1])?,
                pads: array_attr(node, "pads", &[0, 0, 0, 0])?,
                ceil_mode: int_attr(node, "ceil_mode", 0)?,
                count_include_pad: int_attr(node, "count_include_pad", 0)?,
            };
            if node.op_type == "MaxPool" {
                Operator::MaxPool(attributes)
            } else {
                Operator::AveragePool(attributes)
            }
        }
        "BatchNormalization" => Operator::BatchNormalization {
            epsilon: float_attr(node, "epsilon", 1e-5)?,
            momentum: float_attr(node, "momentum", 0.9)?,
            training_mode: int_attr(node, "training_mode", 0)?,
        },
        "HardSigmoid" => Operator::HardSigmoid {
            alpha: float_attr(node, "alpha", 0.2)?,
            beta: float_attr(node, "beta", 0.5)?,
        },
        "Squeeze" => Operator::Squeeze(ints_attr(node, "axes", &[])?),
        "Unsqueeze" => Operator::Unsqueeze(ints_attr(node, "axes", &[])?),
        "Transpose" => Operator::Transpose(ints_attr(node, "perm", &[])?),
        "Softmax" => Operator::Softmax(int_attr(node, "axis", if opset >= 13 { -1 } else { 1 })?),
        "Concat" => Operator::Concat(int_attr(node, "axis", 1)?),
        "ReduceMean" => Operator::ReduceMean {
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
            Operator::Resize(ResizeAttributes {
                scales,
                mode: bytes_attr(node, "mode", b"nearest")?,
                coordinate_transformation_mode: bytes_attr(
                    node,
                    "coordinate_transformation_mode",
                    if opset >= 11 {
                        b"half_pixel"
                    } else {
                        b"asymmetric"
                    },
                )?,
                nearest_mode: bytes_attr(
                    node,
                    "nearest_mode",
                    if opset >= 11 {
                        b"round_prefer_floor"
                    } else {
                        b"floor"
                    },
                )?,
            })
        }
        "Reshape" => Operator::Reshape {
            allowzero: int_attr(node, "allowzero", 0)?,
        },
        "Slice" => {
            if let Some(attribute) = node.attributes.first() {
                return Err(unsupported_attribute(node, &attribute.name));
            }
            Operator::Slice
        }
        "Add" => Operator::Add,
        "Mul" => Operator::Mul,
        "Div" => Operator::Div,
        "Sub" => Operator::Sub,
        "Pow" => Operator::Pow,
        "Sqrt" => Operator::Sqrt,
        "Erf" => Operator::Erf,
        "Relu" => Operator::Relu,
        "MatMul" => Operator::MatMul,
        "Sigmoid" => Operator::Sigmoid,
        "GlobalAveragePool" => Operator::GlobalAveragePool,
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
    bytes: usize,
    generated: usize,
    byte_limit: usize,
    generated_limit: usize,
}
impl Constants {
    fn new(tensors: Vec<RawTensor>, limits: &LoadLimits) -> Result<Self> {
        let mut s = Self {
            items: vec![],
            index: HashMap::new(),
            bytes: 0,
            generated: 0,
            byte_limit: limits.max_constant_bytes,
            generated_limit: limits.max_generated_bytes,
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
        let bytes = data.shape().byte_count(data.dtype())?;
        self.bytes = self
            .bytes
            .checked_add(bytes)
            .filter(|&n| n <= self.byte_limit)
            .ok_or_else(|| {
                ImportError::with_kind(ErrorKind::ResourceLimit, "constant byte limit exceeded")
            })?;
        self.index.insert(name.clone(), self.items.len());
        self.items.push(Some(Constant { name, key, data }));
        Ok(())
    }
    fn can_generate(&self, elements: usize) -> bool {
        elements.checked_mul(4).is_some_and(|bytes| {
            self.bytes
                .checked_add(bytes)
                .is_some_and(|n| n <= self.byte_limit)
                && self
                    .generated
                    .checked_add(bytes)
                    .is_some_and(|n| n <= self.generated_limit)
        })
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

// Resolve every alias once, iteratively. Each edge is visited at most once.
fn canonical_aliases(aliases: HashMap<String, String>) -> Result<HashMap<String, String>> {
    let mut resolved = HashMap::<String, String>::new();
    for start in aliases.keys() {
        if resolved.contains_key(start) {
            continue;
        }
        let mut path = Vec::new();
        let mut visiting = HashSet::new();
        let mut current = start.as_str();
        let target = loop {
            if let Some(target) = resolved.get(current) {
                break target.clone();
            }
            let Some(next) = aliases.get(current) else {
                break current.to_owned();
            };
            if !visiting.insert(current) {
                return Err(error(format!("Identity alias cycle involving {current}")));
            }
            path.push(current);
            current = next;
        };
        for name in path {
            resolved.insert(name.to_owned(), target.clone());
        }
    }
    Ok(resolved)
}
fn resolve(name: &str, aliases: &HashMap<String, String>) -> String {
    aliases.get(name).map_or(name, String::as_str).to_owned()
}
fn metadata_sources(
    name: &str,
    data_input: &str,
    nodes: &[RawNode],
    producer: &HashMap<String, usize>,
    seen: &mut BTreeSet<usize>,
    limit: usize,
) -> Result<()> {
    let mut pending = vec![name];
    while let Some(name) = pending.pop() {
        if name == data_input {
            continue;
        }
        let Some(&i) = producer.get(name) else {
            continue;
        };
        if !seen.insert(i) {
            continue;
        }
        if seen.len() > limit {
            return Err(ImportError::with_kind(
                ErrorKind::ResourceLimit,
                "shape dependency node limit exceeded",
            ));
        }
        let node = &nodes[i];
        if node.op_type != "Shape" {
            pending.extend(node.inputs.iter().map(String::as_str));
        }
    }
    Ok(())
}
#[cfg(test)]
fn normalize(raw: RawModel, limits: &LoadLimits) -> Result<Model> {
    normalize_with_policy(raw, limits, OptimizationPolicy::Optimized)
}
fn normalize_with_policy(
    raw: RawModel,
    limits: &LoadLimits,
    policy: OptimizationPolicy,
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
    if nodes.len() > limits.max_graph_nodes {
        return Err(ImportError::with_kind(
            ErrorKind::ResourceLimit,
            "graph node limit exceeded",
        ));
    }
    let mut constants = Constants::new(initializers, limits)?;
    let mut aliases = HashMap::new();
    let mut producer = HashMap::new();
    for (i, n) in nodes.iter().enumerate() {
        if !domain_key(&n.domain).is_empty() {
            return Err(ImportError::with_kind(
                ErrorKind::Unsupported,
                format!("unsupported operator domain {} at {}", n.domain, n.name),
            ));
        }
        let opset = opsets.get(domain_key(&n.domain)).copied().unwrap_or(0);
        if !(7..=18).contains(&opset)
            || (n.op_type == "Resize" && opset < 10)
            || (n.op_type == "Erf" && opset < 9)
        {
            return Err(ImportError::with_kind(
                ErrorKind::Unsupported,
                format!(
                    "unsupported or missing opset {opset} for {} at {} (supported source opsets: 7..=18)",
                    n.op_type, n.name
                ),
            ));
        }
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
    let aliases = canonical_aliases(aliases)?;
    // Validate source topology and names before any node is erased or folded.
    let mut available = inputs
        .iter()
        .map(|v| v.name.clone())
        .chain(constants.index.keys().cloned())
        .collect::<HashSet<_>>();
    for n in &nodes {
        for input in n.inputs.iter().filter(|n| !n.is_empty()) {
            if !available.contains(input) {
                return Err(error(format!(
                    "{} has unresolved or non-topological input {input}",
                    n.name
                )));
            }
        }
        for output in n.outputs.iter().filter(|n| !n.is_empty()) {
            if !available.insert(output.clone()) {
                return Err(error(format!("duplicate tensor definition {output}")));
            }
        }
    }
    // Canonical names are used by every semantic lowering and shape recipe.
    for n in &mut nodes {
        for input in &mut n.inputs {
            *input = resolve(input, &aliases);
        }
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
        .collect::<Vec<_>>();
    let output_names = outputs
        .iter()
        .map(|v| resolve(&v.name, &aliases))
        .collect::<Vec<_>>();
    if input_names.len() != 1 || output_names.len() != 1 {
        return Err(error("only one-input/one-output OCR graphs are supported"));
    }
    let mut recipes = vec![];
    for node in &nodes {
        if node.op_type == "Reshape" && node.inputs.len() > 1 {
            let shape_input = resolve(&node.inputs[1], &aliases);
            if constants.get(&shape_input).is_none() {
                let data_input = resolve(&node.inputs[0], &aliases);
                let mut sources = BTreeSet::new();
                metadata_sources(
                    &shape_input,
                    &data_input,
                    &nodes,
                    &producer,
                    &mut sources,
                    limits.max_shape_nodes,
                )?;
                recipes.push(ShapeRecipe {
                    node_key: node_key(node),
                    data_input: original(&data_input),
                    shape_input: original(&shape_input),
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
    if metadata_indices.len() > limits.max_shape_nodes {
        return Err(ImportError::with_kind(
            ErrorKind::ResourceLimit,
            "combined shape dependency node limit exceeded",
        ));
    }
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
            validate_shape_constant(&constant.data)?;
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
        validate_shape_constant(&c.data)?;
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
        if node.op_type == "Reshape" && node.inputs.len() > 1 {
            if constants.get(&resolve(&node.inputs[1], &aliases)).is_none() {
                node.inputs.truncate(1);
            }
        } else if node.op_type == "Resize" {
            let opset = opsets[domain_key(&node.domain)];
            if node.inputs.len() > 4 || node.inputs.get(3).is_some_and(|name| !name.is_empty()) {
                return Err(ImportError::with_kind(
                    ErrorKind::Unsupported,
                    format!("{}: Resize sizes input is unsupported", node.name),
                ));
            }
            let scales_index = if opset == 10 { 1 } else { 2 };
            if let Some(scales) = node
                .inputs
                .get(scales_index)
                .and_then(|name| constants.f32(name))
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
            let name = resolve(&node.inputs[1], &aliases);
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
            .collect::<Vec<_>>();
        node.outputs = node
            .outputs
            .iter()
            .map(|name| resolve(name, &aliases))
            .collect::<Vec<_>>();
        transformed.push(WorkNode {
            key: format!("node:{origin}"),
            origin,
            raw: node,
        });
    }
    if policy == OptimizationPolicy::Optimized {
        fold_bn(&mut transformed, &mut constants, &output_names)?;
        merge_branches(&mut transformed, &mut constants, &output_names)?;
    }
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
                metadata.map_or(ShapeSpec::UnknownRank, |v| v.shape.clone()),
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
            domain: n.domain.clone(),
            opset: *opsets.get(domain_key(&n.domain)).unwrap_or(&0),
            source_indices: n.source_indices,
            inputs: ids(&n.inputs)?,
            outputs: ids(&n.outputs)?,
            operator: attributes,
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
            let source = resolve(&node.inputs[0], &aliases);
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
        let Some(elements) = weights.len().checked_add(channels) else {
            continue;
        };
        if !constants.can_generate(elements) {
            continue;
        }
        let mut folded_weights = allocate_f32(weights.len())?;
        let mut folded_bias = allocate_f32(channels)?;
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
        constants.generated += elements * 4;
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
            // Never turn sparse orthogonal branches into a larger dense kernel.
            if len > first.weights.len().saturating_add(second.weights.len()) {
                continue;
            }
            let bias_len = if first.bias.is_some() || second.bias.is_some() {
                first.output_channels
            } else {
                0
            };
            let Some(elements) = len.checked_add(bias_len) else {
                continue;
            };
            if !constants.can_generate(elements) {
                continue;
            }
            let mut weights = allocate_f32(len)?;
            accumulate(&mut weights, &first, kh, kw);
            accumulate(&mut weights, &second, kh, kw);
            let bias = if bias_len != 0 {
                let mut bias = allocate_f32(bias_len)?;
                for (i, value) in bias.iter_mut().enumerate() {
                    *value = first.bias.map_or(0.0, |b| b[i]) + second.bias.map_or(0.0, |b| b[i]);
                }
                Some(bias)
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
            constants.generated += elements * 4;
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
            let removed = item.take().expect("present constant");
            constants.bytes -= removed
                .data
                .shape()
                .byte_count(removed.data.dtype())
                .expect("validated tensor bytes");
        }
    }
    // Keep the names reserved, preserving C# UniqueName collision behavior.
}

fn allocate_f32(len: usize) -> Result<Vec<f32>> {
    let mut values = Vec::new();
    values.try_reserve_exact(len).map_err(|e| {
        ImportError::caused_by(ErrorKind::Allocation, "generated coefficient allocation", e)
    })?;
    values.resize(len, 0.0);
    Ok(values)
}

fn validate_shape_constant(data: &TensorData) -> Result<()> {
    if data.shape().rank() > 1 || data.shape().element_count() > crate::tensor::MAX_RANK {
        return Err(ImportError::with_kind(
            ErrorKind::ResourceLimit,
            "shape constants require at most eight values in a scalar or vector",
        ));
    }
    Ok(())
}
