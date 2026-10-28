//! Streaming ONNX protobuf subset used by the shipped OCR models.
//! Wire decoding is safe, bounded, and independent of graph normalization.

use crate::tensor::{DType, Dim, Shape, ShapeSpec, Tensor, TensorData};
use std::{collections::BTreeMap, fmt, io::BufRead};

pub use crate::error::ImportError;

type Result<T> = std::result::Result<T, ImportError>;

fn push<T>(values: &mut Vec<T>, value: T) -> Result<()> {
    values.try_reserve(1).map_err(|error| {
        ImportError::caused_by(
            crate::error::ErrorKind::Allocation,
            "metadata allocation failed",
            error,
        )
    })?;
    values.push(value);
    Ok(())
}

pub const MAX_SHAPE_NODES: usize = 64;

#[derive(Debug, Clone)]
pub struct LoadLimits {
    pub max_input_bytes: u64,
    pub max_depth: usize,
    pub max_rank: usize,
    /// Maximum source graph nodes, independent of protobuf nesting.
    pub max_graph_nodes: usize,
    /// Maximum combined dynamic shape dependency nodes; at most MAX_SHAPE_NODES.
    pub max_shape_nodes: usize,
    /// Maximum bytes in the normalization constant store, including generated weights.
    pub max_constant_bytes: usize,
    /// Cumulative coefficient bytes an import may generate through optimization.
    pub max_generated_bytes: usize,
}
impl Default for LoadLimits {
    fn default() -> Self {
        Self {
            max_input_bytes: 1 << 30,
            max_depth: 64,
            max_rank: 8,
            max_graph_nodes: 100_000,
            max_shape_nodes: MAX_SHAPE_NODES,
            max_constant_bytes: 1 << 30,
            max_generated_bytes: 1 << 30,
        }
    }
}
#[derive(Debug)]
pub struct RawModel {
    pub ir_version: i64,
    pub opsets: BTreeMap<String, i64>,
    pub graph: RawGraph,
    pub source_bytes: u64,
}
#[derive(Debug, Default)]
pub struct RawGraph {
    pub nodes: Vec<RawNode>,
    pub initializers: Vec<RawTensor>,
    pub inputs: Vec<ValueInfo>,
    pub outputs: Vec<ValueInfo>,
    pub value_info: Vec<ValueInfo>,
}
#[derive(Debug, Clone)]
pub struct RawNode {
    pub source_indices: Vec<usize>,
    pub name: String,
    pub op_type: String,
    pub domain: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attributes: Vec<Attribute>,
}
#[derive(Debug, Clone)]
pub struct Attribute {
    pub name: String,
    pub value: AttributeValue,
}
#[derive(Debug, Clone)]
pub enum AttributeValue {
    Float(f32),
    Int(i64),
    String(Vec<u8>),
    Tensor(Box<RawTensor>),
    Floats(Vec<f32>),
    Ints(Vec<i64>),
    Strings(Vec<Vec<u8>>),
}
#[derive(Debug, Clone)]
pub struct RawTensor {
    pub name: String,
    pub data: TensorData,
}
#[derive(Debug, Clone)]
pub struct ValueInfo {
    pub name: String,
    pub dtype: DType,
    pub shape: ShapeSpec,
}

struct WireReader<R> {
    input: R,
    position: u64,
    ends: Vec<u64>,
    limits: LoadLimits,
}

// Decode the usual type-before-payload order directly into its final allocation.
// Keep the byte fallback for legal protobuf field reordering and repeated types.
enum TensorPayload {
    Bytes(Vec<u8>),
    F32(Vec<f32>),
    I32(Vec<i32>),
    I64(Vec<i64>),
}
impl TensorPayload {
    fn into_tensor(self, dtype: DType, shape: Shape) -> Result<TensorData> {
        match (dtype, self) {
            (DType::F32, Self::F32(v)) => Ok(TensorData::F32(Tensor::from_vec(shape, v)?)),
            (DType::I32, Self::I32(v)) => Ok(TensorData::I32(Tensor::from_vec(shape, v)?)),
            (DType::I64, Self::I64(v)) => Ok(TensorData::I64(Tensor::from_vec(shape, v)?)),
            (_, Self::Bytes(v)) => Ok(TensorData::from_raw(dtype, shape, v)?),
            (_, values) => {
                fn encode<T, const N: usize>(
                    values: Vec<T>,
                    bytes: impl Fn(T) -> [u8; N],
                ) -> Result<Vec<u8>> {
                    let mut result = Vec::new();
                    let length = values
                        .len()
                        .checked_mul(N)
                        .ok_or_else(|| ImportError::new("tensor payload overflow"))?;
                    result.try_reserve_exact(length).map_err(|e| {
                        ImportError::caused_by(
                            crate::error::ErrorKind::Allocation,
                            "tensor payload",
                            e,
                        )
                    })?;
                    for value in values {
                        result.extend_from_slice(&bytes(value));
                    }
                    Ok(result)
                }
                let bytes = match values {
                    Self::F32(v) => encode(v, f32::to_le_bytes)?,
                    Self::I32(v) => encode(v, i32::to_le_bytes)?,
                    Self::I64(v) => encode(v, i64::to_le_bytes)?,
                    Self::Bytes(_) => unreachable!(),
                };
                Ok(TensorData::from_raw(dtype, shape, bytes)?)
            }
        }
    }
}
impl<R: BufRead> WireReader<R> {
    fn new(input: R, limits: LoadLimits) -> Result<Self> {
        if limits.max_input_bytes == 0
            || limits.max_depth == 0
            || limits.max_rank > 8
            || limits.max_shape_nodes > MAX_SHAPE_NODES
        {
            return Err(ImportError::new(
                "loader limits require positive bytes/depth, rank <= 8 and shape nodes <= 64",
            ));
        }
        Ok(Self {
            input,
            position: 0,
            ends: Vec::new(),
            limits,
        })
    }
    fn error(&self, message: impl fmt::Display) -> ImportError {
        ImportError::new(format!("ONNX byte {}: {message}", self.position))
    }
    fn bound(&self) -> u64 {
        self.ends
            .last()
            .copied()
            .unwrap_or(self.limits.max_input_bytes)
    }
    fn check_count(&self, count: u64) -> Result<()> {
        if self
            .position
            .checked_add(count)
            .is_none_or(|end| end > self.bound())
        {
            return Err(self.error("length exceeds message boundary or input limit"));
        }
        Ok(())
    }
    fn byte(&mut self) -> Result<u8> {
        self.check_count(1)?;
        let buffer = self
            .input
            .fill_buf()
            .map_err(|e| ImportError::from(e).context(format!("ONNX byte {}", self.position)))?;
        let Some(&value) = buffer.first() else {
            return Err(self.error("truncated protobuf"));
        };
        self.input.consume(1);
        self.position += 1;
        Ok(value)
    }
    fn varint(&mut self) -> Result<u64> {
        let mut value = 0u64;
        for shift in (0..70).step_by(7) {
            let byte = self.byte()?;
            if shift == 63 && byte > 1 {
                return Err(self.error("varint overflow"));
            }
            value |= u64::from(byte & 127) << shift;
            if byte & 128 == 0 {
                return Ok(value);
            }
        }
        Err(self.error("varint overflow"))
    }
    fn tag(&mut self) -> Result<Option<(u32, u8)>> {
        if self.ends.last().is_some_and(|&end| self.position == end) {
            return Ok(None);
        }
        if self
            .input
            .fill_buf()
            .map_err(|e| ImportError::from(e).context(format!("ONNX byte {}", self.position)))?
            .is_empty()
        {
            if self.ends.is_empty() {
                return Ok(None);
            }
            return Err(self.error("truncated nested message"));
        }
        let tag = self.varint()?;
        if tag > u64::from(u32::MAX) || tag >> 3 == 0 {
            return Err(self.error("invalid protobuf tag"));
        }
        Ok(Some(((tag >> 3) as u32, (tag & 7) as u8)))
    }
    fn enter(&mut self) -> Result<u64> {
        let length = self.varint()?;
        self.check_count(length)?;
        if self.ends.len() >= self.limits.max_depth {
            return Err(ImportError::with_kind(
                crate::error::ErrorKind::ResourceLimit,
                format!(
                    "ONNX byte {}: message nesting limit exceeded",
                    self.position
                ),
            ));
        }
        let end = self.position + length;
        self.ends.push(end);
        Ok(end)
    }
    fn leave(&mut self, end: u64) -> Result<()> {
        if self.position != end || self.ends.pop() != Some(end) {
            return Err(self.error("nested message was not fully consumed"));
        }
        Ok(())
    }
    fn read_into(&mut self, destination: &mut [u8]) -> Result<()> {
        self.check_count(destination.len() as u64)?;
        let mut offset = 0;
        while offset < destination.len() {
            let input = self.input.fill_buf().map_err(|e| {
                ImportError::from(e).context(format!("ONNX byte {}", self.position))
            })?;
            if input.is_empty() {
                return Err(self.error("truncated bytes"));
            }
            let count = input.len().min(destination.len() - offset);
            destination[offset..offset + count].copy_from_slice(&input[..count]);
            self.input.consume(count);
            self.position += count as u64;
            offset += count;
        }
        Ok(())
    }
    fn fixed32(&mut self) -> Result<u32> {
        let mut bytes = [0; 4];
        self.read_into(&mut bytes)?;
        Ok(u32::from_le_bytes(bytes))
    }
    fn bytes(&mut self) -> Result<Vec<u8>> {
        let length = self.varint()?;
        self.check_count(length)?;
        let length = usize::try_from(length).map_err(|_| self.error("length exceeds usize"))?;
        self.bytes_count(length)
    }
    fn bytes_count(&mut self, length: usize) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        // Allocate only for bytes actually available from the reader. A forged
        // long length on a truncated stream must not trigger a large allocation.
        while bytes.len() < length {
            let position = self.position;
            let input = self.input.fill_buf().map_err(|e| {
                ImportError::from(e).context(format!("ONNX byte {}", self.position))
            })?;
            if input.is_empty() {
                return Err(self.error("truncated bytes"));
            }
            let count = input.len().min(length - bytes.len());
            bytes.try_reserve(count).map_err(|error| {
                ImportError::caused_by(
                    crate::error::ErrorKind::Allocation,
                    format!("ONNX byte {position}: allocation failed"),
                    error,
                )
            })?;
            bytes.extend_from_slice(&input[..count]);
            self.input.consume(count);
            self.position += count as u64;
        }
        Ok(bytes)
    }
    fn tensor_payload(&mut self, dtype: i64) -> Result<TensorPayload> {
        let length = self.varint()?;
        self.check_count(length)?;
        let length = usize::try_from(length).map_err(|_| self.error("length exceeds usize"))?;
        Ok(match dtype {
            1 if length.is_multiple_of(4) => {
                TensorPayload::F32(self.numeric_payload(length, f32::from_le_bytes)?)
            }
            6 if length.is_multiple_of(4) => {
                TensorPayload::I32(self.numeric_payload(length, i32::from_le_bytes)?)
            }
            7 if length.is_multiple_of(8) => {
                TensorPayload::I64(self.numeric_payload(length, i64::from_le_bytes)?)
            }
            _ => TensorPayload::Bytes(self.bytes_count(length)?),
        })
    }
    fn numeric_payload<T, const N: usize>(
        &mut self,
        mut remaining: usize,
        convert: impl Fn([u8; N]) -> T,
    ) -> Result<Vec<T>> {
        let mut values = Vec::new();
        while remaining != 0 {
            let input = self.input.fill_buf()?;
            let count = (input.len().min(remaining) / N) * N;
            // Reserve only for bytes actually available, never a declared length.
            if count != 0 {
                values.try_reserve(count / N).map_err(|e| {
                    ImportError::caused_by(crate::error::ErrorKind::Allocation, "tensor payload", e)
                })?;
                values.extend(
                    input[..count]
                        .as_chunks::<N>()
                        .0
                        .iter()
                        .map(|bytes| convert(*bytes)),
                );
                self.input.consume(count);
                self.position += count as u64;
                remaining -= count;
            } else {
                let mut bytes = [0; N];
                self.read_into(&mut bytes)?;
                push(&mut values, convert(bytes))?;
                remaining -= N;
            }
        }
        Ok(values)
    }
    fn string(&mut self) -> Result<String> {
        String::from_utf8(self.bytes()?).map_err(|error| {
            ImportError::caused_by(
                crate::error::ErrorKind::InvalidModel,
                format!("ONNX byte {}: invalid UTF-8", self.position),
                error,
            )
        })
    }
    fn skip_count(&mut self, mut count: u64) -> Result<()> {
        self.check_count(count)?;
        while count != 0 {
            let input = self.input.fill_buf().map_err(|e| {
                ImportError::from(e).context(format!("ONNX byte {}", self.position))
            })?;
            if input.is_empty() {
                return Err(self.error("truncated skipped field"));
            }
            let take = u64::try_from(input.len()).unwrap_or(u64::MAX).min(count) as usize;
            self.input.consume(take);
            self.position += take as u64;
            count -= take as u64;
        }
        Ok(())
    }
    fn skip(&mut self, wire: u8) -> Result<()> {
        match wire {
            0 => {
                self.varint()?;
            }
            1 => self.skip_count(8)?,
            2 => {
                let count = self.varint()?;
                self.skip_count(count)?;
            }
            5 => self.skip_count(4)?,
            _ => return Err(self.error(format!("unsupported wire type {wire}"))),
        }
        Ok(())
    }
    fn integers(&mut self, wire: u8, values: &mut Vec<i64>) -> Result<()> {
        self.integers_limited(wire, values, usize::MAX)
    }
    fn integers_limited(&mut self, wire: u8, values: &mut Vec<i64>, limit: usize) -> Result<()> {
        if wire == 0 {
            if values.len() >= limit {
                return Err(self.error("tensor rank exceeds loader limit"));
            }
            push(values, self.varint()? as i64)?;
        } else if wire == 2 {
            let end = self.enter()?;
            while self.position < end {
                if values.len() >= limit {
                    return Err(self.error("tensor rank exceeds loader limit"));
                }
                push(values, self.varint()? as i64)?;
            }
            self.leave(end)?;
        } else {
            return Err(self.error("integer field has wrong wire type"));
        }
        Ok(())
    }
    fn floats(&mut self, wire: u8, values: &mut Vec<f32>) -> Result<()> {
        if wire == 5 {
            push(values, f32::from_bits(self.fixed32()?))?;
        } else if wire == 2 {
            let end = self.enter()?;
            if !(end - self.position).is_multiple_of(4) {
                return Err(self.error("truncated packed f32"));
            }
            while self.position < end {
                push(values, f32::from_bits(self.fixed32()?))?;
            }
            self.leave(end)?;
        } else {
            return Err(self.error("float field has wrong wire type"));
        }
        Ok(())
    }
}

pub fn parse<R: BufRead>(input: R, limits: &LoadLimits) -> Result<RawModel> {
    let mut reader = WireReader::new(input, limits.clone())?;
    let mut ir_version = 0;
    let mut opsets = BTreeMap::new();
    let mut graph = None;
    while let Some((field, wire)) = reader.tag()? {
        match (field, wire) {
            (1, 0) => {
                ir_version = i64::try_from(reader.varint()?)
                    .map_err(|_| reader.error("IR version overflow"))?
            }
            (8, 2) => {
                let end = reader.enter()?;
                let mut domain = String::new();
                let mut version = 0;
                while let Some((field, wire)) = reader.tag()? {
                    match (field, wire) {
                        (1, 2) => domain = reader.string()?,
                        (2, 0) => {
                            version = i64::try_from(reader.varint()?)
                                .map_err(|_| reader.error("opset version overflow"))?
                        }
                        _ => reader.skip(wire)?,
                    }
                }
                reader.leave(end)?;
                if domain == "ai.onnx" {
                    domain.clear();
                }
                if version <= 0 || opsets.insert(domain, version).is_some() {
                    return Err(reader.error("invalid or duplicate opset import"));
                }
            }
            (7, 2) => {
                if graph.is_some() {
                    return Err(reader.error("multiple ModelProto graphs"));
                }
                let end = reader.enter()?;
                graph = Some(parse_graph(&mut reader)?);
                reader.leave(end)?;
            }
            _ => reader.skip(wire)?,
        }
    }
    if ir_version <= 0 {
        return Err(reader.error("missing or invalid IR version"));
    }
    let graph = graph.ok_or_else(|| reader.error("ModelProto has no graph"))?;
    Ok(RawModel {
        ir_version,
        opsets,
        graph,
        source_bytes: reader.position,
    })
}
fn parse_graph<R: BufRead>(reader: &mut WireReader<R>) -> Result<RawGraph> {
    let mut graph = RawGraph::default();
    while let Some((field, wire)) = reader.tag()? {
        match (field, wire) {
            (1, 2) => {
                let end = reader.enter()?;
                if graph.nodes.len() >= reader.limits.max_graph_nodes {
                    return Err(ImportError::with_kind(
                        crate::error::ErrorKind::ResourceLimit,
                        "graph node limit exceeded",
                    ));
                }
                let node = parse_node(reader, graph.nodes.len())?;
                reader.leave(end)?;
                push(&mut graph.nodes, node)?;
            }
            (5, 2) => {
                let end = reader.enter()?;
                let tensor = parse_tensor(reader)?;
                reader.leave(end)?;
                if tensor.name.is_empty() {
                    return Err(reader.error("graph initializer has no name"));
                }
                push(&mut graph.initializers, tensor)?;
            }
            (11..=13, 2) => {
                let end = reader.enter()?;
                let value = parse_value_info(reader)?;
                reader.leave(end)?;
                match field {
                    11 => push(&mut graph.inputs, value)?,
                    12 => push(&mut graph.outputs, value)?,
                    _ => push(&mut graph.value_info, value)?,
                }
            }
            (15, _) => return Err(reader.error("sparse initializers are unsupported")),
            _ => reader.skip(wire)?,
        }
    }
    Ok(graph)
}
fn parse_node<R: BufRead>(reader: &mut WireReader<R>, index: usize) -> Result<RawNode> {
    let mut node = RawNode {
        source_indices: vec![index],
        name: String::new(),
        op_type: String::new(),
        domain: String::new(),
        inputs: Vec::new(),
        outputs: Vec::new(),
        attributes: Vec::new(),
    };
    while let Some((field, wire)) = reader.tag()? {
        match (field, wire) {
            (1, 2) => push(&mut node.inputs, reader.string()?)?,
            (2, 2) => push(&mut node.outputs, reader.string()?)?,
            (3, 2) => node.name = reader.string()?,
            (4, 2) => node.op_type = reader.string()?,
            (5, 2) => {
                let end = reader.enter()?;
                push(&mut node.attributes, parse_attribute(reader)?)?;
                reader.leave(end)?;
            }
            (7, 2) => node.domain = reader.string()?,
            _ => reader.skip(wire)?,
        }
    }
    if node.op_type.is_empty() {
        return Err(reader.error(format!("node {index} has no op_type")));
    }
    Ok(node)
}
fn parse_attribute<R: BufRead>(reader: &mut WireReader<R>) -> Result<Attribute> {
    let mut name = String::new();
    let mut declared_kind = 0u64;
    let mut present = 0u16;
    let mut scalar = None;
    let mut floats = Vec::new();
    let mut integers = Vec::new();
    let mut strings = Vec::new();
    while let Some((field, wire)) = reader.tag()? {
        match (field, wire) {
            (1, 2) => name = reader.string()?,
            (2, 5) => {
                present |= 1 << 1;
                scalar = Some(AttributeValue::Float(f32::from_bits(reader.fixed32()?)));
            }
            (3, 0) => {
                present |= 1 << 2;
                scalar = Some(AttributeValue::Int(reader.varint()? as i64));
            }
            (4, 2) => {
                present |= 1 << 3;
                scalar = Some(AttributeValue::String(reader.bytes()?));
            }
            (5, 2) => {
                present |= 1 << 4;
                let end = reader.enter()?;
                scalar = Some(AttributeValue::Tensor(Box::new(parse_tensor(reader)?)));
                reader.leave(end)?;
            }
            (7, 2 | 5) => {
                present |= 1 << 6;
                reader.floats(wire, &mut floats)?;
            }
            (8, 0 | 2) => {
                present |= 1 << 7;
                reader.integers(wire, &mut integers)?;
            }
            (9, 2) => {
                present |= 1 << 8;
                push(&mut strings, reader.bytes()?)?;
            }
            (20, 0) => declared_kind = reader.varint()?,
            (6 | 10 | 11 | 22 | 23, _) => {
                return Err(reader.error("graph/list-tensor/sparse attributes are unsupported"));
            }
            _ => reader.skip(wire)?,
        }
    }
    if name.is_empty() {
        return Err(reader.error("attribute has no name"));
    }
    if present.count_ones() > 1 {
        return Err(reader.error(format!("attribute {name} has conflicting payload families")));
    }
    let observed_kind = if present == 0 {
        0
    } else {
        u64::from(present.trailing_zeros())
    };
    if declared_kind != 0 && observed_kind != 0 && declared_kind != observed_kind {
        return Err(reader.error(format!("attribute {name} type/value mismatch")));
    }
    let kind = if declared_kind == 0 {
        observed_kind
    } else {
        declared_kind
    };
    let value = match kind {
        6 => AttributeValue::Floats(floats),
        7 => AttributeValue::Ints(integers),
        8 => AttributeValue::Strings(strings),
        1..=4 => {
            scalar.ok_or_else(|| reader.error(format!("attribute {name} has no scalar value")))?
        }
        _ => {
            return Err(reader.error(format!(
                "attribute {name} has missing or unsupported type {kind}"
            )));
        }
    };
    Ok(Attribute { name, value })
}
fn parse_tensor<R: BufRead>(reader: &mut WireReader<R>) -> Result<RawTensor> {
    let mut name = String::new();
    let mut data_type = 0i64;
    let mut dimensions = Vec::new();
    let mut raw = None;
    let mut payload_families = 0u8;
    let mut floats = Vec::new();
    let mut integers32 = Vec::new();
    let mut integers64 = Vec::new();
    while let Some((field, wire)) = reader.tag()? {
        match (field, wire) {
            (1, 0 | 2) => reader.integers_limited(wire, &mut dimensions, reader.limits.max_rank)?,
            (2, 0) => data_type = reader.varint()? as i64,
            (4, 2 | 5) => {
                payload_families |= 1;
                reader.floats(wire, &mut floats)?;
            }
            (5, 0 | 2) => {
                payload_families |= 2;
                reader.integers(wire, &mut integers32)?;
            }
            (7, 0 | 2) => {
                payload_families |= 4;
                reader.integers(wire, &mut integers64)?;
            }
            (8, 2) => name = reader.string()?,
            (9, 2) => {
                payload_families |= 8;
                raw = Some(reader.tensor_payload(data_type)?);
            }
            (13, _) => return Err(reader.error("external tensor data is unsupported")),
            (14, 0) => {
                if reader.varint()? != 0 {
                    return Err(reader.error("external tensor data is unsupported"));
                }
            }
            (3 | 6 | 10 | 11, _) => {
                return Err(reader.error("segmented/string/double/uint64 tensors are unsupported"));
            }
            _ => reader.skip(wire)?,
        }
    }
    if dimensions.len() > reader.limits.max_rank {
        return Err(reader.error(format!("tensor {name} rank exceeds loader limit")));
    }
    let mut concrete = [0; 8];
    for (index, &dim) in dimensions.iter().enumerate() {
        concrete[index] = usize::try_from(dim).map_err(|_| {
            reader.error(format!(
                "tensor {name} has a negative or overflowing dimension"
            ))
        })?;
    }
    let shape = Shape::new(&concrete[..dimensions.len()])?;
    let dtype = DType::from_onnx(data_type)?;
    if payload_families.count_ones() > 1 {
        return Err(reader.error(format!("tensor {name} mixes payload families")));
    }
    let required_family = match dtype {
        DType::F32 => 1,
        DType::I32 | DType::U8 => 2,
        DType::I64 => 4,
    };
    if payload_families != 0 && payload_families != 8 && payload_families != required_family {
        return Err(reader.error(format!("tensor {name} payload family does not match dtype")));
    }
    let data = if let Some(bytes) = raw {
        bytes.into_tensor(dtype, shape)?
    } else {
        match dtype {
            DType::F32 if integers32.is_empty() && integers64.is_empty() => {
                TensorData::F32(Tensor::from_vec(shape, floats)?)
            }
            DType::I64 if floats.is_empty() && integers32.is_empty() => {
                TensorData::I64(Tensor::from_vec(shape, integers64)?)
            }
            DType::I32 if floats.is_empty() && integers64.is_empty() => {
                // Protobuf int32 negatives use sign-extended 10-byte varints.
                let mut data = Vec::new();
                data.try_reserve_exact(integers32.len()).map_err(|e| {
                    ImportError::caused_by(
                        crate::error::ErrorKind::Allocation,
                        format!("ONNX byte {}: typed tensor allocation", reader.position),
                        e,
                    )
                })?;
                for value in integers32 {
                    data.push(i32::try_from(value).map_err(|_| {
                        reader.error(format!("tensor {name} int32 value out of range"))
                    })?);
                }
                TensorData::I32(Tensor::from_vec(shape, data)?)
            }
            DType::U8 if floats.is_empty() && integers64.is_empty() => {
                let mut data = Vec::new();
                data.try_reserve_exact(integers32.len()).map_err(|e| {
                    ImportError::caused_by(
                        crate::error::ErrorKind::Allocation,
                        format!("ONNX byte {}: typed tensor allocation", reader.position),
                        e,
                    )
                })?;
                for value in integers32 {
                    data.push(u8::try_from(value).map_err(|_| {
                        reader.error(format!("tensor {name} u8 value out of range"))
                    })?);
                }
                TensorData::U8(Tensor::from_vec(shape, data)?)
            }
            _ => {
                return Err(
                    reader.error(format!("tensor {name} typed payload does not match dtype"))
                );
            }
        }
    };
    Ok(RawTensor { name, data })
}
fn parse_value_info<R: BufRead>(reader: &mut WireReader<R>) -> Result<ValueInfo> {
    let mut name = String::new();
    let mut tensor_type = None;
    while let Some((field, wire)) = reader.tag()? {
        match (field, wire) {
            (1, 2) => name = reader.string()?,
            (2, 2) => {
                let end = reader.enter()?;
                while let Some((field, wire)) = reader.tag()? {
                    match (field, wire) {
                        (1, 2) => {
                            let nested = reader.enter()?;
                            tensor_type = Some(parse_tensor_type(reader)?);
                            reader.leave(nested)?;
                        }
                        (4 | 5 | 8 | 9, _) => {
                            return Err(reader.error("non-tensor value types are unsupported"));
                        }
                        _ => reader.skip(wire)?,
                    }
                }
                reader.leave(end)?;
            }
            _ => reader.skip(wire)?,
        }
    }
    if name.is_empty() {
        return Err(reader.error("value_info has no name"));
    }
    let (dtype, shape) =
        tensor_type.ok_or_else(|| reader.error(format!("value {name} has no tensor type")))?;
    Ok(ValueInfo { name, dtype, shape })
}
fn parse_tensor_type<R: BufRead>(reader: &mut WireReader<R>) -> Result<(DType, ShapeSpec)> {
    let mut dtype = None;
    let mut shape = ShapeSpec::UnknownRank;
    while let Some((field, wire)) = reader.tag()? {
        match (field, wire) {
            (1, 0) => dtype = Some(DType::from_onnx(reader.varint()? as i64)?),
            (2, 2) => {
                let end = reader.enter()?;
                shape = ShapeSpec::Ranked(parse_shape(reader)?);
                reader.leave(end)?;
            }
            _ => reader.skip(wire)?,
        }
    }
    Ok((
        dtype.ok_or_else(|| reader.error("tensor type has no dtype"))?,
        shape,
    ))
}
fn parse_shape<R: BufRead>(reader: &mut WireReader<R>) -> Result<Vec<Dim>> {
    let mut dimensions = Vec::new();
    while let Some((field, wire)) = reader.tag()? {
        if field == 1 && wire == 2 {
            let end = reader.enter()?;
            let mut dim = Dim::Unknown;
            while let Some((field, wire)) = reader.tag()? {
                match (field, wire) {
                    (1, 0) => {
                        let value = reader.varint()? as i64;
                        dim =
                            Dim::Known(usize::try_from(value).map_err(|_| {
                                reader.error("negative/overflowing shape dimension")
                            })?);
                    }
                    (2, 2) => dim = Dim::Symbol(reader.string()?),
                    _ => reader.skip(wire)?,
                }
            }
            reader.leave(end)?;
            push(&mut dimensions, dim)?;
            if dimensions.len() > reader.limits.max_rank {
                return Err(reader.error("value shape rank exceeds loader limit"));
            }
        } else {
            reader.skip(wire)?;
        }
    }
    Ok(dimensions)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufReader, Cursor};
    fn var(mut v: u64) -> Vec<u8> {
        let mut out = Vec::new();
        while v >= 128 {
            out.push(v as u8 | 128);
            v >>= 7;
        }
        out.push(v as u8);
        out
    }
    fn int(field: u32, value: u64) -> Vec<u8> {
        [var(u64::from(field) << 3), var(value)].concat()
    }
    fn bytes(field: u32, value: &[u8]) -> Vec<u8> {
        [
            var((u64::from(field) << 3) | 2),
            var(value.len() as u64),
            value.to_vec(),
        ]
        .concat()
    }
    fn model(graph: Vec<u8>) -> Vec<u8> {
        [int(1, 10), bytes(8, &int(2, 14)), bytes(7, &graph)].concat()
    }
    fn tensor(dtype: u64, dims: &[i64], payload: Vec<u8>) -> Vec<u8> {
        let mut parts = vec![bytes(8, b"w"), int(2, dtype)];
        parts.extend(dims.iter().map(|&d| int(1, d as u64)));
        parts.push(payload);
        parts.concat()
    }
    fn read(bytes: &[u8]) -> Result<RawModel> {
        parse(
            BufReader::with_capacity(1, Cursor::new(bytes)),
            &LoadLimits::default(),
        )
    }
    #[test]
    fn raw_dtypes_and_empty_scalar() {
        for (dtype, payload) in [
            (1, 1f32.to_le_bytes().to_vec()),
            (6, (-3i32).to_le_bytes().to_vec()),
            (7, (-9i64).to_le_bytes().to_vec()),
            (2, vec![255]),
        ] {
            let parsed = read(&model(bytes(5, &tensor(dtype, &[], bytes(9, &payload))))).unwrap();
            assert_eq!(parsed.graph.initializers[0].data.len(), 1);
        }
        let parsed = read(&model(bytes(5, &tensor(1, &[0], vec![])))).unwrap();
        assert!(parsed.graph.initializers[0].data.is_empty());
    }

    #[test]
    fn direct_tensor_payload_preserves_bits_and_stream_boundaries() {
        let bits = [
            0x80000000u32,
            0x7fc01234,
            0x7f800000,
            0x00000001,
            0x3f800000,
        ];
        let payload: Vec<_> = bits.iter().flat_map(|v| v.to_le_bytes()).collect();
        let wire = model(bytes(
            5,
            &tensor(1, &[bits.len() as i64], bytes(9, &payload)),
        ));
        for capacity in [1, 3, 4, 7, 8, 19, 65536] {
            let parsed = parse(
                BufReader::with_capacity(capacity, Cursor::new(&wire)),
                &LoadLimits::default(),
            )
            .unwrap();
            assert_eq!(
                parsed.graph.initializers[0]
                    .data
                    .as_f32()
                    .unwrap()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                bits
            );
        }
        for end in 0..wire.len() {
            assert!(read(&wire[..end]).is_err());
        }
        // Protobuf permits a later type field to replace the earlier value.
        let repeated = [tensor(7, &[2], bytes(9, &payload[..8])), int(2, 1)].concat();
        let parsed = read(&model(bytes(5, &repeated))).unwrap();
        assert_eq!(
            parsed.graph.initializers[0].data.as_f32().unwrap()[0].to_bits(),
            bits[0]
        );
    }
    #[test]
    fn packed_unpacked_signed_and_range() {
        for payload in [int(7, (-42i64) as u64), bytes(7, &var((-42i64) as u64))] {
            let parsed = read(&model(bytes(5, &tensor(7, &[], payload)))).unwrap();
            assert_eq!(parsed.graph.initializers[0].data.as_i64().unwrap(), &[-42]);
        }
        assert!(read(&model(bytes(5, &tensor(2, &[], int(5, 256))))).is_err());
        let parsed = read(&model(bytes(5, &tensor(6, &[], int(5, (-7i32) as u64))))).unwrap();
        assert_eq!(parsed.graph.initializers[0].data.as_i32().unwrap(), &[-7]);
        let fixed = [var((4 << 3) | 5), 2.5f32.to_le_bytes().to_vec()].concat();
        for payload in [fixed, bytes(4, &2.5f32.to_le_bytes())] {
            assert_eq!(
                read(&model(bytes(5, &tensor(1, &[], payload))))
                    .unwrap()
                    .graph
                    .initializers[0]
                    .data
                    .as_f32()
                    .unwrap(),
                &[2.5]
            );
        }
    }
    #[test]
    fn metadata_distinguishes_unknown_scalar_zero_symbol() {
        let info = |shape: Option<Vec<u8>>| {
            let mut ty = int(1, 1);
            if let Some(shape) = shape {
                ty.extend(bytes(2, &shape));
            }
            [bytes(1, b"x"), bytes(2, &bytes(1, &ty))].concat()
        };
        let cases = [
            (None, ShapeSpec::UnknownRank),
            (Some(vec![]), ShapeSpec::Ranked(vec![])),
            (
                Some(bytes(1, &int(1, 0))),
                ShapeSpec::Ranked(vec![Dim::Known(0)]),
            ),
            (
                Some(bytes(1, &bytes(2, b"floor(w/8)"))),
                ShapeSpec::Ranked(vec![Dim::Symbol("floor(w/8)".into())]),
            ),
            (Some(bytes(1, &[])), ShapeSpec::Ranked(vec![Dim::Unknown])),
        ];
        for (wire, shape) in cases {
            assert_eq!(
                read(&model(bytes(11, &info(wire)))).unwrap().graph.inputs[0].shape,
                shape
            );
        }
    }
    #[test]
    fn optional_slots_and_field_order() {
        let node = [
            bytes(1, b"x"),
            bytes(1, b""),
            bytes(1, b"scales"),
            bytes(2, b"y"),
            bytes(4, b"Resize"),
        ]
        .concat();
        assert_eq!(
            read(&model(bytes(1, &node))).unwrap().graph.nodes[0].inputs,
            ["x", "", "scales"]
        );
        // raw_data deliberately precedes dimensions and dtype.
        let t = [
            bytes(9, &4f32.to_le_bytes()),
            bytes(8, b"w"),
            int(2, 1),
            int(1, 1),
        ]
        .concat();
        assert_eq!(
            read(&model(bytes(5, &t))).unwrap().graph.initializers[0]
                .data
                .as_f32()
                .unwrap(),
            &[4.]
        );
    }
    #[test]
    fn attribute_presence_and_family_are_independent_of_order() {
        let read_attribute = |payload: Vec<u8>| {
            let attr = [bytes(1, b"axes"), payload].concat();
            let node = [bytes(4, b"Relu"), bytes(5, &attr)].concat();
            read(&model(bytes(1, &node)))
        };
        for (field, kind) in [(7, 6), (8, 7)] {
            let parsed = read_attribute(bytes(field, &[])).unwrap();
            let value = &parsed.graph.nodes[0].attributes[0].value;
            assert!(
                matches!((kind, value), (6, AttributeValue::Floats(v)) if v.is_empty())
                    || matches!((kind, value), (7, AttributeValue::Ints(v)) if v.is_empty())
            );
        }
        let scalar_float = [var((2 << 3) | 5), 1f32.to_le_bytes().to_vec()].concat();
        assert!(read_attribute([int(20, 6), scalar_float.clone()].concat()).is_err());
        for families in [
            [scalar_float.clone(), int(3, 7)].concat(),
            [int(3, 7), scalar_float].concat(),
        ] {
            assert!(read_attribute(families).is_err());
        }
    }
    #[test]
    fn unknown_fields_and_exact_input_limits() {
        let unknowns = [
            int(127, 42),
            bytes(126, b"ignored"),
            [var((125 << 3) | 1), vec![0; 8]].concat(),
            [var((124 << 3) | 5), vec![0; 4]].concat(),
        ]
        .concat();
        let encoded = [model(vec![]), unknowns].concat();
        assert!(
            parse(
                encoded.as_slice(),
                &LoadLimits {
                    max_input_bytes: encoded.len() as u64,
                    ..Default::default()
                }
            )
            .is_ok()
        );
        assert!(
            parse(
                encoded.as_slice(),
                &LoadLimits {
                    max_input_bytes: encoded.len() as u64 - 1,
                    ..Default::default()
                }
            )
            .is_err()
        );
    }
    #[test]
    fn empty_payload_families_still_enforce_dtype() {
        assert!(read(&model(bytes(5, &tensor(7, &[0], bytes(4, &[]))))).is_err());
        assert!(
            read(&model(bytes(
                5,
                &tensor(1, &[0], [bytes(9, &[]), bytes(4, &[])].concat())
            )))
            .is_err()
        );
        let empty = read(&model(bytes(5, &tensor(7, &[0], bytes(7, &[]))))).unwrap();
        assert!(
            empty.graph.initializers[0]
                .data
                .as_i64()
                .unwrap()
                .is_empty()
        );
    }
    #[test]
    fn rank_and_truncated_large_payload_fail_before_collecting_remaining_data() {
        let too_many = model(bytes(5, &tensor(1, &[1; 9], bytes(9, &[]))));
        assert!(
            read(&too_many)
                .unwrap_err()
                .to_string()
                .contains("rank exceeds")
        );
        let mut announced = var(512 * 1024 * 1024);
        announced.extend([1, 2, 3]);
        let mut reader = WireReader::new(announced.as_slice(), LoadLimits::default()).unwrap();
        assert!(
            reader
                .bytes()
                .unwrap_err()
                .to_string()
                .contains("truncated bytes")
        );
    }
    #[test]
    fn malformed_wire_and_limits_are_errors() {
        let valid = model(bytes(5, &tensor(1, &[1], bytes(9, &1f32.to_le_bytes()))));
        for len in 1..valid.len() {
            assert!(read(&valid[..len]).is_err(), "prefix {len}");
        }
        assert!(read(&[8, 255, 255, 255, 255, 255, 255, 255, 255, 255, 2]).is_err());
        assert!(read(&model(bytes(5, &tensor(10, &[], bytes(9, &[0, 0]))))).is_err());
        assert!(read(&model(bytes(5, &tensor(1, &[1], int(14, 1))))).is_err());
        assert!(
            read(&model(bytes(
                1,
                &[26, 1, 255, 34, 4, b'C', b'o', b'n', b'v']
            )))
            .is_err()
        );
        assert!(
            parse(
                valid.as_slice(),
                &LoadLimits {
                    max_input_bytes: 5,
                    ..Default::default()
                }
            )
            .is_err()
        );
        assert!(
            parse(
                valid.as_slice(),
                &LoadLimits {
                    max_depth: 1,
                    ..Default::default()
                }
            )
            .is_err()
        );
    }
}
