use super::{RuntimeError, RuntimeResult};
use crate::{
    model::Model,
    onnx::{AttributeValue, RawNode},
    tensor::{MAX_RANK, Shape, TensorData},
};
use std::collections::HashMap;

#[derive(Clone, Copy)]
pub(super) struct Integers {
    data: [i64; MAX_RANK],
    len: usize,
}
impl Integers {
    fn new(values: &[i64]) -> RuntimeResult<Self> {
        if values.len() > MAX_RANK {
            return Err(RuntimeError::new("shape vector exceeds rank-eight limit"));
        }
        let mut data = [0; MAX_RANK];
        data[..values.len()].copy_from_slice(values);
        Ok(Self {
            data,
            len: values.len(),
        })
    }
    pub(super) fn values(&self) -> &[i64] {
        &self.data[..self.len]
    }
}
enum Value {
    Constant(Integers),
    Shape(usize),
    Slice { input: usize, start: i64, end: i64 },
    Concat(Vec<usize>),
}
pub(super) struct ShapeProgram {
    values: Vec<Value>,
    targets: HashMap<usize, usize>,
}
impl ShapeProgram {
    pub(super) fn compile(model: &Model) -> RuntimeResult<Self> {
        let metadata = model.shape_metadata();
        if metadata.nodes.len() > 64 {
            return Err(RuntimeError::new(
                "CLS shape program exceeds 64-instruction limit",
            ));
        }
        let mut values = Vec::new();
        let mut names = HashMap::new();
        for tensor in &metadata.constants {
            if tensor.data.dimensions().len() != 1 {
                return Err(RuntimeError::new("shape constants must have rank one"));
            }
            let data = match &tensor.data {
                TensorData::I64(t) => Integers::new(t.data())?,
                TensorData::I32(t) => {
                    Integers::new(&t.data().iter().map(|&v| i64::from(v)).collect::<Vec<_>>())?
                }
                _ => return Err(RuntimeError::new("shape constants must be I32/I64")),
            };
            if names.insert(tensor.name.clone(), values.len()).is_some() {
                return Err(RuntimeError::new("duplicate shape constant name"));
            }
            values.push(Value::Constant(data));
        }
        let graph = model.graph();
        let data_names: HashMap<_, _> = graph
            .tensors
            .iter()
            .enumerate()
            .map(|(id, t)| (t.name.as_str(), id))
            .collect();
        for node in &metadata.nodes {
            let fail = |message: &str| {
                RuntimeError::new(format!(
                    "shape node '{}' ({}): {message}",
                    node.name, node.op_type
                ))
            };
            if !matches!(node.domain.as_str(), "" | "ai.onnx")
                || node.outputs.len() != 1
                || node.outputs[0].is_empty()
            {
                return Err(fail("invalid domain or output slots"));
            }
            let source = |name: &str| {
                names
                    .get(name)
                    .copied()
                    .ok_or_else(|| fail("unresolved or non-topological integer input"))
            };
            let value = match node.op_type.as_str() {
                "Shape" => {
                    if node.inputs.len() != 1 || !node.attributes.is_empty() {
                        return Err(fail("CLS Shape requires one input and no attributes"));
                    }
                    Value::Shape(
                        *data_names
                            .get(node.inputs[0].as_str())
                            .ok_or_else(|| fail("unresolved Shape data source"))?,
                    )
                }
                "Slice" => {
                    if node.inputs.len() != 1
                        || node
                            .attributes
                            .iter()
                            .any(|a| !matches!(a.name.as_str(), "starts" | "ends" | "axes"))
                    {
                        return Err(fail(
                            "CLS shape Slice supports attribute-form starts/ends/axes with step one",
                        ));
                    }
                    let starts = ints(node, "starts")?.ok_or_else(|| fail("missing starts"))?;
                    let ends = ints(node, "ends")?.ok_or_else(|| fail("missing ends"))?;
                    let axes = ints(node, "axes")?.unwrap_or(&[0]);
                    if starts.len() != 1
                        || ends.len() != 1
                        || axes.len() != 1
                        || !matches!(axes[0], 0 | -1)
                    {
                        return Err(fail("shape Slice requires one rank-one axis"));
                    }
                    Value::Slice {
                        input: source(&node.inputs[0])?,
                        start: starts[0],
                        end: ends[0],
                    }
                }
                "Concat" => {
                    if node.inputs.is_empty() || node.attributes.iter().any(|a| a.name != "axis") {
                        return Err(fail("invalid Concat inputs/attributes"));
                    }
                    let axis = int(node, "axis")?.ok_or_else(|| fail("missing Concat axis"))?;
                    if !matches!(axis, 0 | -1) {
                        return Err(fail("shape Concat requires rank-one axis"));
                    }
                    Value::Concat(
                        node.inputs
                            .iter()
                            .map(|name| source(name))
                            .collect::<RuntimeResult<_>>()?,
                    )
                }
                _ => return Err(fail("unsupported metadata operator in CLS shape program")),
            };
            if names
                .insert(node.outputs[0].clone(), values.len())
                .is_some()
            {
                return Err(fail("duplicate shape value producer"));
            }
            values.push(value);
        }
        let mut targets = HashMap::new();
        for recipe in model.shape_recipes() {
            let (index, node) = graph
                .nodes
                .iter()
                .enumerate()
                .find(|(_, n)| n.key == recipe.node_key)
                .ok_or_else(|| RuntimeError::new("shape recipe has no retained node"))?;
            if node.op_type != "Reshape"
                || node.inputs.len() != 1
                || node.inputs[0].is_none_or(|id| graph.tensors[id.0].key != recipe.data_input)
            {
                return Err(RuntimeError::new(format!(
                    "{}: shape recipe data relationship is invalid",
                    recipe.node_key
                )));
            }
            let name = recipe
                .shape_input
                .strip_prefix("original:")
                .ok_or_else(|| RuntimeError::new("shape recipe target key is invalid"))?;
            let target = names
                .get(name)
                .copied()
                .ok_or_else(|| RuntimeError::new("shape recipe target is unresolved"))?;
            if targets.insert(index, target).is_some() {
                return Err(RuntimeError::new("duplicate reshape recipe"));
            }
        }
        for (index, node) in graph.nodes.iter().enumerate() {
            if node.op_type == "Reshape" && node.inputs.len() == 1 && !targets.contains_key(&index)
            {
                return Err(RuntimeError::new(format!(
                    "node {index} '{}': dynamic Reshape has no evaluable shape recipe",
                    node.name
                )));
            }
        }
        Ok(Self { values, targets })
    }
    pub(super) fn cache(&self) -> Vec<Option<Integers>> {
        vec![None; self.values.len()]
    }
    pub(super) fn evaluate(
        &self,
        node: usize,
        shapes: &[Option<Shape>],
        cache: &mut [Option<Integers>],
    ) -> RuntimeResult<Integers> {
        let target = *self
            .targets
            .get(&node)
            .ok_or_else(|| RuntimeError::new("missing shape recipe"))?;
        self.value(target, shapes, cache)
    }
    fn value(
        &self,
        id: usize,
        shapes: &[Option<Shape>],
        cache: &mut [Option<Integers>],
    ) -> RuntimeResult<Integers> {
        if let Some(value) = cache[id] {
            return Ok(value);
        }
        let value = match &self.values[id] {
            Value::Constant(value) => *value,
            Value::Shape(tensor) => {
                let shape = shapes[*tensor]
                    .ok_or_else(|| RuntimeError::new("Shape source has not yet been inferred"))?;
                let mut data = [0; MAX_RANK];
                for (axis, &dimension) in shape.dimensions().iter().enumerate() {
                    data[axis] = i64::try_from(dimension)
                        .map_err(|_| RuntimeError::new("shape dimension exceeds I64"))?;
                }
                Integers {
                    data,
                    len: shape.rank(),
                }
            }
            Value::Slice { input, start, end } => {
                let source = self.value(*input, shapes, cache)?;
                let normalize = |v: i64| {
                    let len = source.len as i64;
                    (if v < 0 { v + len } else { v }).clamp(0, len) as usize
                };
                let start = normalize(*start);
                let end = normalize(*end).max(start);
                Integers::new(&source.values()[start..end])?
            }
            Value::Concat(inputs) => {
                let mut output = Integers {
                    data: [0; MAX_RANK],
                    len: 0,
                };
                for &input in inputs {
                    let value = self.value(input, shapes, cache)?;
                    let end = output
                        .len
                        .checked_add(value.len)
                        .filter(|&len| len <= MAX_RANK)
                        .ok_or_else(|| {
                            RuntimeError::new("shape Concat exceeds rank-eight limit")
                        })?;
                    output.data[output.len..end].copy_from_slice(value.values());
                    output.len = end;
                }
                output
            }
        };
        cache[id] = Some(value);
        Ok(value)
    }
}
fn ints<'a>(node: &'a RawNode, name: &str) -> RuntimeResult<Option<&'a [i64]>> {
    match node
        .attributes
        .iter()
        .find(|a| a.name == name)
        .map(|a| &a.value)
    {
        None => Ok(None),
        Some(AttributeValue::Ints(v)) => Ok(Some(v)),
        _ => Err(RuntimeError::new(format!(
            "shape attribute {name} must be INTS"
        ))),
    }
}
fn int(node: &RawNode, name: &str) -> RuntimeResult<Option<i64>> {
    match node
        .attributes
        .iter()
        .find(|a| a.name == name)
        .map(|a| &a.value)
    {
        None => Ok(None),
        Some(AttributeValue::Int(v)) => Ok(Some(*v)),
        _ => Err(RuntimeError::new(format!(
            "shape attribute {name} must be INT"
        ))),
    }
}
pub(super) fn axis(value: i64, rank: usize) -> RuntimeResult<usize> {
    let normalized = if value < 0 {
        value.checked_add(rank as i64)
    } else {
        Some(value)
    };
    normalized
        .and_then(|v| usize::try_from(v).ok())
        .filter(|&v| v < rank)
        .ok_or_else(|| RuntimeError::new("axis is outside tensor rank"))
}
pub(super) fn reshape(input: Shape, target: &[i64], allowzero: bool) -> RuntimeResult<Shape> {
    if target.len() > MAX_RANK {
        return Err(RuntimeError::new("reshape target exceeds rank eight"));
    }
    let mut dims = [0; MAX_RANK];
    let mut inferred = None;
    let mut known = 1usize;
    for (axis, &value) in target.iter().enumerate() {
        let value = if value == 0 && !allowzero {
            *input
                .dimensions()
                .get(axis)
                .ok_or_else(|| RuntimeError::new("reshape zero refers to absent input axis"))?
        } else if value == -1 {
            if inferred.replace(axis).is_some() {
                return Err(RuntimeError::new(
                    "reshape has multiple inferred dimensions",
                ));
            }
            continue;
        } else {
            usize::try_from(value).map_err(|_| {
                RuntimeError::new("reshape dimension is below -1 or does not fit usize")
            })?
        };
        if value == 0 {
            return Err(RuntimeError::new("zero-sized activations are unsupported"));
        }
        dims[axis] = value;
        known = known
            .checked_mul(value)
            .ok_or_else(|| RuntimeError::new("reshape product overflow"))?;
    }
    if let Some(axis) = inferred {
        if !input.element_count().is_multiple_of(known) {
            return Err(RuntimeError::new(
                "reshape inferred dimension is not integral",
            ));
        }
        dims[axis] = input.element_count() / known;
    } else if known != input.element_count() {
        return Err(RuntimeError::new("reshape changes element count"));
    }
    Ok(Shape::new(&dims[..target.len()])?)
}
