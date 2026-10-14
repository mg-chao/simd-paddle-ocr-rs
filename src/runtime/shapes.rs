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
    scalar: bool,
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
            scalar: false,
        })
    }
    fn constant(data: &TensorData) -> RuntimeResult<Self> {
        if data.dimensions().len() > 1 {
            return Err(RuntimeError::new(
                "shape constants must have rank zero or one",
            ));
        }
        if data.shape().element_count() > MAX_RANK {
            return Err(RuntimeError::new(
                "shape constant exceeds rank-eight vector limit",
            ));
        }
        let mut value = match data {
            TensorData::I64(t) => Self::new(t.data())?,
            TensorData::I32(t) => {
                Self::new(&t.data().iter().map(|&v| i64::from(v)).collect::<Vec<_>>())?
            }
            _ => return Err(RuntimeError::new("shape constants must be I32/I64")),
        };
        value.scalar = data.dimensions().is_empty();
        Ok(value)
    }
    pub(super) fn values(&self) -> &[i64] {
        &self.data[..self.len]
    }
    fn vector(&self) -> RuntimeResult<&[i64]> {
        if self.scalar {
            return Err(RuntimeError::new(
                "shape operand requires a vector, found a scalar",
            ));
        }
        Ok(self.values())
    }
    fn shape(self) -> RuntimeResult<Shape> {
        Ok(if self.scalar {
            Shape::new(&[])?
        } else {
            Shape::new(&[self.len])?
        })
    }
    fn with_shape(mut self, shape: Shape) -> RuntimeResult<Self> {
        if shape.rank() > 1 || shape.element_count() != self.len {
            return Err(RuntimeError::new(
                "shape metadata supports only scalar/vector integer values",
            ));
        }
        self.scalar = shape.rank() == 0;
        Ok(self)
    }
}

/// The shipped Slice subset uses constant bounds and positive unit steps.
/// Shared by integer metadata and F32 data slicing, without float conversions.
#[derive(Clone)]
pub(super) struct SliceParameters {
    starts: Vec<i64>,
    ends: Vec<i64>,
    axes: Vec<i64>,
}
impl SliceParameters {
    pub(super) fn new(
        starts: Vec<i64>,
        ends: Vec<i64>,
        axes: Option<Vec<i64>>,
        steps: Option<Vec<i64>>,
    ) -> RuntimeResult<Self> {
        let count = starts.len();
        if count == 0 || count > MAX_RANK || ends.len() != count {
            return Err(RuntimeError::new(
                "Slice starts/ends must have equal nonempty bounded lengths",
            ));
        }
        let axes = axes.unwrap_or_else(|| (0..count).map(|axis| axis as i64).collect());
        let steps = steps.unwrap_or_else(|| vec![1; count]);
        if axes.len() != count || steps.len() != count {
            return Err(RuntimeError::new(
                "Slice axes/steps lengths differ from starts",
            ));
        }
        if steps.iter().any(|&step| step != 1) {
            return Err(RuntimeError::new("Slice supports positive unit steps only"));
        }
        Ok(Self { starts, ends, axes })
    }
    pub(super) fn bind(&self, input: Shape) -> RuntimeResult<(Shape, [usize; MAX_RANK])> {
        if input.rank() == 0 || self.starts.len() > input.rank() {
            return Err(RuntimeError::new(
                "Slice requires a ranked input and at most one bound per axis",
            ));
        }
        let mut dimensions = [0; MAX_RANK];
        dimensions[..input.rank()].copy_from_slice(input.dimensions());
        let mut starts = [0; MAX_RANK];
        let mut seen = [false; MAX_RANK];
        for ((&start, &end), &raw_axis) in self.starts.iter().zip(&self.ends).zip(&self.axes) {
            let axis = axis(raw_axis, input.rank())?;
            if std::mem::replace(&mut seen[axis], true) {
                return Err(RuntimeError::new("Slice axis is duplicated"));
            }
            let dimension = i64::try_from(input.dimensions()[axis])
                .map_err(|_| RuntimeError::new("Slice dimension exceeds I64"))?;
            let normalize = |value: i64| {
                (if value < 0 { value + dimension } else { value }).clamp(0, dimension) as usize
            };
            starts[axis] = normalize(start);
            dimensions[axis] = normalize(end).saturating_sub(starts[axis]);
        }
        Ok((Shape::new(&dimensions[..input.rank()])?, starts))
    }
}

enum Value {
    Constant(Integers),
    Shape(usize),
    Identity(usize),
    Slice {
        input: usize,
        parameters: SliceParameters,
    },
    Squeeze {
        input: usize,
        axes: Vec<i64>,
    },
    Unsqueeze {
        input: usize,
        axes: Vec<i64>,
    },
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
                "shape program exceeds 64-instruction limit",
            ));
        }
        let mut values = Vec::new();
        let mut names = HashMap::new();
        for tensor in &metadata.constants {
            let value = Integers::constant(&tensor.data)?;
            if names.insert(tensor.name.clone(), values.len()).is_some() {
                return Err(RuntimeError::new("duplicate shape constant name"));
            }
            values.push(Value::Constant(value));
        }
        let graph = model.graph();
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
            let constant = |name: &str| -> RuntimeResult<Vec<i64>> {
                let mut id = source(name)?;
                loop {
                    match &values[id] {
                        Value::Constant(value) => return Ok(value.vector()?.to_vec()),
                        Value::Identity(input) => id = *input,
                        _ => return Err(fail("Slice parameters must be constant integer vectors")),
                    }
                }
            };
            let value = match node.op_type.as_str() {
                "Shape" => {
                    if node.inputs.len() != 1 || !node.attributes.is_empty() {
                        return Err(fail("Shape requires one input and no attributes"));
                    }
                    Value::Shape(
                        model
                            .shape_data_source(&node.inputs[0])
                            .ok_or_else(|| fail("unresolved normalized Shape data source"))?
                            .0,
                    )
                }
                "Identity" => {
                    if node.inputs.len() != 1 || !node.attributes.is_empty() {
                        return Err(fail("Identity requires one input and no attributes"));
                    }
                    Value::Identity(source(&node.inputs[0])?)
                }
                "Slice" => {
                    let parameters = if node.inputs.len() == 1 {
                        if node.attributes.iter().any(|a| {
                            !matches!(a.name.as_str(), "starts" | "ends" | "axes" | "steps")
                        }) {
                            return Err(fail("unsupported Slice attributes"));
                        }
                        SliceParameters::new(
                            ints(node, "starts")?
                                .ok_or_else(|| fail("missing starts"))?
                                .to_vec(),
                            ints(node, "ends")?
                                .ok_or_else(|| fail("missing ends"))?
                                .to_vec(),
                            ints(node, "axes")?.map(<[i64]>::to_vec),
                            ints(node, "steps")?.map(<[i64]>::to_vec),
                        )?
                    } else {
                        if !(3..=5).contains(&node.inputs.len()) || !node.attributes.is_empty() {
                            return Err(fail(
                                "Slice requires data, starts, ends and optional axes/steps",
                            ));
                        }
                        let optional = |slot: usize| -> RuntimeResult<Option<Vec<i64>>> {
                            node.inputs
                                .get(slot)
                                .filter(|name| !name.is_empty())
                                .map(|name| constant(name))
                                .transpose()
                        };
                        SliceParameters::new(
                            constant(&node.inputs[1])?,
                            constant(&node.inputs[2])?,
                            optional(3)?,
                            optional(4)?,
                        )?
                    };
                    Value::Slice {
                        input: source(&node.inputs[0])?,
                        parameters,
                    }
                }
                "Squeeze" | "Unsqueeze" => {
                    if !(1..=2).contains(&node.inputs.len())
                        || node.attributes.iter().any(|a| a.name != "axes")
                    {
                        return Err(fail(
                            "Squeeze/Unsqueeze requires attribute or constant-input axes",
                        ));
                    }
                    let attribute_axes = ints(node, "axes")?;
                    let axes = if node.inputs.len() == 2 {
                        if attribute_axes.is_some() {
                            return Err(fail("Squeeze/Unsqueeze cannot mix axes forms"));
                        }
                        constant(&node.inputs[1])?
                    } else {
                        attribute_axes.unwrap_or(&[]).to_vec()
                    };
                    if node.op_type == "Squeeze" {
                        Value::Squeeze {
                            input: source(&node.inputs[0])?,
                            axes,
                        }
                    } else {
                        if axes.is_empty() {
                            return Err(fail("Unsqueeze requires nonempty axes"));
                        }
                        Value::Unsqueeze {
                            input: source(&node.inputs[0])?,
                            axes,
                        }
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
                _ => return Err(fail("unsupported metadata operator in shape program")),
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
        let value = self.value(target, shapes, cache)?;
        value.vector()?;
        Ok(value)
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
            Value::Identity(input) => self.value(*input, shapes, cache)?,
            Value::Shape(tensor) => {
                let shape = shapes[*tensor]
                    .ok_or_else(|| RuntimeError::new("Shape source has not yet been inferred"))?;
                let mut data = [0; MAX_RANK];
                for (axis, &dimension) in shape.dimensions().iter().enumerate() {
                    data[axis] = i64::try_from(dimension)
                        .map_err(|_| RuntimeError::new("shape dimension exceeds I64"))?;
                }
                Integers::new(&data[..shape.rank()])?
            }
            Value::Slice { input, parameters } => {
                let source = self.value(*input, shapes, cache)?;
                source.vector()?;
                let (output, starts) = parameters.bind(source.shape()?)?;
                Integers::new(&source.values()[starts[0]..starts[0] + output.element_count()])?
            }
            Value::Squeeze { input, axes } => {
                let source = self.value(*input, shapes, cache)?;
                source.with_shape(squeeze(source.shape()?, axes)?)?
            }
            Value::Unsqueeze { input, axes } => {
                let source = self.value(*input, shapes, cache)?;
                source.with_shape(unsqueeze(source.shape()?, axes)?)?
            }
            Value::Concat(inputs) => {
                let mut output = Integers::new(&[])?;
                for &input in inputs {
                    let value = self.value(input, shapes, cache)?;
                    let source = value.vector()?;
                    let end = output
                        .len
                        .checked_add(source.len())
                        .filter(|&len| len <= MAX_RANK)
                        .ok_or_else(|| {
                            RuntimeError::new("shape Concat exceeds rank-eight limit")
                        })?;
                    output.data[output.len..end].copy_from_slice(source);
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
pub(super) fn squeeze(input: Shape, axes: &[i64]) -> RuntimeResult<Shape> {
    let mut removed = [false; MAX_RANK];
    if axes.is_empty() {
        for (axis, &dimension) in input.dimensions().iter().enumerate() {
            removed[axis] = dimension == 1;
        }
    } else {
        for &value in axes {
            let axis = axis(value, input.rank())?;
            if removed[axis] {
                return Err(RuntimeError::new("Squeeze axis is duplicated"));
            }
            if input.dimensions()[axis] != 1 {
                return Err(RuntimeError::new("Squeeze axis must have dimension one"));
            }
            removed[axis] = true;
        }
    }
    let mut dimensions = [0; MAX_RANK];
    let mut rank = 0;
    for (axis, &dimension) in input.dimensions().iter().enumerate() {
        if !removed[axis] {
            dimensions[rank] = dimension;
            rank += 1;
        }
    }
    Ok(Shape::new(&dimensions[..rank])?)
}
pub(super) fn unsqueeze(input: Shape, axes: &[i64]) -> RuntimeResult<Shape> {
    let rank = input
        .rank()
        .checked_add(axes.len())
        .filter(|&rank| rank <= MAX_RANK)
        .ok_or_else(|| RuntimeError::new("Unsqueeze output exceeds rank eight"))?;
    let mut inserted = [false; MAX_RANK];
    for &value in axes {
        let axis = axis(value, rank)?;
        if inserted[axis] {
            return Err(RuntimeError::new("Unsqueeze axis is duplicated"));
        }
        inserted[axis] = true;
    }
    let mut dimensions = [1; MAX_RANK];
    let mut source = 0;
    for axis in 0..rank {
        if !inserted[axis] {
            dimensions[axis] = input.dimensions()[source];
            source += 1;
        }
    }
    Ok(Shape::new(&dimensions[..rank])?)
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
