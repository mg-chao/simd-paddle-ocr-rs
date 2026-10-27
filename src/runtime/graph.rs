//! Backend-neutral graph validation, shape binding, and liveness planning.
use super::{
    ExecutionLimits, RuntimeError, RuntimeResult,
    bound::{self, BoundKind, BoundStep},
    shapes::{self, ShapeProgram},
};
use crate::{
    model::{Model, Node, Operator, TensorId},
    tensor::{DType, Dim, Shape, ShapeSpec, TensorData},
};
use std::sync::Arc;

#[derive(Clone)]
pub(super) struct Instruction {
    /// Validated F32 operands in semantic input order. Integer reshape operands
    /// belong to the compiled shape program rather than kernel execution.
    pub(super) inputs: Box<[TensorId]>,
    pub(super) output: TensorId,
    pub(super) kind: Op,
}
#[derive(Clone)]
pub(super) enum Op {
    Add,
    Mul,
    Div,
    Sub,
    Square,
    Sqrt,
    Relu,
    Erf,
    Sigmoid,
    HardSigmoid {
        alpha: f32,
        beta: f32,
    },
    Conv(crate::model::ConvAttributes),
    ConvTranspose,
    MaxPool,
    AveragePool,
    BatchNormalization {
        epsilon: f32,
    },
    Squeeze {
        axes: Vec<i64>,
    },
    Unsqueeze {
        axes: Vec<i64>,
    },
    Transpose {
        permutation: Vec<i64>,
    },
    Slice(shapes::SliceParameters),
    Concat {
        axis: i64,
    },
    Resize {
        factor: usize,
    },
    GlobalAveragePool,
    ReduceMean {
        axes: Vec<i64>,
        keepdims: bool,
        noop: bool,
    },
    Reshape {
        allowzero: bool,
        target: Option<Vec<i64>>,
    },
    MatMul,
    Softmax {
        axis: i64,
    },
}

/// Shape-independent dataflow shared by every session and shape binding.
/// Consumers are stored in source order, including repeated operands such as
/// Add(x, x), so prefix queries retain the same exclusivity rules as a scan.
pub(super) struct GraphUses {
    pub(super) offsets: Vec<usize>,
    pub(super) consumers: Vec<usize>,
    pub(super) producers: Vec<usize>,
}
impl GraphUses {
    pub(super) fn new(instructions: &[Instruction], tensors: usize) -> Self {
        let mut offsets = vec![0; tensors + 1];
        for instruction in instructions {
            for input in &instruction.inputs {
                offsets[input.0 + 1] += 1;
            }
        }
        for index in 1..offsets.len() {
            offsets[index] += offsets[index - 1];
        }
        let mut consumers = vec![0; offsets[tensors]];
        let mut cursor = offsets[..tensors].to_vec();
        let mut producers = vec![usize::MAX; tensors];
        for (index, instruction) in instructions.iter().enumerate() {
            producers[instruction.output.0] = index;
            for input in &instruction.inputs {
                consumers[cursor[input.0]] = index;
                cursor[input.0] += 1;
            }
        }
        Self {
            offsets,
            consumers,
            producers,
        }
    }
    pub(super) fn consumers(&self, id: TensorId) -> &[usize] {
        &self.consumers[self.offsets[id.0]..self.offsets[id.0 + 1]]
    }
    pub(super) fn producer(&self, id: TensorId) -> Option<usize> {
        let index = self.producers[id.0];
        (index != usize::MAX).then_some(index)
    }
    pub(super) fn unique_consumer(&self, id: TensorId, count: usize) -> Option<usize> {
        let (first, rest) = self.consumers(id).split_first()?;
        if *first >= count || rest.first().is_some_and(|&next| next < count) {
            None
        } else {
            Some(*first)
        }
    }
    pub(super) fn used_after(&self, id: TensorId, end: usize, count: usize) -> bool {
        let uses = self.consumers(id);
        let after = uses.partition_point(|&index| index <= end);
        uses.get(after).is_some_and(|&index| index < count)
    }
}

/// Backend-independent validation, dataflow and shape instructions.
pub struct LogicalGraph {
    pub(super) model: Arc<Model>,
    pub(super) instructions: Vec<Instruction>,
    pub(super) shape_program: ShapeProgram,
    pub(super) last_use: Vec<isize>,
    pub(super) graph_uses: GraphUses,
    pub(super) input: TensorId,
    pub(super) output: TensorId,
}
pub(super) type Program = LogicalGraph;
impl LogicalGraph {
    pub fn model(&self) -> &Model {
        &self.model
    }
    pub fn node_count(&self) -> usize {
        self.instructions.len()
    }
    pub fn compile(model: Arc<Model>) -> RuntimeResult<Self> {
        let graph = model.graph();
        if graph.inputs.len() != 1 || graph.outputs.len() != 1 {
            return Err(RuntimeError::new(
                "runtime requires one graph input and output",
            ));
        }
        let input = graph.inputs[0];
        let output = graph.outputs[0];
        if graph.tensors[input.0].dtype != DType::F32 || graph.tensors[output.0].dtype != DType::F32
        {
            return Err(RuntimeError::new("graph input and output must be F32"));
        }
        if graph.tensors[input.0].constant.is_some() || graph.tensors[output.0].constant.is_some() {
            return Err(RuntimeError::new(
                "constant graph inputs/outputs are unsupported by runtime",
            ));
        }
        for tensor in &graph.tensors {
            if let Some(TensorData::F32(values)) = &tensor.constant {
                positive_shape(values.shape())
                    .map_err(|error| error.context(format!("constant {}", tensor.name)))?;
                if values.data().iter().any(|v| !v.is_finite()) {
                    return Err(RuntimeError::new(format!(
                        "constant {} contains nonfinite F32",
                        tensor.name
                    )));
                }
            }
        }
        let mut instructions = Vec::with_capacity(graph.nodes.len());
        let mut last_use = vec![-1; graph.tensors.len()];
        let mut defined: Vec<_> = graph
            .tensors
            .iter()
            .map(|tensor| tensor.constant.is_some())
            .collect();
        defined[input.0] = true;
        for (index, node) in graph.nodes.iter().enumerate() {
            let instruction = compile_node(&model, index, node)?;
            if defined[instruction.output.0] {
                return Err(node_error(
                    index,
                    node,
                    "output overwrites an already defined tensor",
                ));
            }
            for id in node.inputs.iter().flatten() {
                if !defined[id.0] {
                    return Err(node_error(index, node, "input tensor is not yet defined"));
                }
                last_use[id.0] = index as isize;
            }
            defined[instruction.output.0] = true;
            instructions.push(instruction);
        }
        last_use[output.0] = graph.nodes.len() as isize;
        let shape_program = ShapeProgram::compile(&model)?;
        let graph_uses = GraphUses::new(&instructions, graph.tensors.len());
        Ok(Self {
            model,
            instructions,
            shape_program,
            last_use,
            graph_uses,
            input,
            output,
        })
    }
}

pub(super) fn node_error(
    index: usize,
    node: &Node,
    message: impl Into<RuntimeError>,
) -> RuntimeError {
    message.into().context(format!(
        "node {index} '{}' ({}, {})",
        node.name,
        node.op_type(),
        node.key
    ))
}
pub(super) fn integer_values(data: &TensorData) -> RuntimeResult<Vec<i64>> {
    if data.shape().element_count() > crate::tensor::MAX_RANK {
        return Err(RuntimeError::new(
            "integer parameter exceeds rank-eight vector limit",
        ));
    }
    match data {
        TensorData::I64(t) => Ok(t.data().to_vec()),
        TensorData::I32(t) => Ok(t.data().iter().map(|&v| i64::from(v)).collect()),
        _ => Err(RuntimeError::new(
            "expected constant I32/I64 reshape target",
        )),
    }
}
pub(super) fn compile_node(model: &Model, index: usize, node: &Node) -> RuntimeResult<Instruction> {
    let graph = model.graph();
    let fail = |message: &str| node_error(index, node, message);
    if !matches!(node.domain.as_str(), "" | "ai.onnx") {
        return Err(fail("unsupported operator domain"));
    }
    if node.outputs.len() != 1 || node.outputs[0].is_none() {
        return Err(fail("requires exactly one output"));
    }
    let output = node.outputs[0].unwrap();
    if graph.tensors[output.0].dtype != DType::F32 || graph.tensors[output.0].constant.is_some() {
        return Err(fail("output must be a nonconstant F32 tensor"));
    }
    let a = node
        .inputs
        .first()
        .copied()
        .flatten()
        .ok_or_else(|| fail("missing first input"))?;
    if graph.tensors[a.0].dtype != DType::F32 {
        return Err(fail("data input must be F32"));
    }
    let b = node.inputs.get(1).copied().flatten();
    let bias = node.inputs.get(2).copied().flatten();
    let require_inputs = |count: usize| -> RuntimeResult<()> {
        if node.inputs.len() != count || node.inputs.iter().any(Option::is_none) {
            Err(fail("invalid required input slots"))
        } else {
            Ok(())
        }
    };
    let constant_f32 = |id: Option<TensorId>| -> RuntimeResult<()> {
        if id.is_none_or(|id| !matches!(graph.tensors[id.0].constant, Some(TensorData::F32(_)))) {
            Err(fail("requires constant F32 weights/bias"))
        } else {
            Ok(())
        }
    };
    let kind = match &node.operator {
        Operator::Add | Operator::Mul | Operator::Div | Operator::Sub => {
            require_inputs(2)?;
            if graph.tensors[b.unwrap().0].dtype != DType::F32 {
                return Err(fail("second data input must be F32"));
            }
            match node.operator {
                Operator::Add => Op::Add,
                Operator::Mul => Op::Mul,
                Operator::Div => Op::Div,
                _ => Op::Sub,
            }
        }
        Operator::Pow => {
            require_inputs(2)?;
            match &graph.tensors[b.unwrap().0].constant {
                Some(TensorData::F32(t)) if t.data() == [2.0] => Op::Square,
                _ => return Err(fail("Pow supports constant scalar exponent two only")),
            }
        }
        Operator::Sqrt => {
            require_inputs(1)?;
            Op::Sqrt
        }
        Operator::Relu => {
            require_inputs(1)?;
            Op::Relu
        }
        Operator::Erf | Operator::Sigmoid => {
            require_inputs(1)?;
            if matches!(node.operator, Operator::Erf) {
                Op::Erf
            } else {
                Op::Sigmoid
            }
        }
        Operator::HardSigmoid { alpha, beta } => {
            require_inputs(1)?;
            if !alpha.is_finite() || !beta.is_finite() {
                return Err(fail("nonfinite HardSigmoid parameter"));
            }
            Op::HardSigmoid {
                alpha: *alpha,
                beta: *beta,
            }
        }
        Operator::Conv(attributes) => {
            if !(2..=3).contains(&node.inputs.len()) || b.is_none() {
                return Err(fail("Conv requires data, weights and optional bias"));
            }
            constant_f32(b)?;
            if bias.is_some() {
                constant_f32(bias)?;
            }
            Op::Conv(attributes.clone())
        }
        Operator::ConvTranspose(attributes) => {
            if !(2..=3).contains(&node.inputs.len()) || b.is_none() {
                return Err(fail(
                    "ConvTranspose requires data, weights and optional bias",
                ));
            }
            constant_f32(b)?;
            if bias.is_some() {
                constant_f32(bias)?;
            }
            if attributes.kernel_shape != [2, 2]
                || attributes.strides != [2, 2]
                || attributes.dilations != [1, 1]
                || attributes.pads != [0; 4]
                || attributes.group != 1
            {
                return Err(fail(
                    "ConvTranspose requires 2x2 stride-two, group-one, unpadded parameters",
                ));
            }
            Op::ConvTranspose
        }
        Operator::MaxPool(attributes) => {
            require_inputs(1)?;
            if attributes.kernel_shape != [2, 2]
                || attributes.strides != [1, 1]
                || attributes.pads != [0, 0, 1, 1]
                || attributes.ceil_mode != 0
                || attributes.count_include_pad != 0
            {
                return Err(fail(
                    "MaxPool requires 2x2 stride-one with trailing padding",
                ));
            }
            Op::MaxPool
        }
        Operator::AveragePool(attributes) => {
            require_inputs(1)?;
            if attributes.kernel_shape != [3, 2]
                || attributes.strides != [3, 2]
                || attributes.pads != [0; 4]
                || attributes.ceil_mode != 0
                || attributes.count_include_pad != 0
            {
                return Err(fail(
                    "AveragePool requires unpadded 3x2 kernel/stride and zero flags",
                ));
            }
            Op::AveragePool
        }
        Operator::BatchNormalization {
            epsilon,
            momentum,
            training_mode,
        } => {
            require_inputs(5)?;
            if !epsilon.is_finite()
                || *epsilon < 0.0
                || !momentum.is_finite()
                || *training_mode != 0
            {
                return Err(fail(
                    "BatchNormalization requires finite nonnegative epsilon and inference mode",
                ));
            }
            for &id in &node.inputs[1..] {
                constant_f32(id)?;
            }
            Op::BatchNormalization { epsilon: *epsilon }
        }
        Operator::Squeeze(axes) => {
            require_inputs(1)?;
            Op::Squeeze { axes: axes.clone() }
        }
        Operator::Unsqueeze(axes) => {
            require_inputs(1)?;
            if axes.is_empty() {
                return Err(fail("Unsqueeze requires nonempty axes"));
            }
            Op::Unsqueeze { axes: axes.clone() }
        }
        Operator::Transpose(permutation) => {
            require_inputs(1)?;
            if permutation.is_empty() {
                return Err(fail("Transpose requires an explicit permutation"));
            }
            Op::Transpose {
                permutation: permutation.clone(),
            }
        }
        Operator::Concat(axis) => {
            if node.inputs.iter().any(Option::is_none)
                || node
                    .inputs
                    .iter()
                    .flatten()
                    .any(|id| graph.tensors[id.0].dtype != DType::F32)
            {
                return Err(fail("Concat requires nonempty F32 input slots"));
            }
            Op::Concat { axis: *axis }
        }
        Operator::Resize(attributes) => {
            require_inputs(1)?;
            if attributes.mode != b"nearest"
                || attributes.coordinate_transformation_mode != b"asymmetric"
                || attributes.nearest_mode != b"floor"
                || attributes.scales.len() != 4
                || attributes.scales[..2] != [1.0, 1.0]
                || attributes.scales[2] != attributes.scales[3]
            {
                return Err(node_error(
                    index,
                    node,
                    RuntimeError::with_kind(
                        crate::error::ErrorKind::Unsupported,
                        "Resize requires nearest/asymmetric/floor spatial-only equal scales",
                    ),
                ));
            }
            let factor = match attributes.scales[2] {
                2.0 => 2,
                4.0 => 4,
                8.0 => 8,
                _ => return Err(fail("Resize requires spatial scale two, four or eight")),
            };
            Op::Resize { factor }
        }
        Operator::GlobalAveragePool => {
            require_inputs(1)?;
            Op::GlobalAveragePool
        }
        Operator::ReduceMean {
            axes,
            keepdims,
            noop_with_empty_axes,
        } => {
            require_inputs(1)?;
            if !matches!(keepdims, 0 | 1) || !matches!(noop_with_empty_axes, 0 | 1) {
                return Err(fail("ReduceMean flags must be 0 or 1"));
            }
            Op::ReduceMean {
                axes: axes.clone(),
                keepdims: *keepdims == 1,
                noop: *noop_with_empty_axes == 1,
            }
        }
        Operator::Reshape { allowzero } => {
            if !matches!(allowzero, 0 | 1) {
                return Err(fail("Reshape allowzero must be 0 or 1"));
            }
            let target = match node.inputs.as_slice() {
                [Some(_)] => None,
                [Some(_), Some(id)] => {
                    let data = graph.tensors[id.0].constant.as_ref().ok_or_else(|| {
                        fail("reshape target must be constant or a retained shape recipe")
                    })?;
                    if data.dimensions().len() != 1 {
                        return Err(fail("reshape target must have rank one"));
                    }
                    Some(integer_values(data).map_err(|e| node_error(index, node, e))?)
                }
                _ => return Err(fail("invalid Reshape input slots")),
            };
            Op::Reshape {
                allowzero: *allowzero == 1,
                target,
            }
        }
        Operator::MatMul => {
            require_inputs(2)?;
            if graph.tensors[b.unwrap().0].dtype != DType::F32 {
                return Err(fail("MatMul second data input must be F32"));
            }
            Op::MatMul
        }
        Operator::Slice => {
            if !(3..=5).contains(&node.inputs.len()) {
                return Err(fail("Slice requires constant input-form parameters"));
            }
            let parameter = |slot: usize| -> RuntimeResult<Option<Vec<i64>>> {
                node.inputs
                    .get(slot)
                    .copied()
                    .flatten()
                    .map(|id| {
                        let data = graph.tensors[id.0].constant.as_ref().ok_or_else(|| {
                            fail("Slice parameters must be constant integer vectors")
                        })?;
                        if data.dimensions().len() != 1 {
                            return Err(fail("Slice parameters must have rank one"));
                        }
                        integer_values(data)
                    })
                    .transpose()
            };
            Op::Slice(shapes::SliceParameters::new(
                parameter(1)?.ok_or_else(|| fail("Slice is missing starts"))?,
                parameter(2)?.ok_or_else(|| fail("Slice is missing ends"))?,
                parameter(3)?,
                parameter(4)?,
            )?)
        }
        Operator::Softmax(axis) => {
            require_inputs(1)?;
            Op::Softmax { axis: *axis }
        }
    };
    Ok(Instruction {
        inputs: node
            .inputs
            .iter()
            .flatten()
            .copied()
            .filter(|id| graph.tensors[id.0].dtype == DType::F32)
            .collect(),
        output,
        kind,
    })
}

#[derive(Clone)]
pub(super) struct ShapePlan {
    pub(super) input_shape: Shape,
    pub(super) shapes: Vec<Shape>,
    pub(super) steps: Vec<BoundStep>,
    pub(super) slots: Vec<usize>,
    pub(super) lengths: Vec<usize>,
    pub(super) activation_bytes: usize,
    pub(super) naive_activation_bytes: usize,
}
impl ShapePlan {
    pub(super) fn empty(tensors: usize, nodes: usize) -> Self {
        let scalar = Shape::new(&[]).expect("scalar shape");
        Self {
            input_shape: scalar,
            shapes: vec![scalar; tensors],
            steps: Vec::with_capacity(nodes),
            slots: vec![usize::MAX; tensors],
            lengths: Vec::with_capacity(tensors),
            activation_bytes: 0,
            naive_activation_bytes: 0,
        }
    }
}
pub(super) struct BindScratch {
    pub(super) resolved: Vec<Option<Shape>>,
    pub(super) shape_cache: Vec<Option<shapes::Integers>>,
    pub(super) starts: Vec<isize>,
    pub(super) last_use: Vec<isize>,
    pub(super) lifetimes: Vec<(isize, isize, usize, usize)>,
    pub(super) ends: Vec<isize>,
}
impl BindScratch {
    pub(super) fn new(program: &Program) -> Self {
        let count = program.model.graph().tensors.len();
        Self {
            resolved: vec![None; count],
            shape_cache: program.shape_program.cache(),
            starts: vec![-2; count],
            last_use: vec![-1; count],
            lifetimes: Vec::with_capacity(count),
            ends: Vec::with_capacity(count),
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn bind_into(
    program: &Program,
    dimensions: &[usize],
    limits: ExecutionLimits,
    node_count: usize,
    selected_output: TensorId,
    plan: &mut ShapePlan,
    scratch: &mut BindScratch,
) -> RuntimeResult<()> {
    if limits.max_activation_bytes == 0 {
        return Err(RuntimeError::new("max_activation_bytes must be positive"));
    }
    let input_shape = Shape::new(dimensions)?;
    positive_shape(input_shape)?;
    let graph = program.model.graph();
    if let ShapeSpec::Ranked(template) = &graph.tensors[program.input.0].shape {
        if template.len() != dimensions.len() {
            return Err(RuntimeError::new("input rank does not match model"));
        }
        for (dim, &value) in template.iter().zip(dimensions) {
            if matches!(dim, Dim::Known(expected) if *expected != value) {
                return Err(RuntimeError::new("input dimension does not match model"));
            }
        }
    }
    scratch.resolved.fill(None);
    scratch.shape_cache.fill(None);
    let resolved = &mut scratch.resolved;
    for (id, tensor) in graph.tensors.iter().enumerate() {
        if let Some(data) = &tensor.constant {
            resolved[id] = Some(data.shape());
        }
    }
    resolved[program.input.0] = Some(input_shape);
    plan.steps.clear();
    for (index, instruction) in program.instructions[..node_count].iter().enumerate() {
        let node = &graph.nodes[index];
        let a = resolved[instruction.inputs[0].0]
            .ok_or_else(|| node_error(index, node, "unresolved input shape"))?;
        let b = instruction.inputs.get(1).and_then(|id| resolved[id.0]);
        let bias = instruction.inputs.get(2).and_then(|id| resolved[id.0]);
        let result = (|| -> RuntimeResult<(Shape, BoundKind)> {
            match &instruction.kind {
                Op::Add | Op::Mul | Op::Div | Op::Sub => {
                    let b = b.ok_or_else(|| RuntimeError::new("unresolved binary input shape"))?;
                    let output = a.broadcast(b)?;
                    Ok((
                        output,
                        BoundKind::Binary(
                            bound::Broadcast::new(a, b, output),
                            match instruction.kind {
                                Op::Add => bound::BinaryOp::Add,
                                Op::Mul => bound::BinaryOp::Mul,
                                Op::Div => bound::BinaryOp::Div,
                                _ => bound::BinaryOp::Sub,
                            },
                        ),
                    ))
                }
                Op::Square => Ok((a.broadcast(b.unwrap())?, BoundKind::Square)),
                Op::Sqrt => Ok((a, BoundKind::Sqrt)),
                Op::Relu => Ok((a, BoundKind::Relu)),
                Op::Erf => Ok((a, BoundKind::Erf)),
                Op::Sigmoid => Ok((a, BoundKind::Sigmoid)),
                Op::HardSigmoid { alpha, beta } => Ok((a, BoundKind::HardSigmoid(*alpha, *beta))),
                Op::Conv(attrs) => bound::bind_conv(a, b.unwrap(), bias, attrs),
                Op::ConvTranspose => bound::bind_conv_transpose(a, b.unwrap(), bias),
                Op::MaxPool => bound::bind_max_pool(a),
                Op::AveragePool => bound::bind_average_pool(a),
                Op::BatchNormalization { epsilon } => {
                    let mut parameters = [Shape::new(&[])?; 4];
                    for (shape, id) in parameters.iter_mut().zip(&instruction.inputs[1..]) {
                        *shape = resolved[id.0].ok_or_else(|| {
                            RuntimeError::new("unresolved BatchNormalization parameter")
                        })?;
                    }
                    bound::bind_batch_normalization(a, &parameters, *epsilon)
                }
                Op::Squeeze { axes } => Ok((shapes::squeeze(a, axes)?, BoundKind::Copy)),
                Op::Unsqueeze { axes } => Ok((shapes::unsqueeze(a, axes)?, BoundKind::Copy)),
                Op::Transpose { permutation } => bound::bind_transpose(a, permutation),
                Op::Slice(parameters) => bound::bind_slice(a, parameters),
                Op::Concat { axis } => bound::bind_concat(&instruction.inputs, resolved, *axis),
                Op::Resize { factor } => bound::bind_resize(a, *factor),
                Op::GlobalAveragePool => {
                    if a.rank() < 3 {
                        return Err(RuntimeError::new(
                            "GlobalAveragePool requires at least one spatial dimension",
                        ));
                    }
                    let axes = (2..a.rank()).map(|axis| axis as i64).collect::<Vec<_>>();
                    bound::bind_reduce(a, &axes, true, false)
                }
                Op::ReduceMean {
                    axes,
                    keepdims,
                    noop,
                } => bound::bind_reduce(a, axes, *keepdims, *noop),
                Op::Reshape { allowzero, target } => {
                    let evaluated;
                    let values = if let Some(values) = target {
                        values.as_slice()
                    } else {
                        evaluated = program.shape_program.evaluate(
                            index,
                            resolved,
                            &mut scratch.shape_cache,
                        )?;
                        evaluated.values()
                    };
                    Ok((shapes::reshape(a, values, *allowzero)?, BoundKind::Copy))
                }
                Op::MatMul => bound::bind_matmul(a, b.unwrap()),
                Op::Softmax { axis } => {
                    let axis = shapes::axis(*axis, a.rank())?;
                    if a.rank() < 2 || axis != a.rank() - 1 {
                        return Err(RuntimeError::new(
                            "Softmax requires rank-at-least-two, last-axis input",
                        ));
                    }
                    Ok((
                        a,
                        BoundKind::Softmax {
                            rows: a.element_count() / a.dimensions()[axis],
                            columns: a.dimensions()[axis],
                        },
                    ))
                }
            }
        })()
        .map_err(|e| node_error(index, node, e))?;
        positive_shape(result.0).map_err(|e| node_error(index, node, e))?;
        result.0.byte_count(DType::F32)?;
        resolved[instruction.output.0] = Some(result.0);
        plan.steps.push(BoundStep {
            output: instruction.output,
            kind: result.1,
        });
    }
    if node_count == program.instructions.len()
        && let Some(index) = resolved.iter().position(Option::is_none)
    {
        return Err(RuntimeError::new(format!(
            "unresolved tensor shape at {index}"
        )));
    }
    for (destination, &shape) in plan.shapes.iter_mut().zip(resolved.iter()) {
        if let Some(shape) = shape {
            *destination = shape;
        }
    }
    plan.input_shape = input_shape;
    plan_slots_into(program, node_count, selected_output, plan, scratch, limits)
}
pub(super) fn positive_shape(shape: Shape) -> RuntimeResult<()> {
    if shape
        .dimensions()
        .iter()
        .any(|&d| d == 0 || i64::try_from(d).is_err())
    {
        return Err(RuntimeError::new(
            "runtime dimensions must be positive and fit I64",
        ));
    }
    shape.byte_count(DType::F32)?;
    Ok(())
}
pub(super) fn plan_slots_into(
    program: &Program,
    node_count: usize,
    selected_output: TensorId,
    plan: &mut ShapePlan,
    scratch: &mut BindScratch,
    limits: ExecutionLimits,
) -> RuntimeResult<()> {
    let graph = program.model.graph();
    scratch.starts.fill(-2);
    scratch.starts[program.input.0] = -1;
    scratch.last_use.copy_from_slice(&program.last_use);
    if node_count != program.instructions.len() {
        scratch.last_use.fill(-1);
        for (index, instruction) in program.instructions[..node_count].iter().enumerate() {
            for id in &instruction.inputs {
                scratch.last_use[id.0] = index as isize;
            }
        }
    }
    scratch.last_use[selected_output.0] = node_count as isize;
    for (index, instruction) in program.instructions[..node_count].iter().enumerate() {
        scratch.starts[instruction.output.0] = index as isize;
    }
    scratch.lifetimes.clear();
    let mut naive = 0usize;
    for (id, tensor) in graph.tensors.iter().enumerate() {
        if tensor.constant.is_some() || scratch.starts[id] == -2 {
            continue;
        }
        naive = naive
            .checked_add(plan.shapes[id].byte_count(DType::F32)?)
            .ok_or_else(|| RuntimeError::new("activation byte count overflow"))?;
        scratch.lifetimes.push((
            scratch.starts[id],
            scratch.last_use[id].max(scratch.starts[id]),
            id,
            plan.shapes[id].element_count(),
        ));
    }
    scratch
        .lifetimes
        .sort_unstable_by_key(|&(start, _, id, _)| (start, id));
    plan.slots.fill(usize::MAX);
    plan.lengths.clear();
    scratch.ends.clear();
    for &(start, end, id, length) in &scratch.lifetimes {
        let fitting = plan
            .lengths
            .iter()
            .enumerate()
            .filter(|&(i, &len)| scratch.ends[i] < start && len >= length)
            .min_by_key(|&(i, &len)| (len, i))
            .map(|(i, _)| i);
        let reusable = fitting.or_else(|| {
            plan.lengths
                .iter()
                .enumerate()
                .filter(|&(i, _)| scratch.ends[i] < start)
                .max_by_key(|&(i, &len)| (len, std::cmp::Reverse(i)))
                .map(|(i, _)| i)
        });
        let slot = if let Some(slot) = reusable {
            plan.lengths[slot] = plan.lengths[slot].max(length);
            scratch.ends[slot] = end;
            slot
        } else {
            plan.lengths.push(length);
            scratch.ends.push(end);
            plan.lengths.len() - 1
        };
        plan.slots[id] = slot;
    }
    let bytes = plan
        .lengths
        .iter()
        .try_fold(0usize, |sum, &len| {
            len.checked_mul(4).and_then(|bytes| sum.checked_add(bytes))
        })
        .ok_or_else(|| RuntimeError::new("activation plan byte count overflow"))?;
    if bytes > limits.max_activation_bytes {
        return Err(RuntimeError::with_kind(
            crate::error::ErrorKind::ResourceLimit,
            format!(
                "activation plan needs {bytes} bytes, limit is {}",
                limits.max_activation_bytes
            ),
        ));
    }
    plan.activation_bytes = bytes;
    plan.naive_activation_bytes = naive;
    Ok(())
}
