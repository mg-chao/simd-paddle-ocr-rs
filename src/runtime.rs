//! Safe, serial, unfused scalar execution of the shipped CLS operator subset.
//!
//! Models and compiled instructions are immutable and shareable. Each session
//! binds concrete shapes and owns reusable activation buffers. This deliberately
//! does not claim general ONNX execution support.

mod kernels;
mod shapes;

use crate::{
    model::{EffectiveAttributes, Model, Node, TensorId},
    tensor::{DType, Dim, Shape, ShapeSpec, TensorData, TensorError, TensorView},
};
use kernels::{BoundKind, BoundStep};
use shapes::ShapeProgram;
use std::{fmt, sync::Arc};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeError(String);
impl RuntimeError {
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}
impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for RuntimeError {}
impl From<TensorError> for RuntimeError {
    fn from(value: TensorError) -> Self {
        Self::new(value.to_string())
    }
}
pub type RuntimeResult<T> = Result<T, RuntimeError>;

#[derive(Clone, Copy, Debug)]
pub struct ExecutionLimits {
    /// Sum of reusable activation slot lengths, excluding shared model weights.
    pub max_activation_bytes: usize,
}
impl Default for ExecutionLimits {
    fn default() -> Self {
        Self {
            max_activation_bytes: 1 << 30,
        }
    }
}

/// A synchronous observation before subsequent nodes can reuse this storage.
/// The execution index differs from the original ONNX lineage in `key`.
pub struct NodeTrace<'a> {
    pub index: usize,
    pub key: &'a str,
    pub name: &'a str,
    pub op_type: &'a str,
    pub output_id: TensorId,
    pub shape: Shape,
    pub layout: &'static str,
    pub values: &'a [f32],
}

#[derive(Clone)]
struct Instruction {
    a: TensorId,
    b: Option<TensorId>,
    bias: Option<TensorId>,
    output: TensorId,
    kind: Op,
}
#[derive(Clone)]
enum Op {
    Add,
    Mul,
    Relu,
    HardSigmoid {
        alpha: f32,
        beta: f32,
    },
    Conv(crate::model::ConvAttributes),
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

/// Immutable instructions over one owned model; constants are never copied.
pub struct CompiledModel {
    model: Arc<Model>,
    instructions: Vec<Instruction>,
    shape_program: ShapeProgram,
    last_use: Vec<isize>,
    input: TensorId,
    output: TensorId,
}
impl CompiledModel {
    pub fn new(model: Arc<Model>) -> RuntimeResult<Self> {
        let graph = model.graph();
        if graph.inputs.len() != 1 || graph.outputs.len() != 1 {
            return Err(RuntimeError::new(
                "scalar runtime requires one graph input and output",
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
                "constant graph inputs/outputs are unsupported by scalar runtime",
            ));
        }
        for tensor in &graph.tensors {
            if let Some(TensorData::F32(values)) = &tensor.constant {
                positive_shape(values.shape()).map_err(|error| {
                    RuntimeError::new(format!("constant {}: {error}", tensor.name))
                })?;
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
        Ok(Self {
            model,
            instructions,
            shape_program,
            last_use,
            input,
            output,
        })
    }
    pub fn model(&self) -> &Model {
        &self.model
    }
    pub fn node_count(&self) -> usize {
        self.instructions.len()
    }
}

fn node_error(index: usize, node: &Node, message: impl fmt::Display) -> RuntimeError {
    RuntimeError::new(format!(
        "node {index} '{}' ({}, {}): {message}",
        node.name, node.op_type, node.key
    ))
}
fn integer_values(data: &TensorData) -> RuntimeResult<Vec<i64>> {
    match data {
        TensorData::I64(t) => Ok(t.data().to_vec()),
        TensorData::I32(t) => Ok(t.data().iter().map(|&v| i64::from(v)).collect()),
        _ => Err(RuntimeError::new(
            "expected constant I32/I64 reshape target",
        )),
    }
}
fn compile_node(model: &Model, index: usize, node: &Node) -> RuntimeResult<Instruction> {
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
    let kind = match (&*node.op_type, &node.attributes) {
        ("Add", EffectiveAttributes::None) | ("Mul", EffectiveAttributes::None) => {
            require_inputs(2)?;
            if graph.tensors[b.unwrap().0].dtype != DType::F32 {
                return Err(fail("second data input must be F32"));
            }
            if node.op_type == "Add" {
                Op::Add
            } else {
                Op::Mul
            }
        }
        ("Relu", EffectiveAttributes::None) => {
            require_inputs(1)?;
            Op::Relu
        }
        ("HardSigmoid", EffectiveAttributes::HardSigmoid { alpha, beta }) => {
            require_inputs(1)?;
            if !alpha.is_finite() || !beta.is_finite() {
                return Err(fail("nonfinite HardSigmoid parameter"));
            }
            Op::HardSigmoid {
                alpha: *alpha,
                beta: *beta,
            }
        }
        ("Conv", EffectiveAttributes::Conv(attributes)) => {
            if !(2..=3).contains(&node.inputs.len()) || b.is_none() {
                return Err(fail("Conv requires data, weights and optional bias"));
            }
            constant_f32(b)?;
            if bias.is_some() {
                constant_f32(bias)?;
            }
            Op::Conv(attributes.clone())
        }
        (
            "ReduceMean",
            EffectiveAttributes::ReduceMean {
                axes,
                keepdims,
                noop_with_empty_axes,
            },
        ) => {
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
        ("Reshape", EffectiveAttributes::Reshape { allowzero }) => {
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
        ("MatMul", EffectiveAttributes::None) => {
            require_inputs(2)?;
            constant_f32(b)?;
            Op::MatMul
        }
        ("Softmax", EffectiveAttributes::Axis(axis)) => {
            require_inputs(1)?;
            Op::Softmax { axis: *axis }
        }
        _ => {
            return Err(fail(
                "unsupported operator/attributes in scalar CLS runtime",
            ));
        }
    };
    Ok(Instruction {
        a,
        b,
        bias,
        output,
        kind,
    })
}

struct ShapePlan {
    input_shape: Shape,
    shapes: Vec<Shape>,
    steps: Vec<BoundStep>,
    slots: Vec<usize>,
    lengths: Vec<usize>,
    activation_bytes: usize,
    naive_activation_bytes: usize,
}

/// One mutable request. Borrowed output remains valid until the next mutable
/// operation; Rust prevents running or reshaping while that borrow is held.
pub struct Session {
    compiled: Arc<CompiledModel>,
    limits: ExecutionLimits,
    plan: ShapePlan,
    buffers: Vec<Vec<f32>>,
}
impl fmt::Debug for Session {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Session")
            .field("input_shape", &self.input_shape())
            .field("output_shape", &self.output_shape())
            .field("activation_bytes", &self.activation_bytes())
            .finish_non_exhaustive()
    }
}
impl Session {
    pub fn new(
        compiled: Arc<CompiledModel>,
        input_shape: &[usize],
        limits: ExecutionLimits,
    ) -> RuntimeResult<Self> {
        let plan = bind(&compiled, input_shape, limits)?;
        let buffers = allocate(&plan.lengths)?;
        Ok(Self {
            compiled,
            limits,
            plan,
            buffers,
        })
    }
    /// Invalid shapes or failed reservations leave the previous binding intact.
    pub fn reshape(&mut self, input_shape: &[usize]) -> RuntimeResult<()> {
        if input_shape == self.plan.input_shape.dimensions() {
            return Ok(());
        }
        let plan = bind(&self.compiled, input_shape, self.limits)?;
        let buffers = allocate(&plan.lengths)?;
        self.plan = plan;
        self.buffers = buffers;
        Ok(())
    }
    pub fn input_shape(&self) -> Shape {
        self.plan.input_shape
    }
    pub fn output_shape(&self) -> Shape {
        self.plan.shapes[self.compiled.output.0]
    }
    pub fn activation_bytes(&self) -> usize {
        self.plan.activation_bytes
    }
    pub fn naive_activation_bytes(&self) -> usize {
        self.plan.naive_activation_bytes
    }
    pub fn run(&mut self, input: &[f32]) -> RuntimeResult<TensorView<'_, f32>> {
        fn ignore(_: NodeTrace<'_>) -> RuntimeResult<()> {
            Ok(())
        }
        self.execute::<false, _>(input, &mut ignore)?;
        self.output()
    }
    pub fn run_with_trace<F>(
        &mut self,
        input: &[f32],
        mut observer: F,
    ) -> RuntimeResult<TensorView<'_, f32>>
    where
        F: for<'a> FnMut(NodeTrace<'a>) -> RuntimeResult<()>,
    {
        self.execute::<true, _>(input, &mut observer)?;
        self.output()
    }
    fn output(&self) -> RuntimeResult<TensorView<'_, f32>> {
        let id = self.compiled.output;
        TensorView::from_slice(self.plan.shapes[id.0], self.data(id)).map_err(Into::into)
    }
    fn data(&self, id: TensorId) -> &[f32] {
        match &self.compiled.model.graph().tensors[id.0].constant {
            Some(TensorData::F32(values)) => values.data(),
            _ => &self.buffers[self.plan.slots[id.0]][..self.plan.shapes[id.0].element_count()],
        }
    }
    fn execute<const TRACE: bool, F>(
        &mut self,
        input: &[f32],
        observer: &mut F,
    ) -> RuntimeResult<()>
    where
        F: for<'a> FnMut(NodeTrace<'a>) -> RuntimeResult<()>,
    {
        if input.len() != self.plan.input_shape.element_count() {
            return Err(RuntimeError::new(
                "input element count does not match bound shape",
            ));
        }
        if let Some(index) = input.iter().position(|v| !v.is_finite()) {
            return Err(RuntimeError::new(format!(
                "nonfinite input F32 at element {index}"
            )));
        }
        let input_slot = self.plan.slots[self.compiled.input.0];
        self.buffers[input_slot][..input.len()].copy_from_slice(input);
        for index in 0..self.plan.steps.len() {
            let step = &self.plan.steps[index];
            let slot = self.plan.slots[step.output.0];
            // Strict lifetime intervals ensure no current input occupies this
            // slot. Taking the Vec gives a separate safe mutable destination.
            let mut output = std::mem::take(&mut self.buffers[slot]);
            let count = self.plan.shapes[step.output.0].element_count();
            kernels::execute(
                step,
                self.data(step.a),
                step.b.map(|id| self.data(id)),
                step.bias.map(|id| self.data(id)),
                &mut output[..count],
            );
            self.buffers[slot] = output;
            if TRACE {
                let node = &self.compiled.model.graph().nodes[index];
                let shape = self.plan.shapes[step.output.0];
                observer(NodeTrace {
                    index,
                    key: &node.key,
                    name: &node.name,
                    op_type: &node.op_type,
                    output_id: step.output,
                    shape,
                    layout: if shape.rank() == 4 {
                        "nchw"
                    } else {
                        "row_major"
                    },
                    values: self.data(step.output),
                })?;
            }
        }
        if let Some(index) = self
            .data(self.compiled.output)
            .iter()
            .position(|v| !v.is_finite())
        {
            return Err(RuntimeError::new(format!(
                "nonfinite output F32 at element {index}"
            )));
        }
        Ok(())
    }
}

fn allocate(lengths: &[usize]) -> RuntimeResult<Vec<Vec<f32>>> {
    let mut buffers = Vec::new();
    buffers
        .try_reserve_exact(lengths.len())
        .map_err(|e| RuntimeError::new(format!("activation table allocation failed: {e}")))?;
    for &length in lengths {
        let mut buffer = Vec::new();
        buffer
            .try_reserve_exact(length)
            .map_err(|e| RuntimeError::new(format!("activation allocation failed: {e}")))?;
        buffer.resize(length, 0.0);
        buffers.push(buffer);
    }
    Ok(buffers)
}

fn bind(
    compiled: &CompiledModel,
    dimensions: &[usize],
    limits: ExecutionLimits,
) -> RuntimeResult<ShapePlan> {
    if limits.max_activation_bytes == 0 {
        return Err(RuntimeError::new("max_activation_bytes must be positive"));
    }
    let input_shape = Shape::new(dimensions)?;
    positive_shape(input_shape)?;
    let graph = compiled.model.graph();
    if let ShapeSpec::Ranked(template) = &graph.tensors[compiled.input.0].shape {
        if template.len() != dimensions.len() {
            return Err(RuntimeError::new("input rank does not match model"));
        }
        for (dim, &value) in template.iter().zip(dimensions) {
            if matches!(dim, Dim::Known(expected) if *expected != value) {
                return Err(RuntimeError::new("input dimension does not match model"));
            }
        }
    }
    let mut resolved = vec![None; graph.tensors.len()];
    for (id, tensor) in graph.tensors.iter().enumerate() {
        if let Some(data) = &tensor.constant {
            resolved[id] = Some(data.shape());
        }
    }
    resolved[compiled.input.0] = Some(input_shape);
    let mut cache = compiled.shape_program.cache();
    let mut steps = Vec::with_capacity(compiled.instructions.len());
    for (index, instruction) in compiled.instructions.iter().enumerate() {
        let node = &graph.nodes[index];
        let a = resolved[instruction.a.0]
            .ok_or_else(|| node_error(index, node, "unresolved input shape"))?;
        let b = instruction.b.and_then(|id| resolved[id.0]);
        let bias = instruction.bias.and_then(|id| resolved[id.0]);
        let result = (|| -> RuntimeResult<(Shape, BoundKind)> {
            match &instruction.kind {
                Op::Add | Op::Mul => {
                    let b = b.ok_or_else(|| RuntimeError::new("unresolved binary input shape"))?;
                    let output = a.broadcast(b)?;
                    Ok((
                        output,
                        BoundKind::Binary(
                            kernels::Broadcast::new(a, b, output),
                            matches!(instruction.kind, Op::Mul),
                        ),
                    ))
                }
                Op::Relu => Ok((a, BoundKind::Relu)),
                Op::HardSigmoid { alpha, beta } => Ok((a, BoundKind::HardSigmoid(*alpha, *beta))),
                Op::Conv(attrs) => kernels::bind_conv(a, b.unwrap(), bias, attrs),
                Op::ReduceMean {
                    axes,
                    keepdims,
                    noop,
                } => kernels::bind_reduce(a, axes, *keepdims, *noop),
                Op::Reshape { allowzero, target } => {
                    let evaluated;
                    let values = if let Some(values) = target {
                        values.as_slice()
                    } else {
                        evaluated = compiled
                            .shape_program
                            .evaluate(index, &resolved, &mut cache)?;
                        evaluated.values()
                    };
                    Ok((shapes::reshape(a, values, *allowzero)?, BoundKind::Copy))
                }
                Op::MatMul => {
                    let b = b.unwrap();
                    if a.rank() != 2 || b.rank() != 2 {
                        return Err(RuntimeError::new("CLS MatMul requires rank-two inputs"));
                    }
                    let ad = a.dimensions();
                    let bd = b.dimensions();
                    if ad[1] != bd[0] {
                        return Err(RuntimeError::new("MatMul inner dimensions differ"));
                    }
                    Ok((
                        Shape::new(&[ad[0], bd[1]])?,
                        BoundKind::MatMul {
                            rows: ad[0],
                            inner: ad[1],
                            columns: bd[1],
                        },
                    ))
                }
                Op::Softmax { axis } => {
                    let axis = shapes::axis(*axis, a.rank())?;
                    if a.rank() != 2 || axis != 1 {
                        return Err(RuntimeError::new(
                            "CLS Softmax requires rank-two, last-axis input",
                        ));
                    }
                    Ok((
                        a,
                        BoundKind::Softmax {
                            rows: a.dimensions()[0],
                            columns: a.dimensions()[1],
                        },
                    ))
                }
            }
        })()
        .map_err(|e| node_error(index, node, e))?;
        positive_shape(result.0).map_err(|e| node_error(index, node, e))?;
        result.0.byte_count(DType::F32)?;
        resolved[instruction.output.0] = Some(result.0);
        steps.push(BoundStep {
            a: instruction.a,
            b: instruction
                .b
                .filter(|id| graph.tensors[id.0].dtype == DType::F32),
            bias: instruction.bias,
            output: instruction.output,
            kind: result.1,
        });
    }
    let shapes = resolved
        .into_iter()
        .enumerate()
        .map(|(index, shape)| {
            shape.ok_or_else(|| RuntimeError::new(format!("unresolved tensor shape at {index}")))
        })
        .collect::<RuntimeResult<Vec<_>>>()?;
    let (slots, lengths, activation_bytes, naive_activation_bytes) =
        plan_slots(compiled, &shapes, limits)?;
    Ok(ShapePlan {
        input_shape,
        shapes,
        steps,
        slots,
        lengths,
        activation_bytes,
        naive_activation_bytes,
    })
}
fn positive_shape(shape: Shape) -> RuntimeResult<()> {
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
fn plan_slots(
    compiled: &CompiledModel,
    shapes: &[Shape],
    limits: ExecutionLimits,
) -> RuntimeResult<(Vec<usize>, Vec<usize>, usize, usize)> {
    let graph = compiled.model.graph();
    let mut starts = vec![-1; graph.tensors.len()];
    for (index, instruction) in compiled.instructions.iter().enumerate() {
        starts[instruction.output.0] = index as isize;
    }
    let mut lifetimes = Vec::new();
    let mut naive = 0usize;
    for (id, tensor) in graph.tensors.iter().enumerate() {
        if tensor.constant.is_some() {
            continue;
        }
        naive = naive
            .checked_add(shapes[id].byte_count(DType::F32)?)
            .ok_or_else(|| RuntimeError::new("activation byte count overflow"))?;
        lifetimes.push((
            starts[id],
            compiled.last_use[id].max(starts[id]),
            id,
            shapes[id].element_count(),
        ));
    }
    lifetimes.sort_unstable_by_key(|&(start, _, id, _)| (start, id));
    let mut slots = vec![usize::MAX; graph.tensors.len()];
    let mut lengths: Vec<usize> = Vec::new();
    let mut ends: Vec<isize> = Vec::new();
    for (start, end, id, length) in lifetimes {
        let fitting = lengths
            .iter()
            .enumerate()
            .filter(|&(i, &len)| ends[i] < start && len >= length)
            .min_by_key(|&(i, &len)| (len, i))
            .map(|(i, _)| i);
        let reusable = fitting.or_else(|| {
            lengths
                .iter()
                .enumerate()
                .filter(|&(i, _)| ends[i] < start)
                .max_by_key(|&(i, &len)| (len, std::cmp::Reverse(i)))
                .map(|(i, _)| i)
        });
        let slot = if let Some(slot) = reusable {
            lengths[slot] = lengths[slot].max(length);
            ends[slot] = end;
            slot
        } else {
            lengths.push(length);
            ends.push(end);
            lengths.len() - 1
        };
        slots[id] = slot;
    }
    let bytes = lengths
        .iter()
        .try_fold(0usize, |sum, &len| {
            len.checked_mul(4).and_then(|bytes| sum.checked_add(bytes))
        })
        .ok_or_else(|| RuntimeError::new("activation plan byte count overflow"))?;
    if bytes > limits.max_activation_bytes {
        return Err(RuntimeError::new(format!(
            "activation plan needs {bytes} bytes, limit is {}",
            limits.max_activation_bytes
        )));
    }
    Ok((slots, lengths, bytes, naive))
}
