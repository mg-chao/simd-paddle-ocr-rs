//! Scalar and dispatched SIMD CPU execution of the shipped CLS, DET and REC networks.
//!
//! Models and compiled instructions are immutable and shareable. Each session
//! binds concrete shapes and owns reusable activation buffers. This deliberately
//! does not claim general ONNX execution support.

mod kernels;
mod nchw;
mod nhwc;
mod optimized;
pub(crate) mod packing;
pub(crate) mod parallel;
mod shapes;
pub(crate) mod simd;
pub use crate::cpu::CpuContext;
use packing::PackedMatrix;
pub use simd::{Backend, KernelPolicy};
use std::collections::{HashMap, VecDeque};

#[derive(Clone, Copy, Debug, serde::Serialize, PartialEq, Eq)]
pub enum LayoutPolicy {
    Auto,
    Nchw,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct CompileOptions {
    pub kernels: KernelPolicy,
    pub layout: LayoutPolicy,
    pub fusion: bool,
}
impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            kernels: KernelPolicy::Auto,
            layout: LayoutPolicy::Auto,
            fusion: true,
        }
    }
}
impl CompileOptions {
    /// Disable execution rewrites. Pair with `OptimizationPolicy::PreserveArithmetic`
    /// at import time to also preserve source BN and branch arithmetic.
    pub fn reference() -> Self {
        Self {
            kernels: KernelPolicy::Scalar,
            layout: LayoutPolicy::Nchw,
            fusion: false,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ExecutionOptions {
    /// Total retained F32 activation capacity for this Session, including lazy
    /// diagnostic execution. Shared weights, plans and scratch are separate.
    pub max_activation_bytes: usize,
    /// Requested budget, validated against the context capacity before capping at 16.
    pub intra_op_threads: usize,
}
impl Default for ExecutionOptions {
    fn default() -> Self {
        Self {
            max_activation_bytes: 1 << 30,
            intra_op_threads: 1,
        }
    }
}

#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct ExecutionStatistics {
    pub selected_kernels: Vec<KernelSelection>,
    /// Shape-selected physical families, separate from logical operator counts.
    pub specialized_kernels: Vec<KernelSpecialization>,
    pub original_weight_bytes: u64,
    pub normalized_nodes: usize,
    pub physical_steps: usize,
    pub fused_nodes: usize,
    /// Physical convolutions applying their complete epilogue before storing.
    pub store_fused_steps: usize,
    pub nhwc_values: usize,
    pub layout_conversions: usize,
    pub packed_weight_bytes: usize,
    /// All formats cached on the shared parsed Model, including other policies.
    pub shared_cached_weight_bytes: usize,
    pub activation_bytes: usize,
    pub retained_activation_bytes: usize,
    pub scratch_bytes: usize,
    pub plan_cache_hits: u64,
    pub plan_cache_misses: u64,
    pub backend: String,
}
#[derive(Clone, Debug, serde::Serialize)]
pub struct KernelSelection {
    pub operator: String,
    pub layout: String,
    pub packed: bool,
    pub steps: usize,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct KernelSpecialization {
    pub operator: String,
    pub family: String,
    pub isa: String,
    pub store_fused: bool,
    pub steps: usize,
}

fn specialization_counts<'a>(
    selections: impl Iterator<Item = (&'a str, &'a str, String, bool)>,
) -> Vec<KernelSpecialization> {
    let mut counts = std::collections::BTreeMap::new();
    for (operator, family, isa, store_fused) in selections {
        *counts
            .entry((operator, family, isa, store_fused))
            .or_insert(0) += 1;
    }
    counts
        .into_iter()
        .map(
            |((operator, family, isa, store_fused), steps)| KernelSpecialization {
                operator: operator.into(),
                family: family.into(),
                isa,
                store_fused,
                steps,
            },
        )
        .collect()
}
fn kernel_selections<'a>(
    kernels: impl Iterator<Item = (&'a str, &'a str, bool)>,
) -> Vec<KernelSelection> {
    let mut counts = std::collections::BTreeMap::new();
    for (operator, layout, packed) in kernels {
        *counts.entry((operator, layout, packed)).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .map(|((operator, layout, packed), steps)| KernelSelection {
            operator: operator.into(),
            layout: layout.into(),
            packed,
            steps,
        })
        .collect()
}

use crate::{
    model::{Model, Node, Operator, TensorId},
    tensor::{DType, Dim, Shape, ShapeSpec, TensorData, TensorError, TensorView},
};
use kernels::{BoundKind, BoundStep};
use shapes::ShapeProgram;
use std::{fmt, sync::Arc};

crate::error::stage_error!(RuntimeError, Execution);

impl From<TensorError> for RuntimeError {
    fn from(value: TensorError) -> Self {
        Self::caused_by(value.kind(), "tensor", value)
    }
}
impl From<crate::image::ImageError> for RuntimeError {
    fn from(value: crate::image::ImageError) -> Self {
        Self::caused_by(value.kind(), "image stage", value)
    }
}
pub type RuntimeResult<T> = Result<T, RuntimeError>;

#[derive(Clone, Copy, Debug)]
pub struct ExecutionLimits {
    /// Bound on total active/retained F32 activation capacity for one Session,
    /// including its diagnostic/prefix children; weights, plans and scratch are separate.
    /// Transactional rebinding can temporarily hold old and new allocations.
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
    /// Diagnostic kernel wall time, including bounded-pool dispatch, excluding
    /// the observer. Diagnostic execution is unfused and is not production timing.
    pub elapsed_ns: u64,
}

#[derive(Clone)]
struct Instruction {
    /// Validated F32 operands in semantic input order. Integer reshape operands
    /// belong to the compiled shape program rather than kernel execution.
    inputs: Box<[TensorId]>,
    output: TensorId,
    kind: Op,
}
#[derive(Clone)]
enum Op {
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
struct GraphUses {
    offsets: Vec<usize>,
    consumers: Vec<usize>,
    producers: Vec<usize>,
}
impl GraphUses {
    fn new(instructions: &[Instruction], tensors: usize) -> Self {
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
    fn consumers(&self, id: TensorId) -> &[usize] {
        &self.consumers[self.offsets[id.0]..self.offsets[id.0 + 1]]
    }
    fn producer(&self, id: TensorId) -> Option<usize> {
        let index = self.producers[id.0];
        (index != usize::MAX).then_some(index)
    }
    fn unique_consumer(&self, id: TensorId, count: usize) -> Option<usize> {
        let (first, rest) = self.consumers(id).split_first()?;
        if *first >= count || rest.first().is_some_and(|&next| next < count) {
            None
        } else {
            Some(*first)
        }
    }
    fn used_after(&self, id: TensorId, end: usize, count: usize) -> bool {
        let uses = self.consumers(id);
        let after = uses.partition_point(|&index| index <= end);
        uses.get(after).is_some_and(|&index| index < count)
    }
}

/// Immutable instructions over one owned model; constants are never copied.
pub struct CompiledModel {
    model: Arc<Model>,
    instructions: Vec<Instruction>,
    shape_program: ShapeProgram,
    last_use: Vec<isize>,
    graph_uses: GraphUses,
    input: TensorId,
    output: TensorId,
    options: CompileOptions,
    backend: Backend,
    packed: HashMap<TensorId, Arc<PackedMatrix>>,
    conv_packed: HashMap<usize, Arc<PackedMatrix>>,
    transpose_packed: HashMap<usize, Arc<[PackedMatrix; 4]>>,
}
impl CompiledModel {
    pub fn new(model: Arc<Model>) -> RuntimeResult<Self> {
        Self::compile(model, CompileOptions::default())
    }
    pub fn compile(model: Arc<Model>, options: CompileOptions) -> RuntimeResult<Self> {
        let backend = simd::select_backend(options.kernels).map_err(RuntimeError::new)?;
        let graph = model.graph();
        if graph.inputs.len() != 1 || graph.outputs.len() != 1 {
            return Err(RuntimeError::new(
                "CPU runtime requires one graph input and output",
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
                "constant graph inputs/outputs are unsupported by CPU runtime",
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
        let graph_uses = GraphUses::new(&instructions, graph.tensors.len());
        let mut compiled = Self {
            model,
            instructions,
            shape_program,
            last_use,
            graph_uses,
            input,
            output,
            options,
            backend,
            packed: HashMap::new(),
            conv_packed: HashMap::new(),
            transpose_packed: HashMap::new(),
        };
        compiled.prepare_packs()?;
        Ok(compiled)
    }
    pub fn options(&self) -> CompileOptions {
        self.options
    }
    pub fn backend(&self) -> Backend {
        self.backend
    }
    pub fn packed_weight_bytes(&self) -> usize {
        let mut seen = std::collections::HashSet::new();
        let matrices: usize = self
            .packed
            .values()
            .chain(self.conv_packed.values())
            .filter(|p| seen.insert(Arc::as_ptr(p)))
            .map(|p| p.bytes())
            .sum();
        let mut seen_transpose = std::collections::HashSet::new();
        matrices
            + self
                .transpose_packed
                .values()
                .filter(|p| seen_transpose.insert(Arc::as_ptr(p)))
                .map(|p| p.iter().map(PackedMatrix::bytes).sum::<usize>())
                .sum::<usize>()
    }
    pub(crate) fn packed_matrix(&self, id: TensorId) -> Option<&PackedMatrix> {
        self.packed.get(&id).map(Arc::as_ref)
    }
    /// Resolve the format prepared for the physical instruction. Convolution
    /// panels have a different source order from row-major MatMul panels.
    pub(crate) fn instruction_packed(&self, index: usize) -> Option<&PackedMatrix> {
        let instruction = &self.instructions[index];
        match instruction.kind {
            Op::Conv(_) => self.conv_packed.get(&index).map(Arc::as_ref),
            Op::MatMul => instruction
                .inputs
                .get(1)
                .and_then(|id| self.packed_matrix(*id)),
            _ => None,
        }
    }
    fn prepare_packs(&mut self) -> RuntimeResult<()> {
        if self.backend == Backend::Scalar {
            return Ok(());
        }
        for (index, instruction) in self.instructions.iter().enumerate() {
            let Some(&id) = instruction.inputs.get(1) else {
                continue;
            };
            let Some(TensorData::F32(weight)) = &self.model.graph().tensors[id.0].constant else {
                continue;
            };
            let weight_shape = weight.shape();
            let dims = weight_shape.dimensions();
            match &instruction.kind {
                Op::MatMul if dims.len() == 2 => {
                    if let std::collections::hash_map::Entry::Vacant(entry) = self.packed.entry(id)
                    {
                        entry.insert(self.model.packed_weights.matrix(
                            id,
                            packing::MatrixPacking::RowMajor,
                            || {
                                PackedMatrix::new(weight.data(), dims[0], dims[1])
                                    .map_err(RuntimeError::new)
                            },
                        )?);
                    }
                }
                Op::Conv(attr)
                    if dims.len() == 4
                        && (attr.group == 1
                            || (usize::try_from(attr.group).ok() == Some(dims[0])
                                && dims[1] == 1))
                        && (self.options.layout == LayoutPolicy::Auto
                            || nchw::pack_eligible(attr, dims)) =>
                {
                    let pack = self.model.packed_weights.matrix::<RuntimeError>(
                        id,
                        packing::MatrixPacking::ConvOutputMajor,
                        || {
                            let columns = dims[0];
                            let inner = weight.data().len() / columns;
                            let mut transposed = allocate_buffer(weight.data().len())?;
                            for k in 0..inner {
                                for column in 0..columns {
                                    transposed[k * columns + column] =
                                        weight.data()[column * inner + k];
                                }
                            }
                            PackedMatrix::new(&transposed, inner, columns)
                                .map_err(RuntimeError::new)
                        },
                    )?;
                    self.conv_packed.insert(index, pack);
                }
                Op::ConvTranspose
                    if dims.len() == 4
                        && dims[2..] == [2, 2]
                        && dims[1] >= 16
                        && self.options.layout == LayoutPolicy::Auto =>
                {
                    let (inner, columns) = (dims[0], dims[1]);
                    let pack_tap = |tap: usize| -> RuntimeResult<PackedMatrix> {
                        let mut matrix = allocate_buffer(weight.data().len() / 4)?;
                        for (i, value) in matrix.iter_mut().enumerate() {
                            *value = weight.data()[i * 4 + tap];
                        }
                        PackedMatrix::new(&matrix, inner, columns).map_err(RuntimeError::new)
                    };
                    let pack = self
                        .model
                        .packed_weights
                        .transpose::<RuntimeError>(id, || {
                            Ok([pack_tap(0)?, pack_tap(1)?, pack_tap(2)?, pack_tap(3)?])
                        })?;
                    self.transpose_packed.insert(index, pack);
                }
                _ => {}
            }
        }
        Ok(())
    }
    pub fn model(&self) -> &Model {
        &self.model
    }
    pub fn node_count(&self) -> usize {
        self.instructions.len()
    }
    /// Bind metadata without reserving activation storage or executing kernels.
    #[cfg(test)]
    pub(crate) fn node_output_shapes(&self, input_shape: &[usize]) -> RuntimeResult<Vec<Shape>> {
        let plan = bind(self, input_shape, ExecutionLimits::default())?;
        Ok(plan
            .steps
            .iter()
            .map(|step| plan.shapes[step.output.0])
            .collect())
    }
}

fn node_error(index: usize, node: &Node, message: impl Into<RuntimeError>) -> RuntimeError {
    message.into().context(format!(
        "node {index} '{}' ({}, {})",
        node.name,
        node.op_type(),
        node.key
    ))
}
fn integer_values(data: &TensorData) -> RuntimeResult<Vec<i64>> {
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
struct ShapePlan {
    input_shape: Shape,
    shapes: Vec<Shape>,
    steps: Vec<BoundStep>,
    slots: Vec<usize>,
    lengths: Vec<usize>,
    activation_bytes: usize,
    naive_activation_bytes: usize,
}
impl ShapePlan {
    fn empty(tensors: usize, nodes: usize) -> Self {
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
struct BindScratch {
    resolved: Vec<Option<Shape>>,
    shape_cache: Vec<Option<shapes::Integers>>,
    starts: Vec<isize>,
    last_use: Vec<isize>,
    lifetimes: Vec<(isize, isize, usize, usize)>,
    ends: Vec<isize>,
}
impl BindScratch {
    fn new(compiled: &CompiledModel) -> Self {
        let count = compiled.model.graph().tensors.len();
        Self {
            resolved: vec![None; count],
            shape_cache: compiled.shape_program.cache(),
            starts: vec![-2; count],
            last_use: vec![-1; count],
            lifetimes: Vec::with_capacity(count),
            ends: Vec::with_capacity(count),
        }
    }
}

/// One mutable request. Borrowed output remains valid until the next mutable
/// operation; Rust prevents running or reshaping while that borrow is held.
pub struct Session {
    compiled: Arc<CompiledModel>,
    limits: ExecutionLimits,
    plan: ShapePlan,
    staging: ShapePlan,
    scratch: BindScratch,
    buffers: Vec<Vec<f32>>,
    spare_buffers: Vec<Vec<f32>>,
    node_count: usize,
    selected_output: TensorId,
    cpu: Arc<CpuContext>,
    threads: usize,
    optimized: Option<optimized::State>,
    diagnostic: Option<Box<Session>>,
    prefix: Option<Box<Session>>,
    cache: VecDeque<ShapePlan>,
    cache_metadata: bool,
    cache_hits: u64,
    cache_misses: u64,
}
impl fmt::Debug for Session {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Session")
            .field("input_shape", &self.input_shape())
            .field("output_shape", &self.output_shape())
            .field("activation_bytes", &self.activation_bytes())
            .field(
                "retained_activation_bytes",
                &self.retained_activation_bytes(),
            )
            .finish_non_exhaustive()
    }
}
impl Session {
    pub fn new(
        compiled: Arc<CompiledModel>,
        cpu: Arc<CpuContext>,
        input_shape: &[usize],
        options: ExecutionOptions,
    ) -> RuntimeResult<Self> {
        if options.intra_op_threads == 0 || options.intra_op_threads > cpu.capacity() {
            return Err(RuntimeError::new(
                "intra-op threads must be within CPU context capacity",
            ));
        }
        let node_count = compiled.instructions.len();
        let output = compiled.output;
        let mut session = Self::new_selected(
            compiled,
            input_shape,
            ExecutionLimits {
                max_activation_bytes: options.max_activation_bytes,
            },
            node_count,
            output,
        )?;
        session.cpu = cpu;
        session.threads = options.intra_op_threads.min(16);
        Ok(session)
    }
    pub fn new_serial(
        compiled: Arc<CompiledModel>,
        input_shape: &[usize],
        limits: ExecutionLimits,
    ) -> RuntimeResult<Self> {
        Self::new(
            compiled,
            CpuContext::serial(),
            input_shape,
            ExecutionOptions {
                max_activation_bytes: limits.max_activation_bytes,
                intra_op_threads: 1,
            },
        )
    }
    pub fn cpu_context(&self) -> &Arc<CpuContext> {
        &self.cpu
    }
    pub fn execution_limits(&self) -> ExecutionLimits {
        self.limits
    }
    /// Effective intra-op budget, capped at 16 after validating the requested value.
    pub fn intra_op_threads(&self) -> usize {
        self.threads
    }
    pub fn set_execution_context(
        &mut self,
        cpu: Arc<CpuContext>,
        threads: usize,
    ) -> RuntimeResult<()> {
        if threads == 0 || threads > cpu.capacity() {
            return Err(RuntimeError::new(
                "intra-op threads must be within CPU context capacity",
            ));
        }
        self.cpu = cpu;
        self.threads = threads.min(16);
        Ok(())
    }
    pub fn set_intra_op_threads(&mut self, threads: usize) -> RuntimeResult<()> {
        self.set_execution_context(self.cpu.clone(), threads)
    }
    pub fn statistics(&self) -> ExecutionStatistics {
        let mut stats = self
            .optimized
            .as_ref()
            .map_or_else(ExecutionStatistics::default, |state| {
                state.statistics(&self.compiled, self.threads)
            });
        stats.normalized_nodes = self.node_count;
        stats.original_weight_bytes = self.compiled.model().normalized_weight_bytes();
        if self.optimized.is_none() {
            stats.physical_steps = self.node_count;
            stats.selected_kernels = kernel_selections(
                self.compiled.instructions[..self.node_count]
                    .iter()
                    .enumerate()
                    .map(|(i, _)| {
                        (
                            self.compiled.model.graph().nodes[i].op_type(),
                            "canonical",
                            optimized::native_consumes_packed(
                                &self.plan.steps[i].kind,
                                self.compiled.backend,
                                self.compiled.instruction_packed(i).is_some(),
                                self.threads,
                            ),
                        )
                    }),
            );
            stats.specialized_kernels =
                specialization_counts(self.plan.steps[..self.node_count].iter().enumerate().map(
                    |(index, step)| {
                        let (family, isa) = optimized::native_selection(
                            &step.kind,
                            self.compiled.backend,
                            self.compiled.instruction_packed(index).is_some(),
                            self.threads,
                            false,
                        );
                        (
                            self.compiled.model.graph().nodes[index].op_type(),
                            family,
                            isa,
                            false,
                        )
                    },
                ));
        }
        stats.packed_weight_bytes = self.compiled.packed_weight_bytes();
        stats.shared_cached_weight_bytes = self.compiled.model.cached_packed_weight_bytes();
        stats.activation_bytes = self.activation_bytes();
        stats.retained_activation_bytes = self.retained_activation_bytes();
        stats.plan_cache_hits = self.cache_hits;
        stats.plan_cache_misses = self.cache_misses;
        stats.backend = format!("{:?}", self.compiled.backend);
        stats.scratch_bytes = self.retained_scratch_bytes();
        stats
    }
    /// Construct an inclusive prefix with no storage for the unexecuted suffix.
    /// Rebinding preserves this endpoint and the borrowed output is its feature.
    pub(crate) fn new_prefix(
        compiled: Arc<CompiledModel>,
        input_shape: &[usize],
        limits: ExecutionLimits,
        inclusive_last_node: usize,
    ) -> RuntimeResult<Self> {
        let output = compiled
            .instructions
            .get(inclusive_last_node)
            .ok_or_else(|| RuntimeError::new("prefix node index is outside the compiled graph"))?
            .output;
        Self::new_selected(
            compiled,
            input_shape,
            limits,
            inclusive_last_node + 1,
            output,
        )
    }
    fn new_selected(
        compiled: Arc<CompiledModel>,
        input_shape: &[usize],
        limits: ExecutionLimits,
        node_count: usize,
        selected_output: TensorId,
    ) -> RuntimeResult<Self> {
        Self::new_selected_mode(
            compiled,
            input_shape,
            limits,
            node_count,
            selected_output,
            false,
        )
    }
    pub(crate) fn new_diagnostic(
        compiled: Arc<CompiledModel>,
        input_shape: &[usize],
        limits: ExecutionLimits,
    ) -> RuntimeResult<Self> {
        let count = compiled.instructions.len();
        let output = compiled.output;
        Self::new_selected_mode(compiled, input_shape, limits, count, output, true)
    }
    fn new_selected_mode(
        compiled: Arc<CompiledModel>,
        input_shape: &[usize],
        limits: ExecutionLimits,
        node_count: usize,
        selected_output: TensorId,
        diagnostic: bool,
    ) -> RuntimeResult<Self> {
        let tensors = compiled.model.graph().tensors.len();
        let physical = !diagnostic
            && (compiled.options.layout == LayoutPolicy::Auto || compiled.options.fusion);
        let cache_metadata = diagnostic || physical || compiled.backend != Backend::Scalar;
        let mut plan = ShapePlan::empty(tensors, node_count);
        let staging = ShapePlan::empty(tensors, node_count);
        let mut scratch = BindScratch::new(&compiled);
        bind_into(
            &compiled,
            input_shape,
            if physical {
                ExecutionLimits {
                    max_activation_bytes: usize::MAX,
                }
            } else {
                limits
            },
            node_count,
            selected_output,
            &mut plan,
            &mut scratch,
        )?;
        let optimized = if physical {
            Some(optimized::State::new(
                &compiled,
                &plan,
                node_count,
                selected_output,
                limits,
            )?)
        } else {
            None
        };
        let mut buffers = if optimized.is_none() {
            allocate(&plan.lengths)?
        } else {
            Vec::new()
        };
        buffers
            .try_reserve_exact(tensors.saturating_sub(buffers.len()))
            .map_err(|e| {
                RuntimeError::caused_by(
                    crate::error::ErrorKind::Allocation,
                    "activation table allocation failed",
                    e,
                )
            })?;
        check_retained_limit(&buffers, limits)?;
        Ok(Self {
            compiled,
            limits,
            plan,
            staging,
            scratch,
            buffers,
            spare_buffers: Vec::with_capacity(tensors),
            node_count,
            selected_output,
            cpu: CpuContext::serial(),
            threads: 1,
            optimized,
            diagnostic: None,
            prefix: None,
            cache: VecDeque::new(),
            cache_metadata,
            cache_hits: 0,
            cache_misses: 1,
        })
    }
    /// Invalid shapes or failed reservations leave the previous binding intact.
    pub fn reshape(&mut self, input_shape: &[usize]) -> RuntimeResult<()> {
        let limits = self.activation_limits_after(
            self.diagnostic
                .as_ref()
                .map_or(0, |s| s.retained_activation_bytes())
                + self
                    .prefix
                    .as_ref()
                    .map_or(0, |s| s.retained_activation_bytes()),
        )?;
        if input_shape == self.plan.input_shape.dimensions() {
            self.cache_hits += 1;
            return Ok(());
        }
        if self.cache_metadata {
            let index = self
                .cache
                .iter()
                .position(|p| p.input_shape.dimensions() == input_shape);
            let mut next = if let Some(index) = index {
                self.cache_hits += 1;
                self.cache.remove(index).unwrap()
            } else {
                self.cache_misses += 1;
                let mut next = if self.cache.len() >= 8 {
                    self.cache.pop_back().unwrap()
                } else {
                    ShapePlan::empty(self.compiled.model.graph().tensors.len(), self.node_count)
                };
                bind_into(
                    &self.compiled,
                    input_shape,
                    if self.optimized.is_some() {
                        ExecutionLimits {
                            max_activation_bytes: usize::MAX,
                        }
                    } else {
                        limits
                    },
                    self.node_count,
                    self.selected_output,
                    &mut next,
                    &mut self.scratch,
                )?;
                next
            };
            if let Some(optimized) = &mut self.optimized {
                optimized.reshape(
                    &self.compiled,
                    &next,
                    self.node_count,
                    self.selected_output,
                    limits,
                )?;
            } else {
                rebind_buffers(
                    &mut self.buffers,
                    &next.lengths,
                    &mut self.spare_buffers,
                    limits,
                )?;
            }
            std::mem::swap(&mut self.plan, &mut next);
            self.cache.push_front(next);
            self.cache.truncate(8);
            return Ok(());
        }
        bind_into(
            &self.compiled,
            input_shape,
            limits,
            self.node_count,
            self.selected_output,
            &mut self.staging,
            &mut self.scratch,
        )?;
        rebind_buffers(
            &mut self.buffers,
            &self.staging.lengths,
            &mut self.spare_buffers,
            limits,
        )?;
        std::mem::swap(&mut self.plan, &mut self.staging);
        self.cache_misses += 1;
        Ok(())
    }
    pub fn input_shape(&self) -> Shape {
        self.plan.input_shape
    }
    pub fn output_shape(&self) -> Shape {
        self.plan.shapes[self.selected_output.0]
    }
    pub fn activation_bytes(&self) -> usize {
        self.optimized.as_ref().map_or(
            self.plan.activation_bytes,
            optimized::State::activation_bytes,
        )
    }
    fn own_retained_activation_bytes(&self) -> usize {
        self.optimized.as_ref().map_or_else(
            || capacity_bytes(&self.buffers).expect("validated activation capacities"),
            optimized::State::retained_bytes,
        )
    }
    fn activation_limits_after(&self, reserved: usize) -> RuntimeResult<ExecutionLimits> {
        Ok(ExecutionLimits {
            max_activation_bytes: self
                .limits
                .max_activation_bytes
                .checked_sub(reserved)
                .ok_or_else(|| {
                    RuntimeError::new("retained sibling activations exceed max_activation_bytes")
                })?,
        })
    }
    fn reshape_with_limits(
        &mut self,
        shape: &[usize],
        limits: ExecutionLimits,
    ) -> RuntimeResult<()> {
        let previous = self.limits;
        self.limits = limits;
        let result = self.reshape(shape);
        if result.is_err() {
            self.limits = previous;
        }
        result
    }
    /// Total F32 capacity retained by this Session and its cached diagnostic or
    /// prefix execution. Smaller bindings can retain larger slots, within the
    /// configured aggregate activation limit.
    pub fn retained_activation_bytes(&self) -> usize {
        self.own_retained_activation_bytes()
            + self
                .diagnostic
                .as_ref()
                .map_or(0, |s| s.retained_activation_bytes())
            + self
                .prefix
                .as_ref()
                .map_or(0, |s| s.retained_activation_bytes())
    }
    pub fn retained_scratch_bytes(&self) -> usize {
        self.optimized
            .as_ref()
            .map_or(0, optimized::State::scratch_bytes)
            + self
                .diagnostic
                .as_ref()
                .map_or(0, |s| s.retained_scratch_bytes())
            + self
                .prefix
                .as_ref()
                .map_or(0, |s| s.retained_scratch_bytes())
    }
    /// Retained session tables and transactional shape-binding scratch. Shared
    /// compiled metadata and F32 activation capacities are reported separately.
    /// Retained per-session plans and binding scratch, excluding F32 buffers and shared instructions.
    pub fn retained_planning_bytes(&self) -> usize {
        let plan_bytes = |plan: &ShapePlan| {
            plan.shapes.capacity() * size_of::<Shape>()
                + plan.steps.capacity() * size_of::<BoundStep>()
                + (plan.slots.capacity() + plan.lengths.capacity()) * size_of::<usize>()
        };
        self.optimized
            .as_ref()
            .map_or(0, optimized::State::planning_bytes)
            + self
                .diagnostic
                .as_ref()
                .map_or(0, |s| s.retained_planning_bytes())
            + self
                .prefix
                .as_ref()
                .map_or(0, |s| s.retained_planning_bytes())
            + self.cache.iter().map(plan_bytes).sum::<usize>()
            + plan_bytes(&self.plan)
            + plan_bytes(&self.staging)
            + self.scratch.resolved.capacity() * size_of::<Option<Shape>>()
            + self.scratch.shape_cache.capacity() * size_of::<Option<shapes::Integers>>()
            + (self.scratch.starts.capacity()
                + self.scratch.last_use.capacity()
                + self.scratch.ends.capacity())
                * size_of::<isize>()
            + self.scratch.lifetimes.capacity() * size_of::<(isize, isize, usize, usize)>()
            + (self.buffers.capacity() + self.spare_buffers.capacity()) * size_of::<Vec<f32>>()
    }
    pub fn naive_activation_bytes(&self) -> usize {
        self.plan.naive_activation_bytes
    }
    pub fn run(&mut self, input: &[f32]) -> RuntimeResult<TensorView<'_, f32>> {
        self.check_input_length(input)?;
        self.run_with_input(|destination| {
            destination.copy_from_slice(input);
            Ok(())
        })
    }
    pub fn run_with_trace<F>(
        &mut self,
        input: &[f32],
        observer: F,
    ) -> RuntimeResult<TensorView<'_, f32>>
    where
        F: for<'a> FnMut(NodeTrace<'a>) -> RuntimeResult<()>,
    {
        self.check_input_length(input)?;
        self.run_with_input_and_trace(
            |destination| {
                destination.copy_from_slice(input);
                Ok(())
            },
            observer,
        )
    }
    /// Fill the active input directly, then execute. The callback must write
    /// every input element each time: later nodes can reuse the input storage.
    /// Caller callbacks can allocate; successful execution itself does not.
    pub fn run_with_input<F>(&mut self, fill: F) -> RuntimeResult<TensorView<'_, f32>>
    where
        F: FnOnce(&mut [f32]) -> RuntimeResult<()>,
    {
        if let Some(state) = &mut self.optimized {
            state.fill(fill, self.compiled.backend, &self.cpu, self.threads)?;
            self.cpu
                .install(|| state.execute(&self.compiled, &self.cpu, self.threads))?;
            return state.output();
        }
        self.prepare_input(fill)?;
        self.execute::<false, _>(self.plan.steps.len(), &mut |_| Ok(()))?;
        self.output_for(self.selected_output)
    }
    /// Internal preprocessing writes directly to the selected physical input.
    pub(crate) fn run_with_layout_input<F>(&mut self, fill: F) -> RuntimeResult<TensorView<'_, f32>>
    where
        F: FnOnce(&mut [f32], bool) -> RuntimeResult<()>,
    {
        if let Some(state) = &mut self.optimized {
            state.fill_layout(fill)?;
            self.cpu
                .install(|| state.execute(&self.compiled, &self.cpu, self.threads))?;
            return state.output();
        }
        self.run_with_input(|input| fill(input, false))
    }
    /// Fill the input once and synchronously observe each completed node.
    pub fn run_with_input_and_trace<F, O>(
        &mut self,
        fill: F,
        mut observer: O,
    ) -> RuntimeResult<TensorView<'_, f32>>
    where
        F: FnOnce(&mut [f32]) -> RuntimeResult<()>,
        O: for<'a> FnMut(NodeTrace<'a>) -> RuntimeResult<()>,
    {
        if self.optimized.is_some() {
            let limits = self.activation_limits_after(
                self.own_retained_activation_bytes()
                    + self
                        .prefix
                        .as_ref()
                        .map_or(0, |s| s.retained_activation_bytes()),
            )?;
            if self.diagnostic.is_none() {
                let session = Self::new_selected_mode(
                    self.compiled.clone(),
                    self.plan.input_shape.dimensions(),
                    limits,
                    self.node_count,
                    self.selected_output,
                    true,
                )?;
                self.diagnostic = Some(Box::new(session));
            }
            let session = self.diagnostic.as_mut().unwrap();
            session.reshape_with_limits(self.plan.input_shape.dimensions(), limits)?;
            session.set_execution_context(self.cpu.clone(), self.threads)?;
            return session.run_with_input_and_trace(fill, observer);
        }
        self.prepare_input(fill)?;
        self.execute::<true, _>(self.plan.steps.len(), &mut observer)?;
        self.output_for(self.selected_output)
    }
    /// Execute a validated inclusive prefix without evaluating later nodes.
    /// Its feature view has the same borrowing contract as a complete output.
    #[cfg(test)]
    pub(crate) fn run_prefix_with_input<F>(
        &mut self,
        last_node_index: usize,
        fill: F,
    ) -> RuntimeResult<TensorView<'_, f32>>
    where
        F: FnOnce(&mut [f32]) -> RuntimeResult<()>,
    {
        if self.optimized.is_some() {
            if last_node_index >= self.plan.steps.len() {
                return Err(RuntimeError::new(
                    "prefix node index is outside the compiled graph",
                ));
            }
            if last_node_index + 1 == self.node_count {
                return self.run_with_input(fill);
            }
            let limits = self.activation_limits_after(
                self.own_retained_activation_bytes()
                    + self
                        .diagnostic
                        .as_ref()
                        .map_or(0, |s| s.retained_activation_bytes()),
            )?;
            // Prefixes have their own physical region; no fusion can consume a suffix.
            if self
                .prefix
                .as_ref()
                .is_none_or(|p| p.node_count != last_node_index + 1)
            {
                self.prefix = Some(Box::new(Self::new_prefix(
                    self.compiled.clone(),
                    self.plan.input_shape.dimensions(),
                    limits,
                    last_node_index,
                )?));
            }
            let prefix = self.prefix.as_mut().unwrap();
            prefix.reshape_with_limits(self.plan.input_shape.dimensions(), limits)?;
            prefix.set_execution_context(self.cpu.clone(), self.threads)?;
            return prefix.run_with_input(fill);
        }
        let output = self
            .plan
            .steps
            .get(last_node_index)
            .ok_or_else(|| RuntimeError::new("prefix node index is outside the compiled graph"))?
            .output;
        self.prepare_input(fill)?;
        self.execute::<false, _>(last_node_index + 1, &mut |_| Ok(()))?;
        self.output_for(output)
    }
    fn check_input_length(&self, input: &[f32]) -> RuntimeResult<()> {
        if input.len() != self.plan.input_shape.element_count() {
            return Err(RuntimeError::new(
                "input element count does not match bound shape",
            ));
        }
        Ok(())
    }
    fn prepare_input<F>(&mut self, fill: F) -> RuntimeResult<()>
    where
        F: FnOnce(&mut [f32]) -> RuntimeResult<()>,
    {
        let slot = self.plan.slots[self.compiled.input.0];
        let input = &mut self.buffers[slot][..self.plan.input_shape.element_count()];
        fill(input)?;
        if let Some(index) = input.iter().position(|value| !value.is_finite()) {
            return Err(RuntimeError::new(format!(
                "nonfinite input F32 at element {index}"
            )));
        }
        Ok(())
    }
    fn output_for(&self, id: TensorId) -> RuntimeResult<TensorView<'_, f32>> {
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
        node_count: usize,
        observer: &mut F,
    ) -> RuntimeResult<()>
    where
        F: for<'a> FnMut(NodeTrace<'a>) -> RuntimeResult<()>,
    {
        for index in 0..node_count {
            let node_timer = TRACE.then(std::time::Instant::now);
            let step = &self.plan.steps[index];
            let slot = self.plan.slots[step.output.0];
            // Strict lifetime intervals ensure no current input occupies this
            // slot. Taking the Vec gives a separate safe mutable destination.
            let mut output = std::mem::take(&mut self.buffers[slot]);
            let count = self.plan.shapes[step.output.0].element_count();
            self.cpu.install(|| {
                kernels::execute_optimized(
                    &step.kind,
                    &self.compiled.instructions[index].inputs,
                    |id| self.data(id),
                    &mut output[..count],
                    self.compiled.backend,
                    self.compiled.instruction_packed(index),
                    &self.cpu,
                    self.threads,
                )
            });
            self.buffers[slot] = output;
            if TRACE {
                let elapsed_ns = node_timer
                    .unwrap()
                    .elapsed()
                    .as_nanos()
                    .try_into()
                    .unwrap_or(u64::MAX);
                let node = &self.compiled.model.graph().nodes[index];
                let shape = self.plan.shapes[step.output.0];
                observer(NodeTrace {
                    index,
                    key: &node.key,
                    name: &node.name,
                    op_type: node.op_type(),
                    output_id: step.output,
                    shape,
                    layout: if shape.rank() == 4 {
                        "nchw"
                    } else {
                        "row_major"
                    },
                    values: self.data(step.output),
                    elapsed_ns,
                })?;
            }
        }
        let output_id = if node_count == 0 {
            self.selected_output
        } else {
            self.plan.steps[node_count - 1].output
        };
        if let Some(index) = self.data(output_id).iter().position(|v| !v.is_finite()) {
            return Err(RuntimeError::new(format!(
                "nonfinite output F32 at element {index}"
            )));
        }
        Ok(())
    }
}

fn allocate(lengths: &[usize]) -> RuntimeResult<Vec<Vec<f32>>> {
    let mut buffers = Vec::new();
    buffers.try_reserve_exact(lengths.len()).map_err(|e| {
        RuntimeError::caused_by(
            crate::error::ErrorKind::Allocation,
            "activation table allocation failed",
            e,
        )
    })?;
    for &length in lengths {
        buffers.push(allocate_buffer(length)?);
    }
    Ok(buffers)
}

fn allocate_buffer(length: usize) -> RuntimeResult<Vec<f32>> {
    let mut buffer = Vec::new();
    buffer.try_reserve_exact(length).map_err(|e| {
        RuntimeError::caused_by(
            crate::error::ErrorKind::Allocation,
            "activation allocation failed",
            e,
        )
    })?;
    buffer.resize(length, 0.0);
    Ok(buffer)
}

fn byte_total(mut capacities: impl Iterator<Item = usize>) -> RuntimeResult<usize> {
    capacities
        .try_fold(0usize, |sum, capacity| {
            capacity
                .checked_mul(size_of::<f32>())
                .and_then(|bytes| sum.checked_add(bytes))
        })
        .ok_or_else(|| RuntimeError::new("retained activation capacity overflow"))
}

fn capacity_bytes(buffers: &[Vec<f32>]) -> RuntimeResult<usize> {
    byte_total(buffers.iter().map(Vec::capacity))
}

fn check_retained_limit(buffers: &[Vec<f32>], limits: ExecutionLimits) -> RuntimeResult<()> {
    let bytes = capacity_bytes(buffers)?;
    if bytes > limits.max_activation_bytes {
        return Err(RuntimeError::with_kind(
            crate::error::ErrorKind::ResourceLimit,
            format!(
                "retained activation capacity needs {bytes} bytes, limit is {}",
                limits.max_activation_bytes
            ),
        ));
    }
    Ok(())
}

fn rebind_buffers(
    previous: &mut Vec<Vec<f32>>,
    lengths: &[usize],
    prepared: &mut Vec<Vec<f32>>,
    limits: ExecutionLimits,
) -> RuntimeResult<()> {
    if lengths.iter().enumerate().all(|(slot, &length)| {
        previous
            .get(slot)
            .is_some_and(|buffer| buffer.capacity() >= length)
    }) {
        check_retained_limit(previous, limits)?;
        for (buffer, &length) in previous.iter_mut().zip(lengths) {
            buffer.resize(length, 0.0);
        }
        return Ok(());
    }
    prepared.clear();
    let prepare = (|| -> RuntimeResult<bool> {
        let slots = previous.len().max(lengths.len());
        let proposed = byte_total((0..slots).map(|slot| {
            previous
                .get(slot)
                .map_or(0, Vec::capacity)
                .max(lengths.get(slot).copied().unwrap_or(0))
        }))?;
        let compact = proposed > limits.max_activation_bytes;
        let count = if compact { lengths.len() } else { slots };
        prepared.try_reserve_exact(count).map_err(|e| {
            RuntimeError::caused_by(
                crate::error::ErrorKind::Allocation,
                "activation table allocation failed",
                e,
            )
        })?;
        for slot in 0..count {
            let length = lengths.get(slot).copied().unwrap_or(0);
            prepared.push(
                if !compact && previous.get(slot).is_some_and(|b| b.capacity() >= length) {
                    Vec::new()
                } else {
                    allocate_buffer(length)?
                },
            );
        }
        let retained = byte_total(prepared.iter().enumerate().map(|(slot, buffer)| {
            if !compact && buffer.is_empty() {
                previous[slot].capacity()
            } else {
                buffer.capacity()
            }
        }))?;
        if retained > limits.max_activation_bytes {
            // An allocator may return more capacity than requested. Retry with
            // an entirely compact replacement without disturbing old storage.
            prepared.clear();
            for &length in lengths {
                prepared.push(allocate_buffer(length)?);
            }
            check_retained_limit(prepared, limits)?;
            return Ok(false);
        }
        Ok(!compact)
    })();
    let reuse = match prepare {
        Ok(reuse) => reuse,
        Err(error) => {
            prepared.clear();
            return Err(error);
        }
    };
    // All fallible operations are complete. The inactive plan and old binding
    // stay intact until this capacity-qualified ownership transfer commits.
    if reuse {
        for (slot, buffer) in prepared.iter_mut().enumerate() {
            if buffer.is_empty() {
                *buffer = std::mem::take(&mut previous[slot]);
                if let Some(&length) = lengths.get(slot) {
                    buffer.resize(length, 0.0);
                }
            }
        }
    }
    std::mem::swap(previous, prepared);
    prepared.clear();
    Ok(())
}

#[cfg(test)]
fn bind(
    compiled: &CompiledModel,
    dimensions: &[usize],
    limits: ExecutionLimits,
) -> RuntimeResult<ShapePlan> {
    let mut plan = ShapePlan::empty(
        compiled.model.graph().tensors.len(),
        compiled.instructions.len(),
    );
    let mut scratch = BindScratch::new(compiled);
    bind_into(
        compiled,
        dimensions,
        limits,
        compiled.instructions.len(),
        compiled.output,
        &mut plan,
        &mut scratch,
    )?;
    Ok(plan)
}
#[allow(clippy::too_many_arguments)]
fn bind_into(
    compiled: &CompiledModel,
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
    scratch.resolved.fill(None);
    scratch.shape_cache.fill(None);
    let resolved = &mut scratch.resolved;
    for (id, tensor) in graph.tensors.iter().enumerate() {
        if let Some(data) = &tensor.constant {
            resolved[id] = Some(data.shape());
        }
    }
    resolved[compiled.input.0] = Some(input_shape);
    plan.steps.clear();
    for (index, instruction) in compiled.instructions[..node_count].iter().enumerate() {
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
                            kernels::Broadcast::new(a, b, output),
                            match instruction.kind {
                                Op::Add => kernels::BinaryOp::Add,
                                Op::Mul => kernels::BinaryOp::Mul,
                                Op::Div => kernels::BinaryOp::Div,
                                _ => kernels::BinaryOp::Sub,
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
                Op::Conv(attrs) => kernels::bind_conv(a, b.unwrap(), bias, attrs),
                Op::ConvTranspose => kernels::bind_conv_transpose(a, b.unwrap(), bias),
                Op::MaxPool => kernels::bind_max_pool(a),
                Op::AveragePool => kernels::bind_average_pool(a),
                Op::BatchNormalization { epsilon } => {
                    let mut parameters = [Shape::new(&[])?; 4];
                    for (shape, id) in parameters.iter_mut().zip(&instruction.inputs[1..]) {
                        *shape = resolved[id.0].ok_or_else(|| {
                            RuntimeError::new("unresolved BatchNormalization parameter")
                        })?;
                    }
                    kernels::bind_batch_normalization(a, &parameters, *epsilon)
                }
                Op::Squeeze { axes } => Ok((shapes::squeeze(a, axes)?, BoundKind::Copy)),
                Op::Unsqueeze { axes } => Ok((shapes::unsqueeze(a, axes)?, BoundKind::Copy)),
                Op::Transpose { permutation } => kernels::bind_transpose(a, permutation),
                Op::Slice(parameters) => kernels::bind_slice(a, parameters),
                Op::Concat { axis } => kernels::bind_concat(&instruction.inputs, resolved, *axis),
                Op::Resize { factor } => kernels::bind_resize(a, *factor),
                Op::GlobalAveragePool => {
                    if a.rank() < 3 {
                        return Err(RuntimeError::new(
                            "GlobalAveragePool requires at least one spatial dimension",
                        ));
                    }
                    let axes = (2..a.rank()).map(|axis| axis as i64).collect::<Vec<_>>();
                    kernels::bind_reduce(a, &axes, true, false)
                }
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
                        evaluated = compiled.shape_program.evaluate(
                            index,
                            resolved,
                            &mut scratch.shape_cache,
                        )?;
                        evaluated.values()
                    };
                    Ok((shapes::reshape(a, values, *allowzero)?, BoundKind::Copy))
                }
                Op::MatMul => kernels::bind_matmul(a, b.unwrap()),
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
    if node_count == compiled.instructions.len()
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
    plan_slots_into(compiled, node_count, selected_output, plan, scratch, limits)
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
fn plan_slots_into(
    compiled: &CompiledModel,
    node_count: usize,
    selected_output: TensorId,
    plan: &mut ShapePlan,
    scratch: &mut BindScratch,
    limits: ExecutionLimits,
) -> RuntimeResult<()> {
    let graph = compiled.model.graph();
    scratch.starts.fill(-2);
    scratch.starts[compiled.input.0] = -1;
    scratch.last_use.copy_from_slice(&compiled.last_use);
    if node_count != compiled.instructions.len() {
        scratch.last_use.fill(-1);
        for (index, instruction) in compiled.instructions[..node_count].iter().enumerate() {
            for id in &instruction.inputs {
                scratch.last_use[id.0] = index as isize;
            }
        }
    }
    scratch.last_use[selected_output.0] = node_count as isize;
    for (index, instruction) in compiled.instructions[..node_count].iter().enumerate() {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn varint(mut value: u64) -> Vec<u8> {
        let mut bytes = Vec::new();
        loop {
            let next = (value & 127) as u8;
            value >>= 7;
            bytes.push(next | if value == 0 { 0 } else { 128 });
            if value == 0 {
                return bytes;
            }
        }
    }
    fn number(field: u64, value: u64) -> Vec<u8> {
        [varint(field << 3), varint(value)].concat()
    }
    fn bytes(field: u64, value: &[u8]) -> Vec<u8> {
        [
            varint(field << 3 | 2),
            varint(value.len() as u64),
            value.to_vec(),
        ]
        .concat()
    }
    fn info(name: &str) -> Vec<u8> {
        [
            bytes(1, name.as_bytes()),
            bytes(
                2,
                &bytes(
                    1,
                    &[number(1, 1), bytes(2, &bytes(1, &number(1, 2)))].concat(),
                ),
            ),
        ]
        .concat()
    }
    fn matrix_info(name: &str, columns: u64) -> Vec<u8> {
        [
            bytes(1, name.as_bytes()),
            bytes(
                2,
                &bytes(
                    1,
                    &[
                        number(1, 1),
                        bytes(
                            2,
                            &[bytes(1, &bytes(2, b"batch")), bytes(1, &number(1, columns))]
                                .concat(),
                        ),
                    ]
                    .concat(),
                ),
            ),
        ]
        .concat()
    }
    fn wide_head() -> Arc<CompiledModel> {
        let weights = bytes(
            5,
            &[
                number(1, 2),
                number(1, 1024),
                number(2, 1),
                bytes(8, b"weights"),
                bytes(9, &vec![0u8; 2 * 1024 * 4]),
            ]
            .concat(),
        );
        let graph = [
            weights,
            node("Relu", &["x"], "feature"),
            node("MatMul", &["feature", "weights"], "logits"),
            node("Softmax", &["logits"], "y"),
            bytes(11, &matrix_info("x", 2)),
            bytes(12, &matrix_info("y", 1024)),
        ]
        .concat();
        let source = [number(1, 8), bytes(8, &number(2, 7)), bytes(7, &graph)].concat();
        Arc::new(
            CompiledModel::compile(
                Arc::new(Model::load_bytes(&source).unwrap()),
                CompileOptions::reference(),
            )
            .unwrap(),
        )
    }
    fn physical_wide_head() -> Arc<CompiledModel> {
        Arc::new(CompiledModel::new(wide_head().model.clone()).unwrap())
    }

    #[test]
    fn parsed_models_share_packed_weights_across_concurrent_compilation_policies() {
        let model = wide_head().model.clone();
        assert_eq!(model.cached_packed_weight_bytes(), 0);
        let handles: Vec<_> = (0..4)
            .map(|index| {
                let model = model.clone();
                std::thread::spawn(move || {
                    CompiledModel::compile(
                        model,
                        CompileOptions {
                            kernels: KernelPolicy::Portable4,
                            layout: if index % 2 == 0 {
                                LayoutPolicy::Auto
                            } else {
                                LayoutPolicy::Nchw
                            },
                            fusion: index % 2 == 0,
                        },
                    )
                    .unwrap()
                })
            })
            .collect();
        let compiled: Vec<_> = handles
            .into_iter()
            .map(|thread| Arc::new(thread.join().unwrap()))
            .collect();
        let first = &compiled[0];
        assert!(!first.packed.is_empty());
        for other in &compiled[1..] {
            for (id, panel) in &first.packed {
                assert!(Arc::ptr_eq(panel, &other.packed[id]));
            }
        }
        let cached = model.cached_packed_weight_bytes();
        assert_eq!(cached, first.packed_weight_bytes());
        for view in compiled {
            let mut request =
                Session::new_serial(view, &[1, 2], ExecutionLimits::default()).unwrap();
            assert!(
                request
                    .run(&[1., 2.])
                    .unwrap()
                    .data()
                    .iter()
                    .all(|&value| value == 1. / 1024.)
            );
            let stats = request.statistics();
            assert_eq!(stats.shared_cached_weight_bytes, cached);
            assert_eq!(
                stats
                    .specialized_kernels
                    .iter()
                    .map(|entry| entry.steps)
                    .sum::<usize>(),
                stats.physical_steps
            );
            assert!(
                stats
                    .selected_kernels
                    .iter()
                    .any(|entry| entry.operator == "MatMul" && entry.packed)
            );
        }
        assert_eq!(model.cached_packed_weight_bytes(), cached);
        let scalar = CompiledModel::compile(model, CompileOptions::reference()).unwrap();
        assert_eq!(scalar.packed_weight_bytes(), 0);
    }

    #[test]
    fn aggregate_activation_limit_rejects_lazy_trace_before_callbacks() {
        let compiled = physical_wide_head();
        let production =
            Session::new_serial(compiled.clone(), &[1, 2], ExecutionLimits::default()).unwrap();
        let diagnostic =
            Session::new_diagnostic(compiled.clone(), &[1, 2], ExecutionLimits::default()).unwrap();
        let total = production.retained_activation_bytes() + diagnostic.retained_activation_bytes();
        let mut request = Session::new_serial(
            compiled.clone(),
            &[1, 2],
            ExecutionLimits {
                max_activation_bytes: total - 4,
            },
        )
        .unwrap();
        let mut filled = false;
        let mut observed = false;
        assert!(
            request
                .run_with_input_and_trace(
                    |input| {
                        filled = true;
                        input.fill(0.);
                        Ok(())
                    },
                    |_| {
                        observed = true;
                        Ok(())
                    }
                )
                .is_err()
        );
        assert!(!filled && !observed);
        assert!(request.diagnostic.is_none());
        assert_eq!(
            request.run(&[0., 0.]).unwrap().data(),
            vec![1. / 1024.; 1024]
        );
        assert!(request.retained_activation_bytes() <= request.limits.max_activation_bytes);
        let mut fitting = Session::new_serial(
            compiled,
            &[1, 2],
            ExecutionLimits {
                max_activation_bytes: total,
            },
        )
        .unwrap();
        let mut nodes = 0;
        fitting
            .run_with_trace(&[0., 0.], |_| {
                nodes += 1;
                Ok(())
            })
            .unwrap();
        assert_eq!(nodes, 3);
        assert_eq!(fitting.retained_activation_bytes(), total);
    }

    #[test]
    fn aggregate_activation_limit_rebinds_existing_diagnostic_transactionally() {
        let compiled = physical_wide_head();
        let production =
            Session::new_serial(compiled.clone(), &[2, 2], ExecutionLimits::default()).unwrap();
        let diagnostic =
            Session::new_diagnostic(compiled.clone(), &[1, 2], ExecutionLimits::default()).unwrap();
        let limit = production.retained_activation_bytes() + diagnostic.retained_activation_bytes();
        let mut request = Session::new_serial(
            compiled,
            &[1, 2],
            ExecutionLimits {
                max_activation_bytes: limit,
            },
        )
        .unwrap();
        request.run_with_trace(&[0., 0.], |_| Ok(())).unwrap();
        request.reshape(&[2, 2]).unwrap();
        let previous = request.retained_activation_bytes();
        let mut filled = false;
        assert!(
            request
                .run_with_input_and_trace(
                    |_| {
                        filled = true;
                        Ok(())
                    },
                    |_| Ok(())
                )
                .is_err()
        );
        assert!(!filled);
        assert_eq!(request.retained_activation_bytes(), previous);
        assert!(previous <= limit);
        assert_eq!(request.input_shape().dimensions(), &[2, 2]);
        assert_eq!(
            request
                .diagnostic
                .as_ref()
                .unwrap()
                .input_shape()
                .dimensions(),
            &[1, 2]
        );
        assert!(
            request
                .run(&[0.; 4])
                .unwrap()
                .data()
                .iter()
                .all(|&x| x == 1. / 1024.)
        );
        assert!(request.reshape(&[3, 2]).is_err());
        assert_eq!(request.input_shape().dimensions(), &[2, 2]);
        request.reshape(&[1, 2]).unwrap();
        request.run_with_trace(&[0., 0.], |_| Ok(())).unwrap();
        assert!(request.retained_activation_bytes() <= limit);
    }

    #[test]
    fn aggregate_activation_limit_counts_test_prefix_and_diagnostic_siblings() {
        let compiled = physical_wide_head();
        let production =
            Session::new_serial(compiled.clone(), &[1, 2], ExecutionLimits::default()).unwrap();
        let prefix =
            Session::new_prefix(compiled.clone(), &[1, 2], ExecutionLimits::default(), 0).unwrap();
        let primary = production.retained_activation_bytes();
        let prefix_bytes = prefix.retained_activation_bytes();
        let diagnostic =
            Session::new_diagnostic(compiled.clone(), &[1, 2], ExecutionLimits::default()).unwrap();
        let total = primary + prefix_bytes + diagnostic.retained_activation_bytes();
        let mut request = Session::new_serial(
            compiled.clone(),
            &[1, 2],
            ExecutionLimits {
                max_activation_bytes: primary + prefix_bytes - 4,
            },
        )
        .unwrap();
        let mut filled = false;
        assert!(
            request
                .run_prefix_with_input(0, |_| {
                    filled = true;
                    Ok(())
                })
                .is_err()
        );
        assert!(!filled);
        assert!(request.prefix.is_none());
        let mut request = Session::new_serial(
            compiled,
            &[1, 2],
            ExecutionLimits {
                max_activation_bytes: total,
            },
        )
        .unwrap();
        request
            .run_prefix_with_input(0, |input| {
                input.fill(0.);
                Ok(())
            })
            .unwrap();
        request.run_with_trace(&[0., 0.], |_| Ok(())).unwrap();
        assert_eq!(request.retained_activation_bytes(), total);
        let mut replaced = false;
        assert!(
            request
                .run_prefix_with_input(1, |_| {
                    replaced = true;
                    Ok(())
                })
                .is_err()
        );
        assert!(!replaced);
        assert_eq!(request.prefix.as_ref().unwrap().node_count, 1);
        assert_eq!(request.retained_activation_bytes(), total);
        assert!(request.reshape(&[2, 2]).is_err());
        assert_eq!(request.input_shape().dimensions(), &[1, 2]);
        assert!(request.run(&[0., 0.]).is_ok());
    }
    #[test]
    fn requested_intra_op_budgets_are_validated_before_the_effective_cap() {
        let compiled = wide_head();
        #[cfg(feature = "parallel")]
        let (capacity, requests, invalid) = (32, &[1, 16, 17, 32][..], [0, 33]);
        #[cfg(not(feature = "parallel"))]
        let (capacity, requests, invalid) = (1, &[1][..], [0, 2]);
        let cpu = Arc::new(CpuContext::new(capacity).unwrap());
        for &requested in requests {
            let session = Session::new(
                compiled.clone(),
                cpu.clone(),
                &[1, 2],
                ExecutionOptions {
                    intra_op_threads: requested,
                    ..ExecutionOptions::default()
                },
            )
            .unwrap();
            assert_eq!(session.intra_op_threads(), requested.min(16));
            assert!(Arc::ptr_eq(session.cpu_context(), &cpu));
        }
        for requested in invalid {
            assert!(
                Session::new(
                    compiled.clone(),
                    cpu.clone(),
                    &[1, 2],
                    ExecutionOptions {
                        intra_op_threads: requested,
                        ..ExecutionOptions::default()
                    },
                )
                .is_err()
            );
        }
        let mut session =
            Session::new_serial(compiled, &[1, 2], ExecutionLimits::default()).unwrap();
        let original = session.cpu_context().clone();
        for requested in invalid {
            assert!(
                session
                    .set_execution_context(cpu.clone(), requested)
                    .is_err()
            );
            assert!(Arc::ptr_eq(session.cpu_context(), &original));
            assert_eq!(session.intra_op_threads(), 1);
        }
        session
            .set_execution_context(cpu.clone(), capacity)
            .unwrap();
        assert_eq!(session.intra_op_threads(), capacity.min(16));
        for requested in invalid {
            assert!(session.set_intra_op_threads(requested).is_err());
            assert!(Arc::ptr_eq(session.cpu_context(), &cpu));
            assert_eq!(session.intra_op_threads(), capacity.min(16));
        }
        session.set_intra_op_threads(1).unwrap();
        assert_eq!(session.intra_op_threads(), 1);
        assert_eq!(session.run(&[-1., 2.]).unwrap().data()[0], 1. / 1024.);
    }
    #[test]
    fn diagnostic_and_nchw_simd_plans_cache_eight_shapes_without_extra_arenas() {
        let reference = wide_head();
        let simd = Arc::new(
            CompiledModel::compile(
                reference.model.clone(),
                CompileOptions {
                    kernels: KernelPolicy::Portable4,
                    layout: LayoutPolicy::Nchw,
                    fusion: false,
                },
            )
            .unwrap(),
        );
        let limits = ExecutionLimits {
            max_activation_bytes: 128 << 10,
        };
        let diagnostic = Session::new_diagnostic(reference, &[1, 2], limits).unwrap();
        let production = Session::new_serial(simd, &[1, 2], limits).unwrap();
        for mut request in [diagnostic, production] {
            for batch in 2..=9 {
                request.reshape(&[batch, 2]).unwrap();
            }
            assert_eq!(request.cache.len(), 8);
            let misses = request.statistics().plan_cache_misses;
            let arenas = request.buffers.len();
            let retained = request.retained_activation_bytes();
            request.reshape(&[2, 2]).unwrap();
            assert_eq!(request.statistics().plan_cache_misses, misses);
            assert_eq!(request.buffers.len(), arenas);
            assert_eq!(request.retained_activation_bytes(), retained);
            assert!(request.reshape(&[1024, 2]).is_err());
            assert_eq!(request.input_shape().dimensions(), &[2, 2]);
            let output = request.run(&[-1., 2., 3., -4.]).unwrap();
            assert_eq!(output.shape().dimensions(), &[2, 1024]);
            assert!(output.data().iter().all(|&x| x == 1. / 1024.));
            request.reshape(&[10, 2]).unwrap();
            assert_eq!(request.cache.len(), 8);
        }
    }
    #[test]
    fn prefix_binding_excludes_suffix_storage_and_reuses_mixed_shape_metadata() {
        let compiled = wide_head();
        let limits = ExecutionLimits {
            max_activation_bytes: 256,
        };
        assert!(Session::new_serial(compiled.clone(), &[1, 2], limits).is_err());
        assert!(Session::new_prefix(compiled.clone(), &[1, 2], limits, 3).is_err());
        let mut request = Session::new_prefix(compiled, &[1, 2], limits, 0).unwrap();
        assert_eq!(request.activation_bytes(), 16);
        assert_eq!(request.naive_activation_bytes(), 16);
        assert_eq!(request.output_shape().dimensions(), &[1, 2]);
        assert_eq!(request.plan.steps.len(), 1);
        let suffix = request.compiled.instructions[1].output;
        assert_eq!(request.plan.slots[suffix.0], usize::MAX);
        request.reshape(&[16, 2]).unwrap();
        let storage = request
            .buffers
            .iter()
            .map(|buffer| buffer.as_ptr())
            .collect::<Vec<_>>();
        let resolved = request.scratch.resolved.as_ptr();
        let lifetimes = request.scratch.lifetimes.as_ptr();
        let mut step_tables = [request.plan.steps.as_ptr(), request.staging.steps.as_ptr()];
        step_tables.sort();
        for batch in [1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 1, 16] {
            request.reshape(&[batch, 2]).unwrap();
            assert_eq!(request.output_shape().dimensions(), &[batch, 2]);
            assert_eq!(request.scratch.resolved.as_ptr(), resolved);
            assert_eq!(request.scratch.lifetimes.as_ptr(), lifetimes);
            let mut current = [request.plan.steps.as_ptr(), request.staging.steps.as_ptr()];
            current.sort();
            assert_eq!(current, step_tables);
            for (buffer, &pointer) in request.buffers.iter().zip(&storage) {
                assert_eq!(buffer.as_ptr(), pointer);
            }
            let mut observed = 0;
            request
                .run_with_input_and_trace(
                    |input| {
                        input.fill(-1.);
                        Ok(())
                    },
                    |trace| {
                        observed += 1;
                        assert_eq!(trace.index, 0);
                        Ok(())
                    },
                )
                .unwrap();
            assert_eq!(observed, 1);
        }
        assert!(request.reshape(&[17, 2]).is_err());
        assert_eq!(request.input_shape().dimensions(), &[16, 2]);
        assert!(
            request
                .run_with_input_and_trace(
                    |input| {
                        input.fill(1.);
                        Ok(())
                    },
                    |_| { Err(RuntimeError::new("observer failed")) }
                )
                .is_err()
        );
        assert!(
            request
                .run_with_input(|_| Err(RuntimeError::new("fill failed")))
                .is_err()
        );
        assert!(
            request
                .run_with_input(|input| {
                    input.fill(f32::NAN);
                    Ok(())
                })
                .is_err()
        );
        assert!(
            request
                .run_with_input(|input| {
                    input.fill(1.);
                    Ok(())
                })
                .unwrap()
                .data()
                .iter()
                .all(|&value| value == 1.)
        );
    }
    fn node(op: &str, inputs: &[&str], output: &str) -> Vec<u8> {
        bytes(
            1,
            &[
                inputs
                    .iter()
                    .flat_map(|name| bytes(1, name.as_bytes()))
                    .collect(),
                bytes(2, output.as_bytes()),
                bytes(4, op.as_bytes()),
            ]
            .concat(),
        )
    }
    fn prefix_session() -> Session {
        let constant = bytes(
            5,
            &[
                number(1, 1),
                number(2, 1),
                bytes(8, b"large"),
                bytes(9, &f32::MAX.to_le_bytes()),
            ]
            .concat(),
        );
        let graph = [
            constant,
            node("Relu", &["x"], "feature"),
            node("Mul", &["feature", "large"], "overflow"),
            node("Relu", &["overflow"], "y"),
            bytes(11, &info("x")),
            bytes(12, &info("y")),
        ]
        .concat();
        let source = [number(1, 8), bytes(8, &number(2, 7)), bytes(7, &graph)].concat();
        let model = Arc::new(Model::load_bytes(&source).unwrap());
        Session::new_serial(
            Arc::new(CompiledModel::new(model).unwrap()),
            &[2],
            ExecutionLimits::default(),
        )
        .unwrap()
    }

    #[test]
    fn inclusive_prefix_stops_before_failing_suffix_and_checks_index_before_fill() {
        let mut request = prefix_session();
        let mut called = false;
        assert!(
            request
                .run_prefix_with_input(3, |_| {
                    called = true;
                    Ok(())
                })
                .is_err()
        );
        assert!(!called);
        let output = request
            .run_prefix_with_input(0, |input| {
                input.copy_from_slice(&[-1., 2.]);
                Ok(())
            })
            .unwrap();
        assert_eq!(output.shape().dimensions(), &[2]);
        assert_eq!(output.data(), &[0., 2.]);
        assert!(
            request
                .run_prefix_with_input(1, |input| {
                    input.copy_from_slice(&[-1., 2.]);
                    Ok(())
                })
                .unwrap_err()
                .to_string()
                .contains("nonfinite output")
        );
        assert!(
            request
                .run_prefix_with_input(0, |_| Err(RuntimeError::new("fill failed")))
                .is_err()
        );
        assert_eq!(
            request
                .run_prefix_with_input(0, |input| {
                    input.copy_from_slice(&[3., -4.]);
                    Ok(())
                })
                .unwrap()
                .data(),
            &[3., 0.]
        );
        assert!(request.run(&[3., -4.]).is_err());
        assert_eq!(
            request
                .run_prefix_with_input(0, |input| {
                    input.copy_from_slice(&[1., -2.]);
                    Ok(())
                })
                .unwrap()
                .data(),
            &[1., 0.]
        );
    }

    #[test]
    fn failed_buffer_reservation_leaves_previous_storage_owned_and_unchanged() {
        let mut previous = vec![vec![1.0, 2.0, 3.0]];
        let pointer = previous[0].as_ptr();
        // Byte arithmetic fits usize, but Vec's isize allocation bound rejects
        // this capacity before any actual giant allocation is attempted.
        let lengths = [usize::MAX / size_of::<f32>()];
        let mut prepared = Vec::new();
        let error = rebind_buffers(
            &mut previous,
            &lengths,
            &mut prepared,
            ExecutionLimits {
                max_activation_bytes: usize::MAX,
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains("allocation failed"));
        assert_eq!(previous[0].as_ptr(), pointer);
        assert_eq!(previous[0], [1.0, 2.0, 3.0]);
        assert!(prepared.is_empty());
    }

    #[test]
    fn metadata_only_node_shapes_agree_with_session_binding_and_traces() {
        let mut request = prefix_session();
        let shapes = request.compiled.node_output_shapes(&[2]).unwrap();
        assert_eq!(shapes, vec![Shape::new(&[2]).unwrap(); 3]);
        assert!(request.buffers.iter().flatten().all(|&value| value == 0.0));
        let bound = request
            .plan
            .steps
            .iter()
            .map(|step| request.plan.shapes[step.output.0])
            .collect::<Vec<_>>();
        assert_eq!(shapes, bound);
        let mut observed = Vec::new();
        request
            .run_with_trace(&[0., 0.], |trace| {
                observed.push(trace.shape);
                Ok(())
            })
            .unwrap();
        assert_eq!(shapes, observed);
        assert!(request.compiled.node_output_shapes(&[3]).is_err());
    }
}
