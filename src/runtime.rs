//! Shared logical graphs with independent SIMD CPU and native GPU execution.
//!
//! Models, graph lowering and shape binding are shared across execution backends.
//! Compiled instructions are immutable and shareable. Each session binds concrete
//! shapes and owns reusable activation buffers. The optional `gpu` module uses
//! wgpu for tensor and OCR inference. This deliberately does not claim
//! general ONNX execution support.

#[cfg(feature = "gpu")]
pub mod gpu;
pub(crate) mod stage;
pub use stage::Session;
pub use stage::{CompiledModel, ExecutionBackend, ExecutionConfig, ExecutionTarget};
mod graph;
mod graph_patterns;
pub use graph::LogicalGraph;
#[cfg(feature = "gpu")]
use graph::compile_node;
use graph::{BindScratch, Instruction, Op, Program, ShapePlan, bind_into};
mod bound;
mod compiler;
pub use compiler::{Compiler, CompilerOptions};
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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize)]
pub enum GpuPrecision {
    #[default]
    F32,
    MixedF16,
}

/// Arithmetic precision for eligible GPU matrix operations, independent of
/// activation storage precision. F16 converts matrix operands to half precision
/// and accumulates in F32; other operations retain their storage policy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize)]
pub enum GpuMatrixPrecision {
    #[default]
    F32,
    F16,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct GpuCompileOptions {
    pub precision: GpuPrecision,
    pub matrix_precision: GpuMatrixPrecision,
}
/// Settings shared by all compilation backends.
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct CompileOptions {
    pub fusion: bool,
    pub cpu: CpuCompileOptions,
    pub gpu: GpuCompileOptions,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct CpuCompileOptions {
    pub kernels: KernelPolicy,
    pub layout: LayoutPolicy,
}
impl Default for CpuCompileOptions {
    fn default() -> Self {
        Self {
            kernels: KernelPolicy::Auto,
            layout: LayoutPolicy::Auto,
        }
    }
}
impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            fusion: true,
            cpu: Default::default(),
            gpu: Default::default(),
        }
    }
}
impl CompileOptions {
    /// Preserve execution arithmetic. Pair with `OptimizationPolicy::PreserveArithmetic` at import.
    pub fn reference() -> Self {
        Self {
            fusion: false,
            cpu: CpuCompileOptions {
                kernels: KernelPolicy::Scalar,
                layout: LayoutPolicy::Nchw,
            },
            ..Default::default()
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ExecutionOptions {
    /// Total retained F32 activation capacity for this CpuSession, including lazy
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
    pub execution_backend: Option<ExecutionBackend>,
    pub fallback: Option<crate::error::FailureProvenance>,
    pub effective_precision: String,
    pub adapter: Option<String>,
    pub production_submissions: u64,
    pub production_readbacks: u64,
    pub diagnostic_readbacks: u64,
    pub uploaded_bytes: u64,
    pub output_readback_bytes: u64,
    pub diagnostic_readback_bytes: u64,
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
    /// All CPU formats retained by the shared compiler cache, including other policies.
    pub shared_cached_weight_bytes: usize,
    pub activation_bytes: usize,
    pub retained_activation_bytes: usize,
    pub scratch_bytes: usize,
    pub plan_cache_hits: u64,
    pub plan_cache_misses: u64,
    pub backend: String,
}
/// The cumulative fields used to compute a run's deltas, without kernel metadata.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct StatisticsCounters {
    pub production_submissions: u64,
    pub production_readbacks: u64,
    pub diagnostic_readbacks: u64,
    pub uploaded_bytes: u64,
    pub output_readback_bytes: u64,
    pub diagnostic_readback_bytes: u64,
    pub plan_cache_hits: u64,
    pub plan_cache_misses: u64,
}
impl StatisticsCounters {
    #[cfg(feature = "gpu")]
    pub(crate) fn add_diagnostics(&mut self, other: Self) {
        self.diagnostic_readbacks += other.diagnostic_readbacks;
        self.diagnostic_readback_bytes += other.diagnostic_readback_bytes;
    }
}
/// Disjoint retained request allocations. Shared weights and transition peaks are separate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct MemorySnapshot {
    pub host_bytes: usize,
    pub device_bytes: usize,
}
impl MemorySnapshot {
    pub fn total_bytes(self) -> usize {
        self.host_bytes.saturating_add(self.device_bytes)
    }
    pub fn host(host_bytes: usize) -> Self {
        Self {
            host_bytes,
            device_bytes: 0,
        }
    }
}
/// Cumulative traffic, separate from the latest execution's placement and timing.
#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct ExecutionCounters {
    pub runs: u64,
    pub failures: u64,
    pub submissions: u64,
    pub uploaded_bytes: u64,
    pub output_readback_bytes: u64,
    pub control_readback_bytes: u64,
    pub diagnostic_readback_bytes: u64,
}
impl ExecutionStatistics {
    pub(crate) fn retained_bytes(&self) -> usize {
        self.effective_precision.capacity()
            + self.backend.capacity()
            + self.adapter.as_ref().map_or(0, String::capacity)
            + self.fallback.as_ref().map_or(0, |f| f.message.capacity())
            + self.selected_kernels.capacity() * size_of::<KernelSelection>()
            + self
                .selected_kernels
                .iter()
                .map(|k| k.operator.capacity() + k.layout.capacity())
                .sum::<usize>()
            + self.specialized_kernels.capacity() * size_of::<KernelSpecialization>()
            + self
                .specialized_kernels
                .iter()
                .map(|k| k.operator.capacity() + k.family.capacity() + k.isa.capacity())
                .sum::<usize>()
    }

    #[cfg(test)]
    pub(crate) fn counter_snapshot(&self) -> StatisticsCounters {
        StatisticsCounters {
            production_submissions: self.production_submissions,
            production_readbacks: self.production_readbacks,
            diagnostic_readbacks: self.diagnostic_readbacks,
            uploaded_bytes: self.uploaded_bytes,
            output_readback_bytes: self.output_readback_bytes,
            diagnostic_readback_bytes: self.diagnostic_readback_bytes,
            plan_cache_hits: self.plan_cache_hits,
            plan_cache_misses: self.plan_cache_misses,
        }
    }

    pub(crate) fn since(&mut self, previous: &StatisticsCounters) {
        // A fallback may replace a session; its counters start from zero.
        fn delta(now: u64, before: u64) -> u64 {
            now.checked_sub(before).unwrap_or(now)
        }
        self.production_submissions =
            delta(self.production_submissions, previous.production_submissions);
        self.production_readbacks = delta(self.production_readbacks, previous.production_readbacks);
        self.diagnostic_readbacks = delta(self.diagnostic_readbacks, previous.diagnostic_readbacks);
        self.uploaded_bytes = delta(self.uploaded_bytes, previous.uploaded_bytes);
        self.output_readback_bytes =
            delta(self.output_readback_bytes, previous.output_readback_bytes);
        self.diagnostic_readback_bytes = delta(
            self.diagnostic_readback_bytes,
            previous.diagnostic_readback_bytes,
        );
        self.plan_cache_hits = delta(self.plan_cache_hits, previous.plan_cache_hits);
        self.plan_cache_misses = delta(self.plan_cache_misses, previous.plan_cache_misses);
    }
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
    model::{Model, TensorId},
    tensor::{Shape, TensorData, TensorView},
};
use bound::BoundStep;
use std::{fmt, sync::Arc};

pub use crate::error::RuntimeError;

pub type RuntimeResult<T> = Result<T, RuntimeError>;

#[derive(Clone, Copy, Debug)]
pub struct ExecutionLimits {
    /// Bound on total active/retained F32 activation capacity for one CpuSession,
    /// including its diagnostic/prefix children. For a GPU session, this bounds
    /// device activations, device readback staging and host output capacity combined.
    /// Weights, plans and scratch are separate for both execution backends.
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

/// Immutable CPU instructions over one owned model; constants are never copied.
pub struct CpuCompiledModel {
    program: Arc<LogicalGraph>,
    pub(crate) packed_cache: Arc<packing::WeightCache>,
    options: CompileOptions,
    backend: Backend,
    packed: HashMap<TensorId, Arc<PackedMatrix>>,
    conv_packed: HashMap<usize, Arc<PackedMatrix>>,
    transpose_packed: HashMap<usize, Arc<[PackedMatrix; 4]>>,
}
impl CpuCompiledModel {
    pub fn new(model: Arc<Model>) -> RuntimeResult<Self> {
        Self::compile(model, CompileOptions::default())
    }
    pub fn compile(model: Arc<Model>, options: CompileOptions) -> RuntimeResult<Self> {
        Self::from_graph(Arc::new(LogicalGraph::compile(model)?), options)
    }
    pub fn from_graph(program: Arc<LogicalGraph>, options: CompileOptions) -> RuntimeResult<Self> {
        Self::from_graph_cached(program, options, Arc::default())
    }
    pub(crate) fn from_graph_cached(
        program: Arc<LogicalGraph>,
        options: CompileOptions,
        packed_cache: Arc<packing::WeightCache>,
    ) -> RuntimeResult<Self> {
        let backend = simd::select_backend(options.cpu.kernels).map_err(RuntimeError::new)?;
        let mut compiled = Self {
            program,
            packed_cache,
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
        let instruction = &self.program.instructions[index];
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
        for (index, instruction) in self.program.instructions.iter().enumerate() {
            let Some(&id) = instruction.inputs.get(1) else {
                continue;
            };
            let Some(TensorData::F32(weight)) = &self.program.model.graph().tensors[id.0].constant
            else {
                continue;
            };
            let weight_shape = weight.shape();
            let dims = weight_shape.dimensions();
            match &instruction.kind {
                Op::MatMul if dims.len() == 2 => {
                    if let std::collections::hash_map::Entry::Vacant(entry) = self.packed.entry(id)
                    {
                        entry.insert(self.packed_cache.matrix(
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
                        && (self.options.cpu.layout == LayoutPolicy::Auto
                            || nchw::pack_eligible(attr, dims)) =>
                {
                    let pack = self.packed_cache.matrix::<RuntimeError>(
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
                        && self.options.cpu.layout == LayoutPolicy::Auto =>
                {
                    let (inner, columns) = (dims[0], dims[1]);
                    let pack_tap = |tap: usize| -> RuntimeResult<PackedMatrix> {
                        let mut matrix = allocate_buffer(weight.data().len() / 4)?;
                        for (i, value) in matrix.iter_mut().enumerate() {
                            *value = weight.data()[i * 4 + tap];
                        }
                        PackedMatrix::new(&matrix, inner, columns).map_err(RuntimeError::new)
                    };
                    let pack = self.packed_cache.transpose::<RuntimeError>(id, || {
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
        &self.program.model
    }
    pub fn node_count(&self) -> usize {
        self.program.instructions.len()
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

/// One mutable request. Borrowed output remains valid until the next mutable
/// operation; Rust prevents running or reshaping while that borrow is held.
pub struct CpuSession {
    compiled: Arc<CpuCompiledModel>,
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
    diagnostic: Option<Box<CpuSession>>,
    prefix: Option<Box<CpuSession>>,
    cache: VecDeque<ShapePlan>,
    cache_metadata: bool,
    cache_hits: u64,
    cache_misses: u64,
}
impl fmt::Debug for CpuSession {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CpuSession")
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
impl CpuSession {
    pub fn new(
        compiled: Arc<CpuCompiledModel>,
        cpu: Arc<CpuContext>,
        input_shape: &[usize],
        options: ExecutionOptions,
    ) -> RuntimeResult<Self> {
        if options.intra_op_threads == 0 || options.intra_op_threads > cpu.capacity() {
            return Err(RuntimeError::new(
                "intra-op threads must be within CPU context capacity",
            ));
        }
        let node_count = compiled.program.instructions.len();
        let output = compiled.program.output;
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
        compiled: Arc<CpuCompiledModel>,
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
    pub(crate) fn counter_snapshot(&self) -> StatisticsCounters {
        StatisticsCounters {
            plan_cache_hits: self.cache_hits,
            plan_cache_misses: self.cache_misses,
            ..Default::default()
        }
    }
    pub fn statistics(&self) -> ExecutionStatistics {
        let mut stats = self
            .optimized
            .as_ref()
            .map_or_else(ExecutionStatistics::default, |state| {
                state.statistics(&self.compiled, self.threads)
            });
        stats.normalized_nodes = self.node_count;
        stats.execution_backend = Some(ExecutionBackend::Cpu(self.compiled.backend));
        stats.effective_precision = "F32".into();
        stats.original_weight_bytes = self.compiled.model().normalized_weight_bytes();
        if self.optimized.is_none() {
            stats.physical_steps = self.node_count;
            stats.selected_kernels = kernel_selections(
                self.compiled.program.instructions[..self.node_count]
                    .iter()
                    .enumerate()
                    .map(|(i, _)| {
                        (
                            self.compiled.program.model.graph().nodes[i].op_type(),
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
                            self.compiled.program.model.graph().nodes[index].op_type(),
                            family,
                            isa,
                            false,
                        )
                    },
                ));
        }
        stats.packed_weight_bytes = self.compiled.packed_weight_bytes();
        stats.shared_cached_weight_bytes = self.compiled.packed_cache.bytes();
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
        compiled: Arc<CpuCompiledModel>,
        input_shape: &[usize],
        limits: ExecutionLimits,
        inclusive_last_node: usize,
    ) -> RuntimeResult<Self> {
        let output = compiled
            .program
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
        compiled: Arc<CpuCompiledModel>,
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
        compiled: Arc<CpuCompiledModel>,
        input_shape: &[usize],
        limits: ExecutionLimits,
    ) -> RuntimeResult<Self> {
        let count = compiled.program.instructions.len();
        let output = compiled.program.output;
        Self::new_selected_mode(compiled, input_shape, limits, count, output, true)
    }
    fn new_selected_mode(
        compiled: Arc<CpuCompiledModel>,
        input_shape: &[usize],
        limits: ExecutionLimits,
        node_count: usize,
        selected_output: TensorId,
        diagnostic: bool,
    ) -> RuntimeResult<Self> {
        let tensors = compiled.program.model.graph().tensors.len();
        let physical = !diagnostic
            && (compiled.options.cpu.layout == LayoutPolicy::Auto || compiled.options.fusion);
        let cache_metadata = diagnostic || physical || compiled.backend != Backend::Scalar;
        let mut plan = ShapePlan::empty(tensors, node_count);
        let staging = ShapePlan::empty(tensors, node_count);
        let mut scratch = BindScratch::new(&compiled.program);
        bind_into(
            &compiled.program,
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
                    ShapePlan::empty(
                        self.compiled.program.model.graph().tensors.len(),
                        self.node_count,
                    )
                };
                bind_into(
                    &self.compiled.program,
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
            &self.compiled.program,
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
    /// Total F32 capacity retained by this CpuSession and its cached diagnostic or
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
        let slot = self.plan.slots[self.compiled.program.input.0];
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
    fn output_view(&self) -> RuntimeResult<TensorView<'_, f32>> {
        match &self.optimized {
            Some(state) => state.output(),
            None => self.output_for(self.selected_output),
        }
    }
    fn data(&self, id: TensorId) -> &[f32] {
        match &self.compiled.program.model.graph().tensors[id.0].constant {
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
                    &self.compiled.program.instructions[index].inputs,
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
                let node = &self.compiled.program.model.graph().nodes[index];
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
    compiled: &CpuCompiledModel,
    dimensions: &[usize],
    limits: ExecutionLimits,
) -> RuntimeResult<ShapePlan> {
    let mut plan = ShapePlan::empty(
        compiled.program.model.graph().tensors.len(),
        compiled.program.instructions.len(),
    );
    let mut scratch = BindScratch::new(&compiled.program);
    bind_into(
        &compiled.program,
        dimensions,
        limits,
        compiled.program.instructions.len(),
        compiled.program.output,
        &mut plan,
        &mut scratch,
    )?;
    Ok(plan)
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
    fn wide_head_model() -> Arc<Model> {
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
        Arc::new(Model::load_bytes(&source).unwrap())
    }
    fn wide_head() -> Arc<CpuCompiledModel> {
        Arc::new(CpuCompiledModel::compile(wide_head_model(), CompileOptions::reference()).unwrap())
    }
    fn physical_wide_head() -> Arc<CpuCompiledModel> {
        Arc::new(CpuCompiledModel::new(wide_head().program.model.clone()).unwrap())
    }

    #[test]
    #[cfg(not(feature = "gpu"))]
    fn placement_without_gpu_feature_is_explicit() {
        let compile = |target| {
            CompiledModel::compile(
                wide_head_model(),
                CompileOptions::default(),
                ExecutionConfig { target },
            )
        };
        let strict = compile(ExecutionTarget::Gpu).err().unwrap();
        assert_eq!(strict.kind(), crate::error::ErrorKind::Unsupported);
        let auto = compile(ExecutionTarget::Auto).unwrap();
        assert!(matches!(auto.execution_backend(), ExecutionBackend::Cpu(_)));
        assert!(
            auto.fallback_error()
                .unwrap()
                .to_string()
                .contains("gpu Cargo feature")
        );
        let mut session = Session::new_serial(auto, &[1, 2], ExecutionLimits::default()).unwrap();
        assert_eq!(session.run(&[1., 2.]).unwrap().data().len(), 1024);
        assert!(session.statistics().fallback.is_some());
    }

    #[test]
    fn shared_program_binds_dynamic_shapes_without_cpu_weight_packing() {
        let model = wide_head_model();
        let compiler = Compiler::default();
        let program = compiler.resources(model.clone()).unwrap().graph.clone();
        let count = program.instructions.len();
        let mut plan = ShapePlan::empty(model.graph().tensors.len(), count);
        let mut scratch = BindScratch::new(&program);
        bind_into(
            &program,
            &[3, 2],
            ExecutionLimits::default(),
            count,
            program.output,
            &mut plan,
            &mut scratch,
        )
        .unwrap();
        assert_eq!(plan.input_shape.dimensions(), [3, 2]);
        assert_eq!(plan.shapes[program.output.0].dimensions(), [3, 1024]);
        assert_eq!(compiler.cached_packed_weight_bytes(), 0);
    }

    #[test]
    fn parsed_models_share_packed_weights_across_concurrent_compilation_policies() {
        let model = wide_head().program.model.clone();
        let compiler = Arc::new(Compiler::default());
        assert_eq!(compiler.cached_packed_weight_bytes(), 0);
        let handles: Vec<_> = (0..4)
            .map(|index| {
                let model = model.clone();
                let compiler = compiler.clone();
                std::thread::spawn(move || {
                    compiler
                        .compile(
                            model,
                            CompileOptions {
                                cpu: crate::runtime::CpuCompileOptions {
                                    kernels: KernelPolicy::Portable4,
                                    layout: if index % 2 == 0 {
                                        LayoutPolicy::Auto
                                    } else {
                                        LayoutPolicy::Nchw
                                    },
                                },
                                fusion: index % 2 == 0,
                                ..Default::default()
                            },
                            ExecutionTarget::Cpu,
                        )
                        .unwrap()
                        .as_cpu()
                        .unwrap()
                        .clone()
                })
            })
            .collect();
        let compiled: Vec<_> = handles
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect();
        let first = &compiled[0];
        assert!(!first.packed.is_empty());
        for other in &compiled[1..] {
            for (id, panel) in &first.packed {
                assert!(Arc::ptr_eq(panel, &other.packed[id]));
            }
        }
        let cached = compiler.cached_packed_weight_bytes();
        assert_eq!(cached, first.packed_weight_bytes());
        for view in compiled {
            let mut request =
                CpuSession::new_serial(view, &[1, 2], ExecutionLimits::default()).unwrap();
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
        assert_eq!(compiler.cached_packed_weight_bytes(), cached);
        let scalar = CpuCompiledModel::compile(model, CompileOptions::reference()).unwrap();
        assert_eq!(scalar.packed_weight_bytes(), 0);
    }

    #[test]
    fn aggregate_activation_limit_rejects_lazy_trace_before_callbacks() {
        let compiled = physical_wide_head();
        let production =
            CpuSession::new_serial(compiled.clone(), &[1, 2], ExecutionLimits::default()).unwrap();
        let diagnostic =
            CpuSession::new_diagnostic(compiled.clone(), &[1, 2], ExecutionLimits::default())
                .unwrap();
        let total = production.retained_activation_bytes() + diagnostic.retained_activation_bytes();
        let mut request = CpuSession::new_serial(
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
        let mut fitting = CpuSession::new_serial(
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
            CpuSession::new_serial(compiled.clone(), &[2, 2], ExecutionLimits::default()).unwrap();
        let diagnostic =
            CpuSession::new_diagnostic(compiled.clone(), &[1, 2], ExecutionLimits::default())
                .unwrap();
        let limit = production.retained_activation_bytes() + diagnostic.retained_activation_bytes();
        let mut request = CpuSession::new_serial(
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
            CpuSession::new_serial(compiled.clone(), &[1, 2], ExecutionLimits::default()).unwrap();
        let prefix =
            CpuSession::new_prefix(compiled.clone(), &[1, 2], ExecutionLimits::default(), 0)
                .unwrap();
        let primary = production.retained_activation_bytes();
        let prefix_bytes = prefix.retained_activation_bytes();
        let diagnostic =
            CpuSession::new_diagnostic(compiled.clone(), &[1, 2], ExecutionLimits::default())
                .unwrap();
        let total = primary + prefix_bytes + diagnostic.retained_activation_bytes();
        let mut request = CpuSession::new_serial(
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
        let mut request = CpuSession::new_serial(
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
            let session = CpuSession::new(
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
                CpuSession::new(
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
            CpuSession::new_serial(compiled, &[1, 2], ExecutionLimits::default()).unwrap();
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
            CpuCompiledModel::compile(
                reference.program.model.clone(),
                CompileOptions {
                    cpu: crate::runtime::CpuCompileOptions {
                        kernels: KernelPolicy::Portable4,
                        layout: LayoutPolicy::Nchw,
                    },
                    fusion: false,
                    ..Default::default()
                },
            )
            .unwrap(),
        );
        let limits = ExecutionLimits {
            max_activation_bytes: 128 << 10,
        };
        let diagnostic = CpuSession::new_diagnostic(reference, &[1, 2], limits).unwrap();
        let production = CpuSession::new_serial(simd, &[1, 2], limits).unwrap();
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
        assert!(CpuSession::new_serial(compiled.clone(), &[1, 2], limits).is_err());
        assert!(CpuSession::new_prefix(compiled.clone(), &[1, 2], limits, 3).is_err());
        let mut request = CpuSession::new_prefix(compiled, &[1, 2], limits, 0).unwrap();
        assert_eq!(request.activation_bytes(), 16);
        assert_eq!(request.naive_activation_bytes(), 16);
        assert_eq!(request.output_shape().dimensions(), &[1, 2]);
        assert_eq!(request.plan.steps.len(), 1);
        let suffix = request.compiled.program.instructions[1].output;
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
    fn prefix_session() -> CpuSession {
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
        CpuSession::new_serial(
            Arc::new(CpuCompiledModel::new(model).unwrap()),
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
