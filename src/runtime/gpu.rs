//! Native GPU execution with shared weights, reusable storage and transactional shapes.
use super::{
    BindScratch, ExecutionLimits, ExecutionStatistics, NodeTrace, Program, RuntimeError,
    RuntimeResult, ShapePlan, bind_into, bound::BoundKind,
};
use crate::profile::Timer;
use crate::{
    error::ErrorKind,
    model::{Model, TensorId},
    tensor::{DType, Shape, TensorData, TensorView},
};
#[cfg(test)]
use std::sync::mpsc;
use std::{
    cell::RefCell,
    collections::VecDeque,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};
mod api;
mod attention_packed;
use api::BufferUsage;
pub use api::{GpuAdapterInfo, GpuApi, GpuDeviceType, GpuLimits};
mod kernel;
#[cfg(target_os = "macos")]
mod metal_resident_sources;
#[cfg(target_os = "macos")]
mod metal_sources;
use kernel::{KernelId, KernelKey};
#[cfg(not(target_os = "macos"))]
#[path = "gpu/vulkan.rs"]
mod backend;
#[cfg(target_os = "macos")]
#[path = "gpu/metal.rs"]
mod backend;
mod conv_mpp;
mod conv_mpp_lower;
mod conv_mpp_staged;
mod conv_weight_pack;
#[cfg(target_os = "macos")]
mod depthwise_native;
#[cfg(target_os = "macos")]
#[path = "gpu/metal_execution.rs"]
mod executor;
#[cfg(not(target_os = "macos"))]
use backend as executor;
mod fused_activation;
mod fusions;
mod half_native;
pub(crate) mod heads;
mod linear_mpp;
mod linear_native;
mod max_pool;
#[cfg(all(test, target_os = "macos"))]
mod native_binding_tests;
mod operators;
mod physical;
#[cfg(target_os = "macos")]
#[path = "gpu/metal_pipelines.rs"]
mod pipelines;
#[cfg(not(target_os = "macos"))]
mod pipelines;
#[cfg(target_os = "macos")]
mod pointwise_f16;
#[cfg(target_os = "macos")]
mod pointwise_mpp64;
#[cfg(target_os = "macos")]
mod pointwise_static;
mod profile;
pub(crate) mod resident;
mod se_fusions;
mod small_projection;
mod storage;
mod tail_projection;
pub(crate) mod vision;
mod weight_packing;
pub use super::{GpuCompileOptions, GpuMatrixPrecision, GpuPrecision};
pub use resident::{GpuPipelineStatistics, GpuPlacement};
pub use storage::{GpuLayout, GpuTensorDescriptor};
use weight_packing::WeightPacking;
#[cfg(test)]
mod resident_cache_tests;
#[cfg(test)]
mod tests;
pub use profile::{GpuExecutionProfile, GpuKernelProfile, GpuTimingGranularity};
const WORKGROUP_SIZE: usize = 64;
const PLAN_CACHE_CAPACITY: usize = 8;

/// Calling-thread production wall timings, never device timestamp measurements.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct GpuExecutionTimings {
    pub admission_wait_ns: u64,
    pub encoding_ns: u64,
    pub submission_ns: u64,
    pub completion_wait_ns: u64,
    pub readback_ns: u64,
}

/// All unsupported retained-node variants found without acquiring a device.
#[derive(Clone, Debug, Default)]
pub struct GpuSupportReport {
    pub issues: Vec<GpuSupportIssue>,
}
impl GpuSupportReport {
    pub fn is_supported(&self) -> bool {
        self.issues.is_empty()
    }
}
#[derive(Clone, Debug)]
pub struct GpuSupportIssue {
    pub node_index: usize,
    pub node_name: String,
    pub operator: String,
    pub source_indices: Vec<usize>,
    pub reason: String,
}
/// Resource and scheduling limits for the platform GPU executor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuContextOptions {
    /// Concurrent admitted executions. Each request can retain two GPU waves.
    pub max_in_flight: usize,
    /// Independent recognition shapes in one bounded wave (1..=16).
    pub recognition_wave_size: usize,
    /// Maximum orientation-classifier rows prepared together (1..=16).
    pub classification_batch_size: usize,
    pub max_device_bytes: Option<usize>,
}
impl Default for GpuContextOptions {
    fn default() -> Self {
        Self {
            max_in_flight: 1,
            recognition_wave_size: 1,
            classification_batch_size: 4,
            max_device_bytes: None,
        }
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuDeviceMemoryStatistics {
    pub used_bytes: usize,
    pub peak_bytes: usize,
    pub committed_bytes: usize,
    pub peak_committed_bytes: usize,
}
struct Ledger {
    owner_device: Arc<backend::Device>,
    used: AtomicUsize,
    peak: AtomicUsize,
    limit: Option<usize>,
}
impl Ledger {
    fn reserve(self: &Arc<Self>, bytes: usize) -> RuntimeResult<Arc<Reservation>> {
        let mut previous = self.used.load(Ordering::Acquire);
        loop {
            let next = previous
                .checked_add(bytes)
                .ok_or_else(|| resource("GPU allocation count overflow"))?;
            if self.limit.is_some_and(|limit| next > limit) {
                return Err(resource(format!(
                    "GPU device allocations need {next} bytes, context limit is {}",
                    self.limit.unwrap()
                )));
            }
            match self.used.compare_exchange_weak(
                previous,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    self.peak.fetch_max(next, Ordering::AcqRel);
                    return Ok(Arc::new(Reservation {
                        ledger: self.clone(),
                        bytes,
                    }));
                }
                Err(actual) => previous = actual,
            }
        }
    }
}
struct Reservation {
    ledger: Arc<Ledger>,
    bytes: usize,
}
impl Drop for Reservation {
    fn drop(&mut self) {
        self.ledger.used.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}
#[derive(Clone)]
pub(crate) struct GpuBuffer {
    buffer: backend::Buffer,
    reservation: Arc<Reservation>,
}
impl GpuBuffer {
    fn size(&self) -> u64 {
        self.buffer.size()
    }
    fn usage(&self) -> BufferUsage {
        self.buffer.usage()
    }
    fn bytes(&self) -> usize {
        self.reservation.bytes
    }
}
thread_local! {static ACTIVE_CONTEXTS:RefCell<Vec<usize>>=const {RefCell::new(Vec::new())};}
pub(crate) struct Gate<'a> {
    context: &'a GpuContext,
    key: usize,
    // Admission reentrancy is tracked in thread-local state.
    _thread: std::marker::PhantomData<std::rc::Rc<()>>,
}
impl Drop for Gate<'_> {
    fn drop(&mut self) {
        if let Ok(mut count) = self.context.in_flight.lock() {
            *count -= 1;
            // Every waiter must observe a callback-recorded fatal error, even
            // when the first awakened waiter returns without acquiring a gate.
            self.context.available.notify_all();
        }
        ACTIVE_CONTEXTS.with(|v| v.borrow_mut().retain(|&key| key != self.key));
    }
}
pub struct GpuContext {
    backend: backend::Context,
    info: GpuAdapterInfo,
    limits: crate::runtime::gpu::GpuLimits,
    failure: Arc<Mutex<Option<RuntimeError>>>,
    ledger: Arc<Ledger>,
    in_flight: Mutex<usize>,
    available: Arc<Condvar>,
    #[cfg(test)]
    waiting: AtomicUsize,
    options: GpuContextOptions,
    pipelines: Mutex<Option<Arc<pipelines::Pipelines>>>,
    resident_pipelines: Mutex<std::collections::HashMap<KernelKey, Arc<resident::Kernel>>>,
    normalization: Mutex<std::sync::Weak<resident::Normalization>>,
    weights: Mutex<std::collections::HashMap<(usize, GpuPrecision), std::sync::Weak<GpuWeights>>>,
}
impl GpuContext {
    pub fn shared_default() -> RuntimeResult<Arc<Self>> {
        static CONTEXT: std::sync::OnceLock<RuntimeResult<Arc<GpuContext>>> =
            std::sync::OnceLock::new();
        CONTEXT.get_or_init(Self::new).clone()
    }
    pub fn new() -> RuntimeResult<Arc<Self>> {
        Self::new_with_options(GpuContextOptions::default())
    }
    pub fn new_with_options(options: GpuContextOptions) -> RuntimeResult<Arc<Self>> {
        if options.max_in_flight == 0
            || !(1..=16).contains(&options.recognition_wave_size)
            || !(1..=16).contains(&options.classification_batch_size)
            || options.max_device_bytes == Some(0)
        {
            return Err(RuntimeError::with_kind(
                ErrorKind::InvalidInput,
                "GPU limits must be positive; recognition_wave_size and classification_batch_size must be within 1..=16",
            ));
        }
        let failure = Arc::new(Mutex::new(None));
        let available = Arc::new(Condvar::new());
        let (backend, info, limits) = backend::Context::new(failure.clone(), available.clone())?;
        let ledger = Arc::new(Ledger {
            owner_device: backend.owner.clone(),
            used: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            limit: options.max_device_bytes,
        });
        Ok(Arc::new(Self {
            backend,
            info,
            limits,
            failure,
            ledger,
            in_flight: Mutex::new(0),
            available,
            #[cfg(test)]
            waiting: AtomicUsize::new(0),
            options,
            pipelines: Mutex::new(None),
            resident_pipelines: Mutex::new(std::collections::HashMap::new()),
            normalization: Mutex::new(std::sync::Weak::new()),
            weights: Mutex::new(std::collections::HashMap::new()),
        }))
    }
    #[cfg(test)]
    pub(crate) fn inject_failure_for_test(&self, kind: ErrorKind) {
        *self.failure.lock().unwrap() =
            Some(RuntimeError::with_kind(kind, "injected resident failure"));
        self.available.notify_all();
    }
    pub fn adapter_info(&self) -> &GpuAdapterInfo {
        &self.info
    }
    pub fn options(&self) -> GpuContextOptions {
        self.options
    }
    /// Limits used by schedule validation and buffer allocation on this device.
    pub fn limits(&self) -> &crate::runtime::gpu::GpuLimits {
        &self.limits
    }
    pub fn memory_statistics(&self) -> GpuDeviceMemoryStatistics {
        self.backend.collect_garbage();
        GpuDeviceMemoryStatistics {
            used_bytes: self.ledger.used.load(Ordering::Acquire),
            peak_bytes: self.ledger.peak.load(Ordering::Acquire),
            committed_bytes: self.backend.owner.committed_bytes(),
            peak_committed_bytes: self.backend.owner.peak_committed_bytes(),
        }
    }
    fn check_alive(&self) -> RuntimeResult<()> {
        let value = self
            .failure
            .lock()
            .map_err(|_| RuntimeError::new("GPU error state poisoned"))?;
        match value.as_ref() {
            Some(error) => Err(error.clone()),
            None => Ok(()),
        }
    }
    pub(crate) fn acquire_gate(&self) -> RuntimeResult<Gate<'_>> {
        self.check_alive()?;
        let key = self as *const Self as usize;
        if ACTIVE_CONTEXTS.with(|v| v.borrow().contains(&key)) {
            return Err(RuntimeError::new(
                "reentrant GPU execution on the same context is unsupported",
            ));
        }
        let mut count = self
            .in_flight
            .lock()
            .map_err(|_| RuntimeError::new("GPU admission state poisoned"))?;
        while *count >= self.options.max_in_flight {
            #[cfg(test)]
            self.waiting.fetch_add(1, Ordering::AcqRel);
            // Device failures originate outside the admission mutex. A bounded
            // wait also observes a failure notification racing the check above.
            let waited = self
                .available
                .wait_timeout(count, std::time::Duration::from_millis(100));
            #[cfg(test)]
            self.waiting.fetch_sub(1, Ordering::AcqRel);
            count = waited
                .map_err(|_| RuntimeError::new("GPU admission state poisoned"))?
                .0;
            self.check_alive()?;
        }
        self.check_alive()?;
        *count += 1;
        ACTIVE_CONTEXTS.with(|v| v.borrow_mut().push(key));
        Ok(Gate {
            context: self,
            key,
            _thread: std::marker::PhantomData,
        })
    }
    fn reserve(&self, bytes: usize) -> RuntimeResult<Arc<Reservation>> {
        self.check_alive()?;
        self.backend.collect_garbage();
        self.ledger.reserve(bytes)
    }
    fn create_buffer(
        &self,
        label: &str,
        size: u64,
        usage: BufferUsage,
    ) -> RuntimeResult<GpuBuffer> {
        if size == 0 || size > self.limits.max_buffer_size {
            return Err(resource("GPU allocation exceeds device buffer limit"));
        }
        let reservation = self.reserve(
            usize::try_from(size).map_err(|_| resource("GPU buffer exceeds host index range"))?,
        )?;
        let buffer = self.backend.allocate(label, size, usage)?;
        Ok(GpuBuffer {
            buffer,
            reservation,
        })
    }
    fn create_buffer_init(
        &self,
        label: &str,
        contents: &[u8],
        usage: BufferUsage,
    ) -> RuntimeResult<GpuBuffer> {
        let size = contents.len().max(4).div_ceil(4) * 4;
        if size as u64 > self.limits.max_buffer_size {
            return Err(resource("GPU allocation exceeds device buffer limit"));
        }
        let reservation = self.reserve(size)?;
        let buffer = self.backend.allocate(label, size as u64, usage)?;
        let buffer = GpuBuffer {
            buffer,
            reservation,
        };
        let mut padded = vec![0; size];
        padded[..contents.len()].copy_from_slice(contents);
        self.upload(&buffer, 0, &padded)?;
        Ok(buffer)
    }
    pub(super) fn upload(
        &self,
        buffer: &GpuBuffer,
        offset: u64,
        bytes: &[u8],
    ) -> RuntimeResult<()> {
        self.check_alive()?;
        if !Arc::ptr_eq(&buffer.reservation.ledger, &self.ledger) {
            return Err(RuntimeError::with_kind(
                ErrorKind::InvalidInput,
                "foreign GPU upload",
            ));
        }
        self.backend.upload(buffer, offset, bytes)
    }
    pub(super) fn checked<T>(
        &self,
        stage: &str,
        operation: impl FnOnce() -> RuntimeResult<T>,
    ) -> RuntimeResult<T> {
        self.check_alive()?;
        let result = self.backend.checked(stage, operation);
        self.check_alive()?;
        result
    }
}
struct GpuWeights {
    _normalization: Mutex<Option<Arc<resident::Normalization>>>,
    _model: Arc<Model>,
    constants: Vec<Option<GpuBuffer>>,
    // One immutable pack per constant/format, shared by every request.
    // Buffers retain the normal context reservation and device-quota accounting.
    packed: Mutex<std::collections::HashMap<(TensorId, WeightPacking), GpuBuffer>>,
    dummy: GpuBuffer,
    bytes: usize,
}
impl GpuWeights {
    fn packed_bytes(&self) -> usize {
        self.packed
            .lock()
            .map(|weights| weights.values().map(GpuBuffer::bytes).sum())
            .unwrap_or(0)
    }
}
pub struct GpuCompiledModel {
    precision: GpuPrecision,
    matrix_precision: GpuMatrixPrecision,
    full_precision: Mutex<Option<Arc<GpuCompiledModel>>>,
    heads: Mutex<std::collections::HashMap<Vec<u32>, Arc<heads::Head>>>,
    fusion: bool,
    pub(super) program: Arc<super::LogicalGraph>,
    context: Arc<GpuContext>,
    weights: Arc<GpuWeights>,
    pipelines: Arc<pipelines::Pipelines>,
}
impl GpuCompiledModel {
    fn packed_weight_bytes(&self) -> usize {
        self.unique_packed_weight_bytes(&mut std::collections::HashSet::new())
    }
    fn unique_packed_weight_bytes(&self, seen: &mut std::collections::HashSet<usize>) -> usize {
        let mut bytes = if seen.insert(Arc::as_ptr(&self.weights) as usize) {
            self.weights.packed_bytes()
        } else {
            0
        };
        if let Ok(compiled) = self.full_precision.lock()
            && let Some(compiled) = &*compiled
        {
            bytes += compiled.unique_packed_weight_bytes(seen);
        }
        bytes
    }
    /// Cache an FP32-storage sibling without changing matrix operand policy.
    /// Strict resident stages use this to preserve their storage contract.
    pub(crate) fn with_full_precision(self: &Arc<Self>) -> RuntimeResult<Arc<Self>> {
        if self.precision == GpuPrecision::F32 {
            return Ok(self.clone());
        }
        let mut cached = self
            .full_precision
            .lock()
            .map_err(|_| RuntimeError::new("GPU precision cache poisoned"))?;
        if let Some(compiled) = &*cached {
            return Ok(compiled.clone());
        }
        let compiled = Arc::new(Self::from_graph_precision(
            self.program.clone(),
            self.context.clone(),
            self.fusion,
            GpuPrecision::F32,
            self.matrix_precision,
        )?);
        *cached = Some(compiled.clone());
        Ok(compiled)
    }
    pub(crate) fn prepare_images(&self) -> RuntimeResult<()> {
        let mut constants = self
            .weights
            ._normalization
            .lock()
            .map_err(|_| RuntimeError::new("image constants cache poisoned"))?;
        if constants.is_none() {
            *constants = Some(resident::Normalization::shared(&self.context)?);
        }
        Ok(())
    }
    pub fn support_report(model: &Model) -> GpuSupportReport {
        let mut issues = Vec::new();
        for (index, node) in model.graph().nodes.iter().enumerate() {
            if let Err(error) = super::compile_node(model, index, node) {
                issues.push(GpuSupportIssue {
                    node_index: index,
                    node_name: node.name.clone(),
                    operator: node.operator.name().into(),
                    source_indices: node.source_indices.clone(),
                    reason: error.to_string(),
                });
            }
        }
        GpuSupportReport { issues }
    }
    pub fn compile(model: Arc<Model>, context: Arc<GpuContext>) -> RuntimeResult<Self> {
        Self::compile_with_options(model, context, super::CompileOptions::default())
    }
    pub fn compile_with_options(
        model: Arc<Model>,
        context: Arc<GpuContext>,
        options: super::CompileOptions,
    ) -> RuntimeResult<Self> {
        Self::from_graph(
            Arc::new(super::LogicalGraph::compile(model)?),
            context,
            options,
        )
    }
    pub fn from_graph(
        program: Arc<super::LogicalGraph>,
        context: Arc<GpuContext>,
        options: super::CompileOptions,
    ) -> RuntimeResult<Self> {
        Self::from_graph_precision(
            program,
            context,
            options.fusion,
            options.gpu.precision,
            options.gpu.matrix_precision,
        )
    }
    fn from_graph_precision(
        program: Arc<super::LogicalGraph>,
        context: Arc<GpuContext>,
        fusion: bool,
        precision: GpuPrecision,
        matrix_precision: GpuMatrixPrecision,
    ) -> RuntimeResult<Self> {
        if precision == GpuPrecision::MixedF16 && !context.backend.supports_half() {
            return Err(RuntimeError::with_kind(
                ErrorKind::Unsupported,
                "MixedF16 requires native half storage support",
            ));
        }
        let model = &program.model;
        validate_supported(&program)?;
        for tensor in &program.model.graph().tensors {
            if let Some(TensorData::F32(values)) = &tensor.constant {
                checked_storage_bytes(values.shape().element_count(), &context.limits)?;
            }
        }
        let pipelines = pipelines::Pipelines::shared(&context)?;
        let weights = {
            let mut cache = context
                .weights
                .lock()
                .map_err(|_| RuntimeError::new("GPU weight cache poisoned"))?;
            cache.retain(|_, value| value.strong_count() > 0);
            let key = (Arc::as_ptr(model) as usize, precision);
            if let Some(weights) = cache.get(&key).and_then(std::sync::Weak::upgrade) {
                weights
            } else {
                let (constants, dummy) = context.checked("GPU model compilation", || {
                    let constants = program
                        .model
                        .graph()
                        .tensors
                        .iter()
                        .map(|tensor| match &tensor.constant {
                            Some(TensorData::F32(values)) => context
                                .create_buffer_init(
                                    "GPU constant",
                                    &precision.encode(values.data()),
                                    BufferUsage::STORAGE | BufferUsage::COPY_SRC,
                                )
                                .map(Some),
                            _ => Ok(None),
                        })
                        .collect::<RuntimeResult<Vec<_>>>()?;
                    let dummy = context.create_buffer_init(
                        "GPU absent operand",
                        &[0; 4],
                        BufferUsage::STORAGE,
                    )?;
                    Ok((constants, dummy))
                })?;
                let weight_bytes = constants
                    .iter()
                    .flatten()
                    .map(GpuBuffer::bytes)
                    .sum::<usize>()
                    + dummy.bytes();
                let weights = Arc::new(GpuWeights {
                    _normalization: Mutex::new(None),
                    _model: model.clone(),
                    constants,
                    packed: Mutex::new(std::collections::HashMap::new()),
                    dummy,
                    bytes: weight_bytes,
                });
                cache.insert(key, Arc::downgrade(&weights));
                weights
            }
        };
        Ok(Self {
            precision,
            matrix_precision,
            full_precision: Mutex::new(None),
            fusion,
            heads: Mutex::new(std::collections::HashMap::new()),
            program,
            context,
            weights,
            pipelines,
        })
    }
    pub(crate) fn unique_weight_bytes(&self, seen: &mut std::collections::HashSet<usize>) -> usize {
        let mut bytes = 0;
        if seen.insert(Arc::as_ptr(&self.weights) as usize) {
            bytes += self.weights.bytes;
            bytes += self.weights.packed_bytes();
        }
        if let Ok(normalization) = self.weights._normalization.lock()
            && let Some(n) = &*normalization
            && seen.insert(Arc::as_ptr(n) as usize)
        {
            bytes += n.bytes();
        }
        if let Ok(heads) = self.heads.lock() {
            for head in heads.values() {
                if seen.insert(Arc::as_ptr(head) as usize) {
                    bytes += head.retained_bytes();
                }
            }
        }
        if let Ok(full) = self.full_precision.lock()
            && let Some(full) = &*full
        {
            bytes += full.unique_weight_bytes(seen);
        }
        bytes
    }
    pub fn weight_bytes(&self) -> usize {
        self.unique_weight_bytes(&mut std::collections::HashSet::new())
    }
    fn original_weight_bytes(&self) -> usize {
        self.weight_bytes() - self.packed_weight_bytes()
    }
    pub(crate) fn model_arc(&self) -> Arc<Model> {
        self.program.model.clone()
    }
    pub fn model(&self) -> &Model {
        &self.program.model
    }
    pub fn context(&self) -> &Arc<GpuContext> {
        &self.context
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuMemoryStatistics {
    pub device_activation_bytes: usize,
    pub readback_bytes: usize,
    pub host_output_bytes: usize,
    pub host_input_bytes: usize,
    pub diagnostic_bytes: usize,
    pub transition_peak_bytes: usize,
    pub parameter_bytes: usize,
    pub host_parameter_bytes: usize,
    pub upload_bytes: usize,
}
impl GpuMemoryStatistics {
    pub fn total_bytes(&self) -> usize {
        self.device_activation_bytes
            .saturating_add(self.readback_bytes)
            .saturating_add(self.host_output_bytes)
            .saturating_add(self.host_input_bytes)
            .saturating_add(self.diagnostic_bytes)
            .saturating_add(self.parameter_bytes)
            .saturating_add(self.host_parameter_bytes)
            .saturating_add(self.upload_bytes)
    }
}
#[derive(Clone)]
struct Dispatch {
    node: usize,
    nodes: Vec<usize>,
    output: TensorId,
    command: Command,
}
#[derive(Clone)]
enum Command {
    Compute {
        shader: operators::ShaderKind,
        native: Arc<executor::NativeDispatch>,
    },
    Copy {
        source: TensorId,
        output: TensorId,
        source_offset: u64,
        destination_offset: u64,
        bytes: u64,
    },
}
#[derive(Clone)]
struct CachedPlan {
    descriptors: Vec<GpuTensorDescriptor>,
    shape: Shape,
    plan: ShapePlan,
    recipes: Vec<operators::Recipe>,
    bindings: Option<(u64, Vec<Dispatch>)>,
    // CPU packets are immutable; all shapes still share the session's one GPU
    // parameter arena. No cached plan owns additional device allocations.
    parameter_data: Option<Arc<[u8]>>,
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum TransferMode {
    Host,
    Resident,
}
#[derive(Default)]
struct HostTransfers {
    readback: Option<GpuBuffer>,
    input: Vec<f32>,
    upload: Vec<u8>,
    output: Vec<f32>,
}
pub struct GpuSession {
    transfer_mode: TransferMode,
    submissions: Arc<AtomicUsize>,
    uploaded_bytes: u64,
    output_readback_bytes: u64,
    diagnostic_readback_bytes: u64,
    readbacks: u64,
    diagnostic_readbacks: u64,
    descriptors: Vec<GpuTensorDescriptor>,
    // Drop bindings before tracked buffers so device accounting follows ownership.
    dispatches: Vec<Dispatch>,
    parameters: GpuBuffer,
    parameter_data: Arc<[u8]>,
    parameters_dirty: bool,
    limits: ExecutionLimits,
    plan: ShapePlan,
    buffers: Vec<GpuBuffer>,
    transfers: HostTransfers,
    memory: GpuMemoryStatistics,
    execution_timings: GpuExecutionTimings,
    gpu_profile: Option<GpuExecutionProfile>,
    optimized: bool,
    node_count: usize,
    selected_output: TensorId,
    generation: u64,
    recipes: Vec<operators::Recipe>,
    cache: VecDeque<CachedPlan>,
    cache_hits: u64,
    cache_misses: u64,
    diagnostic: Option<Box<GpuSession>>,
    trace_readback: Option<GpuBuffer>,
    trace_output: Vec<f32>,
    compiled: Arc<GpuCompiledModel>,
}
impl GpuSession {
    pub fn new(
        compiled: Arc<GpuCompiledModel>,
        dimensions: &[usize],
        limits: ExecutionLimits,
    ) -> RuntimeResult<Self> {
        let count = compiled.program.instructions.len();
        let output = compiled.program.output;
        let fusion = compiled.fusion;
        Self::new_selected(
            compiled,
            dimensions,
            limits,
            count,
            output,
            fusion,
            TransferMode::Host,
        )
    }
    /// Build a semantic node schedule for tracing on the native executor.
    pub(crate) fn new_diagnostic(
        compiled: Arc<GpuCompiledModel>,
        dimensions: &[usize],
        limits: ExecutionLimits,
    ) -> RuntimeResult<Self> {
        let count = compiled.program.instructions.len();
        let output = compiled.program.output;
        Self::new_selected(
            compiled,
            dimensions,
            limits,
            count,
            output,
            false,
            TransferMode::Host,
        )
    }
    pub(crate) fn new_prefix(
        compiled: Arc<GpuCompiledModel>,
        dimensions: &[usize],
        limits: ExecutionLimits,
        last_node: usize,
    ) -> RuntimeResult<Self> {
        let output = compiled
            .program
            .instructions
            .get(last_node)
            .ok_or_else(|| RuntimeError::new("GPU prefix node out of range"))?
            .output;
        let fusion = compiled.fusion;
        Self::new_selected(
            compiled,
            dimensions,
            limits,
            last_node + 1,
            output,
            fusion,
            TransferMode::Host,
        )
    }

    pub(crate) fn new_resident(
        compiled: Arc<GpuCompiledModel>,
        dimensions: &[usize],
        limits: ExecutionLimits,
        last_node: Option<usize>,
    ) -> RuntimeResult<Self> {
        let (count, output) = match last_node {
            Some(last) => (
                last + 1,
                compiled
                    .program
                    .instructions
                    .get(last)
                    .ok_or_else(|| RuntimeError::new("GPU prefix node out of range"))?
                    .output,
            ),
            None => (compiled.program.instructions.len(), compiled.program.output),
        };
        let fusion = compiled.fusion;
        Self::new_selected(
            compiled,
            dimensions,
            limits,
            count,
            output,
            fusion,
            TransferMode::Resident,
        )
    }
    fn new_selected(
        compiled: Arc<GpuCompiledModel>,
        dimensions: &[usize],
        limits: ExecutionLimits,
        node_count: usize,
        selected_output: TensorId,
        optimized: bool,
        transfer_mode: TransferMode,
    ) -> RuntimeResult<Self> {
        let bound = Self::bind(
            &compiled,
            dimensions,
            limits,
            node_count,
            selected_output,
            optimized,
        )?;
        let built = Self::build(
            &compiled,
            &bound,
            limits,
            &[],
            None,
            None,
            (0, 0),
            0,
            transfer_mode,
        )?;
        let mut session = Self {
            transfer_mode,
            submissions: Arc::new(AtomicUsize::new(0)),
            uploaded_bytes: 0,
            output_readback_bytes: 0,
            diagnostic_readback_bytes: 0,
            readbacks: 0,
            diagnostic_readbacks: 0,
            descriptors: bound.descriptors,
            compiled,
            limits,
            plan: bound.plan,
            buffers: built.buffers,
            transfers: built.transfers,
            dispatches: built.dispatches,
            parameters: built.parameters,
            parameter_data: built.parameter_data,
            parameters_dirty: true,
            memory: built.memory,
            execution_timings: GpuExecutionTimings::default(),
            gpu_profile: None,
            node_count,
            optimized,
            selected_output,
            generation: 0,
            recipes: bound.recipes,
            cache: VecDeque::new(),
            cache_hits: 0,
            cache_misses: 1,
            diagnostic: None,
            trace_readback: None,
            trace_output: Vec::new(),
        };
        session.memory.transition_peak_bytes = session.memory.total_bytes();
        Ok(session)
    }
    fn bind(
        compiled: &GpuCompiledModel,
        dimensions: &[usize],
        limits: ExecutionLimits,
        node_count: usize,
        selected_output: TensorId,
        optimized: bool,
    ) -> RuntimeResult<CachedPlan> {
        let mut plan = ShapePlan::empty(compiled.program.model.graph().tensors.len(), node_count);
        let mut scratch = BindScratch::new(&compiled.program);
        bind_into(
            &compiled.program,
            dimensions,
            // The semantic schedule validates shapes; the committed physical
            // schedule and retained capacities determine the GPU memory limit.
            if optimized && limits.max_activation_bytes > 0 {
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
        validate_plan(&compiled.program, &plan, &compiled.context.limits)?;
        // Without fusion, storing every arithmetic boundary in half precision
        // compounds rounding (notably activation polynomials and normalization).
        // Keep the unfused/diagnostic activation graph in F32 while sharing the
        // same immutable FP16 weights. Production fused plans can store FP16.
        // Attention recognizers also retain F32 activations: half-rounding in
        // their residual stream can change close vocabulary decisions even when
        // the final tensor remains inside the mixed-precision error bound.
        let has_attention = plan
            .steps
            .iter()
            .any(|step| matches!(step.kind, BoundKind::BatchedMatMul(_)));
        let activation_precision = if optimized
            && (!has_attention || compiled.matrix_precision == GpuMatrixPrecision::F16)
        {
            compiled.precision
        } else {
            GpuPrecision::F32
        };
        let mut descriptors: Vec<_> = plan
            .shapes
            .iter()
            .enumerate()
            .map(|(id, &shape)| {
                GpuTensorDescriptor::canonical(
                    shape,
                    if compiled.program.model.graph().tensors[id]
                        .constant
                        .is_some()
                    {
                        compiled.precision
                    } else {
                        activation_precision
                    },
                )
            })
            .collect();
        let mut recipes = if optimized {
            physical::lower(
                &compiled.program,
                &mut plan,
                selected_output,
                &compiled.context.limits,
                activation_precision,
                &mut descriptors,
                compiled.matrix_precision,
                &mut |recipe| {
                    compiled
                        .pipelines
                        .supports_matrix(&compiled.context, recipe)
                },
            )?
        } else {
            operators::recipes(&compiled.program, &plan, &compiled.context.limits)?
        };
        let precision = compiled.precision;
        if precision == GpuPrecision::MixedF16 {
            // Copy commands require four-byte offsets. Compute copies preserve
            // odd half-element tails and concatenation placements without races.
            for recipe in &mut recipes {
                if let operators::RecipeKind::Copy {
                    source_offset,
                    destination_offset,
                    bytes,
                } = recipe.kind
                {
                    recipe.words = [0; 128];
                    recipe.words[0] = (bytes / 4) as u32;
                    recipe.words[2] = 22;
                    recipe.words[3] = (source_offset / 4) as u32;
                    recipe.words[4] = (destination_offset / 4) as u32;
                    (recipe.x, recipe.y) =
                        dispatch_grid((bytes / 4) as usize, &compiled.context.limits)?;
                    recipe.words[1] = recipe.x * 64;
                    recipe.kind = operators::RecipeKind::Compute(operators::ShaderKind::General);
                }
            }
        }
        if optimized
            && compiled.matrix_precision == GpuMatrixPrecision::F16
            && compiled.precision == GpuPrecision::MixedF16
        {
            let generic = recipes.clone();
            half_native::select(
                &mut recipes,
                &descriptors,
                &compiled.context.limits,
                compiled.matrix_precision,
            )?;
            for (recipe, original) in recipes.iter_mut().zip(generic) {
                if let operators::RecipeKind::Compute(kind) = recipe.kind
                    && half_native::supports_kind(kind)
                    && !compiled.pipelines.supports_half(&compiled.context, kind)?
                {
                    *recipe = original;
                }
            }
        }
        if optimized {
            small_projection::select(
                &compiled.program,
                &mut recipes,
                &descriptors,
                &compiled.context.limits,
                compiled.matrix_precision,
            )?;
        }
        plan.activation_bytes = plan
            .lengths
            .iter()
            .map(|&n| activation_precision.bytes(n))
            .sum();
        Ok(CachedPlan {
            descriptors,
            shape: plan.input_shape,
            plan,
            recipes,
            bindings: None,
            parameter_data: None,
        })
    }
    #[allow(clippy::too_many_arguments)]
    fn build(
        compiled: &GpuCompiledModel,
        bound: &CachedPlan,
        limits: ExecutionLimits,
        previous: &[GpuBuffer],
        previous_readback: Option<&GpuBuffer>,
        previous_parameters: Option<&GpuBuffer>,
        previous_host: (usize, usize),
        generation: u64,
        transfer_mode: TransferMode,
    ) -> RuntimeResult<Built> {
        let context = &compiled.context;
        let selected = bound
            .plan
            .steps
            .last()
            .map_or(compiled.program.input, |step| step.output);
        let host = transfer_mode == TransferMode::Host;
        let input_elements = if host { bound.shape.element_count() } else { 0 };
        let output_elements = if host {
            bound.plan.shapes[selected.0].element_count()
        } else {
            0
        };
        let precision = bound.descriptors[compiled.program.input.0].precision;
        let output_bytes = bound.descriptors[selected.0]
            .precision
            .bytes(output_elements);
        let host_output_bytes = output_elements * 4;
        let upload_bytes = precision.bytes(input_elements);
        let host_input_bytes = input_elements
            .checked_mul(4)
            .ok_or_else(|| resource("GPU host input size overflow"))?;
        let stride = 512usize.div_ceil(context.limits.min_uniform_buffer_offset_alignment as usize)
            * context.limits.min_uniform_buffer_offset_alignment as usize;
        let parameter_bytes = bound
            .recipes
            .len()
            .max(1)
            .checked_mul(stride)
            .ok_or_else(|| resource("GPU parameter arena overflow"))?;
        if parameter_bytes > u32::MAX as usize {
            return Err(resource("GPU uniform offsets exceed U32"));
        }
        let fixed_bytes = output_bytes
            .checked_add(host_output_bytes)
            .and_then(|n| n.checked_add(host_input_bytes))
            .and_then(|n| n.checked_add(parameter_bytes * 2 + upload_bytes))
            .ok_or_else(|| resource("GPU activation size overflow"))?;
        check_budget(
            bound
                .plan
                .activation_bytes
                .checked_add(fixed_bytes)
                .ok_or_else(|| resource("GPU activation size overflow"))?,
            limits,
        )?;
        let retained =
            bound
                .plan
                .lengths
                .iter()
                .enumerate()
                .try_fold(0usize, |sum, (slot, &length)| {
                    sum.checked_add(previous.get(slot).map_or(precision.bytes(length), |b| {
                        b.bytes().max(precision.bytes(length))
                    }))
                    .ok_or_else(|| resource("GPU activation capacity overflow"))
                })?;
        let retained_readback =
            previous_readback.map_or(output_bytes, |b| b.bytes().max(output_bytes));
        let retained_host_input = host_input_bytes.max(previous_host.0 * 4);
        let retained_host_output = host_output_bytes.max(previous_host.1 * 4);
        let retained_parameters =
            previous_parameters.map_or(parameter_bytes, |p| p.bytes().max(parameter_bytes));
        let reuse = retained
            .checked_add(retained_readback)
            .and_then(|n| n.checked_add(retained_host_input))
            .and_then(|n| n.checked_add(retained_host_output))
            .and_then(|n| n.checked_add(retained_parameters + parameter_bytes + upload_bytes))
            .is_some_and(|n| n <= limits.max_activation_bytes);
        context.checked("GPU session binding", || {
            let mut buffers = Vec::new();
            for (slot, &length) in bound.plan.lengths.iter().enumerate() {
                let bytes = precision.bytes(length) as u64;
                buffers.push(
                    if reuse
                        && previous
                            .get(slot)
                            .is_some_and(|buffer| buffer.size() >= bytes)
                    {
                        previous[slot].clone()
                    } else {
                        context.create_buffer(
                            "GPU activation slot",
                            bytes,
                            BufferUsage::STORAGE | BufferUsage::COPY_DST | BufferUsage::COPY_SRC,
                        )?
                    },
                );
            }
            let readback = if !host {
                None
            } else {
                Some(
                    if reuse && previous_readback.is_some_and(|b| b.size() >= output_bytes as u64) {
                        previous_readback.unwrap().clone()
                    } else {
                        context.create_buffer(
                            "GPU output readback",
                            output_bytes as u64,
                            BufferUsage::MAP_READ | BufferUsage::COPY_DST,
                        )?
                    },
                )
            };
            let input = if reuse && previous_host.0 >= input_elements {
                Vec::new()
            } else {
                allocate_f32(input_elements)?
            };
            let output = if reuse && previous_host.1 >= output_elements {
                Vec::new()
            } else {
                allocate_f32(output_elements)?
            };
            let upload = vec![0; upload_bytes];
            let parameters = if let Some(old) =
                previous_parameters.filter(|b| reuse && b.bytes() >= parameter_bytes)
            {
                old.clone()
            } else {
                context.create_buffer(
                    "GPU parameter arena",
                    parameter_bytes as u64,
                    BufferUsage::UNIFORM | BufferUsage::COPY_DST,
                )?
            };
            let mut parameter_data = vec![0u8; parameter_bytes];
            let compatible = parameters.buffer
                == previous_parameters
                    .map_or(&parameters.buffer, |b| &b.buffer)
                    .clone()
                && buffers.len() == previous.len()
                && buffers
                    .iter()
                    .zip(previous)
                    .all(|(a, b)| a.buffer == b.buffer);
            let cached_bindings = bound
                .bindings
                .as_ref()
                .filter(|(g, _)| *g == generation && compatible);
            let mut dispatches = Vec::new();
            for (recipe_index, recipe) in bound.recipes.iter().enumerate() {
                let shader = match &recipe.kind {
                    operators::RecipeKind::Compute(shader) => *shader,
                    operators::RecipeKind::Copy {
                        source_offset,
                        destination_offset,
                        bytes,
                    } => {
                        dispatches.push(Dispatch {
                            node: recipe.node,
                            nodes: recipe.nodes.clone(),
                            output: recipe.output,
                            command: Command::Copy {
                                source: recipe.inputs[0],
                                output: recipe.output,
                                source_offset: *source_offset,
                                destination_offset: *destination_offset,
                                bytes: *bytes,
                            },
                        });
                        continue;
                    }
                };
                let offset = recipe_index * stride;
                parameter_data[offset..offset + 512]
                    .copy_from_slice(bytemuck::cast_slice(&recipe.words));
                let native = if let Some((_, cached)) = cached_bindings {
                    let Command::Compute { native, .. } = &cached[recipe_index].command else {
                        unreachable!("cached native recipe kind changed")
                    };
                    native.clone()
                } else {
                    let packed = compiled.specialized_weight(recipe)?;
                    let storage = std::array::from_fn(|i| {
                        if i == 1 && compiled.packed_operand_precision(recipe).is_some() {
                            compiled.packed_operand_precision(recipe).unwrap()
                        } else if i == 5 {
                            bound.descriptors[recipe.output.0].precision
                        } else {
                            recipe
                                .inputs
                                .get(i)
                                .map_or(compiled.precision, |id| bound.descriptors[id.0].precision)
                        }
                    });
                    let pipeline = compiled.pipelines.get_for_recipe(
                        context,
                        shader,
                        storage,
                        &recipe.words,
                    )?;
                    let mut owners: Vec<_> = (0..6)
                        .map(|i| {
                            if i == 1
                                && let Some(packed) = &packed
                            {
                                return packed.clone();
                            }
                            let id = if i == 5 {
                                Some(recipe.output)
                            } else {
                                recipe.inputs.get(i).copied()
                            };
                            id.map_or_else(
                                || compiled.weights.dummy.clone(),
                                |id| {
                                    compiled
                                        .weights
                                        .constants
                                        .get(id.0)
                                        .and_then(Option::as_ref)
                                        .unwrap_or_else(|| &buffers[bound.plan.slots[id.0]])
                                        .clone()
                                },
                            )
                        })
                        .collect();
                    owners.push(parameters.clone());
                    executor::NativeDispatch::prepare(
                        pipeline,
                        owners,
                        vec![0, 0, 0, 0, 0, 0, offset as u64],
                        [recipe.x, recipe.y, 1],
                        &[true, true, true, true, true, false],
                    )?
                };
                dispatches.push(Dispatch {
                    node: recipe.node,
                    nodes: recipe.nodes.clone(),
                    output: recipe.output,
                    command: Command::Compute { shader, native },
                });
            }
            let memory = GpuMemoryStatistics {
                parameter_bytes: parameters.bytes(),
                host_parameter_bytes: parameter_data.capacity(),
                upload_bytes: upload.capacity(),
                device_activation_bytes: buffers.iter().map(GpuBuffer::bytes).sum(),
                readback_bytes: readback.as_ref().map_or(0, GpuBuffer::bytes),
                host_input_bytes: if input.capacity() == 0 {
                    previous_host.0 * 4
                } else {
                    input.capacity() * 4
                },
                host_output_bytes: if output.capacity() == 0 {
                    previous_host.1 * 4
                } else {
                    output.capacity() * 4
                },
                ..Default::default()
            };
            check_budget(memory.total_bytes(), limits)?;
            Ok(Built {
                buffers,
                transfers: HostTransfers {
                    readback,
                    input,
                    upload,
                    output,
                },
                dispatches,
                parameters,
                parameter_data: parameter_data.into(),
                memory,
            })
        })
    }
    pub fn reshape(&mut self, dimensions: &[usize]) -> RuntimeResult<()> {
        self.compiled.context.check_alive()?;
        if dimensions == self.plan.input_shape.dimensions()
            && self.memory.total_bytes() <= self.limits.max_activation_bytes
        {
            self.cache_hits += 1;
            return Ok(());
        }
        let cached = self
            .cache
            .iter()
            .position(|entry| entry.shape.dimensions() == dimensions);
        if let Some(index) = cached
            && self.restore_resident_cached(index)
        {
            self.cache_hits += 1;
            return Ok(());
        }
        let mut bound = if let Some(index) = cached {
            self.cache[index].clone()
        } else {
            Self::bind(
                &self.compiled,
                dimensions,
                self.limits,
                self.node_count,
                self.selected_output,
                self.optimized,
            )?
        };
        let binding_limits = ExecutionLimits {
            max_activation_bytes: self
                .limits
                .max_activation_bytes
                .checked_sub(self.memory.diagnostic_bytes)
                .ok_or_else(|| resource("GPU diagnostics exceed activation limit"))?,
        };
        let mut built = Self::build(
            &self.compiled,
            &bound,
            binding_limits,
            &self.buffers,
            self.transfers.readback.as_ref(),
            Some(&self.parameters),
            (
                self.transfers.input.capacity(),
                self.transfers.output.capacity(),
            ),
            self.generation,
            self.transfer_mode,
        )?;
        check_budget(
            built.memory.total_bytes() + self.memory.diagnostic_bytes,
            self.limits,
        )?;
        let mut old = CachedPlan {
            descriptors: self.descriptors.clone(),
            shape: self.plan.input_shape,
            plan: self.plan.clone(),
            recipes: self.recipes.clone(),
            bindings: Some((self.generation, self.dispatches.clone())),
            parameter_data: (self.transfer_mode == TransferMode::Resident)
                .then(|| self.parameter_data.clone()),
        };
        let mut shared = built
            .buffers
            .iter()
            .filter(|b| {
                self.buffers
                    .iter()
                    .any(|old| Arc::ptr_eq(&b.reservation, &old.reservation))
            })
            .map(GpuBuffer::bytes)
            .sum::<usize>()
            + built
                .transfers
                .readback
                .as_ref()
                .zip(self.transfers.readback.as_ref())
                .filter(|(a, b)| Arc::ptr_eq(&a.reservation, &b.reservation))
                .map_or(0, |(a, _)| a.bytes());
        if self.parameters.buffer == built.parameters.buffer {
            shared += self.parameters.bytes();
        }
        if built.transfers.input.capacity() == 0 {
            shared += self.transfers.input.capacity() * 4;
        }
        if built.transfers.output.capacity() == 0 {
            shared += self.transfers.output.capacity() * 4;
        }

        let peak = self
            .memory
            .total_bytes()
            .saturating_add(built.memory.total_bytes())
            .saturating_sub(shared);
        let compatible = self.parameters.buffer == built.parameters.buffer
            && self.buffers.len() == built.buffers.len()
            && self
                .buffers
                .iter()
                .zip(&built.buffers)
                .all(|(a, b)| a.buffer == b.buffer);
        if !compatible {
            self.generation = self.generation.wrapping_add(1);
            for entry in &mut self.cache {
                entry.bindings = None;
            }
            old.bindings = None;
            bound.bindings = None;
        }
        if self.transfer_mode == TransferMode::Host && built.transfers.input.capacity() == 0 {
            std::mem::swap(&mut built.transfers.input, &mut self.transfers.input);
            built
                .transfers
                .input
                .resize(bound.shape.element_count(), 0.0);
        }
        if self.transfer_mode == TransferMode::Host && built.transfers.output.capacity() == 0 {
            std::mem::swap(&mut built.transfers.output, &mut self.transfers.output);
            built.transfers.output.resize(
                bound.plan.shapes[self.selected_output.0].element_count(),
                0.0,
            );
        }

        if let Some(index) = cached {
            self.cache.remove(index);
            self.cache_hits += 1;
        } else {
            self.cache_misses += 1;
        }
        self.cache.push_front(old);
        self.cache.truncate(PLAN_CACHE_CAPACITY);
        // Release old bindings before reservations for replaced activation buffers.
        drop(std::mem::replace(&mut self.dispatches, built.dispatches));
        self.parameters = built.parameters;
        self.parameter_data = built.parameter_data;
        self.parameters_dirty = true;
        self.recipes = bound.recipes;
        self.descriptors = bound.descriptors;
        self.plan = bound.plan;
        self.buffers = built.buffers;
        self.transfers = built.transfers;
        let diagnostic = self.memory.diagnostic_bytes;
        let previous_peak = self.memory.transition_peak_bytes;
        self.memory = built.memory;
        self.memory.diagnostic_bytes = diagnostic;
        self.memory.transition_peak_bytes = previous_peak.max(peak);
        Ok(())
    }

    /// Reuse only a fully bound resident view of these exact allocations.
    /// Moving cached metadata avoids rebuilding/serializing every recipe; the
    /// next graph still writes its packets through the normal queue ordering.
    fn restore_resident_cached(&mut self, index: usize) -> bool {
        if self.transfer_mode != TransferMode::Resident {
            return false;
        }
        let entry = &self.cache[index];
        let Some((generation, _)) = &entry.bindings else {
            return false;
        };
        let Some(parameter_data) = &entry.parameter_data else {
            return false;
        };
        if *generation != self.generation
            || parameter_data.len() > self.parameters.bytes()
            || entry.plan.lengths.len() != self.buffers.len()
        {
            return false;
        }
        let precision = entry.descriptors[self.compiled.program.input.0].precision;
        if !entry
            .plan
            .lengths
            .iter()
            .zip(&self.buffers)
            .all(|(&length, buffer)| precision.bytes(length) <= buffer.bytes())
        {
            return false;
        }
        let Some(memory) = resident_cached_memory(
            self.memory,
            parameter_data.len(),
            self.limits.max_activation_bytes,
        ) else {
            return false;
        };

        // All fallible checks precede removal. Generation certifies that the
        // cached native snapshots retain the current activation/uniform handles.
        let mut entry = self.cache.remove(index).unwrap();
        let old = CachedPlan {
            shape: self.plan.input_shape,
            plan: std::mem::replace(&mut self.plan, entry.plan),
            recipes: std::mem::replace(&mut self.recipes, entry.recipes),
            descriptors: std::mem::replace(&mut self.descriptors, entry.descriptors),
            bindings: Some((
                self.generation,
                std::mem::replace(&mut self.dispatches, entry.bindings.take().unwrap().1),
            )),
            parameter_data: Some(std::mem::replace(
                &mut self.parameter_data,
                entry.parameter_data.take().unwrap(),
            )),
        };
        self.cache.push_front(old);
        self.cache.truncate(PLAN_CACHE_CAPACITY);
        self.parameters_dirty = true;
        self.memory = memory;
        true
    }
    pub(crate) fn reshape_with_limits(
        &mut self,
        dimensions: &[usize],
        limits: ExecutionLimits,
    ) -> RuntimeResult<()> {
        let previous = self.limits;
        self.limits = limits;
        let result = self.reshape(dimensions);
        if result.is_err() {
            self.limits = previous;
        }
        result
    }
    pub(crate) fn input_data(&self) -> &[f32] {
        &self.transfers.input
    }
    pub fn input_shape(&self) -> Shape {
        self.plan.input_shape
    }
    pub fn output_shape(&self) -> Shape {
        self.plan.shapes[self.selected_output.0]
    }
    pub fn tensor_descriptors(&self) -> &[GpuTensorDescriptor] {
        &self.descriptors
    }
    fn input_precision(&self) -> GpuPrecision {
        self.descriptors[self.compiled.program.input.0].precision
    }
    fn output_precision(&self) -> GpuPrecision {
        self.descriptors[self.selected_output.0].precision
    }
    pub(crate) fn output_view(&self) -> RuntimeResult<TensorView<'_, f32>> {
        TensorView::from_slice(self.output_shape(), &self.transfers.output)
            .map_err(RuntimeError::from)
    }
    pub fn execution_limits(&self) -> ExecutionLimits {
        self.limits
    }
    pub fn activation_bytes(&self) -> usize {
        self.memory.total_bytes()
    }
    pub fn retained_activation_bytes(&self) -> usize {
        self.memory.total_bytes()
    }
    pub fn memory_statistics(&self) -> GpuMemoryStatistics {
        self.memory
    }
    pub(crate) fn gpu_execution_timings(&self) -> GpuExecutionTimings {
        self.execution_timings
    }
    pub fn retained_device_bytes(&self) -> usize {
        self.buffers.iter().map(GpuBuffer::bytes).sum::<usize>()
            + self.transfers.readback.as_ref().map_or(0, GpuBuffer::bytes)
            + self.parameters.bytes()
            + self
                .diagnostic
                .as_ref()
                .map_or(0, |d| d.retained_device_bytes())
            + self.trace_readback.as_ref().map_or(0, GpuBuffer::bytes)
    }
    pub fn retained_planning_bytes(&self) -> usize {
        self.gpu_profile
            .as_ref()
            .map_or(0, GpuExecutionProfile::retained_bytes)
            + self
                .diagnostic
                .as_ref()
                .map_or(0, |d| d.retained_planning_bytes())
            + dispatch_metadata_bytes(&self.dispatches)
            + descriptor_metadata_bytes(&self.descriptors)
            + recipe_metadata_bytes(&self.recipes)
            + self
                .cache
                .iter()
                .map(|entry| {
                    plan_metadata_bytes(&entry.plan)
                        + recipe_metadata_bytes(&entry.recipes)
                        + descriptor_metadata_bytes(&entry.descriptors)
                        + entry
                            .bindings
                            .as_ref()
                            .map_or(0, |(_, d)| dispatch_metadata_bytes(d))
                })
                .sum::<usize>()
            + plan_metadata_bytes(&self.plan)
            + self.buffers.capacity() * std::mem::size_of::<GpuBuffer>()
            + self.cache.capacity() * std::mem::size_of::<CachedPlan>()
            + self.cached_parameter_bytes()
            + self.prepared_native_bytes()
    }
    fn prepared_native_bytes(&self) -> usize {
        let mut seen = std::collections::HashSet::new();
        self.dispatches
            .iter()
            .chain(
                self.cache
                    .iter()
                    .filter_map(|entry| entry.bindings.as_ref())
                    .flat_map(|(_, dispatches)| dispatches),
            )
            .filter_map(|dispatch| match &dispatch.command {
                Command::Compute { native, .. } => Some(native),
                _ => None,
            })
            .filter(|native| seen.insert(Arc::as_ptr(native)))
            .map(|native| native.retained_bytes())
            .sum()
    }
    fn cached_parameter_bytes(&self) -> usize {
        let mut seen = std::collections::HashSet::new();
        seen.insert(self.parameter_data.as_ptr());
        self.cache
            .iter()
            .filter_map(|entry| entry.parameter_data.as_ref())
            .filter(|bytes| seen.insert(bytes.as_ptr()))
            .map(|bytes| bytes.len())
            .sum()
    }
    pub fn retained_scratch_bytes(&self) -> usize {
        0
    }
    pub(crate) fn counter_snapshot(&self) -> super::StatisticsCounters {
        super::StatisticsCounters {
            production_submissions: self.submissions.load(Ordering::Relaxed) as u64,
            uploaded_bytes: self.uploaded_bytes,
            output_readback_bytes: self.output_readback_bytes,
            diagnostic_readback_bytes: self.diagnostic_readback_bytes
                + self
                    .diagnostic
                    .as_ref()
                    .map_or(0, |d| d.diagnostic_readback_bytes),
            production_readbacks: self.readbacks,
            diagnostic_readbacks: self.diagnostic_readbacks
                + self
                    .diagnostic
                    .as_ref()
                    .map_or(0, |d| d.diagnostic_readbacks),
            plan_cache_hits: self.cache_hits,
            plan_cache_misses: self.cache_misses,
        }
    }
    pub fn statistics(&self) -> ExecutionStatistics {
        ExecutionStatistics {
            execution_backend: Some(super::ExecutionBackend::Gpu),
            effective_precision: if self.compiled.matrix_precision == GpuMatrixPrecision::F16 {
                format!(
                    "{} storage; F16 matrix operands/F32 accumulation",
                    if self.compiled.precision == GpuPrecision::MixedF16 {
                        "F16"
                    } else {
                        "F32"
                    }
                )
            } else if self.compiled.precision == GpuPrecision::MixedF16
                && self.input_precision() == GpuPrecision::F32
            {
                "MixedF16 (FP32 activations)".into()
            } else {
                format!("{:?}", self.compiled.precision)
            },
            adapter: Some(self.compiled.context.info.name.clone()),
            production_submissions: self.submissions.load(Ordering::Relaxed) as u64,
            uploaded_bytes: self.uploaded_bytes,
            output_readback_bytes: self.output_readback_bytes,
            diagnostic_readback_bytes: self.diagnostic_readback_bytes
                + self
                    .diagnostic
                    .as_ref()
                    .map_or(0, |d| d.diagnostic_readback_bytes),
            production_readbacks: self.readbacks,
            diagnostic_readbacks: self.diagnostic_readbacks
                + self
                    .diagnostic
                    .as_ref()
                    .map_or(0, |d| d.diagnostic_readbacks),
            nhwc_values: self
                .descriptors
                .iter()
                .filter(|d| matches!(d.layout, GpuLayout::PaddedNhwc { .. }))
                .count(),
            layout_conversions: self
                .recipes
                .iter()
                .filter(|r| {
                    matches!(
                        r.kind,
                        operators::RecipeKind::Compute(operators::ShaderKind::General)
                    ) && r.words[2] == 24
                        || matches!(
                            r.kind,
                            operators::RecipeKind::Compute(operators::ShaderKind::PackNhwc)
                        )
                })
                .count(),
            original_weight_bytes: self.compiled.original_weight_bytes() as u64,
            packed_weight_bytes: self.compiled.packed_weight_bytes(),
            normalized_nodes: self.node_count,
            selected_kernels: super::kernel_selections(self.recipes.iter().filter_map(|r| {
                if let operators::RecipeKind::Compute(kind) = r.kind {
                    Some((
                        kind.name(),
                        if kind == operators::ShaderKind::PointwiseF16
                            || conv_mpp::Kernel::from_shader(kind).is_some()
                        {
                            "nhwc/hwio"
                        } else if conv_mpp_staged::supports_kind(kind) {
                            "nchw/hwio"
                        } else {
                            "nchw/row_major"
                        },
                        self.compiled.packed_operand_precision(r).is_some(),
                    ))
                } else {
                    None
                }
            })),
            physical_steps: self.dispatches.len(),
            fused_nodes: self
                .recipes
                .iter()
                .map(|r| r.nodes.len().saturating_sub(1))
                .sum(),
            store_fused_steps: self.recipes.iter().filter(|r| r.words[64] > 0).count(),
            specialized_kernels: super::specialization_counts(self.recipes.iter().filter_map(
                |r| {
                    if let operators::RecipeKind::Compute(kind) = r.kind {
                        Some((
                            self.compiled.model().graph().nodes[r.nodes[0]].op_type(),
                            kind.name(),
                            format!(
                                "{}_{:?}",
                                format!("{:?}", self.compiled.context.info.backend)
                                    .to_ascii_lowercase(),
                                self.compiled.precision,
                            ),
                            r.words[64] > 0,
                        ))
                    } else {
                        None
                    }
                },
            )),
            activation_bytes: self.activation_bytes(),
            retained_activation_bytes: self.retained_activation_bytes(),
            plan_cache_hits: self.cache_hits,
            plan_cache_misses: self.cache_misses,
            backend: format!(
                "GPU {:?}: {}",
                self.compiled.context.info.backend, self.compiled.context.info.name
            ),
            ..Default::default()
        }
    }
    pub fn stats(&self) -> ExecutionStatistics {
        self.statistics()
    }
    fn resolve(&self, id: TensorId) -> &GpuBuffer {
        self.compiled
            .weights
            .constants
            .get(id.0)
            .and_then(Option::as_ref)
            .unwrap_or_else(|| &self.buffers[self.plan.slots[id.0]])
    }
    pub fn last_gpu_profile(&self) -> Option<&GpuExecutionProfile> {
        self.gpu_profile.as_ref()
    }
    #[cfg(feature = "profiling")]
    pub fn run_profiled(&mut self, input: &[f32]) -> RuntimeResult<TensorView<'_, f32>> {
        if input.len() != self.transfers.input.len() {
            return Err(RuntimeError::with_kind(
                ErrorKind::InvalidInput,
                "GPU input length mismatch",
            ));
        }
        self.run_with_input_profiled::<true>(|output| {
            output.copy_from_slice(input);
            Ok(())
        })
    }
    fn prepare_input(
        &mut self,
        fill: impl FnOnce(&mut [f32]) -> RuntimeResult<()>,
    ) -> RuntimeResult<()> {
        if self.transfer_mode != TransferMode::Host {
            return Err(RuntimeError::with_kind(
                ErrorKind::InvalidInput,
                "resident sessions accept device inputs only",
            ));
        }
        fill(&mut self.transfers.input).map_err(RuntimeError::callback)?;
        if let Some(index) = self.transfers.input.iter().position(|v| !v.is_finite()) {
            return Err(RuntimeError::with_kind(
                ErrorKind::InvalidInput,
                format!("nonfinite input F32 at element {index}"),
            ));
        }
        Ok(())
    }
    pub fn run(&mut self, input: &[f32]) -> RuntimeResult<TensorView<'_, f32>> {
        if input.len() != self.input_shape().element_count() {
            return Err(RuntimeError::with_kind(
                ErrorKind::InvalidInput,
                "GPU input length does not match fixed session shape",
            ));
        }
        self.run_with_input(|destination| {
            destination.copy_from_slice(input);
            Ok(())
        })
    }
    pub fn run_with_input(
        &mut self,
        fill: impl FnOnce(&mut [f32]) -> RuntimeResult<()>,
    ) -> RuntimeResult<TensorView<'_, f32>> {
        self.run_with_input_profiled::<false>(fill)
    }
    pub(crate) fn run_with_input_profiled<const PROFILE: bool>(
        &mut self,
        fill: impl FnOnce(&mut [f32]) -> RuntimeResult<()>,
    ) -> RuntimeResult<TensorView<'_, f32>> {
        self.execution_timings = GpuExecutionTimings::default();
        self.gpu_profile = None;
        self.prepare_input(fill)?;
        self.input_precision()
            .write(&self.transfers.input, &mut self.transfers.upload);
        let context = self.compiled.context.clone();
        let admission = Timer::<PROFILE>::start();
        let gate = context.acquire_gate()?;
        self.execution_timings.admission_wait_ns = admission.elapsed();
        let recording = Timer::<PROFILE>::start();
        context.checked("GPU input upload", || {
            context.upload(
                self.resolve(self.compiled.program.input),
                0,
                &self.transfers.upload,
            )?;
            Ok(())
        })?;
        self.uploaded_bytes += self.transfers.upload.len() as u64;
        let dirty = self.parameters_dirty;
        let upload_ns = recording.elapsed();
        let mut batch = resident::Batch::new(context.clone())?;
        batch.profiling = PROFILE;
        let output = batch.graph(self, "inference")?;
        let precision = self.output_precision();
        let bytes = precision.bytes(self.transfers.output.len());
        let readback = self
            .transfers
            .readback
            .as_ref()
            .expect("host transfer session")
            .clone();
        self.output_readback_bytes += bytes as u64;
        let result = batch.read_into(&output.buffer, &readback, bytes, &gate, |view| {
            precision.decode(view, &mut self.transfers.output)?;
            validate_finite(&self.transfers.output)
        });
        match result {
            Ok(((), statistics)) => {
                self.readbacks += 1;
                self.execution_timings.encoding_ns =
                    upload_ns.saturating_add(statistics.encoding_ns);
                self.execution_timings.submission_ns = statistics.submission_ns;
                self.execution_timings.completion_wait_ns = statistics.completion_wait_ns;
                self.execution_timings.readback_ns = statistics.readback_ns;
                self.gpu_profile = statistics.profile;
            }
            Err(error) => {
                self.parameters_dirty = dirty;
                return Err(error);
            }
        }
        Ok(TensorView::from_slice(
            self.output_shape(),
            &self.transfers.output,
        )?)
    }

    pub fn run_with_trace<F>(
        &mut self,
        input: &[f32],
        observer: F,
    ) -> RuntimeResult<TensorView<'_, f32>>
    where
        F: for<'a> FnMut(NodeTrace<'a>) -> RuntimeResult<()>,
    {
        if input.len() != self.input_shape().element_count() {
            return Err(RuntimeError::new(
                "GPU input length does not match fixed session shape",
            ));
        }
        self.run_with_input_and_trace(
            |destination| {
                destination.copy_from_slice(input);
                Ok(())
            },
            observer,
        )
    }
    pub fn run_with_input_and_trace<I, F>(
        &mut self,
        fill: I,
        mut observer: F,
    ) -> RuntimeResult<TensorView<'_, f32>>
    where
        I: FnOnce(&mut [f32]) -> RuntimeResult<()>,
        F: for<'a> FnMut(NodeTrace<'a>) -> RuntimeResult<()>,
    {
        self.gpu_profile = None;
        if self.optimized {
            let limits = ExecutionLimits {
                max_activation_bytes: self
                    .limits
                    .max_activation_bytes
                    .checked_sub(self.memory.total_bytes() - self.memory.diagnostic_bytes)
                    .ok_or_else(|| resource("GPU diagnostic overlap exceeds limit"))?,
            };
            if self.diagnostic.is_none() {
                self.diagnostic = Some(Box::new(Self::new_diagnostic(
                    self.compiled.clone(),
                    self.input_shape().dimensions(),
                    limits,
                )?));
            }
            let shape = self.input_shape();
            let diagnostic = self.diagnostic.as_mut().unwrap();
            diagnostic.reshape_with_limits(shape.dimensions(), limits)?;
            let result = diagnostic
                .run_with_input_and_trace(fill, observer)
                .map(|_| ());
            self.memory.diagnostic_bytes = diagnostic.retained_activation_bytes();
            self.memory.transition_peak_bytes = self
                .memory
                .transition_peak_bytes
                .max(self.memory.total_bytes());
            result?;
            return TensorView::from_slice(diagnostic.output_shape(), &diagnostic.transfers.output)
                .map_err(Into::into);
        }
        self.prepare_input(fill)?;
        self.input_precision()
            .write(&self.transfers.input, &mut self.transfers.upload);
        let maximum = self
            .plan
            .steps
            .iter()
            .map(|step| self.plan.shapes[step.output.0].element_count())
            .max()
            .unwrap_or(self.transfers.input.len());
        let bytes = GpuPrecision::F32.bytes(maximum);
        if self.trace_output.len() < maximum {
            check_budget(
                self.memory.total_bytes() - self.memory.diagnostic_bytes
                    + bytes
                        .checked_mul(2)
                        .ok_or_else(|| resource("GPU trace capacity overflow"))?,
                self.limits,
            )?;
            let context = &self.compiled.context;
            let (buffer, output) = context.checked("GPU trace allocation", || {
                Ok((
                    context.create_buffer(
                        "GPU trace readback",
                        bytes as u64,
                        BufferUsage::MAP_READ | BufferUsage::COPY_DST,
                    )?,
                    allocate_f32(maximum)?,
                ))
            })?;
            check_budget(
                self.memory.total_bytes() - self.memory.diagnostic_bytes
                    + buffer.bytes()
                    + output.capacity() * 4,
                self.limits,
            )?;
            self.memory.transition_peak_bytes = self.memory.transition_peak_bytes.max(
                self.memory
                    .total_bytes()
                    .saturating_add(buffer.bytes())
                    .saturating_add(output.capacity() * 4),
            );
            self.trace_readback = Some(buffer);
            self.trace_output = output;
            self.memory.diagnostic_bytes = bytes + self.trace_output.capacity() * 4;
        }
        let context = self.compiled.context.clone();
        {
            let _gate = context.acquire_gate()?;
            context.checked("GPU trace input", || {
                if self.parameters_dirty {
                    context.upload(&self.parameters, 0, &self.parameter_data)?;
                }
                context.upload(
                    self.resolve(self.compiled.program.input),
                    0,
                    &self.transfers.upload,
                )?;
                Ok(())
            })?;
        }
        for index in 0..self.node_count {
            let output = self.plan.steps[index].output;
            let shape = self.plan.shapes[output.0];
            let count = shape.element_count();
            let precision = self.descriptors[output.0].precision;
            let timer = std::time::Instant::now();
            {
                let gate = context.acquire_gate()?;
                let mut batch = resident::Batch::new(context.clone())?;
                let tensor = batch.graph_node(self, index, "trace")?;
                let readback = self.trace_readback.as_ref().unwrap().clone();
                batch.read_into(
                    &tensor.buffer,
                    &readback,
                    precision.bytes(count),
                    &gate,
                    |view| precision.decode(view, &mut self.trace_output[..count]),
                )?;
            }
            self.diagnostic_readbacks += 1;
            self.diagnostic_readback_bytes += precision.bytes(count) as u64;
            let node = &self.compiled.model().graph().nodes[index];
            observer(NodeTrace {
                index,
                key: &node.key,
                name: &node.name,
                op_type: node.op_type(),
                output_id: output,
                shape,
                layout: if shape.rank() == 4 {
                    "nchw"
                } else {
                    "row_major"
                },
                values: &self.trace_output[..count],
                elapsed_ns: timer.elapsed().as_nanos().min(u64::MAX as u128) as u64,
            })?;
        }
        if self.node_count == 0 {
            self.transfers.output.copy_from_slice(&self.transfers.input);
        } else {
            let len = self.transfers.output.len();
            self.transfers
                .output
                .copy_from_slice(&self.trace_output[..len]);
        }
        validate_finite(&self.transfers.output)?;
        Ok(TensorView::from_slice(
            self.output_shape(),
            &self.transfers.output,
        )?)
    }
}
struct Built {
    dispatches: Vec<Dispatch>,
    parameters: GpuBuffer,
    parameter_data: Arc<[u8]>,
    buffers: Vec<GpuBuffer>,
    transfers: HostTransfers,
    memory: GpuMemoryStatistics,
}
fn resident_cached_memory(
    mut memory: GpuMemoryStatistics,
    parameter_bytes: usize,
    limit: usize,
) -> Option<GpuMemoryStatistics> {
    let total = memory
        .total_bytes()
        .checked_sub(memory.host_parameter_bytes)?
        .checked_add(parameter_bytes)?;
    if total > limit {
        return None;
    }
    memory.host_parameter_bytes = parameter_bytes;
    memory.transition_peak_bytes = memory.transition_peak_bytes.max(total);
    Some(memory)
}
fn descriptor_metadata_bytes(values: &Vec<GpuTensorDescriptor>) -> usize {
    values.capacity() * std::mem::size_of::<GpuTensorDescriptor>()
        + values
            .iter()
            .map(|d| d.strides.capacity() * std::mem::size_of::<usize>())
            .sum::<usize>()
}
fn dispatch_metadata_bytes(values: &Vec<Dispatch>) -> usize {
    values.capacity() * std::mem::size_of::<Dispatch>()
        + values
            .iter()
            .map(|d| d.nodes.capacity() * std::mem::size_of::<usize>())
            .sum::<usize>()
}
fn recipe_metadata_bytes(values: &Vec<operators::Recipe>) -> usize {
    values.capacity() * std::mem::size_of::<operators::Recipe>()
        + values
            .iter()
            .map(|r| {
                r.inputs.capacity() * std::mem::size_of::<TensorId>()
                    + r.nodes.capacity() * std::mem::size_of::<usize>()
            })
            .sum::<usize>()
}
fn plan_metadata_bytes(plan: &ShapePlan) -> usize {
    plan.shapes.capacity() * std::mem::size_of::<Shape>()
        + plan.steps.capacity() * std::mem::size_of::<super::bound::BoundStep>()
        + plan.slots.capacity() * std::mem::size_of::<usize>()
        + plan.lengths.capacity() * std::mem::size_of::<usize>()
}
fn allocate_f32(elements: usize) -> RuntimeResult<Vec<f32>> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(elements)
        .map_err(|e| RuntimeError::caused_by(ErrorKind::Allocation, "GPU host allocation", e))?;
    output.resize(elements, 0.0);
    Ok(output)
}
fn decode_f32(bytes: &[u8], output: &mut [f32]) -> RuntimeResult<()> {
    if bytes.len() != output.len() * 4 {
        return Err(RuntimeError::new("GPU readback length mismatch"));
    }
    for (destination, bytes) in output.iter_mut().zip(bytes.as_chunks::<4>().0.iter()) {
        *destination = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    }
    Ok(())
}
fn validate_finite(values: &[f32]) -> RuntimeResult<()> {
    if let Some(index) = values.iter().position(|v| !v.is_finite()) {
        return Err(RuntimeError::new(format!(
            "nonfinite output F32 at element {index}"
        )));
    }
    Ok(())
}
pub(crate) fn resource(message: impl Into<String>) -> RuntimeError {
    RuntimeError::with_kind(ErrorKind::ResourceLimit, message)
}

fn check_budget(bytes: usize, limits: ExecutionLimits) -> RuntimeResult<()> {
    if bytes > limits.max_activation_bytes {
        Err(resource(format!(
            "GPU activations and output need {bytes} bytes, limit is {}",
            limits.max_activation_bytes
        )))
    } else {
        Ok(())
    }
}
fn validate_supported(program: &Program) -> RuntimeResult<()> {
    let report = GpuCompiledModel::support_report(&program.model);
    if report.is_supported() {
        Ok(())
    } else {
        Err(RuntimeError::with_kind(
            ErrorKind::Unsupported,
            report
                .issues
                .iter()
                .map(|issue| issue.reason.as_str())
                .collect::<Vec<_>>()
                .join("; "),
        ))
    }
}
fn checked_storage_bytes(
    elements: usize,
    limits: &crate::runtime::gpu::GpuLimits,
) -> RuntimeResult<u64> {
    if elements == 0 || u32::try_from(elements).is_err() {
        return Err(resource(
            "GPU tensor element count must be positive and fit U32",
        ));
    }
    let bytes = (elements as u64)
        .checked_mul(4)
        .ok_or_else(|| resource("GPU storage byte count overflow"))?;
    if bytes > limits.max_buffer_size || bytes > limits.max_storage_buffer_binding_size {
        return Err(resource(format!(
            "GPU storage needs {bytes} bytes, exceeding device buffer or binding limit"
        )));
    }
    Ok(bytes)
}
fn dispatch_grid(
    elements: usize,
    limits: &crate::runtime::gpu::GpuLimits,
) -> RuntimeResult<(u32, u32)> {
    if elements == 0 || u32::try_from(elements).is_err() {
        return Err(resource(
            "GPU dispatch element count must be positive and fit U32",
        ));
    }
    let max = limits.max_compute_workgroups_per_dimension as usize;
    if max == 0 {
        return Err(resource("GPU has no compute workgroup capacity"));
    }
    let groups = elements.div_ceil(WORKGROUP_SIZE);
    let x = groups.min(max);
    let y = groups.div_ceil(x);
    if y > max
        || x.checked_mul(y)
            .and_then(|n| n.checked_mul(WORKGROUP_SIZE))
            .is_none_or(|n| n > u32::MAX as usize)
    {
        return Err(resource(
            "GPU dispatch grid exceeds device limits or U32 index range",
        ));
    }
    Ok((x as u32, y as u32))
}
fn validate_plan(
    program: &Program,
    plan: &ShapePlan,
    limits: &crate::runtime::gpu::GpuLimits,
) -> RuntimeResult<()> {
    for &length in &plan.lengths {
        checked_storage_bytes(length, limits)?;
    }
    for (index, shape) in plan.shapes.iter().enumerate() {
        if program.model.graph().tensors[index].dtype == DType::F32
            && plan.slots[index] != usize::MAX
        {
            checked_storage_bytes(shape.element_count(), limits)?;
        }
    }
    for step in &plan.steps {
        dispatch_grid(plan.shapes[step.output.0].element_count(), limits)?;
        if let BoundKind::Conv(p) = &step.kind {
            let values = [
                p.batch, p.cin, p.cout, p.height, p.width, p.oh, p.ow, p.cpg, p.opg, p.kh, p.kw,
                p.sh, p.sw, p.dh, p.dw, p.pt, p.pl,
            ];
            if values.into_iter().any(|n| u32::try_from(n).is_err()) {
                return Err(resource("GPU convolution parameter exceeds U32"));
            }
            if p.height > i32::MAX as usize || p.width > i32::MAX as usize {
                return Err(resource("GPU convolution input extent exceeds I32"));
            }
            for (output, stride, kernel, dilation, pad) in [
                (p.oh, p.sh, p.kh, p.dh, p.pt),
                (p.ow, p.sw, p.kw, p.dw, p.pl),
            ] {
                let extent = (output - 1).checked_mul(stride).and_then(|n| {
                    (kernel - 1)
                        .checked_mul(dilation)
                        .and_then(|k| n.checked_add(k))
                });
                if extent.is_none_or(|n| n > i32::MAX as usize) || pad > i32::MAX as usize {
                    return Err(resource("GPU convolution coordinate exceeds I32"));
                }
            }
        }
    }
    Ok(())
}
