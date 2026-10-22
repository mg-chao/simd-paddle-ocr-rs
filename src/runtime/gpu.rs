//! Native GPU execution with shared weights, reusable storage and transactional shapes.
use super::{
    BindScratch, ExecutionLimits, ExecutionStatistics, NodeTrace, Program, RuntimeError,
    RuntimeResult, ShapePlan, bind_into, kernels::BoundKind,
};
use crate::profile::Timer;
use crate::{
    error::ErrorKind,
    model::{Model, TensorId},
    tensor::{DType, Shape, TensorData, TensorView},
};
use std::{
    cell::RefCell,
    collections::VecDeque,
    ops::Deref,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
};
use wgpu::util::DeviceExt;
pub mod heads;
mod operators;
pub use heads::CompactRow;
const WORKGROUP_SIZE: usize = 64;
const PLAN_CACHE_CAPACITY: usize = 8;

/// Calling-thread production wall timings, never device timestamp measurements.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct GpuExecutionTimings {
    pub admission_wait_ns: u64,
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
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GpuBackendPreference {
    #[default]
    Auto,
    Vulkan,
    Dx12,
    Metal,
}
#[derive(Clone, Copy, Debug)]
pub struct GpuContextOptions {
    pub backend: GpuBackendPreference,
    pub max_in_flight: usize,
    pub max_device_bytes: Option<usize>,
}
impl Default for GpuContextOptions {
    fn default() -> Self {
        Self {
            backend: GpuBackendPreference::Auto,
            max_in_flight: 1,
            max_device_bytes: None,
        }
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuDeviceMemoryStatistics {
    pub used_bytes: usize,
    pub peak_bytes: usize,
}
struct Ledger {
    used: AtomicUsize,
    peak: AtomicUsize,
    limit: Option<usize>,
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
pub(super) struct GpuBuffer {
    pub(super) buffer: wgpu::Buffer,
    reservation: Arc<Reservation>,
}
impl Deref for GpuBuffer {
    type Target = wgpu::Buffer;
    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}
impl GpuBuffer {
    fn bytes(&self) -> usize {
        self.reservation.bytes
    }
}
thread_local! {static ACTIVE_CONTEXTS:RefCell<Vec<usize>>=const {RefCell::new(Vec::new())};}
pub(super) struct Gate<'a> {
    context: &'a GpuContext,
    key: usize,
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
    pub(super) device: wgpu::Device,
    pub(super) queue: wgpu::Queue,
    info: wgpu::AdapterInfo,
    limits: wgpu::Limits,
    failure: Arc<Mutex<Option<String>>>,
    ledger: Arc<Ledger>,
    in_flight: Mutex<usize>,
    available: Condvar,
    #[cfg(test)]
    waiting: AtomicUsize,
    options: GpuContextOptions,
}
impl GpuContext {
    pub fn new() -> RuntimeResult<Arc<Self>> {
        Self::new_with_options(GpuContextOptions::default())
    }
    pub fn new_with_options(options: GpuContextOptions) -> RuntimeResult<Arc<Self>> {
        if options.max_in_flight == 0 || options.max_device_bytes == Some(0) {
            return Err(RuntimeError::with_kind(
                ErrorKind::InvalidInput,
                "GPU limits must be positive",
            ));
        }
        let backends = match options.backend {
            GpuBackendPreference::Auto => {
                wgpu::Backends::VULKAN | wgpu::Backends::DX12 | wgpu::Backends::METAL
            }
            GpuBackendPreference::Vulkan => wgpu::Backends::VULKAN,
            GpuBackendPreference::Dx12 => wgpu::Backends::DX12,
            GpuBackendPreference::Metal => wgpu::Backends::METAL,
        };
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
            apply_limit_buckets: false,
        }))
        .map_err(|e| RuntimeError::new(format!("GPU adapter initialization failed: {e}")))?;
        let info = adapter.get_info();
        if !matches!(
            info.device_type,
            wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::IntegratedGpu
        ) {
            return Err(RuntimeError::with_kind(
                ErrorKind::Unsupported,
                format!(
                    "GPU requires a physical adapter, found {} ({:?})",
                    info.name, info.device_type
                ),
            ));
        }
        let available = adapter.limits();
        let requested = wgpu::Limits {
            max_buffer_size: available.max_buffer_size,
            max_storage_buffer_binding_size: available.max_storage_buffer_binding_size,
            max_compute_workgroups_per_dimension: available.max_compute_workgroups_per_dimension,
            ..Default::default()
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("simd-paddle-ocr GPU"),
            required_features: wgpu::Features::empty(),
            required_limits: requested,
            ..Default::default()
        }))
        .map_err(|e| RuntimeError::new(format!("GPU device initialization failed: {e}")))?;
        let failure = Arc::new(Mutex::new(None));
        let lost = failure.clone();
        device.set_device_lost_callback(move |reason, message| {
            if let Ok(mut value) = lost.lock() {
                *value = Some(format!("GPU device lost ({reason:?}): {message}"));
            }
        });
        let uncaptured = failure.clone();
        device.on_uncaptured_error(Arc::new(move |error| {
            if let Ok(mut value) = uncaptured.lock() {
                *value = Some(format!("GPU asynchronous error: {error}"));
            }
        }));
        let limits = device.limits();
        Ok(Arc::new(Self {
            device,
            queue,
            info,
            limits,
            failure,
            ledger: Arc::new(Ledger {
                used: AtomicUsize::new(0),
                peak: AtomicUsize::new(0),
                limit: options.max_device_bytes,
            }),
            in_flight: Mutex::new(0),
            available: Condvar::new(),
            #[cfg(test)]
            waiting: AtomicUsize::new(0),
            options,
        }))
    }
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.info
    }
    pub fn options(&self) -> GpuContextOptions {
        self.options
    }
    pub fn memory_statistics(&self) -> GpuDeviceMemoryStatistics {
        GpuDeviceMemoryStatistics {
            used_bytes: self.ledger.used.load(Ordering::Acquire),
            peak_bytes: self.ledger.peak.load(Ordering::Acquire),
        }
    }
    fn check_alive(&self) -> RuntimeResult<()> {
        let value = self
            .failure
            .lock()
            .map_err(|_| RuntimeError::new("GPU error state poisoned"))?;
        match value.as_ref() {
            Some(message) => Err(RuntimeError::new(message.clone())),
            None => Ok(()),
        }
    }
    fn fail(&self, message: String) {
        if let Ok(mut value) = self.failure.lock() {
            *value = Some(message);
        }
        self.available.notify_all();
    }
    pub(super) fn acquire_gate(&self) -> RuntimeResult<Gate<'_>> {
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
            let waited = self.available.wait(count);
            #[cfg(test)]
            self.waiting.fetch_sub(1, Ordering::AcqRel);
            count = waited.map_err(|_| RuntimeError::new("GPU admission state poisoned"))?;
            self.check_alive()?;
        }
        self.check_alive()?;
        *count += 1;
        ACTIVE_CONTEXTS.with(|v| v.borrow_mut().push(key));
        Ok(Gate { context: self, key })
    }
    fn reserve(&self, bytes: usize) -> RuntimeResult<Arc<Reservation>> {
        self.check_alive()?;
        let mut previous = self.ledger.used.load(Ordering::Acquire);
        loop {
            let next = previous
                .checked_add(bytes)
                .ok_or_else(|| resource("GPU allocation count overflow"))?;
            if self.ledger.limit.is_some_and(|limit| next > limit) {
                return Err(resource(format!(
                    "GPU device allocations need {next} bytes, context limit is {}",
                    self.ledger.limit.unwrap()
                )));
            }
            match self.ledger.used.compare_exchange_weak(
                previous,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    self.ledger.peak.fetch_max(next, Ordering::AcqRel);
                    return Ok(Arc::new(Reservation {
                        ledger: self.ledger.clone(),
                        bytes,
                    }));
                }
                Err(actual) => previous = actual,
            }
        }
    }
    pub(super) fn create_buffer(
        &self,
        label: &str,
        size: u64,
        usage: wgpu::BufferUsages,
    ) -> RuntimeResult<GpuBuffer> {
        if size == 0 || size > self.limits.max_buffer_size {
            return Err(resource("GPU allocation exceeds device buffer limit"));
        }
        let reservation = self.reserve(
            usize::try_from(size).map_err(|_| resource("GPU buffer exceeds host index range"))?,
        )?;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        });
        Ok(GpuBuffer {
            buffer,
            reservation,
        })
    }
    pub(super) fn create_buffer_init(
        &self,
        label: &str,
        contents: &[u8],
        usage: wgpu::BufferUsages,
    ) -> RuntimeResult<GpuBuffer> {
        let reservation = self.reserve(contents.len())?;
        let buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage,
            });
        Ok(GpuBuffer {
            buffer,
            reservation,
        })
    }
    pub(super) fn checked<T>(
        &self,
        stage: &str,
        operation: impl FnOnce() -> RuntimeResult<T>,
    ) -> RuntimeResult<T> {
        self.check_alive()?;
        let oom = self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let internal = self.device.push_error_scope(wgpu::ErrorFilter::Internal);
        let validation = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let result = operation();
        let errors = [
            pollster::block_on(validation.pop()),
            pollster::block_on(internal.pop()),
            pollster::block_on(oom.pop()),
        ];
        if let Some(error) = errors.into_iter().flatten().next() {
            let kind = if matches!(error, wgpu::Error::OutOfMemory { .. }) {
                ErrorKind::Allocation
            } else {
                ErrorKind::Execution
            };
            return Err(RuntimeError::caused_by(kind, stage, error));
        }
        self.check_alive()?;
        result.map_err(|e| e.context(stage))
    }
    pub(super) fn submit_and_map<T>(
        &self,
        encoder: wgpu::CommandEncoder,
        readback: &wgpu::Buffer,
        bytes: u64,
        consume: impl FnOnce(&[u8]) -> RuntimeResult<T>,
    ) -> RuntimeResult<T> {
        self.submit_and_map_profiled::<false, T>(
            encoder,
            readback,
            bytes,
            &Timer::<false>::start(),
            &mut GpuExecutionTimings::default(),
            consume,
        )
    }
    fn submit_and_map_profiled<const PROFILE: bool, T>(
        &self,
        encoder: wgpu::CommandEncoder,
        readback: &wgpu::Buffer,
        bytes: u64,
        submission_timer: &Timer<PROFILE>,
        timings: &mut GpuExecutionTimings,
        consume: impl FnOnce(&[u8]) -> RuntimeResult<T>,
    ) -> RuntimeResult<T> {
        let submission = self.queue.submit([encoder.finish()]);
        let (send, receive) = mpsc::sync_channel(1);
        readback
            .slice(..bytes)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = send.send(result);
            });
        timings.submission_ns = submission_timer.elapsed();
        let completion_timer = Timer::<PROFILE>::start();
        if let Err(error) = self.device.poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(std::time::Duration::from_secs(60)),
        }) {
            timings.completion_wait_ns = completion_timer.elapsed();
            readback.unmap();
            let message = format!("GPU completion wait failed: {error}");
            self.fail(message.clone());
            return Err(RuntimeError::new(message));
        }
        match receive.try_recv() {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                timings.completion_wait_ns = completion_timer.elapsed();
                readback.unmap();
                return Err(RuntimeError::new(format!(
                    "GPU output mapping failed: {error}"
                )));
            }
            Err(error) => {
                timings.completion_wait_ns = completion_timer.elapsed();
                readback.unmap();
                return Err(RuntimeError::new(format!(
                    "GPU mapping callback unavailable: {error}"
                )));
            }
        }
        timings.completion_wait_ns = completion_timer.elapsed();
        let readback_timer = Timer::<PROFILE>::start();
        let result = (|| {
            let view = readback
                .slice(..bytes)
                .get_mapped_range()
                .map_err(|e| RuntimeError::new(format!("GPU mapped output access failed: {e}")))?;
            consume(&view)
        })();
        readback.unmap();
        timings.readback_ns = readback_timer.elapsed();
        result
    }
}
pub struct GpuCompiledModel {
    program: Program,
    context: Arc<GpuContext>,
    constants: Vec<Option<GpuBuffer>>,
    dummy: GpuBuffer,
    conv: wgpu::ComputePipeline,
    operators: wgpu::ComputePipeline,
    head_pipelines: heads::HeadPipelines,
    weight_bytes: usize,
}
impl GpuCompiledModel {
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
        let report = Self::support_report(&model);
        if !report.is_supported() {
            return Err(RuntimeError::with_kind(
                ErrorKind::Unsupported,
                report
                    .issues
                    .iter()
                    .map(|issue| {
                        format!(
                            "GPU node {} ({}, {}, sources {:?}): {}",
                            issue.node_index,
                            issue.node_name,
                            issue.operator,
                            issue.source_indices,
                            issue.reason
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("; "),
            ));
        }
        let program = Program::compile(model)?;
        validate_supported(&program)?;
        for tensor in &program.model.graph().tensors {
            if let Some(TensorData::F32(values)) = &tensor.constant {
                checked_storage_bytes(values.shape().element_count(), &context.limits)?;
            }
        }
        let (constants, dummy, conv, operators, head_pipelines) =
            context.checked("GPU model compilation", || {
                let constants = program
                    .model
                    .graph()
                    .tensors
                    .iter()
                    .map(|tensor| match &tensor.constant {
                        Some(TensorData::F32(values)) => context
                            .create_buffer_init(
                                "GPU constant",
                                bytemuck::cast_slice(values.data()),
                                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                            )
                            .map(Some),
                        _ => Ok(None),
                    })
                    .collect::<RuntimeResult<Vec<_>>>()?;
                let dummy = context.create_buffer_init(
                    "GPU absent operand",
                    &[0; 4],
                    wgpu::BufferUsages::STORAGE,
                )?;
                let conv = pipeline(&context.device, "GPU Conv", include_str!("gpu/conv.wgsl"));
                let operators = pipeline(
                    &context.device,
                    "GPU operators",
                    include_str!("gpu/operators.wgsl"),
                );
                let head_pipelines = heads::HeadPipelines::new(&context.device);
                Ok((constants, dummy, conv, operators, head_pipelines))
            })?;
        let weight_bytes = constants
            .iter()
            .flatten()
            .map(GpuBuffer::bytes)
            .sum::<usize>()
            + dummy.bytes();
        Ok(Self {
            program,
            context,
            constants,
            dummy,
            conv,
            operators,
            head_pipelines,
            weight_bytes,
        })
    }
    pub fn weight_bytes(&self) -> usize {
        self.weight_bytes
    }
    pub fn model(&self) -> &Model {
        &self.program.model
    }
    pub fn context(&self) -> &Arc<GpuContext> {
        &self.context
    }
}
fn pipeline(device: &wgpu::Device, label: &str, source: &'static str) -> wgpu::ComputePipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuMemoryStatistics {
    pub device_activation_bytes: usize,
    pub readback_bytes: usize,
    pub host_output_bytes: usize,
    pub host_input_bytes: usize,
    pub head_workspace_bytes: usize,
    pub diagnostic_bytes: usize,
    pub transition_peak_bytes: usize,
}
impl GpuMemoryStatistics {
    pub fn total_bytes(&self) -> usize {
        self.device_activation_bytes
            .saturating_add(self.readback_bytes)
            .saturating_add(self.host_output_bytes)
            .saturating_add(self.host_input_bytes)
            .saturating_add(self.head_workspace_bytes)
            .saturating_add(self.diagnostic_bytes)
    }
}
struct Dispatch {
    node: usize,
    command: Command,
}
enum Command {
    Compute {
        shader: operators::ShaderKind,
        bind_group: wgpu::BindGroup,
        parameters: GpuBuffer,
        x: u32,
        y: u32,
    },
    Copy {
        source: TensorId,
        output: TensorId,
        source_offset: u64,
        destination_offset: u64,
        bytes: u64,
    },
}
impl Dispatch {
    fn parameter_bytes(&self) -> usize {
        match &self.command {
            Command::Compute { parameters, .. } => parameters.bytes(),
            Command::Copy { .. } => 0,
        }
    }
}
#[derive(Clone)]
struct CachedPlan {
    shape: Shape,
    plan: ShapePlan,
    recipes: Vec<operators::Recipe>,
}
pub struct GpuSession {
    // Drop bindings before tracked buffers so device accounting follows ownership.
    dispatches: Vec<Dispatch>,
    head_plan: Option<heads::HeadPlan>,
    limits: ExecutionLimits,
    plan: ShapePlan,
    buffers: Vec<GpuBuffer>,
    readback: GpuBuffer,
    input: Vec<f32>,
    output: Vec<f32>,
    memory: GpuMemoryStatistics,
    execution_timings: GpuExecutionTimings,
    node_count: usize,
    selected_output: TensorId,
    cache: VecDeque<CachedPlan>,
    cache_hits: u64,
    cache_misses: u64,
    head_spec: Option<heads::HeadSpec>,
    compact: Vec<CompactRow>,
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
        Self::new_selected(compiled, dimensions, limits, count, output, None)
    }
    pub(crate) fn new_diagnostic(
        compiled: Arc<GpuCompiledModel>,
        dimensions: &[usize],
        limits: ExecutionLimits,
    ) -> RuntimeResult<Self> {
        Self::new(compiled, dimensions, limits)
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
        Self::new_selected(compiled, dimensions, limits, last_node + 1, output, None)
    }
    pub(crate) fn new_recognition(
        compiled: Arc<GpuCompiledModel>,
        dimensions: &[usize],
        limits: ExecutionLimits,
        classes: usize,
        probability: bool,
    ) -> RuntimeResult<Self> {
        let spec = heads::HeadSpec::recognition(compiled.model(), classes, probability)?;
        Self::new_selected(
            compiled,
            dimensions,
            limits,
            spec.node_count(),
            spec.selected_output(),
            Some(spec),
        )
    }
    pub(crate) fn new_classification(
        compiled: Arc<GpuCompiledModel>,
        dimensions: &[usize],
        limits: ExecutionLimits,
    ) -> RuntimeResult<Self> {
        let spec = heads::HeadSpec::classification(compiled.model())?;
        Self::new_selected(
            compiled,
            dimensions,
            limits,
            spec.node_count(),
            spec.selected_output(),
            Some(spec),
        )
    }
    fn new_selected(
        compiled: Arc<GpuCompiledModel>,
        dimensions: &[usize],
        limits: ExecutionLimits,
        node_count: usize,
        selected_output: TensorId,
        head_spec: Option<heads::HeadSpec>,
    ) -> RuntimeResult<Self> {
        let bound = Self::bind(&compiled, dimensions, limits, node_count, selected_output)?;
        let built = Self::build(&compiled, &bound, limits, &[], None, head_spec.as_ref())?;
        let mut session = Self {
            compiled,
            limits,
            plan: bound.plan,
            buffers: built.buffers,
            readback: built.readback,
            input: built.input,
            output: built.output,
            dispatches: built.dispatches,
            memory: built.memory,
            execution_timings: GpuExecutionTimings::default(),
            node_count,
            selected_output,
            cache: VecDeque::new(),
            cache_hits: 0,
            cache_misses: 1,
            head_spec,
            head_plan: built.head_plan,
            compact: built.compact,
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
    ) -> RuntimeResult<CachedPlan> {
        let mut plan = ShapePlan::empty(compiled.program.model.graph().tensors.len(), node_count);
        let mut scratch = BindScratch::new(&compiled.program);
        bind_into(
            &compiled.program,
            dimensions,
            limits,
            node_count,
            selected_output,
            &mut plan,
            &mut scratch,
        )?;
        validate_plan(&compiled.program, &plan, &compiled.context.limits)?;
        let recipes = operators::recipes(&compiled.program, &plan, &compiled.context.limits)?;
        Ok(CachedPlan {
            shape: plan.input_shape,
            plan,
            recipes,
        })
    }
    fn build(
        compiled: &GpuCompiledModel,
        bound: &CachedPlan,
        limits: ExecutionLimits,
        previous: &[GpuBuffer],
        previous_readback: Option<&GpuBuffer>,
        head_spec: Option<&heads::HeadSpec>,
    ) -> RuntimeResult<Built> {
        let context = &compiled.context;
        let selected = bound
            .plan
            .steps
            .last()
            .map_or(compiled.program.input, |step| step.output);
        let (output_bytes, head_workspace_bytes) = if let Some(spec) = head_spec {
            spec.memory_requirements(bound.plan.shapes[spec.selected_output().0], &context.limits)?
        } else {
            (bound.plan.shapes[selected.0].element_count() * 4, 0)
        };
        let host_output_bytes = if let Some(spec) = head_spec {
            spec.rows(bound.plan.shapes[spec.selected_output().0])?
                .checked_mul(std::mem::size_of::<CompactRow>())
                .ok_or_else(|| resource("GPU compact host size overflow"))?
        } else {
            output_bytes
        };
        let host_input_bytes = bound
            .shape
            .element_count()
            .checked_mul(4)
            .ok_or_else(|| resource("GPU host input size overflow"))?;
        let fixed_bytes = output_bytes
            .checked_add(head_workspace_bytes)
            .and_then(|n| n.checked_add(host_output_bytes))
            .and_then(|n| n.checked_add(host_input_bytes))
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
                    sum.checked_add(
                        previous
                            .get(slot)
                            .map_or(length * 4, |b| b.bytes().max(length * 4)),
                    )
                    .ok_or_else(|| resource("GPU activation capacity overflow"))
                })?;
        let retained_readback =
            previous_readback.map_or(output_bytes, |b| b.bytes().max(output_bytes));
        let reuse = retained
            .checked_add(fixed_bytes - output_bytes)
            .and_then(|n| n.checked_add(retained_readback))
            .is_some_and(|n| n <= limits.max_activation_bytes);
        context.checked("GPU session binding", || {
            let mut buffers = Vec::new();
            for (slot, &length) in bound.plan.lengths.iter().enumerate() {
                let bytes = (length as u64) * 4;
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
                            wgpu::BufferUsages::STORAGE
                                | wgpu::BufferUsages::COPY_DST
                                | wgpu::BufferUsages::COPY_SRC,
                        )?
                    },
                );
            }
            let resolve = |id: TensorId| -> &wgpu::Buffer {
                compiled.constants[id.0]
                    .as_ref()
                    .map(|b| &b.buffer)
                    .unwrap_or_else(|| &buffers[bound.plan.slots[id.0]].buffer)
            };
            let head_plan = head_spec
                .map(|spec| {
                    heads::HeadPlan::new(
                        spec,
                        bound.plan.shapes[spec.selected_output().0],
                        context,
                        &compiled.head_pipelines,
                        resolve,
                    )
                })
                .transpose()?;
            let readback =
                if reuse && previous_readback.is_some_and(|b| b.size() >= output_bytes as u64) {
                    previous_readback.unwrap().clone()
                } else {
                    context.create_buffer(
                        "GPU output readback",
                        output_bytes as u64,
                        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    )?
                };
            let input = allocate_f32(bound.shape.element_count())?;
            let output = if head_plan.is_none() {
                allocate_f32(output_bytes / 4)?
            } else {
                Vec::new()
            };
            let mut compact = Vec::new();
            if let Some(spec) = head_spec {
                let rows = spec.rows(bound.plan.shapes[spec.selected_output().0])?;
                compact.try_reserve_exact(rows).map_err(|e| {
                    RuntimeError::caused_by(
                        ErrorKind::Allocation,
                        "GPU compact output allocation",
                        e,
                    )
                })?;
            }
            let mut dispatches = Vec::new();
            for recipe in &bound.recipes {
                let shader = match &recipe.kind {
                    operators::RecipeKind::Compute(shader) => shader.clone(),
                    operators::RecipeKind::Copy {
                        source_offset,
                        destination_offset,
                        bytes,
                    } => {
                        dispatches.push(Dispatch {
                            node: recipe.node,
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
                let convolution = matches!(shader, operators::ShaderKind::Convolution);
                let parameters = context.create_buffer_init(
                    "GPU dispatch parameters",
                    bytemuck::cast_slice(if convolution {
                        &recipe.words[..32]
                    } else {
                        &recipe.words[..]
                    }),
                    wgpu::BufferUsages::UNIFORM,
                )?;
                let get = |index: usize| {
                    recipe
                        .inputs
                        .get(index)
                        .map(|&id| resolve(id))
                        .unwrap_or(&compiled.dummy.buffer)
                };
                let entries = if convolution {
                    vec![
                        bind_entry(0, get(0)),
                        bind_entry(1, get(1)),
                        bind_entry(2, get(2)),
                        bind_entry(3, resolve(recipe.output)),
                        bind_entry(4, &parameters),
                    ]
                } else {
                    vec![
                        bind_entry(0, get(0)),
                        bind_entry(1, get(1)),
                        bind_entry(2, get(2)),
                        bind_entry(3, get(3)),
                        bind_entry(4, get(4)),
                        bind_entry(5, resolve(recipe.output)),
                        bind_entry(6, &parameters),
                    ]
                };
                let pipeline = if convolution {
                    &compiled.conv
                } else {
                    &compiled.operators
                };
                let bind_group = context
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("GPU node operands"),
                        layout: &pipeline.get_bind_group_layout(0),
                        entries: &entries,
                    });
                dispatches.push(Dispatch {
                    node: recipe.node,
                    command: Command::Compute {
                        shader,
                        bind_group,
                        parameters,
                        x: recipe.x,
                        y: recipe.y,
                    },
                });
            }
            let memory = GpuMemoryStatistics {
                device_activation_bytes: buffers.iter().map(GpuBuffer::bytes).sum(),
                readback_bytes: readback.bytes(),
                host_input_bytes: input.capacity() * 4,
                host_output_bytes: output.capacity() * 4
                    + compact.capacity() * std::mem::size_of::<CompactRow>(),
                head_workspace_bytes: head_plan
                    .as_ref()
                    .map_or(0, heads::HeadPlan::workspace_bytes),
                ..Default::default()
            };
            check_budget(memory.total_bytes(), limits)?;
            Ok(Built {
                buffers,
                readback,
                input,
                output,
                dispatches,
                memory,
                head_plan,
                compact,
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
        let bound = if let Some(index) = cached {
            self.cache[index].clone()
        } else {
            Self::bind(
                &self.compiled,
                dimensions,
                self.limits,
                self.node_count,
                self.selected_output,
            )?
        };
        let binding_limits = ExecutionLimits {
            max_activation_bytes: self
                .limits
                .max_activation_bytes
                .checked_sub(self.memory.diagnostic_bytes)
                .ok_or_else(|| resource("GPU diagnostics exceed activation limit"))?,
        };
        let built = Self::build(
            &self.compiled,
            &bound,
            binding_limits,
            &self.buffers,
            Some(&self.readback),
            self.head_spec.as_ref(),
        )?;
        check_budget(
            built.memory.total_bytes() + self.memory.diagnostic_bytes,
            self.limits,
        )?;
        let old = CachedPlan {
            shape: self.plan.input_shape,
            plan: self.plan.clone(),
            recipes: operators::recipes(
                &self.compiled.program,
                &self.plan,
                &self.compiled.context.limits,
            )?,
        };
        let shared = built
            .buffers
            .iter()
            .filter(|b| {
                self.buffers
                    .iter()
                    .any(|old| Arc::ptr_eq(&b.reservation, &old.reservation))
            })
            .map(GpuBuffer::bytes)
            .sum::<usize>()
            + if Arc::ptr_eq(&built.readback.reservation, &self.readback.reservation) {
                built.readback.bytes()
            } else {
                0
            };
        let peak = self
            .memory
            .total_bytes()
            .saturating_add(built.memory.total_bytes())
            .saturating_sub(shared);
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
        drop(std::mem::replace(&mut self.head_plan, built.head_plan));
        self.plan = bound.plan;
        self.buffers = built.buffers;
        self.readback = built.readback;
        self.input = built.input;
        self.output = built.output;
        self.compact = built.compact;
        let diagnostic = self.memory.diagnostic_bytes;
        let previous_peak = self.memory.transition_peak_bytes;
        self.memory = built.memory;
        self.memory.diagnostic_bytes = diagnostic;
        self.memory.transition_peak_bytes = previous_peak.max(peak);
        Ok(())
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
        &self.input
    }
    pub fn input_shape(&self) -> Shape {
        self.plan.input_shape
    }
    pub fn output_shape(&self) -> Shape {
        self.plan.shapes[self.selected_output.0]
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
            + self.readback.bytes()
            + self
                .dispatches
                .iter()
                .map(Dispatch::parameter_bytes)
                .sum::<usize>()
            + self
                .head_plan
                .as_ref()
                .map_or(0, heads::HeadPlan::retained_device_bytes)
            + self.trace_readback.as_ref().map_or(0, GpuBuffer::bytes)
    }
    pub fn retained_planning_bytes(&self) -> usize {
        self.dispatches.capacity() * std::mem::size_of::<Dispatch>()
            + self
                .dispatches
                .iter()
                .map(Dispatch::parameter_bytes)
                .sum::<usize>()
            + self
                .head_plan
                .as_ref()
                .map_or(0, heads::HeadPlan::parameter_bytes)
            + self
                .cache
                .iter()
                .map(|entry| {
                    plan_metadata_bytes(&entry.plan)
                        + entry.recipes.capacity() * std::mem::size_of::<operators::Recipe>()
                        + entry
                            .recipes
                            .iter()
                            .map(|recipe| {
                                recipe.inputs.capacity() * std::mem::size_of::<TensorId>()
                            })
                            .sum::<usize>()
                })
                .sum::<usize>()
            + plan_metadata_bytes(&self.plan)
            + self.buffers.capacity() * std::mem::size_of::<GpuBuffer>()
            + self.cache.capacity() * std::mem::size_of::<CachedPlan>()
    }
    pub fn retained_scratch_bytes(&self) -> usize {
        0
    }
    pub fn statistics(&self) -> ExecutionStatistics {
        ExecutionStatistics {
            original_weight_bytes: self.compiled.weight_bytes() as u64,
            normalized_nodes: self.node_count,
            physical_steps: self.dispatches.len(),
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
    fn resolve(&self, id: TensorId) -> &wgpu::Buffer {
        self.compiled.constants[id.0]
            .as_ref()
            .map(|b| &b.buffer)
            .unwrap_or_else(|| &self.buffers[self.plan.slots[id.0]].buffer)
    }
    fn encode_graph(&self, encoder: &mut wgpu::CommandEncoder) {
        for dispatch in &self.dispatches {
            self.encode_dispatch(encoder, dispatch);
        }
    }
    fn encode_dispatch(&self, encoder: &mut wgpu::CommandEncoder, dispatch: &Dispatch) {
        match &dispatch.command {
            Command::Copy {
                source,
                output,
                source_offset,
                destination_offset,
                bytes,
            } => encoder.copy_buffer_to_buffer(
                self.resolve(*source),
                *source_offset,
                self.resolve(*output),
                *destination_offset,
                *bytes,
            ),
            Command::Compute {
                shader,
                bind_group,
                x,
                y,
                ..
            } => {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("GPU node"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(if matches!(shader, operators::ShaderKind::Convolution) {
                    &self.compiled.conv
                } else {
                    &self.compiled.operators
                });
                pass.set_bind_group(0, bind_group, &[]);
                pass.dispatch_workgroups(*x, *y, 1);
            }
        }
    }
    fn prepare_input(
        &mut self,
        fill: impl FnOnce(&mut [f32]) -> RuntimeResult<()>,
    ) -> RuntimeResult<()> {
        fill(&mut self.input)?;
        if let Some(index) = self.input.iter().position(|v| !v.is_finite()) {
            return Err(RuntimeError::new(format!(
                "nonfinite input F32 at element {index}"
            )));
        }
        Ok(())
    }
    pub fn run(&mut self, input: &[f32]) -> RuntimeResult<TensorView<'_, f32>> {
        if input.len() != self.input_shape().element_count() {
            return Err(RuntimeError::new(
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
        if self.head_plan.is_some() {
            return Err(RuntimeError::new(
                "compact GPU session requires compact execution",
            ));
        }
        self.prepare_input(fill)?;
        let context = self.compiled.context.clone();
        let admission_timer = Timer::<PROFILE>::start();
        let _gate = context.acquire_gate()?;
        let mut timings = GpuExecutionTimings {
            admission_wait_ns: admission_timer.elapsed(),
            ..Default::default()
        };
        let submission_timer = Timer::<PROFILE>::start();
        let result = context.checked("GPU execution", || {
            context.queue.write_buffer(
                self.resolve(self.compiled.program.input),
                0,
                bytemuck::cast_slice(&self.input),
            );
            let mut encoder =
                context
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("GPU inference"),
                    });
            self.encode_graph(&mut encoder);
            let bytes = self.output.len() as u64 * 4;
            encoder.copy_buffer_to_buffer(
                self.resolve(self.selected_output),
                0,
                &self.readback,
                0,
                bytes,
            );
            context.submit_and_map_profiled::<PROFILE, _>(
                encoder,
                &self.readback,
                bytes,
                &submission_timer,
                &mut timings,
                |view| {
                    decode_f32(view, &mut self.output)?;
                    validate_finite(&self.output)
                },
            )
        });
        self.execution_timings = timings;
        result?;
        Ok(TensorView::from_slice(self.output_shape(), &self.output)?)
    }
    pub(crate) fn run_compact_with_input(
        &mut self,
        fill: impl FnOnce(&mut [f32]) -> RuntimeResult<()>,
    ) -> RuntimeResult<&[CompactRow]> {
        self.run_compact_with_input_profiled::<false>(fill)
    }
    pub(crate) fn run_compact_with_input_profiled<const PROFILE: bool>(
        &mut self,
        fill: impl FnOnce(&mut [f32]) -> RuntimeResult<()>,
    ) -> RuntimeResult<&[CompactRow]> {
        self.execution_timings = GpuExecutionTimings::default();
        self.prepare_input(fill)?;
        let context = self.compiled.context.clone();
        let admission_timer = Timer::<PROFILE>::start();
        let _gate = context.acquire_gate()?;
        let mut timings = GpuExecutionTimings {
            admission_wait_ns: admission_timer.elapsed(),
            ..Default::default()
        };
        let submission_timer = Timer::<PROFILE>::start();
        let result = context.checked("GPU compact execution", || {
            let head = self
                .head_plan
                .as_ref()
                .ok_or_else(|| RuntimeError::new("GPU session has no compact head"))?;
            context.queue.write_buffer(
                self.resolve(self.compiled.program.input),
                0,
                bytemuck::cast_slice(&self.input),
            );
            let mut encoder =
                context
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("GPU compact inference"),
                    });
            self.encode_graph(&mut encoder);
            head.encode(&mut encoder, &self.compiled.head_pipelines);
            encoder.copy_buffer_to_buffer(
                head.output_buffer(),
                0,
                &self.readback,
                0,
                head.output_bytes() as u64,
            );
            context.submit_and_map_profiled::<PROFILE, _>(
                encoder,
                &self.readback,
                head.output_bytes() as u64,
                &submission_timer,
                &mut timings,
                |view| head.decode(view, &mut self.compact),
            )
        });
        self.execution_timings = timings;
        result?;
        Ok(&self.compact)
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
        if self.head_plan.is_some() {
            return Err(RuntimeError::new(
                "GPU tracing requires a dense diagnostic session",
            ));
        }
        self.prepare_input(fill)?;
        let maximum = self
            .plan
            .steps
            .iter()
            .map(|step| self.plan.shapes[step.output.0].element_count())
            .max()
            .unwrap_or(self.input.len());
        let bytes = maximum * 4;
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
                        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
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
                context.queue.write_buffer(
                    self.resolve(self.compiled.program.input),
                    0,
                    bytemuck::cast_slice(&self.input),
                );
                Ok(())
            })?;
        }
        for index in 0..self.node_count {
            let output = self.plan.steps[index].output;
            let shape = self.plan.shapes[output.0];
            let count = shape.element_count();
            let timer = std::time::Instant::now();
            {
                let _gate = context.acquire_gate()?;
                context.checked(&format!("GPU trace node {index}"), || {
                    let mut encoder =
                        context
                            .device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("GPU diagnostic node"),
                            });
                    for dispatch in self
                        .dispatches
                        .iter()
                        .filter(|dispatch| dispatch.node == index)
                    {
                        self.encode_dispatch(&mut encoder, dispatch);
                    }
                    let readback = self.trace_readback.as_ref().unwrap();
                    encoder.copy_buffer_to_buffer(
                        self.resolve(output),
                        0,
                        readback,
                        0,
                        (count * 4) as u64,
                    );
                    context.submit_and_map(encoder, readback, (count * 4) as u64, |view| {
                        decode_f32(view, &mut self.trace_output[..count])
                    })
                })?;
            }
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
            self.output.copy_from_slice(&self.input);
        } else {
            let len = self.output.len();
            self.output.copy_from_slice(&self.trace_output[..len]);
        }
        validate_finite(&self.output)?;
        Ok(TensorView::from_slice(self.output_shape(), &self.output)?)
    }
}
struct Built {
    dispatches: Vec<Dispatch>,
    head_plan: Option<heads::HeadPlan>,
    buffers: Vec<GpuBuffer>,
    readback: GpuBuffer,
    input: Vec<f32>,
    output: Vec<f32>,
    memory: GpuMemoryStatistics,
    compact: Vec<CompactRow>,
}
fn plan_metadata_bytes(plan: &ShapePlan) -> usize {
    plan.shapes.capacity() * std::mem::size_of::<Shape>()
        + plan.steps.capacity() * std::mem::size_of::<super::kernels::BoundStep>()
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
    for (destination, bytes) in output.iter_mut().zip(bytes.chunks_exact(4)) {
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
fn resource(message: impl Into<String>) -> RuntimeError {
    RuntimeError::with_kind(ErrorKind::ResourceLimit, message)
}
fn bind_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
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
fn checked_storage_bytes(elements: usize, limits: &wgpu::Limits) -> RuntimeResult<u64> {
    if elements == 0 || u32::try_from(elements).is_err() {
        return Err(resource(
            "GPU tensor element count must be positive and fit U32",
        ));
    }
    let bytes = (elements as u64)
        .checked_mul(4)
        .ok_or_else(|| resource("GPU storage byte count overflow"))?;
    if bytes > limits.max_buffer_size || bytes > u64::from(limits.max_storage_buffer_binding_size) {
        return Err(resource(format!(
            "GPU storage needs {bytes} bytes, exceeding device buffer or binding limit"
        )));
    }
    Ok(bytes)
}
fn dispatch_grid(elements: usize, limits: &wgpu::Limits) -> RuntimeResult<(u32, u32)> {
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
fn validate_plan(program: &Program, plan: &ShapePlan, limits: &wgpu::Limits) -> RuntimeResult<()> {
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
#[cfg(test)]
mod tests;
