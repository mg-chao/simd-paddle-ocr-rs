//! Optional native GPU execution of a deliberately small, unfused F32 subset.
//!
//! Activations remain NCHW and convolution weights remain OIHW. A compiled
//! model shares immutable device weights and pipelines; each fixed-shape session
//! owns its activation slots, parameter buffers, readback buffer and host output.
//! Explicit GPU construction reports errors and never falls back to CPU.

use super::{
    BindScratch, ExecutionLimits, Op, Program, RuntimeError, RuntimeResult, ShapePlan, bind_into,
    kernels::BoundKind,
};
use crate::{
    error::ErrorKind,
    model::{Model, Operator, TensorId},
    tensor::{DType, Shape, TensorData, TensorView},
};
use std::sync::{Arc, Mutex, mpsc};
use wgpu::util::DeviceExt;

const WORKGROUP_SIZE: usize = 64;
const PARAM_WORDS: usize = 32;

/// One native, headless device. Initialization is synchronous and requires a
/// hardware adapter; no surface, optional GPU features or software fallback.
pub struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    info: wgpu::AdapterInfo,
    limits: wgpu::Limits,
    failure: Arc<Mutex<Option<String>>>,
}

impl GpuContext {
    pub fn new() -> RuntimeResult<Arc<Self>> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::DX12 | wgpu::Backends::METAL,
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
        if info.device_type == wgpu::DeviceType::Cpu {
            return Err(RuntimeError::with_kind(
                ErrorKind::Unsupported,
                format!("GPU requires a hardware adapter, found {}", info.name),
            ));
        }
        let available = adapter.limits();
        let requested = wgpu::Limits {
            max_buffer_size: available.max_buffer_size,
            max_storage_buffer_binding_size: available.max_storage_buffer_binding_size,
            max_compute_workgroups_per_dimension: available.max_compute_workgroups_per_dimension,
            ..wgpu::Limits::default()
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
            if let Ok(mut failure) = lost.lock() {
                *failure = Some(format!("GPU device lost ({reason:?}): {message}"));
            }
        });
        let uncaptured = failure.clone();
        device.on_uncaptured_error(Arc::new(move |error| {
            if let Ok(mut failure) = uncaptured.lock() {
                *failure = Some(format!("GPU asynchronous error: {error}"));
            }
        }));
        let limits = device.limits();
        Ok(Arc::new(Self {
            device,
            queue,
            info,
            limits,
            failure,
        }))
    }

    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.info
    }

    fn check_alive(&self) -> RuntimeResult<()> {
        let failure = self
            .failure
            .lock()
            .map_err(|_| RuntimeError::new("GPU error state poisoned"))?;
        match failure.as_ref() {
            Some(message) => Err(RuntimeError::new(message.clone())),
            None => Ok(()),
        }
    }

    // wgpu 30 scopes are thread-local with std enabled. Always pop all guards,
    // including when a host-side check fails; no process-wide execution lock.
    fn checked<T>(
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
}

/// Immutable GPU weights and pipelines, shareable across independent sessions.
pub struct GpuCompiledModel {
    program: Program,
    context: Arc<GpuContext>,
    constants: Vec<Option<wgpu::Buffer>>,
    dummy: wgpu::Buffer,
    conv: wgpu::ComputePipeline,
    elementwise: wgpu::ComputePipeline,
    weight_bytes: usize,
}

impl GpuCompiledModel {
    pub fn compile(model: Arc<Model>, context: Arc<GpuContext>) -> RuntimeResult<Self> {
        // Validate the complete graph before creating or uploading GPU resources.
        for (index, node) in model.graph().nodes.iter().enumerate() {
            if !matches!(
                node.operator,
                Operator::Conv(_) | Operator::Add | Operator::Mul | Operator::Relu
            ) {
                return Err(RuntimeError::with_kind(
                    ErrorKind::Unsupported,
                    format!(
                        "GPU node {index} ({}): unsupported {}",
                        node.name,
                        node.operator.name()
                    ),
                ));
            }
        }
        let program = Program::compile(model)?;
        validate_supported(&program)?;
        let mut weight_bytes = 4usize; // the shared absent-operand buffer
        for tensor in &program.model.graph().tensors {
            if let Some(TensorData::F32(values)) = &tensor.constant {
                let bytes = checked_storage_bytes(values.shape().element_count(), &context.limits)?;
                weight_bytes = weight_bytes
                    .checked_add(bytes as usize)
                    .ok_or_else(|| resource("GPU weight byte count overflow"))?;
            }
        }
        let (constants, dummy, conv, elementwise) =
            context.checked("GPU model compilation", || {
                let constants = program
                    .model
                    .graph()
                    .tensors
                    .iter()
                    .map(|tensor| match &tensor.constant {
                        Some(TensorData::F32(values)) => Some(context.device.create_buffer_init(
                            &wgpu::util::BufferInitDescriptor {
                                label: Some("GPU constant"),
                                contents: bytemuck::cast_slice(values.data()),
                                usage: wgpu::BufferUsages::STORAGE,
                            },
                        )),
                        _ => None,
                    })
                    .collect();
                let dummy = context
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("GPU absent operand"),
                        contents: &[0; 4],
                        usage: wgpu::BufferUsages::STORAGE,
                    });
                let pipeline = |label, source: &'static str| {
                    let shader =
                        context
                            .device
                            .create_shader_module(wgpu::ShaderModuleDescriptor {
                                label: Some(label),
                                source: wgpu::ShaderSource::Wgsl(source.into()),
                            });
                    context
                        .device
                        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                            label: Some(label),
                            layout: None,
                            module: &shader,
                            entry_point: Some("main"),
                            compilation_options: Default::default(),
                            cache: None,
                        })
                };
                Ok((
                    constants,
                    dummy,
                    pipeline("GPU Conv", include_str!("gpu/conv.wgsl")),
                    pipeline("GPU elementwise", include_str!("gpu/elementwise.wgsl")),
                ))
            })?;
        Ok(Self {
            program,
            context,
            constants,
            dummy,
            conv,
            elementwise,
            weight_bytes,
        })
    }

    /// Device bytes for shared F32 constants, including the absent-operand zero.
    pub fn weight_bytes(&self) -> usize {
        self.weight_bytes
    }

    pub fn model(&self) -> &Model {
        &self.program.model
    }
}

/// Actual retained activation and output capacities. Shared weights and small
/// per-step uniform buffers/bind groups are separate from this activation budget.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuMemoryStatistics {
    pub device_activation_bytes: usize,
    pub readback_bytes: usize,
    pub host_output_bytes: usize,
}

impl GpuMemoryStatistics {
    pub fn total_bytes(&self) -> usize {
        self.device_activation_bytes
            .saturating_add(self.readback_bytes)
            .saturating_add(self.host_output_bytes)
    }
}

struct Dispatch {
    convolution: bool,
    bind_group: wgpu::BindGroup,
    _parameters: wgpu::Buffer,
    x: u32,
    y: u32,
}

/// One fixed-shape GPU request. `run` uploads only the graph input and reads
/// back only its final output. The returned view borrows reusable host storage.
pub struct GpuSession {
    compiled: Arc<GpuCompiledModel>,
    plan: ShapePlan,
    buffers: Vec<wgpu::Buffer>,
    readback: wgpu::Buffer,
    output: Vec<f32>,
    dispatches: Vec<Dispatch>,
    memory: GpuMemoryStatistics,
}

impl GpuSession {
    pub fn new(
        compiled: Arc<GpuCompiledModel>,
        dimensions: &[usize],
        limits: ExecutionLimits,
    ) -> RuntimeResult<Self> {
        let program = &compiled.program;
        let mut plan = ShapePlan::empty(
            program.model.graph().tensors.len(),
            program.instructions.len(),
        );
        let mut scratch = BindScratch::new(program);
        bind_into(
            program,
            dimensions,
            limits,
            program.instructions.len(),
            program.output,
            &mut plan,
            &mut scratch,
        )?;
        validate_plan(program, &plan, &compiled.context.limits)?;
        let elements = plan.shapes[program.output.0].element_count();
        let output_bytes = elements
            .checked_mul(4)
            .ok_or_else(|| resource("GPU output byte count overflow"))?;
        let required = plan
            .activation_bytes
            .checked_add(output_bytes)
            .and_then(|n| n.checked_add(output_bytes))
            .ok_or_else(|| resource("GPU activation byte count overflow"))?;
        check_budget(required, limits)?;
        let mut output = Vec::new();
        output.try_reserve_exact(elements).map_err(|e| {
            RuntimeError::caused_by(ErrorKind::Allocation, "GPU host output allocation", e)
        })?;
        output.resize(elements, 0.0);
        let memory = GpuMemoryStatistics {
            device_activation_bytes: plan.activation_bytes,
            readback_bytes: output_bytes,
            host_output_bytes: output
                .capacity()
                .checked_mul(4)
                .ok_or_else(|| resource("GPU host output byte count overflow"))?,
        };
        check_budget(memory.total_bytes(), limits)?;
        let (buffers, readback, dispatches) =
            compiled.context.checked("GPU session allocation", || {
                let device = &compiled.context.device;
                let buffers: Vec<_> = plan
                    .lengths
                    .iter()
                    .map(|&length| {
                        device.create_buffer(&wgpu::BufferDescriptor {
                            label: Some("GPU activation slot"),
                            size: (length as u64) * 4,
                            usage: wgpu::BufferUsages::STORAGE
                                | wgpu::BufferUsages::COPY_DST
                                | wgpu::BufferUsages::COPY_SRC,
                            mapped_at_creation: false,
                        })
                    })
                    .collect();
                let readback = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("GPU output readback"),
                    size: output_bytes as u64,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let resolve = |id: TensorId| -> &wgpu::Buffer {
                    compiled.constants[id.0]
                        .as_ref()
                        .unwrap_or_else(|| &buffers[plan.slots[id.0]])
                };
                let mut dispatches = Vec::with_capacity(plan.steps.len());
                for (index, step) in plan.steps.iter().enumerate() {
                    let instruction = &program.instructions[index];
                    let count = plan.shapes[step.output.0].element_count();
                    let (x, y) = dispatch_grid(count, &compiled.context.limits)?;
                    let mut words = [0u32; PARAM_WORDS];
                    words[0] = count as u32;
                    words[1] = x * WORKGROUP_SIZE as u32;
                    let convolution = matches!(step.kind, BoundKind::Conv(_));
                    match &step.kind {
                        BoundKind::Conv(p) => {
                            let values = [
                                p.batch, p.cin, p.cout, p.height, p.width, p.oh, p.ow, p.cpg,
                                p.opg, p.kh, p.kw, p.sh, p.sw, p.dh, p.dw, p.pt, p.pl,
                            ];
                            for (destination, value) in words[2..].iter_mut().zip(values) {
                                *destination = value as u32;
                            }
                            words[19] = u32::from(instruction.inputs.len() == 3);
                        }
                        BoundKind::Binary(broadcast, _) => {
                            words[2] = u32::from(matches!(instruction.kind, Op::Mul));
                            words[3] = broadcast.rank as u32;
                            for axis in 0..broadcast.rank {
                                words[4 + axis] = broadcast.dimensions[axis] as u32;
                                words[12 + axis] = broadcast.a_strides[axis] as u32;
                                words[20 + axis] = broadcast.b_strides[axis] as u32;
                            }
                        }
                        BoundKind::Relu => words[2] = 2,
                        _ => return Err(RuntimeError::new("GPU bound operator was not validated")),
                    }
                    let parameters = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("GPU dispatch parameters"),
                        contents: bytemuck::cast_slice(&words),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });
                    let a = resolve(instruction.inputs[0]);
                    let b = instruction
                        .inputs
                        .get(1)
                        .map(|&id| resolve(id))
                        .unwrap_or(&compiled.dummy);
                    let destination = resolve(step.output);
                    let pipeline = if convolution {
                        &compiled.conv
                    } else {
                        &compiled.elementwise
                    };
                    let layout = pipeline.get_bind_group_layout(0);
                    let bind_group = if convolution {
                        let bias = instruction
                            .inputs
                            .get(2)
                            .map(|&id| resolve(id))
                            .unwrap_or(&compiled.dummy);
                        device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("GPU Conv operands"),
                            layout: &layout,
                            entries: &[
                                bind_entry(0, a),
                                bind_entry(1, b),
                                bind_entry(2, bias),
                                bind_entry(3, destination),
                                bind_entry(4, &parameters),
                            ],
                        })
                    } else {
                        device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("GPU elementwise operands"),
                            layout: &layout,
                            entries: &[
                                bind_entry(0, a),
                                bind_entry(1, b),
                                bind_entry(2, destination),
                                bind_entry(3, &parameters),
                            ],
                        })
                    };
                    dispatches.push(Dispatch {
                        convolution,
                        bind_group,
                        _parameters: parameters,
                        x,
                        y,
                    });
                }
                Ok((buffers, readback, dispatches))
            })?;
        Ok(Self {
            compiled,
            plan,
            buffers,
            readback,
            output,
            dispatches,
            memory,
        })
    }

    pub fn input_shape(&self) -> Shape {
        self.plan.input_shape
    }
    pub fn output_shape(&self) -> Shape {
        self.plan.shapes[self.compiled.program.output.0]
    }
    /// Combined device activation slots, readback capacity and host output capacity.
    pub fn activation_bytes(&self) -> usize {
        self.memory.total_bytes()
    }
    pub fn memory_statistics(&self) -> GpuMemoryStatistics {
        self.memory
    }

    pub fn run(&mut self, input: &[f32]) -> RuntimeResult<TensorView<'_, f32>> {
        if input.len() != self.input_shape().element_count() {
            return Err(RuntimeError::new(
                "GPU input length does not match fixed session shape",
            ));
        }
        if let Some(index) = input.iter().position(|v| !v.is_finite()) {
            return Err(RuntimeError::new(format!(
                "nonfinite input F32 at element {index}"
            )));
        }
        let context = &self.compiled.context;
        let output_bytes = self.memory.readback_bytes as u64;
        let program = &self.compiled.program;
        context.checked("GPU execution", || {
            context.queue.write_buffer(
                &self.buffers[self.plan.slots[program.input.0]],
                0,
                bytemuck::cast_slice(input),
            );
            let mut encoder =
                context
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("GPU inference"),
                    });
            // A separate compute pass makes storage hazards and slot reuse
            // explicit to wgpu's resource tracker, including branch joins.
            for dispatch in &self.dispatches {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("GPU node"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(if dispatch.convolution {
                    &self.compiled.conv
                } else {
                    &self.compiled.elementwise
                });
                pass.set_bind_group(0, &dispatch.bind_group, &[]);
                pass.dispatch_workgroups(dispatch.x, dispatch.y, 1);
            }
            encoder.copy_buffer_to_buffer(
                &self.buffers[self.plan.slots[program.output.0]],
                0,
                &self.readback,
                0,
                output_bytes,
            );
            let submission = context.queue.submit([encoder.finish()]);
            let (send, receive) = mpsc::sync_channel(1);
            self.readback
                .slice(..output_bytes)
                .map_async(wgpu::MapMode::Read, move |result| {
                    let _ = send.send(result);
                });
            let polled = context.device.poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: Some(std::time::Duration::from_secs(60)),
            });
            if let Err(error) = polled {
                self.readback.unmap();
                return Err(RuntimeError::new(format!(
                    "GPU completion wait failed: {error}"
                )));
            }
            match receive.try_recv() {
                Ok(Ok(())) => {}
                Ok(Err(error)) => {
                    self.readback.unmap();
                    return Err(RuntimeError::new(format!(
                        "GPU output mapping failed: {error}"
                    )));
                }
                Err(error) => {
                    self.readback.unmap();
                    return Err(RuntimeError::new(format!(
                        "GPU mapping callback unavailable: {error}"
                    )));
                }
            }
            let copy = (|| {
                let view = self
                    .readback
                    .slice(..output_bytes)
                    .get_mapped_range()
                    .map_err(|e| {
                        RuntimeError::new(format!("GPU mapped output access failed: {e}"))
                    })?;
                // Decode bytes without assuming mapped storage has f32 alignment.
                for (destination, bytes) in self.output.iter_mut().zip(view.chunks_exact(4)) {
                    *destination = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                }
                Ok(())
            })();
            self.readback.unmap();
            copy
        })?;
        if let Some(index) = self.output.iter().position(|v| !v.is_finite()) {
            return Err(RuntimeError::new(format!(
                "nonfinite output F32 at element {index}"
            )));
        }
        Ok(TensorView::from_slice(self.output_shape(), &self.output)?)
    }
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
    for (index, instruction) in program.instructions.iter().enumerate() {
        if !matches!(instruction.kind, Op::Conv(_) | Op::Add | Op::Mul | Op::Relu) {
            return Err(RuntimeError::with_kind(
                ErrorKind::Unsupported,
                format!(
                    "GPU node {index}: unsupported {}",
                    program.model.graph().nodes[index].operator.name()
                ),
            ));
        }
    }
    Ok(())
}

fn checked_storage_bytes(elements: usize, limits: &wgpu::Limits) -> RuntimeResult<u64> {
    if elements == 0 || u32::try_from(elements).is_err() {
        return Err(resource(
            "GPU tensor element count must be positive and fit U32",
        ));
    }
    let bytes = u64::try_from(elements)
        .ok()
        .and_then(|n| n.checked_mul(4))
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
    validate_supported(program)?;
    for &length in &plan.lengths {
        checked_storage_bytes(length, limits)?;
    }
    for (index, shape) in plan.shapes.iter().enumerate() {
        if program.model.graph().tensors[index].dtype == DType::F32 {
            checked_storage_bytes(shape.element_count(), limits)?;
            if shape
                .dimensions()
                .iter()
                .any(|&n| u32::try_from(n).is_err())
            {
                return Err(resource("GPU shape dimension exceeds U32"));
            }
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
            // Coordinates use signed subtraction for padding. Reject before
            // narrowing; tensor-index and weight products are checked above.
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
