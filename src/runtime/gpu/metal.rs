//! Existing Metal resource and queue bridge, isolated from shared OCR code.
use super::*;
#[path = "metal_buffers.rs"]
pub(super) mod buffers;
#[cfg(test)]
#[path = "metal_test_support.rs"]
pub(super) mod test_support;
pub(super) struct Device {
    pub(super) raw: wgpu::Device,
}
impl Device {
    pub(super) fn committed_bytes(&self) -> usize {
        0
    }
    pub(super) fn peak_committed_bytes(&self) -> usize {
        0
    }
}
pub(super) struct Context {
    pub(super) owner: Arc<Device>,
    queue: wgpu::Queue,
}
impl Context {
    pub(super) fn collect_garbage(&self) {}
    pub(super) fn new(
        failure: Arc<Mutex<Option<RuntimeError>>>,
        available: Arc<Condvar>,
    ) -> RuntimeResult<(Self, GpuAdapterInfo, GpuLimits)> {
        if !objc2::available!(macos = 26.0) {
            return Err(RuntimeError::with_kind(
                ErrorKind::Unsupported,
                "native Metal GPU execution requires macOS 26 or newer",
            ));
        }
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::METAL,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
            apply_limit_buckets: false,
        }))
        .map_err(|e| {
            RuntimeError::caused_by(ErrorKind::Unsupported, "Metal adapter initialization", e)
        })?;
        let info = adapter.get_info();
        let device_type = match info.device_type {
            wgpu::DeviceType::DiscreteGpu => GpuDeviceType::DiscreteGpu,
            wgpu::DeviceType::IntegratedGpu => GpuDeviceType::IntegratedGpu,
            _ => {
                return Err(RuntimeError::with_kind(
                    ErrorKind::Unsupported,
                    "physical Metal GPU required",
                ));
            }
        };
        let available_limits = adapter.limits();
        let requested = wgpu::Limits {
            max_buffer_size: available_limits.max_buffer_size,
            max_storage_buffer_binding_size: available_limits.max_storage_buffer_binding_size,
            max_compute_workgroups_per_dimension: available_limits
                .max_compute_workgroups_per_dimension,
            max_compute_workgroup_storage_size: available_limits.max_compute_workgroup_storage_size,
            max_compute_invocations_per_workgroup: available_limits
                .max_compute_invocations_per_workgroup,
            max_compute_workgroup_size_x: available_limits.max_compute_workgroup_size_x,
            max_compute_workgroup_size_y: available_limits.max_compute_workgroup_size_y,
            max_compute_workgroup_size_z: available_limits.max_compute_workgroup_size_z,
            ..Default::default()
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("OCR Metal resources"),
            required_features: wgpu::Features::empty(),
            required_limits: requested,
            ..Default::default()
        }))
        .map_err(|e| {
            RuntimeError::caused_by(ErrorKind::Execution, "Metal device initialization", e)
        })?;
        let lost = failure.clone();
        let wake = available.clone();
        device.set_device_lost_callback(move |reason, message| {
            if let Ok(mut v) = lost.lock() {
                *v = Some(RuntimeError::with_kind(
                    ErrorKind::DeviceLost,
                    format!("Metal device lost ({reason:?}): {message}"),
                ));
            }
            wake.notify_all();
        });
        device.on_uncaptured_error(Arc::new(move |error| {
            if let Ok(mut v) = failure.lock() {
                *v = Some(RuntimeError::with_kind(
                    ErrorKind::GpuValidation,
                    format!("Metal asynchronous error: {error}"),
                ));
            }
            available.notify_all();
        }));
        let p = device.limits();
        let limits = GpuLimits {
            max_buffer_size: p.max_buffer_size,
            max_storage_buffer_binding_size: p.max_storage_buffer_binding_size,
            max_uniform_buffer_binding_size: p
                .max_uniform_buffer_binding_size
                .try_into()
                .unwrap_or(u32::MAX),
            max_compute_workgroups_per_dimension: p.max_compute_workgroups_per_dimension,
            max_compute_workgroup_storage_size: p.max_compute_workgroup_storage_size,
            max_compute_invocations_per_workgroup: p.max_compute_invocations_per_workgroup,
            max_compute_workgroup_size_x: p.max_compute_workgroup_size_x,
            max_compute_workgroup_size_y: p.max_compute_workgroup_size_y,
            max_compute_workgroup_size_z: p.max_compute_workgroup_size_z,
            min_uniform_buffer_offset_alignment: p.min_uniform_buffer_offset_alignment,
            min_storage_buffer_offset_alignment: p.min_storage_buffer_offset_alignment,
        };
        let info = GpuAdapterInfo {
            name: info.name,
            vendor: info.vendor,
            device: info.device,
            device_type,
            backend: GpuApi::Metal,
            driver: info.driver,
            driver_info: info.driver_info,
            api_version: 0,
            subgroup_min_size: info.subgroup_min_size,
            subgroup_max_size: info.subgroup_max_size,
        };
        Ok((
            Self {
                owner: Arc::new(Device { raw: device }),
                queue,
            },
            info,
            limits,
        ))
    }
    pub(super) fn supports_half(&self) -> bool {
        true
    }
    pub(super) fn allocate(
        &self,
        label: &str,
        size: u64,
        usage: BufferUsage,
    ) -> RuntimeResult<Buffer> {
        let (raw, native) = buffers::allocate(
            &self.owner.raw,
            label,
            size,
            to_usage(usage) | wgpu::BufferUsages::COPY_DST,
            None,
        )?;
        Ok(Buffer {
            raw,
            native,
            owner: self.owner.clone(),
            usage,
        })
    }
    pub(super) fn upload(
        &self,
        target: &GpuBuffer,
        offset: u64,
        bytes: &[u8],
    ) -> RuntimeResult<()> {
        self.queue.write_buffer(&target.buffer.raw, offset, bytes);
        Ok(())
    }
    pub(super) fn checked<T>(
        &self,
        stage: &str,
        operation: impl FnOnce() -> RuntimeResult<T>,
    ) -> RuntimeResult<T> {
        let oom = self
            .owner
            .raw
            .push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let internal = self.owner.raw.push_error_scope(wgpu::ErrorFilter::Internal);
        let validation = self
            .owner
            .raw
            .push_error_scope(wgpu::ErrorFilter::Validation);
        let result = operation();
        let validation = pollster::block_on(validation.pop());
        let internal = pollster::block_on(internal.pop());
        let oom = pollster::block_on(oom.pop());
        if let Some(e) = oom.or(internal).or(validation) {
            let kind = match e {
                wgpu::Error::OutOfMemory { .. } => ErrorKind::Allocation,
                wgpu::Error::Internal { .. } => ErrorKind::Execution,
                wgpu::Error::Validation { .. } => ErrorKind::GpuValidation,
            };
            return Err(RuntimeError::caused_by(kind, stage, e));
        }
        result.map_err(|e| e.context(stage))
    }
}
fn to_usage(usage: BufferUsage) -> wgpu::BufferUsages {
    let mut flags = wgpu::BufferUsages::empty();
    for (a, b) in [
        (BufferUsage::STORAGE, wgpu::BufferUsages::STORAGE),
        (BufferUsage::UNIFORM, wgpu::BufferUsages::UNIFORM),
        (BufferUsage::COPY_SRC, wgpu::BufferUsages::COPY_SRC),
        (BufferUsage::COPY_DST, wgpu::BufferUsages::COPY_DST),
        (BufferUsage::MAP_READ, wgpu::BufferUsages::MAP_READ),
    ] {
        if usage.contains(a) {
            flags |= b;
        }
    }
    flags
}
#[derive(Clone)]
pub(super) struct Buffer {
    raw: wgpu::Buffer,
    pub(super) native: buffers::NativeBuffer,
    owner: Arc<Device>,
    usage: BufferUsage,
}
impl PartialEq for Buffer {
    fn eq(&self, other: &Self) -> bool {
        self.raw == other.raw
    }
}
impl Eq for Buffer {}
impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.raw.fmt(f)
    }
}
impl Buffer {
    pub(super) fn size(&self) -> u64 {
        self.raw.size()
    }
    pub(super) fn usage(&self) -> BufferUsage {
        self.usage
    }
    pub(super) fn read<T>(
        &self,
        bytes: usize,
        consume: impl FnOnce(&[u8]) -> RuntimeResult<T>,
    ) -> RuntimeResult<T> {
        let (send, receive) = std::sync::mpsc::sync_channel(1);
        self.raw
            .slice(..bytes as u64)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = send.send(result);
            });
        self.owner
            .raw
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(60)),
            })
            .map_err(|e| RuntimeError::new(e.to_string()))?;
        receive
            .try_recv()
            .map_err(|e| RuntimeError::with_kind(ErrorKind::GpuMapping, e.to_string()))?
            .map_err(|e| RuntimeError::with_kind(ErrorKind::GpuMapping, e.to_string()))?;
        let result = (|| {
            let view = self
                .raw
                .slice(..bytes as u64)
                .get_mapped_range()
                .map_err(|e| RuntimeError::with_kind(ErrorKind::GpuMapping, e.to_string()))?;
            consume(&view)
        })();
        self.raw.unmap();
        result
    }
}
pub(super) struct Submission {
    encoding_ns: u64,
    submission_ns: u64,
    device: Arc<Device>,
    index: Option<wgpu::SubmissionIndex>,
}
impl Submission {
    pub(super) fn host_timings(&self) -> (u64, u64) {
        (self.encoding_ns, self.submission_ns)
    }
    pub(super) fn wait(&mut self) -> RuntimeResult<()> {
        if let Some(index) = self.index.clone() {
            self.device
                .raw
                .poll(wgpu::PollType::Wait {
                    submission_index: Some(index),
                    timeout: Some(std::time::Duration::from_secs(60)),
                })
                .map_err(|e| RuntimeError::with_kind(ErrorKind::DeviceLost, e.to_string()))?;
            self.index = None;
        }
        Ok(())
    }
    pub(super) fn timestamps(&self) -> RuntimeResult<Vec<(usize, u64)>> {
        Ok(vec![])
    }
}
impl Drop for Submission {
    fn drop(&mut self) {
        let _ = self.wait();
    }
}
pub(super) fn submit(
    context: &GpuContext,
    mut operations: Vec<(usize, resident::schedule::Operation)>,
    profiling: bool,
    prefer_serial: bool,
) -> RuntimeResult<Submission> {
    let encoding = profiling.then(std::time::Instant::now);
    use resident::schedule::Operation;
    operations.sort_by_key(|(depth, _)| *depth);
    let device = &context.backend.owner.raw;
    let mut commands = Vec::new();
    let mut index = 0;
    while index < operations.len() {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("resident Metal schedule"),
        });
        if matches!(operations[index].1, Operation::Compute { .. }) {
            let mut dispatches = Vec::new();
            while let Some((depth, Operation::Compute { native, .. })) = operations.get(index) {
                dispatches.push((*depth, native.as_ref()));
                index += 1;
            }
            executor::encode(context, &mut encoder, &dispatches, prefer_serial)?;
        } else {
            while let Some((
                _,
                Operation::Copy {
                    source,
                    output,
                    source_offset,
                    output_offset,
                    bytes,
                },
            )) = operations.get(index)
            {
                if !Arc::ptr_eq(&source.reservation.ledger, &context.ledger)
                    || !Arc::ptr_eq(&output.reservation.ledger, &context.ledger)
                {
                    return Err(RuntimeError::with_kind(
                        ErrorKind::InvalidInput,
                        "foreign Metal copy",
                    ));
                }
                encoder.copy_buffer_to_buffer(
                    &source.buffer.raw,
                    *source_offset,
                    &output.buffer.raw,
                    *output_offset,
                    *bytes,
                );
                index += 1;
            }
        }
        commands.push(encoder.finish());
    }
    context.checked("Metal submission", || {
        let encoding_ns = encoding.map_or(0, |timer| timer.elapsed().as_nanos() as u64);
        let submitting = profiling.then(std::time::Instant::now);
        let index = context.backend.queue.submit(commands);
        let submission_ns = submitting.map_or(0, |timer| timer.elapsed().as_nanos() as u64);
        context
            .backend
            .queue
            .on_submitted_work_done(move || drop(operations));
        Ok(Submission {
            encoding_ns,
            submission_ns,
            device: context.backend.owner.clone(),
            index: Some(index),
        })
    })
}
pub(super) fn compile_resident(
    context: &GpuContext,
    key: KernelKey,
    _read_only: &[bool],
) -> RuntimeResult<Arc<executor::NativePipeline>> {
    #[cfg(test)]
    if key.id == KernelId::ProjectionScalarProbe {
        return executor::compile_resident_msl(
            context,
            &include_str!("heads_native.metal").replace("if (width != 32u)", "if (true)"),
            "head_mpp32x64",
            [128, 1, 1],
            5,
        );
    }
    if matches!(
        key.id,
        KernelId::ProjectionMatrixF32 | KernelId::ProjectionMatrixF16
    ) {
        let half = key.id == KernelId::ProjectionMatrixF16;
        return executor::compile_resident_msl(
            context,
            if half {
                include_str!("heads_native_f16.metal")
            } else {
                include_str!("heads_native.metal")
            },
            if half {
                "head_mpp32x64_f16"
            } else {
                "head_mpp32x64"
            },
            [128, 1, 1],
            5,
        );
    }
    executor::compile_wgsl(
        context,
        &super::metal_resident_sources::source(key)?,
        "main",
        [0; 3],
    )
}
