//! Device-resident composition. A batch owns every allocation it records until completion.
use super::*;
use crate::image::{ImageView, PixelFormat};
mod schedule;
use schedule::{Operation, Schedule};

#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct GpuPipelineStatistics {
    pub uploaded_bytes: u64,
    pub output_readback_bytes: u64,
    pub control_readback_bytes: u64,
    pub diagnostic_readback_bytes: u64,
    pub submissions: u64,
    pub cpu_compute_stages: Vec<String>,
    pub kernels: Vec<String>,
    pub peak_device_bytes: usize,
    pub submission_ns: u64,
    pub completion_wait_ns: u64,
    pub readback_ns: u64,
    pub projection_encoding_ns: u64,
    /// Host time encoding resident dispatches and neural graphs; excludes waits.
    pub encoding_ns: u64,
    pub admission_wait_ns: u64,
    pub dispatches: u64,
    pub layout_conversions: u64,
    /// Neural dispatches recorded through the native concurrent Metal path.
    pub native_graph_dispatches: u64,
    /// Image, geometry, projection and decoding dispatches on the native path.
    pub native_resident_dispatches: u64,
    pub scratch_allocations: u64,
    pub scratch_allocated_bytes: u64,
    pub scratch_reuses: u64,
    /// Timestamp data only; never an activation or diagnostic tensor download.
    pub profiling_readback_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub profile: Option<GpuExecutionProfile>,
    pub adapter: String,
    pub backend: String,
    pub driver: String,
    pub driver_info: String,
    pub image_processing: GpuPlacement,
    pub geometry: GpuPlacement,
    pub projection: GpuPlacement,
    pub decoding: GpuPlacement,
}
impl GpuPipelineStatistics {
    pub(crate) fn retained_bytes(&self) -> usize {
        (self.cpu_compute_stages.capacity() + self.kernels.capacity()) * size_of::<String>()
            + self
                .cpu_compute_stages
                .iter()
                .chain(&self.kernels)
                .map(String::capacity)
                .sum::<usize>()
            + self.adapter.capacity()
            + self.backend.capacity()
            + self.driver.capacity()
            + self.driver_info.capacity()
            + self
                .profile
                .as_ref()
                .map_or(0, GpuExecutionProfile::retained_bytes)
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub enum GpuPlacement {
    #[default]
    NotRun,
    Cpu,
    Gpu,
    Hybrid,
}
impl GpuPlacement {
    pub(crate) fn merge(self, other: Self) -> Self {
        if self == Self::NotRun {
            return other;
        }
        if other == Self::NotRun || self == other {
            return self;
        }
        Self::Hybrid
    }
}

#[derive(Clone)]
pub(crate) struct Tensor {
    pub context: Arc<GpuContext>,
    pub buffer: GpuBuffer,
    pub descriptor: GpuTensorDescriptor,
}
impl Tensor {
    pub fn allocate(
        context: Arc<GpuContext>,
        shape: Shape,
        precision: GpuPrecision,
    ) -> RuntimeResult<Self> {
        let bytes = precision.bytes(shape.element_count());
        let buffer = storage(&context, "resident tensor", bytes)?;
        Ok(Self {
            context,
            buffer,
            descriptor: GpuTensorDescriptor::canonical(shape, precision),
        })
    }
    pub fn bytes(&self) -> usize {
        self.descriptor
            .precision
            .bytes(self.descriptor.shape.element_count())
    }
}

#[derive(Clone)]
pub(crate) struct Image {
    pub context: Arc<GpuContext>,
    pub buffer: GpuBuffer,
    pub width: usize,
    pub height: usize,
    pub stride: usize,
    pub format: PixelFormat,
}
impl GpuContext {
    pub(crate) fn upload_resident(&self, buffer: &GpuBuffer, bytes: &[u8]) -> RuntimeResult<()> {
        if !Arc::ptr_eq(&buffer.reservation.ledger, &self.ledger) {
            return Err(RuntimeError::with_kind(
                ErrorKind::InvalidInput,
                "resident upload buffer belongs to another GPU context",
            ));
        }
        self.checked("resident tensor upload", || {
            self.queue.write_buffer(buffer, 0, bytes);
            Ok(())
        })
    }
}
impl Image {
    pub fn upload(batch: &mut Batch, image: ImageView<'_>) -> RuntimeResult<Self> {
        let bytes = (image.height() - 1) * image.stride()
            + image.width() * image.format().bytes_per_pixel();
        if bytes.div_ceil(4) * 4 > batch.context.limits.max_storage_buffer_binding_size as usize
            || image.stride() > u32::MAX as usize
        {
            return Err(resource("source pixels exceed GPU storage binding limit"));
        }
        let mut aligned = vec![0u8; bytes.div_ceil(4) * 4];
        aligned[..bytes].copy_from_slice(&image.pixels()[..bytes]);
        let buffer = batch.context.checked("image upload", || {
            batch.context.create_buffer_init(
                "source pixels",
                &aligned,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            )
        })?;
        batch.statistics.uploaded_bytes += bytes as u64;
        batch.held.push(buffer.clone());
        Ok(Self {
            context: batch.context.clone(),
            buffer,
            width: image.width(),
            height: image.height(),
            stride: image.stride(),
            format: image.format(),
        })
    }
}

pub(crate) fn storage(context: &GpuContext, label: &str, bytes: usize) -> RuntimeResult<GpuBuffer> {
    let bytes = bytes
        .checked_add(3)
        .ok_or_else(|| resource("resident allocation overflow"))?
        / 4
        * 4;
    if bytes == 0 || bytes as u64 > context.limits.max_storage_buffer_binding_size {
        return Err(resource(format!(
            "{label} exceeds GPU storage binding limit"
        )));
    }
    context.checked(label, || {
        context.create_buffer(
            label,
            bytes as u64,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        )
    })
}

pub(super) struct Kernel {
    ledger: Arc<Ledger>,
    pipeline: Arc<metal_execution::NativePipeline>,
    read_only: Box<[bool]>,
}

/// Embedded native compute shader using contiguous buffer bindings: storage
/// operands first, followed by the same 1 KiB uniform packet as WGSL dispatches.
#[derive(Clone, Copy)]
pub(crate) struct NativeShader<'a> {
    pub source: &'a str,
    pub entry_point: &'a str,
    pub workgroup_size: [u32; 3],
}

#[derive(Clone, Copy)]
enum KernelSource<'a> {
    Wgsl(&'a str),
    Native(NativeShader<'a>),
}

impl Kernel {
    pub(super) fn get(
        context: &GpuContext,
        name: &str,
        source: &str,
        read_only: &[bool],
    ) -> RuntimeResult<Arc<Self>> {
        use std::hash::{Hash, Hasher};
        let mut hash = std::collections::hash_map::DefaultHasher::new();
        source.hash(&mut hash);
        Self::get_source(
            context,
            &format!("wgsl/{name}/{:016x}", hash.finish()),
            name,
            KernelSource::Wgsl(source),
            read_only,
        )
    }

    fn get_native(
        context: &GpuContext,
        name: &str,
        shader: NativeShader<'_>,
        read_only: &[bool],
    ) -> RuntimeResult<Arc<Self>> {
        let [x, y, z] = shader.workgroup_size;
        if shader.workgroup_size.contains(&0)
            || x > context.limits.max_compute_workgroup_size_x
            || y > context.limits.max_compute_workgroup_size_y
            || z > context.limits.max_compute_workgroup_size_z
            || x.checked_mul(y)
                .and_then(|n| n.checked_mul(z))
                .is_none_or(|n| n > context.limits.max_compute_invocations_per_workgroup)
        {
            return Err(resource("native resident workgroup exceeds GPU limits"));
        }
        // Source identity keeps native/WGSL variants and development reloads
        // separate even when callers use the same descriptive dispatch name.
        use std::hash::{Hash, Hasher};
        let mut hash = std::collections::hash_map::DefaultHasher::new();
        shader.source.hash(&mut hash);
        shader.entry_point.hash(&mut hash);
        shader.workgroup_size.hash(&mut hash);
        read_only.hash(&mut hash);
        let key = format!("native/{name}/{:016x}", hash.finish());
        Self::get_source(context, &key, name, KernelSource::Native(shader), read_only)
    }

    fn get_source(
        context: &GpuContext,
        key: &str,
        _name: &str,
        source: KernelSource<'_>,
        read_only: &[bool],
    ) -> RuntimeResult<Arc<Self>> {
        context.check_alive()?;
        let mut cache = context
            .resident_pipelines
            .lock()
            .map_err(|_| RuntimeError::new("resident pipeline cache poisoned"))?;
        if let Some(kernel) = cache.get(key) {
            if kernel.read_only.as_ref() != read_only {
                return Err(RuntimeError::with_kind(
                    ErrorKind::InvalidInput,
                    "resident kernel bindings differ from the cached layout",
                ));
            }
            return Ok(kernel.clone());
        }
        if read_only.len() >= 30 {
            return Err(RuntimeError::with_kind(
                ErrorKind::InvalidInput,
                "resident shader storage binding count exceeds its ABI",
            ));
        }
        let pipeline = match source {
            KernelSource::Wgsl(source) => {
                validate_wgsl_layout(source, read_only)?;
                metal_execution::compile_wgsl(context, source, "main", [0; 3])?
            }
            KernelSource::Native(shader) => metal_execution::compile_resident_msl(
                context,
                shader.source,
                shader.entry_point,
                shader.workgroup_size,
                read_only.len() + 1,
            )?,
        };
        let kernel = Arc::new(Self {
            ledger: context.ledger.clone(),
            pipeline,
            read_only: read_only.into(),
        });
        cache.insert(key.into(), kernel.clone());
        Ok(kernel)
    }

    fn validate_native_bindings(
        &self,
        context: &GpuContext,
        buffers: &[(&GpuBuffer, bool)],
        params: &GpuBuffer,
    ) -> RuntimeResult<()> {
        context.check_alive()?;
        if buffers.len() != self.read_only.len()
            || buffers
                .iter()
                .zip(self.read_only.iter())
                .any(|((_, actual), expected)| actual != expected)
        {
            return Err(RuntimeError::with_kind(
                ErrorKind::InvalidInput,
                "resident native bindings differ from the compiled layout",
            ));
        }
        for (index, (buffer, read_only)) in buffers.iter().enumerate() {
            // Reservations are private and created only with their context's
            // actual device, so ledger identity also proves device identity.
            if !Arc::ptr_eq(&buffer.reservation.ledger, &context.ledger)
                || !buffer.usage().contains(wgpu::BufferUsages::STORAGE)
            {
                return Err(RuntimeError::with_kind(
                    ErrorKind::InvalidInput,
                    format!(
                        "resident native storage binding {index} has the wrong context or usage"
                    ),
                ));
            }
            let bytes = buffer.size();
            if bytes == 0
                || !bytes.is_multiple_of(4)
                || bytes > context.limits.max_storage_buffer_binding_size
            {
                return Err(resource(format!(
                    "resident native storage binding {index} exceeds binding size limits or alignment"
                )));
            }
            // Preserve the declared resident storage access contract: a
            // readonly/writeable union conflicts within a dispatch scope.
            // Actual allocation identity includes cloned and scratch views.
            if buffers[..index].iter().any(|(previous, previous_read)| {
                previous_read != read_only
                    && Arc::ptr_eq(&previous.reservation, &buffer.reservation)
            }) {
                return Err(RuntimeError::with_kind(
                    ErrorKind::InvalidInput,
                    "resident native bindings have conflicting read/write aliases",
                ));
            }
        }
        if !Arc::ptr_eq(&params.reservation.ledger, &context.ledger)
            || !params.usage().contains(wgpu::BufferUsages::UNIFORM)
            || params.size() != 1024
            || params.size() > context.limits.max_uniform_buffer_binding_size
        {
            return Err(RuntimeError::with_kind(
                ErrorKind::InvalidInput,
                "resident native parameters require a same-context 1 KiB uniform snapshot",
            ));
        }
        // All bindings cover the entire allocation at offset zero. Imported
        // buffers are fully initialized before publication and storage buffers
        // are never mapped or explicitly destroyed by this private executor.
        // Their owned native handles and reservations survive queue completion.
        Ok(())
    }
}

fn validate_wgsl_layout(source: &str, read_only: &[bool]) -> RuntimeResult<()> {
    let module = naga::front::wgsl::parse_str(source).map_err(|error| {
        RuntimeError::with_kind(ErrorKind::GpuValidation, error.emit_to_string(source))
    })?;
    for (_, global) in module.global_variables.iter() {
        let Some(binding) = global.binding else {
            continue;
        };
        let valid = binding.group == 0
            && match global.space {
                naga::AddressSpace::Storage { access } => read_only
                    .get(binding.binding as usize)
                    .is_some_and(|&read| read == !access.contains(naga::StorageAccess::STORE)),
                naga::AddressSpace::Uniform => binding.binding as usize == read_only.len(),
                _ => false,
            };
        if !valid {
            return Err(RuntimeError::with_kind(
                ErrorKind::InvalidInput,
                "resident WGSL buffer declarations differ from the declared storage/uniform ABI",
            ));
        }
    }
    Ok(())
}

/// Compile once before admitting a native resident operation. Unsupported
/// compiler/shader capabilities retain the generic native GPU kernel; device loss,
/// allocation failures and poisoned state remain visible to the caller.
pub(crate) fn probe_native(
    context: &GpuContext,
    name: &str,
    shader: NativeShader<'_>,
    read_only: &[bool],
) -> RuntimeResult<bool> {
    context.check_alive()?;
    match Kernel::get_native(context, name, shader, read_only) {
        Ok(_) => Ok(true),
        Err(error)
            if matches!(
                error.kind(),
                ErrorKind::Unsupported | ErrorKind::GpuValidation
            ) =>
        {
            context.check_alive()?;
            Ok(false)
        }
        Err(error) => Err(error),
    }
}

pub(crate) struct Batch {
    graphs: Vec<Arc<AtomicUsize>>,
    pub context: Arc<GpuContext>,
    encoder: Option<wgpu::CommandEncoder>,
    schedule: Schedule,
    graph_parameters: std::collections::HashSet<usize>,
    held: Vec<GpuBuffer>,
    pub statistics: GpuPipelineStatistics,
    pub profiling: bool,
    profile: GpuExecutionProfile,
}
/// A submitted batch owns its reservations until the queue completes, including
/// when its caller returns an error or abandons the result. The borrowed gate
/// prevents admission from being released before the outstanding work drains.
pub(crate) struct PendingRead<'a> {
    batch: Batch,
    readback: GpuBuffer,
    bytes: usize,
    submission: Option<wgpu::SubmissionIndex>,
    mapping_requested: bool,
    receive: mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    _gate: &'a Gate<'a>,
}
impl PendingRead<'_> {
    fn wait(&mut self) -> RuntimeResult<()> {
        let Some(submission) = self.submission.take() else {
            return Ok(());
        };
        let timer = self.batch.profiling.then(std::time::Instant::now);
        let result = self.batch.context.device.poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(std::time::Duration::from_secs(60)),
        });
        if let Some(timer) = timer {
            self.batch.statistics.completion_wait_ns += timer.elapsed().as_nanos() as u64;
        }
        if let Err(error) = result {
            let message = format!("GPU completion wait failed: {error}");
            self.batch.context.fail(message.clone());
            return Err(RuntimeError::with_kind(ErrorKind::DeviceLost, message));
        }
        self.batch.context.check_alive()
    }
    pub fn finish(self) -> RuntimeResult<(Vec<u8>, GpuPipelineStatistics)> {
        self.finish_with(|bytes| Ok(bytes.to_vec()))
    }
    pub fn finish_with<T>(
        mut self,
        consume: impl FnOnce(&[u8]) -> RuntimeResult<T>,
    ) -> RuntimeResult<(T, GpuPipelineStatistics)> {
        self.wait()?;
        self.receive
            .try_recv()
            .map_err(|e| {
                RuntimeError::caused_by(
                    ErrorKind::GpuMapping,
                    "resident mapping callback unavailable",
                    e,
                )
            })?
            .map_err(|e| {
                RuntimeError::caused_by(ErrorKind::GpuMapping, "resident output mapping failed", e)
            })?;
        let timer = self.batch.profiling.then(std::time::Instant::now);
        let result = {
            let view = self
                .readback
                .slice(..self.bytes as u64)
                .get_mapped_range()
                .map_err(|e| {
                    RuntimeError::caused_by(
                        ErrorKind::GpuMapping,
                        "resident mapped output access failed",
                        e,
                    )
                })?;
            consume(&view)?
        };
        if let Some(timer) = timer {
            self.batch.statistics.readback_ns += timer.elapsed().as_nanos() as u64;
        }
        if self.batch.profiling {
            self.batch.statistics.profile = Some(std::mem::take(&mut self.batch.profile));
        }
        self.batch.statistics.peak_device_bytes = self.batch.context.memory_statistics().peak_bytes;
        Ok((result, std::mem::take(&mut self.batch.statistics)))
    }
}
impl Drop for PendingRead<'_> {
    fn drop(&mut self) {
        // Normal early returns drain their submission. Native resources also
        // have a queue-completion owner, so a poll timeout cannot release
        // resources still referenced by an unretained Metal command buffer.
        let _ = self.wait();
        if self.mapping_requested {
            self.readback.unmap();
        }
    }
}
impl Batch {
    pub fn new(context: Arc<GpuContext>) -> RuntimeResult<Self> {
        context.check_alive()?;
        let encoder = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("resident OCR"),
            });
        let statistics = GpuPipelineStatistics {
            adapter: context.info.name.clone(),
            backend: format!("{:?}", context.info.backend),
            driver: context.info.driver.clone(),
            driver_info: context.info.driver_info.clone(),
            ..Default::default()
        };
        Ok(Self {
            context,
            graphs: Vec::new(),
            encoder: Some(encoder),
            schedule: Schedule::default(),
            graph_parameters: std::collections::HashSet::new(),
            held: Vec::new(),
            statistics,
            profiling: false,
            profile: GpuExecutionProfile::default(),
        })
    }
    pub fn ensure_context(&self, context: &Arc<GpuContext>) -> RuntimeResult<()> {
        if !Arc::ptr_eq(&self.context, context) {
            return Err(RuntimeError::with_kind(
                ErrorKind::InvalidInput,
                "resident resource belongs to another GPU context",
            ));
        }
        Ok(())
    }
    pub fn storage(&mut self, label: &str, bytes: usize) -> RuntimeResult<GpuBuffer> {
        let buffer = storage(&self.context, label, bytes)?;
        self.statistics.scratch_allocations += 1;
        self.statistics.scratch_allocated_bytes += buffer.bytes() as u64;
        Ok(buffer)
    }
    pub fn dispatch(
        &mut self,
        name: &str,
        source: &str,
        buffers: &[(&GpuBuffer, bool)],
        words: &[u32],
        groups: [u32; 3],
    ) -> RuntimeResult<()> {
        let timer = self.profiling.then(std::time::Instant::now);
        if !self.validate_dispatch(words, groups)? {
            return Ok(());
        }
        let kernel = Kernel::get(
            &self.context,
            name,
            source,
            &buffers.iter().map(|b| b.1).collect::<Vec<_>>(),
        )?;
        self.dispatch_kernel(name, &kernel, buffers, words, groups, timer)
    }

    /// Record a kernel admitted before choosing an optional arithmetic policy.
    /// Keep dispatch validation separate from compilation fallback decisions.
    pub(super) fn dispatch_prepared(
        &mut self,
        name: &str,
        kernel: &Kernel,
        buffers: &[(&GpuBuffer, bool)],
        words: &[u32],
        groups: [u32; 3],
    ) -> RuntimeResult<()> {
        let timer = self.profiling.then(std::time::Instant::now);
        if !self.validate_dispatch(words, groups)? {
            return Ok(());
        }
        if !Arc::ptr_eq(&kernel.ledger, &self.context.ledger) {
            return Err(RuntimeError::with_kind(
                ErrorKind::InvalidInput,
                "prepared resident kernel belongs to another GPU context",
            ));
        }
        if kernel.read_only.len() != buffers.len()
            || kernel
                .read_only
                .iter()
                .zip(buffers)
                .any(|(&expected, (_, actual))| expected != *actual)
        {
            return Err(RuntimeError::with_kind(
                ErrorKind::InvalidInput,
                "prepared resident kernel bindings differ from its layout",
            ));
        }
        self.dispatch_kernel(name, kernel, buffers, words, groups, timer)
    }

    /// Record native Metal work over the existing resident buffers. This shares
    /// parameter packets, binding ownership, hazards, profiling and submission
    /// with `dispatch`; it introduces no activation copy or intermediate tensor.
    pub fn dispatch_native(
        &mut self,
        name: &str,
        shader: NativeShader<'_>,
        buffers: &[(&GpuBuffer, bool)],
        words: &[u32],
        groups: [u32; 3],
    ) -> RuntimeResult<()> {
        let timer = self.profiling.then(std::time::Instant::now);
        if !self.validate_dispatch(words, groups)? {
            return Ok(());
        }
        let kernel = Kernel::get_native(
            &self.context,
            name,
            shader,
            &buffers.iter().map(|b| b.1).collect::<Vec<_>>(),
        )?;
        self.dispatch_kernel(name, &kernel, buffers, words, groups, timer)
    }

    fn validate_dispatch(&self, words: &[u32], groups: [u32; 3]) -> RuntimeResult<bool> {
        if groups.contains(&0) {
            return Ok(false);
        }
        if groups
            .iter()
            .any(|&x| x > self.context.limits.max_compute_workgroups_per_dimension)
            || words.len() > 256
        {
            return Err(resource("resident dispatch exceeds GPU limits"));
        }
        Ok(true)
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_kernel(
        &mut self,
        name: &str,
        kernel: &Kernel,
        buffers: &[(&GpuBuffer, bool)],
        words: &[u32],
        groups: [u32; 3],
        timer: Option<std::time::Instant>,
    ) -> RuntimeResult<()> {
        let mut packet = [0u32; 256];
        packet[..words.len()].copy_from_slice(words);
        packet[255] = u32::MAX;
        let params = self.context.create_buffer_init(
            "resident parameters",
            bytemuck::cast_slice(&packet),
            wgpu::BufferUsages::UNIFORM,
        )?;
        kernel.validate_native_bindings(&self.context, buffers, &params)?;
        let native = metal_execution::NativeDispatch::prepare(
            kernel.pipeline.clone(),
            buffers
                .iter()
                .map(|(buffer, _)| (*buffer).clone())
                .chain(std::iter::once(params.clone()))
                .collect(),
            vec![0; buffers.len() + 1],
            groups,
            &kernel.read_only,
        )?;
        if self.profiling {
            let stage = if name.starts_with("db_") {
                "geometry"
            } else if name.starts_with("projection_") {
                "projection"
            } else if name.starts_with("ctc_") {
                "decoding"
            } else {
                "image"
            };
            self.profile.kernels.push(GpuKernelProfile {
                stage: stage.into(),
                nodes: Vec::new(),
                kernel: if name.starts_with("db_label_") {
                    format!("{name}/phase{}", words[4])
                } else {
                    name.into()
                },
                output_shape: Vec::new(),
                dispatch_count: 1,
                device_ns: None,
                granularity: GpuTimingGranularity::Unavailable,
            });
        }
        self.statistics.native_resident_dispatches += 1;
        self.schedule.push_prepared(
            Operation::Compute {
                native: native.clone(),
            },
            native.hazards(),
        );
        self.held.extend(buffers.iter().map(|(b, _)| (*b).clone()));
        self.held.push(params);
        self.statistics.dispatches += 1;
        if let Some(timer) = timer {
            self.statistics.encoding_ns += timer.elapsed().as_nanos() as u64;
        }
        if !self.statistics.kernels.iter().any(|k| k == name) {
            self.statistics.kernels.push(name.into());
        }
        if name.starts_with("image_")
            || name.starts_with("perspective_")
            || name == "classifier_rotation"
        {
            self.statistics.image_processing = GpuPlacement::Gpu;
        }
        if name.starts_with("db_") {
            self.statistics.geometry = GpuPlacement::Gpu;
        }
        if name.starts_with("projection_") {
            self.statistics.projection = GpuPlacement::Gpu;
        }
        if name.starts_with("ctc_") {
            self.statistics.decoding = GpuPlacement::Gpu;
        }
        Ok(())
    }
    pub fn graph(&mut self, session: &mut GpuSession, stage: &str) -> RuntimeResult<Tensor> {
        self.graph_selected(session, stage, None)
    }
    pub fn graph_node(
        &mut self,
        session: &mut GpuSession,
        node: usize,
        stage: &str,
    ) -> RuntimeResult<Tensor> {
        if node >= session.plan.steps.len() {
            return Err(RuntimeError::with_kind(
                ErrorKind::InvalidInput,
                "native diagnostic node is outside the plan",
            ));
        }
        self.graph_selected(session, stage, Some(node))
    }
    fn graph_selected(
        &mut self,
        session: &mut GpuSession,
        stage: &str,
        node: Option<usize>,
    ) -> RuntimeResult<Tensor> {
        let timer = self.profiling.then(std::time::Instant::now);
        self.ensure_context(&session.compiled.context)?;
        if session.parameters_dirty {
            if self
                .graph_parameters
                .contains(&schedule::identity(&session.parameters))
            {
                return Err(RuntimeError::with_kind(
                    ErrorKind::InvalidInput,
                    "resident session must be submitted before reshaping it again",
                ));
            }
            self.context
                .queue
                .write_buffer(&session.parameters, 0, &session.parameter_data);
            session.parameters_dirty = false;
        }
        self.graph_parameters
            .insert(schedule::identity(&session.parameters));
        for (dispatch, recipe) in session
            .dispatches
            .iter()
            .zip(&session.recipes)
            .filter(|(dispatch, _)| node.is_none_or(|n| dispatch.node == n))
        {
            match &dispatch.command {
                Command::Compute { native, shader } => {
                    self.statistics.native_graph_dispatches += 1;
                    self.statistics.dispatches += 1;
                    if session.compiled.packed_operand_precision(recipe).is_some() {
                        self.held.push(native.owners()[1].clone());
                    }
                    self.schedule.push_prepared(
                        Operation::Compute {
                            native: native.clone(),
                        },
                        native.hazards(),
                    );
                    if self.profiling {
                        self.profile.kernels.push(GpuKernelProfile {
                            stage: stage.into(),
                            nodes: dispatch
                                .nodes
                                .iter()
                                .flat_map(|&n| {
                                    session.compiled.model().graph().nodes[n]
                                        .source_indices
                                        .clone()
                                })
                                .collect(),
                            kernel: shader.name().into(),
                            output_shape: session.plan.shapes[dispatch.output.0]
                                .dimensions()
                                .to_vec(),
                            dispatch_count: 1,
                            device_ns: None,
                            granularity: GpuTimingGranularity::Unavailable,
                        });
                    }
                    self.statistics.layout_conversions += u64::from(
                        matches!(
                            recipe.kind,
                            operators::RecipeKind::Compute(operators::ShaderKind::PackNhwc)
                        ) || (matches!(
                            recipe.kind,
                            operators::RecipeKind::Compute(operators::ShaderKind::General)
                        ) && recipe.words[2] == 24),
                    );
                }
                Command::Copy {
                    source,
                    output,
                    source_offset,
                    destination_offset,
                    bytes,
                } => self.copy(
                    &session.resident_buffer(*source),
                    *source_offset,
                    &session.resident_buffer(*output),
                    *destination_offset,
                    *bytes,
                ),
            }
        }
        self.held.extend(session.buffers.iter().cloned());
        self.held.push(session.parameters.clone());
        self.held
            .extend(session.compiled.weights.constants.iter().flatten().cloned());
        self.held.push(session.compiled.weights.dummy.clone());
        if node.is_none() {
            self.graphs.push(session.submissions.clone());
        }
        if let Some(timer) = timer {
            self.statistics.encoding_ns += timer.elapsed().as_nanos() as u64;
        }
        Ok(session.resident_tensor(
            node.map_or(session.selected_output, |n| session.plan.steps[n].output),
        ))
    }
    pub fn copy(
        &mut self,
        source: &GpuBuffer,
        source_offset: u64,
        output: &GpuBuffer,
        output_offset: u64,
        bytes: u64,
    ) {
        self.schedule.push(
            Operation::Copy {
                source: source.clone(),
                output: output.clone(),
                source_offset,
                output_offset,
                bytes,
            },
            &[(source, true), (output, false)],
        );
        self.held.extend([source.clone(), output.clone()]);
    }
    pub fn read(
        mut self,
        source: &GpuBuffer,
        bytes: usize,
        control: bool,
    ) -> RuntimeResult<(Vec<u8>, GpuPipelineStatistics)> {
        let context = self.context.clone();
        let admission = self.profiling.then(std::time::Instant::now);
        let gate = context.acquire_gate()?;
        if let Some(timer) = admission {
            self.statistics.admission_wait_ns = timer.elapsed().as_nanos() as u64;
        }
        self.submit(source, bytes, control, &gate)?.finish()
    }
    pub(crate) fn submit<'a>(
        self,
        source: &GpuBuffer,
        bytes: usize,
        control: bool,
        gate: &'a Gate<'a>,
    ) -> RuntimeResult<PendingRead<'a>> {
        if !std::ptr::eq(self.context.as_ref(), gate.context) {
            return Err(RuntimeError::with_kind(
                ErrorKind::InvalidInput,
                "resident admission belongs to another GPU context",
            ));
        }
        if bytes == 0 || !bytes.is_multiple_of(4) || bytes as u64 > source.size() {
            return Err(resource("invalid resident readback size"));
        }
        let readback = self.context.create_buffer(
            "resident compact readback",
            bytes as u64,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        )?;
        self.submit_with_readback(source, readback, bytes, control, gate)
    }
    pub(crate) fn read_into<'a, T>(
        self,
        source: &GpuBuffer,
        readback: &GpuBuffer,
        bytes: usize,
        gate: &'a Gate<'a>,
        consume: impl FnOnce(&[u8]) -> RuntimeResult<T>,
    ) -> RuntimeResult<(T, GpuPipelineStatistics)> {
        self.submit_with_readback(source, readback.clone(), bytes, false, gate)?
            .finish_with(consume)
    }
    fn submit_with_readback<'a>(
        mut self,
        source: &GpuBuffer,
        readback: GpuBuffer,
        bytes: usize,
        control: bool,
        gate: &'a Gate<'a>,
    ) -> RuntimeResult<PendingRead<'a>> {
        if !std::ptr::eq(self.context.as_ref(), gate.context)
            || !Arc::ptr_eq(&readback.reservation.ledger, &self.context.ledger)
            || !Arc::ptr_eq(&source.reservation.ledger, &self.context.ledger)
        {
            return Err(RuntimeError::with_kind(
                ErrorKind::InvalidInput,
                "resident readback belongs to another GPU context or admission",
            ));
        }
        if bytes == 0
            || !bytes.is_multiple_of(4)
            || bytes as u64 > source.size()
            || bytes as u64 > readback.size()
            || !readback
                .usage()
                .contains(wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ)
        {
            return Err(resource("invalid resident readback size or usage"));
        }
        self.copy(source, 0, &readback, 0, bytes as u64);
        let encoding = self.profiling.then(std::time::Instant::now);
        let mut commands = self.schedule.encode(
            &self.context,
            self.encoder.as_mut().unwrap(),
            self.graphs.len() == 1,
        )?;
        if let Some(timer) = encoding {
            self.statistics.encoding_ns += timer.elapsed().as_nanos() as u64;
        }
        let context = self.context.clone();
        let encoder = self.encoder.take().unwrap();
        let timer = self.profiling.then(std::time::Instant::now);
        let (send, receive) = mpsc::sync_channel(1);
        // Preserve the submission before checking error scopes. If validation
        // fails, install the owner before propagating the error so Drop drains it.
        let mut submission = None;
        let mut mapping_requested = false;
        let submitted = context.checked("resident submission", || {
            for graph in &self.graphs {
                graph.fetch_add(1, Ordering::Relaxed);
            }
            commands.push(encoder.finish());
            submission = Some(context.queue.submit(commands));
            let native = self.schedule.take_native_resources();
            let held = std::mem::take(&mut self.held);
            context.queue.on_submitted_work_done(move || {
                drop(native);
                drop(held);
            });
            readback
                .slice(..bytes as u64)
                .map_async(wgpu::MapMode::Read, move |result| {
                    let _ = send.send(result);
                });
            mapping_requested = true;
            Ok(())
        });
        if let Some(timer) = timer {
            self.statistics.submission_ns = timer.elapsed().as_nanos() as u64;
        }
        self.statistics.submissions += 1;
        if control {
            self.statistics.control_readback_bytes += bytes as u64;
        } else {
            self.statistics.output_readback_bytes += bytes as u64;
        }
        let pending = PendingRead {
            batch: self,
            readback,
            bytes,
            submission,
            mapping_requested,
            receive,
            _gate: gate,
        };
        submitted?;
        Ok(pending)
    }
}

impl GpuSession {
    fn resident_buffer(&self, id: TensorId) -> GpuBuffer {
        self.compiled
            .weights
            .constants
            .get(id.0)
            .and_then(Option::as_ref)
            .unwrap_or_else(|| &self.buffers[self.plan.slots[id.0]])
            .clone()
    }
    fn resident_tensor(&self, id: TensorId) -> Tensor {
        Tensor {
            context: self.compiled.context.clone(),
            buffer: self.resident_buffer(id),
            descriptor: self.descriptors[id.0].clone(),
        }
    }
    pub(crate) fn resident_input(&self) -> Tensor {
        self.resident_tensor(self.compiled.program.input)
    }
}

/// Runtime shader reload is restricted to explicit physical-device development tests.
pub(crate) fn shader(name: &str, embedded: &'static str) -> std::borrow::Cow<'static, str> {
    #[cfg(test)]
    if std::env::var("SIMD_OCR_GPU_SHADER_RELOAD").as_deref() == Ok("1") {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/runtime/gpu")
            .join(name);
        return std::fs::read_to_string(path)
            .expect("developer shader source")
            .into();
    }
    let _ = name;
    embedded.into()
}

pub(crate) const PARAMETERS: &str = "struct Params { words: array<vec4<u32>,64> };\n";

pub(super) struct Normalization {
    buffers: [GpuBuffer; 2],
}
impl Normalization {
    pub(super) fn bytes(&self) -> usize {
        self.buffers.iter().map(GpuBuffer::bytes).sum()
    }
    pub(super) fn shared(context: &GpuContext) -> RuntimeResult<Arc<Self>> {
        let mut cache = context
            .normalization
            .lock()
            .map_err(|_| RuntimeError::new("normalization cache poisoned"))?;
        if let Some(constants) = cache.upgrade() {
            return Ok(constants);
        }
        let make = |detector: bool| {
            let mut values = Vec::with_capacity(768);
            for c in 0..3 {
                for x in 0..256 {
                    values.push(if detector {
                        ((x as f64 / 255. - [0.485, 0.456, 0.406][c])
                            * (1. / [0.229, 0.224, 0.225][c])) as f32
                    } else {
                        x as f32 * (2.0f32 / 255.0f32) - 1.
                    });
                }
            }
            context.checked("normalization constants", || {
                context.create_buffer_init(
                    "normalization constants",
                    bytemuck::cast_slice(&values),
                    wgpu::BufferUsages::STORAGE,
                )
            })
        };
        let constants = Arc::new(Self {
            buffers: [make(false)?, make(true)?],
        });
        *cache = Arc::downgrade(&constants);
        Ok(constants)
    }
}

/// Byte-domain resize uses the same 11-bit coefficients and two-stage integer rounding as CPU.
pub(crate) fn preprocess(
    batch: &mut Batch,
    image: &Image,
    output: &Tensor,
    source: [usize; 2],
    content_width: usize,
    detector: bool,
) -> RuntimeResult<()> {
    preprocess_at(batch, image, output, source, content_width, detector, 0)
}
pub(crate) fn preprocess_at(
    batch: &mut Batch,
    image: &Image,
    output: &Tensor,
    source: [usize; 2],
    content_width: usize,
    detector: bool,
    sample: usize,
) -> RuntimeResult<()> {
    batch.ensure_context(&image.context)?;
    batch.ensure_context(&output.context)?;
    let shape = output.descriptor.shape.dimensions();
    if shape.len() != 4
        || sample >= shape[0]
        || shape[1] != 3
        || output.descriptor.layout != GpuLayout::Canonical
        || content_width > shape[3]
        || content_width == 0
    {
        return Err(RuntimeError::with_kind(
            ErrorKind::InvalidInput,
            "invalid resident preprocessing shape",
        ));
    }
    let precision = output.descriptor.precision;
    let source_code = format!(
        "{}{}{}",
        if precision == GpuPrecision::MixedF16 {
            "enable f16;\n"
        } else {
            ""
        },
        PARAMETERS,
        shader("image.wgsl", include_str!("image.wgsl")).replace(
            "$OUT",
            if precision == GpuPrecision::MixedF16 {
                "f16"
            } else {
                "f32"
            }
        )
    );
    let constants = Normalization::shared(&batch.context)?;
    let lut = &constants.buffers[usize::from(detector)];
    let words = [
        image.width as u32,
        image.height as u32,
        image.stride as u32,
        image.format.bytes_per_pixel() as u32,
        u32::from(image.format.swap_red_blue()),
        source[0] as u32,
        source[1] as u32,
        content_width as u32,
        shape[2] as u32,
        shape[3] as u32,
        u32::from(detector),
        if !detector && shape[2] == 80 {
            (-1f32).to_bits()
        } else {
            0
        },
        (sample * 3 * shape[2] * shape[3]) as u32,
    ];
    let (x, y) = dispatch_grid(shape[2] * shape[3], &batch.context.limits)?;
    batch.dispatch(
        &format!("image_normalize_{precision:?}"),
        &source_code,
        &[(&image.buffer, true), (lut, true), (&output.buffer, false)],
        &words,
        [x, y, 1],
    )
}

#[cfg(test)]
#[path = "image_pixel_tests.rs"]
mod image_pixel_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_resident_wgsl_layout_validates_declared_access_and_uniform_slots() {
        let source = "@group(0) @binding(0) var<storage,read> a:array<u32>; @group(0) @binding(1) var<storage,read_write> b:array<u32>; @group(0) @binding(2) var<uniform> p:vec4<u32>; @compute @workgroup_size(1) fn main(){b[0]=a[0]+p.x;}";
        validate_wgsl_layout(source, &[true, false]).unwrap();
        for invalid in [
            source.to_owned(),
            source.replace("@binding(2)", "@binding(3)"),
            source.replace("@group(0)", "@group(1)"),
        ] {
            let read_only = if invalid == source {
                &[false, false][..]
            } else {
                &[true, false][..]
            };
            assert_eq!(
                validate_wgsl_layout(&invalid, read_only)
                    .unwrap_err()
                    .kind(),
                ErrorKind::InvalidInput
            );
        }
        assert_eq!(
            validate_wgsl_layout("invalid source", &[false])
                .unwrap_err()
                .kind(),
            ErrorKind::GpuValidation
        );
    }

    #[test]
    #[ignore = "requires physical Metal; reused readback, once-only consumers and foreign context validation"]
    fn gpu_resident_native_read_into_reuses_storage_and_recovers_after_consumer_error() {
        let context = GpuContext::new().unwrap();
        let foreign = GpuContext::new().unwrap();
        let input = storage(&context, "native read_into counter", 4).unwrap();
        let readback = context
            .create_buffer(
                "reused host readback",
                64,
                wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            )
            .unwrap();
        let identity = readback.reservation.clone();
        let used = context.memory_statistics().used_bytes;
        let source = "@group(0) @binding(0) var<storage,read_write> value:array<u32>; @compute @workgroup_size(1) fn main(){value[0]+=1u;}";
        let gate = context.acquire_gate().unwrap();
        for value in [23u32, 41] {
            context
                .upload_resident(&input, bytemuck::bytes_of(&value))
                .unwrap();
            let mut batch = Batch::new(context.clone()).unwrap();
            batch.profiling = true;
            batch
                .dispatch(
                    "native_read_into_counter",
                    source,
                    &[(&input, false)],
                    &[],
                    [1, 1, 1],
                )
                .unwrap();
            let (actual, statistics) = batch
                .read_into(&input, &readback, 4, &gate, |bytes| {
                    Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
                })
                .unwrap();
            assert_eq!(actual, value + 1);
            assert_eq!(statistics.native_resident_dispatches, 1);
            assert_eq!(statistics.output_readback_bytes, 4);
            assert_eq!(statistics.profiling_readback_bytes, 0);
            assert!(statistics.profile.unwrap().compute_ns().is_none());
            assert!(Arc::ptr_eq(&identity, &readback.reservation));
            assert_eq!(context.memory_statistics().used_bytes, used);
        }
        let calls = std::cell::Cell::new(0);
        let failure = Batch::new(context.clone())
            .unwrap()
            .read_into(&input, &readback, 4, &gate, |_| -> RuntimeResult<()> {
                calls.set(calls.get() + 1);
                Err(RuntimeError::with_kind(
                    ErrorKind::InvalidInput,
                    "consumer stopped",
                ))
            })
            .unwrap_err();
        assert_eq!(failure.kind(), ErrorKind::InvalidInput);
        assert_eq!(calls.get(), 1);
        let (actual, _) = Batch::new(context.clone())
            .unwrap()
            .read_into(&input, &readback, 4, &gate, |bytes| {
                Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
            })
            .unwrap();
        assert_eq!(actual, 42);
        let foreign_storage = storage(&foreign, "foreign native storage", 4).unwrap();
        assert_eq!(
            context
                .upload_resident(&foreign_storage, &[0; 4])
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidInput,
            "uploads must not trust colliding wgpu device IDs",
        );
        let mut batch = Batch::new(context.clone()).unwrap();
        assert_eq!(
            batch
                .dispatch(
                    "native_read_into_counter",
                    source,
                    &[(&foreign_storage, false)],
                    &[],
                    [1, 1, 1]
                )
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidInput
        );
        let mut copy = Batch::new(context.clone()).unwrap();
        copy.copy(&foreign_storage, 0, &input, 0, 4);
        assert_eq!(
            copy.read_into(&input, &readback, 4, &gate, |_| Ok(()))
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidInput,
            "transfer recording must validate the same ledger identity before touching the bridge"
        );
        let foreign_readback = foreign
            .create_buffer(
                "foreign readback",
                4,
                wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            )
            .unwrap();
        assert_eq!(
            Batch::new(context.clone())
                .unwrap()
                .read_into(&input, &foreign_readback, 4, &gate, |_| Ok(()))
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidInput
        );
    }
    #[test]
    #[ignore = "requires a physical GPU"]
    fn gpu_resident_pending_readbacks_hold_admission_and_reservations() {
        let context = GpuContext::new_with_options(GpuContextOptions {
            max_in_flight: 1,
            max_device_bytes: Some(4096),
            ..Default::default()
        })
        .unwrap();
        let gate = context.acquire_gate().unwrap();
        let source = storage(&context, "pending source", 1024).unwrap();
        context.queue.write_buffer(&source, 0, &[7; 1024]);
        let first = Batch::new(context.clone())
            .unwrap()
            .submit(&source, 1024, false, &gate)
            .unwrap();
        context.queue.write_buffer(&source, 0, &[8; 1024]);
        let second = Batch::new(context.clone())
            .unwrap()
            .submit(&source, 1024, false, &gate)
            .unwrap();
        drop(source);
        assert_eq!(context.memory_statistics().used_bytes, 3072);
        assert_eq!(
            storage(&context, "over quota while pending", 2048)
                .err()
                .unwrap()
                .kind(),
            ErrorKind::ResourceLimit
        );
        assert!(
            context.acquire_gate().is_err(),
            "pending work retains reentrancy protection"
        );
        assert_eq!(
            first.finish().unwrap().0,
            vec![7; 1024],
            "submissions precede subsequent writes"
        );
        assert_eq!(second.finish().unwrap().0, vec![8; 1024]);
        assert_eq!(context.memory_statistics().used_bytes, 0);
        let early_error = (|| -> RuntimeResult<()> {
            let source = storage(&context, "abandoned source", 1024)?;
            let _pending = Batch::new(context.clone())?.submit(&source, 1024, false, &gate)?;
            storage(&context, "intentional quota error", 4096)?;
            Ok(())
        })();
        assert_eq!(early_error.unwrap_err().kind(), ErrorKind::ResourceLimit);
        assert_eq!(
            context.memory_statistics().used_bytes,
            0,
            "early error drains reservations"
        );
        drop(gate);
        let _next = context.acquire_gate().unwrap();
    }
    #[test]
    #[ignore = "requires a physical GPU"]
    fn gpu_resident_native_profiles_preserve_execution_and_honest_metadata() {
        let context = GpuContext::new_with_options(GpuContextOptions {
            ..Default::default()
        })
        .unwrap();
        let source = "@group(0) @binding(0) var<storage,read_write> value:array<u32>; @compute @workgroup_size(1) fn main(){value[0]+=1u;}";
        for profiling in [true, false] {
            let mut batch = Batch::new(context.clone()).unwrap();
            batch.profiling = profiling;
            let value = batch.storage("profile rollover counter", 4).unwrap();
            for _ in 0..260 {
                batch
                    .dispatch(
                        "profile_rollover",
                        source,
                        &[(&value, false)],
                        &[],
                        [1, 1, 1],
                    )
                    .unwrap();
            }
            let (bytes, statistics) = batch.read(&value, 4, false).unwrap();
            assert_eq!(u32::from_le_bytes(bytes.try_into().unwrap()), 260);
            assert_eq!(statistics.dispatches, 260);
            assert_eq!(statistics.native_resident_dispatches, 260);
            assert_eq!(statistics.diagnostic_readback_bytes, 0);
            assert_eq!(statistics.output_readback_bytes, 4);
            if profiling {
                let profile = statistics.profile.unwrap();
                assert_eq!(profile.kernels.len(), 260);
                assert!(profile.kernels.iter().all(|entry| entry.device_ns.is_none()
                    && entry.granularity == GpuTimingGranularity::Unavailable));
                assert_eq!(statistics.profiling_readback_bytes, 0);
                assert!(profile.compute_ns().is_none());
            } else {
                assert!(statistics.profile.is_none());
                assert_eq!(statistics.profiling_readback_bytes, 0);
            }
        }
    }
    #[test]
    #[ignore = "requires a physical GPU"]
    fn gpu_resident_preprocess_matches_byte_domain_cpu() {
        let context = GpuContext::new_with_options(GpuContextOptions {
            ..Default::default()
        })
        .unwrap();
        for format in [
            PixelFormat::Bgr24,
            PixelFormat::Rgb24,
            PixelFormat::Bgra32,
            PixelFormat::Rgba32,
        ] {
            for (width, height, dw, dh, content, det) in [
                (17, 13, 32, 32, 32, true),
                (37, 23, 160, 80, 129, false),
                (113, 17, 320, 48, 319, false),
                (1, 1, 32, 32, 32, true),
            ] {
                let stride = width * format.bytes_per_pixel() + 7;
                let pixels: Vec<u8> = (0..stride * height)
                    .map(|i| ((i * 53 + i / 7) % 256) as u8)
                    .collect();
                let image = ImageView::new(&pixels, width, height, stride, format).unwrap();
                let mut batch = Batch::new(context.clone()).unwrap();
                let uploaded = Image::upload(&mut batch, image).unwrap();
                let tensor = Tensor::allocate(
                    context.clone(),
                    Shape::new(&[1, 3, dh, dw]).unwrap(),
                    GpuPrecision::F32,
                )
                .unwrap();
                preprocess(
                    &mut batch,
                    &uploaded,
                    &tensor,
                    [width, height],
                    content,
                    det,
                )
                .unwrap();
                let (data, stats) = batch.read(&tensor.buffer, tensor.bytes(), false).unwrap();
                let mut actual = vec![0.; 3 * dh * dw];
                decode_f32(&data, &mut actual).unwrap();
                let mut expected = vec![if dh == 80 { -1. } else { 0. }; actual.len()];
                let mut scratch = crate::preprocess::ResizeWorkspace::default();
                if det {
                    crate::preprocess::det(image, dw, dh, &mut expected, &mut scratch).unwrap();
                } else {
                    // Exercise arbitrary content width using the canonical integer resize.
                    crate::preprocess::resize_for_gpu_test(
                        image,
                        content,
                        dh,
                        dw,
                        &mut expected,
                        &mut scratch,
                    )
                    .unwrap();
                }
                for (i, (&a, &b)) in actual.iter().zip(&expected).enumerate() {
                    assert!(
                        (a - b).abs() < 1e-6,
                        "{format:?} {width}x{height}: {i}: {a} != {b}"
                    );
                }
                assert_eq!(stats.submissions, 1);
                assert_eq!(
                    stats.uploaded_bytes,
                    ((height - 1) * stride + width * format.bytes_per_pixel()) as u64
                );
            }
        }
    }
}
