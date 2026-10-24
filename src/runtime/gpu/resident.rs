//! Device-resident composition. A batch owns every allocation it records until completion.
use super::*;
use crate::image::{ImageView, PixelFormat};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub enum GpuPipelineMode {
    #[default]
    Full,
    InferenceOnly,
}

#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct GpuPipelineStatistics {
    pub mode: GpuPipelineMode,
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

/// Exclusive-request cache. Only allocations with no live view or submitted
/// command owner may be reused. Batches retain reservations through completion.
#[derive(Default)]
pub(crate) struct ScratchPool {
    state: Mutex<ScratchState>,
}
#[derive(Default)]
struct ScratchState {
    context: usize,
    buffers: Vec<GpuBuffer>,
}
impl ScratchPool {
    pub fn retained_bytes(&self) -> usize {
        self.state
            .lock()
            .map(|s| s.buffers.iter().map(GpuBuffer::bytes).sum())
            .unwrap_or(0)
    }
    fn allocate(
        &self,
        context: &GpuContext,
        label: &str,
        bytes: usize,
    ) -> RuntimeResult<(GpuBuffer, bool)> {
        let bytes = bytes
            .checked_add(3)
            .ok_or_else(|| resource("scratch size overflow"))?
            / 4
            * 4;
        let mut state = self
            .state
            .lock()
            .map_err(|_| RuntimeError::new("scratch cache poisoned"))?;
        let identity = context as *const GpuContext as usize;
        if state.context != 0 && state.context != identity {
            return Err(RuntimeError::with_kind(
                ErrorKind::InvalidInput,
                "scratch cache belongs to another GPU context",
            ));
        }
        state.context = identity;
        let free = |b: &GpuBuffer| Arc::strong_count(&b.reservation) == 1;
        if let Some(index) = state
            .buffers
            .iter()
            .enumerate()
            .filter(|(_, b)| {
                free(b) && b.bytes() >= bytes && b.bytes() <= bytes.saturating_mul(2).max(4096)
            })
            .min_by_key(|(_, b)| b.bytes())
            .map(|(i, _)| i)
        {
            return Ok((state.buffers[index].clone(), true));
        }
        let cap = context
            .options
            .max_device_bytes
            .map_or(64 << 20, |n| (n / 4).min(64 << 20));
        let retained = state.buffers.iter().map(GpuBuffer::bytes).sum::<usize>();
        let quota_pressure = context
            .options
            .max_device_bytes
            .is_some_and(|n| context.memory_statistics().used_bytes.saturating_add(bytes) > n);
        if retained.saturating_add(bytes) > cap || state.buffers.len() >= 64 || quota_pressure {
            state.buffers.retain(|b| !free(b));
        }
        let buffer = storage(context, label, bytes)?;
        if state.buffers.len() < 64
            && state
                .buffers
                .iter()
                .map(GpuBuffer::bytes)
                .sum::<usize>()
                .saturating_add(bytes)
                <= cap
        {
            state.buffers.push(buffer.clone());
        }
        Ok((buffer, false))
    }
}

pub(super) struct Kernel {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}
impl Kernel {
    fn get(
        context: &GpuContext,
        name: &str,
        source: &str,
        read_only: &[bool],
    ) -> RuntimeResult<Arc<Self>> {
        let mut cache = context
            .resident_pipelines
            .lock()
            .map_err(|_| RuntimeError::new("resident pipeline cache poisoned"))?;
        if let Some(kernel) = cache.get(name) {
            return Ok(kernel.clone());
        }
        let kernel = context.checked(name, || {
            let mut entries: Vec<_> = read_only
                .iter()
                .enumerate()
                .map(|(binding, &read_only)| wgpu::BindGroupLayoutEntry {
                    binding: binding as u32,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    count: None,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                })
                .collect();
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: read_only.len() as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                count: None,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            });
            let layout =
                context
                    .device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some(name),
                        entries: &entries,
                    });
            let pipeline_layout =
                context
                    .device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some(name),
                        bind_group_layouts: &[Some(&layout)],
                        immediate_size: 0,
                    });
            let shader = context
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(name),
                    source: wgpu::ShaderSource::Wgsl(source.into()),
                });
            let pipeline =
                context
                    .device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some(name),
                        layout: Some(&pipeline_layout),
                        module: &shader,
                        entry_point: Some("main"),
                        compilation_options: Default::default(),
                        cache: None,
                    });
            Ok(Arc::new(Self { layout, pipeline }))
        })?;
        cache.insert(name.into(), kernel.clone());
        Ok(kernel)
    }
}

pub(crate) struct Batch {
    pub scratch: Option<Arc<ScratchPool>>,
    graphs: Vec<Arc<AtomicUsize>>,
    pub context: Arc<GpuContext>,
    encoder: Option<wgpu::CommandEncoder>,
    held: Vec<GpuBuffer>,
    pub statistics: GpuPipelineStatistics,
    pub profiling: bool,
    profiles: Vec<ProfilePage>,
}
struct ProfilePage {
    storage: Option<profile::TimestampStorage>,
    report: GpuExecutionProfile,
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
    pub fn finish(mut self) -> RuntimeResult<(Vec<u8>, GpuPipelineStatistics)> {
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
        let data = self
            .readback
            .slice(..self.bytes as u64)
            .get_mapped_range()
            .map_err(|e| {
                RuntimeError::caused_by(
                    ErrorKind::GpuMapping,
                    "resident mapped output access failed",
                    e,
                )
            })?
            .to_vec();
        if let Some(timer) = timer {
            self.batch.statistics.readback_ns += timer.elapsed().as_nanos() as u64;
        }
        if self.batch.profiling {
            let mut profile = GpuExecutionProfile::default();
            for page in &mut self.batch.profiles {
                if let Some(storage) = page.storage.take() {
                    storage.finish(&self.batch.context, &mut page.report)?;
                }
                profile.kernels.append(&mut page.report.kernels);
            }
            self.batch.statistics.profile = Some(profile);
        }
        self.batch.statistics.peak_device_bytes = self.batch.context.memory_statistics().peak_bytes;
        Ok((data, std::mem::take(&mut self.batch.statistics)))
    }
}
impl Drop for PendingRead<'_> {
    fn drop(&mut self) {
        // Never release live reservations on an early return. A poll failure
        // poisons the context, so no later allocation can reuse that budget.
        let _ = self.wait();
        if self.mapping_requested {
            self.readback.unmap();
            for page in &self.batch.profiles {
                if let Some(storage) = &page.storage {
                    storage.unmap();
                }
            }
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
            scratch: None,
            graphs: Vec::new(),
            encoder: Some(encoder),
            held: Vec::new(),
            statistics,
            profiling: false,
            profiles: Vec::new(),
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
        let pool = self
            .scratch
            .as_ref()
            .filter(|_| self.context.options.cache_resident_scratch);
        let (buffer, reused) = if let Some(pool) = pool {
            pool.allocate(&self.context, label, bytes)?
        } else {
            (storage(&self.context, label, bytes)?, false)
        };
        if reused {
            self.statistics.scratch_reuses += 1;
            self.encoder
                .as_mut()
                .unwrap()
                .clear_buffer(&buffer, 0, None);
            self.held.push(buffer.clone());
        } else {
            self.statistics.scratch_allocations += 1;
            self.statistics.scratch_allocated_bytes += buffer.bytes() as u64;
        }
        Ok(buffer)
    }
    fn profile_range(
        &mut self,
        records: Vec<GpuKernelProfile>,
    ) -> RuntimeResult<Option<(usize, u32)>> {
        if !self.profiling || records.is_empty() {
            return Ok(None);
        }
        let count = u32::try_from(records.len())
            .ok()
            .and_then(|n| n.checked_mul(2))
            .filter(|&n| n <= wgpu::QUERY_SET_MAX_QUERIES)
            .ok_or_else(|| resource("resident profile exceeds query capacity"))?;
        let available = self
            .context
            .timestamp_features
            .contains(wgpu::Features::TIMESTAMP_QUERY);
        if self.profiles.last().is_none_or(|page| {
            page.report.kernels.len() as u32 * 2 + count
                > page.storage.as_ref().map_or(512, |s| s.capacity())
        }) {
            self.profiles.push(ProfilePage {
                storage: if available {
                    Some(profile::TimestampStorage::new(
                        &self.context,
                        count
                            .max(512)
                            .next_power_of_two()
                            .min(wgpu::QUERY_SET_MAX_QUERIES),
                    )?)
                } else {
                    None
                },
                report: GpuExecutionProfile::default(),
            });
        }
        let index = self.profiles.len() - 1;
        let page = &mut self.profiles[index];
        let query = page.report.kernels.len() as u32 * 2;
        page.report.kernels.extend(records);
        Ok(Some((index, query)))
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
        if groups.contains(&0) {
            return Ok(());
        }
        if groups
            .iter()
            .any(|&x| x > self.context.limits.max_compute_workgroups_per_dimension)
            || words.len() > 256
        {
            return Err(resource("resident dispatch exceeds GPU limits"));
        }
        let kernel = Kernel::get(
            &self.context,
            name,
            source,
            &buffers.iter().map(|b| b.1).collect::<Vec<_>>(),
        )?;
        let mut packet = [0u32; 256];
        packet[..words.len()].copy_from_slice(words);
        packet[255] = u32::MAX;
        let params = self.context.create_buffer_init(
            "resident parameters",
            bytemuck::cast_slice(&packet),
            wgpu::BufferUsages::UNIFORM,
        )?;
        let mut entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(i, (b, _))| bind_entry(i as u32, b))
            .collect();
        entries.push(bind_entry(buffers.len() as u32, &params));
        let binding = self.context.checked(name, || {
            Ok(self
                .context
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(name),
                    layout: &kernel.layout,
                    entries: &entries,
                }))
        })?;
        let range = if self.profiling {
            let stage = if name.starts_with("db_") {
                "geometry"
            } else if name.starts_with("projection_") {
                "projection"
            } else if name.starts_with("ctc_") {
                "decoding"
            } else {
                "image"
            };
            self.profile_range(vec![GpuKernelProfile {
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
                granularity: if self
                    .context
                    .timestamp_features
                    .contains(wgpu::Features::TIMESTAMP_QUERY)
                {
                    GpuTimingGranularity::Pass
                } else {
                    GpuTimingGranularity::Unavailable
                },
            }])?
        } else {
            None
        };
        {
            let mut pass =
                self.encoder
                    .as_mut()
                    .unwrap()
                    .begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some(name),
                        timestamp_writes: range.and_then(|(index, query)| {
                            self.profiles[index]
                                .storage
                                .as_ref()
                                .map(|s| s.writes(query))
                        }),
                    });
            pass.set_pipeline(&kernel.pipeline);
            pass.set_bind_group(0, &binding, &[]);
            pass.dispatch_workgroups(groups[0], groups[1], groups[2]);
        }
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
        let timer = self.profiling.then(std::time::Instant::now);
        self.ensure_context(&session.compiled.context)?;
        if session.parameters_dirty {
            self.context
                .queue
                .write_buffer(&session.parameters, 0, &session.parameter_data);
            session.parameters_dirty = false;
        }
        let range = if self.profiling {
            self.profile_range(session.profile_records(stage).kernels)?
        } else {
            None
        };
        let (storage, query) = range.map_or((None, 0), |(index, query)| {
            (self.profiles[index].storage.as_ref(), query)
        });
        session.encode_graph_with_timestamps(self.encoder.as_mut().unwrap(), storage, query);
        self.statistics.dispatches += session
            .dispatches
            .iter()
            .filter(|d| matches!(d.command, Command::Compute { .. }))
            .count() as u64;
        self.statistics.layout_conversions += session
            .recipes
            .iter()
            .filter(|r| {
                matches!(
                    r.kind,
                    operators::RecipeKind::Compute(operators::ShaderKind::General)
                ) && r.words[2] == 24
            })
            .count() as u64;
        self.held.extend(session.buffers.iter().cloned());
        self.held.push(session.parameters.clone());
        self.held
            .extend(session.compiled.weights.constants.iter().flatten().cloned());
        self.held.push(session.compiled.weights.dummy.clone());
        self.graphs.push(session.submissions.clone());
        if let Some(timer) = timer {
            self.statistics.encoding_ns += timer.elapsed().as_nanos() as u64;
        }
        Ok(session.resident_tensor(session.selected_output))
    }
    pub fn copy(
        &mut self,
        source: &GpuBuffer,
        source_offset: u64,
        output: &GpuBuffer,
        output_offset: u64,
        bytes: u64,
    ) {
        self.encoder.as_mut().unwrap().copy_buffer_to_buffer(
            source,
            source_offset,
            output,
            output_offset,
            bytes,
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
        mut self,
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
        self.copy(source, 0, &readback, 0, bytes as u64);
        let context = self.context.clone();
        let timestamps: Vec<_> = self
            .profiles
            .iter()
            .filter_map(|page| {
                page.storage
                    .as_ref()
                    .map(|s| (s, page.report.kernels.len() as u32 * 2))
            })
            .collect();
        for &(storage, count) in &timestamps {
            storage.resolve(self.encoder.as_mut().unwrap(), count);
            self.statistics.profiling_readback_bytes += u64::from(count) * 8;
        }
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
            submission = Some(if timestamps.is_empty() {
                context.queue.submit([encoder.finish()])
            } else {
                let mut copy =
                    context
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("resident timestamp dependency"),
                        });
                for &(storage, count) in &timestamps {
                    storage.copy_results(&mut copy, count);
                }
                context.queue.submit([encoder.finish(), copy.finish()])
            });
            for &(storage, count) in &timestamps {
                storage.map(count);
            }
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
        drop(timestamps);
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
    fn resident_tensor(&self, id: TensorId) -> Tensor {
        let buffer = self
            .compiled
            .weights
            .constants
            .get(id.0)
            .and_then(Option::as_ref)
            .unwrap_or_else(|| &self.buffers[self.plan.slots[id.0]])
            .clone();
        Tensor {
            context: self.compiled.context.clone(),
            buffer,
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
    let (x, y) = dispatch_grid(shape[2] * shape[3] * 3, &batch.context.limits)?;
    batch.dispatch(
        &format!("image_normalize_{precision:?}"),
        &source_code,
        &[(&image.buffer, true), (lut, true), (&output.buffer, false)],
        &words,
        [x, y, 1],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires a physical GPU"]
    fn gpu_resident_pending_readbacks_hold_admission_and_reservations() {
        let backend = match std::env::var("SIMD_OCR_GPU_BACKEND").as_deref() {
            Ok("dx12") => GpuBackendPreference::Dx12,
            Ok("metal") => GpuBackendPreference::Metal,
            _ => GpuBackendPreference::Vulkan,
        };
        let context = GpuContext::new_with_options(GpuContextOptions {
            backend,
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
    fn gpu_resident_profiles_cross_query_pages_without_changing_results() {
        let backend = match std::env::var("SIMD_OCR_GPU_BACKEND").as_deref() {
            Ok("dx12") => GpuBackendPreference::Dx12,
            Ok("metal") => GpuBackendPreference::Metal,
            _ => GpuBackendPreference::Vulkan,
        };
        let context = GpuContext::new_with_options(GpuContextOptions {
            backend,
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
            assert_eq!(statistics.diagnostic_readback_bytes, 0);
            assert_eq!(statistics.output_readback_bytes, 4);
            if profiling {
                let profile = statistics.profile.unwrap();
                assert_eq!(profile.kernels.len(), 260);
                if context
                    .timestamp_features
                    .contains(wgpu::Features::TIMESTAMP_QUERY)
                {
                    assert_eq!(statistics.profiling_readback_bytes, 260 * 16);
                    assert!(profile.compute_ns().is_some_and(|ns| ns > 0));
                }
            } else {
                assert!(statistics.profile.is_none());
                assert_eq!(statistics.profiling_readback_bytes, 0);
            }
        }
    }
    #[test]
    #[ignore = "requires a physical GPU"]
    fn gpu_resident_scratch_lifetime_and_context() {
        let backend = match std::env::var("SIMD_OCR_GPU_BACKEND").as_deref() {
            Ok("dx12") => GpuBackendPreference::Dx12,
            Ok("metal") => GpuBackendPreference::Metal,
            _ => GpuBackendPreference::Vulkan,
        };
        let context = GpuContext::new_with_options(GpuContextOptions {
            backend,
            max_device_bytes: Some(1 << 20),
            cache_resident_scratch: true,
            ..Default::default()
        })
        .unwrap();
        let pool = Arc::new(ScratchPool::default());
        let mut batch = Batch::new(context.clone()).unwrap();
        batch.scratch = Some(pool.clone());
        let first = batch.storage("live first", 1024).unwrap();
        let second = batch.storage("live second", 1024).unwrap();
        assert!(!Arc::ptr_eq(&first.reservation, &second.reservation));
        context.queue.write_buffer(&first, 0, &[7u8; 1024]);
        let (data, _) = batch.read(&first, 1024, false).unwrap();
        assert_eq!(data, vec![7u8; 1024]);
        let identity = first.reservation.clone();
        drop(first);
        drop(second);
        drop(identity);
        let retained = context.memory_statistics().used_bytes;
        let mut batch = Batch::new(context.clone()).unwrap();
        batch.scratch = Some(pool.clone());
        let reused = batch.storage("reused cleared", 1024).unwrap();
        assert_eq!(context.memory_statistics().used_bytes, retained);
        let (data, _) = batch.read(&reused, 1024, false).unwrap();
        assert_eq!(data, vec![0; 1024]);
        let other = GpuContext::new_with_options(GpuContextOptions {
            backend,
            ..Default::default()
        })
        .unwrap();
        assert!(pool.allocate(&other, "wrong context", 1024).is_err());
        assert_eq!(
            pool.allocate(&context, "over quota", 2 << 20)
                .err()
                .unwrap()
                .kind(),
            ErrorKind::ResourceLimit
        );
        drop(reused);
        drop(pool);
        assert_eq!(context.memory_statistics().used_bytes, 0);
    }
    #[test]
    #[ignore = "requires a physical GPU"]
    fn gpu_resident_preprocess_matches_byte_domain_cpu() {
        let backend = match std::env::var("SIMD_OCR_GPU_BACKEND").as_deref() {
            Ok("dx12") => GpuBackendPreference::Dx12,
            Ok("metal") => GpuBackendPreference::Metal,
            _ => GpuBackendPreference::Vulkan,
        };
        let context = GpuContext::new_with_options(GpuContextOptions {
            backend,
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
