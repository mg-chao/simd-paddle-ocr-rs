use super::*;
use crate::runtime::gpu::{
    kernel::{KernelId, KernelKey},
    resident::schedule::Operation,
};
use std::collections::HashMap;

struct ShaderBinding {
    slot: u32,
    uniform: bool,
    read_only: bool,
    minimum: u64,
}
struct ShaderBinary {
    bytes: &'static [u8],
    workgroup: [u32; 3],
    bindings: &'static [ShaderBinding],
    specialization: &'static [(u32, u32)],
}
include!(concat!(env!("OUT_DIR"), "/vulkan_shaders.rs"));

pub(in crate::runtime::gpu) struct NativePipeline {
    raw: vk::Pipeline,
    layout: vk::PipelineLayout,
    descriptors: vk::DescriptorSetLayout,
    owner: Arc<Ledger>,
    bindings: &'static [ShaderBinding],
    pub(in crate::runtime::gpu) workgroup_size: [u32; 3],
}
impl std::fmt::Debug for NativePipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanPipeline")
            .field("workgroup_size", &self.workgroup_size)
            .finish_non_exhaustive()
    }
}
impl PartialEq for NativePipeline {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other)
    }
}
impl Eq for NativePipeline {}
impl NativePipeline {
    pub(in crate::runtime::gpu) fn validate_binding_sizes(
        &self,
        count: usize,
        size: impl Fn(usize) -> u64,
    ) -> RuntimeResult<()> {
        for b in self.bindings {
            if b.slot as usize >= count || size(b.slot as usize) < b.minimum {
                return Err(RuntimeError::with_kind(
                    ErrorKind::InvalidInput,
                    format!("Vulkan binding {} needs {} bytes", b.slot, b.minimum),
                ));
            }
        }
        Ok(())
    }
}
impl Drop for NativePipeline {
    fn drop(&mut self) {
        let device = &self.owner.owner_device.raw;
        // SAFETY: all dispatches/submissions retain their pipeline.
        unsafe {
            device.destroy_pipeline(self.raw, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_set_layout(self.descriptors, None);
        }
    }
}
pub(in crate::runtime::gpu) fn compile(
    context: &GpuContext,
    key: KernelKey,
) -> RuntimeResult<Arc<NativePipeline>> {
    if key.precision & !key.id.precision_mask() != 0 {
        return Err(unsupported("invalid kernel storage signature"));
    }
    let owner = &context.ledger.owner_device;
    if key.precision != 0 && !owner.half {
        return Err(unsupported(
            "FP16 storage requires shaderFloat16 and storageBuffer16BitAccess",
        ));
    }
    if key.id == KernelId::ProjectionMatrixF32
        || (key.id == KernelId::ProjectionMatrixF16 && !owner.cooperative_matrix)
    {
        return Err(unsupported(
            "matrix projection is unavailable on this device",
        ));
    }
    if key.id == KernelId::Heads4Subgroup && !owner.subgroup_reductions {
        return Err(unsupported(
            "subgroup projection is unavailable on this device",
        ));
    }
    let shader = binary(key);
    let [x, y, z] = shader.workgroup;
    let limits = &owner.limits;
    if x > limits.max_compute_workgroup_size_x
        || y > limits.max_compute_workgroup_size_y
        || z > limits.max_compute_workgroup_size_z
        || x.checked_mul(y)
            .and_then(|n| n.checked_mul(z))
            .is_none_or(|n| n > limits.max_compute_invocations_per_workgroup)
    {
        return Err(unsupported("kernel workgroup exceeds device limits"));
    }
    let bindings = shader
        .bindings
        .iter()
        .map(|b| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(b.slot)
                .descriptor_type(if b.uniform {
                    vk::DescriptorType::UNIFORM_BUFFER
                } else {
                    vk::DescriptorType::STORAGE_BUFFER
                })
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        })
        .collect::<Vec<_>>();
    let code = ash::util::read_spv(&mut std::io::Cursor::new(shader.bytes))
        .map_err(|e| RuntimeError::caused_by(ErrorKind::GpuValidation, "embedded SPIR-V", e))?;
    let cache = owner
        .pipeline_cache
        .lock()
        .map_err(|_| RuntimeError::new("Vulkan pipeline cache poisoned"))?;
    // SAFETY: compiled, reflected SPIR-V and layout metadata share one build
    // catalog. Every partial construction is destroyed on failure.
    unsafe {
        let descriptors = owner.result(
            owner.raw.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            ),
            "descriptor layout",
        )?;
        let layouts = [descriptors];
        let layout = match owner.result(
            owner.raw.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts),
                None,
            ),
            "pipeline layout",
        ) {
            Ok(v) => v,
            Err(e) => {
                owner.raw.destroy_descriptor_set_layout(descriptors, None);
                return Err(e);
            }
        };
        let module = match owner.result(
            owner
                .raw
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None),
            "shader module",
        ) {
            Ok(v) => v,
            Err(e) => {
                owner.raw.destroy_pipeline_layout(layout, None);
                owner.raw.destroy_descriptor_set_layout(descriptors, None);
                return Err(e);
            }
        };
        let constants = shader
            .specialization
            .iter()
            .map(|(_, value)| *value)
            .collect::<Vec<_>>();
        let entries = shader
            .specialization
            .iter()
            .enumerate()
            .map(|(i, (id, _))| {
                vk::SpecializationMapEntry::default()
                    .constant_id(*id)
                    .offset((i * 4) as u32)
                    .size(4)
            })
            .collect::<Vec<_>>();
        let specialization = vk::SpecializationInfo::default()
            .map_entries(&entries)
            .data(bytemuck::cast_slice(&constants));
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(c"main")
            .specialization_info(&specialization);
        let result = owner.raw.create_compute_pipelines(
            *cache,
            &[vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(layout)],
            None,
        );
        owner.raw.destroy_shader_module(module, None);
        let raw = match result {
            Ok(v) => v[0],
            Err((partial, e)) => {
                for p in partial {
                    owner.raw.destroy_pipeline(p, None);
                }
                owner.raw.destroy_pipeline_layout(layout, None);
                owner.raw.destroy_descriptor_set_layout(descriptors, None);
                return owner.result(Err(e), "compute pipeline");
            }
        };
        Ok(Arc::new(NativePipeline {
            raw,
            layout,
            descriptors,
            owner: context.ledger.clone(),
            bindings: shader.bindings,
            workgroup_size: shader.workgroup,
        }))
    }
}

pub(in crate::runtime::gpu) struct NativeDispatch {
    pipeline: Arc<NativePipeline>,
    pool: vk::DescriptorPool,
    descriptors: vk::DescriptorSet,
    owners: Vec<GpuBuffer>,
    groups: [u32; 3],
    hazards: Box<[(usize, bool)]>,
}
impl std::fmt::Debug for NativeDispatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanDispatch")
            .field("groups", &self.groups)
            .finish_non_exhaustive()
    }
}
impl NativeDispatch {
    pub(in crate::runtime::gpu) fn prepare(
        pipeline: Arc<NativePipeline>,
        owners: Vec<GpuBuffer>,
        offsets: Vec<u64>,
        groups: [u32; 3],
        read_only: &[bool],
    ) -> RuntimeResult<Arc<Self>> {
        if owners.len() != offsets.len() || owners.len() != read_only.len() + 1 {
            return Err(RuntimeError::with_kind(
                ErrorKind::InvalidInput,
                "Vulkan binding count differs",
            ));
        }
        let device = &pipeline.owner.owner_device;
        if groups
            .iter()
            .zip(device.properties.limits.max_compute_work_group_count)
            .any(|(&n, max)| n > max)
        {
            return Err(resource("Vulkan dispatch exceeds device limits"));
        }
        for (slot, (buffer, offset)) in owners.iter().zip(&offsets).enumerate() {
            let uniform = slot == read_only.len();
            let usage = if uniform {
                BufferUsage::UNIFORM
            } else {
                BufferUsage::STORAGE
            };
            let alignment = if uniform {
                device.limits.min_uniform_buffer_offset_alignment
            } else {
                device.limits.min_storage_buffer_offset_alignment
            } as u64;
            if !Arc::ptr_eq(&buffer.reservation.ledger, &pipeline.owner)
                || !buffer.usage().contains(usage)
            {
                return Err(RuntimeError::with_kind(
                    ErrorKind::InvalidInput,
                    "foreign Vulkan resource or invalid binding usage",
                ));
            }
            if *offset >= buffer.size() || !offset.is_multiple_of(alignment) {
                return Err(resource(
                    "Vulkan binding offset is outside allocation or unaligned",
                ));
            }
        }
        pipeline.validate_binding_sizes(owners.len(), |i| owners[i].size() - offsets[i])?;
        let mut infos = Vec::new();
        for b in pipeline.bindings {
            let slot = b.slot as usize;
            if (b.uniform && slot != read_only.len())
                || (!b.uniform && read_only.get(slot) != Some(&b.read_only))
            {
                return Err(RuntimeError::with_kind(
                    ErrorKind::InvalidInput,
                    "shader binding access differs",
                ));
            }
            let range = if b.uniform {
                b.minimum
            } else {
                owners[slot].size() - offsets[slot]
            };
            if range
                > if b.uniform {
                    device.limits.max_uniform_buffer_binding_size as u64
                } else {
                    device.limits.max_storage_buffer_binding_size
                }
            {
                return Err(resource("Vulkan binding range exceeds device limit"));
            }
            infos.push(
                vk::DescriptorBufferInfo::default()
                    .buffer(owners[slot].buffer.0.raw)
                    .offset(offsets[slot])
                    .range(range),
            );
        }
        let sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(16),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1),
        ];
        // SAFETY: immutable descriptor bindings retain their buffers and pipeline.
        let (pool, descriptors) = unsafe {
            let cached = device
                .descriptor_pools
                .lock()
                .map_err(|_| RuntimeError::new("descriptor cache poisoned"))?
                .pop();
            let pool = if let Some(pool) = cached {
                pool
            } else {
                device.result(
                    device.raw.create_descriptor_pool(
                        &vk::DescriptorPoolCreateInfo::default()
                            .max_sets(1)
                            .pool_sizes(&sizes),
                        None,
                    ),
                    "descriptor pool",
                )?
            };
            let sets = [pipeline.descriptors];
            let descriptors = match device.result(
                device.raw.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(pool)
                        .set_layouts(&sets),
                ),
                "descriptor set",
            ) {
                Ok(v) => v[0],
                Err(e) => {
                    device.raw.destroy_descriptor_pool(pool, None);
                    return Err(e);
                }
            };
            let writes = pipeline
                .bindings
                .iter()
                .zip(&infos)
                .map(|(b, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(descriptors)
                        .dst_binding(b.slot)
                        .descriptor_type(if b.uniform {
                            vk::DescriptorType::UNIFORM_BUFFER
                        } else {
                            vk::DescriptorType::STORAGE_BUFFER
                        })
                        .buffer_info(std::slice::from_ref(info))
                })
                .collect::<Vec<_>>();
            device.raw.update_descriptor_sets(&writes, &[]);
            (pool, descriptors)
        };
        let hazards = owners
            .iter()
            .zip(read_only)
            .map(|(b, &read)| (Arc::as_ptr(&b.reservation) as usize, read))
            .collect();
        Ok(Arc::new(Self {
            pipeline,
            pool,
            descriptors,
            owners,
            groups,
            hazards,
        }))
    }
    pub(in crate::runtime::gpu) fn hazards(&self) -> &[(usize, bool)] {
        &self.hazards
    }
    pub(in crate::runtime::gpu) fn owners(&self) -> &[GpuBuffer] {
        &self.owners
    }
    pub(in crate::runtime::gpu) fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.owners.capacity() * std::mem::size_of::<GpuBuffer>()
            + std::mem::size_of_val(self.hazards.as_ref())
    }
}
impl Drop for NativeDispatch {
    fn drop(&mut self) {
        let device = &self.pipeline.owner.owner_device;
        unsafe {
            if device
                .raw
                .reset_descriptor_pool(self.pool, vk::DescriptorPoolResetFlags::empty())
                .is_ok()
                && let Ok(mut pools) = device.descriptor_pools.lock()
                && pools.len() < 1024
            {
                pools.push(self.pool);
                return;
            }
            device.raw.destroy_descriptor_pool(self.pool, None);
        }
    }
}

pub(in crate::runtime::gpu) struct Submission {
    encoding_ns: u64,
    submission_ns: u64,
    owner: Arc<Device>,
    pool: vk::CommandPool,
    command: vk::CommandBuffer,
    transfer_pool: Arc<TransferPool>,
    value: u64,
    completed: bool,
    wait_error: Option<RuntimeError>,
    operations: Vec<(usize, Operation)>,
    uploads: Vec<Upload>,
    query: Option<vk::QueryPool>,
    query_order: Vec<usize>,
}
impl Submission {
    pub(in crate::runtime::gpu) fn host_timings(&self) -> (u64, u64) {
        (self.encoding_ns, self.submission_ns)
    }
    pub(in crate::runtime::gpu) fn wait(&mut self) -> RuntimeResult<()> {
        self.wait_for(60_000_000_000)
    }
    #[cfg(test)]
    pub(super) fn wait_for_unsignalled_test(&mut self) -> RuntimeResult<()> {
        self.value += 1;
        self.wait_for(0)
    }
    fn wait_for(&mut self, timeout_ns: u64) -> RuntimeResult<()> {
        if let Some(error) = &self.wait_error {
            return Err(error.clone());
        }
        if self.completed {
            return Ok(());
        }
        let semaphores = [self.owner.timeline];
        let values = [self.value];
        let result = unsafe {
            self.owner.raw.wait_semaphores(
                &vk::SemaphoreWaitInfo::default()
                    .semaphores(&semaphores)
                    .values(&values),
                timeout_ns,
            )
        };
        if result == Err(vk::Result::TIMEOUT) {
            let error = RuntimeError::with_kind(
                ErrorKind::Execution,
                "Vulkan completion timed out; context retired",
            );
            self.owner.record_failure(error.clone());
            self.wait_error = Some(error.clone());
            return Err(error);
        }
        if let Err(error) = self.owner.result(result, "wait for Vulkan submission") {
            self.owner.record_failure(error.clone());
            self.wait_error = Some(error.clone());
            return Err(error);
        }
        self.completed = true;
        Ok(())
    }
    pub(in crate::runtime::gpu) fn timestamps(&self) -> RuntimeResult<Vec<(usize, u64)>> {
        let Some(query) = self.query else {
            return Ok(vec![]);
        };
        if !self.completed {
            return Err(RuntimeError::new("timestamps requested before completion"));
        }
        let mut ticks = vec![0u64; self.query_order.len() * 2];
        unsafe {
            self.owner.result(
                self.owner.raw.get_query_pool_results(
                    query,
                    0,
                    &mut ticks,
                    vk::QueryResultFlags::TYPE_64,
                ),
                "timestamp query",
            )?;
        }
        let mask = if self.owner.timestamp_bits == 64 {
            u64::MAX
        } else {
            (1u64 << self.owner.timestamp_bits) - 1
        };
        Ok(self
            .query_order
            .iter()
            .zip(ticks.as_chunks::<2>().0.iter())
            .map(|(&index, t)| {
                (
                    index,
                    ((t[1].wrapping_sub(t[0]) & mask) as f64
                        * self.owner.properties.limits.timestamp_period as f64)
                        .round() as u64,
                )
            })
            .collect())
    }
}
impl Drop for Submission {
    fn drop(&mut self) {
        if !self.completed && self.wait().is_err() {
            // A timed-out wait does not authorize destruction of pending work.
            // Terminal device loss or a successful idle wait makes retirement safe.
            let _serial = self.owner.serial.lock();
            let result = unsafe { self.owner.raw.device_wait_idle() };
            if result.is_err() && result != Err(vk::Result::ERROR_DEVICE_LOST) {
                // Preserve native owners on an indeterminate driver failure.
                std::mem::forget(std::mem::take(&mut self.operations));
                std::mem::forget(std::mem::take(&mut self.uploads));
                std::mem::forget(self.owner.clone());
                return;
            }
        }
        self.transfer_pool.recycle(&mut self.uploads);
        unsafe {
            self.owner
                .raw
                .free_command_buffers(self.pool, &[self.command]);
            if let Some(query) = self.query {
                self.owner.raw.destroy_query_pool(query, None);
            }
            if self
                .owner
                .raw
                .reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty())
                .is_ok()
                && let Ok(mut pools) = self.owner.command_pools.lock()
            {
                pools.push(self.pool);
                return;
            }
            self.owner.raw.destroy_command_pool(self.pool, None);
        }
    }
}

#[derive(Clone, Copy)]
struct Access {
    stage: vk::PipelineStageFlags2,
    access: vk::AccessFlags2,
    write: bool,
}
struct Barriers {
    last: HashMap<usize, Access>,
}
impl Barriers {
    fn apply(
        &mut self,
        device: &Device,
        cmd: vk::CommandBuffer,
        accesses: impl IntoIterator<Item = (usize, Access)>,
    ) {
        let mut src_stage = vk::PipelineStageFlags2::empty();
        let mut dst_stage = src_stage;
        let mut src_access = vk::AccessFlags2::empty();
        let mut dst_access = src_access;
        let mut current: HashMap<usize, Access> = HashMap::new();
        for (id, access) in accesses {
            current
                .entry(id)
                .and_modify(|a| {
                    a.stage |= access.stage;
                    a.access |= access.access;
                    a.write |= access.write;
                })
                .or_insert(access);
        }
        for (&id, &next) in &current {
            if let Some(last) = self.last.get(&id)
                && (last.write || next.write)
            {
                src_stage |= last.stage;
                src_access |= last.access;
                dst_stage |= next.stage;
                dst_access |= next.access;
            }
        }
        if !src_stage.is_empty() {
            barrier(device, cmd, src_stage, src_access, dst_stage, dst_access);
        }
        for (id, next) in current {
            self.last
                .entry(id)
                .and_modify(|old| {
                    if !old.write && !next.write {
                        old.stage |= next.stage;
                        old.access |= next.access;
                    } else {
                        *old = next;
                    }
                })
                .or_insert(next);
        }
    }
}
fn barrier(
    device: &Device,
    cmd: vk::CommandBuffer,
    src: vk::PipelineStageFlags2,
    sa: vk::AccessFlags2,
    dst: vk::PipelineStageFlags2,
    da: vk::AccessFlags2,
) {
    let barriers = [vk::MemoryBarrier2::default()
        .src_stage_mask(src)
        .src_access_mask(sa)
        .dst_stage_mask(dst)
        .dst_access_mask(da)];
    unsafe {
        device.raw.cmd_pipeline_barrier2(
            cmd,
            &vk::DependencyInfo::default().memory_barriers(&barriers),
        );
    }
}
fn identity(b: &Buffer) -> usize {
    Arc::as_ptr(&b.0) as usize
}
const COPY_READ: Access = Access {
    stage: vk::PipelineStageFlags2::COPY,
    access: vk::AccessFlags2::TRANSFER_READ,
    write: false,
};
const COPY_WRITE: Access = Access {
    stage: vk::PipelineStageFlags2::COPY,
    access: vk::AccessFlags2::TRANSFER_WRITE,
    write: true,
};

pub(in crate::runtime::gpu) fn submit(
    context: &GpuContext,
    mut operations: Vec<(usize, Operation)>,
    profiling: bool,
    _prefer_serial: bool,
) -> RuntimeResult<Submission> {
    let encoding = profiling.then(std::time::Instant::now);
    let native = &context.backend;
    let owner = &native.owner;
    let mut serial = owner
        .serial
        .lock()
        .map_err(|_| RuntimeError::new("Vulkan queue poisoned"))?;
    context.check_alive()?;
    let mut uploads = native
        .uploads
        .lock()
        .map_err(|_| RuntimeError::new("Vulkan uploads poisoned"))?;
    uploads.copies.retain_mut(|upload| {
        upload.target_owner = upload.target_reservation.upgrade();
        upload.target_owner.is_some()
    });
    let pool = if let Some(pool) = owner
        .command_pools
        .lock()
        .map_err(|_| RuntimeError::new("command pool cache poisoned"))?
        .pop()
    {
        pool
    } else {
        unsafe {
            owner.result(
                owner.raw.create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(owner.family)
                        .flags(vk::CommandPoolCreateFlags::TRANSIENT),
                    None,
                ),
                "command pool",
            )?
        }
    };
    let mut indeterminate = false;
    let recording = (|| -> RuntimeResult<_> {
        unsafe {
            let cmd = owner.result(
                owner.raw.allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                ),
                "command buffer",
            )?[0];
            owner.result(
                owner.raw.begin_command_buffer(
                    cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                ),
                "begin command buffer",
            )?;
            let mut resources = HashMap::new();
            for (_, op) in &operations {
                match op {
                    Operation::Compute { native, .. } => {
                        if !Arc::ptr_eq(&native.pipeline.owner, &context.ledger) {
                            return Err(RuntimeError::with_kind(
                                ErrorKind::InvalidInput,
                                "foreign Vulkan pipeline",
                            ));
                        }
                        for b in native.owners() {
                            resources.insert(identity(&b.buffer), b.buffer.clone());
                        }
                    }
                    Operation::Copy { source, output, .. } => {
                        for b in [source, output] {
                            if !Arc::ptr_eq(&b.reservation.ledger, &context.ledger) {
                                return Err(RuntimeError::with_kind(
                                    ErrorKind::InvalidInput,
                                    "foreign copy buffer",
                                ));
                            }
                            resources.insert(identity(&b.buffer), b.buffer.clone());
                        }
                    }
                }
            }
            for upload in uploads.copies.iter() {
                resources.insert(identity(&upload.target), upload.target.clone());
            }
            barrier(
                owner,
                cmd,
                vk::PipelineStageFlags2::ALL_COMMANDS,
                vk::AccessFlags2::MEMORY_WRITE | vk::AccessFlags2::MEMORY_READ,
                vk::PipelineStageFlags2::ALL_COMMANDS,
                vk::AccessFlags2::MEMORY_WRITE | vk::AccessFlags2::MEMORY_READ,
            );
            let mut state = Barriers {
                last: HashMap::new(),
            };
            let mut initialized = Vec::new();
            for buffer in resources.values() {
                if !buffer.0.initialized.load(Ordering::Acquire)
                    && !buffer.usage().contains(BufferUsage::MAP_READ)
                {
                    owner
                        .raw
                        .cmd_fill_buffer(cmd, buffer.0.raw, 0, vk::WHOLE_SIZE, 0);
                    state.last.insert(
                        identity(buffer),
                        Access {
                            stage: vk::PipelineStageFlags2::CLEAR,
                            access: vk::AccessFlags2::TRANSFER_WRITE,
                            write: true,
                        },
                    );
                    initialized.push(buffer.clone());
                }
            }
            for upload in uploads.copies.iter() {
                state.apply(owner, cmd, [(identity(&upload.target), COPY_WRITE)]);
                owner.raw.cmd_copy_buffer(
                    cmd,
                    upload.staging.0.raw,
                    upload.target.0.raw,
                    &[vk::BufferCopy::default()
                        .src_offset(upload.source_offset)
                        .dst_offset(upload.offset)
                        .size(upload.bytes)],
                );
            }
            let count = operations
                .iter()
                .filter(|(_, op)| matches!(op, Operation::Compute { .. }))
                .count();
            let query = if profiling && owner.timestamp_bits > 0 && count > 0 {
                Some(
                    owner.result(
                        owner.raw.create_query_pool(
                            &vk::QueryPoolCreateInfo::default()
                                .query_type(vk::QueryType::TIMESTAMP)
                                .query_count((count * 2) as u32),
                            None,
                        ),
                        "timestamp pool",
                    )?,
                )
            } else {
                None
            };
            if let Some(q) = query {
                owner
                    .raw
                    .cmd_reset_query_pool(cmd, q, 0, (count * 2) as u32);
            }
            operations.sort_by_key(|(depth, _)| *depth);
            let mut bound = None;
            let mut query_order = Vec::new();
            for (_, operation) in &operations {
                match operation {
                    Operation::Compute {
                        native,
                        profile_index,
                    } => {
                        state.apply(
                            owner,
                            cmd,
                            native
                                .hazards
                                .iter()
                                .enumerate()
                                .map(|(slot, (_, read))| {
                                    (
                                        identity(&native.owners[slot].buffer),
                                        Access {
                                            stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                                            access: if *read {
                                                vk::AccessFlags2::SHADER_STORAGE_READ
                                            } else {
                                                vk::AccessFlags2::SHADER_STORAGE_READ
                                                    | vk::AccessFlags2::SHADER_STORAGE_WRITE
                                            },
                                            write: !*read,
                                        },
                                    )
                                })
                                .chain(native.owners.last().map(|b| {
                                    (
                                        identity(&b.buffer),
                                        Access {
                                            stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                                            access: vk::AccessFlags2::UNIFORM_READ,
                                            write: false,
                                        },
                                    )
                                })),
                        );
                        if let Some(q) = query {
                            owner.raw.cmd_write_timestamp2(
                                cmd,
                                vk::PipelineStageFlags2::COMPUTE_SHADER,
                                q,
                                (query_order.len() * 2) as u32,
                            );
                        }
                        if bound != Some(native.pipeline.raw) {
                            owner.raw.cmd_bind_pipeline(
                                cmd,
                                vk::PipelineBindPoint::COMPUTE,
                                native.pipeline.raw,
                            );
                            bound = Some(native.pipeline.raw);
                        }
                        owner.raw.cmd_bind_descriptor_sets(
                            cmd,
                            vk::PipelineBindPoint::COMPUTE,
                            native.pipeline.layout,
                            0,
                            &[native.descriptors],
                            &[],
                        );
                        let [x, y, z] = native.groups;
                        owner.raw.cmd_dispatch(cmd, x, y, z);
                        if let Some(q) = query {
                            owner.raw.cmd_write_timestamp2(
                                cmd,
                                vk::PipelineStageFlags2::COMPUTE_SHADER,
                                q,
                                (query_order.len() * 2 + 1) as u32,
                            );
                            query_order.push(*profile_index);
                        }
                    }
                    Operation::Copy {
                        source,
                        output,
                        source_offset,
                        output_offset,
                        bytes,
                    } => {
                        if *bytes == 0
                            || !bytes.is_multiple_of(4)
                            || !source_offset.is_multiple_of(4)
                            || !output_offset.is_multiple_of(4)
                            || (source.buffer == output.buffer
                                && *source_offset < output_offset.saturating_add(*bytes)
                                && *output_offset < source_offset.saturating_add(*bytes))
                            || source_offset
                                .checked_add(*bytes)
                                .is_none_or(|n| n > source.size())
                            || output_offset
                                .checked_add(*bytes)
                                .is_none_or(|n| n > output.size())
                        {
                            if let Some(q) = query {
                                owner.raw.destroy_query_pool(q, None);
                            }
                            return Err(resource("invalid Vulkan copy range"));
                        }
                        state.apply(
                            owner,
                            cmd,
                            [
                                (identity(&source.buffer), COPY_READ),
                                (identity(&output.buffer), COPY_WRITE),
                            ],
                        );
                        owner.raw.cmd_copy_buffer(
                            cmd,
                            source.buffer.0.raw,
                            output.buffer.0.raw,
                            &[vk::BufferCopy::default()
                                .src_offset(*source_offset)
                                .dst_offset(*output_offset)
                                .size(*bytes)],
                        );
                    }
                }
            }
            barrier(
                owner,
                cmd,
                vk::PipelineStageFlags2::ALL_COMMANDS,
                vk::AccessFlags2::MEMORY_WRITE,
                vk::PipelineStageFlags2::HOST,
                vk::AccessFlags2::HOST_READ,
            );
            if let Err(e) = owner.result(owner.raw.end_command_buffer(cmd), "end command buffer") {
                if let Some(q) = query {
                    owner.raw.destroy_query_pool(q, None);
                }
                return Err(e);
            }
            let Some(value) = serial.checked_add(1) else {
                if let Some(q) = query {
                    owner.raw.destroy_query_pool(q, None);
                }
                return Err(resource("Vulkan timeline exhausted"));
            };
            let signals = [vk::SemaphoreSubmitInfo::default()
                .semaphore(owner.timeline)
                .value(value)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
            let commands = [vk::CommandBufferSubmitInfo::default().command_buffer(cmd)];
            let encoding_ns = encoding.map_or(0, |timer| timer.elapsed().as_nanos() as u64);
            let submitting = profiling.then(std::time::Instant::now);
            if let Err(e) = owner.result(
                owner.raw.queue_submit2(
                    owner.queue,
                    &[vk::SubmitInfo2::default()
                        .command_buffer_infos(&commands)
                        .signal_semaphore_infos(&signals)],
                    vk::Fence::null(),
                ),
                "queue submit",
            ) {
                // Failed submission does not establish completion. Retire the
                // context and retain everything until the queue is known idle.
                owner.record_failure(e.clone());
                let idle = owner.raw.queue_wait_idle(owner.queue);
                indeterminate = idle.is_err() && idle != Err(vk::Result::ERROR_DEVICE_LOST);
                if !indeterminate && let Some(q) = query {
                    owner.raw.destroy_query_pool(q, None);
                }
                return Err(e);
            }
            let submission_ns = submitting.map_or(0, |timer| timer.elapsed().as_nanos() as u64);
            for buffer in initialized {
                buffer.0.initialized.store(true, Ordering::Release);
            }
            *serial = value;
            Ok((value, cmd, query, query_order, encoding_ns, submission_ns))
        }
    })();
    let (value, command, query, query_order, encoding_ns, submission_ns) = match recording {
        Ok(v) => v,
        Err(e) => {
            if indeterminate {
                std::mem::forget(operations);
                std::mem::forget(std::mem::take(&mut uploads.copies));
                std::mem::forget(uploads.page.take());
                std::mem::forget(owner.clone());
            } else {
                for upload in &mut uploads.copies {
                    upload.target_owner = None;
                }
                unsafe {
                    owner.raw.destroy_command_pool(pool, None);
                }
            }
            return Err(e);
        }
    };
    uploads.page = None;
    Ok(Submission {
        encoding_ns,
        submission_ns,
        owner: owner.clone(),
        pool,
        command,
        transfer_pool: native.transfer_pool.clone(),
        value,
        completed: false,
        wait_error: None,
        operations,
        uploads: std::mem::take(&mut uploads.copies),
        query,
        query_order,
    })
}
