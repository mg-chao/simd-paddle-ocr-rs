//! Native Vulkan objects. No other module issues Vulkan calls.
use super::*;
use ash::{Entry, vk};
use gpu_allocator::{
    MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc},
};
use std::{
    ffi::CStr,
    sync::atomic::{AtomicBool, AtomicU64},
};
#[path = "vulkan/execution.rs"]
mod execution;
#[cfg(test)]
#[path = "vulkan/tests.rs"]
mod tests;
pub(super) use execution::{NativeDispatch, NativePipeline, Submission, compile, submit};
pub(super) fn compile_resident(
    context: &GpuContext,
    key: KernelKey,
    _read_only: &[bool],
) -> RuntimeResult<Arc<NativePipeline>> {
    if key.id == KernelId::ProjectionMatrixF32
        || (key.id == KernelId::ProjectionMatrixF16 && !context.backend.owner.cooperative_matrix)
    {
        return Err(unsupported(
            "matrix projection is unavailable on this device",
        ));
    }
    let key = if key.id == KernelId::Heads4 && context.backend.owner.subgroup_reductions {
        KernelKey {
            id: KernelId::Heads4Subgroup,
            ..key
        }
    } else {
        key
    };
    compile(context, key)
}

pub(super) struct Device {
    raw: ash::Device,
    instance: ash::Instance,
    _entry: Entry,
    allocator: Mutex<Option<Allocator>>,
    queue: vk::Queue,
    family: u32,
    timeline: vk::Semaphore,
    serial: Mutex<u64>,
    command_pools: Mutex<Vec<vk::CommandPool>>,
    descriptor_pools: Mutex<Vec<vk::DescriptorPool>>,
    pipeline_cache: Mutex<vk::PipelineCache>,
    failure: Arc<Mutex<Option<RuntimeError>>>,
    available: Arc<Condvar>,
    properties: vk::PhysicalDeviceProperties,
    pub(super) limits: GpuLimits,
    pub(super) half: bool,
    cooperative_matrix: bool,
    subgroup_reductions: bool,
    timestamp_bits: u32,
    committed_peak: AtomicU64,
    debug: Option<(
        ash::ext::debug_utils::Instance,
        vk::DebugUtilsMessengerEXT,
        Box<DebugState>,
    )>,
}
impl Device {
    fn result<T>(&self, value: Result<T, vk::Result>, stage: &str) -> RuntimeResult<T> {
        value.map_err(|e| {
            let kind = match e {
                vk::Result::ERROR_DEVICE_LOST => ErrorKind::DeviceLost,
                vk::Result::ERROR_OUT_OF_DEVICE_MEMORY | vk::Result::ERROR_OUT_OF_HOST_MEMORY => {
                    ErrorKind::Allocation
                }
                _ => ErrorKind::Execution,
            };
            let error = RuntimeError::with_kind(kind, format!("{stage}: {e:?}"));
            if kind == ErrorKind::DeviceLost {
                self.record_failure(error.clone());
            }
            error
        })
    }
    fn record_failure(&self, error: RuntimeError) {
        if let Ok(mut failure) = self.failure.lock()
            && failure.is_none()
        {
            *failure = Some(error);
        }
        self.available.notify_all();
    }
    pub(super) fn committed_bytes(&self) -> usize {
        self.allocator
            .lock()
            .ok()
            .and_then(|a| a.as_ref().map(Allocator::capacity))
            .unwrap_or(0) as usize
    }
    pub(super) fn peak_committed_bytes(&self) -> usize {
        self.committed_peak.load(Ordering::Relaxed) as usize
    }
}
impl Drop for Device {
    fn drop(&mut self) {
        // SAFETY: all buffers, pipelines and submissions retain this owner. Last
        // owner destruction therefore happens after their individual retirement.
        unsafe {
            let _ = self.raw.device_wait_idle();
            for pool in self.command_pools.get_mut().unwrap().drain(..) {
                self.raw.destroy_command_pool(pool, None);
            }
            for pool in self.descriptor_pools.get_mut().unwrap().drain(..) {
                self.raw.destroy_descriptor_pool(pool, None);
            }
            self.raw
                .destroy_pipeline_cache(*self.pipeline_cache.get_mut().unwrap(), None);
            drop(self.allocator.get_mut().unwrap().take());
            self.raw.destroy_semaphore(self.timeline, None);
            self.raw.destroy_device(None);
            if let Some((loader, messenger, _)) = &self.debug {
                loader.destroy_debug_utils_messenger(*messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

pub(super) struct Context {
    pub(super) owner: Arc<Device>,
    uploads: Mutex<UploadQueue>,
    transfer_pool: Arc<TransferPool>,
}
#[derive(Default)]
struct UploadQueue {
    copies: Vec<Upload>,
    page: Option<(Buffer, u64)>,
}
struct Upload {
    target: Buffer,
    target_reservation: std::sync::Weak<Reservation>,
    target_owner: Option<Arc<Reservation>>,
    offset: u64,
    staging: Buffer,
    source_offset: u64,
    bytes: u64,
    _reservation: Arc<Reservation>,
}
#[derive(Default)]
struct TransferPool {
    pages: Mutex<Vec<Buffer>>,
}
impl TransferPool {
    fn acquire(&self, device: &Arc<Device>, bytes: u64) -> RuntimeResult<Buffer> {
        let mut pages = self
            .pages
            .lock()
            .map_err(|_| RuntimeError::new("upload pool poisoned"))?;
        if let Some(index) = pages.iter().position(|b| b.size() >= bytes) {
            return Ok(pages.swap_remove(index));
        }
        Buffer::allocate(
            device.clone(),
            "mapped upload arena",
            bytes.max(1 << 20).min(device.limits.max_buffer_size),
            BufferUsage::COPY_SRC,
            Some(MemoryLocation::CpuToGpu),
        )
    }
    fn recycle(&self, uploads: &mut Vec<Upload>) {
        let mut unique = std::collections::HashMap::new();
        for upload in uploads.drain(..) {
            unique
                .entry(Arc::as_ptr(&upload.staging.0) as usize)
                .or_insert(upload.staging);
        }
        if let Ok(mut pages) = self.pages.lock() {
            // Keep a bounded idle cache; allocator capacity remains separately visible.
            for page in unique.into_values() {
                if pages.len() < 8 && page.size() <= 16 << 20 {
                    pages.push(page);
                }
            }
        }
    }
}

struct DebugState {
    failure: Arc<Mutex<Option<RuntimeError>>>,
    available: Arc<Condvar>,
}
// The messenger user pointer must outlive the instance. Context retains it
// through Device's lifetime via its boxed DebugState.
unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    user: *mut std::ffi::c_void,
) -> vk::Bool32 {
    if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR)
        && !data.is_null()
        && !user.is_null()
    {
        // SAFETY: Vulkan supplies the callback data; user is a boxed DebugState
        // retained until the messenger is destroyed.
        let state = unsafe { &*user.cast::<DebugState>() };
        let message = unsafe { CStr::from_ptr((*data).p_message) }.to_string_lossy();
        if let Ok(mut failure) = state.failure.lock()
            && failure.is_none()
        {
            *failure = Some(RuntimeError::with_kind(
                ErrorKind::GpuValidation,
                message.to_string(),
            ));
        }
        state.available.notify_all();
        eprintln!("Vulkan validation: {message}");
    }
    vk::FALSE
}
impl Context {
    pub(super) fn collect_garbage(&self) {
        if let Ok(mut pending) = self.uploads.lock() {
            pending
                .copies
                .retain(|upload| upload.target_reservation.strong_count() > 0);
            if pending.copies.is_empty()
                && let Some((page, _)) = pending.page.take()
                && let Ok(mut pages) = self.transfer_pool.pages.lock()
                && pages.len() < 8
            {
                pages.push(page);
            }
        }
    }
    pub(super) fn supports_half(&self) -> bool {
        self.owner.half
    }
    pub(super) fn checked<T>(
        &self,
        stage: &str,
        operation: impl FnOnce() -> RuntimeResult<T>,
    ) -> RuntimeResult<T> {
        operation().map_err(|e| e.context(stage))
    }
    pub(super) fn new(
        failure: Arc<Mutex<Option<RuntimeError>>>,
        available: Arc<Condvar>,
    ) -> RuntimeResult<(Self, GpuAdapterInfo, GpuLimits)> {
        // SAFETY: loader lifetime is retained by Device, and only discovered
        // Vulkan 1.3 functions/features are used after device creation.
        unsafe {
            let entry = Entry::load().map_err(|e| {
                RuntimeError::caused_by(ErrorKind::Unsupported, "Vulkan loader unavailable", e)
            })?;
            let version = entry
                .try_enumerate_instance_version()
                .map_err(init_error)?
                .unwrap_or(vk::API_VERSION_1_0);
            if version < vk::API_VERSION_1_3 {
                return Err(unsupported("Vulkan 1.3 is required"));
            }
            let validation = std::env::var("SIMD_OCR_VULKAN_VALIDATION").as_deref() == Ok("1");
            let layer = c"VK_LAYER_KHRONOS_validation";
            if validation
                && !entry
                    .enumerate_instance_layer_properties()
                    .map_err(init_error)?
                    .iter()
                    .any(|l| CStr::from_ptr(l.layer_name.as_ptr()) == layer)
            {
                return Err(unsupported(
                    "requested Khronos Vulkan validation layer is unavailable",
                ));
            }
            let layers = if validation {
                vec![layer.as_ptr()]
            } else {
                vec![]
            };
            let extensions = if validation {
                vec![
                    ash::ext::debug_utils::NAME.as_ptr(),
                    ash::ext::validation_features::NAME.as_ptr(),
                ]
            } else {
                vec![]
            };
            let app = vk::ApplicationInfo::default()
                .application_name(c"simd-paddle-ocr")
                .api_version(vk::API_VERSION_1_3);
            let enabled_validation = [vk::ValidationFeatureEnableEXT::SYNCHRONIZATION_VALIDATION];
            let mut validation_info = vk::ValidationFeaturesEXT::default()
                .enabled_validation_features(&enabled_validation);
            let mut create = vk::InstanceCreateInfo::default()
                .application_info(&app)
                .enabled_layer_names(&layers)
                .enabled_extension_names(&extensions);
            if validation {
                create = create.push_next(&mut validation_info);
            }
            let instance = entry.create_instance(&create, None).map_err(init_error)?;
            let debug = if validation {
                let mut state = Box::new(DebugState {
                    failure: failure.clone(),
                    available: available.clone(),
                });
                let loader = ash::ext::debug_utils::Instance::new(&entry, &instance);
                let info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                    .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR)
                    .message_type(
                        vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                            | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                            | vk::DebugUtilsMessageTypeFlagsEXT::GENERAL,
                    )
                    .pfn_user_callback(Some(debug_callback))
                    .user_data((&mut *state as *mut DebugState).cast());
                let messenger = match loader.create_debug_utils_messenger(&info, None) {
                    Ok(value) => value,
                    Err(error) => {
                        instance.destroy_instance(None);
                        return Err(init_error(error));
                    }
                };
                Some((loader, messenger, state))
            } else {
                None
            };
            // Initialization cleanup is explicit: no partially created native
            // object can escape when selecting or constructing a device fails.
            let initialized = (|| -> RuntimeResult<_> {
                let mut devices = Vec::new();
                for physical in instance.enumerate_physical_devices().map_err(init_error)? {
                    let p = instance.get_physical_device_properties(physical);
                    if p.api_version < vk::API_VERSION_1_3
                        || !matches!(
                            p.device_type,
                            vk::PhysicalDeviceType::DISCRETE_GPU
                                | vk::PhysicalDeviceType::INTEGRATED_GPU
                        )
                    {
                        continue;
                    }
                    let families = instance.get_physical_device_queue_family_properties(physical);
                    let Some((family, q)) = families
                        .iter()
                        .enumerate()
                        .filter(|(_, q)| {
                            q.queue_count > 0 && q.queue_flags.contains(vk::QueueFlags::COMPUTE)
                        })
                        .min_by_key(|(_, q)| q.queue_flags.contains(vk::QueueFlags::GRAPHICS))
                    else {
                        continue;
                    };
                    let mut f12 = vk::PhysicalDeviceVulkan12Features::default();
                    let mut f13 = vk::PhysicalDeviceVulkan13Features::default();
                    let mut f11 = vk::PhysicalDeviceVulkan11Features::default();
                    let mut features = vk::PhysicalDeviceFeatures2::default()
                        .push_next(&mut f11)
                        .push_next(&mut f12)
                        .push_next(&mut f13);
                    instance.get_physical_device_features2(physical, &mut features);
                    if f12.timeline_semaphore == 0
                        || f13.synchronization2 == 0
                        || f13.maintenance4 == 0
                    {
                        continue;
                    }
                    devices.push((
                        p.device_type != vk::PhysicalDeviceType::DISCRETE_GPU,
                        physical,
                        p,
                        family as u32,
                        q.timestamp_valid_bits,
                        f11.storage_buffer16_bit_access != 0 && f12.shader_float16 != 0,
                    ));
                }
                devices.sort_by_key(|d| d.0);
                let Some((_, physical, properties, family, timestamp_bits, half)) =
                    devices.into_iter().next()
                else {
                    return Err(unsupported(
                        "no physical Vulkan 1.3 compute device with synchronization2 and timeline semaphores",
                    ));
                };
                let mut driver = vk::PhysicalDeviceDriverProperties::default();
                let mut subgroup = vk::PhysicalDeviceSubgroupProperties::default();
                let mut subgroup_size = vk::PhysicalDeviceSubgroupSizeControlProperties::default();
                let mut allocation_limits = vk::PhysicalDeviceMaintenance3Properties::default();
                let mut buffer_limits = vk::PhysicalDeviceMaintenance4Properties::default();
                let mut props = vk::PhysicalDeviceProperties2::default()
                    .push_next(&mut allocation_limits)
                    .push_next(&mut buffer_limits)
                    .push_next(&mut driver)
                    .push_next(&mut subgroup)
                    .push_next(&mut subgroup_size);
                instance.get_physical_device_properties2(physical, &mut props);
                let extensions = instance
                    .enumerate_device_extension_properties(physical)
                    .map_err(init_error)?;
                let subgroup_reductions = matches!(subgroup.subgroup_size, 32 | 64)
                    && subgroup
                        .supported_stages
                        .contains(vk::ShaderStageFlags::COMPUTE)
                    && subgroup.supported_operations.contains(
                        vk::SubgroupFeatureFlags::BASIC | vk::SubgroupFeatureFlags::ARITHMETIC,
                    );
                let has_matrix = extensions.iter().any(|extension| {
                    CStr::from_ptr(extension.extension_name.as_ptr())
                        == ash::khr::cooperative_matrix::NAME
                });
                let mut matrix_features = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
                let mut memory_model = vk::PhysicalDeviceVulkan12Features::default();
                let mut matrix_properties =
                    vk::PhysicalDeviceCooperativeMatrixPropertiesKHR::default();
                let cooperative_matrix = if has_matrix && half {
                    instance.get_physical_device_features2(
                        physical,
                        &mut vk::PhysicalDeviceFeatures2::default()
                            .push_next(&mut matrix_features)
                            .push_next(&mut memory_model),
                    );
                    instance.get_physical_device_properties2(
                        physical,
                        &mut vk::PhysicalDeviceProperties2::default()
                            .push_next(&mut matrix_properties),
                    );
                    let matrix_loader =
                        ash::khr::cooperative_matrix::Instance::new(&entry, &instance);
                    let shapes = matrix_loader
                        .get_physical_device_cooperative_matrix_properties(physical)
                        .map_err(init_error)?;
                    matrix_features.cooperative_matrix != 0
                        && memory_model.vulkan_memory_model != 0
                        && memory_model.vulkan_memory_model_device_scope != 0
                        && matrix_properties
                            .cooperative_matrix_supported_stages
                            .contains(vk::ShaderStageFlags::COMPUTE)
                        && subgroup
                            .supported_stages
                            .contains(vk::ShaderStageFlags::COMPUTE)
                        && subgroup.supported_operations.contains(
                            vk::SubgroupFeatureFlags::BASIC | vk::SubgroupFeatureFlags::ARITHMETIC,
                        )
                        && subgroup.subgroup_size >= 32
                        && subgroup.subgroup_size <= 128
                        && properties.limits.max_compute_shared_memory_size >= 16384
                        && shapes.iter().any(|s| {
                            s.m_size == 16
                                && s.n_size == 16
                                && s.k_size == 16
                                && s.scope == vk::ScopeKHR::SUBGROUP
                                && s.a_type == vk::ComponentTypeKHR::FLOAT16
                                && s.b_type == vk::ComponentTypeKHR::FLOAT16
                                && s.c_type == vk::ComponentTypeKHR::FLOAT32
                                && s.result_type == vk::ComponentTypeKHR::FLOAT32
                                && s.saturating_accumulation == 0
                        })
                } else {
                    false
                };
                let info = GpuAdapterInfo {
                    name: CStr::from_ptr(properties.device_name.as_ptr())
                        .to_string_lossy()
                        .into_owned(),
                    vendor: properties.vendor_id,
                    device: properties.device_id,
                    device_type: if properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
                        GpuDeviceType::DiscreteGpu
                    } else {
                        GpuDeviceType::IntegratedGpu
                    },
                    backend: GpuApi::Vulkan,
                    driver: CStr::from_ptr(driver.driver_name.as_ptr())
                        .to_string_lossy()
                        .into_owned(),
                    driver_info: CStr::from_ptr(driver.driver_info.as_ptr())
                        .to_string_lossy()
                        .into_owned(),
                    api_version: properties.api_version,
                    subgroup_min_size: if subgroup_size.min_subgroup_size == 0 {
                        subgroup.subgroup_size
                    } else {
                        subgroup_size.min_subgroup_size
                    },
                    subgroup_max_size: subgroup_size.max_subgroup_size.max(subgroup.subgroup_size),
                };
                let p = &properties.limits;
                let limits = GpuLimits {
                    max_buffer_size: buffer_limits
                        .max_buffer_size
                        .min(allocation_limits.max_memory_allocation_size),
                    max_storage_buffer_binding_size: p.max_storage_buffer_range as u64,
                    max_uniform_buffer_binding_size: p.max_uniform_buffer_range,
                    max_compute_workgroups_per_dimension: *p
                        .max_compute_work_group_count
                        .iter()
                        .min()
                        .unwrap(),
                    max_compute_workgroup_storage_size: p.max_compute_shared_memory_size,
                    max_compute_invocations_per_workgroup: p.max_compute_work_group_invocations,
                    max_compute_workgroup_size_x: p.max_compute_work_group_size[0],
                    max_compute_workgroup_size_y: p.max_compute_work_group_size[1],
                    max_compute_workgroup_size_z: p.max_compute_work_group_size[2],
                    min_uniform_buffer_offset_alignment: p
                        .min_uniform_buffer_offset_alignment
                        .try_into()
                        .map_err(|_| unsupported("uniform alignment exceeds host ABI"))?,
                    min_storage_buffer_offset_alignment: p
                        .min_storage_buffer_offset_alignment
                        .try_into()
                        .map_err(|_| unsupported("storage alignment exceeds host ABI"))?,
                };
                let priorities = [1.0];
                let queues = [vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(family)
                    .queue_priorities(&priorities)];
                let mut f11 =
                    vk::PhysicalDeviceVulkan11Features::default().storage_buffer16_bit_access(half);
                let mut f12 = vk::PhysicalDeviceVulkan12Features::default()
                    .timeline_semaphore(true)
                    .shader_float16(half)
                    .vulkan_memory_model(cooperative_matrix)
                    .vulkan_memory_model_device_scope(cooperative_matrix);
                let mut f13 = vk::PhysicalDeviceVulkan13Features::default()
                    .synchronization2(true)
                    .maintenance4(true);
                let enabled_extensions = if cooperative_matrix {
                    vec![ash::khr::cooperative_matrix::NAME.as_ptr()]
                } else {
                    vec![]
                };
                let mut enabled_matrix = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default()
                    .cooperative_matrix(cooperative_matrix);
                let mut device_create = vk::DeviceCreateInfo::default()
                    .queue_create_infos(&queues)
                    .enabled_extension_names(&enabled_extensions)
                    .push_next(&mut f11)
                    .push_next(&mut f12)
                    .push_next(&mut f13);
                if cooperative_matrix {
                    device_create = device_create.push_next(&mut enabled_matrix);
                }
                let raw = instance
                    .create_device(physical, &device_create, None)
                    .map_err(init_error)?;
                let made = (|| -> RuntimeResult<_> {
                    let mut timeline_info = vk::SemaphoreTypeCreateInfo::default()
                        .semaphore_type(vk::SemaphoreType::TIMELINE);
                    let timeline = raw
                        .create_semaphore(
                            &vk::SemaphoreCreateInfo::default().push_next(&mut timeline_info),
                            None,
                        )
                        .map_err(init_error)?;
                    let cache = match raw.create_pipeline_cache(&Default::default(), None) {
                        Ok(c) => c,
                        Err(e) => {
                            raw.destroy_semaphore(timeline, None);
                            return Err(init_error(e));
                        }
                    };
                    let allocation = Allocator::new(&AllocatorCreateDesc {
                        instance: instance.clone(),
                        device: raw.clone(),
                        physical_device: physical,
                        debug_settings: Default::default(),
                        buffer_device_address: false,
                        allocation_sizes: Default::default(),
                    });
                    let allocator = match allocation {
                        Ok(a) => a,
                        Err(e) => {
                            raw.destroy_pipeline_cache(cache, None);
                            raw.destroy_semaphore(timeline, None);
                            return Err(RuntimeError::caused_by(
                                ErrorKind::Allocation,
                                "Vulkan allocator",
                                e,
                            ));
                        }
                    };
                    Ok((timeline, cache, allocator))
                })();
                let (timeline, cache, allocator) = match made {
                    Ok(v) => v,
                    Err(e) => {
                        raw.destroy_device(None);
                        return Err(e);
                    }
                };
                Ok((
                    physical,
                    properties,
                    family,
                    timestamp_bits,
                    half,
                    cooperative_matrix,
                    subgroup_reductions,
                    info,
                    limits,
                    raw,
                    timeline,
                    cache,
                    allocator,
                ))
            })();
            let (
                _physical,
                properties,
                family,
                timestamp_bits,
                half,
                cooperative_matrix,
                subgroup_reductions,
                info,
                limits,
                raw,
                timeline,
                cache,
                allocator,
            ) = match initialized {
                Ok(v) => v,
                Err(e) => {
                    if let Some((loader, messenger, _)) = &debug {
                        loader.destroy_debug_utils_messenger(*messenger, None);
                    }
                    instance.destroy_instance(None);
                    return Err(e);
                }
            };
            let owner = Device {
                queue: raw.get_device_queue(family, 0),
                raw,
                instance,
                _entry: entry,
                allocator: Mutex::new(Some(allocator)),
                family,
                timeline,
                serial: Mutex::new(0),
                command_pools: Mutex::new(Vec::new()),
                descriptor_pools: Mutex::new(Vec::new()),
                pipeline_cache: Mutex::new(cache),
                failure,
                available,
                properties,
                limits: limits.clone(),
                half,
                cooperative_matrix,
                subgroup_reductions,
                timestamp_bits,
                committed_peak: AtomicU64::new(0),
                debug,
            };
            Ok((
                Self {
                    owner: Arc::new(owner),
                    uploads: Mutex::new(UploadQueue::default()),
                    transfer_pool: Arc::new(TransferPool::default()),
                },
                info,
                limits,
            ))
        }
    }
    pub(super) fn allocate(
        &self,
        label: &str,
        size: u64,
        usage: BufferUsage,
    ) -> RuntimeResult<Buffer> {
        Buffer::allocate(
            self.owner.clone(),
            label,
            size,
            usage,
            usage
                .contains(BufferUsage::MAP_READ)
                .then_some(MemoryLocation::GpuToCpu),
        )
    }
    pub(super) fn upload(
        &self,
        target: &GpuBuffer,
        offset: u64,
        bytes: &[u8],
    ) -> RuntimeResult<()> {
        if offset
            .checked_add(bytes.len() as u64)
            .is_none_or(|n| n > target.size())
            || !offset.is_multiple_of(4)
            || !bytes.len().is_multiple_of(4)
        {
            return Err(resource("invalid Vulkan upload range"));
        }
        if bytes.is_empty() {
            return Ok(());
        }
        self.collect_garbage();
        let reservation = target.reservation.ledger.reserve(bytes.len())?;
        let mut pending = self
            .uploads
            .lock()
            .map_err(|_| RuntimeError::new("Vulkan uploads poisoned"))?;
        if pending
            .page
            .as_ref()
            .is_none_or(|(page, at)| page.size() - at < bytes.len() as u64)
        {
            pending.page = Some((
                self.transfer_pool
                    .acquire(&self.owner, bytes.len() as u64)?,
                0,
            ));
        }
        let (page, at) = pending.page.as_mut().unwrap();
        page.write(*at, bytes)?;
        let staging = page.clone();
        let source_offset = *at;
        *at += bytes.len() as u64;
        pending.copies.push(Upload {
            target: target.buffer.clone(),
            target_reservation: Arc::downgrade(&target.reservation),
            target_owner: None,
            offset,
            staging,
            source_offset,
            bytes: bytes.len() as u64,
            _reservation: reservation,
        });
        Ok(())
    }
}
fn init_error(e: vk::Result) -> RuntimeError {
    RuntimeError::with_kind(
        if matches!(
            e,
            vk::Result::ERROR_OUT_OF_DEVICE_MEMORY | vk::Result::ERROR_OUT_OF_HOST_MEMORY
        ) {
            ErrorKind::Allocation
        } else {
            ErrorKind::Unsupported
        },
        format!("Vulkan initialization: {e:?}"),
    )
}
fn unsupported(s: impl Into<String>) -> RuntimeError {
    RuntimeError::with_kind(ErrorKind::Unsupported, s)
}

#[derive(Clone)]
pub(super) struct Buffer(Arc<BufferInner>);
struct BufferInner {
    device: Arc<Device>,
    raw: vk::Buffer,
    allocation: Option<Allocation>,
    size: u64,
    usage: BufferUsage,
    initialized: AtomicBool,
}
impl PartialEq for Buffer {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
impl Eq for Buffer {}
impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanBuffer")
            .field("bytes", &self.size())
            .finish_non_exhaustive()
    }
}
impl Buffer {
    fn allocate(
        device: Arc<Device>,
        label: &str,
        size: u64,
        usage: BufferUsage,
        host: Option<MemoryLocation>,
    ) -> RuntimeResult<Self> {
        if size == 0 || size > device.limits.max_buffer_size {
            return Err(resource("Vulkan buffer size exceeds device limits"));
        }
        let mut flags = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
        if usage.contains(BufferUsage::STORAGE) {
            flags |= vk::BufferUsageFlags::STORAGE_BUFFER;
        }
        if usage.contains(BufferUsage::UNIFORM) {
            flags |= vk::BufferUsageFlags::UNIFORM_BUFFER;
        }
        let mut allocator = device
            .allocator
            .lock()
            .map_err(|_| RuntimeError::new("Vulkan allocator poisoned"))?;
        // SAFETY: the new buffer is private until successfully bound to a valid
        // allocation; the retained device outlives both objects.
        unsafe {
            let raw = device.result(
                device.raw.create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size(size)
                        .usage(flags)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    None,
                ),
                "create buffer",
            )?;
            let mut requirements = device.raw.get_buffer_memory_requirements(raw);
            if host.is_some() {
                // Host suballocations never share non-coherent atoms. Flushing
                // one arena cannot race GPU access to an adjacent allocation.
                let atom = device.properties.limits.non_coherent_atom_size;
                requirements.alignment = requirements.alignment.max(atom);
                requirements.size = requirements.size.div_ceil(atom) * atom;
            }
            let allocator = allocator.as_mut().unwrap();
            let allocation = match allocator.allocate(&AllocationCreateDesc {
                name: label,
                requirements,
                location: host.unwrap_or(MemoryLocation::GpuOnly),
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            }) {
                Ok(v) => v,
                Err(e) => {
                    device.raw.destroy_buffer(raw, None);
                    return Err(RuntimeError::caused_by(
                        ErrorKind::Allocation,
                        "Vulkan allocation",
                        e,
                    ));
                }
            };
            if let Err(error) = device.result(
                device
                    .raw
                    .bind_buffer_memory(raw, allocation.memory(), allocation.offset()),
                "bind buffer memory",
            ) {
                device.raw.destroy_buffer(raw, None);
                let _ = allocator.free(allocation);
                return Err(error);
            }
            device
                .committed_peak
                .fetch_max(allocator.capacity(), Ordering::Relaxed);
            // Release the allocator mutex before retaining the device.
            Ok(Self(Arc::new(BufferInner {
                device: device.clone(),
                raw,
                allocation: Some(allocation),
                size,
                usage,
                initialized: AtomicBool::new(false),
            })))
        }
    }
    pub(super) fn size(&self) -> u64 {
        self.0.size
    }
    pub(super) fn usage(&self) -> BufferUsage {
        self.0.usage
    }
    fn mapped(&self) -> RuntimeResult<(*mut u8, &Allocation)> {
        let allocation = self.0.allocation.as_ref().unwrap();
        let pointer = allocation.mapped_ptr().ok_or_else(|| {
            RuntimeError::with_kind(ErrorKind::GpuMapping, "buffer is not host visible")
        })?;
        Ok((pointer.as_ptr().cast(), allocation))
    }
    fn mapped_range(&self, allocation: &Allocation) -> vk::MappedMemoryRange<'_> {
        let atom = self.0.device.properties.limits.non_coherent_atom_size;
        debug_assert!(allocation.offset().is_multiple_of(atom));
        debug_assert!(allocation.size().is_multiple_of(atom));
        unsafe {
            vk::MappedMemoryRange::default()
                .memory(allocation.memory())
                .offset(allocation.offset())
                .size(allocation.size())
        }
    }
    fn write(&self, offset: u64, bytes: &[u8]) -> RuntimeResult<()> {
        if offset
            .checked_add(bytes.len() as u64)
            .is_none_or(|end| end > self.size())
        {
            return Err(resource("mapped upload exceeds allocation"));
        }
        let (pointer, allocation) = self.mapped()?;
        // SAFETY: staging is unpublished, exclusively initialized here, and the
        // bounds were validated before allocation.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                pointer.add(offset as usize),
                bytes.len(),
            );
            if !allocation
                .memory_properties()
                .contains(vk::MemoryPropertyFlags::HOST_COHERENT)
            {
                self.0.device.result(
                    self.0
                        .device
                        .raw
                        .flush_mapped_memory_ranges(&[self.mapped_range(allocation)]),
                    "flush upload",
                )?;
            }
        }
        self.0.initialized.store(true, Ordering::Release);
        Ok(())
    }
    pub(super) fn read<T>(
        &self,
        bytes: usize,
        consume: impl FnOnce(&[u8]) -> RuntimeResult<T>,
    ) -> RuntimeResult<T> {
        if bytes as u64 > self.size() {
            return Err(resource("readback exceeds Vulkan allocation"));
        }
        let (pointer, allocation) = self.mapped()?;
        // SAFETY: caller has waited for this submission, and the exclusive
        // session/readback lease prevents simultaneous host or GPU writes.
        unsafe {
            if !allocation
                .memory_properties()
                .contains(vk::MemoryPropertyFlags::HOST_COHERENT)
            {
                self.0.device.result(
                    self.0
                        .device
                        .raw
                        .invalidate_mapped_memory_ranges(&[self.mapped_range(allocation)]),
                    "invalidate readback",
                )?;
            }
            consume(std::slice::from_raw_parts(pointer, bytes))
        }
    }
}
impl Drop for BufferInner {
    fn drop(&mut self) {
        // SAFETY: submissions retain buffers until completion.
        unsafe {
            self.device.raw.destroy_buffer(self.raw, None);
        }
        if let Some(allocation) = self.allocation.take()
            && let Ok(mut allocator) = self.device.allocator.lock()
            && let Some(allocator) = allocator.as_mut()
        {
            let _ = allocator.free(allocation);
        }
    }
}
