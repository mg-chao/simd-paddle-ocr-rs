//! The sole compute executor inside the wgpu-owned Metal resource/queue bridge.
//!
//! Imported buffers retain one Metal resource across compute and transfer APIs. This module
//! records every compute encoder; the bridge owns the command buffer,
//! submission, completion, readbacks, and error scopes throughout.
use super::metal_buffers::NativeBuffer;
use super::*;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSError, NSString};
use objc2_metal::{
    MTLBarrierScope, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCompileOptions,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLDispatchType,
    MTLLanguageVersion, MTLLibrary, MTLSize,
};
use std::ptr::NonNull;

const SIZES_BUFFER_SLOT: usize = 30;

/// Immutable compiled Metal state and the metadata normally supplied by HAL.
pub(super) struct NativePipeline {
    raw: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    owner: Arc<Ledger>,
    pub(super) workgroup_size: [u32; 3],
    /// Runtime-array buffers in Naga GLOBAL declaration order, not slot order.
    sized_bindings: Vec<usize>,
    /// Active workgroup globals in declaration order; Metal requires 16B sizes.
    workgroup_memory: Vec<usize>,
    /// Active binding slots and their reflected WGSL minimum byte sizes.
    /// Manual MSL declares storage/uniform minima; its caller proves tensor extents.
    required_bindings: Vec<(usize, u64)>,
}

impl std::fmt::Debug for NativePipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativePipeline")
            .field("workgroup_size", &self.workgroup_size)
            .field("sized_bindings", &self.sized_bindings)
            .field("workgroup_memory", &self.workgroup_memory)
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
    pub(super) fn validate_binding_sizes(
        &self,
        count: usize,
        size: impl Fn(usize) -> u64,
    ) -> RuntimeResult<()> {
        for &(binding, minimum) in &self.required_bindings {
            if binding >= count {
                return Err(RuntimeError::with_kind(
                    ErrorKind::InvalidInput,
                    "native dispatch is missing a required buffer",
                ));
            }
            let actual = size(binding);
            if actual < minimum {
                return Err(RuntimeError::with_kind(
                    ErrorKind::InvalidInput,
                    format!(
                        "native binding {binding} has {actual} bytes, shader requires at least {minimum}"
                    ),
                ));
            }
        }
        Ok(())
    }
}

pub(super) struct NativeDispatch {
    // These bindings have no mutable/public access after preparation. The raw
    // resource length, PSO minima and runtime-size proof stay valid for the
    // complete lifetime of every shared Arc, including cached shape views.
    pipeline: Arc<NativePipeline>,
    buffers: Vec<NativeBuffer>,
    offsets: Vec<u64>,
    groups: [u32; 3],
    // Raw Metal handles keep allocations alive, but only these tracked owners
    // retain their device-quota reservations. No session/context backreference.
    owners: Vec<GpuBuffer>,
    hazards: Box<[(usize, bool)]>,
    runtime_sizes: Box<[u32]>,
}

impl std::fmt::Debug for NativeDispatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeDispatch")
            .field("groups", &self.groups)
            .field("bindings", &self.owners.len())
            .finish_non_exhaustive()
    }
}

impl NativeDispatch {
    /// Prepare immutable same-device bindings once. Every buffer owns its raw
    /// Metal resource; invalid ranges and foreign contexts remain errors.
    pub(super) fn prepare(
        pipeline: Arc<NativePipeline>,
        owners: Vec<GpuBuffer>,
        offsets: Vec<u64>,
        groups: [u32; 3],
        read_only: &[bool],
    ) -> RuntimeResult<Arc<Self>> {
        if owners.len() != offsets.len()
            || owners.len() != read_only.len() + 1
            || owners.len() > SIZES_BUFFER_SLOT
        {
            return Err(RuntimeError::with_kind(
                ErrorKind::InvalidInput,
                "native preparation requires matching storage, uniform and offset bindings",
            ));
        }
        if owners
            .iter()
            .any(|buffer| !Arc::ptr_eq(&pipeline.owner, &buffer.reservation.ledger))
        {
            return Err(RuntimeError::with_kind(
                ErrorKind::InvalidInput,
                "native dispatch buffers and pipeline must belong to the same GPU context",
            ));
        }
        if groups.iter().any(|&n| {
            n > pipeline
                .owner
                .owner_device
                .limits()
                .max_compute_workgroups_per_dimension
        }) {
            return Err(resource("native dispatch groups exceed device limits"));
        }
        for (index, owner) in owners.iter().enumerate() {
            let usage = if index + 1 == owners.len() {
                wgpu::BufferUsages::UNIFORM
            } else {
                wgpu::BufferUsages::STORAGE
            };
            if !owner.usage().contains(usage) {
                return Err(RuntimeError::with_kind(
                    ErrorKind::InvalidInput,
                    "native binding usage differs from its storage/uniform ABI",
                ));
            }
        }
        let buffers: Vec<_> = owners.iter().map(|buffer| buffer.native.clone()).collect();
        let mut remaining = [0; SIZES_BUFFER_SLOT];
        for (index, (buffer, &offset)) in buffers.iter().zip(&offsets).enumerate() {
            // Query the actual Metal resource once. Its allocation length is
            // immutable; tracked GpuBuffer owners survive every encode using
            // this dispatch and never expose buffer destruction here.
            let bytes = buffer.raw.length() as u64;
            if offset >= bytes || !offset.is_multiple_of(4) {
                return Err(resource(
                    "native buffer binding offset is outside its allocation or unaligned",
                ));
            }
            remaining[index] = bytes - offset;
        }
        pipeline.validate_binding_sizes(owners.len(), |binding| remaining[binding])?;
        let runtime_sizes =
            checked_runtime_sizes(&pipeline.sized_bindings, &remaining[..buffers.len()])?;
        let hazards = owners
            .iter()
            .zip(read_only)
            .map(|(buffer, &read)| (Arc::as_ptr(&buffer.reservation) as usize, read))
            .collect();
        Ok(Arc::new(Self {
            pipeline,
            buffers,
            offsets,
            groups,
            owners,
            hazards,
            runtime_sizes,
        }))
    }

    pub(super) fn groups(&self) -> [u32; 3] {
        self.groups
    }

    pub(super) fn hazards(&self) -> &[(usize, bool)] {
        &self.hazards
    }

    pub(super) fn owners(&self) -> &[GpuBuffer] {
        &self.owners
    }

    #[cfg(test)]
    pub(super) fn offsets(&self) -> &[u64] {
        &self.offsets
    }

    #[cfg(test)]
    pub(super) fn runtime_sizes(&self) -> &[u32] {
        &self.runtime_sizes
    }

    #[cfg(test)]
    pub(super) fn pipeline_for_test(&self) -> Arc<NativePipeline> {
        self.pipeline.clone()
    }

    pub(super) fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.buffers.capacity() * std::mem::size_of::<NativeBuffer>()
            + self.offsets.capacity() * std::mem::size_of::<u64>()
            + self.owners.capacity() * std::mem::size_of::<GpuBuffer>()
            + std::mem::size_of_val(self.hazards.as_ref())
            + std::mem::size_of_val(self.runtime_sizes.as_ref())
    }
}

fn checked_runtime_sizes(bindings: &[usize], remaining: &[u64]) -> RuntimeResult<Box<[u32]>> {
    bindings
        .iter()
        .map(|&binding| {
            let &bytes = remaining
                .get(binding)
                .ok_or_else(|| resource("native runtime-array size binding is absent"))?;
            u32::try_from(bytes)
                .map_err(|_| resource("native runtime-array binding exceeds U32 byte sizes"))
        })
        .collect()
}

fn unsupported(message: impl Into<String>) -> RuntimeError {
    RuntimeError::with_kind(ErrorKind::Unsupported, message)
}

fn native_device(device: &wgpu::Device) -> RuntimeResult<Retained<ProtocolObject<dyn MTLDevice>>> {
    // SAFETY: Borrow only the matching backend's device, clone its retained
    // Objective-C handle, and never mutate or destroy HAL-owned state.
    let hal = unsafe { device.as_hal::<wgpu::hal::api::Metal>() }
        .ok_or_else(|| unsupported("native execution requires the actual Metal device"))?;
    Ok(hal.raw_device().clone())
}

fn compiler_error(stage: &str, error: &NSError) -> RuntimeError {
    // Source/feature compiler rejection is optional. Internal driver/compiler
    // failure remains an execution error and must never qualify a fallback.
    let domain = error.domain().to_string();
    let kind = if domain == "MTLLibraryErrorDomain" && matches!(error.code(), 1 | 3 | 4 | 5) {
        ErrorKind::Unsupported
    } else {
        ErrorKind::Execution
    };
    RuntimeError::with_kind(kind, format!("{stage}: {error}"))
}

fn compile(
    context: &GpuContext,
    source: &str,
    entry: &str,
    workgroup_size: [u32; 3],
    sized_bindings: Vec<usize>,
    workgroup_memory: Vec<usize>,
    required_bindings: Vec<(usize, u64)>,
) -> RuntimeResult<Arc<NativePipeline>> {
    objc2::rc::autoreleasepool(|_| {
        compile_inner(
            context,
            source,
            entry,
            workgroup_size,
            sized_bindings,
            workgroup_memory,
            required_bindings,
        )
    })
}

fn compile_inner(
    context: &GpuContext,
    source: &str,
    entry: &str,
    workgroup_size: [u32; 3],
    sized_bindings: Vec<usize>,
    workgroup_memory: Vec<usize>,
    required_bindings: Vec<(usize, u64)>,
) -> RuntimeResult<Arc<NativePipeline>> {
    let threads = workgroup_size
        .into_iter()
        .try_fold(1u32, u32::checked_mul)
        .filter(|&n| n != 0)
        .ok_or_else(|| unsupported("native compute requires a nonzero workgroup size"))?;
    let limits = context.device.limits();
    if workgroup_size[0] > limits.max_compute_workgroup_size_x
        || workgroup_size[1] > limits.max_compute_workgroup_size_y
        || workgroup_size[2] > limits.max_compute_workgroup_size_z
        || threads > limits.max_compute_invocations_per_workgroup
    {
        return Err(unsupported(
            "native workgroup geometry exceeds device dispatch limits",
        ));
    }
    let owner = context.ledger.clone();
    let device = native_device(&context.device)?;
    let options = MTLCompileOptions::new();
    options.setLanguageVersion(MTLLanguageVersion::Version4_0);
    // Match wgpu Metal compilation so scheduling preserves shader arithmetic.
    options.setPreserveInvariance(true);
    let library = device
        .newLibraryWithSource_options_error(&NSString::from_str(source), Some(&options))
        .map_err(|error| compiler_error("native Metal source compilation", &error))?;
    let function = library
        .newFunctionWithName(&NSString::from_str(entry))
        .ok_or_else(|| unsupported(format!("native Metal entry {entry} is missing")))?;
    let raw = device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|error| compiler_error("native Metal pipeline compilation", &error))?;
    if threads as usize > raw.maxTotalThreadsPerThreadgroup() {
        return Err(unsupported(
            "native kernel exceeds its maximum threadgroup size",
        ));
    }
    let memory = workgroup_memory
        .iter()
        .try_fold(raw.staticThreadgroupMemoryLength(), |sum, &n| {
            sum.checked_add(n)
        })
        .ok_or_else(|| resource("native threadgroup memory size overflow"))?;
    if memory > device.maxThreadgroupMemoryLength()
        || memory > limits.max_compute_workgroup_storage_size as usize
    {
        return Err(unsupported(
            "native kernel exceeds device threadgroup memory",
        ));
    }
    Ok(Arc::new(NativePipeline {
        raw,
        owner,
        workgroup_size,
        sized_bindings,
        workgroup_memory,
        required_bindings,
    }))
}

/// Compile a shader using the shared native seven-buffer ABI.
pub(super) fn compile_msl(
    context: &GpuContext,
    source: &str,
    entry: &str,
    workgroup_size: [u32; 3],
) -> RuntimeResult<Arc<NativePipeline>> {
    compile(
        context,
        source,
        entry,
        workgroup_size,
        Vec::new(),
        Vec::new(),
        (0..7)
            .map(|binding| (binding, if binding == 6 { 512 } else { 4 }))
            .collect(),
    )
}

/// Compile an embedded translated kernel with its frozen Naga ABI metadata.
/// The caller must keep runtime sizes, dynamic threadgroup slots and minimum
/// binding extents consistent with the source; preparation validates each
/// actual buffer against these requirements before recording a dispatch.
pub(super) fn compile_msl_metadata(
    context: &GpuContext,
    source: &str,
    entry: &str,
    workgroup_size: [u32; 3],
    sized_bindings: Vec<usize>,
    workgroup_memory: Vec<usize>,
    required_bindings: Vec<(usize, u64)>,
) -> RuntimeResult<Arc<NativePipeline>> {
    compile(
        context,
        source,
        entry,
        workgroup_size,
        sized_bindings,
        workgroup_memory,
        required_bindings,
    )
}

/// Resident shaders use their declared contiguous storage operands followed by
/// one uniform buffer; unlike neural shaders they need not have seven bindings.
pub(super) fn compile_resident_msl(
    context: &GpuContext,
    source: &str,
    entry: &str,
    workgroup_size: [u32; 3],
    bindings: usize,
) -> RuntimeResult<Arc<NativePipeline>> {
    if bindings == 0 || bindings > SIZES_BUFFER_SLOT {
        return Err(unsupported("native resident binding count exceeds its ABI"));
    }
    compile(
        context,
        source,
        entry,
        workgroup_size,
        Vec::new(),
        Vec::new(),
        (0..bindings)
            .map(|binding| (binding, if binding + 1 == bindings { 1024 } else { 4 }))
            .collect(),
    )
}

/// Translate WGSL using the same bounds and workgroup initialization policies
/// as HAL. The WGSL entry declaration is authoritative for workgroup geometry;
/// the extra size argument keeps the registration call shared with native MSL.
pub(super) fn compile_wgsl(
    context: &GpuContext,
    source: &str,
    entry: &str,
    _workgroup_size: [u32; 3],
) -> RuntimeResult<Arc<NativePipeline>> {
    let module = naga::front::wgsl::parse_str(source).map_err(|error| {
        unsupported(format!(
            "native WGSL parsing: {}",
            error.emit_to_string(source)
        ))
    })?;
    if !module.overrides.is_empty() {
        return Err(unsupported(
            "native WGSL pipeline overrides require explicit specialization",
        ));
    }
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::back::msl::supported_capabilities(),
    )
    .validate(&module)
    .map_err(|error| unsupported(format!("native WGSL validation: {error}")))?;
    let entry_index = module
        .entry_points
        .iter()
        .position(|ep| ep.name == entry && ep.stage == naga::ShaderStage::Compute)
        .ok_or_else(|| unsupported(format!("native WGSL compute entry {entry} is missing")))?;
    let entry_info = info.get_entry_point(entry_index);
    let mut resources = naga::back::msl::EntryPointResources {
        sizes_buffer: Some(SIZES_BUFFER_SLOT as u8),
        ..Default::default()
    };
    let mut sized_bindings = Vec::new();
    let mut required_bindings = Vec::new();
    let mut workgroup_memory = Vec::new();
    for (handle, global) in module.global_variables.iter() {
        if global.space == naga::AddressSpace::WorkGroup && !entry_info[handle].is_empty() {
            let size = module.types[global.ty].inner.size(module.to_ctx());
            let size = size
                .checked_add(15)
                .map(|n| n / 16 * 16)
                .ok_or_else(|| resource("native workgroup allocation overflow"))?;
            workgroup_memory.push(size as usize);
        }
        let Some(binding) = global.binding else {
            continue;
        };
        if binding.group != 0
            || binding.binding as usize >= SIZES_BUFFER_SLOT
            || !matches!(
                global.space,
                naga::AddressSpace::Uniform | naga::AddressSpace::Storage { .. }
            )
        {
            return Err(unsupported(
                "native WGSL requires group-zero buffer bindings below slot30",
            ));
        }
        if matches!(
            module.types[global.ty].inner,
            naga::TypeInner::BindingArray { .. }
        ) {
            return Err(unsupported(
                "native WGSL buffer binding arrays are unsupported",
            ));
        }
        let mutable = matches!(global.space, naga::AddressSpace::Storage { access } if access.contains(naga::StorageAccess::STORE));
        resources.resources.insert(
            binding,
            naga::back::msl::BindTarget {
                buffer: Some(binding.binding as u8),
                mutable,
                ..Default::default()
            },
        );
        if !entry_info[handle].is_empty() {
            // Match wgpu-core's ResourceType::Buffer requirement. This includes
            // struct padding and one element of a trailing runtime array;
            // checking only scalar alignment is insufficient for fixed arrays.
            let minimum = module.types[global.ty].inner.size(module.to_ctx());
            required_bindings.push((binding.binding as usize, u64::from(minimum)));
        }
        if module.types[global.ty]
            .inner
            .needs_host_buffer_byte_size(&module.types)
        {
            // This loop intentionally includes unused runtime arrays, exactly
            // matching Naga's generated _mslBufferSizes struct declaration.
            sized_bindings.push(binding.binding as usize);
        }
    }
    let options = naga::back::msl::Options {
        lang_version: (4, 0),
        per_entry_point_map: std::collections::BTreeMap::from([(entry.to_owned(), resources)]),
        fake_missing_bindings: false,
        bounds_check_policies: naga::proc::BoundsCheckPolicies {
            index: naga::proc::BoundsCheckPolicy::Restrict,
            buffer: naga::proc::BoundsCheckPolicy::Restrict,
            image_load: naga::proc::BoundsCheckPolicy::Restrict,
            binding_array: naga::proc::BoundsCheckPolicy::Unchecked,
        },
        zero_initialize_workgroup_memory: true,
        ..Default::default()
    };
    let pipeline_options = naga::back::msl::PipelineOptions {
        entry_point: Some((naga::ShaderStage::Compute, entry.to_owned())),
        ..Default::default()
    };
    let (msl, translated) =
        naga::back::msl::write_string(&module, &info, &options, &pipeline_options)
            .map_err(|error| unsupported(format!("native WGSL Metal translation: {error}")))?;
    let translated_entry = translated
        .entry_point_names
        .first()
        .ok_or_else(|| unsupported("native WGSL translation produced no entry"))?
        .as_ref()
        .map_err(|error| unsupported(format!("native WGSL entry translation: {error}")))?;
    compile(
        context,
        &msl,
        translated_entry,
        module.entry_points[entry_index].workgroup_size,
        sized_bindings,
        workgroup_memory,
        required_bindings,
    )
}

fn dispatch_type<T>(dispatches: &[(usize, T)], prefer_serial: bool) -> MTLDispatchType {
    if !prefer_serial && dispatches.windows(2).any(|pair| pair[0].0 == pair[1].0) {
        MTLDispatchType::Concurrent
    } else {
        MTLDispatchType::Serial
    }
}

/// Record a complete depth-sorted compute chunk. Each level must contain
/// independent accesses; the resident scheduler derives depths from actual
/// imported buffer identities, including aliasing and packed weight resources.
pub(super) fn encode(
    context: &GpuContext,
    encoder: &mut wgpu::CommandEncoder,
    dispatches: &[(usize, &NativeDispatch)],
    prefer_serial: bool,
) -> RuntimeResult<()> {
    if dispatches.is_empty() {
        return Ok(());
    }
    let mut previous_depth = dispatches[0].0;
    for &(depth, dispatch) in dispatches {
        if !Arc::ptr_eq(&dispatch.pipeline.owner, &context.ledger) {
            return Err(RuntimeError::with_kind(
                ErrorKind::InvalidInput,
                "native pipeline belongs to another encoder context",
            ));
        }
        if depth < previous_depth
            || dispatch.buffers.len() != dispatch.offsets.len()
            || dispatch.buffers.len() > SIZES_BUFFER_SLOT
        {
            return Err(RuntimeError::with_kind(
                ErrorKind::InvalidInput,
                "native dispatches require sorted depths and matching buffer offsets",
            ));
        }
        previous_depth = depth;
    }
    let dispatch_type = dispatch_type(dispatches, prefer_serial);
    // SAFETY: The caller supplies a fresh raw-only wgpu encoder, because wgpu
    // forbids mixing raw compute and transfer recording on the same encoder. We create
    // and end only a compute encoder on its HAL command buffer;
    // wgpu finishes/submits it in order with the surrounding transfer buffers.
    // All imported buffers remain owned until the entire submission completes.
    objc2::rc::autoreleasepool(|_| unsafe {
        encoder.as_hal_mut::<wgpu::hal::api::Metal, _, _>(|hal| {
            let hal =
                hal.ok_or_else(|| unsupported("native encoding requires a Metal command encoder"))?;
            let command = hal.raw_command_buffer().ok_or_else(|| {
                RuntimeError::with_kind(ErrorKind::Execution, "Metal command buffer is unavailable")
            })?;
            let compute = command
                .computeCommandEncoderWithDispatchType(dispatch_type)
                .ok_or_else(|| {
                    RuntimeError::with_kind(
                        ErrorKind::Allocation,
                        "Metal compute encoder allocation failed",
                    )
                })?;
            let mut previous_depth = dispatches[0].0;
            let mut bound_pipeline = None;
            let mut bound_buffers: [Option<(&NativeBuffer, u64)>; SIZES_BUFFER_SLOT] =
                [None; SIZES_BUFFER_SLOT];
            for &(depth, dispatch) in dispatches {
                let sizes = &dispatch.runtime_sizes;
                if depth != previous_depth {
                    // Serial encoders execute dispatches in order and our
                    // imported buffers explicitly use tracked hazards. RAW,
                    // WAR and WAW therefore require no explicit barrier for a
                    // single graph, even across its internal DAG branches, or
                    // a chunk with one dispatch at each depth. Other batches
                    // retain concurrent encoding and independent-level overlap.
                    if dispatch_type == MTLDispatchType::Concurrent {
                        compute.memoryBarrierWithScope(MTLBarrierScope::Buffers);
                    }
                    previous_depth = depth;
                }
                if bound_pipeline.is_none_or(|previous| !Arc::ptr_eq(previous, &dispatch.pipeline))
                {
                    compute.setComputePipelineState(&dispatch.pipeline.raw);
                    bound_pipeline = Some(&dispatch.pipeline);
                }
                for (index, (buffer, &offset)) in
                    dispatch.buffers.iter().zip(&dispatch.offsets).enumerate()
                {
                    // Metal buffer bindings persist across pipeline changes and
                    // barriers. Compare the actual retained object, including
                    // the offset: graph parameter packets share one allocation.
                    // This cache is local to this encoder; transfer boundaries
                    // and later command buffers always start with empty state.
                    if !bound_buffers[index].is_some_and(|(previous, previous_offset)| {
                        previous_offset == offset && std::ptr::eq(&*previous.raw, &*buffer.raw)
                    }) {
                        compute.setBuffer_offset_atIndex(Some(&buffer.raw), offset as usize, index);
                        bound_buffers[index] = Some((buffer, offset));
                    }
                }
                if !sizes.is_empty() {
                    compute.setBytes_length_atIndex(
                        NonNull::new_unchecked(sizes.as_ptr().cast_mut().cast()),
                        sizes.len() * std::mem::size_of::<u32>(),
                        SIZES_BUFFER_SLOT,
                    );
                }
                for (index, &bytes) in dispatch.pipeline.workgroup_memory.iter().enumerate() {
                    compute.setThreadgroupMemoryLength_atIndex(bytes, index);
                }
                let groups = dispatch.groups();
                if groups.iter().all(|&n| n != 0) {
                    compute.dispatchThreadgroups_threadsPerThreadgroup(
                        MTLSize {
                            width: groups[0] as usize,
                            height: groups[1] as usize,
                            depth: groups[2] as usize,
                        },
                        MTLSize {
                            width: dispatch.pipeline.workgroup_size[0] as usize,
                            height: dispatch.pipeline.workgroup_size[1] as usize,
                            depth: dispatch.pipeline.workgroup_size[2] as usize,
                        },
                    );
                }
            }
            compute.endEncoding();
            Ok(())
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepared_runtime_size_table_preserves_declaration_order_and_resource_errors() {
        assert_eq!(
            checked_runtime_sizes(&[2, 0, 1], &[4, 512, 128])
                .unwrap()
                .as_ref(),
            &[128, 4, 512],
        );
        assert!(checked_runtime_sizes(&[], &[]).unwrap().is_empty());
        assert_eq!(
            checked_runtime_sizes(&[1], &[4]).unwrap_err().kind(),
            ErrorKind::ResourceLimit,
        );
        assert_eq!(
            checked_runtime_sizes(&[0], &[u64::from(u32::MAX) + 1])
                .unwrap_err()
                .kind(),
            ErrorKind::ResourceLimit,
        );
        assert_eq!(
            checked_runtime_sizes(&[0], &[u64::from(u32::MAX)])
                .unwrap()
                .as_ref(),
            &[u32::MAX]
        );
    }

    #[test]
    fn encoder_policy_preserves_parallel_batches_and_serial_single_graphs() {
        // Chunk boundaries can leave gaps or a nonzero first depth. None of
        // these chains offers simultaneous work, including a single dispatch.
        for depths in [vec![], vec![7], vec![0, 1, 2, 3], vec![7, 9, 14]] {
            let dispatches = depths
                .into_iter()
                .map(|depth| (depth, ()))
                .collect::<Vec<_>>();
            assert_eq!(dispatch_type(&dispatches, false), MTLDispatchType::Serial);
            assert_eq!(dispatch_type(&dispatches, true), MTLDispatchType::Serial);
        }
        // Resident-only and multiple-graph batches retain concurrent fork/join
        // work. One graph instead uses ordered serial dispatch for its entire
        // DAG, including parallel branches and the resident work around it.
        for depths in [vec![0, 1, 1, 2], vec![7, 7, 8, 8, 9], vec![0, 1, 2, 2]] {
            let dispatches = depths
                .into_iter()
                .map(|depth| (depth, ()))
                .collect::<Vec<_>>();
            assert_eq!(
                dispatch_type(&dispatches, false),
                MTLDispatchType::Concurrent
            );
            assert_eq!(dispatch_type(&dispatches, true), MTLDispatchType::Serial);
        }
    }
}
