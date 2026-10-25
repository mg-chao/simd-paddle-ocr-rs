use super::*;

/// A kernel's immutable, native buffer interface. The final binding is its
/// parameter packet; accesses and byte ranges are checked before recording.
pub(crate) struct NativePipeline {
    pub(super) raw: Native<dyn MTLComputePipelineState>,
    _library: Native<dyn MTLLibrary>,
    owner: Arc<Ledger>,
    pub(crate) workgroup_size: [u32; 3],
    pub(super) minimum_bytes: Vec<u64>,
    pub(super) read_only: Vec<bool>,
}
impl std::fmt::Debug for NativePipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalPipeline")
            .field("workgroup", &self.workgroup_size)
            .finish()
    }
}
impl PartialEq for NativePipeline {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other)
    }
}
impl Eq for NativePipeline {}

pub(crate) struct NativeDispatch {
    pub(super) pipeline: Arc<NativePipeline>,
    pub(super) buffers: Vec<GpuBuffer>,
    pub(super) offsets: Vec<u64>,
    pub(super) groups: [u32; 3],
    hazards: Box<[(usize, bool)]>,
}
impl std::fmt::Debug for NativeDispatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalDispatch")
            .field("groups", &self.groups)
            .finish()
    }
}
impl NativeDispatch {
    #[cfg(test)]
    pub(crate) fn pipeline_for_test(&self) -> Arc<NativePipeline> {
        self.pipeline.clone()
    }
    #[cfg(test)]
    pub(crate) fn offsets(&self) -> &[u64] {
        &self.offsets
    }

    pub(crate) fn prepare(
        pipeline: Arc<NativePipeline>,
        buffers: Vec<GpuBuffer>,
        offsets: Vec<u64>,
        groups: [u32; 3],
        read_only: &[bool],
    ) -> RuntimeResult<Arc<Self>> {
        if buffers.len() != offsets.len()
            || buffers.len() != pipeline.minimum_bytes.len()
            || read_only != pipeline.read_only
            || buffers.len() != read_only.len() + 1
        {
            return Err(RuntimeError::with_kind(
                ErrorKind::InvalidInput,
                "Metal kernel binding layout differs",
            ));
        }
        if groups.contains(&0)
            || groups.iter().any(|&v| {
                v > pipeline
                    .owner
                    .owner_device
                    .limits
                    .max_compute_workgroups_per_dimension
            })
        {
            return Err(resource("Metal dispatch grid exceeds planner limits"));
        }
        let invocations = groups
            .iter()
            .chain(&pipeline.workgroup_size)
            .try_fold(1u64, |count, &dimension| {
                count.checked_mul(u64::from(dimension))
            });
        if invocations.is_none_or(|count| count > u64::from(u32::MAX)) {
            return Err(resource("Metal dispatch exceeds U32 kernel addressing"));
        }
        for (i, ((buffer, &offset), &minimum)) in buffers
            .iter()
            .zip(&offsets)
            .zip(&pipeline.minimum_bytes)
            .enumerate()
        {
            if !Arc::ptr_eq(&buffer.reservation.ledger, &pipeline.owner) {
                return Err(RuntimeError::with_kind(
                    ErrorKind::InvalidInput,
                    "foreign Metal binding",
                ));
            }
            let uniform = i == buffers.len() - 1;
            let alignment = if uniform { 16 } else { 4 };
            if !buffer.size().is_multiple_of(4)
                || !offset.is_multiple_of(alignment)
                || offset
                    .checked_add(minimum)
                    .is_none_or(|end| end > buffer.size())
            {
                return Err(resource(
                    "Metal binding is undersized, outside its allocation, or unaligned",
                ));
            }
            if !buffer.usage().contains(if uniform {
                BufferUsage::UNIFORM
            } else {
                BufferUsage::STORAGE
            }) {
                return Err(RuntimeError::with_kind(
                    ErrorKind::InvalidInput,
                    "Metal binding storage role differs",
                ));
            }
        }
        let hazards = buffers
            .iter()
            .enumerate()
            .map(|(i, b)| {
                (
                    resident::schedule::identity(b),
                    read_only.get(i).copied().unwrap_or(true),
                )
            })
            .collect();
        Ok(Arc::new(Self {
            pipeline,
            buffers,
            offsets,
            groups,
            hazards,
        }))
    }
    pub(crate) fn hazards(&self) -> &[(usize, bool)] {
        &self.hazards
    }
    pub(crate) fn owners(&self) -> &[GpuBuffer] {
        &self.buffers
    }
    pub(crate) fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.buffers.capacity() * std::mem::size_of::<GpuBuffer>()
            + self.offsets.capacity() * 8
            + std::mem::size_of_val(self.hazards.as_ref())
    }
}

pub(super) fn compile(
    context: &GpuContext,
    source: &str,
    entry: &str,
    workgroup: [u32; 3],
    minimum_bytes: Vec<u64>,
    read_only: Vec<bool>,
) -> RuntimeResult<Arc<NativePipeline>> {
    compile_constants(
        context,
        source,
        entry,
        workgroup,
        minimum_bytes,
        read_only,
        &[],
        MTLMathFloatingPointFunctions::Precise,
    )
}
#[allow(clippy::too_many_arguments)]
pub(super) fn compile_constants(
    context: &GpuContext,
    source: &str,
    entry: &str,
    workgroup: [u32; 3],
    minimum_bytes: Vec<u64>,
    read_only: Vec<bool>,
    constants: &[(usize, u32)],
    functions: MTLMathFloatingPointFunctions,
) -> RuntimeResult<Arc<NativePipeline>> {
    context.check_alive()?;
    objc2::rc::autoreleasepool(|_| {
        let device = &context.backend.owner;
        let options = MTLCompileOptions::new();
        options.setLanguageVersion(MTLLanguageVersion::Version4_0);
        options.setPreserveInvariance(true);
        // F32 is the reference arithmetic policy. Optimized kernels explicitly
        // opt into matrix instructions; compiler-wide reassociation is disabled.
        options.setMathMode(MTLMathMode::Safe);
        options.setMathFloatingPointFunctions(functions);
        let descriptor = MTL4LibraryDescriptor::new();
        descriptor.setSource(Some(&NSString::from_str(source)));
        descriptor.setOptions(Some(&options));
        let library = device
            .compiler
            .newLibraryWithDescriptor_error(&descriptor)
            .map_err(|e| native_error(entry, &e))?;
        let function = MTL4LibraryFunctionDescriptor::new();
        function.setLibrary(Some(&library));
        function.setName(Some(&NSString::from_str(entry)));
        let descriptor = MTL4ComputePipelineDescriptor::new();
        let specialized = MTL4SpecializedFunctionDescriptor::new();
        specialized.setFunctionDescriptor(Some(&function));
        let values = MTLFunctionConstantValues::new();
        for (index, value) in constants {
            // SAFETY: Metal copies a properly aligned u32 while this borrowed
            // value is live. The typed catalog defines these constant indices.
            unsafe {
                values.setConstantValue_type_atIndex(
                    std::ptr::NonNull::from(value).cast(),
                    MTLDataType::UInt,
                    *index,
                );
            }
        }
        specialized.setConstantValues(Some(&values));
        descriptor.setComputeFunctionDescriptor(Some(&specialized));
        descriptor.setRequiredThreadsPerThreadgroup(size(workgroup));
        let raw = device
            .compiler
            .newComputePipelineStateWithDescriptor_compilerTaskOptions_error(&descriptor, None)
            .map_err(|e| native_error(entry, &e))?;
        let threads = workgroup
            .iter()
            .try_fold(1u32, |a, &b| a.checked_mul(b))
            .ok_or_else(|| resource("Metal workgroup overflow"))?;
        if threads == 0
            || threads as usize > raw.maxTotalThreadsPerThreadgroup()
            || raw.staticThreadgroupMemoryLength() > device.raw.maxThreadgroupMemoryLength()
        {
            return Err(unsupported("Metal kernel exceeds device execution limits"));
        }
        Ok(Arc::new(NativePipeline {
            raw,
            _library: library,
            owner: context.ledger.clone(),
            workgroup_size: workgroup,
            minimum_bytes,
            read_only,
        }))
    })
}
