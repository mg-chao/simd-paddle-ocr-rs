//! Native Metal ownership and submission. All platform objects stay in this module.
use super::*;
use block2::RcBlock;
use dispatch2::DispatchData;
use objc2::{
    rc::{Retained, autoreleasepool},
    runtime::ProtocolObject,
};
use objc2_foundation::NSString;
use objc2_metal::*;
use resident::schedule::{Operation, identity};
use std::{
    collections::HashMap,
    ptr::NonNull,
    sync::{OnceLock, Weak},
    time::{Duration, Instant},
};
mod counters;
use counters::Counters;
mod archive;
use archive::Archive;

struct Shader {
    name: String,
    threads: u32,
    reads: &'static [bool],
    packet: usize,
}
include!(concat!(env!("OUT_DIR"), "/metal_catalog.rs"));

pub(super) struct Device {
    raw: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    library: OnceLock<RuntimeResult<Retained<ProtocolObject<dyn MTLLibrary>>>>,
    archive: OnceLock<Option<Archive>>,
    archive_hits: AtomicUsize,
    archive_misses: AtomicUsize,
    serial: Mutex<()>,
    failure: Arc<Mutex<Option<RuntimeError>>>,
    available: Arc<Condvar>,
    pub(super) limits: GpuLimits,
    committed: AtomicUsize,
    peak: AtomicUsize,
}
impl Device {
    fn retire(&self, error: RuntimeError) {
        if let Ok(mut state) = self.failure.lock()
            && state.is_none()
        {
            *state = Some(error);
        }
        self.available.notify_all();
    }
    pub(super) fn committed_bytes(&self) -> usize {
        self.committed.load(Ordering::Acquire)
    }
    pub(super) fn peak_committed_bytes(&self) -> usize {
        self.peak.load(Ordering::Acquire)
    }
    fn library(&self) -> RuntimeResult<&ProtocolObject<dyn MTLLibrary>> {
        self.library
            .get_or_init(|| {
                let bytes = embedded_resource(
                    include_bytes!(concat!(env!("OUT_DIR"), "/ocr.metallib.deflate")),
                    env!("SIMD_OCR_METAL_IR_BYTES"),
                )?;
                let data = DispatchData::from_bytes(&bytes);
                self.raw.newLibraryWithData_error(&data).map_err(|e| {
                    RuntimeError::with_kind(
                        ErrorKind::GpuValidation,
                        format!("load embedded Metal library: {e}"),
                    )
                })
            })
            .as_ref()
            .map(|x| &**x)
            .map_err(Clone::clone)
    }
}

fn embedded_resource(bytes: &[u8], expected: &str) -> RuntimeResult<Vec<u8>> {
    let expected: usize = expected
        .parse()
        .expect("build-generated Metal resource length");
    let data =
        miniz_oxide::inflate::decompress_to_vec_with_limit(bytes, expected).map_err(|e| {
            RuntimeError::with_kind(
                ErrorKind::GpuValidation,
                format!("invalid embedded Metal resource: {e:?}"),
            )
        })?;
    if data.len() != expected {
        return Err(RuntimeError::with_kind(
            ErrorKind::GpuValidation,
            "truncated embedded Metal resource",
        ));
    }
    Ok(data)
}

pub(super) struct Context {
    pub(super) owner: Arc<Device>,
    uploads: Mutex<Vec<Upload>>,
}
struct Upload {
    target: Buffer,
    reservation: Weak<Reservation>,
    target_owner: Option<Arc<Reservation>>,
    source: Buffer,
    offset: usize,
    bytes: usize,
    _reservation: Arc<Reservation>,
}
struct PendingUploads<'a> {
    context: &'a Context,
    entries: Vec<Upload>,
}
impl Drop for PendingUploads<'_> {
    fn drop(&mut self) {
        // Recording failed before commit. Restore snapshots so a recoverable
        // validation/allocation error cannot silently discard model uploads.
        if !self.entries.is_empty() {
            for upload in &mut self.entries {
                upload.target_owner = None;
            }
            if let Ok(mut pending) = self.context.uploads.lock() {
                pending.splice(0..0, self.entries.drain(..));
            }
        }
    }
}
impl Context {
    pub(super) fn new(
        failure: Arc<Mutex<Option<RuntimeError>>>,
        available: Arc<Condvar>,
    ) -> RuntimeResult<(Self, GpuAdapterInfo, GpuLimits)> {
        autoreleasepool(|_| {
            if objc2_foundation::NSProcessInfo::processInfo()
                .operatingSystemVersion()
                .majorVersion
                < 14
            {
                return Err(unsupported("Metal OCR requires macOS 14 or newer"));
            }
            if !cfg!(target_arch = "aarch64") {
                return Err(unsupported("Metal OCR requires Apple Silicon"));
            }
            let raw =
                MTLCreateSystemDefaultDevice().ok_or_else(|| unsupported("no Metal device"))?;
            if !raw.hasUnifiedMemory() || !raw.supportsFamily(MTLGPUFamily::Apple7) {
                return Err(unsupported(
                    "Metal OCR requires an Apple7-or-newer unified-memory GPU",
                ));
            }
            let queue = raw.newCommandQueue().ok_or_else(|| {
                RuntimeError::with_kind(ErrorKind::Allocation, "create Metal command queue")
            })?;
            let maximum = raw.maxThreadsPerThreadgroup();
            let limits = GpuLimits {
                max_buffer_size: raw.maxBufferLength() as u64,
                max_storage_buffer_binding_size: raw.maxBufferLength() as u64,
                max_uniform_buffer_binding_size: 65536,
                max_compute_workgroups_per_dimension: 65535,
                max_compute_workgroup_storage_size: raw.maxThreadgroupMemoryLength() as u32,
                max_compute_invocations_per_workgroup: maximum.width as u32,
                max_compute_workgroup_size_x: maximum.width as u32,
                max_compute_workgroup_size_y: maximum.height as u32,
                max_compute_workgroup_size_z: maximum.depth as u32,
                min_uniform_buffer_offset_alignment: 16,
                min_storage_buffer_offset_alignment: 16,
            };
            let info = GpuAdapterInfo {
                name: raw.name().to_string(),
                vendor: 0x106b,
                device: 0,
                device_type: GpuDeviceType::IntegratedGpu,
                backend: GpuApi::Metal,
                driver: "Apple Metal".into(),
                driver_info: "native command queue; embedded MSL 3.1 library".into(),
                api_version: 0x30001,
                subgroup_min_size: 32,
                subgroup_max_size: 32,
            };
            let owner = Arc::new(Device {
                raw,
                queue,
                limits: limits.clone(),
                failure,
                available,
                library: OnceLock::new(),
                archive: OnceLock::new(),
                archive_hits: AtomicUsize::new(0),
                archive_misses: AtomicUsize::new(0),
                serial: Mutex::new(()),
                committed: AtomicUsize::new(0),
                peak: AtomicUsize::new(0),
            });
            Ok((
                Self {
                    owner,
                    uploads: Mutex::new(Vec::new()),
                },
                info,
                limits,
            ))
        })
    }
    pub(super) fn supports_half(&self) -> bool {
        true
    }
    pub(super) fn collect_garbage(&self) {
        if let Ok(mut uploads) = self.uploads.lock() {
            uploads.retain(|upload| upload.reservation.strong_count() != 0);
        }
    }
    pub(super) fn checked<T>(
        &self,
        _: &str,
        operation: impl FnOnce() -> RuntimeResult<T>,
    ) -> RuntimeResult<T> {
        operation()
    }
    pub(super) fn allocate(
        &self,
        label: &str,
        size: u64,
        usage: BufferUsage,
    ) -> RuntimeResult<Buffer> {
        Buffer::allocate(self.owner.clone(), label, size, usage)
    }
    pub(super) fn upload(
        &self,
        target: &GpuBuffer,
        offset: u64,
        bytes: &[u8],
    ) -> RuntimeResult<()> {
        if offset
            .checked_add(bytes.len() as u64)
            .is_none_or(|end| end > target.size())
            || !offset.is_multiple_of(4)
            || !bytes.len().is_multiple_of(4)
        {
            return Err(resource("invalid Metal upload range"));
        }
        if bytes.is_empty() {
            return Ok(());
        }
        let reservation = target.reservation.ledger.reserve(bytes.len())?;
        let source = self.allocate("upload snapshot", bytes.len() as u64, BufferUsage::COPY_SRC)?;
        // SAFETY: source is new and has never been published to another thread/GPU command.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                source.0.raw.contents().as_ptr().cast(),
                bytes.len(),
            );
        }
        self.uploads
            .lock()
            .map_err(|_| RuntimeError::new("Metal upload queue poisoned"))?
            .push(Upload {
                target: target.buffer.clone(),
                reservation: Arc::downgrade(&target.reservation),
                target_owner: None,
                source,
                offset: offset as usize,
                bytes: bytes.len(),
                _reservation: reservation,
            });
        Ok(())
    }
}

#[derive(Clone)]
pub(super) struct Buffer(Arc<BufferInner>);
struct BufferInner {
    raw: Retained<ProtocolObject<dyn MTLBuffer>>,
    owner: Arc<Device>,
    size: usize,
    allocated: usize,
    usage: BufferUsage,
}
// SAFETY: native buffer metadata is immutable after construction. CPU writes
// target unpublished upload snapshots only; CPU reads use the executor's
// exclusive readback lease after completion. GPU access is ordered on the
// context's single queue, with barriers for concurrent dispatch dependencies.
// No raw pointer or mutable slice is exposed outside this platform module.
unsafe impl Send for BufferInner {}
unsafe impl Sync for BufferInner {}
impl Buffer {
    fn allocate(
        owner: Arc<Device>,
        label: &str,
        size: u64,
        usage: BufferUsage,
    ) -> RuntimeResult<Self> {
        let size = usize::try_from(size)
            .map_err(|_| resource("Metal allocation exceeds host index range"))?;
        if size == 0 || size as u64 > owner.limits.max_buffer_size {
            return Err(resource("Metal allocation exceeds device limit"));
        }
        let raw = owner
            .raw
            .newBufferWithLength_options(size, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| {
                RuntimeError::with_kind(
                    ErrorKind::Allocation,
                    format!("Metal allocation of {size} bytes failed"),
                )
            })?;
        raw.setLabel(Some(&NSString::from_str(label)));
        // SAFETY: allocation has not escaped, and every byte is initialized before first use.
        unsafe {
            std::ptr::write_bytes(raw.contents().as_ptr().cast::<u8>(), 0, size);
        }
        let allocated = MTLResource::allocatedSize(&*raw);
        let committed = owner.committed.fetch_add(allocated, Ordering::AcqRel) + allocated;
        owner.peak.fetch_max(committed, Ordering::AcqRel);
        Ok(Self(Arc::new(BufferInner {
            raw,
            owner,
            size,
            allocated,
            usage,
        })))
    }
    pub(super) fn size(&self) -> u64 {
        self.0.size as u64
    }
    pub(super) fn usage(&self) -> BufferUsage {
        self.0.usage
    }
    pub(super) fn read<T>(
        &self,
        bytes: usize,
        consume: impl FnOnce(&[u8]) -> RuntimeResult<T>,
    ) -> RuntimeResult<T> {
        if bytes > self.0.size || !self.usage().contains(BufferUsage::MAP_READ) {
            return Err(resource("invalid Metal readback range or usage"));
        }
        // SAFETY: the shared executor waits for completion and exclusively leases the
        // readback buffer until this callback returns. Apple unified memory is coherent.
        let data =
            unsafe { std::slice::from_raw_parts(self.0.raw.contents().as_ptr().cast(), bytes) };
        consume(data)
    }
}
impl PartialEq for Buffer {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
impl Eq for Buffer {}
impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalBuffer")
            .field("bytes", &self.0.size)
            .finish()
    }
}
impl Drop for BufferInner {
    fn drop(&mut self) {
        self.owner
            .committed
            .fetch_sub(self.allocated, Ordering::AcqRel);
    }
}

pub(super) struct NativePipeline {
    owner: Arc<Ledger>,
    raw: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    spec: Shader,
    pub(super) workgroup_size: [u32; 3],
}
impl NativePipeline {
    pub(super) fn validate_binding_sizes(
        &self,
        count: usize,
        remaining: impl Fn(usize) -> u64,
    ) -> RuntimeResult<()> {
        if count != self.spec.reads.len() + 1 {
            return Err(RuntimeError::with_kind(
                ErrorKind::InvalidInput,
                "Metal binding count differs from catalog",
            ));
        }
        for index in 0..count {
            let required = if index + 1 == count {
                self.spec.packet
            } else {
                4
            };
            if remaining(index) < required as u64 {
                return Err(RuntimeError::with_kind(
                    ErrorKind::InvalidInput,
                    format!("Metal binding {index} is smaller than {required} bytes"),
                ));
            }
        }
        Ok(())
    }
}
impl PartialEq for NativePipeline {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(&*self.raw, &*other.raw)
    }
}
impl Eq for NativePipeline {}
impl std::fmt::Debug for NativePipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("MetalPipeline")
            .field(&self.spec.name)
            .finish()
    }
}

pub(super) fn compile(context: &GpuContext, key: KernelKey) -> RuntimeResult<Arc<NativePipeline>> {
    context.check_alive()?;
    autoreleasepool(|_| {
        let spec = shader(key)?;
        let device = &context.backend.owner;
        let function = device
            .library()?
            .newFunctionWithName(&NSString::from_str(&spec.name))
            .ok_or_else(|| {
                RuntimeError::with_kind(
                    ErrorKind::GpuValidation,
                    format!("missing Metal entry {}", spec.name),
                )
            })?;
        let cached = device
            .archive
            .get_or_init(|| Archive::load(device))
            .as_ref()
            .and_then(|archive| archive.pipeline(device, &function));
        let raw = if let Some(raw) = cached {
            device.archive_hits.fetch_add(1, Ordering::Relaxed);
            raw
        } else {
            device.archive_misses.fetch_add(1, Ordering::Relaxed);
            device
                .raw
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|error| {
                    RuntimeError::with_kind(
                        ErrorKind::GpuValidation,
                        format!("Metal pipeline {}: {error}", spec.name),
                    )
                })?
        };
        if spec.threads as usize > raw.maxTotalThreadsPerThreadgroup() {
            return Err(resource(
                "Metal pipeline threadgroup exceeds its native limit",
            ));
        }
        Ok(Arc::new(NativePipeline {
            owner: context.ledger.clone(),
            workgroup_size: [spec.threads, 1, 1],
            raw,
            spec,
        }))
    })
}
pub(super) fn compile_resident(
    context: &GpuContext,
    key: KernelKey,
    read_only: &[bool],
) -> RuntimeResult<Arc<NativePipeline>> {
    let pipeline = compile(context, key)?;
    if pipeline.spec.reads != read_only {
        return Err(RuntimeError::with_kind(
            ErrorKind::InvalidInput,
            "Metal access contract differs from catalog",
        ));
    }
    Ok(pipeline)
}

// Encoder state is local to one native compute pass. Pipeline and buffer
// identities stay live in the retained operation list until completion.
#[derive(Default)]
struct EncodingState {
    pipeline: Option<usize>,
    buffers: Vec<(usize, u64)>,
}

pub(super) struct NativeDispatch {
    pipeline: Arc<NativePipeline>,
    owners: Vec<GpuBuffer>,
    offsets: Vec<u64>,
    groups: [u32; 3],
    hazards: Vec<(usize, bool)>,
}
impl NativeDispatch {
    pub(super) fn prepare(
        pipeline: Arc<NativePipeline>,
        owners: Vec<GpuBuffer>,
        offsets: Vec<u64>,
        groups: [u32; 3],
        read_only: &[bool],
    ) -> RuntimeResult<Arc<Self>> {
        if owners.len() != offsets.len()
            || owners.len() != read_only.len() + 1
            || pipeline.spec.reads != read_only
        {
            return Err(RuntimeError::with_kind(
                ErrorKind::InvalidInput,
                "Metal binding/access count differs",
            ));
        }
        let limits = &pipeline.owner.owner_device.limits;
        if groups
            .iter()
            .any(|&n| n == 0 || n > limits.max_compute_workgroups_per_dimension)
        {
            return Err(resource("Metal dispatch grid exceeds device limits"));
        }
        let mut accesses = HashMap::new();
        for (i, (buffer, offset)) in owners.iter().zip(&offsets).enumerate() {
            let uniform = i == read_only.len();
            let (usage, alignment) = if uniform {
                (
                    BufferUsage::UNIFORM,
                    limits.min_uniform_buffer_offset_alignment,
                )
            } else {
                (
                    BufferUsage::STORAGE,
                    limits.min_storage_buffer_offset_alignment,
                )
            };
            if !Arc::ptr_eq(&buffer.reservation.ledger, &pipeline.owner)
                || !buffer.usage().contains(usage)
            {
                return Err(RuntimeError::with_kind(
                    ErrorKind::InvalidInput,
                    "foreign Metal resource or incorrect buffer usage",
                ));
            }
            if *offset >= buffer.size() || !offset.is_multiple_of(alignment as u64) {
                return Err(resource(
                    "Metal binding offset is outside allocation or unaligned",
                ));
            }
            let read = uniform || read_only[i];
            accesses
                .entry(identity(buffer))
                .and_modify(|old: &mut bool| *old &= read)
                .or_insert(read);
        }
        pipeline.validate_binding_sizes(owners.len(), |i| owners[i].size() - offsets[i])?;
        Ok(Arc::new(Self {
            pipeline,
            owners,
            offsets,
            groups,
            hazards: accesses.into_iter().collect(),
        }))
    }
    pub(super) fn hazards(&self) -> &[(usize, bool)] {
        &self.hazards
    }
    pub(super) fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.owners.capacity() * std::mem::size_of::<GpuBuffer>()
            + self.offsets.capacity() * 8
            + self.hazards.capacity() * std::mem::size_of::<(usize, bool)>()
    }
    fn encode(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        state: &mut EncodingState,
    ) {
        let pipeline = Arc::as_ptr(&self.pipeline) as usize;
        if state.pipeline != Some(pipeline) {
            encoder.setComputePipelineState(&self.pipeline.raw);
            state.pipeline = Some(pipeline);
        }
        if state.buffers.len() < self.owners.len() {
            state.buffers.resize(self.owners.len(), (0, u64::MAX));
        }
        for (i, (buffer, offset)) in self.owners.iter().zip(&self.offsets).enumerate() {
            let binding = (Arc::as_ptr(&buffer.buffer.0) as usize, *offset);
            if state.buffers[i] == binding {
                continue;
            }
            // SAFETY: preparation validates ownership, alignment and ranges; the
            // command's completion handler retains all buffers and this pipeline.
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&buffer.buffer.0.raw), *offset as usize, i);
            }
            state.buffers[i] = binding;
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: self.groups[0] as usize,
                height: self.groups[1] as usize,
                depth: self.groups[2] as usize,
            },
            MTLSize {
                width: self.pipeline.workgroup_size[0] as usize,
                height: 1,
                depth: 1,
            },
        );
    }
}
impl std::fmt::Debug for NativeDispatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalDispatch")
            .field("pipeline", &self.pipeline)
            .field("groups", &self.groups)
            .finish()
    }
}

struct Completion {
    result: Mutex<Option<RuntimeResult<()>>>,
    available: Condvar,
}
pub(super) struct Submission {
    owner: Arc<Device>,
    completion: Arc<Completion>,
    _command: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    encoding_ns: u64,
    submission_ns: u64,
    counters: Option<Arc<Counters>>,
}
impl Submission {
    pub(super) fn host_timings(&self) -> (u64, u64) {
        (self.encoding_ns, self.submission_ns)
    }
    pub(super) fn wait(&mut self) -> RuntimeResult<()> {
        let result = self
            .completion
            .result
            .lock()
            .map_err(|_| RuntimeError::new("Metal completion state poisoned"))?;
        let (result, timeout) = self
            .completion
            .available
            .wait_timeout_while(result, Duration::from_secs(60), |r| r.is_none())
            .map_err(|_| RuntimeError::new("Metal completion state poisoned"))?;
        if timeout.timed_out() && result.is_none() {
            let error = RuntimeError::with_kind(
                ErrorKind::DeviceLost,
                "Metal completion timed out; context retired",
            );
            self.owner.retire(error.clone());
            return Err(error);
        }
        result
            .as_ref()
            .expect("completion condition checked")
            .clone()
    }
    pub(super) fn timestamps(&self) -> RuntimeResult<Vec<(usize, u64)>> {
        let state = self
            .completion
            .result
            .lock()
            .map_err(|_| RuntimeError::new("Metal completion state poisoned"))?;
        state
            .as_ref()
            .ok_or_else(|| RuntimeError::new("Metal timestamps requested before completion"))?
            .clone()?;
        self.counters
            .as_ref()
            .map_or_else(|| Ok(Vec::new()), |counters| counters.resolve())
    }
}

pub(super) fn submit(
    context: &GpuContext,
    mut operations: Vec<(usize, Operation)>,
    profiling: bool,
    prefer_serial: bool,
) -> RuntimeResult<Submission> {
    autoreleasepool(|_| {
        let encoding = profiling.then(Instant::now);
        let owner = &context.backend.owner;
        let _serial = owner
            .serial
            .lock()
            .map_err(|_| RuntimeError::new("Metal queue poisoned"))?;
        context.check_alive()?;
        // Validate the complete transfer graph before creating native encoders.
        for (_, operation) in &operations {
            match operation {
                Operation::Compute { native, .. } => {
                    if !Arc::ptr_eq(&native.pipeline.owner, &context.ledger) {
                        return Err(RuntimeError::with_kind(
                            ErrorKind::InvalidInput,
                            "foreign Metal dispatch",
                        ));
                    }
                }
                Operation::Copy {
                    source,
                    output,
                    source_offset,
                    output_offset,
                    bytes,
                } => {
                    if !Arc::ptr_eq(&source.reservation.ledger, &context.ledger)
                        || !Arc::ptr_eq(&output.reservation.ledger, &context.ledger)
                    {
                        return Err(RuntimeError::with_kind(
                            ErrorKind::InvalidInput,
                            "foreign Metal copy resource",
                        ));
                    }
                    if source_offset
                        .checked_add(*bytes)
                        .is_none_or(|end| end > source.size())
                        || output_offset
                            .checked_add(*bytes)
                            .is_none_or(|end| end > output.size())
                        || !source.usage().contains(BufferUsage::COPY_SRC)
                        || !output.usage().contains(BufferUsage::COPY_DST)
                        || !source_offset.is_multiple_of(4)
                        || !output_offset.is_multiple_of(4)
                        || !bytes.is_multiple_of(4)
                    {
                        return Err(resource("invalid Metal copy ownership, usage or range"));
                    }
                    if source.buffer == output.buffer
                        && *source_offset < output_offset.saturating_add(*bytes)
                        && *output_offset < source_offset.saturating_add(*bytes)
                    {
                        return Err(resource("overlapping Metal buffer copy"));
                    }
                }
            }
        }
        let mut uploads = PendingUploads {
            context: &context.backend,
            entries: std::mem::take(
                &mut *context
                    .backend
                    .uploads
                    .lock()
                    .map_err(|_| RuntimeError::new("Metal upload queue poisoned"))?,
            ),
        };
        uploads.entries.retain_mut(|upload| {
            upload.target_owner = upload.reservation.upgrade();
            upload.target_owner.is_some()
        });
        let command = owner.queue.commandBuffer().ok_or_else(|| {
            RuntimeError::with_kind(ErrorKind::Allocation, "create Metal command buffer")
        })?;
        if !uploads.entries.is_empty() {
            let blit = command
                .blitCommandEncoder()
                .ok_or_else(|| RuntimeError::new("create Metal blit encoder"))?;
            for upload in &uploads.entries {
                // SAFETY: snapshots are immutable, ranges checked at upload, and owners retained through completion.
                unsafe {
                    blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                        &upload.source.0.raw,
                        0,
                        &upload.target.0.raw,
                        upload.offset,
                        upload.bytes,
                    );
                }
            }
            blit.endEncoding();
        }
        operations.sort_by_key(|(depth, _)| *depth);
        let counters = if profiling {
            Counters::new(context, &operations)?
        } else {
            None
        };
        let mut dispatch_index = 0;
        let mut compute: Option<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>> = None;
        let mut previous_depth = None;
        let mut encoding_state = EncodingState::default();
        for (depth, operation) in &operations {
            match operation {
                Operation::Compute { native, .. } => {
                    if let Some(counters) = &counters {
                        let encoder = counters.encoder(&command, dispatch_index)?;
                        native.encode(&encoder, &mut EncodingState::default());
                        encoder.endEncoding();
                        dispatch_index += 1;
                        continue;
                    }
                    if compute.is_none() {
                        compute = command.computeCommandEncoderWithDispatchType(if prefer_serial {
                            MTLDispatchType::Serial
                        } else {
                            MTLDispatchType::Concurrent
                        });
                        if compute.is_none() {
                            return Err(RuntimeError::new("create Metal compute encoder"));
                        }
                        previous_depth = None;
                        encoding_state = EncodingState::default();
                    }
                    let encoder = compute.as_ref().unwrap();
                    if !prefer_serial && previous_depth.is_some_and(|previous| previous != *depth) {
                        encoder.memoryBarrierWithScope(MTLBarrierScope::Buffers);
                    }
                    native.encode(encoder, &mut encoding_state);
                    previous_depth = Some(*depth);
                }
                Operation::Copy {
                    source,
                    output,
                    source_offset,
                    output_offset,
                    bytes,
                } => {
                    if let Some(encoder) = compute.take() {
                        encoder.endEncoding();
                    }
                    if *bytes > 0 {
                        let blit = command
                            .blitCommandEncoder()
                            .ok_or_else(|| RuntimeError::new("create Metal copy encoder"))?;
                        // SAFETY: both ranges were checked; tracked buffers order cross-encoder dependencies.
                        unsafe {
                            blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                                &source.buffer.0.raw,
                                *source_offset as usize,
                                &output.buffer.0.raw,
                                *output_offset as usize,
                                *bytes as usize,
                            );
                        }
                        blit.endEncoding();
                    }
                }
            }
        }
        if let Some(encoder) = compute {
            encoder.endEncoding();
        }
        let completion = Arc::new(Completion {
            result: Mutex::new(None),
            available: Condvar::new(),
        });
        let notified = completion.clone();
        let device = owner.clone();
        let completed_counters = counters.clone();
        // Metal retains this handler until completion. It owns ALL logical quota
        // reservations as well as native handles, even if the waiter times out or drops.
        let retained = Mutex::new(Some((operations, std::mem::take(&mut uploads.entries))));
        let handler = RcBlock::new(
            move |completed: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
                // SAFETY: Metal invokes the handler with the live completed command buffer.
                let completed = unsafe { completed.as_ref() };
                if let Some(counters) = &completed_counters {
                    counters.complete(&device);
                }
                let result = if completed.status() == MTLCommandBufferStatus::Completed {
                    Ok(())
                } else {
                    let native_error = completed.error();
                    let kind = native_error.as_ref().map_or(ErrorKind::Execution, |e| {
                        if e.domain()
                            .isEqualToString(unsafe { MTLCommandBufferErrorDomain })
                            && matches!(e.code(), 2 | 4 | 11)
                        {
                            ErrorKind::DeviceLost
                        } else {
                            ErrorKind::Execution
                        }
                    });
                    let error = RuntimeError::with_kind(
                        kind,
                        format!("Metal command failed: {native_error:?}"),
                    );
                    device.retire(error.clone());
                    Err(error)
                };
                if let Ok(mut resources) = retained.lock() {
                    resources.take();
                }
                if let Ok(mut state) = notified.result.lock() {
                    *state = Some(result);
                }
                notified.available.notify_all();
            },
        );
        // SAFETY: handler captures only synchronized Send/Sync state and lives until completion.
        unsafe {
            command.addCompletedHandler(RcBlock::as_ptr(&handler));
        }
        let encoding_ns = encoding.map_or(0, |start| start.elapsed().as_nanos() as u64);
        let submission = profiling.then(Instant::now);
        command.commit();
        let submission_ns = submission.map_or(0, |start| start.elapsed().as_nanos() as u64);
        Ok(Submission {
            owner: owner.clone(),
            completion,
            _command: command,
            encoding_ns,
            submission_ns,
            counters,
        })
    })
}

fn unsupported(message: impl Into<String>) -> RuntimeError {
    RuntimeError::with_kind(ErrorKind::Unsupported, message)
}

#[cfg(test)]
mod tests;
