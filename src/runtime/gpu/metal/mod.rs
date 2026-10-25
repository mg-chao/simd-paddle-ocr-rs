//! Apple Silicon execution. All native objects and unsafe API calls live here.
use super::*;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSRange, NSString};
use objc2_metal::*;
use std::sync::atomic::AtomicU64;

mod buffers;
mod commands;
pub(crate) mod execution;
pub(crate) mod pipelines;
mod shaders;
pub(crate) use buffers::Buffer;
pub(crate) use commands::{Submission, submit};

type Native<T> = Retained<ProtocolObject<T>>;

#[derive(Default)]
struct Accounting {
    committed: AtomicUsize,
    peak: AtomicUsize,
}

pub(crate) struct Device {
    raw: Native<dyn MTLDevice>,
    compiler: Native<dyn MTL4Compiler>,
    queue: Native<dyn MTL4CommandQueue>,
    event: Native<dyn MTLSharedEvent>,
    accounting: Arc<Accounting>,
    failure: Arc<Mutex<Option<RuntimeError>>>,
    available: Arc<Condvar>,
    stream: Mutex<commands::Stream>,
    uploads: Mutex<buffers::UploadPool>,
    readbacks: Mutex<buffers::ReadbackPool>,
    limits: GpuLimits,
}
// SAFETY: Device/queue/compiler support concurrent use. Event operations are
// atomic Metal timeline operations. Command state and mapped page writes are
// exclusively accessed under their mutexes; no encoder is shared by threads.
unsafe impl Send for Device {}
unsafe impl Sync for Device {}

impl Device {
    pub(crate) fn committed_bytes(&self) -> usize {
        self.accounting.committed.load(Ordering::Acquire)
    }
    pub(crate) fn peak_committed_bytes(&self) -> usize {
        self.accounting.peak.load(Ordering::Acquire)
    }
    fn check(&self) -> RuntimeResult<()> {
        self.failure
            .lock()
            .map_err(|_| RuntimeError::new("Metal error state poisoned"))?
            .clone()
            .map_or(Ok(()), Err)
    }
    fn fail(&self, error: RuntimeError) -> RuntimeError {
        if let Ok(mut failure) = self.failure.lock() {
            failure.get_or_insert_with(|| error.clone());
        }
        self.available.notify_all();
        error
    }
}

pub(crate) struct Context {
    pub(crate) owner: Arc<Device>,
}
impl Context {
    pub(crate) fn new(
        failure: Arc<Mutex<Option<RuntimeError>>>,
        available: Arc<Condvar>,
    ) -> RuntimeResult<(Self, GpuAdapterInfo, GpuLimits)> {
        objc2::rc::autoreleasepool(|_| {
            if !cfg!(target_arch = "aarch64") || !objc2::available!(macos = 26.0) {
                return Err(unsupported(
                    "Metal execution requires Apple Silicon and macOS 26+",
                ));
            }
            let raw =
                MTLCreateSystemDefaultDevice().ok_or_else(|| unsupported("no Metal device"))?;
            if !raw.hasUnifiedMemory() || !raw.supportsFamily(MTLGPUFamily::Metal4) {
                return Err(unsupported(
                    "the device does not support Apple Silicon Metal 4",
                ));
            }
            let queue = raw
                .newMTL4CommandQueue()
                .ok_or_else(|| allocation("command queue"))?;
            let event = raw
                .newSharedEvent()
                .ok_or_else(|| allocation("completion event"))?;
            let compiler = raw
                .newCompilerWithDescriptor_error(&MTL4CompilerDescriptor::new())
                .map_err(|e| native_error("compiler", &e))?;
            let threads = raw.maxThreadsPerThreadgroup();
            let limits = GpuLimits {
                max_buffer_size: raw.maxBufferLength() as u64,
                max_storage_buffer_binding_size: raw.maxBufferLength() as u64,
                max_uniform_buffer_binding_size: 65536,
                // The shared planner flattens dispatch indices into U32. This
                // is a planner bound, not an artificial resource-size limit.
                max_compute_workgroups_per_dimension: 65535,
                max_compute_workgroup_storage_size: raw.maxThreadgroupMemoryLength() as u32,
                max_compute_invocations_per_workgroup: threads.width as u32,
                max_compute_workgroup_size_x: threads.width as u32,
                max_compute_workgroup_size_y: threads.height as u32,
                max_compute_workgroup_size_z: threads.depth as u32,
                min_uniform_buffer_offset_alignment: 16,
                min_storage_buffer_offset_alignment: 4,
            };
            let info = GpuAdapterInfo {
                name: raw.name().to_string(),
                vendor: 0x106b,
                device: 0,
                device_type: GpuDeviceType::IntegratedGpu,
                backend: GpuApi::Metal,
                driver: "Metal 4".into(),
                driver_info: String::new(),
                api_version: 4,
                subgroup_min_size: 32,
                subgroup_max_size: 32,
            };
            let owner = Arc::new(Device {
                raw,
                compiler,
                queue,
                event,
                accounting: Arc::new(Accounting::default()),
                failure,
                available,
                stream: Mutex::new(commands::Stream::default()),
                uploads: Mutex::new(buffers::UploadPool::default()),
                readbacks: Mutex::new(buffers::ReadbackPool::default()),
                limits: limits.clone(),
            });
            Ok((Self { owner }, info, limits))
        })
    }
    pub(crate) fn collect_garbage(&self) {
        if let Ok(mut stream) = self.owner.stream.lock() {
            stream.collect(&self.owner);
        }
        if let Ok(mut uploads) = self.owner.uploads.lock() {
            uploads.trim();
        }
        if let Ok(mut readbacks) = self.owner.readbacks.lock() {
            readbacks.trim();
        }
    }
    pub(crate) fn supports_half(&self) -> bool {
        true
    }
    pub(crate) fn allocate(
        &self,
        label: &str,
        size: u64,
        usage: BufferUsage,
    ) -> RuntimeResult<Buffer> {
        buffers::allocate(self.owner.clone(), label, size, usage)
    }
    pub(crate) fn upload(
        &self,
        target: &GpuBuffer,
        offset: u64,
        bytes: &[u8],
    ) -> RuntimeResult<()> {
        buffers::upload(&self.owner, target, offset, bytes)
    }
    pub(crate) fn checked<T>(
        &self,
        stage: &str,
        operation: impl FnOnce() -> RuntimeResult<T>,
    ) -> RuntimeResult<T> {
        self.owner.check()?;
        operation().map_err(|e| e.context(stage))
    }
}

pub(crate) fn compile_resident(
    context: &GpuContext,
    key: KernelKey,
    read_only: &[bool],
) -> RuntimeResult<Arc<execution::NativePipeline>> {
    shaders::resident(context, key, read_only)
}
fn unsupported(message: impl Into<String>) -> RuntimeError {
    RuntimeError::with_kind(ErrorKind::Unsupported, message)
}
fn allocation(what: &str) -> RuntimeError {
    RuntimeError::with_kind(
        ErrorKind::Allocation,
        format!("Metal {what} allocation failed"),
    )
}
fn native_error(stage: &str, error: &objc2_foundation::NSError) -> RuntimeError {
    RuntimeError::with_kind(ErrorKind::Execution, format!("Metal {stage}: {error}"))
}
fn size(v: [u32; 3]) -> MTLSize {
    MTLSize {
        width: v[0] as usize,
        height: v[1] as usize,
        depth: v[2] as usize,
    }
}

#[cfg(test)]
mod tests;
