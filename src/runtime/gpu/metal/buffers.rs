use super::*;

pub(super) struct Allocation {
    pub raw: Native<dyn MTLBuffer>,
    accounting: Arc<Accounting>,
    bytes: usize,
}
// SAFETY: The handle is immutable. Mapped writes are confined to exclusive,
// disjoint upload ranges; GPU access is ordered by the command stream.
unsafe impl Send for Allocation {}
unsafe impl Sync for Allocation {}
impl Drop for Allocation {
    fn drop(&mut self) {
        self.accounting
            .committed
            .fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

impl Allocation {
    fn new(device: &Device, label: &str, bytes: usize, shared: bool) -> RuntimeResult<Arc<Self>> {
        let mode = if shared {
            MTLResourceOptions::StorageModeShared
        } else {
            MTLResourceOptions::StorageModePrivate
        };
        let raw = device
            .raw
            .newBufferWithLength_options(
                bytes,
                mode | MTLResourceOptions::HazardTrackingModeUntracked,
            )
            .ok_or_else(|| allocation(label))?;
        raw.setLabel(Some(&NSString::from_str(label)));
        let allocated = MTLResource::allocatedSize(&*raw);
        let committed = device
            .accounting
            .committed
            .fetch_add(allocated, Ordering::AcqRel)
            + allocated;
        device
            .accounting
            .peak
            .fetch_max(committed, Ordering::AcqRel);
        if shared {
            // SAFETY: This unpublished shared allocation has no GPU or CPU users.
            unsafe {
                raw.contents().cast::<u8>().as_ptr().write_bytes(0, bytes);
            }
        }
        Ok(Arc::new(Self {
            raw,
            accounting: device.accounting.clone(),
            bytes: allocated,
        }))
    }
}

#[derive(Clone)]
pub(super) struct Upload {
    pub page: Arc<Allocation>,
    pub source_offset: usize,
    pub destination_offset: usize,
    pub bytes: usize,
    pub _reservation: Arc<Reservation>,
}
pub(super) struct Contents {
    pub initialized: bool,
    pub uploads: Vec<Upload>,
}
pub(super) struct Storage {
    pub allocation: Arc<Allocation>,
    pub contents: Mutex<Contents>,
    pub last_use: AtomicU64,
    pub recording: Mutex<()>,
    logical_size: u64,
}
#[derive(Clone)]
pub(crate) struct Buffer {
    pub(super) storage: Arc<Storage>,
    pub(super) owner: Arc<Device>,
    usage: BufferUsage,
}
impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalBuffer")
            .field("bytes", &self.size())
            .field("usage", &self.usage)
            .finish()
    }
}
impl PartialEq for Buffer {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.storage, &other.storage)
    }
}
impl Eq for Buffer {}
impl Buffer {
    pub(crate) fn size(&self) -> u64 {
        self.storage.logical_size
    }
    pub(crate) fn usage(&self) -> BufferUsage {
        self.usage
    }
    pub(super) fn raw(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.storage.allocation.raw
    }
    pub(crate) fn read<T>(
        &self,
        bytes: usize,
        consume: impl FnOnce(&[u8]) -> RuntimeResult<T>,
    ) -> RuntimeResult<T> {
        if !self.usage.contains(BufferUsage::MAP_READ) || bytes as u64 > self.size() {
            return Err(resource("Metal readback range or storage mode is invalid"));
        }
        self.owner.check()?;
        // Hold the stream only while copying, never across a user callback.
        let guard = self
            .owner
            .stream
            .lock()
            .map_err(|_| RuntimeError::new("Metal stream poisoned"))?;
        if self.owner.event.signaledValue() < self.storage.last_use.load(Ordering::Acquire) {
            return Err(RuntimeError::with_kind(
                ErrorKind::GpuMapping,
                "Metal readback is still in flight",
            ));
        }
        // SAFETY: MAP_READ allocations use shared storage. Completion and the
        // stream lock exclude GPU writes throughout this bounded CPU copy.
        let result = unsafe {
            std::slice::from_raw_parts(self.raw().contents().cast::<u8>().as_ptr(), bytes)
        }
        .to_vec();
        drop(guard);
        consume(&result)
    }
}
pub(super) fn allocate(
    owner: Arc<Device>,
    label: &str,
    size: u64,
    usage: BufferUsage,
) -> RuntimeResult<Buffer> {
    owner.check()?;
    if size == 0 || size > owner.limits.max_buffer_size || size > isize::MAX as u64 {
        return Err(resource("Metal allocation size is invalid"));
    }
    let shared = usage.contains(BufferUsage::MAP_READ) || usage.contains(BufferUsage::UNIFORM);
    let allocation = if usage.contains(BufferUsage::MAP_READ) {
        owner
            .readbacks
            .lock()
            .map_err(|_| RuntimeError::new("Metal readback pool poisoned"))?
            .allocate(&owner, label, size as usize)?
    } else {
        Allocation::new(&owner, label, size as usize, shared)?
    };
    Ok(Buffer {
        storage: Arc::new(Storage {
            allocation,
            contents: Mutex::new(Contents {
                initialized: shared,
                uploads: Vec::new(),
            }),
            last_use: AtomicU64::new(0),
            recording: Mutex::new(()),
            logical_size: size,
        }),
        owner,
        usage,
    })
}

#[derive(Default)]
pub(super) struct UploadPool {
    pages: Vec<(Arc<Allocation>, usize)>,
}
impl UploadPool {
    pub(super) fn trim(&mut self) {
        let mut idle = 0;
        self.pages.retain(|(page, _)| {
            if Arc::strong_count(page) > 1 {
                return true;
            }
            idle += page.raw.length();
            idle <= 4 * 1024 * 1024
        });
    }

    fn write(&mut self, device: &Device, bytes: &[u8]) -> RuntimeResult<(Arc<Allocation>, usize)> {
        let length = bytes.len().div_ceil(16) * 16;
        let mut chosen = None;
        for (i, (page, cursor)) in self.pages.iter_mut().enumerate() {
            if Arc::strong_count(page) == 1 {
                *cursor = 0;
            }
            if page.raw.length() - *cursor >= length {
                chosen = Some(i);
                break;
            }
        }
        let index = match chosen {
            Some(index) => index,
            None => {
                let page = Allocation::new(device, "Metal upload arena", length.max(65536), true)?;
                self.pages.push((page, 0));
                self.pages.len() - 1
            }
        };
        let (page, cursor) = &mut self.pages[index];
        let offset = *cursor;
        *cursor += length;
        // SAFETY: The pool mutex protects cursor assignment. The reserved range
        // is disjoint from all outstanding uploads and within shared storage.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                page.raw.contents().cast::<u8>().as_ptr().add(offset),
                bytes.len(),
            );
        }
        Ok((page.clone(), offset))
    }
}
pub(super) fn upload(
    device: &Device,
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
        return Err(resource("Metal upload range is invalid or unaligned"));
    }
    if bytes.is_empty() {
        return Ok(());
    }
    let reservation = target.reservation.ledger.reserve(bytes.len())?;
    let (page, source_offset) = device
        .uploads
        .lock()
        .map_err(|_| RuntimeError::new("Metal upload pool poisoned"))?
        .write(device, bytes)?;
    target
        .buffer
        .storage
        .contents
        .lock()
        .map_err(|_| RuntimeError::new("Metal buffer state poisoned"))?
        .uploads
        .push(Upload {
            page,
            source_offset,
            destination_offset: offset as usize,
            bytes: bytes.len(),
            _reservation: reservation,
        });
    Ok(())
}

/// Whole-page readback leases. Reuse requires exclusive pool ownership, which
/// excludes both CPU readers and every residency set still visible to the GPU.
#[derive(Default)]
pub(super) struct ReadbackPool {
    pages: Vec<Arc<Allocation>>,
}
impl ReadbackPool {
    pub(super) fn trim(&mut self) {
        let mut idle = 0;
        self.pages.retain(|page| {
            if Arc::strong_count(page) > 1 {
                return true;
            }
            idle += page.raw.length();
            idle <= 16 * 1024 * 1024
        });
    }

    fn allocate(
        &mut self,
        device: &Device,
        label: &str,
        bytes: usize,
    ) -> RuntimeResult<Arc<Allocation>> {
        if let Some(page) = self
            .pages
            .iter()
            .filter(|p| Arc::strong_count(p) == 1 && p.raw.length() >= bytes)
            .min_by_key(|p| p.raw.length())
        {
            return Ok(page.clone());
        }
        // Bound unused capacity independently from live logical quotas.
        let mut idle = 0;
        self.pages.retain(|page| {
            if Arc::strong_count(page) > 1 {
                return true;
            }
            idle += page.raw.length();
            idle <= 4 * 1024 * 1024
        });
        let page = Allocation::new(device, label, bytes.max(65536), true)?;
        self.pages.push(page.clone());
        Ok(page)
    }
}
