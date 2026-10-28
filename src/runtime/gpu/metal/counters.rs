//! Opt-in dispatch timings. One dispatch per sampled pass on Apple GPUs.
use super::*;
use objc2_foundation::NSRange;

// Apple Silicon limits a timestamp sample buffer to 32 KiB. Each dispatch
// needs two 8-byte samples; large OCR waves therefore use multiple buffers.
const DISPATCHES_PER_BUFFER: usize = 2048;

pub(super) struct Counters {
    buffers: Vec<Retained<ProtocolObject<dyn MTLCounterSampleBuffer>>>,
    indices: Vec<usize>,
    start: (u64, u64),
    end: Mutex<Option<(u64, u64)>>,
    _reservation: Arc<Reservation>,
}
// SAFETY: sampling is encoded on one thread before commit. The completion
// callback publishes the end timestamp; resolution is permitted only after
// completion, with immutable metadata and no further GPU writes.
unsafe impl Send for Counters {}
unsafe impl Sync for Counters {}

fn clock(device: &Device) -> (u64, u64) {
    let (mut cpu, mut gpu) = (0, 0);
    // SAFETY: both timestamp pointers are valid, distinct stack values.
    unsafe {
        device
            .raw
            .sampleTimestamps_gpuTimestamp(NonNull::from(&mut cpu), NonNull::from(&mut gpu));
    }
    (cpu, gpu)
}

impl Counters {
    pub(super) fn new(
        context: &GpuContext,
        operations: &[(usize, Operation)],
    ) -> RuntimeResult<Option<Arc<Self>>> {
        let device = &context.backend.owner;
        if !device
            .raw
            .supportsCounterSampling(MTLCounterSamplingPoint::AtStageBoundary)
        {
            return Ok(None);
        }
        let Some(sets) = device.raw.counterSets() else {
            return Ok(None);
        };
        let Some(set) = sets
            .iter()
            .find(|set| set.name().to_string() == "timestamp")
        else {
            return Ok(None);
        };
        let indices: Vec<_> = operations
            .iter()
            .filter_map(|(_, op)| match op {
                Operation::Compute { profile_index, .. } => Some(*profile_index),
                _ => None,
            })
            .collect();
        if indices.is_empty() {
            return Ok(None);
        }
        let count = indices
            .len()
            .checked_mul(2)
            .ok_or_else(|| resource("Metal counter count overflow"))?;
        let reservation = context.ledger.reserve(
            count
                .checked_mul(8)
                .ok_or_else(|| resource("Metal counter size overflow"))?,
        )?;
        let mut buffers = Vec::with_capacity(indices.len().div_ceil(DISPATCHES_PER_BUFFER));
        for chunk in indices.chunks(DISPATCHES_PER_BUFFER) {
            let descriptor = MTLCounterSampleBufferDescriptor::new();
            descriptor.setCounterSet(Some(&set));
            // SAFETY: nonempty chunks have at most 4096 timestamp samples.
            unsafe {
                descriptor.setSampleCount(chunk.len() * 2);
            }
            descriptor.setStorageMode(MTLStorageMode::Shared);
            let buffer = device
                .raw
                .newCounterSampleBufferWithDescriptor_error(&descriptor)
                .map_err(|e| {
                    RuntimeError::with_kind(
                        ErrorKind::GpuValidation,
                        format!("Metal counter buffer: {e}"),
                    )
                })?;
            buffers.push(buffer);
        }
        Ok(Some(Arc::new(Self {
            buffers,
            indices,
            start: clock(device),
            end: Mutex::new(None),
            _reservation: reservation,
        })))
    }
    pub(super) fn encoder(
        &self,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
        dispatch: usize,
    ) -> RuntimeResult<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>> {
        if dispatch >= self.indices.len() {
            return Err(resource("Metal counter index out of range"));
        }
        let descriptor = MTLComputePassDescriptor::new();
        descriptor.setDispatchType(MTLDispatchType::Serial);
        // SAFETY: slot zero exists, and both sample indices fit the checked allocation.
        unsafe {
            let attachment = descriptor
                .sampleBufferAttachments()
                .objectAtIndexedSubscript(0);
            attachment.setSampleBuffer(Some(&self.buffers[dispatch / DISPATCHES_PER_BUFFER]));
            attachment.setStartOfEncoderSampleIndex((dispatch % DISPATCHES_PER_BUFFER) * 2);
            attachment.setEndOfEncoderSampleIndex((dispatch % DISPATCHES_PER_BUFFER) * 2 + 1);
        }
        command
            .computeCommandEncoderWithDescriptor(&descriptor)
            .ok_or_else(|| RuntimeError::new("create sampled Metal pass"))
    }
    pub(super) fn complete(&self, device: &Device) {
        if let Ok(mut end) = self.end.lock() {
            *end = Some(clock(device));
        }
    }
    pub(super) fn resolve(&self) -> RuntimeResult<Vec<(usize, u64)>> {
        let end = self
            .end
            .lock()
            .map_err(|_| RuntimeError::new("Metal counter clock poisoned"))?
            .ok_or_else(|| RuntimeError::new("Metal counters requested before completion"))?;
        let cpu_span = end
            .0
            .checked_sub(self.start.0)
            .filter(|n| *n != 0)
            .ok_or_else(|| resource("invalid Metal CPU clock interval"))?;
        let gpu_span = end
            .1
            .checked_sub(self.start.1)
            .filter(|n| *n != 0)
            .ok_or_else(|| resource("invalid Metal GPU clock interval"))?;
        let mut bytes = Vec::with_capacity(self.indices.len() * 16);
        for (buffer, chunk) in self
            .buffers
            .iter()
            .zip(self.indices.chunks(DISPATCHES_PER_BUFFER))
        {
            // SAFETY: completion publishes the end clock after all GPU writes;
            // the requested range is exactly this buffer's sample count.
            let data = unsafe { buffer.resolveCounterRange(NSRange::new(0, chunk.len() * 2)) }
                .ok_or_else(|| RuntimeError::new("resolve Metal timestamps"))?;
            let resolved = data.to_vec();
            if resolved.len() != chunk.len() * 16 {
                return Err(resource("Metal timestamp count mismatch"));
            }
            bytes.extend_from_slice(&resolved);
        }
        self.indices
            .iter()
            .zip(bytes.as_chunks::<16>().0)
            .map(|(&index, pair)| {
                let begin = u64::from_ne_bytes(pair[..8].try_into().unwrap());
                let finish = u64::from_ne_bytes(pair[8..].try_into().unwrap());
                if begin == 0 || finish == u64::MAX || finish < begin {
                    return Err(resource("invalid Metal timestamp sample"));
                }
                // Apple's sampled CPU clock is in nanoseconds; GPU clocks need
                // calibration from the paired before/after samples.
                let ns = ((finish - begin) as u128 * cpu_span as u128 / gpu_span as u128)
                    .min(u64::MAX as u128) as u64;
                Ok((index, ns))
            })
            .collect()
    }
}
