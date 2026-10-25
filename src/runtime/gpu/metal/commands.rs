use super::buffers::{Allocation, Upload};
use super::*;
use resident::schedule::Operation;
use std::{
    collections::HashSet,
    ptr::NonNull,
    time::{Duration, Instant},
};

struct Recording {
    allocator: Native<dyn MTL4CommandAllocator>,
    command: Native<dyn MTL4CommandBuffer>,
    arguments: Native<dyn MTL4ArgumentTable>,
    residency: Native<dyn MTLResidencySet>,
    allocations: Vec<Arc<Allocation>>,
}
// SAFETY: Recording is moved between exclusive owners. Its native mutable
// objects are never used concurrently and its allocator is reset only after
// the associated timeline completion is observed.
unsafe impl Send for Recording {}
impl Recording {
    fn new(device: &Device) -> RuntimeResult<Self> {
        let allocator = device
            .raw
            .newCommandAllocator()
            .ok_or_else(|| allocation("command allocator"))?;
        let command = device
            .raw
            .newCommandBuffer()
            .ok_or_else(|| allocation("command buffer"))?;
        let desc = MTL4ArgumentTableDescriptor::new();
        desc.setMaxBufferBindCount(31);
        desc.setInitializeBindings(true);
        let arguments = device
            .raw
            .newArgumentTableWithDescriptor_error(&desc)
            .map_err(|e| native_error("argument table", &e))?;
        let residency = device
            .raw
            .newResidencySetWithDescriptor_error(&MTLResidencySetDescriptor::new())
            .map_err(|e| native_error("residency set", &e))?;
        Ok(Self {
            allocator,
            command,
            arguments,
            residency,
            allocations: Vec::new(),
        })
    }
}
struct Resources {
    _operations: Vec<(usize, Operation)>,
    _buffers: Vec<GpuBuffer>,
    _uploads: Vec<Upload>,
}
struct Flight {
    ticket: u64,
    completion: Mutex<Option<RuntimeResult<()>>>,
    ready: Condvar,
    recording: Mutex<Option<Recording>>,
    resources: Mutex<Option<Resources>>,
    counters: Option<Native<dyn MTL4CounterHeap>>,
    profile_indices: Vec<usize>,
}
// SAFETY: The only non-thread-safe native objects are exclusively owned by
// Recording. Counter heaps are immutable after completion, and accessed only
// after the shared event. Resource handles themselves are immutable.
unsafe impl Send for Flight {}
unsafe impl Sync for Flight {}

#[derive(Default)]
pub(super) struct Stream {
    ticket: u64,
    idle: Vec<Recording>,
    flights: Vec<Arc<Flight>>,
}
impl Stream {
    pub(super) fn collect(&mut self, device: &Device) {
        let completed = device.event.signaledValue();
        self.flights.retain(|flight| {
            let ready =
                flight.ticket <= completed && flight.completion.lock().is_ok_and(|v| v.is_some());
            if ready
                && let Ok(mut recording) = flight.recording.lock()
                && let Some(mut recording) = recording.take()
            {
                recording.residency.removeAllAllocations();
                recording.residency.commit();
                recording.allocations.clear();
                recording.allocator.reset();
                if self.idle.len() < 32 {
                    self.idle.push(recording);
                }
            }
            !ready
        });
    }
}

pub(crate) struct Submission {
    device: Arc<Device>,
    flight: Arc<Flight>,
    encoding_ns: u64,
    submission_ns: u64,
}
impl Submission {
    pub(crate) fn host_timings(&self) -> (u64, u64) {
        (self.encoding_ns, self.submission_ns)
    }
    pub(crate) fn wait(&mut self) -> RuntimeResult<()> {
        self.wait_for(Duration::from_secs(60))
    }
    fn wait_for(&mut self, timeout: Duration) -> RuntimeResult<()> {
        self.device.check()?;
        let until = Instant::now() + timeout;
        let mut complete = self
            .flight
            .completion
            .lock()
            .map_err(|_| RuntimeError::new("Metal completion poisoned"))?;
        while complete.is_none() {
            let remaining = until.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(self.device.fail(RuntimeError::with_kind(
                    ErrorKind::DeviceLost,
                    "Metal completion timed out; resources remain retained",
                )));
            }
            complete = self
                .flight
                .ready
                .wait_timeout(complete, remaining)
                .map_err(|_| RuntimeError::new("Metal completion poisoned"))?
                .0;
        }
        complete.as_ref().unwrap().clone()?;
        drop(complete);
        if !self.device.event.waitUntilSignaledValue_timeoutMS(
            self.flight.ticket,
            until.saturating_duration_since(Instant::now()).as_millis() as u64,
        ) {
            return Err(self.device.fail(RuntimeError::with_kind(
                ErrorKind::DeviceLost,
                "Metal completion event timed out",
            )));
        }
        self.device.check()?;
        if let Ok(mut stream) = self.device.stream.lock() {
            stream.collect(&self.device);
        }
        Ok(())
    }
    pub(crate) fn timestamps(&self) -> RuntimeResult<Vec<(usize, u64)>> {
        let Some(counters) = &self.flight.counters else {
            return Ok(Vec::new());
        };
        if self.device.event.signaledValue() < self.flight.ticket {
            return Err(resource("Metal counters still in flight"));
        }
        let count = self.flight.profile_indices.len() * 2;
        // SAFETY: The range was allocated for exactly these timestamp pairs;
        // the shared-event value proves all GPU writes have completed.
        let data = unsafe { counters.resolveCounterRange(NSRange::new(0, count)) }
            .ok_or_else(|| RuntimeError::new("Metal timestamp resolve failed"))?;
        // SAFETY: The immutable NSData stays retained through the complete
        // decode below; no pointer escapes or mutable reference aliases it.
        let bytes = unsafe { data.as_bytes_unchecked() };
        if bytes.len() != count * 8 {
            return Err(RuntimeError::new("Metal timestamp result size differs"));
        }
        let frequency = self.device.raw.queryTimestampFrequency();
        if frequency == 0 {
            return Err(RuntimeError::new("Metal timestamp frequency is zero"));
        }
        Ok(self
            .flight
            .profile_indices
            .iter()
            .zip(bytes.as_chunks::<16>().0.iter())
            .map(|(&index, pair)| {
                let start = u64::from_ne_bytes(pair[..8].try_into().unwrap());
                let end = u64::from_ne_bytes(pair[8..].try_into().unwrap());
                (
                    index,
                    (u128::from(end.saturating_sub(start)) * 1_000_000_000 / u128::from(frequency))
                        .min(u128::from(u64::MAX)) as u64,
                )
            })
            .collect())
    }
}
impl Drop for Submission {
    fn drop(&mut self) {
        let _ = self.wait();
    }
}

pub(crate) fn submit(
    context: &GpuContext,
    operations: Vec<(usize, Operation)>,
    profiling: bool,
    _prefer_serial: bool,
) -> RuntimeResult<Submission> {
    objc2::rc::autoreleasepool(|_| submit_inner(context, operations, profiling))
}
fn submit_inner(
    context: &GpuContext,
    mut operations: Vec<(usize, Operation)>,
    profiling: bool,
) -> RuntimeResult<Submission> {
    let timer = profiling.then(Instant::now);
    context.check_alive()?;
    let device = &context.backend.owner;
    operations.sort_by_key(|(depth, _)| *depth);
    let mut identities = HashSet::new();
    let mut buffers = Vec::new();
    for (_, operation) in &operations {
        let operands: &[GpuBuffer] = match operation {
            Operation::Compute { native, .. } => native.owners(),
            Operation::Copy {
                source,
                output,
                source_offset,
                output_offset,
                bytes,
            } => {
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
                    return Err(resource("invalid Metal copy range or usage"));
                }
                if source.buffer == output.buffer
                    && *bytes != 0
                    && *source_offset < output_offset + bytes
                    && *output_offset < source_offset + bytes
                {
                    return Err(resource("overlapping Metal copy"));
                }
                for buffer in [source, output] {
                    if identities.insert(resident::schedule::identity(buffer)) {
                        buffers.push(buffer.clone());
                    }
                }
                &[]
            }
        };
        for buffer in operands {
            if identities.insert(resident::schedule::identity(buffer)) {
                buffers.push(buffer.clone());
            }
        }
    }
    if buffers
        .iter()
        .any(|b| !Arc::ptr_eq(&b.reservation.ledger, &context.ledger))
    {
        return Err(RuntimeError::with_kind(
            ErrorKind::InvalidInput,
            "foreign Metal submission resource",
        ));
    }
    // Acquire allocation gates in a stable order. Initialized read-only weights
    // need no gate, so independent sessions can encode concurrently. A transfer
    // or mutation owns its allocation until publication in the queue order.
    let mut gated = HashSet::new();
    for (_, operation) in &operations {
        match operation {
            Operation::Compute { native, .. } => {
                for &(identity, read_only) in native.hazards() {
                    if !read_only {
                        gated.insert(identity);
                    }
                }
            }
            Operation::Copy { output, .. } => {
                gated.insert(resident::schedule::identity(output));
            }
        }
    }
    for buffer in &buffers {
        let state = buffer
            .buffer
            .storage
            .contents
            .lock()
            .map_err(|_| RuntimeError::new("Metal buffer state poisoned"))?;
        if !state.initialized || !state.uploads.is_empty() {
            gated.insert(resident::schedule::identity(buffer));
        }
    }
    let mut ordered: Vec<_> = buffers
        .iter()
        .filter(|b| gated.contains(&resident::schedule::identity(b)))
        .collect();
    ordered.sort_by_key(|b| resident::schedule::identity(b));
    let _guards = ordered
        .iter()
        .map(|b| {
            b.buffer
                .storage
                .recording
                .lock()
                .map_err(|_| RuntimeError::new("Metal allocation recording poisoned"))
        })
        .collect::<RuntimeResult<Vec<_>>>()?;
    let mut stream = device
        .stream
        .lock()
        .map_err(|_| RuntimeError::new("Metal stream poisoned"))?;
    device.check()?;
    stream.collect(device);
    let mut recording = match stream.idle.pop() {
        Some(r) => r,
        None => Recording::new(device)?,
    };
    drop(stream);
    let mut snapshots = Vec::new();
    let mut uploads = Vec::new();
    for buffer in &buffers {
        recording
            .residency
            .addAllocation(ProtocolObject::from_ref(buffer.buffer.raw()));
        recording
            .allocations
            .push(buffer.buffer.storage.allocation.clone());
        // Read-only allocations inspected as initialized before gating do not
        // consume later uploads. Those belong to a subsequent publication;
        // consuming them without the allocation gate could duplicate a transfer.
        if !gated.contains(&resident::schedule::identity(buffer)) {
            snapshots.push((true, 0));
            continue;
        }
        let state = buffer
            .buffer
            .storage
            .contents
            .lock()
            .map_err(|_| RuntimeError::new("Metal buffer state poisoned"))?;
        snapshots.push((state.initialized, state.uploads.len()));
        for upload in &state.uploads {
            recording
                .residency
                .addAllocation(ProtocolObject::from_ref(&*upload.page.raw));
            recording.allocations.push(upload.page.clone());
            uploads.push(upload.clone());
        }
    }
    for (_, operation) in &operations {
        if let Operation::Compute { native, .. } = operation {
            recording
                .residency
                .addAllocation(ProtocolObject::from_ref(&*native.pipeline.raw));
        }
    }
    let profile_indices: Vec<_> = operations
        .iter()
        .filter_map(|(_, op)| match op {
            Operation::Compute { profile_index, .. } if profiling => Some(*profile_index),
            _ => None,
        })
        .collect();
    let counters = if profile_indices.is_empty() {
        None
    } else {
        let descriptor = MTL4CounterHeapDescriptor::new();
        descriptor.setType(MTL4CounterHeapType::Timestamp);
        // SAFETY: Each entry reserves exactly two timestamp indices. The Vec's
        // allocation bound also proves this usize multiplication cannot wrap.
        unsafe {
            descriptor.setCount(profile_indices.len() * 2);
        }
        Some(
            device
                .raw
                .newCounterHeapWithDescriptor_error(&descriptor)
                .map_err(|e| native_error("counter heap", &e))?,
        )
    };
    recording.residency.commit();
    recording
        .command
        .beginCommandBufferWithAllocator(&recording.allocator);
    recording.command.useResidencySet(&recording.residency);
    let encoder = recording
        .command
        .computeCommandEncoder()
        .ok_or_else(|| allocation("compute encoder"))?;
    let all = MTLStages::Dispatch | MTLStages::Blit;
    encoder.barrierAfterQueueStages_beforeStages_visibilityOptions(
        all,
        all,
        MTL4VisibilityOptions::Device,
    );
    encoder.setArgumentTable(Some(&recording.arguments));
    let mut upload_index = 0;
    for (buffer, &(initialized, count)) in buffers.iter().zip(&snapshots) {
        if !initialized {
            // SAFETY: The logical range is within the retained allocation, and
            // its exclusive recording gate orders this clear before all users.
            unsafe {
                encoder.fillBuffer_range_value(
                    buffer.buffer.raw(),
                    NSRange::new(0, buffer.size() as usize),
                    0,
                );
            }
        }
        // Multiple writes to one allocation are ordered, including its clear.
        for upload in &uploads[upload_index..upload_index + count] {
            encoder.barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
                MTLStages::Blit,
                MTLStages::Blit,
                MTL4VisibilityOptions::Device,
            );
            // SAFETY: Upload creation checked both ranges. Their reservations
            // and pages remain retained in this submission until completion.
            unsafe {
                encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    &upload.page.raw,
                    upload.source_offset,
                    buffer.buffer.raw(),
                    upload.destination_offset,
                    upload.bytes,
                );
            }
        }
        upload_index += count;
    }
    encoder.barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
        MTLStages::Blit,
        all,
        MTL4VisibilityOptions::Device,
    );
    let mut previous = None;
    let mut sample = 0;
    for (depth, operation) in &operations {
        if previous.is_some_and(|old| old != *depth) {
            encoder.barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
                all,
                all,
                MTL4VisibilityOptions::Device,
            );
        }
        previous = Some(*depth);
        match operation {
            Operation::Compute { native, .. } => {
                encoder.setComputePipelineState(&native.pipeline.raw);
                for (i, (buffer, &offset)) in native.buffers.iter().zip(&native.offsets).enumerate()
                {
                    // SAFETY: Prepared bindings validate addresses, offsets,
                    // and slot counts; residency and Flight retain every owner.
                    unsafe {
                        recording
                            .arguments
                            .setAddress_atIndex(buffer.buffer.raw().gpuAddress() + offset, i);
                    }
                }
                if let Some(heap) = &counters {
                    // SAFETY: sample is the dispatch ordinal times two, within
                    // the heap's two-entry reservation for this dispatch.
                    unsafe {
                        encoder.writeTimestampWithGranularity_intoHeap_atIndex(
                            MTL4TimestampGranularity::Precise,
                            heap,
                            sample,
                        );
                    }
                }
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    size(native.groups),
                    size(native.pipeline.workgroup_size),
                );
                if let Some(heap) = &counters {
                    // SAFETY: The matching end sample occupies the second slot
                    // of the same reserved timestamp pair.
                    unsafe {
                        encoder.writeTimestampWithGranularity_intoHeap_atIndex(
                            MTL4TimestampGranularity::Precise,
                            heap,
                            sample + 1,
                        );
                    }
                }
                sample += 2;
            }
            Operation::Copy {
                source,
                output,
                source_offset,
                output_offset,
                bytes,
                // SAFETY: Submission validation checked both copy ranges, usages,
                // alignment and overlap; residency and Flight retain both buffers.
            } => unsafe {
                encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    source.buffer.raw(),
                    *source_offset as usize,
                    output.buffer.raw(),
                    *output_offset as usize,
                    *bytes as usize,
                );
            },
        }
    }
    encoder.endEncoding();
    recording.command.endCommandBuffer();
    let command = recording.command.clone();
    // Only publication and completion-value assignment are serialized. No
    // mutable encoder or argument table is shared between active executions.
    let mut stream = device
        .stream
        .lock()
        .map_err(|_| RuntimeError::new("Metal stream poisoned"))?;
    device.check()?;
    let ticket = stream
        .ticket
        .checked_add(1)
        .ok_or_else(|| resource("Metal timeline exhausted"))?;
    let flight = Arc::new(Flight {
        ticket,
        completion: Mutex::new(None),
        ready: Condvar::new(),
        recording: Mutex::new(Some(recording)),
        resources: Mutex::new(Some(Resources {
            _operations: operations,
            _buffers: buffers.clone(),
            _uploads: uploads,
        })),
        counters,
        profile_indices,
    });
    let callback_flight = flight.clone();
    let failure = device.failure.clone();
    let available = device.available.clone();
    let callback = block2::RcBlock::new(
        move |feedback: NonNull<ProtocolObject<dyn MTL4CommitFeedback>>| {
            // SAFETY: Metal supplies a live feedback object for this callback.
            let result = unsafe { feedback.as_ref() }
                .error()
                .map_or(Ok(()), |e| Err(native_error("GPU execution", &e)));
            if let Err(error) = &result {
                if let Ok(mut state) = failure.lock() {
                    state.get_or_insert_with(|| error.clone());
                }
                available.notify_all();
            }
            // Feedback is delivered after the workload completes, even on errors.
            // This breaks resource/device ownership cycles without CPU polling.
            if let Ok(mut resources) = callback_flight.resources.lock() {
                resources.take();
            }
            if let Ok(mut completion) = callback_flight.completion.lock() {
                *completion = Some(result);
            }
            callback_flight.ready.notify_all();
        },
    );
    let options = MTL4CommitOptions::new();
    // SAFETY: Metal copies the block. Its captures own all callback state and
    // remain thread-safe until Metal delivers completion feedback.
    unsafe {
        options.addFeedbackHandler(&*callback as *const _ as *mut _);
    }
    let encoding_ns = timer.map_or(0, |t| t.elapsed().as_nanos() as u64);
    let submitting = profiling.then(Instant::now);
    let mut pointer = NonNull::from(&*command);
    // SAFETY: This one-element pointer array and its command are live for the
    // synchronous commit call. Flight retains the allocator and all resources.
    unsafe {
        device
            .queue
            .commit_count_options(NonNull::from(&mut pointer), 1, &options);
    }
    device
        .queue
        .signalEvent_value(ProtocolObject::from_ref(&*device.event), ticket);
    stream.ticket = ticket;
    stream.flights.push(flight.clone());
    for (buffer, &(_, count)) in buffers.iter().zip(&snapshots) {
        buffer
            .buffer
            .storage
            .last_use
            .store(ticket, Ordering::Release);
        let mut state = buffer.buffer.storage.contents.lock().map_err(|_| {
            device.fail(RuntimeError::with_kind(
                ErrorKind::DeviceLost,
                "Metal submitted buffer state poisoned",
            ))
        })?;
        state.initialized = true;
        state.uploads.drain(..count);
    }
    let submission_ns = submitting.map_or(0, |t| t.elapsed().as_nanos() as u64);
    Ok(Submission {
        device: device.clone(),
        flight,
        encoding_ns,
        submission_ns,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires physical Metal allocations; injects host-only timeout without submitting GPU work"]
    fn native_timeout_retires_without_releasing_pending_reservations() {
        let context = GpuContext::new().unwrap();
        let buffer = context
            .create_buffer("retirement reservation", 64, BufferUsage::STORAGE)
            .unwrap();
        let reservation = Arc::downgrade(&buffer.reservation);
        let flight = Arc::new(Flight {
            ticket: 1,
            completion: Mutex::new(None),
            ready: Condvar::new(),
            recording: Mutex::new(None),
            resources: Mutex::new(Some(Resources {
                _operations: Vec::new(),
                _buffers: vec![buffer],
                _uploads: Vec::new(),
            })),
            counters: None,
            profile_indices: Vec::new(),
        });
        // No work is submitted: this exercises exactly the timeout branch with
        // an absent completion notification, without risking a GPU hang.
        let mut submission = Submission {
            device: context.backend.owner.clone(),
            flight: flight.clone(),
            encoding_ns: 0,
            submission_ns: 0,
        };
        assert_eq!(
            submission.wait_for(Duration::ZERO).unwrap_err().kind(),
            ErrorKind::DeviceLost
        );
        drop(submission);
        assert!(reservation.upgrade().is_some());
        assert!(context.check_alive().is_err());
        // The synthetic flight has no device use; explicitly model completion.
        flight.resources.lock().unwrap().take();
        assert!(reservation.upgrade().is_none());
    }
}
