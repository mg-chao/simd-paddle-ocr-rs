# Metal 4 ownership and execution

The shared graph compiler owns tensor shapes, precision policy, activation slot
reuse, physical packing, OCR scheduling, and resource quotas. The macOS module
`src/runtime/gpu/metal` owns all Objective-C objects and unsafe native calls.
It requires Apple Silicon, unified memory, Metal 4, and macOS 26+.

## Immutable interfaces

`ShaderKind`, `KernelKey`, and storage signatures identify kernels independently
of native object addresses. The catalog specifies the entry point, parameter
size, access modes, minimum binding sizes, and threadgroup geometry. Shared graph
lowering checks shape products, tensor extents, and packed storage before creating
bindings. Native binding preparation checks context ownership, usage, offsets,
minimum ranges, and grid limits. Prepared dispatches retain their pipelines,
buffer owners, offsets, and allocation access identities.

Neural parameter packets contain 128 `u32` words (512 bytes); resident OCR packets
contain 256 words (1024 bytes). Rust serializes those layouts explicitly. MSL
uses matching arrays and integer indexing. Storage types are compile-time
parameters, while native function constants select intermediate rounding.
Spatial MPP descriptors specialize input dimensions; staged convolution
specializes channels only. The pipeline cache is bounded to 512 entries.

Compilation uses `MTL4Compiler` with MSL 4 and safe arithmetic. F32 storage uses
precise floating-point functions; half storage preserves its native fast
transcendental-function policy before explicit IEEE rounding. Accumulation stays
F32 in both modes. A cached pipeline retains its library. Shape capability
admission precedes changes to tensor layouts. A compiler defect is an error,
not a request to select another execution backend. MPP uses explicit F32
accumulation; half operands require the explicit F16 matrix policy.

## Memory and CPU pointer invariants

Model weights and activation slots are private Metal buffers. Upload pages,
parameter buffers and readback buffers use shared storage. Allocation accounting
records native allocated capacity independently of the logical quota ledger.
The upload pool assigns aligned, disjoint ranges under its mutex. CPU writes
occur only within those reserved ranges. A page can be reused only when no
pending transfer or recording retains it. Parameter updates stage immutable
snapshots; publishing another update never mutates an in-flight snapshot.
Readback pages use exclusive leases and are reused only after all recording and
reader references release them. Buffer views expose their requested logical
size, while capacity accounting includes the full pooled allocation. Idle upload
and readback capacity is bounded separately.

Raw CPU pointers are confined to allocation initialization, upload copies,
completed readback copies, and timestamp resolution. Each access checks the
allocation range and storage mode. Readback copies hold the publication lock,
verify the allocation's completion value, and release the lock before invoking
any caller callback.

## Recording and visibility

Each active recording exclusively owns an `MTL4CommandAllocator`, command buffer,
argument table, and residency set. Allocation gates are acquired in a stable
identity order for transfers and mutations. Initialized immutable inputs need no
gate, permitting concurrent encoding with shared model weights. The queue lock
covers only recycling and publication, including completion-value assignment.
No mutable encoder or argument table is shared between active recordings.

Uploads, clears, compute dispatches and readback copies share one compute encoder.
The shared schedule derives dependency depths from allocation identities and
read/write accesses, including aliases and copies. Dispatch/blit stage barriers
with device visibility separate dependent depths. A queue-stage barrier imports
prior submissions' writes. Transfer barriers order clears and overlapping upload
ranges before computation. This does not rely on automatic hazard tracking.

Every submission installs explicit resource residency and retains buffer owners,
upload reservations, pipelines, recording resources and optional counter heaps.
Argument table addresses are captured by each dispatch. Residency allocations
remain retained and accounted until the completed recording is recycled.

## Completion and failure

Queue commits and shared-event values are monotonically serialized. Commit
feedback supplies execution errors and wakes waiters. Recycling requires both
feedback and the matching shared-event value. Only then may the allocator reset,
the residency set release allocations, or an upload page return to reuse.

Waits are bounded to 60 seconds. A timeout or terminal execution failure retires
the context, wakes admission waiters and rejects future publication. A timeout
never releases resources which may still be accessed by the GPU. Fault tests
retire a healthy context through bounded host-side injection; they do not submit
an intentionally nonterminating shader.

## Profiling

Enabled profiling allocates a Metal 4 timestamp counter heap with two samples
per dispatch. Samples bracket the dispatch at precise granularity. Resolved
ticks use the device timestamp frequency and populate `device_ns`. Host encoding,
submission, completion waiting and readback times are reported separately.
Disabled profiling allocates no timestamp resources.

The execution model follows Apple's [Metal 4 core API](https://developer.apple.com/documentation/metal/understanding-the-metal-4-core-api).
