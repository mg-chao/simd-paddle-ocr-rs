# GPU architecture

Shared OCR planning owns tensor layouts, logical recipes, scheduling dependencies,
admission, resource reservations, and statistics. Platform modules own buffers,
pipelines, descriptor bindings, transfers, submission, completion, and destruction.
The shared execution path does not expose graphics API resource types.

## Vulkan ownership and submission

Each native allocation, pipeline, descriptor binding, and submission retains its
device owner. A context owns the queue's serialized submission stream and reusable
mapped transfer pages. `gpu-allocator` suballocates device and host memory. Host
suballocations are aligned and padded to non-coherent atom boundaries before
flush/invalidate operations, preventing adjacent allocations from sharing atoms.

Uploads are staged into reusable mapped pages. Abandoned, unsubmitted targets are
discarded. Submitted transfers retain both the target reservation and staging
reservation. Timeline completion retires work before recycling transfer pages,
command pools, or descriptor pools. Timeouts retire the context without freeing
resources still referenced by the GPU. Device loss wakes admission waiters and
prevents new submissions.

Each resident wave becomes one primary command buffer on one compute-capable
queue. First use explicitly clears storage, scratch, counters, and padding.
Dependency tracking merges aliases by allocation identity. Read/read accesses
need no barrier; conflicting accesses generate synchronization2 memory barriers
with compute, transfer, and host stages/accesses. The implementation follows the
[Khronos synchronization model](https://docs.vulkan.org/guide/latest/synchronization_examples.html).

Neural uniforms use aligned arenas with 512-byte packet ranges. Resident uniforms
use aligned arenas with 1-KiB packets. Descriptor ranges describe the individual
packet rather than the arena capacity. Prepared bindings remain immutable and
retain every referenced resource; pipeline caches use typed kernel identities
and storage signatures.

## Shader interfaces

The native catalog names GLSL kernels, storage precision masks, binding access,
uniform sizes, workgroup geometry, and specialization constants. Cargo builds
the declared variants with pinned shaderc and reflects their SPIR-V descriptor
types, access decorations, uniform layouts, workgroup sizes, and specialization
constants against the catalog. Embedded
SPIR-V is the only Windows runtime shader input.

Portable kernels retain explicit half rounding, F32 accumulators, activation
and reduction semantics, and compensated geometry. Cooperative-matrix projection
uses capability admission and explicitly padded shared tiles, including matrix
tails. It is enabled only by the explicit F16 matrix policy. The F32 projection
uses subgroup reductions when compute-stage basic/arithmetic operations and
32- or 64-lane subgroups are available; other devices retain shared-memory
reductions. No vendor names select a kernel.

Device dispatch timestamps are separate from host work. Both resident statistics
and `StageProfile` expose encoding and submission time separately. Trace traffic
from primary diagnostic sessions remains visible alongside retained resident
sessions.

macOS currently supports CPU execution only. The previous macOS GPU
implementation and its shader catalog have been removed. GPU modules are
compiled only for non-macOS targets with the `gpu` feature enabled.
