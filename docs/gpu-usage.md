# GPU execution

Enable the `gpu` Cargo feature to use `ExecutionTarget::Gpu`. CPU execution and
F32 storage remain the defaults. Automatic execution retains its existing CPU
selection and fallback policy; an explicit GPU request reports initialization,
capability, allocation, validation, and execution failures to the caller.

## Windows

The Windows implementation uses `ash` 0.38 and `gpu-allocator` 0.28. It requires a
physical integrated or discrete GPU supporting Vulkan 1.3, synchronization2,
timeline semaphores, and maintenance4. Device selection prefers discrete GPUs
and compute-only queues. No surface, window, or graphics queue is required.

Build with the repository's pinned Rust toolchain, Visual Studio C++ Build Tools,
CMake, and Python available. Cargo builds pinned `shaderc` 0.10.1 from its bundled
source. GLSL and catalog metadata in `src/runtime/gpu/vulkan/shaders` are compiled
and checked during the build. Neither a Vulkan SDK nor a runtime shader compiler
is needed by the deployed application. CPU-only builds do not build shaderc.

```powershell
cargo build --release --locked --features gpu --example ocr
```

F32 is the portable default. `GpuCompileOptions::precision =
GpuPrecision::MixedF16` uses explicit half storage boundaries and F32 arithmetic
and accumulation. It requires storage-buffer 16-bit access and shader FP16
features. `GpuMatrixPrecision::F16` additionally admits supported cooperative
matrix projections with FP16 operands and F32 accumulation. Admission checks the
reported matrix shapes, component types, subgroup operations, shared-memory
capacity, and Vulkan memory-model/device-scope support. Other cases use portable GLSL kernels.

`GpuContext::adapter_info()` returns crate-owned `GpuAdapterInfo`, with `GpuApi`
and `GpuDeviceType`. `GpuContext::limits()` returns `GpuLimits`. Consumers should
use these types directly; platform resource handles remain private.

`GpuContextOptions::max_device_bytes` bounds live logical buffer storage,
including active upload ranges, readback buffers, and pending work. Allocator
pool capacity is reported separately by `GpuDeviceMemoryStatistics::committed_bytes`
and `peak_committed_bytes`; idle pooled capacity can exceed live bytes. Prepared
bindings and in-flight submissions retain their resources until completion.

Profiling records dispatch-level device timestamps when the queue supports them.
`GpuKernelProfile::device_ns` and `GpuTimingGranularity::Dispatch` describe GPU
durations. Host encoding, queue submission, completion waiting, and readback
durations remain separate. `StageProfile::gpu_encoding_ns` measures host
preparation/encoding, while `gpu_submission_ns` measures queue submission.
Disabled profiling allocates no timestamp queries.

## macOS

The previous macOS GPU backend has been removed. macOS currently supports CPU
execution only, including builds with the `gpu` feature enabled. The
`runtime::gpu` module and GPU context APIs are available only on non-macOS targets.
`ExecutionTarget::Gpu` returns `ErrorKind::Unsupported` on macOS;
`ExecutionTarget::Auto` falls back to CPU and records the unsupported reason.

See [local GPU validation](gpu-validation.md) for physical-device Vulkan checks.
