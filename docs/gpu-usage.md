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

On Apple Silicon with macOS 14 or newer, `gpu` selects native Metal. The same
`GpuContext`, `ExecutionTarget`, precision, quota, and resident OCR APIs apply.
Intel Macs and older operating systems report `ErrorKind::Unsupported` for an
explicit GPU request; `Auto` retains its documented fallback policy.

Install Xcode and select its developer directory, complete its first-launch
setup, then install the additional compiler component:

```sh
xcodebuild -downloadComponent MetalToolchain
xcrun metal --version
cargo build --release --locked --features gpu --example ocr
target/release/examples/ocr --execution gpu --statistics input.png
```

Cargo compiles MSL 3.1 into an embedded Metal library and a native M4 pipeline
archive for the installed SDK's macOS major version. The native archive contains
the common homogeneous FP32 and FP16 storage variants; portable IR retains all
binding precision combinations. Deployment requires no Xcode installation or
source shader compiler. Matching M4 systems load common pipelines from the
archive; other binding signatures, Apple GPUs, OS versions, and archive misses compile the
embedded IR through the native Metal driver. Both paths execute on the GPU.
Pipeline state objects are cached per context. Tensor dimensions are parameter
data; changing a shape does not compile dimension-specific shaders.

The archive is extracted into a private process-local temporary directory and
removed with its context. It is not a persistent application shader cache.
Physical-device archive tests use `FailOnBinaryArchiveMiss` to verify actual
hits. The archive retains reflection metadata because the tested Metal runtime
cannot find its pipelines when that metadata is omitted.

F32 remains the default. MixedF16 stores intermediate tensors in half precision;
SIMD-group matrix kernels accumulate in F32. Explicit `--gpu-matrix f16` also
allows half projection operands and the existing reduced-precision OCR policy.
Metal's resident detector retains F32 tensors and weights by default: half
quantization can change DB threshold decisions and thin-text crops. On macOS,
`GpuContextOptions::metal_detector_precision = GpuPrecision::MixedF16` explicitly
permits mixed detector storage when both compile storage and matrix policy also
select F16. Qualify this option for each model and workload; the tiny model's
border fixture requires the default FP32 policy. Classification and recognition
retain the selected precision. Per-stage
statistics expose the effective precision; all three stages execute on Metal.
The backend uses unified-memory buffers with immutable upload snapshots, checked
bindings, dependency barriers, and completion-owned resources. Device failure
retires the context and wakes admission waiters.

Profiling uses calibrated Metal timestamp counters. On Apple GPUs it records one
dispatch per sampled compute pass; this instrumentation changes scheduling and
is excluded from latency benchmarks. Unsupported counters report unavailable
timing rather than substituted host durations.

See [Metal validation](metal-validation.md) for commands and the current limits
of correctness and performance evidence.

See [local GPU validation](gpu-validation.md) for physical-device Vulkan checks.
