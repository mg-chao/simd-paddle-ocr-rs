# Native GPU inference

Enable `gpu` to build the wgpu **30.0.1** backend. Vulkan, DX12 and Metal are
enabled; WGSL kernels use portable compute operations. The default execution
target remains CPU, and GPU defaults to FP32. No ONNX Runtime, C# runtime, or
model download is involved in production inference.

```powershell
cargo run --release --locked --features gpu --example ocr -- --execution gpu --gpu-backend vulkan --statistics input.jpg
cargo run --release --locked --features gpu --example ocr -- --execution gpu --gpu-backend dx12 --gpu-precision mixed-f16 --json input.jpg
```

`--execution cpu|auto|gpu` applies to all stages. `--det-execution`,
`--rec-execution`, and `--cls-execution` override it independently. `--statistics`
prints actual stage backends, precision, adapter, kernels, retained memory,
submission/readback counts, and any fallback reason to stderr.

## Library API

```rust,no_run
use simd_paddle_ocr_rs::{ExecutionTarget, ModelPreset, OcrModelSet};
use simd_paddle_ocr_rs::runtime::gpu::{GpuBackendPreference, GpuContext, GpuContextOptions, GpuPrecision};
# fn main() -> Result<(), Box<dyn std::error::Error>> {
let gpu = GpuContext::new_with_options(GpuContextOptions {
    backend: GpuBackendPreference::Vulkan,
    precision: GpuPrecision::F32,
    max_device_bytes: Some(2 << 30),
    ..Default::default()
})?;
let models = OcrModelSet::from_directory("models", ModelPreset::ChineseV6Medium, true)?;
let mut request = models.builder()
    .execution_target(ExecutionTarget::Gpu)
    .classifier_target(ExecutionTarget::Cpu)
    .gpu_context(gpu)
    .build()?;
let independent_request = request.fork()?;
# Ok(())
# }
```

`gpu_options` on the builder is an alternative to constructing a context. It
allows `Auto` to handle an unavailable adapter or requested feature. A supplied
context takes precedence over options. Models compiled on a context share its
device, queue, immutable weight uploads, and pipelines. Forks own separate
activation storage and bounded shape caches.

For tensor inference, use backend-neutral `CompiledModel` and `Session`:

```rust,no_run
use std::sync::Arc;
use simd_paddle_ocr_rs::{CompiledModel, ExecutionConfig, ExecutionTarget, Session};
use simd_paddle_ocr_rs::{model::Model, runtime::{CompileOptions, ExecutionLimits}};
# fn main() -> Result<(), Box<dyn std::error::Error>> {
let model = Arc::new(Model::load_path("cls.onnx")?);
let compiled = CompiledModel::compile(model, CompileOptions::default(), ExecutionConfig {
    target: ExecutionTarget::Gpu,
    ..Default::default()
})?;
let mut session = Session::new_serial(compiled, &[1, 3, 80, 160], ExecutionLimits::default())?;
let input = vec![0.; 3 * 80 * 160];
let output = session.run(&input)?; // logical F32; borrows the exclusive session
println!("{:?}", output.shape());
# Ok(())
# }
```

`Session::new_prefix` selects the output of a validated logical node. Its
boundary is protected from fusion. `LogicalGraph` holds shared validation,
shape instructions, dependencies and source lineage. `CpuCompiledModel` and
`GpuCompiledModel` can independently compile an `Arc<LogicalGraph>`.

Breaking migration: the old CPU-only `CompiledModel` and `Session` are now
`CpuCompiledModel` and `CpuSession`. The common names refer to enum-dispatched
interfaces. CPU SIMD `KernelPolicy`/`Backend` selects arithmetic kernels;
`ExecutionTarget` selects placement. Existing borrowed-output and exclusive
mutable-session contracts remain in effect.

## Precision, failures, and memory

`GpuPrecision::MixedF16` stores weights and fused-plan activations in FP16 and
accumulates in FP32. Attention recognizers retain FP32 activations because half
rounding in their residual streams can change close vocabulary decisions.
Unfused and diagnostic plans also retain FP32 activations to
avoid accumulating a half-rounding error at every arithmetic boundary; they
share the same FP16 weights. Statistics report this effective storage policy.
It requires the adapter's `SHADER_F16` feature. `subgroups: true` opts
into reductions using the advertised subgroup feature, without assuming a lane
width. FP32 portable kernels require neither feature. There are no cooperative
matrix extensions or native API escape paths.

Strict `Gpu` returns errors. `Auto` may use CPU for unsupported capabilities or
shapes, and can retry a failed invocation once after device loss. Full CPU packs
are created lazily on fallback. Input errors, allocation/resource limits, GPU
validation/mapping errors and callback errors are returned; callbacks are never
replayed. Read the session's statistics to distinguish placement from intent.

A normal invocation uploads input, submits the graph once, keeps intermediates
on GPU, and reads only the selected output. The shape cache retains eight previous
schedules plus the current one; pipelines live independently on the context. Liveness allocation
reuses separate buffers bounded by storage-binding limits. Canonical reshapes
alias only with extended lifetimes. Compatible concat producers write directly
into disjoint destination regions. Failed reshapes preserve the old session.

`GpuMemoryStatistics` accounts for activations, readback, host input/output,
upload storage, parameters, diagnostics, profiling and allocation-transition
peaks. Context quotas include simultaneous old/new device allocations. Driver
and wgpu internal allocations are outside the application's allocation ledger.
`OcrEngine` includes retained request GPU buffers in idle eviction accounting;
shared model weights are not charged per request. GPU completion waits hold no
device-wide mutex and do not occupy CPU inference workers.

Tracing executes an unfused diagnostic GPU schedule and returns logical F32
tensors. It deliberately adds submissions/readbacks; diagnostic counts and
timings are separate from production. `CompileOptions { fusion: false, .. }`
also permits ordinary unfused graph execution.

Preprocessing, DB postprocessing and compact REC/CLS projection/decoding stay
on CPU. Full low-level GPU graph outputs are supported too. GPU image processing,
specialized projection heads and cross-shape submission batching are excluded.

## Implementation map

`runtime::LogicalGraph` owns validation, host shape evaluation, dependencies and
source lineage. `runtime::stage` dispatches immutable models and exclusive
sessions. The CPU implementation owns its SIMD packs and physical layouts;
`runtime::gpu` owns device weights, shape schedules, liveness allocations and
readbacks. GPU compilation never constructs a `CpuCompiledModel`. OCR prepares
only its required CPU projection resources; an `Auto` model's complete CPU plan
is initialized once, on first fallback.

The C# reference checkout used during development was
`33ff8a81672452041d110e8ba52dcd2639e54228`. Relevant source-to-rewrite mapping:

| C# Vulkan source | Rust implementation |
| --- | --- |
| `GpuGraph.cs`, `GpuRunner.cs` | `gpu.rs`, `gpu/physical.rs`, `gpu/fusions.rs` |
| `gemm_nc.comp`, convolution shaders | `tiled.wgsl`, `pointwise.wgsl`, `depthwise.wgsl`, `transposed.wgsl` |
| `se_fused.comp`, `resize4add.comp` | `se.wgsl`, portable resize/add recipe |
| `layernorm.comp`, `attn.comp` | `reduce.wgsl`, `attention.wgsl` |

Attention retains the online-softmax algorithm, with a portable four-query,
sixteen-key tile and FP32 scalar accumulators. Selection is explicit and based
on shapes and enabled capabilities. Startup does not benchmark or autotune.
Unsupported fusion patterns use ordinary GPU kernels. Diagnostic plans disable
fusion and physical rewrites, including concat placement.

## Reproducible local validation

The seven models and hashes are listed in [CPU usage](cpu-usage.md). The
developer harness uses identical deterministic NCHW tensors and checks finite
outputs, shapes, and `abs_error <= atol + rtol * abs(reference)`. FP32 uses
`atol=rtol=1e-3`; mixed storage uses `atol=5e-2`, `rtol=1e-2`.

```powershell
cargo build --release --locked --features gpu --examples
python tools/gpu_validation.py inventory
python tools/gpu_validation.py compare --backend vulkan
python tools/gpu_validation.py compare --backend dx12 --unfused
python tools/gpu_validation.py compare --backend vulkan --precision mixed
python tools/gpu_ocr_smoke.py
cargo test --release --locked --all-features gpu_hardware -- --ignored --nocapture --test-threads=1
cargo test --release --locked --all-features gpu_integration_tests -- --ignored --nocapture --test-threads=1
```

Set `SIMD_OCR_GPU_BACKEND=dx12` to run the physical operator and integration
tests on DX12. These explicitly requested tests fail if the adapter is absent;
they do not silently skip or select software rendering.

Without `--model`/`--shape`, the comparison covers CLS batches 1/2, DET 128x128
and 96x160, and REC widths 32/128/320 with batches 1/2: 26 cases per
backend/precision/fusion configuration. JSON reports record model and executable
hashes, input hashes, options, shapes, adapter identity and selected kernels.
Reports and raw tensors are developer outputs under ignored `reports/`.

The optional C# oracle requires the sibling reference checkout and .NET 10:

```powershell
dotnet build tools/reference/Reference.csproj -c Release
python tools/gpu_validation.py compare --model cls --shape 1 3 80 160 --reference --boundary p2o.pd_op.flatten.0.0
```

`--boundary` prunes both runtimes to an **original ONNX tensor name**, avoiding
node-index comparisons after normalization/fusion. The C# raw GPU graph does
not implement the final CLS MatMul; compare its backbone boundary instead.
The C# GPU oracle uses mixed precision, so its comparison uses the mixed gate.
Its inability to execute a boundary is recorded as an error, never a pass.
Set `SIMD_OCR_TRACE_COMPARE` to a JSON output path to compare every available
logical Rust CPU/GPU diagnostic boundary by its tensor name. These readbacks
are diagnostic and are counted separately from the production invocation.

The ignored `gpu_bounded_performance` test measures only pointwise, dense and
attention cases, one warmup and three measured calls per path. It is separate
from correctness tests and does not autotune production startup.

See [local validation results](gpu-validation.md) for physical-device evidence.
Metal and other GPUs require validation on those physical devices.
