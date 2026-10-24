# Native GPU OCR

Enable `gpu` to build the wgpu **30.0.1** backend. Vulkan, DX12 and Metal are
enabled; WGSL kernels use portable compute operations. The default execution
target remains CPU. GPU-selected OCR defaults to the full resident pipeline and FP32. No ONNX Runtime, C# runtime, or
model download is involved in production inference.

```powershell
cargo run --release --locked --features gpu --example ocr -- --execution gpu --gpu-backend vulkan --statistics input.jpg
cargo run --release --locked --features gpu --example ocr -- --execution gpu --gpu-backend dx12 --gpu-precision mixed-f16 --json input.jpg
```

`--execution cpu|auto|gpu` applies to all stages. `--det-execution`,
`--rec-execution`, and `--cls-execution` override it independently. `--statistics`
prints actual stage backends, precision, adapter, kernels, retained memory,
submission/readback counts, and any fallback reason to stderr. `--gpu-pipeline
full|inference-only` selects resident OCR or the earlier hybrid path. Explicit
per-stage CPU overrides remain hybrid and are reported as such.

## Library API

```rust,no_run
use simd_paddle_ocr_rs::{ExecutionTarget, GpuPipelineMode, ModelPreset, OcrModelSet};
use simd_paddle_ocr_rs::runtime::gpu::{GpuBackendPreference, GpuContext, GpuContextOptions};
# fn main() -> Result<(), Box<dyn std::error::Error>> {
let gpu = GpuContext::new_with_options(GpuContextOptions {
    backend: GpuBackendPreference::Vulkan,
    max_device_bytes: Some(2 << 30),
    ..Default::default()
})?;
let models = OcrModelSet::from_directory("models", ModelPreset::ChineseV6Medium, true)?;
let plan = models.builder()
    .execution_target(ExecutionTarget::Gpu)
    .gpu_pipeline_mode(GpuPipelineMode::Full)
    .gpu_context(gpu)
    .build()?;
let mut request = plan.request()?;
let independent_request = plan.request()?;
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
use simd_paddle_ocr_rs::{Compiler, ExecutionTarget, Session};
use simd_paddle_ocr_rs::{model::Model, runtime::{CompileOptions, ExecutionLimits}};
# fn main() -> Result<(), Box<dyn std::error::Error>> {
let model = Arc::new(Model::load_path("cls.onnx")?);
let compiler = Compiler::default();
let compiled = compiler.compile(model, CompileOptions::default(), ExecutionTarget::Gpu)?;
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
Select it with `CompileOptions.gpu.precision` and request
`GpuContextOptions { shader_f16: true, ..Default::default() }` when creating the
context. FP32 and mixed plans can share that context. It requires the adapter's
`SHADER_F16` feature. `subgroups: true` opts
into reductions using the advertised subgroup feature, without assuming a lane
width. FP32 portable kernels require neither feature.
`experimental_cooperative_matrix: true` (CLI
`--gpu-experimental-cooperative-matrix`) opts in to the unqualified 8×8 FP32
matrix tier. It checks advertised scalar types, dimensions, subgroup ranges,
and the experimental feature before requesting it. Unsupported configurations
retain portable GPU kernels. It does not enable C#'s FP16 cooperative paths.
No startup benchmarking or vendor-name selection occurs. See the pinned
[wgpu feature contract](https://docs.rs/wgpu/30.0.1/wgpu/struct.Features.html#associatedconstant.EXPERIMENTAL_COOPERATIVE_MATRIX).

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

Full OCR uploads source pixels once, including valid arbitrary row strides and
RGB/BGR/alpha formats. Integer byte-domain resize, padding and normalization,
DB threshold/dilation, 8-connected foreground and 4-connected enclosed holes,
component ordering, hulls, rectangles, scores, unclip and source mapping execute
on GPU. Perspective sampling uses two-FP32 arithmetic and preserves bicubic byte
rounding, vertical counterclockwise rotation, and classifier-controlled rotation.
No native FP64 capability is required.

Compact heads retain bias order and lowest-index ties. Probability scores use
a stable denominator over the complete vocabulary. CTC performs repeat/blank
handling, token/span compaction, UTF-8 byte-budget checks and line-score
reduction on GPU. The host maps token IDs to dictionary strings and assembles
borrowed public results. Low-level tensor APIs and tracing still return dense
outputs when requested; oriented-crop APIs return their requested pixel bytes.

The encoder composes preprocessing, graph execution, heads and decoding before
submission. Recognition lines of the same width form bounded batches; different
widths occupy independent request-owned slots, bounded by `max_in_flight`.
Orientation classification groups up to four adjacent crops independently of
recognition width. Its batch shape stays fixed while commands are pending;
unused tail rows are ignored. Fixed-batch-one models and activation limits that
cannot admit a larger classifier batch retain single-crop execution. This keeps
at most three prepared crops beyond the current recognition unit. Classification
batching does not change the public recognition batch-size setting.
Available units are submitted without an external batch-filling wait. Results
are restored to original line order. A recognition request holds one admission
guard and at most two submitted waves, allowing encoding to overlap device
work. Pending waves retain their buffer reservations and readbacks until
completion, including on early errors. Shape and candidate counts cross compact
control-readback boundaries. Intermediate image, feature and vocabulary
buffers do not cross to the host during normal full-GPU OCR.

Dense 2×2 convolutions use the same bounded tiled kernels as eligible 1×1 and
3×3 convolutions. On Metal, FP32 pointwise convolutions with feature-map height
at most six can also use direct NCHW addressing for large channel reductions.
Other backends retain the existing large-channel policy. These choices require
no startup tuning. The [M4 optimization results](gpu-optimization-2026-10-24.md)
record the measured gains and hardware limitations.

Immutable weights and head/pipeline caches are shared. Forks have independent
activations and scratch. `cache_resident_scratch: true` opts in to the bounded
scratch cache; inconclusive local timings keep it disabled by default. Reuse requires no live views or pending batch
owners, is capped at 64 buffers and 64 MiB (or one quarter of the context quota),
and participates in request-pool eviction accounting. Capacity errors remain
visible; candidate limits retain the reference selection policy.

`OcrRequest::gpu_pipeline_statistics()` returns the latest resident OCR run's transfer,
placement, kernel and timing statistics. `latest_run()` includes typed stage
reports; `counters()` provides cumulative totals separately. Diagnostic bytes
in the latest run cover only that invocation. Standalone low-level session
statistics retain cumulative counters. Image processing,
geometry, projection and decoding have independent placement fields. Output,
control and diagnostic readbacks are counted separately. Peak device bytes are
the context's application allocation high-water mark; opaque driver allocations
are excluded. Statistics do not make full GPU synonymous with faster execution.
Upload counters measure image/tensor payloads; shared immutable model uploads
and small uniform parameter packets are excluded. They are not a PCIe traffic
counter. Readback counters include the requested buffer bytes.

Full-pipeline detector and recognizer inference retain FP32 weights and
activations even in a mixed context. Half-precision threshold crossings changed
contour topology, moved validation boxes by 1–2 source pixels, and shifted CTC
span boundaries despite FP32 attention residuals. Classifier inference retains
mixed storage. Each FP32 variant shares the same
device, is cached independently by precision, and is included in weight/quota
accounting. Stage statistics report its effective precision. Low-level tensor
and `InferenceOnly` execution retain the requested mixed policy; their numerical
error gates alone do not guarantee identical discrete OCR decisions.
Traced calls retain unfused low-level execution at the requested model precision,
including diagnostic CPU image/result processing. Their results can therefore
differ from guarded mixed Full execution; normal residency claims exclude traces.

## Implementation map

`runtime::LogicalGraph` owns validation, host shape evaluation, dependencies and
source lineage. `runtime::stage` dispatches immutable models and exclusive
sessions. `runtime::Compiler` owns shared validated graphs, CPU packs and lowered plans.
The CPU implementation selects physical layouts;
`runtime::gpu` owns device weights, shape schedules, liveness allocations and
readbacks. `ocr::gpu` owns resident OCR orchestration and reusable text/span assembly.
Resident sessions allocate no dense host transfer buffers. GPU compilation never constructs a `CpuCompiledModel`. OCR prepares
GPU projection resources; CPU projection packs are lazy for GPU models. An
`Auto` model's complete CPU plan is initialized once, on first allowed fallback.

The C# reference checkout used during development was
`33ff8a81672452041d110e8ba52dcd2639e54228`. Relevant source-to-rewrite mapping:

| C# Vulkan source | Rust implementation |
| --- | --- |
| `GpuGraph.cs`, `GpuRunner.cs` | `gpu.rs`, `gpu/physical.rs`, `gpu/fusions.rs` |
| `gemm_nc.comp`, convolution shaders | `tiled.wgsl`, `pointwise.wgsl`, `depthwise.wgsl`, `transposed.wgsl` |
| `se_fused.comp`, `resize4add.comp` | `se_chain.wgsl` after a separate reduction, `se.wgsl`, prescaling, portable resize/add |
| `layernorm.comp`, `attn.comp` | `reduce.wgsl`, `attention.wgsl` |

Attention retains the online-softmax algorithm, with a portable four-query,
sixteen-key tile and FP32 scalar accumulators. Private Q/K/V reshape, transpose
and slice chains are addressed directly from their source; escaping values and
selected prefixes retain materialization. Compatible pointwise chains retain
NHWC storage through explicit boundaries. Selection is explicit and based
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

See [full-pipeline qualification](full-gpu-validation.md) and the
[earlier inference baseline](gpu-validation.md) for physical-device evidence.
Metal and other GPUs require validation on those physical devices.

For source-pinned performance comparisons, build with profiling and use the
bounded harness described in [the performance phase](gpu-performance-phase.md):

```powershell
cargo build --release --locked --all-features --example gpu_ocr_bench
dotnet build tools/reference/Reference.csproj -c Release
python tools/gpu_performance.py --backend vulkan --corpus reports/full-gpu/release-vulkan/corpus --report reports/gpu-performance/new-run --presets tiny --cases sparse --configs 1:1:2 --modes cpu inference-only full --csharp cpu gpu --measurements 3 --profile
```

Use a fresh report directory. `--configs` means requests:GPU-slots:recognition-batch;
profiling calls are separate from warm measurements. The harness freezes binaries
and input identities, validates complete public OCR outputs, and retains failed
comparisons. C# GPU is a hybrid mixed-precision reference, with fallback checked
explicitly. Vertical crop rotation differs between the two libraries; such
configurations are reported as unqualified rather than used for speed claims.
