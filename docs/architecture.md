# Plans, requests, and backend resources

The library separates imported model data, compilation, and mutable execution.
The numerical CPU kernels and GPU shaders remain backend-specific.

## Ownership

- `Model` owns normalized ONNX data. It has no execution caches.
- `Compiler` owns validated graphs, lowered plans and CPU packing caches, and resolves its GPU
  context once, on demand. Reuse an `Arc<Compiler>` across builders to share
  these resources. Dropping the compiler releases its cache; compiled artifacts
  retain the resources they still need.
- Stage plans and `OcrPlan` hold immutable models and validated configuration.
  Creating plans does not allocate request activation storage.
- `OcrRequest` owns its selected execution state, bounded workers, diagnostics,
  and reusable results. Each request requires exclusive mutable access to run.
- `OcrEngine` leases independent requests from one shared plan and accounts for
  their retained host and device memory. Shared weights are not charged per lease.

Bound-operation geometry belongs to the common runtime. CPU kernels and GPU
lowering both consume it. GPU graph storage is distinct from host transfer
storage: resident image stages do not allocate dense input/output staging.
OCR composition owns stage ordering, compact result assembly, and fallback;
GPU runtime primitives own encoding, submission, resource reservations, and
completion. Submitted work retains its resources until completion on all exits.

## Migrating callers

The builder now produces an immutable plan. Create a request explicitly:

```rust,no_run
use simd_paddle_ocr_rs::{ModelPreset, OcrModelSet, RunOptions};
use simd_paddle_ocr_rs::image::{ImageView, PixelFormat};
# fn main() -> Result<(), Box<dyn std::error::Error>> {
let models = OcrModelSet::from_directory("models", ModelPreset::ChineseV6Tiny, true)?;
let plan = models.builder().build()?;
let mut request = plan.request()?;
let pixels = vec![255; 640 * 480 * 3];
let image = ImageView::new(&pixels, 640, 480, 0, PixelFormat::Rgb24)?;
let result = request.run(image, RunOptions { capture_spans: true })?;
let owned = result.try_to_owned()?;
println!("{}", owned.text);
# Ok(())
# }
```

`Ocr` is renamed to `OcrRequest`. `run` takes `RunOptions` in place of a capture
boolean. Borrowed results remain valid until the next mutable use of the request;
use `try_to_owned` when the result must outlive it.

Replace `into_factory` with a cloned plan and `plan.request()`. A plan retains no
warm request buffers. Construct an engine with
`OcrEngine::new(Arc::new(plan), limits)` or use `OcrBuilder::build_engine`.

Use `DetectorPlan`, `ClassifierPlan`, `RecognizerPlan`, and `OrientedLinePlan`
for standalone stages. Their `request()` methods create independent execution
storage while sharing compiled resources. Request forks preserve their configuration.

Use immutable `extractor()` and `recognizer()` accessors for inspection.
Replace nested `extractor_mut().detector_mut().set_resize_options(...)` calls
with `request.set_detector_resize(...)`. Resize and batching changes are
validated before updating the request's configuration. A failed update leaves
the previous configuration usable.

Strict GPU placement returns errors. Auto fallback remains restricted to
unsupported execution and device loss; input, callback, quota, allocation,
validation, and mapping errors remain visible. Diagnostic observers are never
replayed. CPU fallback resources are prepared lazily.

## Compilation settings

`CompileOptions.fusion` is common to both lowerings. CPU kernel and layout
policy are `CompileOptions.cpu: CpuCompileOptions`; GPU storage policy is
`CompileOptions.gpu: GpuCompileOptions`. These settings types are available in
CPU-only builds; GPU execution requires the `gpu` feature.
`GpuContextOptions` describes device capabilities (`shader_f16`, subgroups),
admission limits, backend selection and allocation quotas. It no longer
chooses model precision. FP32 remains the compilation default.

```rust,no_run
use std::sync::Arc;
use simd_paddle_ocr_rs::{Compiler, ModelPreset, OcrModelSet};
# fn main() -> Result<(), Box<dyn std::error::Error>> {
let compiler = Arc::new(Compiler::default());
let models = OcrModelSet::from_directory("models", ModelPreset::ChineseV6Tiny, false)?;
let first = models.clone().builder().compiler(compiler.clone()).build()?;
let second = models.builder().compiler(compiler).build()?;
// first and second share lowering/packing resources for identical model Arcs.
let request_a = first.request()?;
let request_b = second.request()?;
# Ok(())
# }
```

Cache identity is the imported `Arc<Model>` plus CPU policy/fusion or GPU
precision/fusion within the compiler's device context. A separately imported
copy is a different model. There is no process-wide compilation registry.
CPU-only builds and disabled stages never initialize a GPU. An initialization
failure is retained and reused by all subsequent compilations on that compiler.
When a builder receives an explicit compiler, that compiler owns device selection;
builder `gpu_context` and `gpu_options` settings apply only to its implicit compiler.

## Reporting and memory

`OcrRequest::latest_run()` describes one completed invocation, including failed
invocations. Stage keys are `OcrStage`, and execution is `OcrExecution::Cpu`,
`Hybrid`, or `Resident`. Tensor session counters in that report are deltas;
`OcrRequest::counters()` holds cumulative traffic and invocation counts.
`execution_statistics()` remains an inspection of the active sessions' bound
plans and cumulative session counters. Uninitialized sessions report no backend.

`memory_snapshot()` returns disjoint retained `host_bytes` and `device_bytes`.
The engine obtains one snapshot when a lease returns and uses its sum for idle
eviction. `OcrPlan::shared_weights()` reports shared host coefficients/packs and
device weights separately. GPU transition peaks and the context allocation
high-water mark remain separate from retained request capacities. wgpu/driver
internal allocations and allocator overhead are outside this accounting.

Preserved fallback errors are available on compiled models and sessions via `compiled.fallback_error()` / `session.fallback_error()`. The returned error
retains its source chain. Serializable execution provenance contains its kind,
origin, stage and message. Callback errors cannot become replayable by carrying
an `Unsupported` or `DeviceLost` kind.

## Module boundaries

| Boundary | Responsibility |
| --- | --- |
| `model`, `onnx` | Import, normalization, original coefficients |
| `runtime::graph`, `runtime::bound`, `runtime::shapes` | Validation, shared operations, concrete shape/liveness plans |
| `runtime::compiler` | Explicit shared cache and device resolution ownership |
| CPU runtime and kernels | CPU lowering, layouts, packed execution |
| `runtime::gpu` | Device lowering, graph buffers, recording, submission and readback |
| Stage plans / requests | Model contracts and exclusive stage execution |
| `ocr` / `ocr::gpu` | Composition, fallback policy, ordering, reusable public results |
| `engine` | Admission, leases, panic discard and idle eviction |

## Local validation

Run the nonignored correctness matrix, formatting, Clippy, examples and doctests
on the physical development machine:

```powershell
python tools/check_architecture.py
```

Add `--release --hardware vulkan dx12` to exercise the selected physical-device
regressions. Hardware groups have a 120-second timeout and exclude performance
tests. Logs and a command/result manifest are written to
`target/architecture-validation`. The matrix covers all eight combinations of
parallel/serial, GPU, and profiling. The compile-fail doctests verify that views
cannot escape requests or engine leases.

For a bounded real-model comparison using existing local assets:

```powershell
cargo build --release --locked --all-features --example ocr
python tools/full_gpu_validation.py --backend vulkan --presets tiny --cases typical --report target/architecture-real-vulkan
```

This runs one image once in each of CPU, hybrid and resident mode, checking text,
ordering, boxes, scores and captured spans against the CPU result. It does not
measure performance. Metal and NEON require a separate physical Apple Silicon
machine; Windows x64 checks do not qualify those backends.

### Refactor validation record — 2026-10-24

Local Windows x64 checks used the pinned `nightly-2026-09-08` toolchain and an
AMD Radeon RX 6750 XT (driver `32.0.21045.16010`). Each final feature combination
was invoked once with nonignored library tests:

| CPU execution | GPU feature | Profiling feature | Passed |
| --- | --- | --- | ---: |
| Parallel | Enabled | Enabled | 369 |
| Parallel | Enabled | Disabled | 364 |
| Parallel | Disabled | Enabled | 360 |
| Parallel | Disabled | Disabled | 355 |
| Serial | Enabled | Enabled | 357 |
| Serial | Enabled | Disabled | 352 |
| Serial | Disabled | Enabled | 348 |
| Serial | Disabled | Disabled | 343 |

The all-feature release suite also passed 369 tests. The selected 13 hardware
correctness tests passed once per backend on Vulkan and DX12, including eight
injected-failure cases in the fallback test. One real `tiny` model sample passed
once in each CPU, inference-only and resident mode on each backend, using the
existing comparison tolerances. Formatting, Clippy with warnings denied,
all-feature and serial example builds, and all seven compile-fail doctests in
both feature configurations passed. Rustdoc also built with warnings denied.
No benchmarks were run. Metal, NEON, other
adapters and the complete real-model corpus were not qualified by this run.
