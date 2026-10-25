# Plans, requests, and backend resources

The library separates imported model data, compilation, and mutable execution.
The numerical CPU kernels and Metal shaders have separate execution machinery.

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
`GpuContextOptions` contains only admission/resource bounds: `max_in_flight`,
`recognition_wave_size`, `classification_batch_size` and `max_device_bytes`.
Device capabilities are discovered internally. GPU work always uses the native
Metal executor; there is no alternate backend or pipeline-mode selector. FP32
remains the compilation default, with explicit F16 storage/matrix policies.

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
When a builder receives an explicit compiler, that compiler owns its Metal context;
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

Add `--release --hardware` on a physical Mac to exercise Metal integration
regressions. Hardware checks have a 120-second timeout and exclude performance
tests. Logs and a command/result manifest are written to
`target/architecture-validation`. CPU-only combinations do not initialize Metal.
The compile-fail doctests verify that views cannot escape requests or engine leases.

For a bounded real-model comparison using existing local assets:

```sh
cargo build --release --locked --all-features --example gpu_ocr_bench
python tools/full_gpu_validation.py --presets tiny --cases typical --report target/metal-qualification
```

This runs one image once on CPU and once on the current Metal implementation,
checking its unchanged Rust text/geometry/score/CTC contract. It performs no
benchmark repetitions. Explicit per-stage CPU overrides remain supported and
are reported as Hybrid composition; they do not select a second GPU implementation.
See [GPU validation](gpu-validation.md) and [CPU validation](macos-cpu-validation.md).
