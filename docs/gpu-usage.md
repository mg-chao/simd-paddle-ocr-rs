# Native Metal GPU execution

Enable the optional `gpu` feature to use the single GPU implementation: native
Metal on macOS 26 or later. Windows GPU support is not implemented. CPU execution remains
available without a GPU runtime and remains the default.

The locally validated device is Apple M4 with macOS 27. Metal shaders are
compiled using Metal 4. Explicit GPU requests report unsupported devices or
compilation failures; they do not select another GPU backend. `ExecutionTarget::Auto`
may fall back to CPU only under the documented unsupported/device-loss policy.
Input, callback, allocation, quota and validation errors remain visible.

```sh
cargo run --release --locked --features gpu --example ocr -- \
  --execution gpu --statistics input.jpg
cargo run --release --locked --features gpu --example ocr -- \
  --execution gpu --gpu-precision mixed-f16 --gpu-matrix f16 --alignment input.jpg
```

`--gpu-precision` selects F32 or MixedF16 storage. `--gpu-matrix` independently
selects F32 or explicit F16 matrix operands with F32 accumulation. Both default
to F32. These are numerical policies on the same Metal executor. They are not
backend or pipeline alternatives. Explicit F16 accuracy is qualified separately;
it can change accepted crop dimensions and scores under its existing contract.

## Library configuration

```rust,no_run
use simd_paddle_ocr_rs::{ExecutionTarget, ModelPreset, OcrModelSet};
use simd_paddle_ocr_rs::runtime::{CompileOptions, GpuCompileOptions, GpuMatrixPrecision, GpuPrecision};
use simd_paddle_ocr_rs::runtime::gpu::{GpuContext, GpuContextOptions};
# fn main() -> Result<(), Box<dyn std::error::Error>> {
let context = GpuContext::new_with_options(GpuContextOptions {
    max_in_flight: 1,
    recognition_wave_size: 16,
    classification_batch_size: 16,
    max_device_bytes: None,
})?;
let models = OcrModelSet::from_directory("models", ModelPreset::ChineseV6Small, true)?;
let plan = models.builder()
    .execution_target(ExecutionTarget::Gpu)
    .gpu_context(context)
    .compile_options(CompileOptions {
        gpu: GpuCompileOptions {
            precision: GpuPrecision::MixedF16,
            matrix_precision: GpuMatrixPrecision::F16,
        },
        ..Default::default()
    })
    .build()?;
let mut request = plan.request()?;
# let _ = &mut request;
# Ok(())
# }
```

`GpuContextOptions` contains exactly four resource controls:

- `max_in_flight`: positive admission bound, default 1.
- `recognition_wave_size`: 1–16 independent shapes per wave, default 1.
- `classification_batch_size`: 1–16 classifier rows, default 4. Fixed model
  batches and activation quotas may reduce the effective capacity.
- `max_device_bytes`: optional context allocation quota.

Capabilities are discovered internally. There are no backend, encoder,
experimental matrix, or per-kernel enable/disable switches. Qualified MPP and
specialized kernels share one Metal executor with their bounded scalar or
WGSL-to-MSL math fallbacks. Compile rejection of an optional specialization
preserves the same-engine fallback; fatal errors propagate.

GPU stages with compatible contexts compose resident preprocessing, detector
geometry, crops, classification, recognition, projection and CTC. GPU stages always use the resident pipeline. Explicit `.detector_target(...)`,
`.classifier_target(...)` and `.recognizer_target(...)` CPU overrides remain
supported. Such a mixed CPU/Metal composition is reported as Hybrid. Combining
resident GPU stages from different contexts is rejected before execution.

## Ownership and inspection

A compiler shares immutable model resources and weights. Each request owns its
activation and parameter arenas. Submitted work retains buffers, bindings and
pipeline owners through completion, including on failures. CPU-only builds and
CPU-selected stages do not initialize Metal.

`latest_run()`, `execution_statistics()` and `gpu_pipeline_statistics()` report
actual stage placement, selected kernels, traffic, submissions and allocation
high-water marks. Device memory reservations cover application-owned buffers;
Metal/driver allocation overhead is outside that ledger. GPU execution never
uses software adapters. Diagnostic calls use the same Metal implementation;
benchmark them separately from uninstrumented wall latency.

See [architecture](architecture.md), [GPU validation](gpu-validation.md) and
[performance scope](gpu-performance.md). CPU setup and platform support remain
in [CPU usage](cpu-usage.md).
