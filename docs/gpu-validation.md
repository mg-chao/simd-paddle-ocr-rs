# Local native GPU validation — 2026-10-22

Tested locally on Windows x64, **AMD Radeon RX 6750 XT**, using native Vulkan
and DX12, wgpu **30.0.1**, and Rust `nightly-2026-09-08`. Vulkan reports AMD
proprietary driver **26.10.1**; DX12 reports **32.0.21045.16010**. All execution
checks used release binaries and the physical adapter. Metal and other physical
devices remain **unverified**.

## Full graph comparisons

**208/208 passed**, with finite outputs and identical shapes. Each configuration
covers all seven models: CLS batches 1/2 at 80×160; each DET model at 128×128 and
96×160; each REC model at widths 32/128/320, height 48, batches 1/2. Every case
uses the same deterministic tensor for the CPU and GPU reference. These are
correctness invocations, not performance samples.

The unchanged elementwise gates are `1e-3 + 1e-3 * abs(reference)` for FP32 and
`5e-2 + 1e-2 * abs(reference)` for mixed storage. The maximum absolute error is
reported below; the gate itself includes the relative term.

| Backend | Precision | Schedule | Passed | Maximum absolute error |
| --- | --- | --- | --- | --- |
| vulkan | f32 | fused | 26/26 | 5.033612e-05 |
| vulkan | f32 | unfused | 26/26 | 0.0003367662 |
| vulkan | mixed | fused | 26/26 | 0.04756138 |
| vulkan | mixed | unfused | 26/26 | 0.04449579 |
| dx12 | f32 | fused | 26/26 | 5.099177e-05 |
| dx12 | f32 | unfused | 26/26 | 0.0003359318 |
| dx12 | mixed | fused | 26/26 | 0.04267856 |
| dx12 | mixed | unfused | 26/26 | 0.04449657 |

Each strict GPU invocation asserts no CPU weight packing during compilation,
one production submission, one selected-output readback, and no diagnostic
readbacks. Unfused schedules remain entirely on GPU. Mixed storage uses FP16
weights with FP32 accumulation. Unfused/diagnostic plans and attention
recognizers retain FP32 activations; other optimized plans use FP16 activations.
The effective policy is visible in statistics and tensor descriptors. This
policy fixes accumulated half-store error and preserves close vocabulary
choices without changing numerical tolerances.

## Regression and lifecycle checks

- `cargo test --release --all-features --lib --offline`: **365 passed**, 30
  explicitly ignored tests. GPU tests below are invoked separately.
- `cargo test --release --no-default-features --lib --offline`: **341 passed**,
  8 explicitly ignored tests. The probability-mode test now limits thread counts
  to what the compiled `parallel` feature supports.
- **18 operator/lifecycle tests plus 3 OCR integration tests passed on each
  native backend**. Coverage includes irregular tails, grouped/dilated and
  transposed convolution, broadcasting, rank-five transpose, pooling, resize,
  reductions, normalization, attention, SE, concat placement, fusion guards,
  FP16 storage, optional subgroup variants, and traced logical tensors.
  Four additional medium-REC full-graph comparisons pass with subgroup variants
  enabled (both native backends, FP32 and mixed, batch 1 at width 128).
- Lifecycle coverage includes bounded shape-cache eviction, buffer growth and
  reuse, allocation/reshape rollback, device quotas, shared weight uploads,
  independent concurrent sessions, callback errors, injected device loss,
  strict GPU behavior, lazy Auto fallback, and request-pool eviction accounting.
- All-target checks cover default CPU features, all features, and `gpu` without
  default features. Examples compile. A direct compiler check accepts ordinary
  borrowed output and rejects overlapping mutable session use with **E0499**.

Injected device-loss tests validate recovery policy; no physical device reset
was forced. Allocation totals cover application-owned buffers and host storage,
not opaque driver/wgpu allocations.

## Fixed OCR sample

The public C# checkout's `examples/sample.jpg` was run once per configuration:
three CPU baselines and **12 GPU runs** (tiny/small/medium × Vulkan/DX12 ×
FP32/mixed). All GPU runs match their CPU baseline's **text and orientation**
exactly, with 16 recognized lines. The OCR pipeline uses GPU backbones and the
existing compact CPU projection/decoding paths.

Sample SHA-256:
`30c417c9f758a3b62718729f5a944f7d2e10cdd2bde0e8ce6785523ddb68ffe9`.

## C# reference comparison

The developer oracle used C# checkout
`33ff8a81672452041d110e8ba52dcd2639e54228`, .NET 10, and identical raw F32 inputs.
Boundaries are original ONNX tensor names: CLS `p2o.pd_op.flatten.0.0`, tiny REC
`p2o.pd_op.add.40.0`, small/medium REC `p2o.pd_op.transpose.8.0`, and DET graph
outputs. Pruned-model and original-model hashes are both recorded.

C# CPU agrees with Rust CPU on **all seven** boundaries under the FP32 gate.
The C# GPU reference agrees on CLS and DET under the mixed gate. Its three REC
backbone outputs exceed that gate; those failures are retained as reference
differences, not counted as passing evidence for Rust. Rust GPU independently
passes its CPU comparisons above. The raw C# GPU graph also does not support
the full CLS projection, so its backbone boundary is used.

| Model | C# CPU maximum error | C# GPU maximum error | C# GPU elements outside gate |
| --- | --- | --- | --- |
| cls | 2.700835e-08 | 0.002676917 | 0 |
| det | 1.192093e-07 | 7.152557e-06 | 0 |
| small_det | 1.192093e-07 | 0.0004124045 | 0 |
| medium_det | 2.384186e-07 | 0.004369915 | 0 |
| rec | 5.865097e-05 | 0.6106482 | 801 |
| small_rec | 4.768372e-05 | 0.5726871 | 762 |
| medium_rec | 0.0001125336 | 0.6398754 | 1342 |

## Bounded selection measurements

Each experiment used 1–3 inputs, **one correctness warmup and three measured
invocations per path**, with no automatic repetition or startup autotuning.
Each performance execution finished well below 60 seconds. Host timings were
noisy, so a separate three-input device-timestamp check resolved pointwise and
attention selection. Values below are median compute microseconds; pointwise
includes its layout-conversion dispatch.

| Case | Generic GPU | Optimized GPU |
| --- | --- | --- |
| Pointwise 65 → 67, 32×33 | 315.00 | 244.48 |
| Pointwise 128 → 128, 64×64 | 3809.96 | 1260.44 |
| Attention 1×4×128×32 | 708.32 | 423.08 |

Pointwise uses the current mixed policy: generic/unfused activations are FP32,
optimized activations are FP16, and both share FP16 weights. Attention uses
FP32. The separate dense 32 → 64, 3×3, 32×33 case measured median end-to-end
invocation times of **2.3502 ms generic / 0.7669 ms tiled-16**. Three samples
are sufficient for the bounded selection check, not a throughput guarantee;
there were timing outliers. No cross-device performance claims are made.

The initial attention mapping with private dynamic arrays was replaced by a
four-query/sixteen-key workgroup tile with scalar FP32 accumulators. Narrow
dense matrices use 16×16 tiles before the 32×32 occupancy threshold. Mixed
pointwise kernels use padded NHWC only under explicit channel/spatial guards.

## Reproduction and evidence

See [GPU usage](gpu-usage.md) for commands and APIs. `tools/gpu_validation.py`
records complete operator/attribute inventories, input shapes, original tensor
names, hashes, adapter identity, selected kernels, errors, and raw F32 tensors.
`tools/gpu_ocr_smoke.py` checks the fixed sample. `gpu_compare ... measure`
requires the `profiling` feature and performs exactly one warmup and three
measurements; it reports both wall time and optional device timestamps.

Local generated evidence is under ignored `reports/native-wgpu/`:
`validated/`, `ocr-final/`, `reference-final/`, `pointwise-timestamps/`,
`operators-validated-*`, `integration-validated-*`, `regression-validated.txt`,
`regression-no-default-validated.txt`, and `borrow-contract/`. Earlier reports
are historical and are not substituted for these checks.

The full matrix comparison executable SHA-256 was
`ee95ce40cf8b217bc07e58ad928505a516f76e31dffbce612d240da7e03d75ec`. Each manifest pins its own executable and input hash;
the optional timestamp instrumentation was added to the developer example
subsequently, without changing inference kernels.

| Model | Source nodes | Model SHA-256 |
| --- | --- | --- |
| `cls` | 143 | `dd8b2b61983d76ab230a58da9e0e0e84956b71c3877f2ce6e438fe22d74d2cf2` |
| `det` | 242 | `193bab7a04fca699a6c82e6abb5b81bdb28177f0abd4062552b04908dafb19f8` |
| `small_det` | 242 | `d73e0058b7a8086bbd57f3d10b8bcd4ff95363f67e06e2762b5e814fe9c9410e` |
| `medium_det` | 276 | `eb13b44b25bb36f89528b68720af8a61d9cf381176107f465db1757b65d086e1` |
| `rec` | 219 | `9ef676d6ed3c88256a2d92c640c44f25b0c40947e111b14b8be8f594091563e6` |
| `small_rec` | 481 | `5435fd747c9e0efe15a96d0b378d5bd157e9492ed8fd80edf08f30d02fa24634` |
| `medium_rec` | 508 | `9c09abf0957f7968c7586464b7397b84ad2387a0497a351af40e9acc71b673ba` |
