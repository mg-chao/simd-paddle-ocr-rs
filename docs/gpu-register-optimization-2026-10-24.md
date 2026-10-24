# GPU convolution optimization — 2026-10-24

The convolution changes reduce warm full-GPU OCR latency by **3.4% for Tiny,
10.1% for Small, and 20.1% for Medium** on the physical Apple M4. The same model,
FP32 precision, inputs, batching, and output checks are used before and after.
Accounted GPU memory usage and transfer volume are unchanged. These gains do
not yet make the GPU faster than the measured SIMD CPU path.

This follows the [current-backend review](gpu-current-review-2026-10-24.md).
The baseline is runtime revision `21971c5`; each experiment preserves its own
source hashes, and final results below belong to the combined implementation.

## Implementation

1. **Reuse pointwise operands in registers.**
   [pointwise_blocked.wgsl](../src/runtime/gpu/pointwise_blocked.wgsl) computes
   a 32×32 output tile with 64 threads and a reduction tile of 16. Each thread
   accumulates four rows of four adjacent columns using `vec4<f32>` values.
   Every loaded weight contributes to four outputs, and each feature vector
   contributes to four output channels. Workgroup storage remains 4 KiB, and
   no layout conversion is needed. Bias initialization, reduction order,
   fused epilogues, batch indexing, partial tiles, and destination offsets
   retain the existing semantics. The new pipeline is created lazily.
   [physical.rs](../src/runtime/gpu/physical.rs) selects it for eligible FP32
   Metal 1×1 convolutions with input height at most six and at least 128 input
   channels, after the existing 32×32 tile occupancy checks. Prescaled inputs,
   tall maps, other backends, and mixed-storage graphs retain their earlier
   pointwise policy.
2. **Use larger tiles for costly spatial convolutions.**
   [operators.rs](../src/runtime/gpu/operators.rs) makes the existing tiled
   shader available to dense 5×5 and 7×7 FP32 Metal convolutions. For 3×3,
   5×5, and 7×7 layers, 32×32 tiles are preferred when input channels are at
   least 32, output channels are divisible by 32, and output spatial size is
   at least 1024. Existing occupancy, stride, dilation, and resource checks
   still apply. This improves reuse in detector layers with only 32 or 64
   output channels; their large spatial maps supply sufficient workgroups.

The choices are automatic and require no API or configuration changes.
Models, recognition batch defaults, slot count, precision defaults, and CPU
execution are unchanged. Larger caches and more slots were not adopted because
the preceding review did not demonstrate a latency benefit from them.

## Measurements

Physical Mac mini, Apple M4 with 10 GPU cores and 16 GiB unified memory,
macOS 27.0.1 (26A434), Metal, wgpu 30.0.1, nightly-2026-09-08, locked release
builds with all features. Full-GPU FP32 OCR uses one request, one slot, and
CTC span capture. Recognition batch size is two except where explicitly eight.
CPU references use four line workers, eight detector threads, and sixteen
preprocessing workers.

Typical is the repository's 500×500 sample with 16 lines. Dense is a 1000×1000
2×2 repetition with 64 lines. These related fixtures do not constitute a
diverse document corpus. Each configuration has one cold correctness/warm-up
call, three warm measurements, and one separate profile call. Warm wall time
includes OCR execution and worker launch; it excludes image decoding, model
construction, owned-result conversion, validation, serialization, and profiling.

| Workload | Before, median ms | After, median ms | Latency reduction | After warm samples, ms |
|---|---:|---:|---:|---|
| Tiny, typical | 72.510 | 70.058 | 3.4% | 70.098, 70.058, 69.622 |
| Small, typical | 227.325 | 204.304 | 10.1% | 204.576, 204.135, 204.304 |
| Medium, typical | 897.951 | 717.252 | 20.1% | 717.252, 718.639, 712.787 |
| Tiny, dense, batch 2 | 241.154 | 233.640 | 3.1% | 234.872, 233.640, 226.750 |
| Tiny, dense, batch 8 | 220.275 | 208.888 | 5.2% | 209.064, 208.888, 208.267 |

Every row compares identical settings. The dense batch-two samples have an
8.1 ms range; the magnitude of that modest improvement is uncertain. Batch
eight remains an optional workload-specific setting: its lower dense latency
does not justify increasing the universal default. No unchanged version was
rerun to obtain more favorable samples.

The preceding SIMD CPU medians were 49.97, 124.05, and 351.04 ms for typical
Tiny, Small, and Medium, and 173.17 ms for dense Tiny. They remain lower than
these GPU results; CPU code was not changed or rebenchmarked in this pass.

### Evidence for the kernel choices

Focused kernel experiments used one correctness/warm-up call and three
measured calls per candidate and shape. Wall time includes command creation,
submission, completion, and output readback; these are not device-only times.
They are separate source versions from the final whole-pipeline measurements.

| Kernel case | Previous median µs | Selected median µs |
|---|---:|---:|
| Pointwise `[1,1536,3,240] → 768` | 5078.083 | 2793.958 |
| Pointwise `[1,1024,6,240] → 512` | 4018.625 | 2496.291 |
| Pointwise `[2,192,6,136] → 1024` | 2149.792 | 1852.042 |
| 3×3, `[1,256,128,128] → 64`, tile 16 → 32 | 22666.292 | 12405.333 |
| 5×5, `[1,32,128,128] → 32`, generic → tile 32 | 9188.500 | 2757.375 |
| 7×7, `[1,32,128,128] → 32`, generic → tile 32 | 16043.625 | 4720.542 |

For the final Medium profile, recognizer device time falls from 699.20 to
551.27 ms and detector time from 163.63 to 124.09 ms. Original detector ONNX
nodes 268, 246, 251, and 8 fall from 21.83/15.40/8.48/10.91 ms to
11.60/4.12/2.09/5.86 ms, respectively. These measurements support the selection
policy, but individual profiles do not establish exact contributions to the
wall-time gain.

Profiling splits execution into measured passes and changes scheduling.
Profile sums are not an additive decomposition of ordinary warm latency.
Final Tiny/Small/Medium typical profiles have 9/5/4 unavailable timestamps;
dense batch-two/eight profiles have 21/9. Missing durations are omitted from
available-time summaries, not treated as measured zero. The quoted Medium
detector-node durations are all present.

### Memory and transfers

All five configurations retain exactly the same accounted device bytes and
peak bytes as their corresponding baseline. Typical Tiny/Small/Medium retain
36.86/81.19/233.64 MiB, with peaks of 41.87/91.35/247.29 MiB. Dense Tiny retains
101.16 MiB at batch two and 108.90 MiB at batch eight. Accounting excludes
opaque driver allocations and does not measure register pressure.

Upload, compact output/control readback, dispatch, and submission counts are
unchanged. Each measured full-GPU run reports zero CPU compute stages, zero
diagnostic tensor readbacks, and zero layout conversions. The speedup therefore
comes without extra host transfers or increased accounted buffer capacity.

## Local validation and provenance

All validation ran on the physical machine. Focused correctness preceded
performance measurements, and every measured source version was rebuilt.

- Release library suite: **392 passed**, 50 intentionally ignored.
- **Six focused physical GPU tests passed**, covering register-block tails,
  bias/no bias, output offsets, fused residual/activation, selection boundaries,
  3×3/5×5/7×7 asymmetric padding and stride, existing resident fusions, mixed
  storage/subgroups, and resident-storage/reshape invariants.
- **Twelve CPU/GPU OCR comparisons passed**: three presets × three fixtures
  (typical, vertical, dense) in FP32, plus three presets on typical with
  mixed-f16 requested. Maximum box-coordinate difference was zero. Text,
  line order/count, rotations, widths, CTC steps/spans, and confidence gates
  passed with the existing tolerances. Full detector/recognizer precision
  policy remains unchanged when mixed-f16 is requested.
- Strict release Clippy passed for all features/all targets and for serial
  GPU without default features/profiling. The benchmark's profile-vector
  declaration now avoids an unused-mut warning when profiling is disabled.
  Formatting and whitespace checks passed.

Final performance execution took **12.15 seconds**, excluding builds: five
configurations, five cold/warm-up calls, fifteen warm samples, and five separate
profiles. Earlier pointwise and spatial experiments took 0.28 and 1.20 seconds,
respectively; the separately recorded pointwise-only integration pass took
12.05 seconds. Each source version stayed below the 60-second performance
budget. Correctness runs are not used as latency samples.

Reproduction commands and full evidence are in the ignored local directory
`reports/gpu-register-optimization-2026-10-24/`:

- `final-typical/`, `final-dense/`, and `final-dense-batch8/`: raw measurements,
  profiles, frozen executables, source archives, input/model/dictionary hashes,
  and exact commands in each `manifest.json`.
- `final-qualification/` and `final-qualification-mixed/`: correctness outputs
  and manifests; `*-final.log`: unit and focused GPU test logs.
- `final-build-checks.json`: successful build, lint, format, and whitespace
  checks after fixing the benchmark warning; `final-performance-budget.json`:
  execution budget accounting; `summary.json`: baseline comparisons.
- `register-*`, `spatial-*`, `typical/`, `dense/`, and `dense-batch8/` preserve
  earlier experiments separately; their samples are not pooled into final results.

All 162 source hashes in each final performance manifest match the working
tree; frozen executable and archive hashes match as well. The 157 production
source hashes in each final correctness manifest also match. Baseline/current
input, model, dictionary, and execution settings were checked for equality.

The measurements qualify these choices on Apple M4 only. Other Metal devices
and a broader document corpus still need local qualification; Vulkan and DX12
were not exercised in this pass and retain their prior selection policy.
