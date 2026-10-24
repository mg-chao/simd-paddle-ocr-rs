GPU pointwise matrix optimization — 2026-10-24

The default Metal GPU path now uses FP32 cooperative matrix instructions for
eligible recognition pointwise convolutions. On the physical Apple M4, the
measured Full OCR median fell by **24.5% for Medium** and **12.8% for Small**.
Tiny improved by 3.2% in this small sample. Selected layers also avoid duplicate
packed weights, reducing Medium's measured peak device allocation by 48.5 MiB.

This implements the first priority from the
[cross-backend analysis](gpu-cross-backend-opportunities-2026-10-24.md).
FP32 weights, activations and accumulation are retained. A native Metal backend,
FP16 conversion and a new activation layout were not needed for this improvement.

**Implementation.**

The original hot kernels used scalar/vector FP32 register blocks. The new
`pointwise_cooperative32` kernel expresses the same NCHW operation as
`weights[Cout, Cin] × input[Cin, H×W]`. Each subgroup accumulates sixteen 8×8
matrix fragments, making one 32×32 output tile. It loads operands directly from
canonical device buffers; shared memory is used only to initialize bias and
apply the existing fused epilogue. Residuals, activations, batches and output
offsets retain their existing behavior.

Two logical tiles are assigned to each 64-thread workgroup. The shader handles
one subgroup with two phases, masks extra subgroups, and masks the unused tile
when the tile count is odd. A uniform workgroup load establishes the control
flow required by WGSL matrix operations and barriers. Scratch reserves sixteen
64-float slots, covering subgroup widths from 4 to 64. The legacy experimental
MatMul scratch was also enlarged to cover that minimum width safely.

The old capability check rejected this M4 because it advertises a minimum
subgroup size of four. Device initialization now accepts the supported range
and verifies non-saturating FP32 8×8×8 matrix operands and accumulators before
requesting wgpu's experimental cooperative feature.

Selection is deliberately limited to the existing Metal packed/register
pointwise policy, with at least 128 input channels, 64 output channels and
64 spatial elements. Input channels, output channels and spatial elements
must all be divisible by eight, and all bound tensors must have canonical
FP32 storage. A partial 32×32 tile is supported; an incomplete 8×8 fragment
uses the existing kernel. Prescaled convolutions, mixed-storage plans and other
backends retain their previous selection. There is no startup benchmark.

`GpuContextOptions::cooperative_pointwise` defaults to `true`; setting it to
`false` disables this path. Generic MatMul remains controlled separately by
`experimental_cooperative_matrix`, which still defaults to `false`. No explicit
option is needed for the measured Full OCR improvement. See
[gpu-usage.md](gpu-usage.md) for the public configuration.

Implementation files:
[gpu.rs](../src/runtime/gpu.rs),
[physical.rs](../src/runtime/gpu/physical.rs),
[operators.rs](../src/runtime/gpu/operators.rs),
[pointwise_cooperative.wgsl](../src/runtime/gpu/pointwise_cooperative.wgsl), and
[pipelines.rs](../src/runtime/gpu/pipelines.rs).

**Full OCR measurements.**

One 500×500 image, sixteen lines, one request, one GPU slot, recognition batch
two, CTC spans enabled, release build, FP32 Full placement. Each preset/version
has one cold warm-up and three ordinary measured calls. A separate profiled
call follows the timings. The baseline is the frozen, source-verified
[prior comparison](../reports/csharp-comparison-2026-10-24/performance/manifest.json);
it was not rerun. The new run uses identical image/RGB, model and dictionary
hashes, adapter and settings.

| Preset | Before median, ms | After median, ms | Latency reduction | Before samples, ms | After samples, ms |
|---|---:|---:|---:|---|---|
| Tiny | 63.925 | 61.875 | 3.2% | 63.953, 63.925, 63.911 | 62.231, 61.875, 61.602 |
| Small | 181.491 | 158.170 | 12.8% | 181.895, 180.087, 181.491 | 158.323, 157.858, 158.170 |
| Medium | 598.185 | 451.450 | 24.5% | 598.185, 602.398, 595.088 | 451.450, 453.128, 450.487 |

These are three-sample local observations, not confidence intervals or claims
for other devices. Tiny's small difference warrants particular caution. No
unchanged workload was repeated to obtain more favorable numbers.

The separate profiles confirm 86 / 269 / 286 matrix dispatches for
Tiny / Small / Medium. Total recognizer pointwise device time changed from
19.801 / 66.589 / 326.794 ms to 17.774 / 44.652 / 178.128 ms. Medium's full
recognizer device duration changed from 437.538 to 288.587 ms, while detector
duration stayed near 122 ms. Profiles split neural dispatches into individual
passes and are not additive components of the ordinary wall times. New profiles
have 8 / 6 / 9 missing timestamps elsewhere; missing values are not zeroes.

Measured peak device allocation fell by 1,499,136 / 9,510,912 / 50,855,936 bytes
(1.43 / 9.07 / 48.50 MiB). Retained allocation fell by the same amounts because
selected layers bind canonical weights instead of creating additional packs.
Normal samples retain zero CPU compute stages, zero diagnostic tensor downloads
and zero layout conversions.

**Focused kernel comparison.**

Three representative hot shapes were compared within the final binary, one
correctness warm-up and three measured calls per kernel/shape. Each call includes
allocation, submission, completion and readback; these are wall medians, not
pure device execution durations. CPU scalar agreement is checked before timing.

| Input N×Cin×H×W → Cout | Packed kernel, μs | Matrix kernel, μs | Wall reduction |
|---|---:|---:|---:|
| 1×1536×3×240 → 768 | 3053.708 | 1561.459 | 48.9% |
| 1×1024×6×240 → 512 | 2940.666 | 1563.000 | 46.8% |
| 2×192×6×136 → 1024 | 2169.375 | 1656.834 | 23.6% |

Some matrix timestamp queries were unavailable, so no device-only median is
inferred from the incomplete samples. The kernel and full OCR timing harnesses
together consumed 6.10 seconds for the final code version, including warm-ups
and diagnostic calls, below the 60-second execution budget. Build time is
excluded. Earlier prototypes were rebuilt and recorded separately; their
samples are not pooled with these results. Shared-memory staging and 16×16
matrix tiles were slower, while 64×64 tiles were less consistent than 32×32.
Only the selected 32×32 implementation remains in production code.

**Local correctness and build checks.**

- 393 release library tests passed; 66 explicitly ignored tests were not part
  of that general run.
- Seven focused physical-GPU tests passed: matrix selection/fusion, matrix
  tails/subgroups, packed-weight tails, register-tile tails/offsets, register
  fusion/fallback, recognition-policy boundaries and the legacy experimental
  MatMul/fusion path. The new direct kernel uses three tail fixtures, with
  32/64/128-thread test workgroups to exercise one/two/four actual M4 subgroups;
  production uses 64 threads. Scalar tolerance remains `1e-4 + 1e-4×|reference|`.
- The selection test verifies the default path, explicit opt-out, an unaligned
  fallback, and that enabling pointwise capability does not select experimental
  generic MatMul. A device-independent check rejects incomplete fragments and
  mixed storage, and verifies an odd tile grid.
- Nine FP32 Full OCR cases passed: all three presets on typical, dense and
  rotated images. Three additional Full cases passed with a MixedF16 context,
  one typical image per preset. Each compares against CPU output. Text, line
  counts, orientation, recognition dimensions and CTC spans agree; scores pass
  the existing tolerances, coordinates have zero observed error, and the Full
  residency checks pass. Full DET/REC continue to use FP32 in a mixed context.
- `cargo clippy --locked --all-features --all-targets -- -D warnings`,
  `cargo fmt --all --check`, `git diff --check`, and the locked release example
  builds passed. No CI, VM or remote runner was used.

The policy regression test initially expected the previous packed kernel for
three now-eligible aligned shapes. Those expectations were updated to the new
capability-dependent choice; scalar comparison was retained. No numerical
thresholds or OCR expectations were relaxed.

Validation is limited to the physical Apple M4/Metal machine. Other adapters
and operating systems have not been qualified by this work. The change narrows
the C# performance gap without establishing parity: C# uses different precision,
layout, scheduling and output contracts. Detector convolutions, remaining
portable pointwise layers and projection still account for substantial work.

**Reproduction and evidence.**

The [summary and audit](../reports/gpu-matrix-optimization-2026-10-24/summary.json)
records the comparisons and validation counts. It checks all 165 final timing
source/build-input hashes and all 163 kernel-input hashes against the working
tree, plus 161 shared starting-source hashes against the saved baseline.
The [audit script](../reports/gpu-matrix-optimization-2026-10-24/summarize.py)
only reads saved records; it does not rerun workloads.

- [Final OCR timing manifest](../reports/gpu-matrix-optimization-2026-10-24/performance/manifest.json), frozen executable and source archive in the same directory.
- [Final kernel manifest](../reports/gpu-matrix-optimization-2026-10-24/final-kernels/manifest.json), frozen test binary and source archive in the same directory.
- [FP32 qualification](../reports/gpu-matrix-optimization-2026-10-24/final-qualification/manifest.json) and [mixed-context qualification](../reports/gpu-matrix-optimization-2026-10-24/mixed-qualification/manifest.json), including per-case outputs and statistics.
- [Performance execution budget](../reports/gpu-matrix-optimization-2026-10-24/performance-budget.json).

Build with `cargo build --release --locked --all-features --example ocr --example gpu_ocr_bench`.
Qualification uses `tools/full_gpu_validation.py`; timings use
`tools/gpu_performance.py` with `--backend metal --presets tiny small medium
--cases typical --configs 1:1:2 --modes full --measurements 3 --alignment --profile`.
Pass the saved qualification corpus and a new report directory explicitly.
The ignored `gpu_pointwise_cooperative_candidates` test contains the bounded
three-shape kernel comparison.
