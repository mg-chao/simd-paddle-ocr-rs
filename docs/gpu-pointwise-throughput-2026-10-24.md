GPU pointwise throughput optimization — 2026-10-24

The selected changes reduce Full GPU OCR warm latency by **3.3% for Tiny,
4.9% for Small and 2.1% for Medium** on the physical Apple M4. Dense Tiny
improves **5.2% at batch two and 3.7% at batch eight**. All twelve final
CPU/GPU OCR comparisons pass. The improvement costs additional shared immutable
GPU weights: 1.43 / 9.07 / 48.50 MiB for the typical presets.

This implements the measured priorities in the
[backend evaluation](gpu-backend-evaluation-2026-10-24.md). Baseline is `f902d85`;
the selected working tree is pinned by source hashes, source archives and frozen
executables. CPU code, model files, score/precision policies, batching, GPU-slot
defaults and public APIs are unchanged. The SIMD CPU remains faster in the
preceding measurements; these gains do not establish GPU/CPU parity.

The implementation makes three changes:

1. **Load immutable pointwise weights as contiguous vectors.**
   [gpu.rs](../src/runtime/gpu.rs) lazily creates a K-major pack with four
   adjacent output channels per vector. It retains canonical weights for
   other kernels and diagnostics. Packs are shared by compiled views and
   requests using the same model/context/precision, with at most one pack per
   constant. Padding covers channel tails. The normal device allocation ledger,
   quota, lifetime and unique-weight accounting include each pack. Statistics
   expose `packed_weight_bytes` and mark the selected kernel as packed.

   [pointwise_weights_packed.wgsl](../src/runtime/gpu/pointwise_weights_packed.wgsl)
   loads one vector instead of gathering four separated scalars. Both weight
   layouts share [pointwise_blocked.wgsl](../src/runtime/gpu/pointwise_blocked.wgsl),
   retaining the 32×32 output tile, K=16, 64 threads, four-by-four register block,
   accumulation order, bias handling, destination offsets and fused epilogues.
   Selection is limited to constant-weight Metal FP32 pointwise layers on the
   existing short feature maps, plus the measured H12 192→96 geometry. Dynamic
   weights, mixed storage and other backends retain their prior policy.
2. **Promote measured mid-channel maps to register tiles.**
   [operators.rs](../src/runtime/gpu/operators.rs) permits tile32 for dense
   stride-one, zero-padding 192→96 pointwise maps at height six or twelve,
   retaining the minimum of 64 tile32 groups. These are actual Tiny and Small
   recognition layers, previously selected for tile16 because output channels
   were below 128. [physical.rs](../src/runtime/gpu/physical.rs) uses direct
   NCHW addressing and the packed kernel for eligible plain convolutions.
   Existing prescaled paths preserve their arithmetic and fusion.
3. **Cover the remaining narrow H1 shapes.**
   The Metal FP32 H1 pointwise threshold falls from 64 to 48 tile16 groups.
   Existing guards remain: dense 1×1, stride one, no spatial padding, input
   channels at least 128, output channels at least 64 and width at least 64.
   For example, `[1,384,1,104]→120` has 56 groups and
   `[1,768,1,80]→192` has 60; both previously used the generic convolution.
   Smaller and unrelated cases keep their earlier choices.

All execution ran locally on a physical Mac mini with Apple M4, ten GPU cores,
16 GiB unified memory, macOS 27, Metal, wgpu 30.0.1 and nightly-2026-09-08.
Builds used release mode, all features and locked dependencies. Full GPU used
FP32, one request, one slot, span capture and default Logit scores. Recognition
batch was two unless specified. CPU options were four line workers, eight
detector threads and sixteen preprocessing workers. Optional subgroup,
cooperative-matrix and scratch-cache settings remained off.

Typical is the 500×500 repository sample with sixteen lines. Dense is its
1000×1000 2×2 repetition with sixty-four lines. A rotated fixture was used for
correctness. These related inputs are a limited corpus.

Each selected timed configuration used one cold correctness/warm-up invocation,
three ordinary latency samples and one separate diagnostic profile. Five
configurations took **8.29 seconds** including startup and harness work, within
one declared 59-second budget. Builds and prior correctness checks are excluded.
The interval includes worker launch through inference return, excluding image
decode, construction, owned-result conversion, validation, serialization and
profiles. No unchanged configuration was rerun for more favorable numbers.

| Full GPU workload | Baseline median ms | Selected median ms | Reduction | Selected warm samples ms |
|---|---:|---:|---:|---|
| Tiny, typical | 66.934 | 64.755 | 3.3% | 64.755, 64.067, 65.008 |
| Small, typical | 191.568 | 182.133 | 4.9% | 182.133, 183.824, 182.043 |
| Medium, typical | 617.220 | 604.136 | 2.1% | 604.136, 605.640, 601.660 |
| Tiny, dense, batch 2 | 218.603 | 207.323 | 5.2% | 207.323, 208.959, 203.042 |
| Tiny, dense, batch 8 | 198.950 | 191.557 | 3.7% | 193.659, 188.809, 191.557 |

Every selected sample is below every corresponding baseline sample. These are
small sequential comparisons on one machine, not confidence intervals or a
cross-device claim. Dense sample ranges remain several milliseconds wide.
The baseline is the immediately preceding audit, preserved separately; samples
from earlier optimization reports are not pooled into this comparison.

Focused comparisons preceded integration. Each candidate/shape had one
correctness warm-up and three measurements. The following are wall medians for
encoding, submission, completion and readback, not pure kernel durations.
Packing itself occurred outside the measured kernel invocation.

| Focused input → output channels | Previous median µs | Candidate median µs | Candidate |
|---|---:|---:|---|
| `[1,1536,3,240]→768` | 3135.958 | 2737.167 | Packed global weights |
| `[1,1024,6,240]→512` | 2691.209 | 2551.291 | Packed global weights |
| `[2,192,6,136]→1024` | 1793.916 | 1710.500 | Packed global weights |
| `[1,384,1,104]→120` | 1180.000 | 668.750 | Generic → direct16 |
| `[1,768,1,80]→192` | 1778.958 | 598.125 | Generic → direct16 |
| `[1,160,1,84]→160` | 614.084 | 382.334 | Generic → direct16 |
| `[1,192,6,248]→96` | 1199.209 | 888.917 | Direct16 → packed32 |
| `[2,192,6,136]→96` | 1269.917 | 571.291 | Direct16 → packed32 |
| `[1,192,12,216]→96` | 1192.500 | 682.250 | Direct16 → packed32 |

Timestamp coverage was sparse in these short experiments; the mid-channel and
narrow experiments had no available device durations. Their wall samples guide
candidates rather than establish hardware bottlenecks. For mid-channel maps,
unpacked register32 also beat direct16 and was faster than packed32 in the
first case. That mixed result motivated a separate whole-OCR comparison.

A load-only lane remapping was rejected as inconclusive: it helped one shape
but did not establish a consistent gain on the large reductions. An alternate
whole-OCR build kept the new selection rules and used unpacked register32.
Its typical medians were 64.586 / 182.765 / 613.880 ms. Tiny and Small overlap
the selected version's ranges; Medium's three alternate samples were all
above the selected samples. The packed version lowers Medium latency by a
further 1.6%, at its disclosed memory cost. There is no claim that packing
alone provides a material Tiny/Small OCR speedup.

Selected separate profiles put recognizer graph device time at
36.249 / 128.438 / 442.943 ms, versus 37.286 / 141.280 / 454.463 ms in the
baseline. Normal execution and profiling group GPU passes differently, so
these are not additive decompositions of ordinary latency or isolated
contributions from each change. All neural and projection records are present;
missing image metadata, geometry and CTC records total 6/5/5 for typical
Tiny/Small/Medium and 18/11 for dense batch two/eight. Missing durations are
excluded, not treated as measured zero.

| Selected Full configuration | Retained / peak MiB | Increase over baseline MiB |
|---|---:|---:|
| Tiny typical | 38.29 / 43.30 | 1.43 |
| Small typical | 90.26 / 100.42 | 9.07 |
| Medium typical | 282.14 / 295.79 | 48.50 |
| Tiny dense b2 | 102.59 / 120.38 | 1.43 |
| Tiny dense b8 | 110.42 / 128.21 | 1.53 |

Packs add model-shared storage rather than per-request activation storage.
Warm upload payloads, compact output/control bytes, submissions, dispatches,
scratch-allocation counts and layout-conversion counts match the baseline.
Packing does incur an initial immutable upload, which the existing per-run
payload counters exclude. Application accounting excludes opaque driver
allocations. All selected Full runs report GPU placement, no stage fallback,
no CPU compute stages, no diagnostic tensor download and no layout conversion.

Final validation on the physical machine:

- Release library suite: **392 passed**, 63 intentionally ignored.
- **Nine focused GPU tests passed**, covering packed weights and tails,
  bias/no bias, offsets and fused epilogues, real production selection and
  exclusion boundaries, channel-gate paths, concurrent cache sharing, quota
  failure without reservation leakage, unique ownership accounting, resident
  fusions, mixed/subgroup paths and reshape/storage invariants.
- **Twelve CPU/GPU OCR comparisons passed:** three presets × typical/dense/
  vertical in FP32, plus all three presets on typical with mixed-f16 requested.
  Full DET/REC retain F32 under the existing policy. Text, counts/order,
  rotation, recognition widths/time steps, CTC spans and score gates passed;
  maximum coordinate difference was zero.
- All five selected benchmark configurations passed CPU-reference checks,
  exact warm-output repeatability and profile-output equivalence. The three
  alternate configurations also passed their CPU/reference and repeatability
  checks. Semantic `qualified` and `errors` fields were inspected.
- Strict release Clippy passed for all features/all targets and for serial
  GPU without default features/profiling. Formatting and whitespace checks
  passed. No CI, remote runner or virtual machine was used.

An initial broad test-name filter also selected an unrelated ignored AVX2 CPU
benchmark, which refused this ARM64 host before measuring. Final focused GPU
filters exclude it. Initial compilation/lint issues in the new test harness
were corrected before final qualification and measurements. The earlier
profiling-availability issue is not represented as resolved.

Local evidence is preserved in the ignored directory
`reports/gpu-throughput-optimization-2026-10-24/`. `final-typical/` and
`final-dense/` hold the selected measurements, source archives, frozen binaries,
hashes and exact commands. `final-qualification/`, `final-qualification-mixed/`
and `final-check-*.log` hold final accuracy, unit-test and lint evidence.
`final-performance-budget.json` and `run_final_performance.py` preserve the
selected execution scope. `weight-load-candidates/`,
`packed-and-narrow-candidates/`, `mid-channel-candidates/` and
`unpacked-policy-typical/` preserve separate experimental versions. Earlier
`qualification*` directories predate test-harness/comment cleanup and are not
used as final provenance. No rejected lane-remapping or temporary selection
override remains in production.

`summary.json` records extracted comparisons and `final-validation.json`
records the completion checks. All 164 source/build-input hashes in each
selected performance manifest match the current tree; frozen/current binaries,
source archives, final accuracy build inputs, models, dictionaries, baseline
inputs and benchmark settings were independently checked for equality.

The changes are qualified on this M4 only. Other Metal devices, Vulkan, DX12,
a broader document corpus and deployment-specific memory limits need their
own local qualification. Other backends retain their earlier selection policy.
