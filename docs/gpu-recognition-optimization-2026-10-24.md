# GPU recognition optimization — 2026-10-24

The selected implementation reduces full-GPU OCR warm latency by **8.0% for
Tiny, 8.4% for Small and 15.7% for Medium** on the physical Apple M4. Dense
Tiny improves 7.7% at recognition batch two and 5.7% at batch eight.
Accounted GPU memory, transfer volume, submissions and dispatch counts remain
unchanged. All twelve final CPU/GPU OCR accuracy comparisons pass.

This implements the measured priorities from the
[stage audit](gpu-stage-audit-2026-10-24.md). The baseline is `a7930a4`; final
measurements describe the current working-tree changes, pinned by complete
source hashes and archives. No API, model, precision, batching or slot defaults
were changed.

The implementation has four parts:

1. **Pack shared pointwise weights across output rows.**
   [pointwise_blocked.wgsl](../src/runtime/gpu/pointwise_blocked.wgsl) retains
   its 32×32 output tile, 64 threads, K=16 and four-by-four register block.
   Shared weights now use K-major `vec4<f32>` entries containing the four
   output rows reused by a lane. Each reduction step reads one weight vector
   instead of four separated scalar entries. Global OI weight storage,
   accumulation order, bias, tails, output offsets and fused epilogues remain
   intact. Shared memory is still 4 KiB. No weight repacking allocation,
   layout conversion, extra capability or extra dispatch is needed.
2. **Use fast paths for measured recognition shapes.**
   [operators.rs](../src/runtime/gpu/operators.rs) admits narrow Metal FP32
   dense 1×1 H=1 maps to 16×16 tiling at 64 groups instead of the general
   128-group threshold. It requires stride one, unchanged spatial dimensions,
   input channels at least 128, output channels at least 64 and width at least
   64. Smaller and unrelated shapes keep their previous policy.
   [physical.rs](../src/runtime/gpu/physical.rs) additionally uses the packed
   pointwise kernel for Metal FP32 H=12, 512→256 plain Conv32 layers. Prescaled
   H=12 inputs keep their prior kernel; newly eligible H=1 channel gates retain
   the existing prescaled path.
3. **Tile RGB stem convolutions.**
   The existing implicit tiled shader already supports reduction length 27.
   `operators.rs` now admits ungrouped Metal FP32 RGB 3×3 stride-two stems with
   at least sixteen output channels. They prefer tile32 when the existing
   occupancy rules permit it, otherwise tile16. The 128 tile16 / 64 tile32
   group guards remain. Other kernels, input channel counts, strides and
   precisions retain their previous selection policy. This improves the
   measured detector and recognition stems.
4. **Skip unused Logit probability reduction.**
   [heads.wgsl](../src/runtime/gpu/heads.wgsl) and
   [heads4.wgsl](../src/runtime/gpu/heads4.wgsl) return after their existing
   winner/invalid reduction when the uniform mode flag selects Logit. They
   write the same zero sum field and avoid the probability sum section and
   seven barriers. The Probability tail is unchanged. Lowest-class ties,
   invalid flags, class/row tails and CTC error propagation are preserved.

The convolution policies are scoped to Metal FP32. The projection fast path is
shared WGSL; it was exercised on Metal with both F32 and F16 feature storage.
Vulkan, DX12 and other Metal devices were not locally qualified.

The physical machine was a Mac mini, Apple M4 with ten GPU cores and 16 GiB
unified memory, macOS 27, Metal, wgpu 30.0.1 and nightly-2026-09-08. Builds used
release mode, all features and locked dependencies. Full GPU used FP32, one
request, one slot, capture enabled and Logit recognition scores. Recognition
batch was two unless labeled eight. CPU options were four line workers, eight
detector threads and sixteen preprocessing workers.

Typical is the repository's 500×500 sample with sixteen lines. Dense is a
1000×1000 2×2 repetition with sixty-four lines. A rotated sample was used only
for correctness. These related fixtures do not represent a diverse corpus.

Each final timed configuration had one cold correctness/warm-up invocation,
three ordinary warm measurements and one separate profile. There were five
configurations: three typical presets and dense Tiny at batches two and eight.
Total final performance execution was **10.6 seconds**, including startup,
warm-ups, profiles and harness work, within a single declared 59-second budget.
Timings include worker launch through inference return, excluding decoding,
construction, owned-result conversion, validation, serialization and profiles.

| Workload | Before median ms | After median ms | Latency reduction | After warm samples ms |
|---|---:|---:|---:|---|
| Tiny, typical | 71.822 | 66.099 | 8.0% | 66.099, 65.909, 66.236 |
| Small, typical | 206.378 | 188.964 | 8.4% | 188.964, 189.002, 187.385 |
| Medium, typical | 723.023 | 609.571 | 15.7% | 608.610, 609.571, 620.321 |
| Tiny, dense, batch 2 | 228.800 | 211.270 | 7.7% | 211.159, 211.270, 212.311 |
| Tiny, dense, batch 8 | 209.215 | 197.323 | 5.7% | 197.193, 197.682, 197.323 |

All three final samples are below all three corresponding baseline samples.
These are small, sequential comparisons rather than a statistical claim across
machines or documents. Baseline and final harness, compiler, Cargo, models,
dictionaries, image/RGB/reference hashes and execution settings match.
Historical samples are preserved separately, not pooled with final samples.
The earlier CPU medians remain lower; CPU execution was not rebenchmarked in
this implementation pass. Batch eight remains an optional workload setting.

Focused comparisons guided the choices. Every candidate used one correctness
warm-up and three measured calls per shape. The following are wall times for
encoding, submission, completion and output readback, not pure kernel times.

| Focused shape | Previous kernel median µs | Selected kernel median µs |
|---|---:|---:|
| Pointwise `[1,1536,3,240]→768`, scalar shared weights → packed | 3562.834 | 2483.584 |
| Pointwise `[1,1024,6,240]→512`, scalar shared weights → packed | 3860.500 | 2217.250 |
| Pointwise `[2,192,6,136]→1024`, scalar shared weights → packed | 2156.125 | 1779.875 |
| H=12 `[1,512,12,128]→256`, implicit32 → packed pointwise | 2204.625 | 1372.333 |
| H=1 `[1,768,1,124]→192`, generic → direct16 | 1981.500 | 682.167 |
| H=1 `[1,160,1,124]→160`, generic → direct16 | 647.292 | 395.041 |
| RGB `[1,3,512,512]→16`, generic → implicit32 | 2753.417 | 1059.584 |
| RGB `[2,3,48,256]→24`, generic → implicit32 | 880.875 | 432.750 |
| RGB `[2,3,48,256]→64`, generic → implicit32 | 827.291 | 612.083 |

These rows belong to separately frozen experimental versions. They are not
additional final OCR repetitions. Timestamp availability was sparse in some
microbenchmarks, so wall samples and numerical checks guided selection.
The last RGB tile16/tile32 wall ranges overlap; tile32 has the lower median
and available device times on that case, while the final OCR comparison
validates the combined policy.

The isolated projection experiment improved scalar-head and large-inner-head
wall medians but did not establish a gain for the four-row 80-feature case.
Final projection stage results are similarly modest/noisy; no universal
projection speedup or exact contribution to OCR gain is claimed.

An explicit full-K unrolling candidate was rejected: it was approximately
17–28% slower in wall time on all three tested hot pointwise shapes. The first
H1/H12/projection integration also had mixed OCR latency results, so it was
not used as proof of the final gain. The shared-weight and RGB experiments
followed it. Those earlier binaries, sources and results remain separate;
no rejected shader or temporary kernel kind remains in production.

The final separate profiles support the neural-kernel priorities:

| Device graph time ms | Tiny before → after | Small before → after | Medium before → after |
|---|---:|---:|---:|
| Recognizer | 40.527 → 37.175 | 152.626 → 133.017 | 556.215 → 450.567 |
| Detector | 11.339 → 10.933 | 27.575 → 26.643 | 127.900 → 121.753 |
| Recognition RGB stem, original node 0 | 2.025 → 0.840 | 4.325 → 1.632 | 5.777 → 2.061 |

Medium recognition's original H=12 nodes 83/112 fall from 19.116/19.274 ms to
7.718/7.733 ms across the page. Narrow H=1 nodes use direct16 where the new
guard permits it; smaller widths remain generic. These profiles do not
isolate each change's wall-time contribution. All neural and projection
timestamp records are present; geometry/image metadata/CTC have unavailable
records. Final missing counts are 6/5/8 for typical Tiny/Small/Medium and
22/12 for dense batch two/eight. Missing entries are excluded, not measured as
zero. Profiling changes Metal pass grouping, so device sums cannot be added
or subtracted to decompose ordinary warm wall latency.

Corresponding before/after runs have exactly identical retained and peak
accounted GPU bytes, upload/readback volume, submissions, dispatches and
scratch-allocation counts. Typical retained/peak MiB are 36.86/41.87,
81.19/91.35 and 233.64/247.29. Dense values are 101.16/118.95 at batch two and
108.90/126.68 at batch eight. Counters exclude opaque driver allocations and
do not measure register pressure or physical memory traffic. Every measured
Full call has zero layout conversions, CPU compute stages, diagnostic tensor
downloads and stage fallbacks.

Final local validation:

- Release library suite: **392 passed**, 58 intentionally ignored.
- **Nine focused physical Metal tests passed**, covering shared-weight tails,
  bias/no bias, destination offsets, residual/activation and channel-gate
  fusions, real production H1/H12/RGB selection and exclusion boundaries,
  asymmetric RGB padding, existing resident fusions and mixed/subgroup paths.
  Projection cases cover scalar/F16/four-row paths, row/class tails, ties
  across class tiles, Probability scores and NaN/±Inf rejection through CTC.
- **Twelve CPU/GPU OCR comparisons passed:** all three presets on typical,
  dense and vertical in FP32, plus all three on typical with mixed-f16
  requested. Full DET/REC retain F32 under the existing precision policy.
  Text, counts/order, rotation, recognition widths/time steps, CTC spans and
  scores pass the existing gates; maximum box-coordinate error is zero.
  FP32 score gates use `abs_error <= 0.001 + 0.001*abs(reference)`.
- All five final timed configurations pass CPU-reference checks, exact warm
  output repeatability and profiled-output equivalence. Semantic `errors`
  and `qualified` fields were explicitly inspected.
- Strict release Clippy passed for all features/all targets and for serial
  GPU without default features/profiling. Formatting and whitespace checks
  passed. All checks ran on the physical machine, without CI or remote runners.

The prior audit's tiny profiling-availability test was not rerun as an accuracy
gate. Timestamp incompleteness remains a disclosed measurement limitation;
ordinary latency results do not depend on a positive complete timestamp sum.

Evidence is preserved in the ignored directory
`reports/gpu-recognition-optimization-2026-10-24/`:

- `accepted-typical/`, `accepted-dense/`, `accepted-dense-batch8/`: final raw
  samples, profiles, frozen executable, source archive, exact commands and
  hashes; `accepted-summary.json` extracts the baseline comparison.
- `accepted-qualification/`, `accepted-qualification-mixed/`: final outputs,
  input/model hashes, source provenance and correctness manifests.
- `accepted-*.log`, `accepted-focused-checks.json`: final build, lint and test
  evidence; `accepted-performance-budget.json`: cumulative performance budget.
- `candidates/`, `shared-weight-candidate/`, `rgb-candidates/`: separately
  frozen kernel comparisons; the earlier `final-*` directories record the
  first, superseded integration, not the accepted implementation.

All 162 source/build-input hashes in each accepted performance manifest match
the working tree, as do frozen/current executable hashes and source archives.
The final qualification manifests also match current sources and OCR binary.
Models, dictionaries, inputs and CPU references match the baseline audit.
Each experimental source version stayed below its bounded performance budget;
unchanged versions were not rerun for more favorable numbers.
