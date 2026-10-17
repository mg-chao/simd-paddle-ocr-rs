# Current Windows OCR latency and optimization analysis

For the newer `38e054c` implementation on a 24-thread context, see the
[current CPU24 stage analysis](windows-cpu24-stage-analysis.md). This report
retains the earlier `061c33b` eight-CPU measurements; the two experiments must
not be pooled.

The current implementation still has useful optimization targets, concentrated in detector compute for sparse pages and recognition compute for dense pages. Dense Small and Medium spend **78.0% and 89.5%** of profiled elapsed time in recognition. Current recognition backbone probes attribute **79.7% and 90.0%** of physical-step time to 1×1 convolutions. Orientation classification is also substantial for Tiny: **24.97 ms, or 36.0%**, on the dense fixture.

These results describe production revision `061c33b60fd9d14f63119f19552bdc0278e2f083` on 2026-10-17, after the recognition scheduling, short-reduction pointwise and detector upsampling changes. The older initial tables in [the original stage analysis](windows-cpu-stage-analysis.md) describe `4c88abc` and are separate experiments. No production code or defaults changed during this analysis.

The full measurements, source/executable hashes, jobs, output checks, retained capacities and raw recognition step logs are in [the evidence JSON](reports/windows-x64-2026-10-17-current-pipeline-analysis.json). Local runner and logs are in `target/current-pipeline-analysis-20261017/`.

## Measurement conditions

All validation ran locally on the physical Windows machine: AMD Ryzen 9 5950X, 12 enabled cores and 24 logical processors exposed by Windows, approximately 64 GiB RAM. Page runs used eight logical CPUs pinned with affinity `0xff`, one request, release builds, automatic SIMD/layout, fusion, detector limit 960, adaptive recognition widths, batch size one, classification enabled and alignment capture disabled. Eight logical CPUs do not imply eight separate physical cores.

Three inputs cover different bottlenecks: sparse `imgs_1` (720×1150, three lines), dense short text `imgs_00018069` (430×267, 69 lines), and mixed widths `imgs_00006737` (896×528, 31 lines). Small and Medium use the same dense input to answer whether increasing model size changes the dominant stage. Each model uses its pinned dictionary and detector threshold; model-size comparisons are workload costs, not an accuracy comparison.

Uninstrumented OCR timings include conversion to a complete owned result from predecoded BGR24 pixels. Five separate profiles measure the borrowed-result production path with stage timers. Each profile has one warm-up and one measured call; profiles are localization evidence, not distributions of stage latency. Worker counters sum overlapping elapsed calls and include nested kernel waits. They are neither process CPU time nor extra serial stages.

## Complete OCR latency and cold setup

| Model and input | Three measured owned-result calls, ms | Median, ms | Import, compile and request setup, ms | First inference, ms |
|---|---|---:|---:|---:|
| Tiny sparse | 45.1623, 43.6833, 43.9571 | **43.9571** | 16.4914 | 51.9587 |
| Tiny dense | 69.3443, 69.2114, 68.4106 | **69.2114** | 15.8334 | 73.6617 |
| Tiny mixed | 80.9221, 80.9112, 81.7919 | **80.9221** | 15.5421 | 89.5197 |
| Small dense | 198.1405, 197.8247, 199.3953 | **198.1405** | 59.2740 | 205.9808 |
| Medium dense | 757.3393, 764.9710, 760.9956 | **760.9956** | 254.7887 | 781.9464 |

The Tiny rows reuse the classification-enabled measurements in [the current-revision orientation comparison](reports/windows-x64-2026-10-17-no-cls-latency.json). Its complete source digest, production executable and original manifest match the current files, so unchanged code was not timed again. Small and Medium were measured once each with the same production executable and settings. Each ordinary worker records one first inference, one warm-up and exactly three measured calls; complete outputs equal its first-inference control.

Setup is separate from first inference and measured inference. Reloading models and rebuilding the full pipeline for every image can therefore add a material cost, especially for Medium. Keep models and requests alive, or use an appropriately retained `OcrEngine` lease. Creating or forking another request from shared compiled models does not necessarily repeat this whole cost. These setup samples include local file reads with uncontrolled filesystem-cache state; they are not cold-storage measurements. Parsing, normalization, packing and request construction are aggregated.

## Elapsed time of each pipeline stage

All values below are milliseconds from one current-revision instrumented sample per case.

| Elapsed stage | Tiny sparse | Tiny dense | Tiny mixed | Small dense | Medium dense |
|---|---:|---:|---:|---:|---:|
| DET resize and normalization | 3.0681 | 0.3118 | 1.3930 | 0.4201 | 0.3983 |
| DET inference | **37.2592** | 8.7833 | 28.1919 | 16.9255 | 50.6162 |
| DB boxes and reading order | 0.6622 | 0.5678 | 0.7964 | 0.4414 | 0.4727 |
| Crop planning and extraction | 1.0311 | 1.3887 | 2.6994 | 1.2908 | 1.2682 |
| Classification and orientation wave | 2.7485 | **24.9667** | 11.3250 | 24.8446 | 25.1707 |
| Recognition wave and assembly | 9.3459 | **33.3801** | **41.1401** | **155.8152** | **664.5799** |
| Unattributed elapsed remainder | 0.0636 | 0.0482 | 0.0623 | 0.0657 | 0.0614 |
| Profiled borrowed-result total | **54.1786** | **69.4466** | **85.6081** | **199.8033** | **742.5674** |

The six named stages execute in order. The remainder contains orchestration and timer boundaries, including some binding and pool-entry work. Classification completes before recognition starts. Classification includes resize, fixed-shape backbone, two-class head and optional in-place 180° rotation; the current timers do not split those operations. Crop planning includes geometric rotation for vertical crops, which remains separate from optional orientation classification.

The sparse profile total is higher than the independently measured 43.96 ms owned-result median. Different executions and instrumentation need not agree; this single sample is retained without retries. Its DET inference still dominates at 68.8%. Do not subtract the two totals to estimate owned-copy overhead, rescale stage counters to a median, or average instrumented and ordinary measurements.

| Share of profiled total | Tiny sparse | Tiny dense | Tiny mixed | Small dense | Medium dense |
|---|---:|---:|---:|---:|---:|
| DET inference | 68.8% | 12.6% | 32.9% | 8.5% | 6.8% |
| Classification | 5.1% | 36.0% | 13.2% | 12.4% | 3.4% |
| Recognition | 17.3% | 48.1% | 48.1% | 78.0% | 89.5% |

## Work inside parallel stages

These milliseconds sum owner-call durations and overlap within their elapsed wave. **Do not add this table to the elapsed-stage table or divide every entry by eight to infer latency.** Ownership, widths, nested threads and end-of-wave imbalance differ between inputs.

| Summed elapsed work | Tiny sparse | Tiny dense | Tiny mixed | Small dense | Medium dense |
|---|---:|---:|---:|---:|---:|
| Crop workers | 3.5985 | 8.9338 | 11.7357 | 8.7501 | 8.9635 |
| Classifier and rotation workers | 6.0739 | 189.0141 | 85.0514 | 186.8030 | 187.8402 |
| REC resize and normalization | 0.6998 | 5.4728 | 7.0727 | 5.1572 | 5.4992 |
| REC backbone | **16.5581** | **230.6676** | **247.3159** | **1095.0800** | **4987.5149** |
| Vocabulary projection and argmax | 1.2582 | 20.7059 | 21.2522 | 91.2318 | 158.9378 |
| CTC text decoding | 0.0059 | 0.0362 | 0.0296 | 0.0756 | 0.1743 |
| DET and REC reshape calls | 0.0023 | 0.0533 | 0.0492 | 0.0911 | 0.1285 |
| Binding and preparation | 0.0081 | 0.0210 | 0.0141 | 0.0372 | 0.0460 |
| Result storage | 0.0111 | 0.0271 | 0.0185 | 0.0396 | 0.0456 |

Reshape and binding combine serial DET counters with overlapping REC owner counters. Result storage combines preparation, worker-slot writes and final assembly; ordinary owned conversion is outside these counters. Projection represents 8.2%, 7.7% and 3.1% of summed backbone-plus-projection work for dense Tiny, Small and Medium. That is a work composition, not the same share of page latency. CTC, warm reshape and result storage have very small measured costs.

## Implementation trace and remaining opportunities

| Link | Current implementation | Consequence for optimization |
|---|---|---|
| Input acquisition | [`ImageView`](../src/image.rs) borrows RGB/BGR/alpha pixels with a validated stride. The example application performs file decoding. | Capture, I/O and JPEG/PNG decode are outside these measurements. Caller-side conversion copies can be avoided where an accepted pixel format is already available. |
| Import and request creation | [`builder.rs`](../src/builder.rs) and [`model.rs`](../src/model.rs) parse/normalize ONNX, compile supported execution paths, share weights and construct mutable requests. | Amortize the 15.5–254.8 ms measured setup. Parallel import/compile is a cold-start hypothesis and needs its own memory/latency experiment if cold start matters. |
| DET shape and preprocessing | [`preprocess.rs`](../src/preprocess.rs) limits the long side, rounds dimensions to 32, caches interpolation maps/rows, uses SIMD and normalization lookup tables, and writes directly into the chosen layout. | Already optimized. Vectorizing remaining vertical interpolation/stores is plausible, but its measured cost is secondary to inference. |
| DET inference | [`runtime/optimized.rs`](../src/runtime/optimized.rs) propagates layouts, fuses stores, aliases eligible operations and reuses activation arenas; [`nhwc/transpose.rs`](../src/runtime/nhwc/transpose.rs) already contains direct-output AVX2 upsampling. | Optimize actual remaining convolution shapes. Recommending generic SIMD, packing or fusion would duplicate existing work. See physical-step attribution below. |
| DB postprocessing | [`db.rs`](../src/db.rs) uses SIMD threshold/scan helpers and reusable scratch, followed by component geometry, scoring, unclipping and reading-order sorting. | Only 0.44–0.80 ms here. Its minimum-area rectangle scan is quadratic in hull vertices; rotating calipers or component parallelism could matter on more complex contours, with tie/order parity checks. |
| Perspective crops | [`lines.rs`](../src/lines.rs) retains a contiguous crop slab and splits extraction into 16-row jobs. [`crop.rs`](../src/crop.rs) maps output coordinates and [`pixel_kernels.rs`](../src/pixel_kernels.rs) samples bicubic pixels. | Multi-output sampling or an affine/axis-aligned fast path is concrete future work. The current sampler performs 16 source reads per output pixel. Reusing temporary job vectors is smaller work. The current full crop-stage ceiling is 1.03–2.70 ms. |
| Classification | [`orientation.rs`](../src/orientation.rs) executes a fixed `[1,3,80,160]` prefix and compact two-class head, with one kernel thread per owner. [`lines.rs`](../src/lines.rs) applies rotations before starting REC. | Tiny dense pays about 25 ms. Skipping CLS is useful only with a validated orientation contract. Classifier batching or changing the phase barrier requires a separate output/memory/latency comparison; earlier overlap attempts are already recorded in the repository as rejected. |
| REC shape and scheduling | [`recognizer.rs`](../src/recognizer.rs) chooses height 48 and adaptive widths rounded to 32. [`ocr.rs`](../src/ocr.rs) orders widths longest-first and assigns deterministic owners by accumulated width cost. [`cpu.rs`](../src/cpu.rs) already selects eight automatic owners on AVX2/FMA with at least eight units. | The old suggestion to switch dense pages from four to eight owners is already implemented. Mixed-page widths reach 1536, so forcing width 320 can compress long text. Record per-owner completion before changing width-cost weighting. |
| Shape caches and memory | [`runtime.rs`](../src/runtime.rs) and [`runtime/optimized.rs`](../src/runtime/optimized.rs) retain eight prior logical/physical shape plans and activation high-water capacity. | Shape caching and arena reuse already exist. Stable ownership helps keep them warm. Tight engine idle-memory limits can evict useful warmed requests. |
| REC preprocessing | [`preprocess.rs`](../src/preprocess.rs) resizes directly into the selected layout and pads the chosen width. | Writing only padding instead of zeroing the entire padded tensor is a small candidate. Dense-page summed resize work is 5.2–5.5 ms and overlaps across owners. |
| REC backbone | [`nhwc/pointwise.rs`](../src/runtime/nhwc/pointwise.rs) uses packed six-row by sixteen-column tiles, fused final stores, and a 96-row reuse group for large channel products. The Windows inline short-reduction path already exists. | Channel-heavy 1×1 kernels are the strongest remaining dense Small/Medium target. The latest short-reduction change does not target most of the large-channel work. |
| Vocabulary head | [`projection.rs`](../src/projection.rs), [`recognizer.rs`](../src/recognizer.rs) and [`simd/argmax.rs`](../src/runtime/simd/argmax.rs) stop the backbone before the dense vocabulary head, project through packed panels and retain winning classes. | Full vocabulary probability tensors are already avoided. Tune the compact head only after the larger backbone cost. Preserve class tie behavior and the public uncalibrated-logit score semantics. |
| CTC and output | [`recognizer.rs`](../src/recognizer.rs) reuses decode storage; [`ocr.rs`](../src/ocr.rs) copies owner outputs through reusable slots into ordered result slabs. Owned conversion makes additional copies. | Measured CTC and storage are tiny. Removing a copy or combining decode passes is unlikely to be the first useful speed change for these inputs. |
| Concurrent requests | [`engine.rs`](../src/engine.rs) shares models, leases requests, limits idle retention and coordinates a shared CPU pool. | This experiment measures one request. Queueing, throughput, SMT topology, retention eviction and batch-dependent ownership need their own bounded workloads before changing global thread defaults. |

## Convolution attribution

The existing detector candidate measurements are reusable because their frozen production executable matches the current Tiny benchmark binary. The five recorded changed-source hashes, eleven input hashes and three frozen binary hashes also match. This provenance list is a changed-file manifest, not a complete source-tree digest. The new evidence retains the original [detector report](reports/windows-x64-2026-10-17-detector-optimization.json) reference and candidate profiles.

| Current sparse DET physical steps, ms | Tiny | Small | Medium |
|---|---:|---:|---:|
| All physical steps | 29.6661 | 76.7515 | 275.0140 |
| 1×1 group-one convolutions | 8.9960 | 25.7110 | 135.8640 |
| 1×1 share | 30.3% | 33.5% | 49.4% |
| All transposed convolutions | 2.8686 | 3.9279 | 8.6298 |

These use the actual sparse image at `[1,3,960,608]`, eight logical CPUs, one warm-up and one instrumented production pass per model. They exclude preprocessing and some session orchestration. They are not additional measurements of the current page-stage table. Tiny's remaining compute is distributed across regular and depthwise convolutions as well as pointwise work; a single universal pointwise rewrite will not remove the whole detector cost. Small has a particularly large node-2 sample, so one instrumented sample is insufficient to select that layer alone.

Fresh current-revision REC probes execute the fused production feature prefix, excluding the final vocabulary projection. Each uses deterministic synthetic input, one logical CPU, one warm-up and one timed pass with exact output-bit agreement:

| REC probe | Physical steps, ms | 1×1 convolutions, ms | 1×1 share | Layout conversion, ms |
|---|---:|---:|---:|---:|
| Small `[1,3,48,128]` | 9.4067 | 7.4926 | **79.7%** | 0.0377 |
| Medium `[1,3,48,640]` | 212.3399 | 191.1623 | **90.0%** | 0.0134 |

Small's largest layers include `384→768` and `768→384`; Medium includes `512→1024`, `1024→512` and `768→1536`. At the Medium probe shape, individual 1×1 steps reach approximately 10 ms. These serial shape-specific percentages locate kernel work; they cannot be multiplied directly into the eight-owner page timings to predict a page speedup. Hardware cache misses, bandwidth and register spills were not separately measured, so a particular microarchitectural cause is still a hypothesis.

The already adopted short-reduction inline path covers input channels 96–192 and output channels at least 128. Its [prior experiment](reports/windows-x64-2026-10-17-pointwise-optimization.json) found overlapping kernel ranges and no established Small/Medium page improvement. The dominant larger channel products still use the existing full-reduction path. Further pointwise work should target those products rather than repeat the short-reduction experiment.

## Optimization priorities

1. **Dense Small and Medium: target large-channel REC pointwise kernels.** Start with one actual `384→768` or `512→1024` shape. The subsequent [implementation-path analysis](windows-large-channel-convolution-analysis.md) found no established gain from doubling the 96-row group to 192 and identified existing rejected K=256 blocking/input-packing experiments. A new experiment should therefore change the mechanism, for example register-tile geometry or frequent fused-store dispatch, while preserving bias-first ascending-channel arithmetic, fused stores and tails. Require a production page improvement, complete-result parity and retained-memory accounting after a useful focused kernel result. Exact-width batch size two is a separate candidate: it can improve weight reuse but reduces independent recognition units, may reduce active owners, and serializes preprocessing inside each batch. This 69-line dense input still has enough units for eight owners at batch size two.

2. **Sparse pages: target DET compute.** The fresh Tiny sparse profile spends 68.8% in DET inference. The upsampling optimization is already present; inspect the remaining regular, depthwise and pointwise layers. Lower detector resolution also reduces work, but changes small-text detection and requires accuracy checks rather than being treated as a free kernel speedup.

3. **Upright Tiny workloads: consider omitting orientation classification.** The existing same-source experiment reduced dense median latency from **69.2114 to 44.1710 ms, 36.2% lower**, preserving all 69 texts, recognition scores, geometry and recognition dimensions. Classification metadata intentionally changes. On sparse input all three texts changed, and on mixed input one text changed. Use `.direction_classification(false)` only when the input contract and representative checks support it; geometric vertical-crop rotation still occurs. See [the complete toggle comparison](windows-no-cls-latency.md).

4. **Mixed widths: investigate stragglers before scheduler changes.** Recognition and DET account for approximately 48.1% and 32.9% of the mixed profile. There is a width-1536 line among much shorter units. Width-weighted deterministic ownership already exists; finer cost weights or batching should be supported by owner completion measurements and cache/memory checks.

5. **Treat DB, crop allocations, warm reshape, CTC and result copies as secondary.** They offer small absolute ceilings on these inputs. Even eliminating the entire DB stage saves less than 0.8 ms per profiled page. Eliminating crop work saves at most the measured 2.7 ms, not its summed worker duration.

For scale, halving the entire recognition wave would reduce the profiled Small/Medium total by approximately **39.0%/44.7%**; halving Tiny sparse DET inference would reduce its profiled total by **34.4%**. These are conditional critical-path calculations, not predictions for any proposed kernel change. A single faster operator, different memory contention or a scheduling tail can produce a much smaller page improvement.

## Validation and reproduction

Four profiling semantic tests passed before timing. Five real-image profiles matched complete warm-up outputs exactly. Both new ordinary OCR workers matched complete results across first, warm-up and measured calls. The two REC prefix probes matched output bits. Model/input hashes were checked, source and executable hashes stayed fixed during measurements, and the existing Tiny and detector evidence was verified before reuse.

New performance execution totaled **10.481 seconds**, excluding builds. Adding the reused Tiny orientation experiment (2.733 seconds) and the reused detector candidate work (7.060 seconds) gives **20.274 seconds** of conservatively counted execution, within 60 seconds. Every fresh process had a deadline of at most ten seconds, with an aggregate 60-second cap. There was no automatic sample growth, retry for nicer numbers, CI, remote runner or VM. The existing Windows `UNBOUND_AFFINITY` unused-import warning appeared during example builds.

The measured scope is three representative images, one eight-logical-CPU page configuration, five stage cases, two new three-sample owned-OCR cases and two serial kernel shapes. It does not qualify a full corpus, rotated-image accuracy, concurrent throughput, other thread counts or other architectures. Separate CLS preprocessing/backbone/head/rotation, acquisition/decode, owned-copy overhead and cold-import substeps have no individual timers here; their aggregate boundaries are stated above.

```powershell
cargo test --release --locked --features profiling --lib profile::tests -- --test-threads=1
cargo build --release --locked --example cpu_benchmark
cargo build --release --locked --features profiling --example cpu_profile

# One invocation per model/input: three Tiny inputs, then Small/Medium dense.
./target/release/examples/cpu_profile.exe tiny imgs_1 8
./target/release/examples/cpu_profile.exe tiny imgs_00018069 8
./target/release/examples/cpu_profile.exe tiny imgs_00006737 8
./target/release/examples/cpu_profile.exe small imgs_00018069 8
./target/release/examples/cpu_profile.exe medium imgs_00018069 8

# Internal Rust-only jobs preserve the settings and predecoded-pixel hashes.
./target/release/examples/cpu_benchmark.exe _worker target/current-pipeline-analysis-20261017/small-dense.job.json
./target/release/examples/cpu_benchmark.exe _worker target/current-pipeline-analysis-20261017/medium-dense.job.json

# Serial fused feature-prefix attribution, one warm-up and one pass per shape.
cargo test --release --locked --features profiling --lib runtime::optimized::tests::performance::recognition_backbone_physical_steps -- --exact --ignored --nocapture --test-threads=1
```

Run inference only after builds finish and preserve the external deadlines. The local `run.ps1` records the complete bounded execution; its machine-local test executable path is the one emitted by the focused correctness build. The raw jobs refer to retained local pixel files; regenerate them through the existing benchmark setup if those files have been removed.
