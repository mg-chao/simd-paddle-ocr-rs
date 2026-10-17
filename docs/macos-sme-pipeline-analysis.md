# Current Apple M4 OCR pipeline latency and optimization analysis

The subsequent [SME tile and crossover optimization](macos-arm64-sme-tile-optimization.md) implements the convolution recommendations and records the resulting gains and limitations. Measurements below describe revision `52ed1be` before that follow-up.

At revision `52ed1be343384c18bb24053599f8fd4928b56b2d`, detection remains the main cost on sparse pages, while recognition dominates dense pages. Detector inference accounts for **68.2–77.2%** of sparse-page wall time. Recognition accounts for **58.6%, 79.0% and 85.4%** of dense Tiny, Small and Medium wall time. **The next useful work is convolution execution and recognition scheduling; DB geometry, CTC decoding and result copying have small absolute ceilings on these inputs.** Tiny dense orientation is a separate substantial target at **26.5%** of its profile.

This analysis covers the current SME implementation, including its fallbacks. It supersedes the bottleneck ranking in the [pre-SME pipeline analysis](macos-current-pipeline-analysis.md); the [SME optimization report](macos-arm64-sme-optimization.md) retains the paired before/after speedup evidence. The [complete evidence](reports/macos-arm64-2026-10-17-head-52ed1be-pipeline-analysis.json) contains current-source hashes, all page samples, nested profiles, owner records, fresh operator timings, validation logs and reproduction scripts.

The page samples were already measured locally on 17 October 2026. Every `src/**/*.rs` hash matched the analysis checkout at `52ed1be`; benchmark examples, harness and build configuration were unchanged from their recorded base. Frozen benchmark/profile executable hashes also matched. Those samples were retained without rerunning unchanged code. Five new operator probes answered the outstanding question of which network operations dominate after SME. No samples from the pre-SME revision or rejected prototypes are pooled with these results. The subsequent tile optimization has separate source hashes and validation.

The physical machine is Apple M4, 10 cores, 16 GiB RAM, macOS 27.0.1. Builds use native release, pinned `nightly-2026-09-08`, default parallel features, automatic NEON/FMA with guarded SME, automatic layouts and fusion. Page measurements use eight shared workers, one request, automatic line ownership, adaptive recognition widths, batch one, detector limit 960, classification enabled and alignment disabled. Scheduling is unpinned. The host is on AC power, and `pmset` reports no recorded thermal/performance warning; neither core placement nor clock frequency is measured.

The three inputs are sparse `imgs_1` (720×1150), dense `imgs_00018069` (430×267), and mixed-width `imgs_00006737` (896×528). Their detector tensors are 608×960, 416×256 and 896×544. Different bundles can return different line counts and text, so cross-model speed comparisons do not establish equivalent recognition accuracy.

Ordinary latency starts from hash-verified decoded BGR24 and includes complete owned result conversion. Each of nine cases has one separately recorded first call, one warm-up and three measured calls in a fresh process. Load includes import, compilation, CPU-context and initial request creation. OS file caches were not flushed, so load is not disk-cold latency. Peak RSS is the process lifetime resident high-water mark, including setup and shared pages.

| Bundle / page | Lines | Median ms | Min–max ms | Load ms | First call ms | Peak RSS MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Tiny / sparse | 3 | 42.66 | 41.85–42.72 | 14.40 | 51.87 | 68.67 |
| Tiny / dense | 69 | 65.62 | 64.55–70.46 | 6.51 | 69.07 | 44.95 |
| Tiny / mixed | 31 | 73.81 | 73.63–75.55 | 6.48 | 77.14 | 76.81 |
| Small / sparse | 2 | 90.25 | 90.21–90.46 | 29.32 | 96.48 | 148.67 |
| Small / dense | 69 | 168.31 | 166.68–172.24 | 19.97 | 170.98 | 119.12 |
| Small / mixed | 33 | 207.49 | 202.68–207.85 | 19.62 | 216.89 | 180.16 |
| Medium / sparse | 2 | 329.55 | 321.03–336.11 | 91.41 | 319.93 | 454.89 |
| Medium / dense | 69 | 554.52 | 542.99–560.55 | 92.73 | 513.28 | 357.81 |
| Medium / mixed | 31 | 681.31 | 679.17–722.21 | 84.97 | 656.08 | 480.78 |

Measured ranges span 0.3–9.0% of their medians, with Tiny dense the widest. Medium sparse/mixed span 4.6%/6.3%. These three samples support broad bottleneck prioritization, not p95/p99 estimates or small-effect significance. First calls need not always be slower than later calls. Every original first/warm/measured result was checked for complete equality within its case.

The library path is borrowed pixels → detector preparation → detector network → DB boxes → perspective crops → orientation and rotation → recognition preparation/backbone/head/CTC → ordered results. File I/O, image decoding, application serialization and engine admission waiting are outside the reported inference latency. These costs must be measured in the consuming application if its end-to-end latency is the target.

The following elapsed stages come from separate profiling-enabled production runs, each with one warm-up and one instrumented call. All values are milliseconds. The six wall phases are sequential; their sum leaves only 0.012–0.016 ms of orchestration in total. They must not be combined with the ordinary medians above to estimate overhead or a stage speedup.

| Bundle / page | DET prepare | DET infer | DB boxes | Crop | CLS | REC | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Tiny / sparse | 2.77 | 29.70 | 0.44 | 0.39 | 1.34 | 7.47 | 42.13 |
| Tiny / dense | 0.57 | 8.12 | 0.31 | 0.70 | 17.24 | 38.19 | 65.14 |
| Tiny / mixed | 2.07 | 23.89 | 0.53 | 1.33 | 8.32 | 36.76 | 72.91 |
| Small / sparse | 2.94 | 61.29 | 0.43 | 0.42 | 1.41 | 23.31 | 89.83 |
| Small / dense | 0.51 | 15.00 | 0.29 | 0.67 | 17.55 | 128.06 | 162.11 |
| Small / mixed | 2.03 | 54.19 | 0.57 | 1.39 | 9.08 | 140.48 | 207.75 |
| Medium / sparse | 4.13 | 260.18 | 0.50 | 0.59 | 1.49 | 70.18 | 337.09 |
| Medium / dense | 0.73 | 59.31 | 0.31 | 0.87 | 21.36 | 484.49 | 567.09 |
| Medium / mixed | 2.78 | 220.64 | 0.56 | 1.18 | 10.49 | 479.68 | 715.35 |

CLS includes resize/normalization, its backbone, compact head and selected 180-degree pixel rotations. REC includes scheduling, resizing, feature extraction, vocabulary projection, CTC and ordered assembly. Crop includes geometry/slab preparation and row-task dispatch. Complete owned conversion is outside profile total and takes **0.0018–0.0130 ms** in separate post-profile measurements.

Parallel nested counters sum elapsed worker calls, including active-call stalls; they are neither additional wall phases nor process CPU time. DET/REC reshape and binding also overlap the enclosing phases. The dense-page breakdown is:

| Nested worker work ms | Tiny dense | Small dense | Medium dense |
| --- | ---: | ---: | ---: |
| Crop workers | 3.6252 | 3.6391 | 4.2930 |
| CLS resize and normalize | 7.8755 | 7.6998 | 8.5849 |
| CLS backbone | 112.6809 | 109.2149 | 112.6491 |
| CLS compact head | 0.1195 | 0.1222 | 0.1328 |
| CLS pixel rotation | 0.0000 | 0.0000 | 0.0000 |
| REC resize and normalize | 5.2613 | 5.8147 | 5.7909 |
| REC backbone | 218.9125 | 922.4689 | 2593.8554 |
| Vocabulary projection and argmax | 16.7035 | 68.6611 | 111.8464 |
| CTC decode | 0.0180 | 0.0296 | 0.0416 |
| DET and REC reshape | 0.0216 | 0.0389 | 0.0515 |
| Binding and validation | 0.0098 | 0.0173 | 0.0309 |
| Result storage | 0.0122 | 0.0163 | 0.0265 |

CLS backbone is 92.8–93.4% of its dense worker orientation time. Vocabulary projection is 7.1%, 6.9% and 4.1% of dense backbone-plus-projection work. CTC stays below 0.042 ms summed work, and warm reshape below 0.052 ms across all nine cases. These figures give little reason to prioritize decoder, result-ownership or warm shape-cache rewrites. Dense profiles apply no 180-degree rotations, while sparse profiles rotate all 3/2/2 lines and mixed profiles rotate 1/2/3; the dense fixture alone does not justify removing classification.

Source review connects each measured stage to its implementation:

| Link | Current implementation | Optimization implication |
| --- | --- | --- |
| Input and setup | [ImageView](../src/image.rs) borrows validated pixels. The [builder](../src/builder.rs) imports/compiles models; [engine leases](../src/engine.rs) reuse mutable requests with bounded idle retention. | Reuse models, compiled requests and the CPU pool. Rebuilding every image adds the load costs above. Engine queueing is workload-dependent. |
| DET preparation | [Preprocessing](../src/preprocess.rs) caches interpolation maps and rows, uses fixed-point byte interpolation and normalization tables, and writes directly into the chosen tensor layout. It partitions rows across the stage budget. | 0.51–4.13 ms. The vertical interpolation/table/store loop remains a concrete secondary SIMD target; transient job allocations are also present but their individual costs are unmeasured. |
| DET network | The [physical runtime](../src/runtime/optimized.rs) fuses eligible operations and stores, aliases values, reuses arenas and reserves scratch. Dedicated NCHW/NHWC SIMD paths and narrow-channel kernels are already active. | Remaining convolution arithmetic is the main sparse-page target. Further opportunities must be tied to measured shapes rather than reintroducing existing fusion or SIMD. |
| DB boxes | [DB](../src/db.rs) thresholds/scans the map, fills components and holes, computes hulls/minimum-area rectangles, scores/unclips polygons and sorts reading order. Scratch storage and scan kernels are retained. | 0.29–0.57 ms here. Geometry complexity could matter on other contours, but these pages provide little return from a rewrite. |
| Perspective crops | [Line extraction](../src/lines.rs) writes a retained BGR slab in 16-row tasks. [Cropping](../src/crop.rs) uses SIMD coordinate mapping and interior cubic sampling, with clamped borders and folded vertical rotation. | 0.39–1.39 ms. Crop-to-network fusion would alter the cubic-byte-then-bilinear numerical path and needs correctness/accuracy evidence. |
| Orientation | The [classifier](../src/orientation.rs) uses fixed 80×160 input, a feature prefix and a compact two-class head. Up to eight single-thread owners classify independent lines, then CLS completes before REC starts. | Tiny dense spends 17.24 ms here. Backbone improvements or true classifier batching are useful candidates. REC batch configuration does not batch CLS. |
| REC sizing and binding | The [recognizer](../src/recognizer.rs) uses height 48 and adaptive widths rounded to 32. [Sessions](../src/runtime.rs) retain eight previous shape plans and high-water buffers. | Warm binding is already cheap. Dense pages mostly use widths 96/128; mixed pages reach 1472/1536. Fixed width 320 changes long-text sampling and needs accuracy evaluation. |
| REC scheduling | [OCR](../src/ocr.rs) sorts units by width and greedily assigns them to stable least-loaded owners. Automatic recognition uses up to eight owners; one dominant long-crop owner can expose two kernel bands. | Width balances nominal work but not measured runtime after SME. Owner completion evidence below merits a targeted scheduling investigation. |
| REC convolutions | [Pointwise dispatch](../src/runtime/nhwc/pointwise.rs) requires at least 32 executing rows and both channel dimensions ≥384 for SME; other shapes retain NEON. [SME](../src/runtime/nhwc/pointwise/sme.rs) packs up to 64 input rows, computes 16×16 tiles, stages results and applies fused stores. | Large pointwise layers remain expensive; below-threshold 192↔384 and 256↔512 layers are additional specific targets. Packing, repeated tile entry and result staging are candidates to isolate, not established causes. |
| Vocabulary head and CTC | The [compact head](../src/recognizer.rs) computes class winners without materializing the full vocabulary tensor. CTC collapses winners into reusable text/span storage. | Projection is secondary; CTC is negligible on these inputs. Alignment was disabled and has separate output work. |
| Results and memory | [Ordered assembly](../src/ocr.rs) restores reading order and supports borrowed/owned results. Each owner retains private activations, shape plans, resize buffers and kernel scratch. | Copying is cheap. More workers, dynamic ownership and batching can increase retained memory or disrupt warm reuse. |

Separate instrumented startup observations span 3.43–3.79 ms import plus 1.81–1.92 ms compile/request for Tiny, 10.11–10.23 plus 8.50–8.74 ms for Small, and 37.78–38.19 plus 45.04–45.23 ms for Medium. CPU-context creation spans 0.076–0.124 ms. These are observations from separate processes, not pieces to add to an ordinary inference median.

The new fused-operator diagnostics retain production layouts, fusion and kernels. DET consumes the real sparse image at `[1,3,960,608]` with eight threads; REC consumes deterministic synthetic tensors through the feature prefix at the stated width with one thread. Each shape gets one warm-up and one instrumented pass, and every output preserves float bits. Public trace execution is unfused and was deliberately not used as production timing.

| Fused operator probe | All steps ms | 1×1 convolution ms | 1×1 share | Layout conversion ms |
| --- | ---: | ---: | ---: | ---: |
| Tiny DET sparse, 8 threads | 28.404 | 11.492 | 40.5% | 0.0000 |
| Small DET sparse, 8 threads | 66.514 | 32.493 | 48.9% | 2.3965 |
| Medium DET sparse, 8 threads | 341.722 | 138.482 | 40.5% | 0.0000 |
| Small REC width 128, 1 thread | 9.616 | 7.828 | 81.4% | 0.0022 |
| Medium REC width 640, 1 thread | 147.409 | 132.520 | 89.9% | 0.0095 |

**Small REC:** the 192→384 layers over 192 spatial rows take 0.401–0.421 ms each; the reverse 384→192 layers take about 0.290–0.293 ms. Neither direction passes the current SME channel threshold. Existing SME-eligible 384→768 layers over 96 rows still take 0.421–0.427 ms. A useful next experiment is the exact 192-row, K=192, N=384 shape, including input packing and fused stores, to establish whether a lower crossover or a better NEON kernel helps.

**Medium REC:** repeated 512→1024 layers over 960 rows take 6.67–6.94 ms, while 768→1536 layers take 5.89–6.02 ms. These are already SME-eligible. Below-threshold 256→512 layers over 1920 rows take 5.84–5.87 ms, with reverse layers around 4.45–4.50 ms. Both improving retained SME execution and testing the narrower crossover have concrete targets. Attention MatMul, softmax, layer normalization and transpose together account for only 1.3%/1.2% of Small/Medium probe time.

**DET:** Tiny's 64→16 3×3 layer takes 1.88 ms, and two 32→64 pointwise layers take 1.78/1.81 ms. Small's 96→24 3×3 takes 4.98 ms, its first transposed convolution 2.83 ms, and 48→96 pointwise layers 2.54/2.71 ms. These smaller channels are not addressed by the ≥384 SME gate. Small layout conversion totals 2.40 ms, materially less than its convolution work.

The Medium DET diagnostic is noticeably slower than its separate page profile: 341.72 ms of step timings versus 260.18 ms of detector wall time. Its 128→64 stride-two 3×3 and 32→64 2×2 steps record 43.95/39.61 ms, while 256→64 3×3 records 27.90 ms. The cause is unresolved with one diagnostic pass. These are candidate shapes, not stable isolated latency estimates or evidence of a regression. The unfavorable observation is retained without a cleaner rerun. Operator percentages belong to that diagnostic and must not be substituted into page-stage totals.

Recognition owner completion provides another optimization lead. Width cost is the sum of scheduled recognition tensor widths; finish offsets share one dispatch origin. All listed owners begin within 0.5 ms. Dense owners use one kernel thread each. Mixed pages give only the longest single-crop owner two threads.

| Bundle / page | Width cost range | Owner finish range ms | Longest crop owner finish ms | Last owner maximum crop width |
| --- | ---: | ---: | ---: | ---: |
| Tiny / dense | 1024–1120 | 22.31–38.17 | — | 224 |
| Tiny / mixed | 1088–1536 | 22.00–36.75 | 36.50 | 384 |
| Small / dense | 1024–1120 | 118.86–128.00 | — | 192 |
| Small / mixed | 1024–1472 | 104.84–140.44 | 137.41 | 320 |
| Medium / dense | 992–1088 | 239.64–484.46 | — | 192 |
| Medium / mixed | 1056–1472 | 259.08–479.60 | 411.51 | 480 |

For Medium dense, width costs differ by only 9.7%, but completion ranges from 239.64 to 484.46 ms. The slowest owner uses 484.01 ms of calling-thread CPU during its 484.45 ms wall interval. Medium mixed's 1472-wide crop finishes at 411.51 ms, **68.08 ms before** the last owner, whose maximum crop width is 480. Tiny and Small's longest crops also finish before their last owners. Further special-casing only the longest crop therefore misses the observed critical path.

These observations support investigating owner cost estimation, available parallel work and SME execution under concurrent owners. They do not prove a P/E-core placement issue, memory-bandwidth bottleneck or SME hardware contention. Thread CPU clocks exclude helpers and do not distinguish useful arithmetic from active stalls. A next scheduling probe should record per-owner kernel selection and hardware placement if available, then test a deterministic policy with bounded memory. Blind work stealing would risk the stable ownership and buffer-reuse properties that the current code intentionally preserves.

| Bundle / page | Retained activations MiB | Retained workspace MiB |
| --- | ---: | ---: |
| Tiny / mixed | 41.79 | 10.68 |
| Small / mixed | 84.79 | 21.78 |
| Medium / mixed | 175.57 | 25.83 |

Medium mixed additionally peaks at 480.78 MiB RSS. Retained activation/workspace sizes and RSS measure different things; immutable model/packed weights, allocator behavior and shared pages also contribute. More simultaneous requests can multiply private scratch, so a latency gain needs memory validation.

The practical priority depends on the workload:

1. **Small/Medium dense or mixed pages:** optimize pointwise convolutions and diagnose the Medium owner imbalance. Test one exact below-threshold shape and one already-SME shape before changing dispatch broadly. Preserve reduction order, bias, tails, fused epilogues and fallback behavior.
2. **Sparse pages:** optimize detector convolution, especially measured smaller-channel pointwise and 3×3 layers. The recent SME work did not remove detector dominance.
3. **Tiny dense pages:** target orientation backbone or classifier batching alongside recognition. Removing CLS is an application option only with an upright-input contract and output checks; it changes behavior on the rotated sparse/mixed inputs.
4. **Secondary work:** detector resize/normalization, compact vocabulary projection, and verified exact-width REC batching. Batch support already exists; it is not automatically faster and can reduce independent work. Smaller models or detector resolutions require a separate ground-truth accuracy/latency tradeoff study.
5. **Defer for these inputs:** DB geometry, CTC, result copying and more warm shape caching. Their measured costs cannot explain the main latency.

For scale, a hypothetical **20% reduction of the whole REC phase** saves **7.64/25.61/96.90 ms** on dense Tiny/Small/Medium, or **11.7%/15.8%/17.1%** of profiled page time. Halving Tiny dense CLS saves **8.62 ms (13.2%)**. A 20% DET inference reduction saves **14.1%/13.6%/15.4%** of sparse Tiny/Small/Medium time. These are sensitivity calculations with other stages fixed, not measured candidate speedups; kernel-only improvements have smaller page effects.

Local validation in this analysis passed **7 profile semantic tests**, **4 SME correctness tests**, and **5 exact operator-output comparisons**. All **9 archived page results** were rechecked against saved goldens with unchanged 0.5-pixel coordinate and `1e-4 + 1e-4×abs(expected)` score tolerances; all text, counts, widths, labels and rotations matched. Model/pixel hashes and source/binary provenance passed. The earlier full 213-request qualification belongs to the SME implementation report and was not rerun here. Production source, model assets, goldens and tolerances are unchanged.

New performance execution took **1.245 seconds**: three DET and two REC warm-ups, then one measured pass for each. Together with the retained **21.353 seconds** of current-version page/profile execution, cumulative performance execution is **22.599 seconds**, below the 60-second cap. The cumulative count is 9 first calls + 9 ordinary warm-ups + 27 ordinary samples + 9 profile warm-ups + 9 profiles + 5 operator warm-ups + 5 operator passes = **73 executions**. Process startup/loading count toward the budget; builds and focused correctness checks do not. New diagnostic subprocesses had a 12-second timeout capped by the remaining aggregate budget, with no retries or sampling growth.

The reproduction driver and raw new logs are in `target/pipeline-analysis-20261017-head-52ed1be`. The portable JSON embeds the scripts and all current evidence. For a new reproduction, use a fresh directory at the same depth, retain the hash-matched prior page artifacts, run the following focused checks into the driver's two log paths, and execute its `run.py` then `package.py`:

```sh
cargo test --release --locked --features profiling --lib profile::tests -- --test-threads=1
cargo test --release --locked --lib runtime::nhwc::pointwise::sme::tests -- --test-threads=1
```

For a future code change, rebuild and reset its measured run count; these samples then become a baseline, not validation of the new version. Start with the affected kernel/stage and one warm-up plus three single-call release samples. Expand to the corresponding page only if that answers a concrete remaining question. One physical host, three page inputs and one page configuration do not establish other-platform speed, concurrent throughput, application accuracy or tail latency.
