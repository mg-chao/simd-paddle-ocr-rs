# Apple M4 pipeline latency after fused panel stores

The current working tree still spends most OCR time in neural-network execution. Sparse pages spend **71.0–79.8%** of their profiled latency in the detector network. Dense pages spend **48.9%, 71.2% and 82.4%** in recognition for Tiny, Small and Medium. Tiny dense also spends **17.80 ms, or 32.7%**, in orientation classification. The most useful next work is spatial convolution in the detector, pointwise convolution in recognition, and investigation of recognition worker imbalance. Classifier acceleration is particularly valuable for Tiny dense pages.

This analysis covers `ca0996a` plus the uncommitted [fused panel-store implementation](macos-arm64-panel-store-optimization.md), not clean `ca0996a`. The wider NEON/SME stores, expanded SME crossover, input packing, operation fusion and stable recognition ownership are already implemented. The measurements below include those changes. No production code was changed for this analysis.

[Raw evidence](reports/macos-arm64-2026-10-18-panel-store-pipeline-analysis.json) contains ordinary samples, all stage and owner counters, operator shapes, model/input provenance, source and executable hashes, the source patch, validation logs and reproduction scripts.

**Measurement conditions.** Physical Apple M4, four performance cores and six efficiency cores, 16 GiB RAM, macOS 27.0.1, AC power, native ARM64 release builds and pinned `nightly-2026-09-08`. One active request shares eight CPU workers. SIMD/layout/fusion and line ownership are automatic; recognition batch size is one, width is adaptive, detector limit is 960, classification is enabled with threshold zero, dilation and alignment are off. Model-specific box thresholds and the remaining manifest settings are unchanged. macOS scheduling is unpinned; thread indices do not identify physical cores. No recorded thermal warning was reported, but frequencies and temperatures were not sampled.

All 109 Rust source files match the final panel-store verification. Rebuilt ordinary benchmark and profile executables match the recorded executable hashes byte-for-byte; harness, examples, Cargo files and toolchain have no differences from the recorded baseline. Therefore the nine ordinary page cases and three existing stage profiles are reused. Six missing stage profiles and five fused-operator shapes were measured locally for this analysis. Reused and new measurements retain their separate origins; no samples from different implementations or timing methods are pooled.

**Ordinary OCR latency.** Each page case has a separately measured first call, one explicit warm-up and three measured calls returning owned results. Input is predecoded BGR24. Model setup, image-file decoding, serialization and service admission waiting are outside these timers. The first call includes lazy request/worker initialization and first-use effects; it is not a cold-disk measurement.

| Bundle / page | Lines | Warm median ms | Range ms | First call ms |
| --- | --- | --- | --- | --- |
| Tiny / sparse | 3 | 38.54 | 37.95–38.59 | 73.10 |
| Tiny / dense | 69 | 55.14 | 54.03–55.61 | 57.29 |
| Tiny / mixed | 31 | 65.41 | 65.21–75.37 | 64.23 |
| Small / sparse | 2 | 82.54 | 81.41–83.68 | 90.42 |
| Small / dense | 69 | 112.79 | 111.18–114.43 | 113.55 |
| Small / mixed | 33 | 152.11 | 151.82–152.91 | 149.41 |
| Medium / sparse | 2 | 252.41 | 240.71–263.39 | 249.81 |
| Medium / dense | 69 | 365.23 | 356.76–373.92 | 364.52 |
| Medium / mixed | 31 | 509.82 | 505.75–517.76 | 504.87 |

Sparse is `imgs_1`: 720×1150 source pixels, 608×960 detector tensor, and only 2–3 lines. Dense is `imgs_00018069`: 69 lines and a 416×256 detector tensor. Mixed is `imgs_00006737`: 31–33 lines and an 896×544 detector tensor. Recognition uses height 48. Dense crops mainly use widths 96/128; mixed pages include width 1536 for Tiny and 1472 for Small/Medium. Thus text count alone does not predict latency: sparse pages have the largest detector tensor, while mixed pages include expensive long recognition crops. Different bundles produce different boxes, so their page times do not compare identical kernel workloads or establish an accuracy ranking.

**Elapsed time through the pipeline, milliseconds.** Each profile uses one ordinary warm-up and one instrumented call, requiring complete owned output equality. Tiny dense, Small dense and Medium mixed are the reused profiles; the other six are new. These six wall phases execute sequentially in the measured configuration.

| Bundle / page | DET prep | DET network | DB boxes | Crop | CLS and rotation | REC and assembly | Total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Tiny / sparse | 2.74 | 25.73 | 0.42 | 0.44 | 1.17 | 5.73 | 36.25 |
| Tiny / dense | 0.63 | 8.22 | 0.30 | 0.79 | 17.80 | 26.62 | 54.38 |
| Tiny / mixed | 2.05 | 21.55 | 0.53 | 1.06 | 7.43 | 27.13 | 59.78 |
| Small / sparse | 2.92 | 52.00 | 0.46 | 0.43 | 1.30 | 15.55 | 72.68 |
| Small / dense | 0.51 | 13.55 | 0.29 | 0.61 | 17.14 | 79.38 | 111.49 |
| Small / mixed | 2.08 | 46.98 | 0.64 | 1.46 | 8.82 | 86.65 | 146.65 |
| Medium / sparse | 3.83 | 203.36 | 0.49 | 0.47 | 1.33 | 45.23 | 254.72 |
| Medium / dense | 0.66 | 48.72 | 0.32 | 0.92 | 17.72 | 321.01 | 389.36 |
| Medium / mixed | 2.86 | 172.39 | 0.56 | 1.43 | 9.28 | 296.47 | 483.01 |

The unrounded phase sum leaves only **0.011–0.021 ms** inside total for outer validation, binding and orchestration. REC includes dispatch, joins and ordered result assembly. Owned conversion is timed separately after total. Instrumented totals are not components of the ordinary median: Medium dense, for example, profiles at 389.36 ms while its ordinary median is 365.23 ms. Scheduling and instrumentation differ; that gap is not a measured regression or timer-overhead estimate.

```mermaid
flowchart LR
    A[Borrowed pixels] --> B[Resize and normalize]
    B --> C[Detector network]
    C --> D[DB boxes and reading order]
    D --> E[Perspective crops]
    E --> F[Orientation classification]
    F --> G[Recognition backbone]
    G --> H[Compact projection and CTC]
    H --> I[Ordered results]
```

**Code path and optimization implications.**

| Stage | Current implementation | Interpretation |
| --- | --- | --- |
| Input and setup | [ImageView](../src/image.rs) borrows validated raw pixels. [Builder](../src/builder.rs) shares immutable models; [engine leases](../src/engine.rs) retain mutable requests under a bounded pool. | Reuse models, requests and CPU contexts. A service's queue time needs a separate workload; the library timings here start with available pixels and a request. |
| Detector preprocessing | [Preprocessing](../src/preprocess.rs) caches resize coefficients and row scratch, uses fixed-point byte interpolation and normalization lookup tables, writes the selected tensor layout directly, and divides rows among workers. | 0.51–3.83 ms here. Per-call job allocations and interpolation are secondary opportunities; their individual costs are not isolated by this profile. |
| Detector network | [Detector](../src/detector.rs) reshapes, fills the selected input layout, executes the optimized graph and returns its probability map. [Runtime](../src/runtime/optimized.rs) fuses eligible activation/residual stores and aliases reusable buffers. | 8.22–203.36 ms. Optimize the measured fused convolution shapes, rather than adding fusion or SIMD that already exists. |
| DB postprocessing | [DB](../src/db.rs) thresholds the map, finds components/contours, builds hulls and rectangles, scores and expands boxes, and sorts reading order with reusable scratch. | 0.29–0.64 ms. This corpus does not justify a geometry rewrite. Dilation is disabled, so these times do not characterize that option. |
| Crop extraction | [Line extraction](../src/lines.rs) prepares transforms and a retained BGR slab, then schedules 16-row crop jobs. [Crop kernels](../src/crop.rs) use SIMD coordinates and cubic sampling with border handling and geometric rotation. | 0.43–1.46 ms. Crop-to-tensor fusion would need to preserve the existing cubic-byte then bilinear-resize semantics; it is a low priority at this share. |
| Classification | [Orientation](../src/orientation.rs) resizes to fixed 80×160 input, runs a feature prefix and evaluates a compact two-class head. Up to eight single-thread owners process round-robin line lists. | 1.17–17.80 ms wall time. A barrier precedes recognition. Backbone execution dominates; recognition batching does not batch this classifier. |
| Recognition | [Recognizer](../src/recognizer.rs) selects width, reshapes to height 48, resizes/normalizes, executes the feature prefix, projects vocabulary maxima and decodes CTC. | 5.73–321.01 ms including scheduling and assembly. Dense Small/Medium need backbone and ownership improvements. |
| Output | [OCR](../src/ocr.rs) restores reading order from stable worker slots and appends text/metadata to retained storage; owned conversion copies the requested result. | Storage is at most 0.035 ms of summed work and owned copy at most 0.035 ms. A lock-free result rewrite would not address the measured bottleneck. |

The [CLI](../examples/ocr.rs) additionally reads/decodes JPEG or PNG and formats output. Those costs, engine admission waiting, optional token alignment and application transport are outside this workload and remain unmeasured.

**Classification details, summed worker milliseconds.** These counters overlap across owners inside the CLS wall phase. They are not extra wall phases or process CPU time.

| Bundle / page | Resize / normalize | Backbone | Head | 180° rotation |
| --- | --- | --- | --- | --- |
| Tiny / sparse | 0.3714 | 3.0444 | 0.0050 | 0.0220 |
| Tiny / dense | 8.6810 | 109.9864 | 0.1248 | 0.0000 |
| Tiny / mixed | 4.3103 | 44.9871 | 0.0560 | 0.0011 |
| Small / sparse | 0.2977 | 2.2475 | 0.0042 | 0.0205 |
| Small / dense | 8.0715 | 99.3287 | 0.1230 | 0.0000 |
| Small / mixed | 4.8882 | 47.2592 | 0.0595 | 0.0010 |
| Medium / sparse | 0.2984 | 2.3268 | 0.0040 | 0.0204 |
| Medium / dense | 8.6419 | 102.9312 | 0.1310 | 0.0000 |
| Medium / mixed | 4.7107 | 48.3883 | 0.0562 | 0.0063 |

Dense-page backbone execution contributes about **92–93%** of the measured classifier preprocessing/backbone/head/rotation work. No dense-page crop is rotated; sparse pages rotate all 3/2/2 lines and mixed pages rotate 1/2/3 for Tiny/Small/Medium. The cost is repeated network evaluation, not pixel rotation. Disabling classification would change behavior on this corpus and is appropriate only when an application can guarantee upright crops. A useful implementation experiment is classifier backbone profiling followed by one fixed-shape batching alternative. Its numerical output, memory and parallelism must be checked separately from recognition batching.

**Recognition details, summed worker milliseconds.** These counters overlap between owners and are nested inside REC wall time.

| Bundle / page | Resize / normalize | Backbone | Projection | CTC |
| --- | --- | --- | --- | --- |
| Tiny / sparse | 0.7890 | 10.3183 | 1.0051 | 0.0027 |
| Tiny / dense | 6.0704 | 156.6701 | 18.9735 | 0.0282 |
| Tiny / mixed | 7.9228 | 153.6405 | 19.4016 | 0.0220 |
| Small / sparse | 0.8728 | 26.4574 | 1.8129 | 0.0027 |
| Small / dense | 6.0071 | 528.3084 | 79.4626 | 0.0354 |
| Small / mixed | 8.2484 | 534.0166 | 75.4885 | 0.0292 |
| Medium / sparse | 0.9135 | 77.1263 | 4.4862 | 0.0046 |
| Medium / dense | 5.7084 | 1579.6240 | 110.1257 | 0.0461 |
| Medium / mixed | 7.2903 | 1699.5846 | 110.5424 | 0.0360 |

The compact projection already avoids materializing the full vocabulary tensor and its final softmax. Projection is **10.8%, 13.1% and 6.5%** of dense Tiny/Small/Medium backbone-plus-projection work. It is a secondary target, particularly for Small. A candidate should retain compact maxima, deterministic ties and score semantics; a full dense output would discard an existing memory optimization. CTC is at most **0.0461 ms** summed across the page, so decoder work has negligible leverage here.

**Binding and result costs, milliseconds.** Binding, reshape and result storage are nested work counters. Owned copy is a separate elapsed timer after profile total.

| Bundle / page | Binding | DET and REC reshape | Result storage | Owned copy |
| --- | --- | --- | --- | --- |
| Tiny / sparse | 0.0011 | 0.0007 | 0.0013 | 0.0016 |
| Tiny / dense | 0.0163 | 0.0371 | 0.0133 | 0.0073 |
| Tiny / mixed | 0.0217 | 0.0289 | 0.0066 | 0.0347 |
| Small / sparse | 0.0022 | 0.0008 | 0.0015 | 0.0025 |
| Small / dense | 0.0272 | 0.0460 | 0.0208 | 0.0057 |
| Small / mixed | 0.0244 | 0.0505 | 0.0136 | 0.0040 |
| Medium / sparse | 0.0014 | 0.0007 | 0.0015 | 0.0020 |
| Medium / dense | 0.0502 | 0.0745 | 0.0347 | 0.0085 |
| Medium / mixed | 0.0519 | 0.0712 | 0.0189 | 0.0038 |

The [runtime](../src/runtime.rs) caches eight previous shape plans and retains activation high-water buffers. Warm DET/REC reshape costs at most **0.075 ms** summed across owners. The classifier's fixed-shape binding is part of request construction. Increasing shape-cache capacity is not supported as a latency priority by these warm profiles.

**Fresh diagnosis of fused operators.** Detector probes use the real sparse image at `[1,3,960,608]` with eight threads. Recognition probes use deterministic synthetic input and the production feature prefix at the stated width, one thread, excluding the vocabulary head. Each shape has one warm-up and one instrumented pass, with bit-for-bit output equality. The probes retain production fusion, layout and kernel selection; they do not use the separate unfused trace path. Serial recognition probes do not reproduce eight concurrent owners.

| Fused probe | Step sum ms | Pointwise ms | Pointwise share | Layout conversion ms |
| --- | --- | --- | --- | --- |
| Tiny DET sparse / 8 threads | 25.050 | 8.838 | 35.3% | 0.0000 |
| Small DET sparse / 8 threads | 50.266 | 18.761 | 37.3% | 2.0622 |
| Medium DET sparse / 8 threads | 200.381 | 66.484 | 33.2% | 0.0000 |
| Small REC width 128 / 1 thread | 4.523 | 2.896 | 64.0% | 0.0022 |
| Medium REC width 640 / 1 thread | 55.372 | 40.310 | 72.8% | 0.0082 |

The concrete hot operations are:

- **Tiny detector:** 64→16 3×3 takes 1.719 ms; the first transposed convolution takes 1.646 ms; 16→8 2×2 takes 1.565 ms. Pointwise work totals 8.838 ms, including a 32→64 layer at 1.297 ms. These narrow channel counts do not meet the SME gate. Spatial reuse, narrow NEON kernels and the upsampling path are the relevant candidates.
- **Small detector:** 96→24 3×3 takes 4.831 ms; the first transposed convolution takes 2.558 ms; the initial 3→24 3×3 and 48→24 stride-two 3×3 take 2.267/2.256 ms. Layout conversions total 2.062 ms, **4.1%** of step time, making them measurable but secondary.
- **Medium detector:** 256→64 3×3 takes 28.151 ms and 128→64 stride-two 3×3 takes 14.519 ms. Together they consume **42.670 ms, or 21.3%**, of detector step time. All dense 3×3 work totals 51.898 ms. A 256-channel depthwise 9×9 takes 9.538 ms, the first transposed convolution 9.289 ms, and a dense 32→32 7×7 9.267 ms. Large spatial convolutions deserve attention alongside the 66.484 ms of pointwise work.
- **Small recognition at width 128:** the leading layer is 96→48 stride-two 3×3 at 0.321 ms; two early 2×2 convolutions take 0.237/0.212 ms. Pointwise work is still **64.0%** of the backbone. The top 96→192 pointwise layers take 0.202/0.197 ms and already qualify for SME.
- **Medium recognition at width 640:** the first 128→64 stride-two 3×3 takes 2.854 ms. Six 512→1024 pointwise layers over 960 spatial rows take 2.159–2.476 ms each; pointwise work totals **40.310 ms, or 72.8%** of the backbone. Attention MatMul, softmax, layer normalization and transpose together are about 3.3%, so attention is a much smaller target in this probe.

The [dense kernel](../src/runtime/nhwc/dense.rs) already packs weights, groups six-row tiles, uses 512-element reduction blocks, precomputes interior geometry and applies sixteen-lane fused stores on ARM64. It does not materialize a full im2col buffer. A bounded experiment could compare one spatial-reuse improvement or tiled input packing for the two Medium 3×3 shapes. Extra packing, scratch and border handling must be included in the timing; an isolated matrix multiply would not answer whether the convolution gets faster.

The [SME implementation](../src/runtime/nhwc/pointwise/sme.rs) already packs 64-row groups with NEON transposes, uses 32×32 tiles and 16×16 tails, and applies the new wide fused stores. Remaining possibilities include amortizing streaming-mode transitions and changing result staging. They are hypotheses: these measurements do not isolate packing, compute, staging or shared hardware contention. Any candidate must preserve live-ZA/TPIDR2/FPCR guards, vector-length checks, bias-first ascending-K accumulation and fallback semantics. Repeating the previous crossover expansion or store widening would duplicate completed work.

**Recognition worker completion.** Finish times share one dispatch origin. Assigned width sum is the current cost estimate, not measured work. Dense owners use one kernel thread; a dominant single-crop owner on mixed pages can borrow a second thread from the same pool.

| Bundle / page | Assigned width sum | Owner finish range ms | Longest crop finishes ms | Last owner maximum width |
| --- | --- | --- | --- | --- |
| Tiny / dense | 1024–1120 | 17.60–26.59 | — | 192 |
| Tiny / mixed | 1088–1536 | 17.36–27.12 | 20.41 | 448 |
| Small / dense | 1024–1120 | 72.86–79.36 | — | 224 |
| Small / mixed | 1024–1472 | 64.55–86.62 | 80.07 | 384 |
| Medium / dense | 992–1088 | 150.28–320.90 | — | 192 |
| Medium / mixed | 1056–1472 | 149.84–296.46 | 296.46 | 1472 |

Medium dense assigns width sums of 992–1088, a 9.7% spread, yet completion spans **150.28–320.90 ms**. Its last owner has only width 192 as its largest crop and accumulates 320.26 ms of calling-thread CPU time during 320.88 ms of active wall time. Medium mixed instead finishes with the width-1472 crop at 296.46 ms despite the two-thread hint. Tiny and Small mixed finish on other owners after their longest crops complete. Therefore “make the longest crop faster” does not address every page's critical path.

This establishes completion imbalance, but does not identify its cause. The machine has heterogeneous cores; core placement, shape-dependent costs and concurrent cache/SME resource effects are plausible explanations. Thread CPU time excludes nested helpers and includes active stalls. Neither its ratio nor the finish-time spread is an achievable speedup estimate. Record actual per-owner shape/kernel costs and compare one owner-budget or deterministic cost-model alternative before changing policy. Preserve output ordering and retained memory: unrestricted work stealing can populate different worker-private arenas and shape caches on repeated pages.

**Setup and retained memory.** Setup ranges below span the three separate diagnostic processes per bundle. The ordinary benchmark independently times its combined setup. They are separate observations rather than additive columns.

| Bundle | Model import ms | CPU context ms | Compile and request ms | Ordinary load and compile ms |
| --- | --- | --- | --- | --- |
| Tiny | 3.599–4.534 | 0.089–0.128 | 1.857–2.114 | 6.37–15.56 |
| Small | 10.098–10.510 | 0.076–0.084 | 8.576–9.174 | 20.36–20.56 |
| Medium | 38.221–44.360 | 0.085–0.090 | 46.954–52.791 | 91.41–96.02 |

Recreating a Medium model/request for every image adds roughly 91–96 ms in the ordinary measurements before inference. Reusing the model, CPU context and warmed request is the clearest application-level improvement when callers currently reconstruct them. The first-inference column above captures additional first-use work separately.

| Bundle / page | Retained activations MiB | Retained workspace MiB | Peak RSS MiB |
| --- | --- | --- | --- |
| Tiny / sparse | 39.06 | 6.34 | 70.20 |
| Tiny / dense | 12.42 | 10.34 | 46.17 |
| Tiny / mixed | 41.79 | 11.98 | 77.95 |
| Small / sparse | 64.29 | 8.93 | 148.61 |
| Small / dense | 26.85 | 20.04 | 118.77 |
| Small / mixed | 84.79 | 21.77 | 179.80 |
| Medium / sparse | 156.36 | 12.43 | 454.64 |
| Medium / dense | 46.95 | 23.93 | 357.73 |
| Medium / mixed | 175.57 | 25.81 | 481.05 |

Activations and workspace are retained capacities; workspace includes kernel scratch. Peak RSS includes model weights, allocator state and other resident pages, so these columns are not additive. Medium mixed retains 175.57 MiB of activations and peaks at 481.05 MiB RSS. More owners or larger batches can increase private activation capacity; latency candidates should preserve these diagnostics alongside outputs.

**Prioritized experiments and potential effect.** These are proposed next steps, not measured candidate gains. The sensitivity column assumes only the named elapsed phase becomes shorter and everything else stays fixed; it does not translate summed worker work into wall savings.

| Priority | Concrete next experiment | Reason | Sensitivity |
| --- | --- | --- | --- |
| 1 for dense or mixed Small/Medium | Profile one representative pointwise layer under concurrent owners; compare one SME transition/staging change. | Pointwise is 64.0–72.8% of the serial backbone; recognition dominates these pages. | A 20% shorter REC phase saves 15.88 ms / 14.2% on Small dense and 64.20 ms / 16.5% on Medium dense. |
| 1 for sparse pages | Optimize Medium's 256→64 and 128→64 spatial 3×3 shapes, including data preparation and stores. | Two steps consume 21.3% of the Medium detector probe; DET dominates sparse latency. | A 20% shorter DET phase saves 40.67 ms / 16.0% on Medium sparse. |
| 2 for Medium dense | Compare one bounded owner-count or shape-cost policy using the same page and recorded owner costs. | Nearly balanced width sums have a large finish spread. | No numeric gain is inferred from the spread; core placement and resource effects are unresolved. |
| 1 for Tiny dense | Isolate the fixed classifier backbone, then compare one small classifier batch. | CLS is 17.80 ms / 32.7% of Tiny dense latency. | Halving CLS saves 8.90 ms / 16.4% on that profile. |
| 3 | Optimize compact vocabulary projection, especially Small. | It is 13.1% of Small dense backbone-plus-projection work after the store improvements. | Needs a production projection measurement; its summed work is not directly removable wall time. |

Exact-width recognition batching is already supported and can be evaluated as one alternative for repeated-width dense crops, with the same output and memory checks. Disabling classification, lowering detector resolution, reducing recognition width or changing models are separate accuracy/sampling tradeoffs. None is established as a safe default by these timing results. DB geometry, CTC, extra shape caching and result locking should remain lower priorities for the measured workload.

**Validation and execution budget.** Fresh local release validation passed **12 focused tests**: four prepared-store tests, one fused panel-store regression test and seven profiling/accounting tests. The latter cover ordinary/profiled result parity and serial/parallel owner accounting. All six new page profiles match their warm-up outputs exactly; all five new operator shapes match warm-up tensors bit-for-bit. All nine complete profile-output hashes also match the previously recorded pre-store baseline. The existing exact-source evidence additionally records 405 default checks, 343 serial library tests, 360 profiling library tests, clippy, x86 compilation and 213/213 reference qualification requests; these broader suites were not rerun for this documentation-only analysis.

New performance execution was **8.786 seconds**, including process startup, asset checks/loading, warm-ups and measurements. Adding the prior source-version charge, including its conservative prototype allowance, gives **22.466 seconds**, below the 60-second budget. Six new stage cases each used one warm-up and one measured profile; the three detector shapes and two recognition shapes also each used one warm-up and one measured pass. All performance subprocesses ran sequentially, after builds and correctness checks completed, with a 12-second per-command cap and the remaining aggregate budget. No timing run was repeated for cleaner numbers.

Three ordinary samples and single diagnostic passes establish local bottlenecks, not p95/p99 latency or universal speedups. This report applies to the physical M4 and the stated single-request configuration; it does not establish Windows/x86, other ARM chips, throughput under concurrent requests, arbitrary image sizes or ground-truth OCR accuracy.

**Reproduction.** Local artifacts are in `target/pipeline-analysis-20261018-panel-stores`; the portable JSON embeds the measurement and packaging scripts. Build and validate before running any timing workload:

```sh
cargo test --release --locked --lib optimized::store::tests -- --test-threads=1
cargo test --release --locked --lib neon_panel_stores -- --test-threads=1
cargo build --release --locked --example cpu_benchmark
cargo build --release --locked --features profiling --example cpu_profile
cargo test --release --locked --features profiling --lib profile::tests -- --test-threads=1
```

For one missing production stage case, run `target/release/examples/cpu_profile small imgs_00006737 8`. The ignored `sparse_detector_production_steps` and `recognition_backbone_physical_steps` tests contain the fixed operator probes. The recorded runner invokes the already-built test binary to keep build activity outside measurement. Ordinary page records were produced with `cpu_benchmark _worker JOB.json`; the job objects and every raw sample are included in the evidence. Use external deadlines, count all warm-ups and internal repetitions, and do not rerun these unchanged cases merely to refine their numbers.
