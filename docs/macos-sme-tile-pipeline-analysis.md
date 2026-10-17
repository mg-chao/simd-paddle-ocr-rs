# Apple M4 OCR latency after SME tile optimization

The subsequent [SME packing and crossover optimization](macos-arm64-sme-packing-optimization.md) implements the smaller-channel convolution recommendations and records its measured gains, memory tradeoff and validation. Measurements below describe the preceding `83fc0c7` implementation.

At revision `83fc0c71580f473d9933fc7f4d49a5a6193c8c0e`, **detector inference accounts for 69.8–77.4% of sparse-page latency, while recognition accounts for 55.8–80.8% on dense pages**. There is further optimization potential in convolution execution and recognition scheduling. Tiny dense orientation classification is also substantial at **28.5%**. DB geometry, CTC decoding, warm shape binding and result copying have much smaller absolute costs on these inputs.

This analysis covers the implementation after the [32×32 SME tile and crossover changes](macos-arm64-sme-tile-optimization.md). The [preceding analysis](macos-sme-pipeline-analysis.md) describes the earlier 16×16 implementation; its below-threshold 192↔384 recommendation is already implemented. The new priority for Small includes narrower 96→192 layers, while Medium still spends most recognition arithmetic in larger SME-eligible pointwise layers. No production code or accuracy settings were changed for this analysis.

[Portable evidence](reports/macos-arm64-2026-10-17-head-83fc0c7-pipeline-analysis.json) contains all nine current page results, every stage counter, worker records, five new operator probes, source/binary verification, local correctness logs and reproduction scripts.

The machine is a physical Apple M4 with 10 cores and 16 GiB RAM, macOS 27.0.1, running natively on AC power. Release builds use the pinned `nightly-2026-09-08` toolchain, eight shared workers, one request, automatic SIMD/layout/fusion and line ownership, detector limit 960, adaptive recognition widths, recognition batch size one, classification enabled and alignment disabled. macOS scheduling is unpinned. Core placement and clock frequency were not measured.

The inputs are sparse `imgs_1` (720×1150), dense `imgs_00018069` (430×267), and mixed `imgs_00006737` (896×528). Detector tensors are respectively 608×960, 416×256 and 896×544. All three shipped bundles are included because their network shapes and bottlenecks differ; other settings are fixed. The bundles can produce different text and line counts, so their latency differences do not establish equal accuracy.

**Ordinary page latency.** These are the current version's existing local measurements from 17 October 2026. Rebuilt standard and profiling executables match their saved executable SHA-256 hashes byte-for-byte; all current Rust source hashes match the recorded final source. The benchmark harness, build configuration, manifest, model assets and selected image files were also verified. These measurements were reused rather than repeated. Each case has a separately recorded first call, one explicit warm-up and three measured calls. The first call additionally warms process/request state. Timing includes conversion to complete owned results and begins with predecoded BGR24.

| Bundle / page | Lines | Median ms | Range ms | Load ms | First call ms | Peak RSS MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Tiny / sparse | 3 | 46.90 | 46.31–47.11 | 14.45 | 127.93 | 68.80 |
| Tiny / dense | 69 | 63.37 | 63.35–86.30 | 6.23 | 71.23 | 45.00 |
| Tiny / mixed | 31 | 74.85 | 73.45–75.58 | 6.15 | 75.83 | 76.84 |
| Small / sparse | 2 | 82.83 | 81.21–85.41 | 20.23 | 86.44 | 148.55 |
| Small / dense | 69 | 138.33 | 136.40–138.62 | 19.92 | 142.27 | 119.03 |
| Small / mixed | 33 | 172.19 | 170.72–173.05 | 20.19 | 178.98 | 179.80 |
| Medium / sparse | 2 | 303.82 | 302.60–310.43 | 92.95 | 316.46 | 455.33 |
| Medium / dense | 69 | 433.83 | 425.75–449.32 | 92.25 | 430.30 | 357.83 |
| Medium / mixed | 31 | 579.25 | 557.83–583.09 | 86.97 | 615.62 | 481.59 |

Load includes model import, compilation, CPU context and initial request creation. It excludes first inference; file caches were not flushed, so it is not disk-cold loading. Tiny sparse has a 127.93 ms first-call observation and Tiny dense has an 86.30 ms measured sample. Both are retained. The earlier paired comparison also showed Tiny regressions whose cause remains unresolved; this analysis does not replace those observations with the more favorable direct-candidate round. Three samples are insufficient for p95/p99 or small-effect significance.

**Elapsed stages in milliseconds.** Four existing profiles were reused: Tiny sparse, Small dense, Medium dense and Medium mixed. Five missing profiles were measured once each after one warm-up. Each checks complete owned-output equality against its uninstrumented warm-up. The following profiles are separate runs from the ordinary samples above; they describe stage distribution and must not be substituted into, or added to, an ordinary median.

| Bundle / page | DET prep | DET infer | DB | Crop | CLS | REC | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Tiny / sparse | 3.44 | 30.23 | 0.42 | 0.34 | 1.39 | 7.51 | 43.34 |
| Tiny / dense | 0.54 | 8.32 | 0.34 | 0.64 | 17.99 | 35.21 | 63.06 |
| Tiny / mixed | 2.05 | 24.01 | 0.51 | 1.41 | 8.41 | 36.93 | 73.35 |
| Small / sparse | 2.85 | 58.97 | 0.46 | 0.42 | 1.45 | 20.38 | 84.55 |
| Small / dense | 0.68 | 16.94 | 0.29 | 0.78 | 20.46 | 117.43 | 156.60 |
| Small / mixed | 2.30 | 49.51 | 0.55 | 1.51 | 9.23 | 109.31 | 172.45 |
| Medium / sparse | 3.68 | 226.98 | 0.48 | 0.53 | 1.47 | 59.93 | 293.08 |
| Medium / dense | 0.74 | 57.75 | 0.31 | 0.86 | 20.38 | 336.26 | 416.34 |
| Medium / mixed | 2.84 | 201.41 | 0.52 | 1.46 | 9.94 | 355.98 | 572.18 |

The six elapsed phases are sequential in this configuration. Their unrounded sum leaves 0.014–0.038 ms of validation, binding and orchestration inside total. Profile total excludes the subsequent owned copy. Small dense's profile total is 156.60 ms versus an ordinary median of 138.33 ms: instrumentation, scheduling and separate execution can differ materially. This difference is not an isolated measurement of timer overhead. Source-level stage coverage and output checks pass, but single-pass substage times remain diagnostic.

The full library path is borrowed pixels → DET resize/normalization → detector network → DB boxes → perspective crops → orientation/rotation → REC resize/backbone/projection/CTC → ordered results. The [CLI example](../examples/ocr.rs) additionally reads/decodes the image and serializes its output. Those application costs, optional token boxes and [engine admission waiting](../src/engine.rs) are outside these inference measurements, rather than assumed to cost zero.

**Nested classification work in milliseconds.** These counters sum elapsed calls across workers and overlap the CLS wall phase. They are not extra wall time or process CPU time.

| Bundle / page | Resize / normalize | Backbone | Head | Pixel rotation |
| --- | ---: | ---: | ---: | ---: |
| Tiny / sparse | 0.3885 | 3.6564 | 0.0046 | 0.0211 |
| Tiny / dense | 7.7800 | 111.6535 | 0.1213 | 0.0000 |
| Tiny / mixed | 4.3669 | 49.7607 | 0.0540 | 0.0006 |
| Small / sparse | 0.3005 | 2.5725 | 0.0037 | 0.0195 |
| Small / dense | 8.6431 | 120.1694 | 0.1239 | 0.0000 |
| Small / mixed | 5.2039 | 53.2859 | 0.0587 | 0.0008 |
| Medium / sparse | 0.3136 | 2.5808 | 0.0040 | 0.0243 |
| Medium / dense | 9.2648 | 121.6546 | 0.1245 | 0.0000 |
| Medium / mixed | 5.1515 | 53.7244 | 0.0562 | 0.0060 |

Classification uses fixed 80×160 input and a feature prefix plus compact two-class head. In the dense profiles its backbone is about 93% of measured orientation worker work. No dense lines need 180-degree rotation, but sparse profiles rotate all 3/2/2 lines and mixed profiles rotate 1/2/3. Disabling CLS therefore requires an application contract that guarantees upright inputs and an accuracy check on that workload; it is not a transparent optimization of this corpus.

**Nested recognition work in milliseconds.** Recognition backbone, projection and preprocessing overlap across owners, so their summed durations can exceed the REC wall phase.

| Bundle / page | Resize / normalize | Backbone | Projection | CTC |
| --- | ---: | ---: | ---: | ---: |
| Tiny / sparse | 0.8286 | 14.0232 | 0.8916 | 0.0025 |
| Tiny / dense | 5.0947 | 217.7958 | 16.5735 | 0.0174 |
| Tiny / mixed | 6.7517 | 232.2454 | 16.2492 | 0.0221 |
| Small / sparse | 0.8970 | 32.5947 | 2.3718 | 0.0039 |
| Small / dense | 6.7624 | 751.3717 | 84.8407 | 0.0465 |
| Small / mixed | 7.6950 | 731.1895 | 66.7363 | 0.0225 |
| Medium / sparse | 0.9216 | 94.3569 | 5.2443 | 0.0049 |
| Medium / dense | 6.1265 | 2169.2245 | 116.9723 | 0.0513 |
| Medium / mixed | 7.4440 | 2103.1884 | 109.2703 | 0.0338 |

The compact vocabulary projection already avoids materializing the complete vocabulary tensor. It contributes 7.1%, 10.1% and 5.1% of backbone-plus-projection work for dense Tiny, Small and Medium, respectively. It is a secondary arithmetic target, especially for Small after the backbone improvements. CTC remains at or below 0.052 ms summed work across all nine profiles. The raw evidence also preserves crop-worker work separately from crop wall time.

**Binding and output accounting in milliseconds.** Binding, reshape and storage are nested work counters; only owned copy is a separate post-profile elapsed measurement. DET and REC reshape are summed together. CLS has a fixed shape bound during request creation.

| Bundle / page | Binding | DET + REC reshape | Result storage | Owned copy |
| --- | ---: | ---: | ---: | ---: |
| Tiny / sparse | 0.0015 | 0.0004 | 0.0011 | 0.0018 |
| Tiny / dense | 0.0178 | 0.0264 | 0.0127 | 0.0084 |
| Tiny / mixed | 0.0085 | 0.0230 | 0.0053 | 0.0073 |
| Small / sparse | 0.0024 | 0.0008 | 0.0016 | 0.0020 |
| Small / dense | 0.0408 | 0.0788 | 0.0256 | 0.0053 |
| Small / mixed | 0.0231 | 0.0448 | 0.0104 | 0.0115 |
| Medium / sparse | 0.0012 | 0.0007 | 0.0017 | 0.0026 |
| Medium / dense | 0.0904 | 0.0917 | 0.0287 | 0.0118 |
| Medium / mixed | 0.0603 | 0.0800 | 0.0166 | 0.0037 |

Warm reshape remains below 0.092 ms summed work. The [runtime](../src/runtime.rs) already retains eight previous shape plans and activation high-water buffers. More shape caching, lock removal around ordered output slots, or a CTC rewrite cannot address the dominant latency on these pages. Separate setup profiles observe import/compile-request ranges of 3.55–3.69/1.91–2.07 ms for Tiny, 10.33–12.88/8.68–9.24 ms for Small and 39.54–39.69/47.62–50.24 ms for Medium. Reuse models, compiled requests and the CPU pool instead of rebuilding them per image.

Source review connects each stage to a concrete remaining opportunity:

| Stage | Current implementation | Implication |
| --- | --- | --- |
| Input and setup | [ImageView](../src/image.rs) borrows validated pixels; the [builder](../src/builder.rs) shares immutable models; [engine leases](../src/engine.rs) retain bounded mutable requests. | Setup amortization matters for short-lived callers. Service queueing needs a separate workload measurement. |
| DET preparation | [Preprocessing](../src/preprocess.rs) caches interpolation maps/rows, uses fixed-point byte interpolation and lookup-table normalization, writes the selected layout directly, and partitions rows. | The vertical interpolation/table/store loop and transient task allocation are secondary targets; the whole measured phase is only 0.54–3.68 ms. Their individual costs are not isolated. |
| Detector network | [Optimized execution](../src/runtime/optimized.rs) already fuses eligible operations and stores, aliases tensors and reserves reusable arenas/scratch. | Target the measured convolution shapes below. Adding generic fusion or SIMD duplicates existing work. |
| DB boxes | [DB](../src/db.rs) thresholds/scans probabilities, finds components/contours, forms hulls and rectangles, scores/unclips boxes and sorts reading order, with retained scratch. | 0.29–0.55 ms here. Defer a geometry rewrite unless another corpus shows expensive contour behavior. |
| Perspective crop | [Line extraction](../src/lines.rs) writes a retained BGR slab in 16-row tasks; [crop kernels](../src/crop.rs) use SIMD coordinate mapping and cubic interior sampling with clamped borders and folded geometric rotation. | 0.34–1.51 ms. Direct crop-to-tensor fusion changes the cubic-byte-then-bilinear numerical path and requires correctness/accuracy evidence. |
| Orientation | Up to eight single-thread owners classify independent lines round-robin. All classification completes before recognition starts. | Tiny dense's 17.99 ms wall phase makes CLS backbone optimization or real classifier batching useful candidates. REC batch size does not batch CLS. Overlapping stages would require shared-pool and memory evaluation. |
| REC dimensions | [Recognition](../src/recognizer.rs) uses height 48, adaptive width rounded to 32, and exact-width batches when configured. | Dense lines are mostly width 96/128; mixed pages reach 1472/1536. Reducing width or fixing 320 changes long-text sampling and needs ground-truth accuracy testing. |
| REC scheduling | [OCR scheduling](../src/ocr.rs) assigns descending-width units to stable least-loaded owners. Dense pages use eight owners with one kernel thread each; the dominant single-crop owner on mixed pages can use two. | Similar width sums still produce different finish times. Investigate measured cost and concurrent execution while retaining bounded memory and deterministic output order. |
| REC convolution | [Pointwise dispatch](../src/runtime/nhwc/pointwise.rs) admits SME when executing rows ≥32, both channel dimensions ≥192, and one dimension ≥384. [SME](../src/runtime/nhwc/pointwise/sme.rs) packs 64 rows, computes 32×32 blocks, handles tails with 16×16 tiles, and applies fused stores. | Wider tiles and 192↔384/256↔512 crossover are already present. Isolate remaining packing, guard/streaming transitions and result staging before changing them; smaller-channel layers still use NEON. |
| Projection, decode and results | [Compact projection](../src/recognizer.rs) computes class winners; reusable CTC/text storage feeds reading-order assembly and optional owned copies. | Projection has a measurable secondary share. Decoder/storage costs are too small to lead the next optimization round. |

**Fresh operator diagnostics.** Five probes retain production fusion, layouts and kernels. DET uses the real sparse image at `[1,3,960,608]` with eight threads. REC runs the feature prefix with deterministic synthetic input at the stated width and one thread; the vocabulary head is excluded. Each has one warm-up and one instrumented pass, with bit-identical network outputs. The public unfused trace path was not used for these timings.

| Fused operator probe | All steps ms | Pointwise ms | Pointwise share | Layout conversion ms |
| --- | ---: | ---: | ---: | ---: |
| Tiny DET sparse, 8 threads | 29.551 | 11.971 | 40.5% | 0.0000 |
| Small DET sparse, 8 threads | 59.542 | 25.869 | 43.4% | 2.4037 |
| Medium DET sparse, 8 threads | 233.243 | 102.085 | 43.8% | 0.0000 |
| Small REC width 128, 1 thread | 6.499 | 4.742 | 73.0% | 0.0025 |
| Medium REC width 640, 1 thread | 74.447 | 59.293 | 79.6% | 0.0099 |

For **Small REC**, the leading operation is a 96→48 stride-two 3×3 convolution at 0.336 ms. Three 96→192 pointwise layers over 384 spatial rows take 0.282–0.313 ms each and remain below the current SME gate. These are the next concrete NEON/crossover candidates. Eligible 192→384 and 384→768 layers still appear among the larger costs, but extending the old ≥384 gate is no longer outstanding work.

For **Medium REC**, repeated 512→1024 pointwise layers over 960 rows take 3.10–3.62 ms each; a 768→1536 layer takes 2.91 ms. These already pass SME eligibility. Input packing, repeated streaming entries and staged fused stores are code-level candidates, but this probe does not split their costs or establish hardware contention. MatMul, softmax, layer normalization and transpose together contribute about 1.9% of the Small probe and 2.3% of Medium; attention is not the leading measured opportunity.

For **Tiny DET**, a 64→16 3×3 convolution takes 2.07 ms and two 32→64 pointwise layers take 1.73/1.86 ms. **Small DET** spends 4.90 ms on 96→24 3×3, 2.64/2.95 ms on 48→96 pointwise and 2.55 ms on its first transposed convolution. Its layout conversions total 2.40 ms, or 4.0% of the probe. These smaller-channel kernels require different work from the existing large-channel SME path.

For **Medium DET**, 256→64 3×3 costs 27.06 ms, 128→64 stride-two 3×3 costs 12.34 ms, and two 128→256 pointwise layers cost 11.20/10.72 ms. A 256-channel depthwise 9×9 layer costs 10.22 ms. These provide focused dense-convolution, narrow-pointwise and depthwise targets. The 233.24 ms step total is reasonably close to the separate 226.98 ms DET phase, but neither is a paired kernel speedup measurement. Historical operator numbers from the prior implementation are not pooled with these probes.

**Recognition completion imbalance.** The following finish times share a dispatch origin and come from the nine production stage profiles. Width cost is the sum of assigned tensor widths. The longest-crop column applies to mixed pages, whose longest-crop owner has two kernel threads.

| Bundle / page | Width cost range | Finish range ms | Longest crop finishes ms | Last owner max width |
| --- | ---: | ---: | ---: | ---: |
| Tiny / dense | 1024–1120 | 23.77–35.16 | — | 192 |
| Tiny / mixed | 1088–1536 | 25.07–36.92 | 35.99 | 672 |
| Small / dense | 1024–1120 | 91.51–117.40 | — | 224 |
| Small / mixed | 1024–1472 | 91.62–109.29 | 108.77 | 288 |
| Medium / dense | 992–1088 | 249.60–336.17 | — | 192 |
| Medium / mixed | 1056–1472 | 196.37–355.89 | 339.50 | 352 |

Medium dense owners have width costs differing by 9.7%, yet finish between 249.60 and 336.17 ms. Medium mixed finishes between 196.37 and 355.89 ms; the 1472-wide crop finishes at 339.50 ms, 16.39 ms before the last owner, whose largest crop is only 352 pixels wide. That last owner's calling-thread CPU time is 354.59 ms. The mismatch supports a scheduling investigation, but does not prove that P/E-core placement, memory bandwidth, the cost estimator or SME contention is the cause. Thread CPU time excludes nested helpers and does not distinguish useful arithmetic from active stalls. The finish spread is not an achievable speedup estimate.

Record per-owner shape/kernel selection and execution placement where available, then test one bounded alternative against eight owners. A calibrated deterministic cost model or exact-width batching may help, but fewer owners or batching can also lose independent work. Unrestricted work stealing can make identical pages grow different worker-private arenas and shape caches. Preserve warm-storage bounds when evaluating any scheduling change.

| Bundle / mixed page | Retained activations MiB | Retained workspace MiB | Packed weights MiB |
| --- | ---: | ---: | ---: |
| Tiny | 41.79 | 10.67 | 6.79 |
| Small | 84.79 | 21.77 | 30.41 |
| Medium | 175.57 | 25.81 | 132.20 |

Medium mixed peaks at 481.59 MiB process RSS. RSS includes setup, weights, allocator effects and shared resident pages, whereas retained counters measure specific library capacities. Additional requests and workers can multiply private storage. A recognition latency improvement should therefore report retained activation/workspace growth as well as page time.

The next optimization order should follow the workload:

1. **Small/Medium dense or mixed:** optimize the measured pointwise shapes and investigate Medium owner completion. Start with Small rows/K/N = 384/96/192 and Medium 960/512/1024, including packing and fused stores. Keep the existing channel crossover unless a bounded paired probe supports changing it.
2. **Sparse pages:** prioritize detector 3×3 and narrow pointwise convolutions; Medium's 256→64 3×3 is a particularly large individual target. Evaluate existing strip/tile reuse on the measured 9×9 depthwise shape before extending its implementation.
3. **Tiny dense:** target the classifier backbone or classifier batching. Half of its CLS wall time would save 8.99 ms, about 14.3% of its profiled page time, if all other work stayed fixed.
4. **Secondary:** compact vocabulary projection, detector normalization and verified exact-width recognition batching. A smaller model, lower detector limit or disabled classification requires a separate accuracy/latency decision.
5. **Defer on this corpus:** DB geometry, CTC, result copying and additional warm shape caching.

For scale, a hypothetical 20% reduction of the entire REC phase saves 7.04/23.49/67.25 ms on dense Tiny/Small/Medium, or 11.2%/15.0%/16.2% of those profile totals. A 20% DET inference reduction saves 6.05/11.79/45.40 ms on sparse pages, or 14.0%/14.0%/15.5%. These are sensitivity calculations with other stages fixed, not measured candidate gains. They show why arithmetic and scheduling deserve attention before sub-millisecond stages.

Local validation ran 16 focused correctness checks: three default profile-accounting tests, six SME/selection/scratch tests, and seven profiling-enabled accounting/output tests. All passed. The two diagnostic test commands cover five network shapes and preserve output bits; the five new OCR profiles preserve complete owned outputs. The matching revision's earlier full 213-request reference qualification is retained in the linked optimization report; it was not rerun for this analysis-only change.

New performance execution was **6.205 seconds**, including process startup, loading and all warm-up/measured calls. Combined with the already recorded **21.652 seconds** for this version, execution totals **27.857 seconds**, below the 60-second budget. New work comprises five OCR warm-ups plus five profiled OCR calls, and five network warm-ups plus five operator-profile calls. Existing page evidence has nine cases × three measured samples, separate first calls and warm-ups; four stage profiles are reused. The version's earlier candidate-selection calls remain charged to the budget but are not pooled into the page table. Builds and correctness checks are excluded. Each new subprocess had a 12-second cap and the remaining aggregate budget; there were no retries or automatic sampling loops.

Reproduction sources and raw logs are in `target/pipeline-analysis-20261017-head-83fc0c7`; the JSON embeds the scripts so the method survives removal of local target artifacts. The scripts verify the retained comparison artifacts and refuse to overwrite a measurement marker. For a new version, rebuild and use a fresh evidence directory with a new run count. The commands used for focused correctness and binary construction are:

```sh
cargo test --release --locked --lib profile::tests -- --test-threads=1
cargo test --release --locked --lib sme -- --test-threads=1
cargo test --release --locked --features profiling --lib profile::tests -- --test-threads=1
cargo build --release --locked --example cpu_benchmark
cargo build --release --locked --features profiling --example cpu_profile
```

These results establish bottlenecks on one physical M4, three representative pages and one thread/concurrency configuration. They do not establish native x86 performance, service throughput, ground-truth OCR accuracy, tail latency or a guaranteed speedup for the proposed changes.
