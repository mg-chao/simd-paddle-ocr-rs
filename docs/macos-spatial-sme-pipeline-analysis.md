# Apple M4 pipeline latency after spatial SME convolution

The current implementation still has meaningful optimization opportunities in network execution. **Recognition consumes 73.1% of Small dense-page latency and 80.3% of Medium dense-page latency; orientation classification consumes 34.0% of Tiny dense-page latency.** Sparse pages spend 69.9–74.4% in the detector network. The next work should target pointwise convolution and recognition owner imbalance, the ARM64 transposed-convolution store path, and the remaining spatial/depthwise layers. DB postprocessing, CTC, shape-cache lookup and result copying have much less potential.

This report analyzes clean production source at `2316ab0` on 2026-10-18, including the [spatial SME optimization](macos-arm64-spatial-sme-optimization.md). No production or test source was changed. The [portable evidence](reports/macos-arm64-2026-10-18-spatial-sme-pipeline-analysis.json) contains every page sample, stage counter, owner record, operator timing, source/executable hash, validation log and reproduction script.

**Scope and measurement method.** Physical Apple M4, four performance cores and six efficiency cores, 16 GiB RAM, macOS 27.0.1, AC power, native ARM64 release builds and pinned `nightly-2026-09-08`. One active request shares eight CPU workers. SIMD/layout/fusion and line ownership are automatic; recognition batch size is one, widths are adaptive, detector limit is 960, classification is enabled with threshold zero, and alignment and dilation are off. Model-specific box thresholds and remaining manifest settings are unchanged. macOS scheduling is unpinned; worker indices are not physical-core identities. No thermal/performance warning was recorded before or after measurement; frequencies and temperatures were not measured.

All 112 Rust source hashes match the final spatial-SME evidence. Both rebuilt ordinary/profile executables are byte-identical to those used for its final measurements; examples, harness, Cargo files and toolchain also match. Therefore nine ordinary page benchmarks and three stage profiles are reused. Six missing stage profiles and five fused-operator shapes were measured locally for this analysis. New and reused evidence retain separate origins. No samples from different source versions or measurement methods are pooled.

The three representative images cover different bottlenecks: sparse `imgs_1` is 720×1150 pixels and produces a 608×960 detector tensor with 2–3 lines; dense `imgs_00018069` produces a 416×256 detector tensor with 69 lines; mixed `imgs_00006737` produces an 896×544 detector tensor with 31–33 lines. Dense recognition crops mainly have tensor widths 96/128. Mixed pages include width 1536 for Tiny and 1472 for Small/Medium. Recognition height is 48. Each bundle can detect different boxes, so these are not identical network workloads or an accuracy ranking.

**Ordinary complete OCR latency.** Three fixed measured calls per case, one explicit warm-up, and a separately recorded first inference. Results are owned; pixels are predecoded BGR24. Setup, image-file I/O/decoding, serialization, application queues and engine-lease waiting are outside these timers. The first inference includes lazy initialization and first-use effects; it is not a cold-disk measurement.

| Bundle / page | Lines | Warm median ms | Range ms | First call ms |
| --- | --- | --- | --- | --- |
| Tiny / sparse | 3 | 35.45 | 34.98–35.63 | 46.98 |
| Tiny / dense | 69 | 53.19 | 49.96–54.88 | 56.08 |
| Tiny / mixed | 31 | 60.67 | 59.12–61.31 | 62.85 |
| Small / sparse | 2 | 67.02 | 66.33–69.07 | 71.62 |
| Small / dense | 69 | 113.17 | 107.59–116.04 | 116.54 |
| Small / mixed | 33 | 145.16 | 142.29–147.94 | 138.27 |
| Medium / sparse | 2 | 228.84 | 225.16–229.96 | 224.15 |
| Medium / dense | 69 | 341.17 | 330.51–357.38 | 327.69 |
| Medium / mixed | 31 | 456.58 | 446.90–465.87 | 453.00 |

These current-source samples were already collected by the spatial-SME experiment and were not repeated to obtain cleaner numbers. Three samples do not estimate p95/p99 or establish statistical significance. A first call being faster than later samples, as on some Medium cases, illustrates scheduling variation rather than negative initialization cost.

**Sequential pipeline phases, elapsed milliseconds.** Tiny sparse, Small dense and Medium mixed profiles are reused; the other six are new. Each has one ordinary warm-up and one instrumented production call with complete owned-output equality. The six columns below are sequential wall phases in this configuration.

| Bundle / page | DET prep | DET network | DB boxes | Crop | CLS + rotation | REC + assembly | Total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Tiny / sparse | 2.82 | 25.17 | 0.40 | 0.37 | 1.19 | 5.92 | 35.88 |
| Tiny / dense | 0.66 | 8.14 | 0.31 | 0.79 | 18.88 | 26.74 | 55.53 |
| Tiny / mixed | 2.60 | 24.69 | 0.53 | 1.42 | 8.40 | 26.58 | 64.24 |
| Small / sparse | 3.65 | 55.40 | 0.56 | 0.49 | 1.19 | 17.98 | 79.31 |
| Small / dense | 0.54 | 12.73 | 0.28 | 0.65 | 16.83 | 84.44 | 115.49 |
| Small / mixed | 2.48 | 46.53 | 0.59 | 1.24 | 9.08 | 86.78 | 146.72 |
| Medium / sparse | 3.91 | 169.43 | 0.59 | 0.59 | 2.12 | 50.98 | 227.64 |
| Medium / dense | 0.81 | 46.24 | 0.34 | 1.18 | 21.74 | 286.46 | 356.79 |
| Medium / mixed | 2.79 | 137.30 | 0.62 | 1.55 | 8.55 | 299.55 | 450.38 |

Unrounded phase sums leave only 0.012–0.024 ms in total for outer validation, orchestration and preparation. Recognition includes scheduling, joins and ordered assembly. Owned conversion is timed separately after profile total. Instrumented totals are not components of the ordinary medians: Small sparse, for example, has a 79.31 ms diagnostic total versus a 67.02 ms ordinary median from a separate process/run. This difference cannot isolate instrumentation overhead, scheduling or a regression. The single profiles identify where time goes, not a precise repeatable latency for every stage.

```mermaid
flowchart LR
    A[Borrowed raw pixels] --> B[Resize and normalize]
    B --> C[Detector network]
    C --> D[DB boxes and reading order]
    D --> E[Perspective crops]
    E --> F[Orientation classification]
    F --> G[Recognition backbone]
    G --> H[Compact projection and CTC]
    H --> I[Ordered borrowed or owned results]
```

**Implementation review of each stage.**

| Stage | Current code path | Performance implication |
| --- | --- | --- |
| Input and lifetime | [ImageView](../src/image.rs) borrows raw pixels; [builder](../src/builder.rs) shares models; [engine](../src/engine.rs) retains mutable requests under a bounded pool. | Reuse loaded models, CPU context and warmed requests. JPEG/PNG decode is outside the library and unmeasured here. Service queueing needs a separate workload. |
| DET resize/normalization | [Preprocessing](../src/preprocess.rs) uses fixed-point separable resize, cached horizontal coefficients/rows, normalization lookup tables and disjoint row bands, writing directly in the requested layout. | 0.54–3.91 ms wall. A secondary Tiny/sparse target; allocation or additional SIMD changes should be justified by focused measurements. Lowering detector resolution changes detection behavior. |
| DET graph | [Runtime](../src/runtime.rs) and [optimized plan](../src/runtime/optimized.rs) reuse packed weights and activations, select layouts, fuse operations, and dispatch NEON/guarded SME kernels. | The main sparse-page cost. Existing 3×3 SME acceleration is already included; repeating that proposal would miss the current hotspots. |
| DB boxes | [DB](../src/db.rs) uses SIMD threshold/interior masks, component/contour scanning, scoring, geometry/unclip and reading-order sorting. | Only 0.28–0.62 ms total. These substages are covered by one aggregate timer; their individual costs were not instrumented. |
| Perspective crops | [Crop](../src/crop.rs) uses SIMD coordinate/interior sampling with clamped border fallback; [line extraction](../src/lines.rs) partitions a retained BGR slab into 16-row jobs. | 0.37–1.55 ms wall. Bicubic sampling and geometric vertical rotation belong here; 180° classifier correction belongs to CLS. |
| Orientation | [Classifier](../src/orientation.rs) runs a fixed `[1,3,80,160]` feature prefix, a compact two-class head and the public probability calculation. Crops are assigned round-robin to up to eight single-threaded classifier workers. | 16.83–21.74 ms on 69-line pages. The network, not pixel rotation or the two-class head, is the expensive part. |
| REC scheduling | [OCR](../src/ocr.rs) sorts descending tensor widths, batches only equal widths, and greedily assigns immutable jobs by width × batch count. [CPU budgets](../src/cpu.rs) select eight owners for sufficiently populated pages. | Stable private buffers avoid repeated arena growth. Width cost does not fully predict owner completion; see the records below. |
| REC resize/backbone | [Recognizer](../src/recognizer.rs) selects adaptive widths, reshapes a cached plan, normalizes directly into the graph input and executes the feature prefix. | The dominant dense/mixed Small/Medium cost. Sparse requests use fewer owners with more kernel threads; dense owners normally use one each. |
| Projection and CTC | [Compact projection](../src/runtime/simd/argmax.rs) reuses packed weight panels and keeps maxima/indices; [CTC](../src/recognizer.rs) collapses blanks/repeats and optionally records spans. | Full vocabulary activations and final vocabulary softmax are already avoided in production. Projection is measurable; CTC is negligible here. Production scores are uncalibrated projection maxima, so an optimization must preserve their existing semantics. |
| Storage/output | [OCR assembly](../src/ocr.rs) restores detection order into reusable text/metadata slabs and returns borrowed views; `try_to_owned` copies on request. | Warm assembly and copying are microsecond-scale. Removing those copies cannot materially improve these page latencies. |

**Nested classification work, milliseconds summed across workers.** These columns overlap in time between workers and must not be added to the wall-phase table. They include elapsed worker intervals, not a process-CPU measurement.

| Bundle / page | CLS resize | CLS backbone | Two-class head | 180° rotation | Rotated lines |
| --- | --- | --- | --- | --- | --- |
| Tiny / sparse | 0.3653 | 3.0819 | 0.0043 | 0.0168 | 3 |
| Tiny / dense | 8.4970 | 111.0109 | 0.1210 | 0.0000 | 0 |
| Tiny / mixed | 4.6656 | 46.5056 | 0.0541 | 0.0005 | 1 |
| Small / sparse | 0.2887 | 2.0602 | 0.0035 | 0.0187 | 2 |
| Small / dense | 7.9210 | 101.7814 | 0.1208 | 0.0000 | 0 |
| Small / mixed | 5.5495 | 49.6548 | 0.0570 | 0.0007 | 2 |
| Medium / sparse | 0.5310 | 2.8218 | 0.0047 | 0.0212 | 2 |
| Medium / dense | 8.8760 | 113.6320 | 0.1258 | 0.0000 | 0 |
| Medium / mixed | 4.8645 | 46.9943 | 0.0556 | 0.0055 | 3 |

The dense page applies no 180° rotations, yet its classifier backbones consume 101.78–113.63 ms of summed work. This rules out rotation and the two-class head as useful primary targets. For Tiny dense, a hypothetical 50% reduction in the complete CLS wall phase would save 9.44 ms, or 17.0% of profiled page latency. That is sensitivity, not a measured achievable gain. A bounded fixed-shape classifier-kernel or small-batch experiment is warranted for that workload. Disabling classification is an application option only when orientation is guaranteed upstream: the sparse fixtures rotate every detected line, and mixed fixtures rotate 1–3 lines. A faster classifier must preserve scores, decisions and image orientation.

**Nested recognition work, milliseconds summed across owners.** Backbone/projection values can greatly exceed page wall time because owners run concurrently; a multithreaded kernel is timed as elapsed time on its calling owner, not the sum of its helper-thread CPU time.

| Bundle / page | REC resize | REC backbone | Projection | CTC |
| --- | --- | --- | --- | --- |
| Tiny / sparse | 0.7924 | 10.8768 | 0.9576 | 0.0027 |
| Tiny / dense | 6.0411 | 154.1433 | 19.3357 | 0.0194 |
| Tiny / mixed | 8.1564 | 165.6407 | 20.4998 | 0.0243 |
| Small / sparse | 0.8118 | 29.2632 | 2.9977 | 0.0039 |
| Small / dense | 5.7134 | 507.1121 | 76.2135 | 0.0348 |
| Small / mixed | 7.7767 | 532.1530 | 74.7186 | 0.0274 |
| Medium / sparse | 0.9617 | 91.6475 | 4.5223 | 0.0053 |
| Medium / dense | 6.4901 | 1690.8336 | 115.7868 | 0.0750 |
| Medium / mixed | 8.4144 | 1746.3093 | 131.5106 | 0.0400 |

Projection accounts for 11.1%, 13.1% and 6.4% of dense Tiny/Small/Medium backbone-plus-projection work. It is a secondary target, especially for Small: consider packed row/column reuse or a compact SME variant only if it still avoids a dense vocabulary output and preserves tie/score behavior. CTC is at most 0.0750 ms summed across a page. Attention MatMul, softmax, layer normalization and transpose together occupy only 2.9%/3.3% of the Small/Medium serial backbone probes below; they are lower priority than convolution in those shapes.

**Binding and result costs, milliseconds.** Binding/reshape/storage are nested work counters; owned copy is separate elapsed time after total. They are not additional sequential columns.

| Bundle / page | Binding | DET + REC reshape | Result storage | Owned copy |
| --- | --- | --- | --- | --- |
| Tiny / sparse | 0.0013 | 0.0005 | 0.0013 | 0.0013 |
| Tiny / dense | 0.0296 | 0.0565 | 0.0147 | 0.0053 |
| Tiny / mixed | 0.0127 | 0.0316 | 0.0072 | 0.0067 |
| Small / sparse | 0.0042 | 0.0008 | 0.0015 | 0.0015 |
| Small / dense | 0.0455 | 0.0616 | 0.0329 | 0.0079 |
| Small / mixed | 0.0465 | 0.0844 | 0.0108 | 0.0078 |
| Medium / sparse | 0.0055 | 0.0007 | 0.0020 | 0.0022 |
| Medium / dense | 0.0996 | 0.1188 | 0.0275 | 0.0055 |
| Medium / mixed | 0.0383 | 0.0644 | 0.0125 | 0.0038 |

The runtime keeps eight previous shape plans and high-water activation buffers. Warm DET/REC reshape work is at most 0.119 ms across owners. Classifier shape binding occurs during request construction. Increasing shape-cache capacity, optimizing CTC strings or changing result locks is not supported as a high-impact warm-latency priority by these measurements.

**Fused-operator diagnosis on the current source.** Detector probes use the sparse image at `[1,3,960,608]`, eight threads. Recognition probes use deterministic synthetic input and the production feature prefix at widths 128/640, one thread, excluding vocabulary projection. Each shape has one warm-up and one instrumented pass with bit-for-bit tensor equality. Fusion, layout and kernel selection remain production-like; these probes do not use the separate unfused trace path. Serial recognition does not reproduce concurrent-owner contention.

| Fused probe | Step sum ms | 1×1 conv ms | 1×1 share | Layout conversion ms |
| --- | --- | --- | --- | --- |
| Tiny DET sparse / 8 threads | 27.708 | 10.459 | 37.7% | 0.0000 |
| Small DET sparse / 8 threads | 54.130 | 22.066 | 40.8% | 2.9807 |
| Medium DET sparse / 8 threads | 174.381 | 71.525 | 41.0% | 0.0000 |
| Small REC width 128 / 1 thread | 4.217 | 2.868 | 68.0% | 0.0019 |
| Medium REC width 640 / 1 thread | 51.740 | 39.681 | 76.7% | 0.0108 |

The following costs guide the next experiments; all are single-pass diagnostic observations:

- **Tiny DET:** the two transposed convolutions take 1.837/1.592 ms; the two early dense 2×2 layers take 1.623/1.278 ms. Pointwise work totals 10.459 ms. The previously prominent 64→16 3×3 is now 1.210 ms and eligible for spatial SME.
- **Small DET:** the first transposed convolution takes 3.027 ms; initial 3→24 3×3 takes 2.764 ms; early 2×2 layers take 2.290/1.910 ms. The 96→24 3×3 costs 2.146 ms. Pointwise work totals 22.066 ms. Layout conversion is 2.981 ms, 5.5% of step time: worthwhile to inspect after the larger kernels, not the main explanation for total latency.
- **Medium DET:** leading individual steps are dense 32→32 7×7 at 10.268 ms, depthwise 256-channel 9×9 at 9.633 ms, first transposed convolution at 9.478 ms, and 32→64 2×2 at 8.299 ms. The other 2×2 is 6.945 ms. Dense 3×3 work totals 18.818 ms, while pointwise work totals 71.525 ms. The 256→64 3×3 is 7.745 ms and the 128→64 stride-two 3×3 is 4.948 ms. Thus the earlier emphasis on those two 3×3 layers must be revised after the retained optimization.
- **Small REC, width 128:** pointwise layers take 68.0% of the backbone. The two early 2×2 layers are 0.228/0.201 ms; the leading 96→192 pointwise is 0.201 ms. There is no single overwhelmingly expensive layer.
- **Medium REC, width 640:** pointwise layers take 76.7% of the backbone. Six 512→1024 convolutions over 960 spatial positions each take 2.109–2.389 ms. Their repeated packing/compute/store path is a stronger target than decoder or attention work.

The [shared SME executor](../src/runtime/nhwc/pointwise/sme.rs) already packs 64-row groups, uses 32×32 tiles with 16×16 tails, and applies sixteen-lane fused activation/residual stores. Each tile enters/exits streaming mode and stages results through a local array before the final store. Amortizing transitions or reducing staging is a concrete hypothesis, but these timings do not isolate the cost of either. Measure a full selected layer under realistic concurrency before adopting a change, retaining ZA/TPIDR2/FPCR guards, fallback behavior and accumulation order.

The ARM64 [packed transposed-convolution path](../src/runtime/nhwc.rs) performs four panel matrix products into reusable temporary storage, then scalar transformed stores into interleaved output positions. An existing direct-store specialization is Windows AVX2-only in [transpose.rs](../src/runtime/nhwc/transpose.rs). A NEON direct-store variant is a concrete unimplemented target, especially for the first Medium upsampling layer. It must include the matrix product, interleaving and fused epilogue in the comparison; measuring only a copy loop would be insufficient.

The [spatial SME selector](../src/runtime/nhwc/dense.rs) currently admits dense 3×3, at least 32 input/16 output channels and 32 output positions. It excludes 2×2, 5×5, 7×7 and depthwise layers. A bounded experiment on the Medium 32→32 7×7 or early 2×2 shapes could evaluate extending receptive-field packing. Larger kernels need more scratch and can lose to direct NEON at small channel counts. Depthwise 9×9 needs a separate spatial-reuse/strip-kernel investigation; treating it as dense matrix multiplication would waste work. No speedup from these candidates is claimed here.

**Recognition owner completion.** All finish offsets share one dispatch origin. Assigned width sum is a scheduling estimate, not measured computation. Dense owners use one kernel thread. On mixed pages, the single longest-crop owner can borrow a second thread within the same fixed CPU pool.

| Bundle / page | Assigned width sum | Owner finish range ms | Longest crop finishes ms | Last owner max width |
| --- | --- | --- | --- | --- |
| Tiny / dense | 1024–1120 | 18.63–26.71 | — | 192 |
| Tiny / mixed | 1088–1536 | 22.30–26.56 | 24.87 | 672 |
| Small / dense | 1024–1120 | 65.64–84.37 | — | 192 |
| Small / mixed | 1024–1472 | 62.51–86.76 | 81.33 | 320 |
| Medium / dense | 992–1088 | 165.11–286.45 | — | 224 |
| Medium / mixed | 1056–1472 | 184.43–299.50 | 221.39 | 672 |

Medium dense assigns width sums of 992–1088, only a 9.7% spread, but owner completion spans 165.11–286.45 ms. Medium mixed's width-1472 owner finishes at 221.39 ms; another owner with three lines, maximum width 672 and width sum 1088 finishes last at 299.50 ms. That last owner records 298.23 ms of calling-thread CPU time. In this current-source snapshot, making only the longest crop faster would not shorten the recognition critical path. Tiny and Small mixed likewise finish on other owners.

These records establish uneven completion but not its cause or recoverable time. Heterogeneous cores, per-shape kernel costs, cache/SME contention and OS scheduling remain plausible; thread CPU time includes active stalls and excludes nested helper threads. Do not subtract fastest from slowest and call that attainable savings. The next scheduling experiment should compare automatic eight owners with one bounded alternative, or calibrate a deterministic width/shape cost model, recording exact outputs and retained capacity. Unrestricted work stealing can destabilize private arenas and shape caches. More threads or larger batches are not assumed to be faster.

**Setup and memory.** The first three columns span separate diagnostic processes; the last is combined setup measured independently by the ordinary worker. They are not additive.

| Bundle | Model import ms | CPU context ms | Compile + request ms | Ordinary load + compile ms |
| --- | --- | --- | --- | --- |
| Tiny | 3.690–4.806 | 0.083–0.168 | 1.968–2.196 | 6.049–14.689 |
| Small | 10.280–20.611 | 0.078–0.152 | 8.928–17.008 | 20.065–21.055 |
| Medium | 38.501–61.694 | 0.083–0.115 | 46.558–58.312 | 89.118–97.174 |

Rebuilding a Medium model/request for every image adds about 89–97 ms in the ordinary observations before the first inference. Application-level reuse is the clearest improvement when callers currently reconstruct these objects. Cold filesystem I/O and external image decoding were not isolated. The observed setup ranges also demonstrate that timings from separate processes should not be treated as a precise decomposition of one another.

| Bundle / page | Activation MiB | Workspace MiB | Kernel scratch MiB¹ | Peak RSS MiB |
| --- | --- | --- | --- | --- |
| Tiny / sparse | 39.06 | 6.84 | 1.59 | 70.59 |
| Tiny / dense | 12.42 | 10.84 | 1.75 | 46.61 |
| Tiny / mixed | 41.79 | 12.48 | 1.83 | 78.42 |
| Small / sparse | 64.29 | 9.30 | 3.38 | 148.97 |
| Small / dense | 26.85 | 20.41 | 3.38 | 119.75 |
| Small / mixed | 84.79 | 22.17 | 3.59 | 180.66 |
| Medium / sparse | 156.36 | 13.43 | 7.50 | 455.83 |
| Medium / dense | 46.95 | 24.93 | 7.50 | 359.06 |
| Medium / mixed | 175.57 | 26.81 | 7.88 | 481.52 |

¹ Kernel scratch is included in workspace. Retained activation/workspace capacities and peak process RSS are not additive; RSS also contains models, allocator state and other resident pages. Medium mixed retains 175.57 MiB of activations and peaks at 481.52 MiB RSS. Changing owner count, batches or receptive-field packing can increase private memory even when latency improves, so evaluate those capacities together with outputs.

**Priority and quantitative sensitivity.** The figures below assume only the named elapsed phase gets shorter and all other work stays fixed. They are potential-effect calculations, not benchmarked candidate gains; summed worker durations are never substituted for wall time.

| Priority / workload | Next bounded experiment | Reason and sensitivity |
| --- | --- | --- |
| 1: Small/Medium dense or mixed | One representative pointwise-layer change, including packing and final stores; then the affected page. | A 20% shorter entire REC phase saves 16.89/17.36 ms on Small dense/mixed, or 14.6%/11.8% total. Medium saves 57.29/59.91 ms, or 16.1%/13.3%. Accelerating only one kernel saves less. |
| 1 alongside kernels: Medium recognition | One owner-budget or deterministic cost-model comparison with fixed memory/output checks. | Large owner finish spread despite balanced width sums; no attainable gain can yet be quantified. The longest crop is not the last finisher in the measured mixed case. |
| 1: sparse images | ARM64 direct-store transposed convolution; one remaining 7×7/2×2 shape or pointwise layer. | Detector is 69.9–74.4% of page time. A 20% shorter whole DET network would reduce sparse page latency by 14.0–14.9%. |
| 1: Tiny dense | Fixed-shape CLS backbone kernel or bounded batching experiment. | CLS is 18.88 ms/34.0%. Halving its wall phase would reduce the page by 9.44 ms/17.0%. |
| 2 | Compact recognition projection, Small detector layout transitions, and Tiny DET resize. | Measurable secondary costs. Preserve compact maxima, numerical behavior, layouts and fixed-point resize results. |
| Low | DB geometry, CTC, warm shape-cache lookups, text assembly/copies. | Each is small in these profiles; even eliminating them cannot deliver the gains available in network stages. |

Reducing detector resolution, recognition width or model size can also reduce work, but changes recognition/detection quality and requires a separate accuracy study. The measured outputs and thresholds were kept unchanged here. The existing spatial-SME adoption experiment already records sparse-page median reductions of 3.3%/7.9%/12.4% for Tiny/Small/Medium with disjoint sample ranges; those gains are included in the current numbers, not new gains from this analysis.

**Local validation and execution budget.** Before new timing, 12 focused SME/selection/accounting correctness tests and seven profiling-contract tests passed in release mode. One ignored crossover benchmark was not run. All six new production profiles equal their own warm-up outputs; all nine profile output hashes match the earlier complete-output controls. All five new fused-operator shapes match their warm-up tensors bit-for-bit. The exact-source prior evidence additionally records the full default/serial/profiling suites, clippy, x86 compilation and 213/213 local C# qualification requests; those broader checks were not rerun for this documentation-only analysis.

New performance execution was **6.508 seconds**, including subprocess startup, asset checking/loading, warm-ups and measurements. Adding the prior conservative version charge of 15.350 seconds gives **21.858 seconds**, below the 60-second version budget. The six new page-stage cases, three detector shapes and two recognition shapes each used one warm-up and one measured diagnostic pass. All timing subprocesses ran sequentially after builds/correctness checks, with a 12-second per-command timeout and the remaining aggregate budget. No unchanged case was rerun for more favorable numbers.

These results apply to the physical M4, the three fixture images and the stated eight-worker single-request configuration. They do not establish other-hardware performance, arbitrary image sizes, concurrent-request throughput, tail latency or ground-truth OCR accuracy. Internal SME packing/compute/staging, per-operator CLS costs, and individual DB substage times remain unisolated; the proposed experiments identify where additional instrumentation would be useful.

**Reproduction.** Local artifacts are in `target/pipeline-analysis-20261018-spatial-sme`; the portable JSON embeds `validate.py`, `run.py` and `package.py`, commands and raw logs. Validation/builds used:

```sh
cargo test --release --locked --lib sme -- --test-threads=1
cargo test --release --locked --features profiling --lib profile::tests -- --test-threads=1
cargo build --release --locked --example cpu_benchmark
cargo build --release --locked --features profiling --example cpu_profile
```

For a single missing stage case, `target/release/examples/cpu_profile medium imgs_00018069 8` performs one warm-up and one profile. The ignored `runtime::optimized::tests::performance::sparse_detector_production_steps` and `recognition_backbone_physical_steps` probes are invoked directly from the already-built release test binary. The evidence preserves all ordinary `_worker JOB.json` inputs and samples. The runner prevents accidental repeated measurement, verifies matching source/executable/asset hashes, and enforces the fixed budget. Do not run an additional timing matrix just to refine unchanged results.
