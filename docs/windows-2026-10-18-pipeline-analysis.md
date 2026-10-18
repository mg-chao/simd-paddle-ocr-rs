# Current Windows OCR pipeline analysis — 2026-10-18

This records the pre-optimization `b30436a` implementation. The subsequent
[parallel PadEnd2 optimization](windows-avx2-pad-end-optimization.md) adopts a
measured detector improvement and records its separate before/after results.

There is substantial work worth targeting, concentrated in neural inference. On revision `b30436a`, **DET inference occupies 76.2% of sparse Tiny elapsed time; recognition occupies 67.4% of dense Small and 86.5% of dense Medium**. The current serial recognition probes attribute **80.6% / 90.5%** of Small / Medium physical-step time to 1×1 convolutions. Dense Tiny also spends **12.414 ms / 27.5%** in orientation classification.

Fresh owner records refine the scheduling diagnosis: the mixed page's width-1536 line finishes first, and an owner with five shorter lines finishes last. A scheduler that accounts for per-line overhead is a more relevant hypothesis here than automatically giving the longest line more resources. No optimization speedup is claimed: this task measures and analyzes the current implementation, without changing production code or defaults.

The [portable evidence JSON](reports/windows-x64-2026-10-18-pipeline-analysis.json) contains every ordinary sample, stage counter, owner interval, operator log, output digest, source/executable/model hash, validation/build log, derived calculation and runner source. Earlier [CPU24 stage timings](windows-cpu24-stage-analysis.md), [depthwise before/after results](windows-avx2-depthwise-optimization.md), and macOS results describe separate experiments; none are pooled into this report.

**Machine, configuration and timing boundaries.** All execution was local on the physical Windows x64 ASUS desktop: AMD Ryzen 9 5950X, **12 enabled physical cores / 24 logical processors** reported by Windows, 63.91 GiB RAM, Balanced power plan. Page runs used a shared 24-thread CPU context, affinity `ffffff`, one request, automatic AVX2/FMA/layout/fusion, automatic line ownership, batch size one, detector limit 960, adaptive recognition height 48 and widths rounded to 32, classification enabled at threshold 0.0, and alignment disabled. Detector/preprocessing budgets cap at sixteen; dense recognition uses sixteen owners with one intra-op thread each. Logical CPUs are not separate physical cores.

Builds use pinned `nightly-2026-09-08`, release optimization and default parallel features, without custom RUSTFLAGS. Ordinary and profiling binaries were built separately and frozen before timing. Three fixtures cover sparse `imgs_1` (720×1150, 3 lines), dense `imgs_00018069` (430×267, 69 lines), and mixed `imgs_00006737` (896×528, 31 lines). Small and Medium reuse the dense fixture to determine how model size changes stage composition. Their model-specific thresholds and dictionaries remain pinned; these are workload costs, not an accuracy ranking.

The measured OCR API starts from predecoded BGR24 pixels. Ordinary calls include complete owned-result conversion. Profiles stop at the borrowed result and time conversion separately. Capture, file I/O / JPEG-PNG decode, JSON serialization and external engine-lease queueing are outside the samples. Setup is measured separately below. These excluded application costs are not assumed to be zero.

**Complete OCR latency.** Each independent ordinary process executes one first-call warm-up and three measured calls. There is no extra first inference. Every result equals its complete warm-up control, and complete serialized output digests match the corresponding profiling build.

| Case | Three owned-result calls, ms | Median, ms | Observed range, ms |
| --- | --- | --- | --- |
| Tiny sparse | 51.219, 55.586, 50.281 | 51.219 | 50.281–55.586 |
| Tiny dense | 44.664, 44.664, 44.570 | 44.664 | 44.570–44.664 |
| Tiny mixed | 69.988, 69.353, 68.541 | 69.353 | 68.541–69.988 |
| Small dense | 117.216, 117.300, 125.958 | 117.300 | 117.216–125.958 |
| Medium dense | 440.340, 454.722, 434.418 | 440.340 | 434.418–454.722 |

The sparse, Small and Medium ranges show host variation; three calls do not establish steady state, p95/p99 or confidence intervals. No unchanged case was rerun for cleaner numbers. These medians cannot establish improvement or regression against a report from another session or source revision.

**Elapsed pipeline stages.** The following values are one instrumented production call per case after one warm-up. DET, DB, crop, classification and recognition execute sequentially. Their elapsed durations plus the remainder equal that profile's total.

| Stage / milliseconds | Tiny sparse | Tiny dense | Tiny mixed | Small dense | Medium dense |
| --- | --- | --- | --- | --- | --- |
| DET resize / normalization | 0.921 | 0.330 | 0.901 | 0.441 | 0.345 |
| DET inference | 36.013 | 12.280 | 29.449 | 25.806 | 48.830 |
| DB boxes / reading order | 0.635 | 0.464 | 0.799 | 0.525 | 0.470 |
| Crop planning / extraction | 0.631 | 0.876 | 2.122 | 0.958 | 0.747 |
| Classification / orientation wave | 1.628 | 12.414 | 5.697 | 13.104 | 11.559 |
| Recognition wave / assembly | 7.384 | 18.797 | 29.241 | 84.540 | 398.495 |
| Remaining orchestration / timer boundaries | 0.038 | 0.045 | 0.035 | 0.055 | 0.050 |
| Profiled borrowed-result total | 47.250 | 45.206 | 68.245 | 125.429 | 460.497 |
| Separate owned conversion | 0.004 | 0.013 | 0.009 | 0.014 | 0.014 |

| Share of profile total | Tiny sparse | Tiny dense | Tiny mixed | Small dense | Medium dense |
| --- | --- | --- | --- | --- | --- |
| DET inference | 76.2% | 27.2% | 43.2% | 20.6% | 10.6% |
| Classification | 3.4% | 27.5% | 8.3% | 10.4% | 2.5% |
| Recognition | 15.6% | 41.6% | 42.8% | 67.4% | 86.5% |

The total includes CPU-pool entry and orchestration; the remainder includes timing boundaries, validation and some binding/storage work. The recognition wave includes its scheduling and ordered output assembly. Classification finishes before recognition starts. Crop includes geometric rotation of vertical crops; classifier-triggered 180° correction belongs to classification.

The stage profile and ordinary median are separate executions. For example, Medium's profile is 460.497 ms and its ordinary median is 440.340 ms. Instrumentation, separate binaries and host variation can affect them. Do not rescale the stage values to the ordinary median, combine them into one distribution, or subtract those totals to estimate timer/copy overhead.

**Work inside parallel stages.** These are sums of overlapping owner-call durations, including nested kernel waits. They are not process CPU time or extra serial latency. Do not add this table to the wall table or divide every entry by a fixed worker count.

| Stage / milliseconds | Tiny sparse | Tiny dense | Tiny mixed | Small dense | Medium dense |
| --- | --- | --- | --- | --- | --- |
| Crop workers | 2.485 | 7.338 | 10.064 | 9.157 | 7.215 |
| CLS inclusive workers | 4.786 | 148.796 | 73.326 | 170.046 | 152.671 |
| CLS preprocessing (nested) | 0.311 | 8.692 | 4.870 | 9.271 | 8.603 |
| CLS backbone (nested) | 4.401 | 139.900 | 68.356 | 160.542 | 143.851 |
| CLS compact head (nested) | 0.011 | 0.161 | 0.075 | 0.177 | 0.166 |
| Applied 180° rotations (nested) | 0.061 | 0.000 | 0.002 | 0.000 | 0.000 |
| REC preprocessing | 0.609 | 5.399 | 5.621 | 5.409 | 5.593 |
| REC backbone | 16.269 | 230.054 | 197.928 | 1126.381 | 5666.148 |
| Vocabulary projection / argmax | 0.783 | 19.757 | 9.901 | 84.558 | 230.648 |
| CTC text decoding | 0.017 | 0.077 | 0.056 | 0.105 | 0.249 |
| DET + REC reshape calls | 0.002 | 0.079 | 0.068 | 0.146 | 0.218 |
| Binding / preparation | 0.003 | 0.034 | 0.017 | 0.045 | 0.073 |
| Result storage | 0.012 | 0.034 | 0.020 | 0.044 | 0.078 |

CLS nested counters belong to its inclusive worker work. The classifier backbone accounts for **91.9–94.4%** of that work; rotation and the two-class head are small. All three sparse lines receive a 180° correction, one mixed line does, and dense pages receive none. The dense classifier still evaluates all 69 lines.

The REC backbone accounts for **90.1% / 92.6% / 96.0%** of measured preprocessing + backbone + projection + CTC work on dense Tiny / Small / Medium. Projection accounts for approximately **7.7% / 7.0% / 3.9%** of that work. These are worker-work compositions, not corresponding percentages of page latency. Warm binding/reshape, CTC and result storage have small measured costs. Owned conversion is only 0.0045–0.0144 ms per page in the directly timed samples.

**Recognition scheduling and completion.** Owners are assigned deterministically by descending width and accumulated `width × batch_count` cost. This preserves worker-local arenas and shape caches. The current Windows policy already expands to sixteen owners for at least 32 units on 24–31-thread contexts; the mixed page has 31 units and uses eight owners. The extra-thread borrowing rule in `recognition_borrowing_owner` is NEON-only and does not apply here.

| Case | Owners × intra-op threads | Width-cost range | First / median / last finish, ms | Last-minus-penultimate finish, ms |
| --- | --- | --- | --- | --- |
| Tiny sparse | 3 × 8 | 160–608 | 4.896 / 5.479 / 7.375 | 1.896 |
| Tiny dense | 16 × 1 | 512–576 | 12.661 / 16.412 / 18.707 | 0.005 |
| Tiny mixed | 8 × 3 | 1088–1536 | 22.519 / 27.306 / 29.218 | 1.072 |
| Small dense | 16 × 1 | 512–576 | 68.664 / 77.337 / 84.456 | 3.304 |
| Medium dense | 16 × 1 | 480–576 | 327.966 / 373.306 / 398.426 | 2.806 |

Offsets share a dispatch origin and exclude planning before that origin. Every owner starts within 0.139 ms of dispatch in these samples. The final-owner-only tails are small relative to the neural work, especially the 2.806 ms tail within Medium's 398.495 ms recognition wave. This is not evidence that perfect balancing would save only that tail: several owners may finish early. It does show that one isolated late owner is not the whole Medium problem.

| Owner | Lines | Width cost | Maximum width | Finish after dispatch, ms |
| --- | --- | --- | --- | --- |
| 0 | 1 | 1536 | 1536 | 22.519 |
| 1 | 4 | 1216 | 672 | 28.146 |
| 2 | 5 | 1216 | 448 | 29.218 |
| 3 | 5 | 1152 | 416 | 27.467 |
| 4 | 4 | 1088 | 384 | 27.375 |
| 5 | 4 | 1088 | 352 | 26.301 |
| 6 | 4 | 1088 | 320 | 25.718 |
| 7 | 4 | 1088 | 320 | 27.236 |

For mixed Tiny, the largest width-cost owner completes at 22.519 ms, while owner 2, with smaller width cost but five calls, completes at 29.218 ms. This is consistent with a per-call fixed cost and width-dependent kernel efficiency, but does not prove either cause; nested scheduling, SMT/cache contention and host noise can contribute. A deterministic model-specific estimate such as `a × sum(tensor_width) + b × line_count`, calibrated with isolated real shapes, is a reasonable next scheduling experiment. Preserve stable ownership, exact ordering and retained-memory bounds. Do not hard-code coefficients from this single sample or remove adaptive widths: compressing long text to width 320 changes the recognition input.

**Physical operator attribution.** The existing opt-in diagnostics run the fused production plan, with one warm-up and one instrumented pass and bitwise output agreement. Ordinary observer tracing executes a different diagnostic path and was not used for timing. These probes deliberately isolate different questions and use different thread counts from CPU24 pages; their totals are not additional components of the page-stage table.

The detector probe uses the real sparse image at `[1,3,960,608]`, eight logical CPUs with affinity `ff`, for all three bundles. It excludes image preprocessing and some session overhead. The columns do not sum to all steps because pooling, concatenation and other operators are omitted from the compact table; the evidence contains all families and steps.

| Sparse DET / eight logical CPUs | All steps, ms | 1×1, ms | Other group-one convolutions, ms | Grouped / depthwise, ms | ConvTranspose, ms | Layout Convert, ms |
| --- | --- | --- | --- | --- | --- | --- |
| Tiny | 32.521 | 10.687 | 9.460 | 2.658 | 2.850 | 0.790 |
| Small | 83.193 | 29.761 | 29.890 | 5.913 | 6.693 | 2.067 |
| Medium | 305.767 | 153.072 | 88.066 | 28.090 | 9.021 | 0.000 |

Tiny's largest individual sampled steps are node 2 (`16→8`, 2×2) at 2.384 ms, node 4 (`8→16`, 2×2) at 2.259 ms, node 19 (`32→64`, 1×1) at 2.070 ms and node 239 (final transposed convolution) at 1.865 ms. Other group-one spatial convolutions total 9.460 ms, alongside 10.687 ms of pointwise work. Short-depthwise interiors and direct AVX2 upsampling are already implemented. Remaining sparse-page work is distributed across operators; optimizing only depthwise or only pointwise cannot remove the whole detector cost.

The recognition probe uses deterministic synthetic tensors at one logical CPU with affinity `1`, stopping before the vocabulary projection. Small width 128 appears in the dense page; Medium width 640 is a long-line shape, not the dense page's exact width distribution. These probes identify current kernel costs, not the precise operator percentages of sixteen-owner dense-page latency.

| REC feature prefix / one logical CPU | All steps, ms | 1×1, ms | 1×1 share | Layout Convert, ms |
| --- | --- | --- | --- | --- |
| Small [1,3,48,128] | 9.522 | 7.678 | 80.6% | 0.040 |
| Medium [1,3,48,640] | 222.644 | 201.558 | 90.5% | 0.014 |

Small's largest sampled step is node 201 (`384→768`, 1×1) at 0.568 ms, followed by another `384→768` and `768→384` projections. Medium's node 133 (`512→1024`, 1×1) is 11.244 ms; several other large-channel pointwise layers cost about 9–11 ms each. Layout conversion is small in these REC probes. Hardware cache misses, bandwidth, IPC, register spills and clock behavior were not measured, so a specific microarchitectural cause remains unproven.

**Setup and reuse.** These single setup samples come from the ordinary executable. Import includes local model/dictionary reads, ONNX parsing and normalization. Compile/request includes graph planning, weight preparation and initial mutable requests; lazy line workers and shape-dependent work can still occur in first inference.

| Case | Import, ms | CPU context, ms | Compile / request, ms | Setup total, ms | First-call warm-up, ms |
| --- | --- | --- | --- | --- | --- |
| Tiny sparse | 10.891 | 0.583 | 6.452 | 17.925 | 59.219 |
| Tiny dense | 9.696 | 0.552 | 5.013 | 15.260 | 55.616 |
| Tiny mixed | 9.189 | 0.562 | 4.551 | 14.301 | 81.854 |
| Small dense | 31.384 | 0.446 | 22.315 | 54.144 | 135.883 |
| Medium dense | 123.622 | 0.614 | 133.544 | 257.781 | 448.517 |

Keep parsed/compiled models and mutable requests alive, or retain an `OcrEngine` lease pool appropriately. Recreating the complete pipeline per image adds roughly **14–258 ms** in these samples. The helper hashes assets before import, so cache state is uncontrolled and often warm; these are not cold-disk measurements. Import parsing, packing and compilation substeps were not individually timed. Parallel compilation would be a separate cold-start experiment, with memory costs.

| Case | Retained activations, MiB | Retained workspace, MiB | Combined request buffers, MiB | Process CPU / wall ratio |
| --- | --- | --- | --- | --- |
| Tiny sparse | 39.062 | 5.463 | 44.525 | 10.046 |
| Tiny dense | 17.330 | 14.289 | 31.619 | 12.253 |
| Tiny mixed | 43.744 | 12.272 | 56.016 | 13.229 |
| Small dense | 39.400 | 27.353 | 66.753 | 13.220 |
| Medium dense | 63.900 | 27.548 | 91.449 | 13.786 |

Retained request buffers exclude shared immutable weights; kernel scratch is already included in workspace. Process memory peaks in the evidence include helper asset checks and decoding all helper fixtures, not just a deployed request. Engine idle-byte limits below the retained size can evict warmed requests. CPU/wall ratios use summed measured process CPU time divided by summed measured elapsed time; Windows counters are coarse, and these ratios do not measure physical-core utilization or prove a bandwidth bottleneck. Per-owner thread CPU time is unavailable on Windows.

**Implementation review, from input to output.**

| Link | Existing mechanism | Assessment |
| --- | --- | --- |
| Acquisition / decode | [ImageView](../src/image.rs) borrows BGR/RGB/BGRA/RGBA with checked stride; [the CLI](../examples/ocr.rs) separately decodes files. | Pass supported caller pixels directly where possible. Decode/capture need application-level measurement; no latency is assigned here. |
| Model import / compile | [builder.rs](../src/builder.rs) loads ONNX/dictionaries, compiles DET/REC/optional CLS and constructs requests. | Reuse first. Whole-pipeline setup is material, particularly for Medium. |
| Engine admission / reuse | [engine.rs](../src/engine.rs) leases requests and caps idle retained memory; [cpu.rs](../src/cpu.rs) shares bounded CPU capacity. | This run has one request. Queueing, concurrent throughput and idle eviction need their own workloads before changing global thread defaults. |
| Shape binding | [runtime.rs](../src/runtime.rs) and [optimized.rs](../src/runtime/optimized.rs) retain eight prior logical/physical shape plans and high-water arenas. | Warm binding/reshape is tiny. Recommending shape caching or arena reuse would duplicate existing work. |
| Detector preprocessing | [preprocess.rs](../src/preprocess.rs) caches resize maps/rows, uses SIMD pixel operations and writes the selected layout directly. | 0.330–0.921 ms elapsed here. Smaller opportunity than inference. |
| Detector network | [optimized runtime](../src/runtime/optimized.rs), [NHWC kernels](../src/runtime/nhwc.rs) and [NCHW kernels](../src/runtime/nchw.rs) use packed weights, fusion, aliasing and bounded scratch. | Main sparse-page target. Investigate actual remaining 2×2, 3×3, pointwise and transpose shapes; select by measured layer cost. |
| DB boxes / reading order | [db.rs](../src/db.rs) thresholds/scans with SIMD, reuses component/geometry scratch, scores, unclips and sorts boxes. | 0.464–0.799 ms here. Its minimum-area rectangle loops over hull edges and vertices, so complex contours could justify rotating calipers, with geometry/tie/order validation. |
| Perspective crops | [lines.rs](../src/lines.rs) retains a contiguous pixel slab and dispatches 16-row bands; [crop.rs](../src/crop.rs) and [pixel kernels](../src/pixel_kernels.rs) perform bicubic sampling. | 0.631–2.122 ms. An equivalent affine/axis-aligned fast path or multi-pixel sampler is plausible; changing interpolation changes pixels. Temporary job vectors are still allocated, but allocation-specific cost was not measured. |
| Classification / orientation | [orientation.rs](../src/orientation.rs) executes a fixed `[1,3,80,160]` backbone and compact two-class head, then [lines.rs](../src/lines.rs) rotates selected crops. | Dense Tiny's 12.414 ms wave is significant, overwhelmingly backbone work. Head/rotation tuning has little ceiling. |
| Recognition shapes / scheduling | [recognizer.rs](../src/recognizer.rs) chooses adaptive widths; [ocr.rs](../src/ocr.rs) assigns deterministic width-based owners and supports exact-width batches. | Current owner traces support investigating per-line overhead in the cost model. Eight-to-sixteen-owner expansion already exists. |
| Recognition preprocessing | [preprocess.rs](../src/preprocess.rs) writes tensors directly; when padding is needed it clears the entire output before writing content. | Clearing only padding is a concrete smaller candidate. Multi-line measured preprocessing is 5.4–5.6 ms summed overlapping work, not that many milliseconds of page latency. |
| Recognition backbone | [pointwise.rs](../src/runtime/nhwc/pointwise.rs) uses six-row × sixteen-output AVX2 tiles, full ascending-K reductions, fused final stores and 96-row groups for large channels; [avx2.rs](../src/runtime/nhwc/pointwise/avx2.rs) specializes K=96–192. | Best dense Small/Medium target remains large-channel 1×1 work. Current ARM SME/NEON extensions are guarded and do not accelerate this Ryzen host. |
| Vocabulary projection | [recognizer.rs](../src/recognizer.rs) and [argmax.rs](../src/runtime/simd/argmax.rs) reuse packed panels and retain compact winners. | Full final vocabulary probabilities/softmax are already avoided in production recognition. Preserve tie order and public score semantics; prioritize the larger backbone first. |
| CTC / storage / owned output | [recognizer.rs](../src/recognizer.rs) reuses decode storage; [ocr.rs](../src/ocr.rs) assembles ordered result slabs from worker slots. | Very small measured costs. Alignment/token-box capture is disabled, so its incremental cost is unmeasured. |

**Recommended priorities and realistic limits.**

1. **Dense Small/Medium: investigate one dominant large-channel pointwise layer first.** Use the actual Small `384→768` or Medium `512→1024` geometry and obtain cache/compute/assembly evidence before choosing a new mechanism. The [earlier large-channel experiments](windows-large-channel-optimization.md) and [implementation analysis](windows-large-channel-convolution-analysis.md) already rejected K blocking/input packing, larger grouping, alternate tiles and broad store specialization as established broad wins. Require exact output parity and a bounded page-level gain before adoption; a faster isolated kernel alone is insufficient.
2. **Sparse Tiny: prioritize detector spatial and pointwise computation.** The profile's DET inference is 36.013 ms / 76.2%. Investigate the measured early 2×2 layers and remaining 3×3/pointwise shapes, with both canonical and NHWC paths represented. Existing AVX2 depthwise and transpose improvements are already present. Lower resolution or a smaller model are quality/performance choices, requiring representative detection and text-quality checks.
3. **Dense Tiny: consider whether orientation classification is required.** The current 12.414 ms wave is a meaningful cost. Skipping it is appropriate only under a verified input-orientation contract. The [historical toggle test](windows-no-cls-latency.md) preserved dense text but changed sparse and mixed outputs; its speedup is not a current CPU24 prediction. Current profiles apply rotations on those two inputs. If CLS remains required, target its fixed-shape backbone before its head or rotation.
4. **Mixed pages: test a stable width-plus-call-count scheduler cost.** The longest crop is already the earliest finisher in this profile. Simply prioritizing it further is unsupported. Validate any revised model across sparse/dense/mixed pages, exact outputs and retained capacities. Hardware/topology effects remain an alternative explanation. Automatic work stealing could grow different per-owner caches between calls.
5. **Defer small glue-stage rewrites and indiscriminate batching/thread increases.** The [existing batch-two experiment](windows-avx2-depthwise-optimization.md) found no consistent page benefit; batching also changes scheduling-unit counts. Model/context reuse is already useful if the application rebuilds per image. DB, CTC and output-copy work have limited ceilings on these fixtures.

For scale, **halving the entire** sparse Tiny detector inference would save 18.007 ms, or **38.1%** of that profile. Halving the entire Small/Medium recognition wave would save 42.270/199.248 ms, or **33.7%/43.3%**. These are conditional Amdahl calculations with all other elapsed time fixed, not predicted gains from a kernel change. Even eliminating all DB work saves less than 0.8 ms here. No quantified optimization speedup has been demonstrated by this analysis.

**Validation, budget and reproduction.** Seven focused release/profiling tests passed before timing, covering disabled counters, merge accounting, serial/parallel/batched owner records and complete-output parity. All model, source-image and decoded-pixel hash checks passed. Every ordinary/profile call matched its complete warm-up result, all five cross-build complete-result digests matched, and the five operator graphs/prefixes matched output bits. Production source and frozen executable hashes were stable throughout measurement. Both builds succeeded with the pre-existing Windows unused `UNBOUND_AFFINITY` import warning.

There were **15 ordinary measured calls + 5 profiled calls + 5 operator calls**, and **15 warm-ups: 40 inference calls total**. Conservative process execution, including startup, asset checks, decoding, setup and collection waits, totaled **11.447 seconds**, within the single aggregate **60-second** budget. Builds and correctness tests are excluded. Page/REC diagnostic processes had ten-second deadlines; the three-detector diagnostic had fifteen seconds. No sampling growth, benchmark retries, CI, VM or remote runner was used.

The executable runner and logs remain in `target/pipeline-analysis-20261018/`; the tracked evidence embeds the runner, analysis source and all measurements. To reproduce a separately justified run, use a fresh output directory (the runner refuses to overwrite measurements), rebuild/freeze the executables, and enforce the same aggregate budget:

```powershell
cargo test --release --locked --features profiling --lib profile::tests
cargo build --release --locked --example cpu_owner_probe
cargo build --release --locked --features profiling --example cpu_profile

# For each of the five selected cases, one invocation of each:
./target/release/examples/cpu_owner_probe.exe tiny imgs_1 0 1
./target/release/examples/cpu_profile.exe tiny imgs_1 24 0
# Operator diagnostics are the exact ignored tests named in the saved runner.
```

This scope establishes stage composition on three source images and one physical Windows machine. It does not qualify OCR accuracy against ground truth, other CPUs/architectures, concurrent throughput, service queueing, capture/decode, alignment overhead or tail-latency distributions. Per-operator probes are single samples at their stated independent thread configurations. Native CPU hardware counters and cold-import substeps remain unmeasured.
