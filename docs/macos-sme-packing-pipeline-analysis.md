# Apple M4 pipeline latency after SME packing

The current implementation at `ca0996a` is still dominated by neural-network execution. Detector inference accounts for **72.4–80.0% of sparse-page wall time**. Recognition accounts for **54.9%, 73.3% and 80.2% of dense-page wall time** for Tiny, Small and Medium. Tiny's dense-page orientation phase is also substantial at **17.52 ms / 29.2%**. The next useful work is workload-specific convolution optimization, recognition scheduling diagnosis, and classifier acceleration. DB geometry, CTC decoding and result copying are small on these inputs.

[Raw evidence and reproduction scripts](reports/macos-arm64-2026-10-17-head-ca0996a-pipeline-analysis.json) preserve all nine current page measurements, nine stage profiles, five fused-operator probes, source/binary hashes, owner records and local validation logs. This analysis changes documentation only. The [preceding optimization report](macos-arm64-sme-packing-optimization.md) records the already implemented SME crossover and packing improvements; those improvements are not proposed again here.

**Measurement scope and provenance.** All execution is on this physical Apple M4: 10 cores, 16 GiB RAM, macOS 27.0.1, native ARM64 release builds, pinned `nightly-2026-09-08`, AC power. The system reports no recorded thermal/performance warning; temperature, frequency and physical core placement were not sampled. The configuration is eight shared CPU workers, one active request, automatic SIMD/layout/fusion and line ownership, recognition batch size one, adaptive widths, detector limit 960, classification enabled, classifier threshold 0, and alignment disabled. The manifest's detector thresholds and all models/pixels are unchanged. The manifest's `cpu_threads: 1` is overridden by each job's effective eight-thread setting.

The nine ordinary page cases and three stage profiles were already measured for this exact implementation. All current `src/**/*.rs` hashes match its final verification; freshly built benchmark/profile binaries are byte-identical to the measured executables. The harness, Cargo files, toolchain and examples are unchanged from the recorded baseline; model/fixture hashes are verified. Reusing these records avoids rerunning unchanged code for cleaner numbers. Six missing stage cases and five fused-operator shapes were newly profiled. Tables identify separate measurement types: **ordinary page medians, instrumented wall phases, and overlapping summed worker work**. They must not be mixed into a single additive breakdown.

**Ordinary page latency, milliseconds.** Each case uses a fresh process, separately records its first call, performs one explicit warm-up, then measures three owned-result calls. Timers exclude model setup, image decoding, serialization and service admission waiting. The first call includes lazy request initialization and first-use work, but is not a cold-disk measurement. These are current-version samples from the retained comparison, not new samples or a statistical tail estimate.

| Bundle / page | Lines | Warm median ms | Measured range ms | First call ms |
| --- | ---: | ---: | ---: | ---: |
| Tiny / sparse | 3 | 42.52 | 41.99–44.31 | 49.58 |
| Tiny / dense | 69 | 60.15 | 60.14–60.70 | 61.34 |
| Tiny / mixed | 31 | 68.24 | 67.67–71.02 | 72.26 |
| Small / sparse | 2 | 80.97 | 79.19–81.37 | 88.87 |
| Small / dense | 69 | 137.06 | 134.28–140.79 | 138.85 |
| Small / mixed | 33 | 165.72 | 163.98–178.98 | 180.11 |
| Medium / sparse | 2 | 259.08 | 254.07–264.80 | 267.64 |
| Medium / dense | 69 | 403.69 | 384.81–434.50 | 415.83 |
| Medium / mixed | 31 | 550.41 | 543.17–551.94 | 550.18 |

The sparse image is `imgs_1` (720×1150 source, 608×960 DET tensor, 2–3 detected lines). Dense `imgs_00018069` has a 416×256 DET tensor and 69 lines. Mixed `imgs_00006737` has an 896×544 DET tensor and 31–33 lines. Recognition always uses height 48: dense pages concentrate at widths 96/128, while mixed pages contain a width-1536 Tiny crop or width-1472 Small/Medium crop. Thus “sparse” describes text count, not detector work. Detector cost follows image size and model; recognition cost follows crop shapes, counts and scheduling. The bundles also produce different boxes, so model comparisons are whole-workload comparisons, not equal-shape kernel comparisons or accuracy rankings.

**Production stage latency, elapsed milliseconds.** Each profile contains one unprofiled warm-up and one instrumented call, with complete owned output equality. Tiny dense, Small dense and Medium mixed profiles are reused; the other six are new. The six measured phases run sequentially in this configuration.

| Bundle / page | DET prep | DET network | DB boxes | Crop | CLS + rotation | REC + assembly | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Tiny / sparse | 1.89 | 30.23 | 0.42 | 0.41 | 1.29 | 7.14 | 41.39 |
| Tiny / dense | 0.29 | 8.26 | 0.32 | 0.68 | 17.52 | 33.02 | 60.10 |
| Tiny / mixed | 1.13 | 24.17 | 0.56 | 1.07 | 8.05 | 31.96 | 66.94 |
| Small / sparse | 1.55 | 56.46 | 0.45 | 0.44 | 1.34 | 17.72 | 77.96 |
| Small / dense | 0.38 | 15.34 | 0.32 | 0.69 | 19.37 | 99.28 | 135.40 |
| Small / mixed | 1.11 | 48.41 | 0.55 | 1.13 | 8.97 | 110.57 | 170.75 |
| Medium / sparse | 2.00 | 216.77 | 0.61 | 0.52 | 1.46 | 49.73 | 271.11 |
| Medium / dense | 0.48 | 51.56 | 0.32 | 0.84 | 19.67 | 295.39 | 368.29 |
| Medium / mixed | 1.62 | 183.94 | 0.56 | 1.50 | 10.13 | 328.72 | 526.49 |

The unrounded phase sum leaves **0.012–0.029 ms** of validation, binding and orchestration inside total. REC includes its dispatch/join and ordered result assembly. Owned conversion is measured after profile total. Instrumentation and scheduling affect these runs: for example, Medium dense's profile is 368.29 ms versus its ordinary median of 403.69 ms. This difference is neither a speedup nor an estimate of timer overhead. Use the profiles to locate work and the ordinary samples to describe observed page latency.

The source path is:

```mermaid
flowchart LR
    A[Borrowed pixels] --> B[DET resize and normalize]
    B --> C[Detector network]
    C --> D[DB boxes and reading order]
    D --> E[Perspective crops]
    E --> F[Orientation classifier and rotation]
    F --> G[Recognition resize and backbone]
    G --> H[Vocabulary projection and CTC]
    H --> I[Ordered borrowed or owned results]
```

The [CLI](../examples/ocr.rs) additionally reads and decodes files and serializes results. The [engine](../src/engine.rs) can wait for an available request lease. Those costs, optional token alignment and application/network transport are outside this workload and remain unmeasured, rather than being assigned zero latency.

**Classification details, summed worker milliseconds.** These calls overlap across owners within the CLS wall phase and must not be added to that phase or treated as process CPU time.

| Bundle / page | Resize / normalize | Backbone | Head | 180° rotation |
| --- | ---: | ---: | ---: | ---: |
| Tiny / sparse | 0.2237 | 3.5790 | 0.0056 | 0.0177 |
| Tiny / dense | 5.2101 | 114.7760 | 0.1232 | 0.0000 |
| Tiny / mixed | 2.8175 | 49.6671 | 0.0542 | 0.0004 |
| Small / sparse | 0.1702 | 2.4583 | 0.0041 | 0.0194 |
| Small / dense | 5.3088 | 113.8166 | 0.1193 | 0.0000 |
| Small / mixed | 3.2070 | 52.8535 | 0.0597 | 0.0008 |
| Medium / sparse | 0.1848 | 2.6599 | 0.0034 | 0.0209 |
| Medium / dense | 5.4143 | 112.2864 | 0.1240 | 0.0000 |
| Medium / mixed | 2.8922 | 51.5285 | 0.0564 | 0.0055 |

[Classification](../src/orientation.rs) uses fixed 80×160 input, a network prefix and a compact two-class head. The backbone is approximately 95% of measured dense-page classification work. Pixel rotation is tiny; repeatedly evaluating the backbone is the cost. Up to eight single-thread classifier owners process round-robin line lists, with a barrier before recognition. Recognition batching does not batch the classifier.

No dense-page line is rotated, but sparse pages rotate all 3/2/2 lines and mixed pages rotate 1/2/3 for Tiny/Small/Medium. Disabling classification is an application-level accuracy tradeoff requiring known upright inputs; it would change results on this corpus. For Tiny dense, halving the CLS wall phase would save 8.76 ms, or 14.6% of its profiled page time, if everything else stayed fixed. That is a sensitivity calculation, not a measured candidate gain. A classifier backbone profile or real classifier batching is a useful next focused experiment. Stage overlap would also need evaluation of the shared CPU budget and retained memory.

**Recognition details, summed worker milliseconds.** These counters overlap between recognition owners; they are nested inside REC wall time.

| Bundle / page | Resize / normalize | Backbone | Vocabulary projection | CTC |
| --- | ---: | ---: | ---: | ---: |
| Tiny / sparse | 0.4366 | 12.5807 | 0.8896 | 0.0021 |
| Tiny / dense | 3.6062 | 193.5093 | 19.1516 | 0.0224 |
| Tiny / mixed | 4.6995 | 198.9218 | 19.4905 | 0.0170 |
| Small / sparse | 0.4600 | 31.4500 | 1.7668 | 0.0029 |
| Small / dense | 3.9904 | 660.3667 | 82.6627 | 0.0444 |
| Small / mixed | 4.3300 | 691.6341 | 77.5510 | 0.0290 |
| Medium / sparse | 0.4889 | 85.9382 | 4.0247 | 0.0032 |
| Medium / dense | 3.5917 | 1994.2876 | 119.5694 | 0.0546 |
| Medium / mixed | 4.6892 | 2206.1275 | 138.0415 | 0.0393 |

[Recognition](../src/recognizer.rs) resizes each crop to height 48, rounds adaptive width upward to a multiple of 32, runs the feature prefix and computes compact vocabulary maxima. It already avoids materializing the full vocabulary tensor and its final softmax. Dense Tiny/Small/Medium projection is **9.0% / 11.1% / 5.7%** of backbone-plus-projection work. This makes the compact head a secondary target, particularly for Small; a new full-vocabulary tensor would undermine existing memory savings. CTC takes at most **0.055 ms** of summed work. Attention's MatMul, softmax, layer normalization and transpose together account for only 2.1% of the Small backbone probe and 2.6% of Medium below.

**Bookkeeping, milliseconds.** Binding, reshape and result storage are nested work counters; owned copy is separate elapsed time after the profile. CLS's fixed-shape binding occurs during request construction.

| Bundle / page | Binding | DET + REC reshape | Result storage | Owned copy |
| --- | ---: | ---: | ---: | ---: |
| Tiny / sparse | 0.0012 | 0.0005 | 0.0015 | 0.0016 |
| Tiny / dense | 0.0155 | 0.0318 | 0.0146 | 0.0077 |
| Tiny / mixed | 0.0115 | 0.0385 | 0.0065 | 0.0089 |
| Small / sparse | 0.0032 | 0.0006 | 0.0015 | 0.0017 |
| Small / dense | 0.0234 | 0.0603 | 0.0310 | 0.0049 |
| Small / mixed | 0.0307 | 0.0445 | 0.0142 | 0.0096 |
| Medium / sparse | 0.0030 | 0.0005 | 0.0017 | 0.0025 |
| Medium / dense | 0.0570 | 0.0695 | 0.0318 | 0.0156 |
| Medium / mixed | 0.0270 | 0.0689 | 0.0127 | 0.0042 |

Warm reshape is at most 0.070 ms summed across DET and REC; result storage is at most 0.032 ms and owned copy at most 0.016 ms. The [runtime](../src/runtime.rs) retains eight previous shape plans and activation high-water buffers. More shape caching, a lock-free output rewrite or a new CTC decoder cannot address the dominant costs here.

**Code review of the remaining pipeline stages.**

| Stage | Existing implementation | Consequence for optimization |
| --- | --- | --- |
| Input and setup | [ImageView](../src/image.rs) borrows validated pixels; [builder](../src/builder.rs) shares immutable models; [engine leases](../src/engine.rs) retain bounded mutable requests. | Reuse models, compiled requests and the CPU context. Measure admission waiting separately for a loaded service. |
| DET preprocessing | [Preprocessing](../src/preprocess.rs) caches interpolation maps/rows, uses fixed-point byte interpolation and normalization lookup tables, writes the selected layout directly, and partitions rows. | Its entire phase is only 0.29–2.00 ms. Vertical interpolation/table/store work and transient job allocations are secondary candidates; individual costs were not isolated. |
| Neural execution | [Optimized execution](../src/runtime/optimized.rs) already fuses eligible operations/stores, aliases buffers and retains activation/scratch storage. | Target actual fused shapes; generic advice to add SIMD, fusion or buffer reuse duplicates existing work. |
| DB postprocessing | [DB](../src/db.rs) thresholds/scans, finds components/contours, constructs hulls/rectangles, scores/unclips and sorts boxes, using reusable scratch. | 0.32–0.61 ms on these pages. Geometry changes need another workload to justify them. Dilation is disabled in this measurement. |
| Perspective crops | [Extraction](../src/lines.rs) fills a retained BGR slab in 16-row jobs; [crop kernels](../src/crop.rs) use SIMD mapping and cubic interior sampling with clamped borders and folded geometric rotation. | 0.41–1.50 ms. Crop-to-tensor fusion would change the current cubic-byte then bilinear-resize path and needs numerical/accuracy validation. |
| Recognition shapes/batching | Exact selected-width batches are supported; batch size is one here. Dense inputs have many repeated widths. | Test one bounded same-width batch alternative if optimizing dense pages. More batching can reduce independent work and increase private activations. Fixed width 320 or lower resolution changes long-text sampling. |
| Recognition ownership | [OCR scheduling](../src/ocr.rs) sorts descending-width units and assigns them to the least-loaded stable owner, using width × batch count as cost. A dominant single-crop owner can borrow a second kernel thread. | This preserves deterministic output and warm storage. Owner finish records below still justify investigating the cost estimate and concurrent execution. |
| SME pointwise convolution | [Dispatch](../src/runtime/nhwc/pointwise.rs) now admits bands with at least 32 rows, both channel dimensions ≥96 and one ≥192 on supported macOS ARM64. [Packing](../src/runtime/nhwc/pointwise/sme.rs) already uses 4×4 NEON transposes, 64-row scratch, 32×32 tiles and 16×16 tails. | The 96↔192 and 128↔256 crossover expansion is already implemented. Investigate remaining tile transitions/staging and smaller ineligible kernels only with shape-specific evidence. |

**Fresh fused-operator diagnosis.** DET probes use the real sparse image at `[1,3,960,608]` with eight threads. REC probes use deterministic synthetic input and the production feature prefix at the stated width, one thread, excluding the vocabulary head. Each probe has one warm-up and one instrumented pass; all network outputs match bit-for-bit. They retain production fusion/layout/kernel selection; the public unfused trace path is not used. These serial REC probes do not reproduce eight concurrent recognition owners.

| Fused probe | Step sum ms | Pointwise ms | Pointwise share | Layout conversion ms |
| --- | ---: | ---: | ---: | ---: |
| Tiny DET sparse / 8 threads | 28.234 | 11.698 | 41.4% | 0.0000 |
| Small DET sparse / 8 threads | 56.449 | 24.039 | 42.6% | 2.1168 |
| Medium DET sparse / 8 threads | 212.779 | 78.842 | 37.1% | 0.0000 |
| Small REC width 128 / 1 thread | 6.366 | 4.488 | 70.5% | 0.0029 |
| Medium REC width 640 / 1 thread | 72.053 | 55.964 | 77.7% | 0.0088 |

Concrete hot operations are:

- **Tiny DET:** 64→16 3×3 takes 1.840 ms; two 32→64 pointwise layers take 1.810/1.726 ms. These pointwise shapes are below the SME channel gate. The first transposed convolution takes 1.615 ms. Prioritize narrow kernels and reuse across output pixels if sparse Tiny latency matters.
- **Small DET:** 96→24 3×3 takes 4.675 ms; two 48→96 pointwise layers take 2.874/2.810 ms; the first transposed convolution takes 2.820 ms. Layout conversions sum to 2.117 ms, 3.8% of this probe: measurable but secondary to convolution.
- **Medium DET:** 256→64 3×3 takes 27.997 ms, and 128→64 stride-two 3×3 takes 15.020 ms. Together these two operations consume **20.2%** of detector step time. A 32→32 dense 7×7 takes 9.719 ms, a 256-channel depthwise 9×9 takes 9.492 ms, and a transposed convolution takes 8.917 ms. All dense 3×3 work totals 53.053 ms. The pointwise gate has already been widened; dense spatial convolution and large-filter reuse deserve direct attention now.
- **Small REC width 128:** 96→48 stride-two 3×3 leads at 0.359 ms. The top 96→192 pointwise layers take 0.296/0.286 ms; they are now SME-eligible, so recommending the old gate expansion again is stale. Eligible 384→768 pointwise layers take 0.276/0.273 ms. Profile the real fused path before attributing residual costs to input packing.
- **Medium REC width 640:** six 512→1024 pointwise layers over 960 spatial rows take 3.216–3.369 ms each. The pointwise family remains **77.7%** of the backbone, even after SME changes. These are concrete targets for packing/compute/store decomposition and concurrency tests.

The SME source still enters/exits streaming mode per tile, checks thread state and stages 32×32 results in a local array before applying fused stores. Hoisting eligible checks, processing more work per streaming region or changing result staging are plausible candidates, **not measured causes**. Preserve live-ZA/TPIDR2/FPCR guards, vector-length checks, bias-first ascending-K accumulation, fused-store semantics and fallback behavior. The previous optimization already rejected a four-way reduction unroll because its fixed samples did not show consistent benefit; rerunning that unchanged idea is not a priority. No hardware-counter evidence currently establishes memory bandwidth, cache misses or SME contention as the bottleneck.

**Recognition completion imbalance.** All finish times use the same dispatch origin within their profile. Assigned width sum is the scheduler's cost proxy; “last owner max width” describes the slowest owner's largest crop. The mixed-page longest-crop owner has two kernel threads; dense owners have one each.

| Bundle / page | Assigned width sum | Owner finish range ms | Longest crop finishes ms | Last owner max width |
| --- | ---: | ---: | ---: | ---: |
| Tiny / dense | 1024–1120 | 21.35–33.00 | — | 256 |
| Tiny / mixed | 1088–1536 | 25.39–31.95 | 31.95 | 1536 |
| Small / dense | 1024–1120 | 84.33–99.19 | — | 192 |
| Small / mixed | 1024–1472 | 79.78–110.48 | 100.69 | 288 |
| Medium / dense | 992–1088 | 207.04–295.35 | — | 192 |
| Medium / mixed | 1056–1472 | 255.73–328.67 | 295.15 | 384 |

Medium dense's width sums differ by only 9.7%, yet owners finish between **207.04 and 295.35 ms**. Medium mixed's longest crop finishes at **295.15 ms**, while an owner whose largest crop is only 384 pixels finishes at **328.67 ms**. The latter's calling-thread CPU time is 322.85 ms, close to its 328.64 ms active wall interval. This establishes a finish-time imbalance; it does not identify whether shape costs, physical core placement, concurrent memory traffic or SME resources cause it. Thread CPU time excludes nested helpers and does not separate useful arithmetic from active stalls. The finish spread is not an achievable speedup estimate.

A focused scheduling experiment should record each owner's actual shape/kernel mix, then compare one deterministic cost-model or owner-count alternative. Keep output order and retained capacity checks. Unrestricted work stealing can make identical warmed pages populate different worker-private arenas and shape caches; merely increasing worker count is not justified by this data.

**Setup and memory.** The first three setup columns are ranges over the three separate diagnostic processes; the last is the ordinary benchmark's independently timed setup range. They are not additive across columns or a promise for uncached storage.

| Bundle | Model import ms | CPU context ms | Request compile ms | Standard load + compile ms |
| --- | ---: | ---: | ---: | ---: |
| Tiny | 3.52–3.69 | 0.08–0.16 | 1.94–1.96 | 6.36–12.26 |
| Small | 10.22–11.43 | 0.08–0.10 | 8.81–9.13 | 20.72–21.36 |
| Medium | 39.38–40.54 | 0.07–0.08 | 46.81–49.61 | 91.25–95.83 |

Rebuilding the model/request per page can add tens of milliseconds for Medium before inference. Reuse should be the first application-level check. Shape diversity and service concurrency need separate measurements from this warm, repeated-image workload.

| Bundle / mixed | Activations MiB | Workspace MiB | Cached weights MiB | Peak RSS MiB |
| --- | ---: | ---: | ---: | ---: |
| Tiny | 41.79 | 11.98 | 6.79 | 78.14 |
| Small | 84.79 | 21.77 | 30.41 | 180.44 |
| Medium | 175.57 | 25.81 | 132.20 | 481.02 |

Medium mixed retains 175.57 MiB of activations and 25.81 MiB of workspace, plus 132.20 MiB of shared cached weights; peak process RSS is 481.02 MiB. RSS also includes original model data, setup, allocator effects and resident shared pages, so it is not the sum of these counters. Extra requests and workers multiply private state. The latest SME change already added 1.08–1.31 MiB of Tiny workspace while improving latency; further scheduling/batching work should report retained memory as well as speed.

**Priorities and expected scale.**

1. For **Small/Medium dense or mixed pages**, target measured pointwise shapes and investigate the owner imbalance. Separate packing, streaming transitions, arithmetic and fused stores before selecting another SME change. Evaluate at most one bounded batching or scheduling alternative at a time.
2. For **sparse pages**, prioritize detector spatial/narrow convolutions. Medium's 256→64 3×3 and 128→64 stride-two 3×3 are particularly useful starting points; Tiny/Small need smaller-channel kernels than the existing SME gate.
3. For **Tiny dense pages**, profile/accelerate the classifier backbone or implement actual classifier batching. The phase is almost a third of page latency. Skipping CLS is appropriate only under an upright-input contract with accuracy validation.
4. Consider **compact vocabulary projection** next, especially Small. Detector preprocessing and layout conversion are secondary. Defer DB, CTC and output-copy rewrites on this corpus.

The table below holds other phases fixed and applies a hypothetical time reduction to an entire phase. These are sensitivity estimates, not measured speedups or forecasts for a particular kernel change.

| Hypothetical change | Tiny saved ms / page % | Small saved ms / page % | Medium saved ms / page % |
| --- | ---: | ---: | ---: |
| 20% less DET network time / sparse | 6.05 / 14.6% | 11.29 / 14.5% | 43.35 / 16.0% |
| 20% less REC phase time / dense | 6.60 / 11.0% | 19.86 / 14.7% | 59.08 / 16.0% |

A 20% reduction of Medium mixed REC would save 65.74 ms / 12.5% of its profiled page time; a 20% DET reduction would save 36.79 ms / 7.0%. The arithmetic shows why network work and scheduling have more headroom than sub-millisecond bookkeeping. Lower detector resolution, a smaller model, fixed recognition width and disabled classification are separate accuracy/latency decisions, not transparent implementation optimizations.

**Local validation and execution budget.** All 17 focused correctness checks passed: three default profile-accounting tests, seven SME tests and seven profiling-enabled accounting/output tests. Six new production profiles preserve complete owned output relative to their warm-up; five fused-network probes preserve output bits. The matching version's prior full 213-request reference qualification remains recorded in the optimization report and was not rerun for this documentation-only analysis. That qualification establishes reference compatibility on its corpus, not ground-truth OCR accuracy.

New performance execution totals **6.927 seconds**, including subprocess startup/loading and every warm-up/measured call. Combined with the version's previously recorded **14.066 seconds**, cumulative execution is **20.993 seconds**, below 60 seconds. New work is six OCR warm-ups plus six profiles, and five network warm-ups plus five operator profiles. Reused evidence has nine page cases × three measured samples, separate first calls and one explicit warm-up each, plus three stage profiles. The version's earlier kernel candidate-selection calls remain charged to its budget and are not pooled into these tables. Builds and correctness checks are excluded. Each new subprocess had a 12-second timeout and the remaining aggregate budget; no retries or sampling-growth loops were used.

Reproduction files are in `target/pipeline-analysis-20261017-head-ca0996a`; the portable JSON embeds logs and scripts. The driver refuses to overwrite its measurement marker or accept changed sources/binaries. Its three prior input files (`comparison.json`, `profiles.json`, `verification.json`) are also preserved as corresponding top-level objects in the linked packing-optimization evidence, allowing reconstruction if local target artifacts are removed. To repeat on a changed implementation, rebuild into a fresh evidence directory and start a new version's run count. Focused correctness and construction commands were:

```sh
cargo test --release --locked --lib profile::tests -- --test-threads=1
cargo test --release --locked --lib sme -- --test-threads=1
cargo test --release --locked --features profiling --lib profile::tests -- --test-threads=1
cargo build --release --locked --example cpu_benchmark
cargo build --release --locked --features profiling --example cpu_profile
```

The conclusion applies to this physical M4, three representative pages and one effective thread/concurrency configuration. Native x86 performance, other SME hardware, service throughput, p95/p99, ground-truth accuracy and the proposed optimizations' actual speedups remain unmeasured.
