# Apple M4 current OCR pipeline latency analysis

The subsequent [SME recognition optimization](macos-arm64-sme-optimization.md) implements the large-channel convolution recommendation and records the new page and stage timings. The measurements below remain the pre-SME baseline at `2e5287f`.

Convolution execution is the strongest remaining optimization target in the current implementation. **Detector inference accounts for 67.1–71.6% of sparse-page elapsed time. Recognition accounts for 56.4%, 81.5% and 89.1% of dense-page time for Tiny, Small and Medium.** Small and Medium recognition should prioritize large-channel 1×1 convolutions. Tiny dense pages also spend 29.0% in orientation classification. DB geometry, CTC decoding and result copying have much smaller absolute costs.

This analysis measures commit `2e5287fcf8d22e194c71cd60a7985ed7689a9464` on 17 October 2026. It includes the retained narrow NEON convolution paths and the long-crop two-worker scheduling rule. The [portable evidence](reports/macos-arm64-2026-10-17-head-2e5287f-pipeline-analysis.json) contains raw samples, complete outputs, profiles, owner records, operator timings, source/executable/model/pixel hashes, validation and the reproduction driver. The [earlier revision evidence](reports/macos-arm64-2026-10-17-current-pipeline-analysis.json) remains historical; its samples are not pooled with these measurements or used to claim a new speedup.

The physical host is an Apple M4, 10 cores, 16 GiB RAM, macOS 27.0.1, running on AC power. No thermal or performance warning was recorded by `pmset`; CPU frequency and physical-core placement were not measured. All builds are native release with pinned `nightly-2026-09-08`, default parallel features, automatic NEON/FMA/layout selection and fusion. Measurements use one request, eight shared workers, automatic line ownership, detector limit 960, adaptive recognition widths, batch one, classification enabled and alignment disabled. macOS scheduling is unpinned.

The library accepts decoded pixels, so its measured path is input binding → detector resize/normalization → detector network → DB boxes → perspective crops → orientation classifier/rotation → recognition resize/backbone/projection/CTC → ordered results. JPEG/PNG decoding, file acquisition and JSON serialization belong to the CLI/application and are outside this library benchmark.

Inputs are `imgs_1` (sparse, 720×1150), `imgs_00018069` (dense, 430×267), and `imgs_00006737` (mixed widths, 896×528). Detector tensors are respectively 608×960, 416×256 and 896×544. Tiny finds 3/69/31 lines, Small 2/69/33, and Medium 2/69/31; different model outputs mean latency comparisons between bundles do not establish equivalent accuracy.

Ordinary latency includes conversion to complete owned results from predecoded BGR24. Each fresh worker performs one separately recorded first inference, one warm-up, then exactly three measured calls. Load includes model import, compilation, CPU context and initial request construction. Peak RSS is the process lifetime resident high-water mark, including shared pages and setup; it is not retained activation memory alone.

| Model / page | Lines | Median ms | Min–max ms | Load ms | First inference ms | Peak RSS MiB |
| --- | --- | --- | --- | --- | --- | --- |
| tiny / sparse | 3 | 41.968 | 41.716–42.405 | 15.182 | 51.280 | 69.00 |
| tiny / dense | 69 | 65.269 | 63.132–65.860 | 6.622 | 86.754 | 44.95 |
| tiny / mixed | 31 | 73.748 | 72.767–76.992 | 6.484 | 82.047 | 76.86 |
| small / sparse | 2 | 93.090 | 92.916–96.111 | 21.210 | 102.971 | 146.11 |
| small / dense | 69 | 183.948 | 176.934–188.297 | 21.976 | 188.210 | 116.30 |
| small / mixed | 33 | 220.038 | 218.452–222.285 | 20.903 | 229.967 | 177.27 |
| medium / sparse | 2 | 368.775 | 363.050–382.523 | 94.088 | 382.173 | 449.14 |
| medium / dense | 69 | 668.296 | 660.951–686.451 | 93.513 | 671.148 | 351.91 |
| medium / mixed | 31 | 898.082 | 890.178–906.291 | 89.246 | 923.586 | 475.20 |

Observed sample ranges are 1.6–6.2% of the corresponding medians. Small dense is the widest relative range: 176.934, 183.948 and 188.297 ms. These fixed samples describe this local run; they do not establish statistical significance or tail-latency percentiles. Every first/warm/measured result was exactly equal within its case.

Separate instrumented processes split startup into import, CPU-context creation and compilation/request creation. The following ranges span one observation in each of the three fixture processes for a model. They are not components to add to an ordinary inference median. OS file caches were not flushed, so startup is fresh-process timing rather than a disk-cold claim.

| Model | Import ms range | CPU context ms range | Compile and request ms range |
| --- | --- | --- | --- |
| tiny | 3.434–3.621 | 0.080–0.129 | 1.850–1.929 |
| small | 10.438–10.797 | 0.087–0.089 | 8.604–9.359 |
| medium | 38.656–39.031 | 0.095–0.113 | 46.740–48.636 |

Keeping compiled models, CPU contexts and requests warm avoids repeatedly paying these costs. First-use activation allocation and shape-dependent packing remain in the separately recorded first inference.

The elapsed stages below come from one warm-up and one production-path profile per case. All values are milliseconds. The six phases run sequentially; total returns borrowed results and also includes small orchestration costs. **These are separate runs from the ordinary medians**, so the profile columns must not be rescaled to the ordinary table or subtracted from it to estimate overhead.

| Model / page | DET prepare | DET infer | DB | Crop | CLS | REC | Total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tiny / sparse | 2.850 | 29.331 | 0.430 | 0.338 | 1.394 | 7.918 | 42.274 |
| tiny / dense | 0.529 | 8.293 | 0.303 | 0.600 | 19.250 | 37.471 | 66.461 |
| tiny / mixed | 2.042 | 24.144 | 0.531 | 1.351 | 8.770 | 36.344 | 73.197 |
| small / sparse | 3.337 | 63.024 | 0.450 | 0.422 | 1.462 | 25.251 | 93.961 |
| small / dense | 0.530 | 15.049 | 0.295 | 0.670 | 18.506 | 154.502 | 189.572 |
| small / mixed | 2.119 | 51.303 | 0.616 | 1.120 | 9.070 | 154.621 | 218.865 |
| medium / sparse | 2.926 | 261.539 | 0.445 | 0.445 | 1.508 | 98.187 | 365.070 |
| medium / dense | 0.607 | 52.013 | 0.313 | 0.680 | 18.894 | 590.693 | 663.218 |
| medium / mixed | 1.996 | 213.122 | 0.631 | 1.360 | 9.288 | 657.084 | 883.495 |

Unassigned wall time outside the six phases is only 0.013–0.020 ms. DB costs 0.295–0.631 ms and perspective cropping 0.338–1.360 ms. Owned conversion, timed separately after the profiled call, costs 0.0013–0.0082 ms. Recognition wall includes scheduling and ordered result assembly. Classification wall includes preprocessing, backbone, compact head and applied 180-degree rotations.

The following counters expose nested work on the 69-line dense page. Parallel counters sum elapsed worker durations, including any time a worker call is active; they are **not process CPU time and not extra wall phases**. DET/REC reshape and binding counters include sites already nested in elapsed phases. Summing this table into the preceding totals would double-count work.

| Nested work ms | Tiny dense | Small dense | Medium dense |
| --- | --- | --- | --- |
| Crop workers | 3.52924 | 3.51112 | 3.95808 |
| CLS preprocessing | 7.88766 | 7.79204 | 8.62180 |
| CLS backbone | 108.82934 | 111.70229 | 112.38070 |
| CLS compact head | 0.11754 | 0.12825 | 0.12396 |
| CLS pixel rotation | 0.00000 | 0.00000 | 0.00000 |
| REC preprocessing | 5.38208 | 5.43446 | 5.85379 |
| REC backbone | 221.74917 | 986.17287 | 4318.81329 |
| Vocabulary projection and argmax | 16.70838 | 68.33658 | 107.05738 |
| CTC decode | 0.01733 | 0.02667 | 0.04575 |
| DET and REC reshape calls | 0.02692 | 0.04596 | 0.06613 |
| Binding and validation | 0.01341 | 0.02825 | 0.04263 |
| Result storage | 0.00988 | 0.03659 | 0.02558 |

The CLS backbone is about 93% of dense orientation worker time. Vocabulary projection is 7.0%, 6.5% and 2.4% of backbone-plus-projection worker time for Tiny, Small and Medium. CTC costs at most 0.046 ms summed work across all nine profiles, and warm reshape calls at most 0.067 ms. These costs make a decoder rewrite, different public result ownership, or more shape caching poor first performance targets for the sampled pages.

All dense profiles applied zero 180-degree rotations, whereas the sparse profiles rotated all 3/2/2 detected lines. Mixed pages rotated 1/2/3 lines. A zero-rotation dense fixture does not justify disabling classification for arbitrary inputs.

The source review identifies the existing mechanism and the remaining opportunity at each link:

| Pipeline link | Current implementation | Implication from this measurement |
| --- | --- | --- |
| Input and model/request setup | [ImageView](../src/image.rs) borrows validated pixels. The [builder](../src/builder.rs) compiles shared weights and creates requests; [engine leases](../src/engine.rs) reuse requests with bounded idle retention. | Reuse is already supported. Per-call reconstruction incurs the measured setup costs; engine admission waiting and concurrent service latency need their own workload. |
| Detector preprocessing | [Resize and normalization](../src/preprocess.rs) use cached interpolation maps/rows, fixed-point byte interpolation, normalization lookup tables and direct selected-layout writes. | 0.529–3.337 ms. The final vertical interpolation/store loop is a possible focused target, especially Tiny sparse (6.7% of its profile), but offers less than the network. |
| Detector inference | The [optimized runtime](../src/runtime/optimized.rs) fuses operations/stores, aliases eligible values and reuses arenas; packed weights and native SIMD are already present. Narrow 8/12/24-channel NEON paths are enabled for the selected 2×2/3×3 families. | Optimize the remaining measured convolution shapes. The previous report's proposal to enable these narrow paths is already implemented. |
| DB postprocessing | [DB processing](../src/db.rs) thresholds/scans the map, fills components, constructs hulls/minimum-area boxes, scores and unclips polygons, then sorts reading order. It retains scratch storage and uses pixel scan kernels. | Geometry can grow with complex contours, but this workload is below 0.64 ms. Larger-contour evidence would be needed to justify geometry work. |
| Perspective crops | [Line extraction](../src/lines.rs) fills a retained BGR slab using 16-row tasks. [Crop mapping](../src/crop.rs) already uses portable SIMD mapping and interior cubic sampling on ARM, with clamped borders; vertical rotation is folded into extraction. | Further sampling specialization is secondary here. Direct crop-to-network fusion changes the cubic-byte-then-bilinear numerical contract. |
| Orientation | [Classifier](../src/orientation.rs) preprocesses to 80×160, runs a feature prefix and compact two-class head, then rotates selected crops. Independent lines use up to eight one-thread classifier owners. CLS finishes before REC. | Tiny dense spends 19.250 ms here. Target the backbone or investigate batching if orientation support is required. Omission needs an upright-input contract and separate accuracy checks. |
| Recognition sizing and binding | [Recognizer](../src/recognizer.rs) uses height 48 and adaptive widths rounded to 32; [sessions](../src/runtime.rs) retain eight prior shape plans and high-water buffers. | Dense crops mostly use widths 96/128; mixed pages reach 1536 (Tiny) or 1472 (Small/Medium). Warm binding is already cheap. Fixed width 320 can compress long text and changes the accuracy tradeoff. |
| Recognition scheduling | [OCR scheduling](../src/ocr.rs) assigns width-sorted units to stable owners; [CPU policy](../src/cpu.rs) selects eight owners for enough independent units. A uniquely dominant single crop can use two nested workers from the same fixed pool. | The long-crop fix is active. Owner measurements below quantify the remaining tail; arbitrary extra threads or work stealing are not established improvements. |
| Recognition backbone | [NEON pointwise execution](../src/runtime/nhwc/pointwise/neon.rs) already uses six rows × sixteen channels, ordered FMA and packed panels. Dense convolutions and depthwise kernels have dedicated paths. | Large-channel 1×1 layers dominate the Small/Medium probes. Investigate input/weight reuse and tile/store overhead on concrete shapes before changing broad dispatch. |
| Projection and decoding | The [recognizer head](../src/recognizer.rs) computes compact winning classes/probabilities without materializing full vocabulary logits; [CTC decode](../src/recognizer.rs) collapses winners into reusable text/span storage. | Projection is secondary, especially for Medium. CTC itself is negligible in this workload. |
| Results and retained memory | [OCR result assembly](../src/ocr.rs) preserves input order in stable slabs; callers can borrow or own results. Workers retain private activations and resize buffers. | Copy cost is negligible. Medium mixed retains 175.57 MiB activations plus 19.01 MiB workspace and peaks at 475.20 MiB RSS. Batching/more owners must be evaluated against memory and warm-buffer retention. |

Fused production operator diagnostics narrow the network targets. DET uses the actual sparse image at `[1,3,960,608]`, eight threads, one warm-up and one measured pass per model. REC uses deterministic synthetic input for one serial feature-prefix warm-up and one measured pass at each listed shape; it excludes the vocabulary head. All five diagnostic outputs preserved float bits. Operator timing is a separate diagnostic and cannot be multiplied directly into page-stage time.

| Production operator probe | All steps ms | 1×1 conv ms | 1×1 share | Layout conversion ms |
| --- | --- | --- | --- | --- |
| tiny DET | 27.736 | 11.432 | 41.2% | 0.0000 |
| small DET | 61.106 | 29.950 | 49.0% | 1.9698 |
| medium DET | 266.457 | 146.884 | 55.1% | 0.0000 |
| small REC width 128 | 11.018 | 9.173 | 83.3% | 0.0025 |
| medium REC width 640 | 210.807 | 195.751 | 92.9% | 0.0081 |

Small REC repeatedly spends 0.560–0.680 ms in its 384↔768 pointwise layers over 96 spatial rows. Medium width 640 spends about 9.65–10.25 ms in representative 512→1024 and 768→1536 layers. Attention-related matrix multiplication, softmax, layer normalization and transpose together occupy only about 1.2% and 0.8% of the two prefixes. Large pointwise convolution is a more promising target than attention, layout-copy removal or CTC.

Sparse DET's 1×1 work is 41.2%, 49.0% and 55.1% of operator time. Other concrete targets are Tiny's 64→16 3×3 layer at 1.980 ms, Small's 96→24 3×3 at 4.572 ms, and Medium's 256→64 3×3 at 23.938 ms and 128→64 stride-two 3×3 at 13.287 ms. Small's 96→24 layer now runs NHWC, so its remaining cost is not an absent channel-tail path. Small's separately measured layout conversion is 1.970 ms; removing it alone has a much lower ceiling than reducing convolution work.

Recognition owner intervals explain the mixed-width critical path. All owners begin within 0.026 ms of dispatch in these three profiles. The longest crop is assigned two kernel workers; other owners have one. Completion offsets share one dispatch origin.

| Mixed page | Long width | Kernel workers | Long owner ends ms | Last other ends ms | Residual tail ms | Tail / total |
| --- | --- | --- | --- | --- | --- | --- |
| tiny | 1536 | 2 | 36.309 | 35.295 | 1.014 | 1.39% |
| small | 1472 | 2 | 154.610 | 142.441 | 12.169 | 5.56% |
| medium | 1472 | 2 | 657.072 | 639.275 | 17.798 | 2.01% |

With every other completion fixed, eliminating only the final lone-owner tail would remove at most 1.4%, 5.6% and 2.0% of these page profiles. This is a conditional ceiling for that specific tail, not an upper bound on all possible scheduling changes. Medium's long owner records 653.5 ms of calling-thread CPU during its 657.1 ms interval; helper CPU is excluded. This points toward active computation rather than a late launch. Dense owners still vary: Tiny completes in 24.7–37.4 ms, Small in 120.6–154.4 ms, and Medium in 515.4–590.6 ms. Width balance does not guarantee equal execution cost or core placement, and a Rayon worker index does not identify a physical CPU.

Paired process CPU/wall ratios are approximately 5.2–6.1 on sparse pages and 6.3–7.4 on dense pages. These sum all process threads and indicate parallel CPU use; they do not by themselves prove cache bandwidth, register spilling, thermal behavior or a P/E-core scheduling cause.

The optimization order supported by the current evidence is:

1. **Small/Medium recognition pointwise convolution.** Start with one exact shape, such as Small rows/K/channels = 96/384/768 or Medium = 960/512/1024. Inspect generated code and memory reuse before choosing a mechanism. Keep ordered reductions, tail handling and fused epilogues intact. Earlier [48/96/192-row groups and four/six-row tiles](macos-arm64-recognition-optimization.md) were inconclusive; [K=256/512 blocking](macos-arm64-narrow-convolution-optimization.md) did not establish a useful gain. Repeating these changes without new evidence is a low-value experiment.
2. **Sparse detector convolution.** Target the pointwise layers and the remaining 3×3 shapes above. Optimize actual arithmetic and data reuse; narrow-path enablement is already retained. A lower detector resolution or a smaller model may be faster, but changes recognition accuracy and requires a representative accuracy evaluation.
3. **Tiny dense orientation.** Investigate the fixed-shape CLS backbone or true classifier batching. Same-width REC batching is already available and is a separate option. Unconditional CLS/REC overlap competes for the same bounded pool and has previously rejected variants; measure a concrete new policy before adopting it. Disabling CLS is only an application option when inputs are guaranteed upright.
4. **Scheduling and batching only with a demonstrated case.** Consider a cost model including per-line overhead or validated exact-width batching on repeated widths. Preserve stable ownership and cache reuse. The measured long-crop tail is now comparatively small; extra threads and default batching could increase memory or reduce independent work.
5. **Defer small stages.** Preprocessing is the most plausible secondary target for Tiny sparse. DB, crop, warm binding, CTC and owned conversion have small absolute ceilings on these images.

For scale, a hypothetical **20% reduction in the whole REC phase** would reduce dense Tiny/Small/Medium profiled page latency by **7.49/30.90/118.14 ms**, or **11.3%/16.3%/17.8%**, if other phases remained unchanged. Halving the whole Tiny dense CLS wave would save **9.63 ms (14.5%)**. A 20% DET inference reduction would save **13.4–14.3%** of sparse page time. These are sensitivity calculations, not forecasts or measured candidate improvements; a kernel gain is diluted by other work, scheduling and memory contention.

A next implementation experiment should first pass focused correctness checks, then use one warm-up and three single-call release measurements on one or two affected layer shapes. Only a useful result should expand to the corresponding page with three samples, identical settings and output/memory checks. Keep total execution within 60 seconds per version and rebuild after any source change. The present analysis needs no further unchanged-code runs.

Local validation passed **7 profiling semantic tests**, **9 saved-golden comparisons** with unchanged 0.5-pixel coordinate and `1e-4 + 1e-4×abs(expected)` score tolerances, **9 exact warm-up/profile output comparisons**, and **5 exact operator output-bit comparisons**. Each ordinary worker also checked complete equality of its first, warm-up and three measured results. Model assets and predecoded pixels passed pinned hashes. No production code, dependency, golden output or tolerance changed.

Performance execution totaled **26.133 seconds** under one shared 60-second cap, including process startup/model loading and excluding builds and focused correctness tests. Counts were 9 ordinary first calls + 9 ordinary warm-ups + 27 ordinary measured calls; 9 profile warm-ups + 9 profiles; and 5 operator warm-ups + 5 instrumented passes: **73 inference/prefix executions**. Ordinary/profile processes had a 10-second timeout; grouped operator diagnostics had 12 seconds, always capped by the remaining aggregate budget. There were no retries or automatic sample growth. Source, harness and executable hashes were stable.

Build and correctness commands were:

```sh
cargo test --release --locked --features profiling --lib profile::tests -- --test-threads=1
cargo build --release --locked --example cpu_benchmark
cargo build --release --locked --features profiling --example cpu_profile
```

Raw local artifacts and the driver are in `target/pipeline-analysis-20261017-head-2e5287f`. For reproduction, use a fresh directory at the same depth, rebuild/copy the two examples, save the focused test log, and execute the embedded driver. It hash-checks existing predecoded benchmark inputs; generate those with the standard benchmark harness first if absent. The portable report includes every invoked worker/test command and all measured data.

The scope is one physical machine, three inputs, three bundles and one concurrency/thread configuration. No CI, VM, remote runner or cross-platform timing was used. This is neither full performance qualification nor a C# comparison. Decode/serialization, alignment/token boxes, engine admission waiting, concurrent throughput, other hardware and application accuracy changes remain outside these measurements. One stage sample per case supports bottleneck prioritization, not small-effect claims.
