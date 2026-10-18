# Current Windows CPU24 OCR latency analysis

For fresh measurements of revision `b30436a`, including recognition owner
completion times and physical operator attribution, see the
[2026-10-18 pipeline analysis](windows-2026-10-18-pipeline-analysis.md).

These timings precede the subsequently retained [AVX2 short-depthwise optimization](windows-avx2-depthwise-optimization.md). That record contains separate matched before/after measurements; the stage timings below are not post-optimization results.

There is room to improve recognition speed, but the useful target depends on the workload. On the current implementation, detector inference occupies **74.9%** of the sparse Tiny profile. Dense Small and Medium spend **67.0% and 85.9%** of elapsed time in recognition. Dense Tiny is more evenly divided: **26.7% detection inference, 29.4% classification and 40.1% recognition**. Classification's neural backbone accounts for 92–95% of its summed worker time; rotation and the two-class head are small.

These are fresh measurements of production revision `38e054c` plus the diagnostic changes included in the [raw evidence](reports/windows-x64-2026-10-17-cpu24-stage-analysis.json). OCR arithmetic, model settings and scheduling policy were retained. The diagnostic changes add CLS substage counters, permit the profiler to use the local 24-thread context, and report setup, owned conversion, output hashes and recognition width distributions. The previous [eight-CPU analysis](windows-current-pipeline-analysis.md) describes an earlier revision and a different CPU configuration; its page timings are not pooled with these results.

All work ran locally on the physical Windows x64 machine on 2026-10-17: AMD Ryzen 9 5950X, 12 enabled cores / 24 logical processors reported by Windows, approximately 64 GiB RAM. The tests used the pinned `nightly-2026-09-08`, release optimization, default parallel features, one request, automatic SIMD/layout and fusion, and process affinity `ffffff`. Detector, preprocessing and owner budgets cap at 16; nested recognition work still shares the full 24-thread pool. Twenty-four logical CPUs are not twenty-four physical cores.

Classification is enabled at threshold 0.0, detector limit is 960, recognition height is 48 with adaptive widths rounded to 32, batch size is one, and alignment capture is disabled. Model-specific detector thresholds and pinned dictionaries are unchanged. Page timings start from predecoded BGR24 and include complete owned-result construction. Capture, file I/O/image decoding, model setup, engine lease acquisition and JSON serialization are outside these timings.

**Inputs and scheduling.** Three source images cover different costs. Small and Medium reuse the dense image to isolate how model size changes stage composition; this is not an accuracy comparison.

| Case | Source size | DET tensor H×W | Detected lines | Active CLS owners | Active REC owners / intra-op threads per owner |
|---|---:|---:|---:|---:|---:|
| Tiny sparse / `imgs_1` | 720×1150 | 960×608 | 3 | 3 | 3 / 8 |
| Tiny dense / `imgs_00018069` | 430×267 | 256×416 | 69 | 16 | 16 / 1 |
| Tiny mixed / `imgs_00006737` | 896×528 | 544×896 | 31 | 16 | 8 / 3 |
| Small dense | 430×267 | 256×416 | 69 | 16 | 16 / 1 |
| Medium dense | 430×267 | 256×416 | 69 | 16 | 16 / 1 |

Owner/thread counts follow the current source policy and reported budgets for one uncontended request; they are not measurements of simultaneous CPU utilization. The newer sixteen-owner policy applies on Windows AVX2/FMA contexts of capacity 24–31 with at least 32 recognition units. The 31-line mixed page retains eight owners. Classification completes before recognition begins.

**Ordinary complete OCR latency.** Each separate, uninstrumented process performs one first-call warm-up and exactly three measured calls. Every complete `OwnedOcr` result matches its warm-up control. There is no additional separately executed first inference.

| Case | Three samples, ms | Median, ms | Observed range, ms |
|---|---|---:|---:|
| Tiny sparse | 51.0788, 47.9658, 48.3816 | **48.3816** | 47.9658–51.0788 |
| Tiny dense | 49.5496, 50.5574, 45.3641 | **49.5496** | 45.3641–50.5574 |
| Tiny mixed | 76.9393, 68.4793, 72.8305 | **72.8305** | 68.4793–76.9393 |
| Small dense | 110.8518, 112.1846, 118.9714 | **112.1846** | 110.8518–118.9714 |
| Medium dense | 413.6565, 509.1008, 473.0911 | **473.0911** | 413.6565–509.1008 |

These bounded samples show appreciable noise, especially for Medium. They are not confidence intervals or proof of steady state. The previous CPU24 experiment's medians are a separate session, not a paired baseline for declaring a regression or improvement here. No unchanged case was rerun for cleaner numbers.

**Sequential elapsed stages.** Values below are milliseconds from one instrumented call per case, after one warm-up in a separate process. They describe the borrowed-result production path. The six named phases execute in order; their sum plus the remainder equals that profile's total. Summed worker counters are shown separately below.

| Elapsed stage, ms | Tiny sparse | Tiny dense | Tiny mixed | Small dense | Medium dense |
|---|---:|---:|---:|---:|---:|
| DET resize / normalization | 1.044 | 0.360 | 0.881 | 0.350 | 0.316 |
| DET inference | **35.526** | 11.898 | **31.288** | 25.479 | 46.826 |
| DB postprocessing / reading order | 0.651 | 0.545 | 0.865 | 0.447 | 0.508 |
| Crop planning / extraction | 0.618 | 0.756 | 1.899 | 0.894 | 0.892 |
| Classification / orientation wave | 1.719 | **13.140** | 6.040 | 14.910 | 12.982 |
| Recognition wave / assembly | 7.799 | **17.905** | **29.767** | **85.743** | **374.591** |
| Remaining orchestration / timer boundaries | 0.048 | 0.033 | 0.046 | 0.057 | 0.052 |
| Profiled borrowed total | **47.404** | **44.638** | **70.786** | **127.880** | **436.166** |
| Separate owned conversion | 0.0057 | 0.0135 | 0.0100 | 0.0168 | 0.0157 |

The profiled total is a separate execution from the ordinary median. For example, Small's profile is 127.88 ms while its ordinary median is 112.18 ms. Instrumentation and host variation both affect these observations. Do not subtract the two to estimate timer overhead or owned-copy cost, or rescale stage timings to the ordinary median. Owned conversion is timed directly outside `run_profiled`.

| Share of profiled total | Tiny sparse | Tiny dense | Tiny mixed | Small dense | Medium dense |
|---|---:|---:|---:|---:|---:|
| DET inference | 74.9% | 26.7% | 44.2% | 19.9% | 10.7% |
| Classification | 3.6% | 29.4% | 8.5% | 11.7% | 3.0% |
| Recognition | 16.5% | 40.1% | 42.1% | 67.0% | 85.9% |

**Work within parallel phases.** Each entry sums elapsed durations of owner calls, including nested waits. Entries overlap across owners and with the enclosing wall phase. They are not process CPU time, must not be added to the elapsed-stage table, and must not be divided by a fixed worker count to infer page latency.

| Summed worker work, ms | Tiny sparse | Tiny dense | Tiny mixed | Small dense | Medium dense |
|---|---:|---:|---:|---:|---:|
| Crop workers | 2.494 | 8.145 | 10.679 | 7.522 | 8.441 |
| CLS inclusive workers | 5.060 | 154.887 | 79.803 | 191.398 | 163.480 |
| ↳ CLS preprocessing | 0.322 | 8.156 | 4.730 | 9.691 | 9.289 |
| ↳ CLS backbone / session execution | **4.663** | **146.531** | **74.965** | **181.453** | **153.976** |
| ↳ CLS compact head | 0.008 | 0.157 | 0.078 | 0.197 | 0.167 |
| ↳ Applied 180° rotations | 0.065 | 0 | 0.0025 | 0 | 0 |
| REC preprocessing | 0.618 | 5.094 | 6.875 | 5.393 | 5.488 |
| REC backbone / session execution | **15.671** | **220.531** | **204.617** | **1196.438** | **5214.066** |
| Vocabulary projection / argmax | 0.681 | 19.435 | 11.044 | 101.970 | 195.526 |
| CTC text decode | 0.0063 | 0.0401 | 0.0481 | 0.1105 | 0.1822 |
| DET + REC reshape calls | 0.0012 | 0.0674 | 0.0873 | 0.1384 | 0.1980 |
| Binding / preparation | 0.0026 | 0.0235 | 0.0179 | 0.0471 | 0.0652 |
| Result storage | 0.0017 | 0.0192 | 0.0141 | 0.0426 | 0.0544 |

The indented CLS counters are contained in CLS inclusive work; its small remainder includes validation, context entry, metadata and timer boundaries. Backbone counters for DET, CLS and REC include session execution overhead and subtract the measured preprocessing callback. Result storage combines serial preparation, worker-slot copies and final ordered assembly. Binding/reshape/storage are mixed serial and worker counters, not extra independent wall stages.

All three sparse crops receive a 180° correction; the mixed page has one correction. Dense pages have none, but still execute the classifier for every line. The new counters show that optimizing rotation or the two-class head will recover little of dense Tiny's 13.14 ms classification wave. The neural backbone is the target if classification remains required.

Among measured REC preprocessing + backbone + projection + CTC work, the backbone accounts for **90.0% Tiny dense, 91.8% Small dense and 96.3% Medium dense**. These are work compositions, not those operators' direct shares of page latency.

**Setup and retained state.** The following single setup samples come from the ordinary executable. Import includes model/dictionary file reads, ONNX parsing and normalization. Compilation/request setup includes graph planning and initial requests; lazy per-line workers and shape-dependent first use can still appear in the first-call warm-up.

| Case | Import, ms | CPU context, ms | Compile / request, ms | Setup total, ms | First-call warm-up, ms |
|---|---:|---:|---:|---:|---:|
| Tiny sparse | 9.659 | 0.614 | 4.439 | 14.712 | 54.909 |
| Tiny dense | 15.401 | 0.848 | 7.093 | 23.341 | 59.118 |
| Tiny mixed | 10.883 | 0.608 | 4.893 | 16.384 | 92.967 |
| Small dense | 32.245 | 0.456 | 20.902 | 53.602 | 132.148 |
| Medium dense | 128.033 | 0.717 | 136.377 | 265.127 | 477.885 |

Asset hashing runs before import, and filesystem cache state is uncontrolled. These are fresh request/process setup samples, not cold-disk measurements. Keep models and mutable requests alive to amortize setup; recreating the entire pipeline per image can add approximately 15–265 ms in these samples.

| Case | Retained activation, MiB | Retained workspace, MiB | Combined request buffers, MiB | Process CPU / wall equivalents |
|---|---:|---:|---:|---:|
| Tiny sparse | 39.06 | 5.46 | 44.53 | 10.28 |
| Tiny dense | 17.33 | 14.29 | 31.62 | 13.10 |
| Tiny mixed | 43.74 | 12.27 | 56.02 | 13.89 |
| Small dense | 39.40 | 27.35 | 66.75 | 12.56 |
| Medium dense | 63.90 | 27.55 | 91.45 | 13.59 |

CPU equivalents are the sum of three measured process CPU durations divided by the sum of their wall durations. Windows counters are coarse; these values describe average process CPU consumption, not physical-core occupancy, IPC or proof of a bandwidth bottleneck. Request buffers exclude shared immutable weights. Kernel scratch is already part of workspace. Raw process private-commit peaks include model validation, decoding all helper fixtures and setup; they are not an isolated deployed-OCR memory requirement. An engine idle-byte cap below a warmed request's retained size can cause eviction and lose its reuse benefit.

**Trace of the current implementation and optimization assessment.**

| Pipeline link | Current implementation | Performance implication |
|---|---|---|
| Pixel acquisition / decode | [`ImageView`](../src/image.rs) accepts borrowed BGR/RGB/BGRA/RGBA with stride; [`ocr` example](../examples/ocr.rs) separately decodes image files. | Avoid caller-side repacking when the input already has a supported format. Capture, decode and format-conversion latency were not individually measured here. |
| Import / compilation / requests | [`builder.rs`](../src/builder.rs) loads models and dictionaries, compiles DET/REC/optional CLS sequentially, and constructs reusable requests. | Reuse is the first application-level improvement if setup currently occurs per frame. Parallel cold compilation is an untested setup-only candidate with memory tradeoffs. |
| Admission / concurrency | [`engine.rs`](../src/engine.rs) leases retained requests; [`cpu.rs`](../src/cpu.rs) shares one bounded pool and adjusts nested budgets under demand. | No lease queue or concurrent throughput was measured. Increasing worker counts indiscriminately is not justified by this single-request run. |
| Validation / binding / shapes | [`detector.rs`](../src/detector.rs), [`recognizer.rs`](../src/recognizer.rs), [`runtime.rs`](../src/runtime.rs) validate shapes and retain eight cached logical/physical plans. | Warm reshape/preparation is tiny. Shape caching already exists; preserving warmed request state matters more than another metadata rewrite. |
| DET resize / normalize | [`preprocess.rs`](../src/preprocess.rs) caches interpolation maps/rows, uses SIMD pixel kernels and writes the selected layout directly. | Only 0.32–1.04 ms elapsed here. Further pixel optimization has a small ceiling compared with DET inference. |
| DET inference | [`runtime/optimized.rs`](../src/runtime/optimized.rs) propagates layouts, fuses stores, aliases eligible operations and reuses arenas. AVX2 detector upsampling was already optimized. | The first sparse-page target: 35.53 ms in Tiny. Inspect remaining regular, depthwise and pointwise convolutions at the actual 960×608 shape. Lower detector resolution changes detection behavior and needs accuracy evidence. |
| DB boxes / reading order | [`db.rs`](../src/db.rs) uses SIMD threshold/scan helpers and reusable component/geometry scratch. | 0.45–0.87 ms elapsed. Its hull rectangle search can become expensive on complex contours, but these inputs do not justify prioritizing it. |
| Perspective crops | [`lines.rs`](../src/lines.rs) keeps a contiguous slab and distributes 16-row jobs; [`crop.rs`](../src/crop.rs) and [`pixel_kernels.rs`](../src/pixel_kernels.rs) perform bicubic sampling. | 0.62–1.90 ms elapsed. A validated affine/axis-aligned fast path or multi-output sampling is plausible; retaining temporary job vectors is a smaller candidate. Preserve interpolation and geometry parity. |
| Classification / orientation | [`orientation.rs`](../src/orientation.rs) runs a fixed `[1,3,80,160]` prefix and compact two-class head, one intra-op thread per owner; [`lines.rs`](../src/lines.rs) rotates before REC. | Dense Tiny spends 13.14 ms here, overwhelmingly in its backbone. Omit CLS only when the input orientation contract permits it, or investigate the fixed-shape backbone. Rotation/head optimization has little measured value. |
| REC widths / scheduling / batches | [`ocr.rs`](../src/ocr.rs) sorts widths longest-first, assigns deterministic owners by accumulated width cost, and supports exact-width batches; [`cpu.rs`](../src/cpu.rs) already selects sixteen owners for these dense CPU24 pages. | Switching from eight to sixteen owners is already implemented. The mixed page has a width-1536 line and uses eight owners. Record owner completion times before changing cost weights or the 32-unit threshold. |
| REC preprocessing | [`preprocess.rs`](../src/preprocess.rs) resizes directly into the chosen layout and initializes padded tensors when required. | Roughly 5–7 ms summed work on multi-line inputs, overlapping across owners. Initializing only right padding is a small candidate, not a substitute for backbone optimization. |
| REC backbone | [`runtime/nhwc/pointwise.rs`](../src/runtime/nhwc/pointwise.rs) uses packed full-K six-row × sixteen-output tiles, direct fused stores and 96-row groups for large channel products. | Strongest dense Small/Medium compute target. The short-K inline path already exists; larger channel products remain important. Existing evidence has rejected several straightforward tile/block/store changes. |
| Vocabulary projection | [`projection.rs`](../src/projection.rs) and [`runtime/simd/argmax.rs`](../src/runtime/simd/argmax.rs) keep packed panels and compact winners rather than full vocabulary probabilities. | Removing dense softmax/full logits would duplicate existing work. Projection is 7.8% / 3.6% of dense Small/Medium measured REC component work, below the backbone. Preserve class ties and public score semantics. |
| CTC / geometry / result storage | [`recognizer.rs`](../src/recognizer.rs) reuses CTC storage; [`ocr.rs`](../src/ocr.rs) assembles ordered reusable output slabs. Optional alignment/token-box work is disabled here. | CTC, warm storage and owned conversion are small. This run does not establish the cost of alignment-enabled output. |

**What to optimize next.**

1. **Dense Small/Medium: investigate backbone compute with actual convolution shapes.** Earlier serial prefix attribution on the unchanged convolution arithmetic found 1×1 convolutions responsible for 79.7% of Small and 90.0% of Medium physical-step time. Those are historical, serial shape probes, not percentages to multiply into this CPU24 page table. The [large-channel analysis](windows-large-channel-convolution-analysis.md) and [subsequent experiments](windows-large-channel-optimization.md) already rejected K=256 blocking, input packing, 96→192 grouping, alternate 8×8/4×24 tiles and broad fused-store specialization as established improvements. Hardware cache/IPC/bandwidth evidence is still missing. Obtain it for one real large-channel layer before another speculative kernel rewrite. Require complete-result parity and a page-level improvement before adoption.

2. **Test exact-width batch size two as a separate bounded experiment.** The implementation already exposes `.recognition_batch_size(2)`. Batching may improve weight reuse, but also changes kernel shapes, preprocessing and scheduling units. Sixty-nine lines produce at least 35 batch units, preserving the current sixteen-owner threshold; other pages may cross below 32 units and select eight owners. No batch-size speedup is claimed by this experiment.

3. **Sparse Tiny: focus on DET convolutions.** The measured DET inference ceiling is 35.53 ms / 74.9% of the profile. New changes should target remaining shapes, since direct AVX2 upsampling and other graph optimizations are already present. Model size and detector resolution are application tradeoffs, requiring recognition/detection quality checks.

4. **Upright Tiny workloads: evaluate skipping classification against actual orientation needs.** Its 13.14 ms dense wave is a meaningful opportunity. A separate historical [eight-CPU toggle test](windows-no-cls-latency.md) preserved dense text but changed all three sparse texts and one mixed text when CLS was disabled. That historical speedup is not a CPU24 prediction. Geometric vertical-crop rotation remains separate from optional CLS. If CLS is required, its backbone deserves attention before resize, head or rotation.

5. **Mixed pages: measure the long-line scheduling tail.** This page spends 44.2% in DET inference and 42.1% in recognition. The recognition width histogram includes one width-1536 and one width-672 line among many widths 64–448. The deterministic width-weighted policy already exists. A per-owner completion trace can determine whether better cost weights, exact-width grouping or a different nested-thread budget can shorten the critical path. Forcing width 320 may compress long text and is not a free optimization.

For scale, halving the entire recognition wave would reduce the profiled Small/Medium total by **33.5% / 42.9%**. Halving sparse Tiny DET inference would reduce its profiled total by **37.5%**. These are conditional critical-path bounds, not forecasts for any particular implementation. Even eliminating the whole measured DB stage saves less than 0.9 ms per page, and eliminating owned conversion saves at most 0.017 ms in these samples.

**Validation and reproduction.** The release profiling library suite passed **329 tests**, with 12 diagnostic tests ignored, including serial/parallel classification, rotation, disabled-counter and complete-output parity checks. Separate release executables were built with profiling disabled for ordinary latency and enabled for stage diagnosis. All model, source-image and decoded-pixel checks passed their pinned hashes. Five ordinary processes and five profile processes verified stable complete outputs internally; their complete serialized output digests also matched across build modes for every case. Source and executable hashes remained fixed during measurement.

There were **15 ordinary measured calls, five profiled measured calls and ten warm-up calls: 30 inference calls total**. Conservative process execution, including helper startup, asset checks, decoding, import, setup and collection waits, totaled **8.935 seconds**, below the aggregate 60-second cap. Every process had at most a 10-second external deadline. No retries, sampling growth, CI, remote runner or VM were used. Builds were excluded from the execution budget. The existing Windows unused `UNBOUND_AFFINITY` import warning remains unrelated.

The [evidence JSON](reports/windows-x64-2026-10-17-cpu24-stage-analysis.json) contains every sample and counter, source manifest and patch, executable/model/input hashes, widths, memory figures, process CPU samples, settings and commands. The bounded runner and individual logs remain in `target/cpu24-stage-analysis-20261017/`. Its runner refuses to overwrite prior measurements; use a fresh directory for a separately justified future experiment.

```powershell
cargo test --release --locked --features profiling --lib
cargo build --release --locked --example cpu_owner_probe
cargo build --release --locked --features profiling --example cpu_profile

# One ordinary invocation and one profile invocation per selected case.
# OWNERS=0 preserves the automatic policy; it does not request zero workers.
./target/release/examples/cpu_owner_probe.exe tiny imgs_1 0
./target/release/examples/cpu_profile.exe tiny imgs_1 24 0
# Repeat only for tiny/dense, tiny/mixed, small/dense, medium/dense.
# Run inference after builds finish, with external 10-second deadlines
# and a total 60-second execution cap as in the saved run.ps1.
```

This scoped analysis establishes stage composition on three corpus images and one host/configuration. It does not qualify ground-truth OCR accuracy, concurrent throughput, engine queueing, capture/decode, alignment output, other CPU placements or other architectures. Native hardware counters, per-owner finish times and cold-import substeps remain unmeasured. No production speedup is attributed to the diagnostic changes.
