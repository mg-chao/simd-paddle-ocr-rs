The current CPU pipeline has a measurable scheduling opportunity and two main compute targets: detection on large sparse pages, and recognition backbones on dense pages. On this physical Windows x64 host, a focused uninstrumented Tiny/dense comparison reduced median latency from **84.966 to 69.326 ms (18.4% lower; 1.226× speed)** by selecting eight line workers in an eight-thread CPU context. Complete owned outputs matched exactly. Production defaults and kernels were left unchanged during this analysis.

The subsequent [AVX2/FMA adoption record](windows-avx2-recognition-scheduling.md)
implements Optimization 1 and records separate baseline/candidate measurements
across sparse/mixed pages, all three model sizes and concurrent requests. The
numbers and recommendations below describe the original analysis revision.

The subsequent [classification-disabled latency comparison](windows-no-cls-latency.md)
measures the orientation toggle on revision `061c33b`: Tiny/dense latency falls
from 69.2114 to 44.1710 ms with unchanged text; other fixtures expose the output
tradeoffs when crops need orientation correction.

The [Optimization 3 implementation](#optimization-3-detector-upsampling-on-windows-avx2fma)
at the end of this document records the subsequent detector change, measured
against `b446afd` with Optimizations 1 and 2 already present.

This analysis was performed on 2026-10-17 (Asia/Shanghai) against production commit `4c88abc2daaff3e93867c0fe7e099a304dafa039`. The machine identifies as an AMD Ryzen 9 5950X; Windows exposes 12 enabled cores / 24 logical processors and approximately 64 GiB RAM. All page measurements used eight logical CPUs with mask `0xff`, automatic SIMD/layout, fusion, adaptive REC widths, batch size one, classification enabled, and alignment capture disabled. This mask is eight logical processors, not a claim of eight separate physical cores. The host uses the AVX2/FMA path; this run does not validate ARM64 or AVX-512.

The [durable raw evidence](reports/windows-x64-2026-10-17-stage-analysis.json) records samples, stage counters, hashes, output checks, memory capacities, and diagnostic output. Full worker results remain in `target/cpu-benchmark/1792221478849-17784`; investigation files are in `target/stage-analysis-20261017`.

The ordinary release benchmark measured complete owned OCR results from predecoded BGR24 pixels. Each fixture received one separately reported first inference, one warm-up, and three measured calls. All three Rust/C# comparisons passed the existing correctness, timing, and memory gates. These are three of the 36 required qualification cells, not a full qualification. C# is a reference implementation of the same models and settings, not an accuracy target for different model presets.

| Tiny fixture | Lines | DET tensor W×H | Rust samples, ms | Rust median | C# median | Rust lower latency | Rust peak private MiB |
|---|---:|---|---|---:|---:|---:|---:|
| `imgs_1` | 3 | 608×960 | 45.376, 44.628, 45.433 | 45.376 | 53.891 | 15.8% | 64.58 |
| `imgs_00018069` | 69 | 416×256 | 85.580, 84.009, 82.670 | 84.009 | 126.568 | 33.6% | 35.80 |
| `imgs_00006737` | 31 | 896×544 | 90.331, 89.565, 88.908 | 89.565 | 125.179 | 28.5% | 67.73 |

Tiny model load/compile/request setup took 15.19–15.73 ms in these fresh worker processes. First inference was 57.239 / 87.859 / 96.573 ms for sparse / dense / mixed; warm-up was 44.745 / 84.897 / 91.962 ms. Reusing an `Ocr` request or an `OcrEngine` lease avoids paying setup and first-use costs on every image. Model setup includes parsing, normalization, compilation, weight packing and request construction; those substeps were not individually timed.

Separate release builds with `profiling` measured production stages: one warm-up plus one profiled call for each of the five cases below. Every profiled result equaled its warm-up result. Sparse is `imgs_1` (720×1150, three lines); dense is `imgs_00018069` (430×267, 69 lines); mixed is `imgs_00006737` (896×528, 31 lines). Small and Medium were expanded only on the same dense fixture to check whether the recognition bottleneck changes with model size.

| Elapsed stage, ms | Tiny sparse | Tiny dense | Tiny mixed | Small dense | Medium dense |
|---|---:|---:|---:|---:|---:|
| Detector resize / normalization | 1.763 | 0.311 | 1.631 | 0.454 | 0.427 |
| Detector inference | 35.536 | 8.504 | 25.557 | 23.994 | 51.187 |
| DB postprocessing | 0.713 | 0.461 | 0.809 | 0.456 | 0.451 |
| Crop planning / perspective sampling | 0.796 | 1.298 | 2.319 | 1.382 | 1.160 |
| Classification / rotation | 1.638 | 24.715 | 11.275 | 24.933 | 24.767 |
| Recognition wave / assembly | 7.843 | 47.985 | 47.132 | 180.464 | 713.746 |
| Total borrowed-result OCR | 48.333 | 83.317 | 88.769 | 231.735 | 791.793 |
| Unattributed elapsed remainder | 0.044 | 0.043 | 0.047 | 0.053 | 0.055 |

These elapsed stage counters are from the instrumented samples; do not add them to, average them with, or rescale them to the uninstrumented benchmark medians. The remainder contains uninstrumented orchestration, pool entry, binding/reshape outside the elapsed stage boundaries, and timer overhead. `recognition_wall_ns` includes scheduling and final result assembly. `crop_ns` includes planning as well as the crop wave. CLS preprocessing, its backbone, the classifier head, and 180-degree rotation are aggregated in the classification counter.

| Summed line-worker elapsed work, ms | Tiny sparse | Tiny dense | Tiny mixed | Small dense | Medium dense |
|---|---:|---:|---:|---:|---:|
| REC resize / normalization | 0.8155 | 4.6087 | 5.4481 | 4.3499 | 4.6773 |
| REC backbone | 15.2473 | 173.1805 | 159.9634 | 650.0982 | 2713.9466 |
| Vocabulary projection / argmax | 0.8679 | 12.6480 | 12.0196 | 50.4754 | 99.1153 |
| CTC decode | 0.0057 | 0.0444 | 0.0301 | 0.0861 | 0.1332 |
| DET + REC reshape calls (mixed accounting) | 0.0016 | 0.0498 | 0.0490 | 0.0771 | 0.0803 |
| Binding / preparation (mixed accounting) | 0.0026 | 0.0174 | 0.0107 | 0.0379 | 0.0482 |
| Result storage (mixed accounting) | 0.0018 | 0.0183 | 0.0088 | 0.0274 | 0.0396 |

The REC rows sum each line owner's elapsed call durations, including nested parallel kernel waits. They are neither additive wall stages nor process CPU time. Reshape and binding mix serial DET work with REC owner work. Result storage combines preparation, worker slot writes, and final assembly; much of it is already inside recognition wall time. Therefore none of these rows should be added to the previous table. Cropping likewise has a separate summed `crop_work_ns` in the JSON. The 0.0016–0.0803 ms reshape and sub-0.14 ms CTC sums make those poor primary targets for the measured workloads.

The implementation was traced through these links:

| Link | Current behavior and source | Performance interpretation |
|---|---|---|
| Image acquisition / decode | The public library takes a borrowed [`ImageView`](../src/image.rs); JPEG/PNG decode belongs to the example/application. | File I/O and decoding are outside these inference times. No timing claim is made for application ingestion. |
| Model import / compile | [`builder.rs`](../src/builder.rs), [`model.rs`](../src/model.rs), and [`runtime.rs`](../src/runtime.rs) load local ONNX, normalize/fold supported operations, select ISA/layout, share weights and packs, and build mutable requests. | Keep models and requests alive; the measured Tiny setup cost is substantial relative to a 45 ms sparse request. Cold filesystem-cache behavior was not isolated. |
| DET preprocessing | [`preprocess.rs`](../src/preprocess.rs) limits the longest side to 960, rounds tensor dimensions to multiples of 32, and writes normalized pixels directly in the selected layout with bounded row parallelism. | 0.31–1.76 ms here; already a small fraction of total latency. |
| DET inference | [`runtime/optimized.rs`](../src/runtime/optimized.rs) executes a fused physical plan with packed convolution kernels, NHWC propagation, aliasing, and reusable arenas. | Tiny sparse DET is 73.5% of profiled time; tiny mixed is 28.8%. This is the first kernel target for sparse documents. Exact DET kernel ranking was not measured in this run. |
| DB postprocessing | [`db.rs`](../src/db.rs) thresholds, finds components/boundaries, constructs hulls/quads, scores and unclips them, then sorts reading order. SIMD threshold/scan helpers and scratch reuse already exist. | 0.45–0.81 ms; do not prioritize a contour rewrite before inference. |
| Cropping | [`lines.rs`](../src/lines.rs) prepares a contiguous crop slab; [`crop.rs`](../src/crop.rs) applies perspective sampling, split into 16-row jobs for parallel extraction. | 0.80–2.32 ms. Reusing temporary crop/classification job vectors may trim allocations, but the measured upper bound is small. |
| Orientation | [`orientation.rs`](../src/orientation.rs) runs fixed-shape classification; [`lines.rs`](../src/lines.rs) applies rotations and completes this wave before recognition. | Tiny dense costs 24.72 ms (29.7%). Automatic classification already uses eight workers here. If upstream guarantees upright text, disabling classification is a useful quality-dependent option; this run did not measure or validate that ablation. |
| REC shape / scheduling | [`ocr.rs`](../src/ocr.rs) sorts by descending selected width, assigns stable ownership by accumulated width cost, and optionally groups exact-width batches. [`cpu.rs`](../src/cpu.rs) defaults x64 to four owners. [`recognizer.rs`](../src/recognizer.rs) rounds natural widths to multiples of 32; sessions cache eight previous shapes. | Shape binding is already cheap after warm-up. Dense Tiny has 36 of 69 lines at width 96, so short-line scheduling is worth testing. The mixed page includes a width-1536 line; forcing a short fixed width compresses its horizontal scale and can change recognition. |
| REC preprocessing | [`preprocess.rs`](../src/preprocess.rs) resizes each crop to height 48 and zero pads to the chosen width. | Summed work is 0.82–5.45 ms, usually overlapped across owners. Dense Tiny padding is 12.4% of aggregate tensor width; sparse/mixed are 2.8%/4.9%. Reducing padding changes shape semantics and needs accuracy checks. |
| REC backbone | The production prefix stops before the vocabulary head; kernels are in [`runtime/nhwc/pointwise.rs`](../src/runtime/nhwc/pointwise.rs), [`dense.rs`](../src/runtime/nhwc/dense.rs), and depthwise paths. | Dominant owner work. Recognition wall time is 57.6% of Tiny dense, 77.9% of Small dense, and 90.1% of Medium dense. |
| Projection / CTC | [`recognizer.rs`](../src/recognizer.rs) and [`runtime/simd/argmax.rs`](../src/runtime/simd/argmax.rs) use packed, cache-panelled compact projection and stable argmax, followed by blank/repeat decoding. | Production already avoids materializing the full vocabulary probability tensor. Projection is 3.5–7.2% of backbone-plus-projection work in the dense cases; CTC itself is negligible. The full dense diagnostic graph is not a valid production latency proxy. |
| Output / concurrency | [`ocr.rs`](../src/ocr.rs) writes reusable result slabs; [`engine.rs`](../src/engine.rs) can lease warmed requests under a shared CPU cap and idle-memory limit. | Storage timers are tiny; owned conversion is included only in ordinary latency measurements. Concurrent-request queueing, token boxes and memory eviction were not benchmarked. |

The serial fused recognition-prefix diagnostic isolates the kernel opportunity without executing the dense vocabulary head. It uses synthetic tensors and one logical CPU, so its absolute times are not comparable to eight-thread page timings. Each shape ran once for warm-up and once with physical-step timers; output bits matched. At Small `[1,3,48,128]`, 1×1 group-one convolutions consumed **7.543 / 9.508 ms (79.3%)**. At Medium `[1,3,48,640]`, they consumed **193.415 / 213.246 ms (90.7%)**. Large examples include `384→768`, `768→384`, `512→1024`, and `768→1536` channel transforms. Layout-conversion steps cost only 0.0417 / 0.0158 ms in these two probes.

Pointwise kernels already use packed six-row / sixteen-column tiles, fused stores, and a larger 96-row weight-reuse group for channel-heavy AVX layers. The next kernel experiment should target those actual shapes: compare a bounded alternative panel/group size or reduction blocking, inspect memory traffic/register pressure, and preserve reduction order or the existing numerical contract. Adding SIMD, packing, fusion, or NHWC generally would duplicate existing optimizations. These measurements locate the target; they do not establish an unimplemented kernel speedup.

The tested scheduling change is narrower and immediately actionable. With `CpuContext::new(8)`, setting `CpuOptions.line_workers = 8` changes recognition from four owners with two nested kernel threads to eight owners with one thread. DET and preprocessing remain at eight; classification already had eight available workers. A stage-only diagnostic changed recognition wall time from 47.985 to 32.505 ms, while classification stayed near 25 ms. A fresh, separately built [production scheduling probe](../examples/cpu_scheduling_probe.rs) then compared both configurations without profiling:

| Setting | Three measured owned-OCR samples, ms | Median, ms | Retained activation MiB | Retained workspace MiB |
|---|---|---:|---:|---:|
| Automatic, four owners × two kernel threads | 84.9660, 84.4457, 86.3432 | 84.9660 | 10.7363 | 6.6541 |
| Eight owners × one kernel thread | 69.6232, 69.0578, 69.3260 | 69.3260 | 12.4238 | 9.1078 |

The observed ranges do not overlap. Every complete owned result, including scores, coordinates and metadata, equaled the automatic warm-up output exactly. The extra retained capacity is **4.14 MiB per request**, excluding shared model weights. This is capacity accounting, not a separately measured process-memory comparison. There was one warm-up and three calls per configuration, automatic first, in one process sharing immutable models and the CPU pool. Order effects and other workloads remain untested. The earlier 84.009 ms Rust/C# benchmark median is not pooled into this A/B result. Keep the current defaults until sparse/mixed inputs, larger models and concurrent requests justify broader adoption.

For an application dominated by this tested dense Tiny workload, the explicit configuration is:

```rust
.cpu_context(Arc::new(CpuContext::new(8)?))
.cpu_options(CpuOptions {
    line_workers: 8,
    preprocess_workers: 8,
    detector_threads: 8,
})
```

Priorities follow the measured critical paths:

1. Use the verified eight-owner setting for matching dense Tiny workloads, accepting the measured retained-memory increase. Broader automatic dispatch should remain a separate bounded experiment.
2. Optimize the REC pointwise kernels for dense Small/Medium. As an illustrative upper-bound exercise, halving the *whole recognition wave* would reduce the profiled Small/Medium dense totals by about 38.9%/45.1%; that is not a prediction for one kernel change.
3. Optimize DET compute for sparse pages. Halving the *whole DET inference stage* in Tiny sparse would reduce total time by about 36.8%. Lowering detector resolution could also reduce work, but trades against small-text detection and was not tested.
4. Consider skipping orientation only when the input contract guarantees upright text. Its measured Tiny/dense share is 29.7%; removing that work is a theoretical ceiling, not a validated end-to-end result, because rotations can also affect recognition.
5. Treat projection, crop allocations, DB, shape cache, CTC and result copies as secondary targets. Exact-width batching is available, but metadata/CTC costs are already tiny; any benefit must come from kernel efficiency and needs a bounded memory/latency comparison.

Validation used the physical host only. Four focused profiling-contract tests passed before benchmarks. Three live Tiny Rust/C# cases passed existing correctness/timing/memory gates. Five primary profiles and one scheduling profile matched their warm-up outputs; the two backbone probes matched output bits; the separate scheduling A/B matched complete owned results. No full corpus, 36-cell qualification, rotated-image accuracy, throughput matrix, or new kernel candidate was run. Ordinary Rust measurements and diagnostics used approximately **8.35 s** of total worker process execution; C# used **3.04 s**, excluding builds. Original-harness Rust execution was 7.16 s; the new scheduling harness was rebuilt and measured separately for 1.19 s. No automatic sampling growth or unchanged-code retries occurred. The existing Windows `UNBOUND_AFFINITY` unused-import warning remains unrelated to this analysis.

Reproduction commands (keep other inference/build activity idle; the original run used external 10-second deadlines for individual diagnostics):

```powershell
cargo test --release --locked --features profiling --lib profile::tests -- --test-threads=1
cargo build --release --locked --example cpu_benchmark
cargo build --release --locked --features profiling --example cpu_profile
./target/release/examples/cpu_benchmark.exe --model tiny --threads 8 --fixture imgs_1,imgs_00018069,imgs_00006737 --workload latency --runs 3 --budget-seconds 30 --case-timeout-seconds 10
# One call per MODEL/FIXTURE: Tiny sparse/dense/mixed; Small dense; Medium dense.
./target/release/examples/cpu_profile.exe tiny imgs_00018069 8
./target/release/examples/cpu_profile.exe tiny imgs_00018069 8 8
cargo test --release --locked --features profiling --lib runtime::optimized::tests::performance::recognition_backbone_physical_steps -- --exact --ignored --nocapture --test-threads=1
cargo build --release --locked --example cpu_scheduling_probe
./target/release/examples/cpu_scheduling_probe.exe tiny imgs_00018069
```

## Optimization 3: detector upsampling on Windows AVX2/FMA

Implemented on 2026-10-17 against `b446afd`. The chosen change removes temporary
tile writes and scalar scattering from the detector's packed 2×2, stride-two
transposed convolution. Four source pixels share each coefficient walk and write
directly to their exclusive output positions, applying the fused store transform
as vectors. The implementation reuses the original GEMM reduction, preserving
bias-first, ascending-input-channel FMA arithmetic. Models, detector resolution,
thresholds, orientation and recognition settings are unchanged.

The [kernel](../src/runtime/nhwc/transpose.rs) is selected only for Windows x64
AVX2/FMA packed multi-channel transpose convolutions. Other policies retain their
existing paths. Partial channel vectors use bounded scalar stores; source-pixel
tails and worker bands remain disjoint. Existing workspace capacity is retained,
so this change reduces traffic without claiming a retained-memory reduction.

The decision follows a complete fused DET step profile on the actual `imgs_1`
pixels, using `[1,3,960,608]`, eight logical CPUs and affinity `ff`. The old Tiny
16→16 transpose was its largest individual step (3.516 ms). The corresponding
Small and Medium layers were also expensive. Across the complete old detector,
1×1 convolutions totaled 8.861 / 28.554 / 141.150 ms for Tiny / Small / Medium;
optimizing this upsampling layer does not remove those remaining costs.

| Profiled layer, ms | Baseline | Direct output | Reduction |
|---|---:|---:|---:|
| Tiny 16→16 transpose, node 236 | 3.5162 | 1.0822 | 69.2% |
| Small 24→24 transpose, node 236 | 5.5656 | 2.1357 | 61.6% |
| Medium 64→64 transpose, node 222 | 15.5709 | 6.1619 | 60.4% |

These are single measured, instrumented passes after one warm-up per model and
version. Total physical-step time changed from 33.1062→29.6661 ms for Tiny,
79.2005→76.7515 ms for Small, and 294.9548→275.0140 ms for Medium. Step timings
include worker waits and timer overhead, but exclude some session orchestration.
They must not be pooled with the following ordinary release measurements.

The uninstrumented comparison used frozen baseline and rebuilt candidate
executables, matching input/model hashes, automatic layout/fusion, and the same
eight-logical-CPU configuration. Raw DET used deterministic F32 input with one
warm-up and three measured calls. Complete owned OCR used the same predecoded
BGR24 sparse page, one separately recorded cold inference, one warm-up, and three
measured calls. Version order was baseline/candidate for Tiny, candidate/baseline
for Small, and baseline/candidate for Medium. Builds were idle during timing.

| Model | DET graph median, baseline→candidate ms | Complete OCR median, baseline→candidate ms | OCR latency reduction |
|---|---:|---:|---:|
| Tiny | 36.9221→32.6516 | 49.9202→43.8308 | **12.2%** |
| Small | 74.5906→77.5239 | 104.0515→105.1410 | −1.1%; inconclusive |
| Medium | 286.4283→282.1205 | 400.6078→380.9916 | **4.9%** |

| Model | Complete OCR baseline samples, ms | Candidate samples, ms |
|---|---|---|
| Tiny | 57.3221, 49.9202, 49.2653 | 43.8308, 44.1668, 43.0103 |
| Small | 106.3083, 104.0515, 103.1422 | 105.1410, 121.6993, 99.5011 |
| Medium | 396.4054, 400.6078, 402.6811 | 380.9916, 387.4477, 380.7825 |

Tiny and Medium complete-OCR sample ranges are separated. Small's graph and OCR
ranges overlap; their medians were 3.9% and 1.1% slower, respectively. Therefore
the faster Small transpose does **not** establish a full-page improvement.
Medium's raw-graph ranges also overlap. All samples, including outliers, are
retained; unchanged code was not rerun for cleaner numbers.

Every cross-version raw DET output SHA-256 matched, as did every complete owned
OCR result, including coordinates, scores and metadata. Retained activation and
workspace capacities were identical. Validation on this physical host passed
14 baseline kernel checks, 44 candidate NHWC checks (four timing diagnostics
remained ignored), and the separate fused-store integration check. The new
bitwise comparison covers five geometries, bias present/absent, channel and pixel
tails, two batches, four row-band sizes, fused residual/ReLU stores and guard
sentinels. All six step profiles matched their own warm-up output bits.

The [durable evidence](reports/windows-x64-2026-10-17-detector-optimization.json)
contains source/executable/model hashes, all physical steps, raw graph samples,
complete OCR outputs, memory diagnostics and execution accounting. Frozen
executables and raw logs are in `target/detector-optimization-20261017`.
Total performance process execution was **7.69 seconds baseline / 7.06 seconds
candidate**, excluding builds and correctness checks. Graph/OCR processes had
15-second deadlines, the three-model profile had a 20-second deadline, and the
comparison harness had a 40-second per-version aggregate limit within the
60-second overall budget. No remote runner or VM was used.

This is a measured improvement for the tested Windows AVX2/FMA workloads, not a
global optimum or full corpus/throughput qualification. ARM64 and AVX-512 were
not measured. The existing Windows `UNBOUND_AFFINITY` example warning is unrelated.

Reproduction uses the same commands for each frozen source version; preserve
executables before rebuilding the other version and compare hashes and complete
outputs before interpreting timing:

```powershell
cargo test --release --locked --lib runtime::nhwc -- --test-threads=1
cargo test --release --locked --lib runtime::optimized::tests::convolution_store_programs_preserve_order_tails_and_worker_bands -- --exact --test-threads=1
cargo build --release --locked --example cpu_adoption_probe --example cpu_benchmark
# Once per version: all three detectors, one warm-up and one timed pass each.
cargo test --release --locked --lib runtime::optimized::tests::performance::sparse_detector_production_steps -- --exact --ignored --nocapture --test-threads=1
# Once per model/version: use small_det.onnx or medium_det.onnx for other presets.
./target/release/examples/cpu_adoption_probe.exe --model ../SimdPaddleOCR/models/det.onnx --shape 1,3,960,608 --threads 8 --budget-seconds 15
# Existing worker job specifies the predecoded pixels, model, settings and runs.
./target/release/examples/cpu_benchmark.exe _worker target/detector-optimization-20261017/tiny-sparse.job.json
```
