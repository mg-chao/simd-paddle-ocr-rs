# Windows AVX2 parallel PadEnd2 optimization — 2026-10-18

Follow-up: the [paired Medium investigation](windows-medium-regression-investigation.md)
did not reproduce the 10.6% Medium slowdown below: fresh baseline/current medians
were 434.7797 / 435.0426 ms (+0.06%), while Tiny stayed faster in every matched pair.
The original samples below remain a separate experiment.

The adopted change connects the existing native eight-output 2×2 pad-end convolution to parallel row shards. Previously, the serial dispatcher used this optimized reduction while the parallel dispatcher repeated generic geometry and weight indexing inside the input-channel loop. Both paths now share the native reduction and destination-band traversal.

The change is confined to eligible x86 eight-lane, eight-output, stride-one pad-end geometry. It keeps reduction order, bias seeding, non-FMA borders/tails, global fused-store indices and exclusive NCHW output ownership. It adds no packing, scratch allocation, layout conversion, public API or OCR quality-setting change. AVX-512 retains its existing sixteen-lane path.

The [preceding pipeline analysis](windows-2026-10-18-pipeline-analysis.md) identified this Tiny detector stem as a concrete target. The [portable evidence](reports/windows-x64-2026-10-18-pad-end-optimization.json) contains every sample, output digest, source/executable hash, operator record, rejected worker experiment, runner and validation log.

**Focused comparison.** Release builds ran locally on the physical Windows AMD Ryzen 9 5950X machine (12 enabled physical cores, 24 logical processors). Each version used the same single-node ONNX model, deterministic input, AVX2/FMA, eight threads, affinity `ff`, one warm-up and three measured calls per shape. The model represents the 16→8 Tiny detector stem. Times are complete session calls, not isolated arithmetic or complete OCR.

| Input shape | Baseline samples, ms | Candidate samples, ms | Median before → after, ms | Reduction |
| --- | --- | --- | --- | --- |
| 1×16×480×304 | 4.1405, 4.7331, 4.0559 | 2.1325, 2.3599, 2.2371 | 4.1405 → 2.2371 | 46.0% |
| 1×16×128×208 | 0.9116, 0.8763, 0.8124 | 0.4318, 0.3988, 0.5177 | 0.8763 → 0.4318 | 50.7% |

All eight output digests per shape (both warm-ups and six measured calls) match exactly across versions. The smaller shape represents the dense fixture; the larger shape represents the sparse fixture.

**Production graph check.** A separate instrumented probe executes the actual sparse detector graph at eight threads, once after one warm-up for each model. The targeted Tiny node changes from **2.3841 to 1.1663 ms**. This is a single instrumented sample per version, corroborating the affected operation rather than establishing a distribution.

| Detector | Baseline physical-step sum, ms | Candidate physical-step sum, ms |
| --- | --- | --- |
| tiny | 32.5207 | 32.9179 |
| small | 83.1934 | 82.6753 |
| medium | 305.7673 | 280.5127 |

Medium's two detector stem convolutions use NHWC and do not select the changed canonical PadEnd2 writer. Small's canonical 2×2 layers can use the refactored serial writer, so Small is not an unaffected control. These single-sample aggregate timings do not establish a causal effect of the patch; host/build variation remains a confounder. Each graph asserts output-bit equality against its own warm-up; cross-version complete OCR equality is checked below.

**Complete OCR check.** The original three predecoded BGR24 images and five model/page cases retain CPU24 affinity `ffffff`, automatic owners, batch one, detector limit 960, adaptive recognition widths and enabled orientation classification. Each case uses one warm-up and three measured owned-result calls. The same-day baseline is reused without rerunning unchanged code. These non-interleaved samples are useful regression evidence but cannot prove small whole-page gains.

| Model / fixture | Baseline samples, ms | Candidate samples, ms | Median before → after, ms | Observed reduction |
| --- | --- | --- | --- | --- |
| tiny imgs_1 | 51.2193, 55.5863, 50.2808 | 45.9112, 46.5351, 42.9055 | 51.2193 → 45.9112 | 10.4% |
| tiny imgs_00018069 | 44.6641, 44.6643, 44.5703 | 47.0807, 46.4089, 45.3434 | 44.6641 → 46.4089 | -3.9% |
| tiny imgs_00006737 | 69.9881, 69.3532, 68.5409 | 70.2461, 72.2424, 65.0216 | 69.3532 → 70.2461 | -1.3% |
| small imgs_00018069 | 117.2156, 117.2999, 125.9579 | 118.0121, 119.9384, 117.9403 | 117.2999 → 118.0121 | -0.6% |
| medium imgs_00018069 | 440.3400, 454.7224, 434.4181 | 424.1598, 487.1084, 508.0666 | 440.3400 → 487.1084 | -10.6% |

Tiny sparse has a lower median and a non-overlapping observed range. Tiny dense is 3.9% slower, Tiny mixed 1.3% slower, Small dense 0.6% slower and Medium dense 10.6% slower by these medians. Medium ranges from 424.160 to 508.067 ms, spanning the earlier 434.418–454.722 ms range. These outcomes establish neither broad page gains nor absence of regressions. The strong adoption evidence is the repeated affected-convolution result, corroborated by its actual graph node; the sparse-page observation is supportive but its entire 10.4% change cannot be attributed to that one node. No unchanged case was rerun for more favorable numbers.

Every complete owned-result digest matches the original baseline, including text, scores, boxes and ordering. Reported retained activation, workspace and kernel-scratch bytes are unchanged in all five cases.

**Rejected scheduling alternative.** Before changing the kernel, a bounded recognition-only probe compared sixteen and twelve owners, each with one intra-op thread, on identical oriented dense-page crops. It used one warm-up and three measured waves per model/configuration, with AB/BA/AB measurement order. Unlike the earlier whole-page ownership probe, classification and intra-op budget do not change between these configurations. Per-line owned conversion is included; final sorting/copying and full-page orchestration are excluded. Both worker sets remain resident, so this is a selection diagnostic.

| Model | Sixteen-owner samples, ms | Twelve-owner samples, ms | Median sixteen / twelve, ms |
| --- | --- | --- | --- |
| tiny | 18.0158, 18.0657, 16.9340 | 22.8425, 20.6446, 20.8964 | 18.0158 / 20.8964 |
| small | 83.1427, 75.3539, 78.6896 | 93.5884, 84.8170, 82.7280 | 78.6896 / 84.8170 |
| medium | 423.0857, 435.0983, 437.6943 | 420.6133, 440.8485, 503.6668 | 435.0983 / 440.8485 |

Twelve owners regress Tiny and Small and provide no consistent Medium improvement. Complete recognition outputs match across all configurations. The production worker policy therefore remains unchanged. Existing large-channel packing/tile experiments also do not justify a broad recognition-kernel replacement; their tradeoffs are documented in the earlier large-channel reports.

**Correctness and bounds.** Focused release tests ran before performance measurement, followed by the complete non-ignored release library suite. The new independent oracle covers discontiguous row bands at nonzero batch/channel/row offsets, bias and no bias, widths 7/9/17, FMA and separate multiply/add, bottom/right boundaries, exact global store indices, exactly-once writes and unchanged guards. Existing tests cover production-size row sharding, thread budgets 1/2/4/8, explicit backend selection, cancellation-sensitive reductions, multiple batches and output channel tiles.

Validation commands:

```powershell
cargo test --release --locked --features profiling --lib pad_end -- --test-threads=1
# Execute the newly built release test binary with these filters:
pad-v1-tests.exe row_shard_guards --test-threads=1
pad-v1-tests.exe --test-threads=1
cargo check --release --locked --no-default-features --lib
cargo build --release --locked --example cpu_recognition_probe
```

Recorded results: test result: ok. 7 passed; 0 failed; 0 ignored; 0 measured; 352 filtered out; finished in 0.10s; test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 358 filtered out; finished in 0.01s; test result: ok. 345 passed; 0 failed; 14 ignored; 0 measured; 0 filtered out; finished in 4.85s.

Conservative process-wall performance accounting is **17.898 s baseline** (including the preceding pipeline analysis and recognition selection) and **6.861 s candidate**, excluding builds and correctness checks. Candidate execution comprises eight synthetic graph calls, twenty complete OCR calls and six instrumented detector graph calls. Page and synthetic processes have ten-second limits; the operator process has a fifteen-second limit. No automatic sampling growth or retries occurred.

The result supports the affected convolution optimization. Recognition backbone work still dominates dense Small/Medium latency; this patch does not claim to solve that bottleneck. Three samples do not establish p95/p99 or a confidence interval. Validation covers this physical Windows AVX2 host; AVX-512 and ARM64 performance were not measured. No CI, remote runner or VM was used. Input capture, image decoding and external request queueing remain outside the timed OCR calls.
