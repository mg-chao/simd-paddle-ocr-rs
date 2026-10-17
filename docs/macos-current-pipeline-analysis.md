# Apple M4 current OCR pipeline latency analysis

The current implementation has room to improve recognition speed, with different targets for different pages. Sparse inputs spend **71.7–74.0%** of profiled elapsed time in detector inference. Dense inputs spend **57.6%, 78.3% and 89.6%** in recognition for Tiny, Small and Medium. Large-channel pointwise convolution is the strongest measured kernel target for Small and Medium recognition; Tiny dense also pays a substantial orientation-classification cost.

Measurements use commit `7ba96f9dc1bef61a7ad4a15c09f4f6278904f703` on 17 October 2026, a physical Apple M4 with four performance and six efficiency cores, 16 GiB RAM and macOS 27.0.1. All runs are native release builds with the pinned nightly toolchain, default parallel features, automatic NEON/FMA and layout selection, and fusion enabled. One request uses an eight-worker CPU context with unpinned macOS scheduling, automatic line ownership, detector limit 960, adaptive recognition width, batch size one, classification enabled and alignment disabled. Production source was clean before measurement and stayed unchanged.

The [portable evidence](reports/macos-arm64-2026-10-17-current-pipeline-analysis.json) contains raw ordinary samples and OCR outputs, complete stage profiles, fused-operator measurements, commands, source/executable/model/input provenance and the local reproduction driver. Existing Windows or earlier ARM timings are not pooled into these results.

The three inputs are `imgs_1` at 720×1150 pixels, `imgs_00018069` at 430×267, and `imgs_00006737` at 896×528. Detector tensor sizes are 608×960, 416×256 and 896×544 respectively. Tiny finds 3/69/31 lines, Small 2/69/33 and Medium 2/69/31. The model-specific outputs differ, so model costs do not imply equivalent accuracy.

Ordinary latency below includes complete owned-result conversion from predecoded BGR24 pixels. Each fresh process runs one separately reported first inference, one warm-up and exactly three measured inferences. Loading includes model import, compilation, CPU context and request creation; first inference is separate. All times are milliseconds; resident memory is the process high-water mark, not just activations.

| Model / input | Lines | Median ms | Sample range ms | Load ms | First inference ms | Peak resident MiB |
| --- | --- | --- | --- | --- | --- | --- |
| tiny / sparse | 3 | 49.531 | 48.441–49.569 | 14.820 | 57.634 | 68.66 |
| tiny / dense | 69 | 67.313 | 66.147–67.636 | 6.386 | 67.195 | 45.02 |
| tiny / mixed | 31 | 81.254 | 80.221–83.321 | 6.411 | 92.218 | 76.89 |
| small / sparse | 2 | 124.013 | 123.980–125.569 | 20.885 | 132.697 | 139.16 |
| small / dense | 69 | 200.779 | 192.989–203.464 | 20.679 | 214.503 | 115.16 |
| small / mixed | 33 | 272.163 | 271.829–272.711 | 20.232 | 282.474 | 171.73 |
| medium / sparse | 2 | 376.384 | 372.242–378.183 | 88.064 | 393.851 | 448.78 |
| medium / dense | 69 | 683.142 | 678.693–707.114 | 93.216 | 685.233 | 351.66 |
| medium / mixed | 31 | 987.715 | 971.389–1227.219 | 87.147 | 1012.155 | 474.59 |

Medium mixed is noisy: its three samples are 1227.219, 987.715 and 971.389 ms, a range equal to 25.9% of its median. Other cases have observed ranges of approximately 0.3–5.2% of their medians. Three samples do not establish statistical significance, and these measurements contain no optimization before/after comparison.

The following durations come from separate production-path instrumented executions, one warm-up and one measured pass per case. The six elapsed phases run in sequence. Recognition includes scheduling and ordered result assembly; classification includes preprocessing, backbone, compact head and any 180-degree rotations. The total returns borrowed results, with owned conversion measured separately. Do not rescale these times to the ordinary median or subtract the two totals to infer copying overhead.

| Model / input | DET prepare | DET infer | DB | Crop | CLS | REC | Profile total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tiny / sparse | 2.826 | 34.256 | 0.448 | 0.347 | 1.506 | 7.982 | 47.377 |
| tiny / dense | 0.531 | 9.701 | 0.316 | 0.671 | 17.881 | 39.483 | 68.601 |
| tiny / mixed | 2.078 | 28.095 | 0.542 | 1.081 | 8.653 | 43.551 | 84.014 |
| small / sparse | 2.941 | 90.645 | 0.413 | 0.405 | 1.397 | 26.731 | 122.544 |
| small / dense | 0.519 | 21.549 | 0.279 | 0.690 | 18.528 | 150.105 | 191.683 |
| small / mixed | 2.199 | 78.897 | 0.529 | 1.146 | 9.205 | 177.802 | 269.797 |
| medium / sparse | 3.007 | 270.353 | 0.536 | 0.543 | 1.640 | 100.795 | 376.891 |
| medium / dense | 0.689 | 54.256 | 0.307 | 0.803 | 18.194 | 639.646 | 713.915 |
| medium / mixed | 2.945 | 257.453 | 0.551 | 1.600 | 9.720 | 793.999 | 1066.298 |

Elapsed orchestration outside the six phases is 0.012–0.030 ms. DB costs only 0.279–0.551 ms and cropping 0.347–1.600 ms across these inputs. Eliminating either stage cannot remove its summed parallel-worker work from page latency.

The dense-page counters below explain work inside the elapsed phases. Parallel-stage counters sum worker elapsed durations, including delays while each call is active; bookkeeping counters combine nested DET and REC measurements. They are neither process CPU time nor additional elapsed phases, and worker sums can exceed total page latency. Each dense page contains 69 lines; all three profiles applied zero 180-degree rotations. All nine raw profiles, including width histograms, are retained in the evidence.

| Nested work and bookkeeping ms | Tiny dense | Small dense | Medium dense |
| --- | --- | --- | --- |
| Crop workers | 3.81720 | 3.90867 | 4.68988 |
| CLS resize and normalization | 8.69108 | 8.02584 | 8.94892 |
| CLS backbone | 118.99917 | 116.24033 | 118.19378 |
| CLS compact head | 0.13017 | 0.12187 | 0.12892 |
| CLS 180 degree rotation | 0.00000 | 0.00000 | 0.00000 |
| REC resize and normalization | 5.31200 | 5.36862 | 6.58871 |
| REC backbone | 225.37288 | 1071.84225 | 4736.09559 |
| Vocabulary projection and argmax | 16.69491 | 65.17683 | 131.41950 |
| CTC decode | 0.02287 | 0.02987 | 0.07004 |
| DET and REC reshape calls | 0.03787 | 0.05016 | 0.10637 |
| Binding and preparation | 0.01142 | 0.02096 | 0.05667 |
| Result storage | 0.01270 | 0.01887 | 0.03212 |

CLS backbone accounts for about 93% of its dense worker orientation time. Vocabulary projection is 6.9%, 5.7% and 2.7% of dense backbone-plus-projection work for Tiny, Small and Medium. CTC and warm reshape/storage costs are small. The separately measured owned conversion spans 0.0016–0.0099 ms across the nine profiles; changing the public result ownership model is therefore a low priority for these pages.

The implementation trace identifies what each link already does and where further work could help.

| Link | Current behavior | Performance implication |
| --- | --- | --- |
| Input and setup | [ImageView](../src/image.rs) borrows validated RGB/BGR/alpha pixels. [Model loading](../src/model.rs), [builder](../src/builder.rs) and [engine](../src/engine.rs) separate shared weights from reusable requests. The CLI decodes files before OCR. | Keep models and requests warm in a service. JPEG/PNG decode, I/O and serialization are outside these measurements. Profile setup separates import, CPU-context creation and request compilation. |
| DET preparation | [Preprocessing](../src/preprocess.rs#L397) computes a limited, 32-aligned detector shape, caches interpolation maps, uses fixed-point byte interpolation and normalization tables, and writes the chosen tensor layout. | The final vertical interpolation/store loop remains a candidate, but measured preparation is only 0.519–3.007 ms. |
| DET inference | [Optimized runtime](../src/runtime/optimized.rs) selects layouts, fuses stores and residual/activation work, aliases eligible values, and reuses activation arenas. Native NEON kernels and shared packed weights are already present. | Sparse pages are dominated by actual convolution work. Optimize the measured shapes instead of adding generic packing, fusion or SIMD that already exists. |
| DB postprocessing | [DB](../src/db.rs#L105) thresholds and scans the bitmap, finds components, constructs hulls and minimum-area boxes, scores/unclips them and sorts reading order. Scratch and SIMD scan helpers are retained. | Serial hull rectangle search is quadratic in hull size and polygon scoring scans pixels. Better geometry algorithms may matter for complex contours, but this sample stage stays below 0.56 ms. |
| Perspective crop | [Line extraction](../src/lines.rs#L689) fills a retained BGR slab through 16-row tasks. [Crop sampling](../src/crop.rs#L196) uses a double-precision projective mapping and 4×4 cubic interpolation per output pixel; vertical rotation is folded into the crop. | Multi-pixel SIMD or exact axis-aligned fast paths are plausible. Direct crop-to-network resampling changes the current cubic-byte-then-bilinear contract and needs accuracy validation. |
| Orientation | [Classifier](../src/orientation.rs#L202) resizes each crop to 80×160, executes a fixed prefix and compact two-class head with one kernel thread per owner, then rotates when selected. The CLS phase finishes before REC. | Tiny dense spends 17.881 ms here. Batch CLS or target its backbone if arbitrary orientation must be supported. Skipping CLS requires an upright-input contract; all sparse lines in these profiles were rotated. |
| REC width and ownership | [Recognizer](../src/recognizer.rs#L1111) uses height 48 and adaptive widths rounded to 32. [OCR scheduling](../src/ocr.rs#L829) sorts widths and assigns width-weighted units to deterministic owners. [CPU policy](../src/cpu.rs#L139) already uses eight NEON owners for at least eight units. | Mixed pages contain widths up to 1536 for Tiny and 1472 for Small/Medium. Measure per-owner completion before altering scheduling; fixed width 320 can compress long text. Exact-width batching is available but changes owner count and memory. |
| REC execution | [Sessions](../src/runtime.rs#L1236) retain eight previous shape plans and activation high-water storage. [Pointwise NEON](../src/runtime/nhwc/pointwise/neon.rs#L24) uses six rows by sixteen channels with ordered FMA and packed weights. | REC backbone is the dominant dense Small/Medium cost. Useful experiments should target large channel products, tile/store overhead and weight reuse while retaining fused epilogues and numerical behavior. |
| Projection and CTC | [Compact projection](../src/projection.rs) computes winning vocabulary classes without materializing a full logits/probabilities tensor. [CTC](../src/recognizer.rs#L615) collapses winners into reusable text/span storage. | The compact head is already optimized structurally. Projection is secondary to the backbone; CTC is too small here to be the first target. |
| Results and concurrency | [OCR](../src/ocr.rs#L997) copies worker outputs into stable ordered slabs. [Engine](../src/engine.rs) shares models, leases requests and bounds idle retention through a shared CPU context. | Borrowed results already avoid owned copies. Tight retention can evict warm requests. Admission waits and simultaneous-request throughput require a separate workload. |

Fused-operator timings further narrow the compute targets. The detector diagnostic runs the real sparse image at `[1,3,960,608]` with eight threads, one warm-up and one measured pass per model. The recognition diagnostic runs synthetic feature prefixes serially at the two stated shapes, excluding the vocabulary head. All five comparisons preserved output bits. The tables describe those diagnostic executions and cannot be multiplied directly into page-stage times.

| Fused production probe | All steps ms | 1×1 convolution ms | 1×1 share | Layout conversion ms |
| --- | --- | --- | --- | --- |
| tiny DET | 38.806 | 13.738 | 35.4% | 0.5286 |
| small DET | 103.368 | 36.065 | 34.9% | 1.2769 |
| medium DET | 296.464 | 164.963 | 55.6% | 0.0000 |
| small REC width 128 | 11.151 | 8.682 | 77.9% | 0.0272 |
| medium REC width 640 | 215.004 | 200.124 | 93.1% | 0.0083 |

Small REC repeatedly spends roughly 0.53–0.64 ms per 384→768 / 768→384 pointwise layer in its width-128 probe. Medium width-640 includes approximately 9.8–10.5 ms 512→1024 and 768→1536 pointwise layers. Attention-related matrix multiplication, softmax, layer normalization and transpose together account for only about 1.1% of Small and 0.8% of Medium prefix steps. Layout conversions are also small. These measurements favor pointwise work over attention or layout-copy changes.

Sparse DET has another concrete target: [NHWC dense eligibility](../src/runtime/optimized.rs#L210) requires output channels divisible by sixteen and at least sixteen. Small's 96→24 3×3, 48→24 stride-2 3×3, and 24→12 2×2 layers therefore use canonical paths; they cost 18.726, 12.862 and 9.325 ms in the operator probe. Tiny's 16→8 2×2 layer costs 6.083 ms. A tail-capable NHWC path or specialized canonical kernel could address these cases. The expensive part is their convolution computation; removing the separately measured conversion time alone would save little. Medium DET also spends 55.6% of operator time in pointwise work. Depthwise totals are only 6.6%, 5.3% and 7.2% for the three detector probes, and the large ARM64 depthwise strip path is already present.

The optimization order supported by these measurements is:

1. **Dense Small/Medium: large-channel REC pointwise kernels.** Start with one measured layer shape, such as Small 384→768 over 96 spatial rows or Medium 512→1024 at its recorded shape. Compare one mechanism at a time, including tile/store overhead or weight reuse. Existing ARM records contain inconclusive tile/cache-group experiments, so changing a group size alone is not an established improvement. Require a useful focused result before testing a complete page, then check full OCR parity and retained memory.
2. **Sparse pages: detector convolution paths.** Target the channel-tail layers above and Medium pointwise work. NEON support for those tails or an improved canonical implementation is a stronger candidate than general layout cleanup. A lower detector limit is an application accuracy tradeoff.
3. **Tiny dense: orientation backbone or conditional CLS omission.** The measured CLS wave is 26.1% of this profile. Halving the whole wave would reduce total latency by about 13.0% if all other phases stayed unchanged. Removing CLS could save at most its present 17.881 ms wave under the same assumption, but this analysis did not measure the toggle, and zero rotations in one image does not establish an upright-input guarantee.
4. **Mixed widths: measure owner completion and exact-width batching.** Width balancing and eight owners already exist. Long lines can still create a tail. Existing batch sizes two to four may improve weight reuse on repeated widths but reduce independent work and increase activations; do not change the default without a paired latency/memory/result comparison.
5. **Treat crop, DB, warm binding, CTC and result copies as secondary.** They have small absolute ceilings on these inputs. The M4 core mix also makes four/eight/ten-worker DET a plausible separate experiment; equal Rayon bands do not prove efficient placement across performance and efficiency cores. No alternative worker count was measured here.

For scale, halving the entire REC wave, with every other phase unchanged, would reduce profiled dense Small latency by 39.2% and Medium by 44.8%. Halving Tiny sparse detector inference would reduce its total by 36.2%. These are critical-path ceilings for whole phases, not predictions for a specific kernel change. Kernel speedups can be diluted by other operators, scheduling and memory contention.

Local validation passed five focused profiling semantic tests, nine saved-golden comparisons with the existing 0.5-pixel and `1e-4 + 1e-4×abs(expected)` score tolerances, exact complete warm-up/profile output equality in nine cases, and exact float-bit operator equality in five probes. Each ordinary worker also checked complete output equality across its first, warm-up and three measured calls. Model assets and predecoded inputs passed their pinned hashes. No golden, tolerance, dependency or production-code changes were made.

Performance execution totaled **28.062 seconds**, including process startup and model loading and excluding builds/correctness tests, under a shared 60-second limit. Counts were nine ordinary first inferences, nine ordinary warm-ups and 27 ordinary measured inferences; nine profile warm-ups and nine profiles; and five operator warm-ups and five instrumented passes. That is 73 inference/prefix executions overall. Every performance subprocess had a 10-second deadline, or 12 seconds for each grouped operator diagnostic, further capped by the remaining aggregate budget. There were no retries, automatic sampling growth or unchanged-code reruns.

These measurements cover one physical machine, one request at a time, three images and three model bundles. File acquisition/decode, token-box/alignment output, JSON serialization, admission queueing, simultaneous-request throughput and other architectures remain outside this latency scope. Stage profiles have one sample per case, so small differences should not drive changes. Source-level hypotheses about cache misses, register spills or heterogeneous-core placement need their own measurements.

Build and focused correctness commands were:

```sh
cargo build --release --locked --example cpu_benchmark
cargo test --release --locked --features profiling --lib profile::tests
cargo build --release --locked --features profiling --example cpu_profile
```

The preserved executables, nine hash-checked worker jobs, raw results, diagnostic logs and driver are in `target/pipeline-analysis-20261017-current`. For any new measurement, rebuild and use a fresh output directory, keep other inference idle, and preserve fixed counts and deadlines. The portable JSON records every invoked worker/test command and the exact local test-binary path.
