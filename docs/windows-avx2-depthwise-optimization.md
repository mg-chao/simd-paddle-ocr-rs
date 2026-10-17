# Windows AVX2 short-depthwise optimization

The retained change removes repeated input-bound checks from complete eight-pixel tiles of packed NHWC depthwise convolutions. On this physical Windows machine, the two focused kernels take **58–60% less time**, and complete Tiny OCR takes **4.7% less time on the dense page and 5.8% less time on the mixed page** in the final matched comparison. Sparse Tiny, Small and Medium remain inconclusive. This is the best validated change among the candidates investigated here; it does not resolve the large pointwise-convolution bottleneck in Small and Medium recognition.

The [raw evidence](reports/windows-x64-2026-10-17-depthwise-optimization.json) preserves samples, output hashes, source manifests and patches, executable hashes, rejected experiments, runner sources and validation logs. The preceding [CPU24 stage analysis](windows-cpu24-stage-analysis.md) explains the target selection and contains separate, pre-optimization timings.

## Implementation

Only the x86 AVX2/FMA depthwise wrapper enables the new path. The shared traversal validates storage and overflow-safe geometry once per output band, then proves that an eight-pixel tile has a complete receptive field before using unaligned SIMD loads without repeated tap/pixel bounds checks. Eligibility depends on geometry and storage, with kernels up to 5×5 and at least eight channels. Maps without a complete tile use the existing traversal.

Each channel vector retains eight independent pixel sums, initialized with bias and updated in the existing kernel-row/kernel-column order using FMA. The final store retains the global output index and channel index for fused channel operations and residuals. Borders, short tiles and scalar channel tails use the existing checked arithmetic. Kernels of height six or greater retain the existing strip path. No additional activation buffers, weight packs or scratch allocations are needed.

The production change is in [the depthwise dispatch](../src/runtime/nhwc.rs) and [the interior kernel](../src/runtime/nhwc/short_depthwise.rs). Classification, thresholds, model data, recognition batching and scheduling defaults retain their previous settings. The developer [owner probe](../examples/cpu_owner_probe.rs) now accepts an optional `BATCH_SIZES` argument so the rejected batching alternative can be reproduced.

## Final matched page comparison

The baseline is revision `38e054c` plus the preceding diagnostic work and batch-selector helper. The candidate adds the final shared-traversal depthwise implementation. Both frozen executables use ordinary release builds with default parallel features, without profiling or custom Rust flags. Tests ran locally on the AMD Ryzen 9 5950X host, with 12 enabled cores / 24 logical CPUs reported by Windows, affinity `ffffff`, one 24-thread context and automatic line ownership. The pinned toolchain is `nightly-2026-09-08`.

All cases use the manifest's models/settings, classification enabled, batch size one, adaptive recognition widths, detector limit 960 and identical predecoded BGR24 bytes. Each binary runs once per case in its own process, with one warm-up and exactly three measured complete owned-result calls. Old/new process order alternates across cases. There is no separately executed first inference. Decoding, model import, request setup and result serialization are outside the measured OCR call.

| Case | Baseline samples, ms | Candidate samples, ms | Median before → after, ms | Median change |
|---|---|---|---:|---:|
| Tiny sparse / `imgs_1` | 46.1048, 46.1690, 45.3974 | 46.1541, 45.3123, 43.8662 | 46.1048 → 45.3123 | 1.72% lower; overlapping ranges |
| Tiny dense / `imgs_00018069` | 46.1866, 46.0815, 47.0068 | 44.0130, 43.3548, 44.7946 | 46.1866 → 44.0130 | **4.71% lower** |
| Tiny mixed / `imgs_00006737` | 70.6271, 71.0936, 70.3764 | 66.5621, 69.9760, 66.0410 | 70.6271 → 66.5621 | **5.76% lower** |
| Small dense | 115.5311, 119.3501, 121.0835 | 115.4282, 124.4413, 121.1692 | 119.3501 → 121.1692 | 1.52% higher; overlapping ranges |
| Medium dense | 428.8271, 426.2493, 402.6589 | 426.5911, 424.1587, 389.0927 | 426.2493 → 424.1587 | 0.49% lower; overlapping ranges |

Tiny dense and mixed have disjoint observed ranges in this bounded comparison. Sparse, Small and Medium do not establish an improvement or regression; Medium also shows downward sample drift. Three samples do not establish confidence intervals or steady state. These results are complete OCR latency, not separately measured recognition-stage speedups. This final experiment does not reuse or pool the preceding stage-analysis, selection or rejected-candidate samples.

All five cross-version comparisons have identical SHA-256 digests of the complete serialized `OwnedOcr` results. Every warm-up and measured result within each process is also exactly equal. Retained activation bytes, workspace bytes, kernel scratch and post-run CPU budgets match across versions. Process private-commit peaks are preserved in the raw evidence and include setup; they are not isolated kernel memory measurements.

## Focused kernel result

The paired diagnostic uses the same shared traversal with the interior path disabled/enabled, one logical CPU, cached synthetic inputs and packed weights, and an identity final store. These are two actual Tiny detector shapes. Each variant receives one warm-up and three single-call samples, in AB/BA/AB order; there are no hidden sampling loops. Outputs match bit for bit.

| Shape, NHWC | Checked samples, ms | Interior samples, ms | Median before → after, ms | Reduction |
|---|---|---|---:|---:|
| H240 × W152 × C64, 5×5, stride 1, pad 2 | 7.4410, 7.5052, 7.4771 | 3.2411, 3.0325, 3.1200 | 7.4771 → 3.1200 | **58.27%** |
| H240 × W152 × C32, 3×3, stride 1, pad 1 | 1.4636, 1.4879, 1.4518 | 0.5813, 0.5701, 0.5893 | 1.4636 → 0.5813 | **60.28%** |

The single-thread kernel reduction cannot be applied directly to the parallel page timings or interpreted as a reduction in all neural inference work.

## Alternatives and rejected first candidate

Exact-width batch size two preserved complete outputs but did not establish a consistent page benefit on the dense input. Batch-one/batch-two medians were 45.6531/46.9442 ms for Tiny, 122.8272/123.5138 ms for Small and 411.5945/404.6698 ms for Medium. Small and Medium ranges overlap, while Tiny became slower. The default remains one.

A real-model allocation census inspected cached large 1×1 recognition weight matrices without running inference. One of five Small panels had split 32-byte loads, representing 11.11% of their weight storage; none of Medium's nineteen panels did. This is one allocation sample per model, not a traffic-weighted measurement or proof that allocations are always aligned. It did not justify changing all packed storage. Earlier tile/blocking/store alternatives remain recorded in [the large-channel optimization investigation](windows-large-channel-optimization.md).

The first depthwise candidate added a separate whole-layer traversal. Its page comparison showed Medium regressions of 5.83% at batch one and 9.77% at batch two, despite faster focused depthwise kernels, so that candidate was rejected. A bounded serial recognition diagnosis also found faster depthwise groups but slower unchanged pointwise work. Static inspection of both ordinary and profiling executables found identical full-K pointwise leaf bytes and identical normalized prepared-store pointwise instructions. Function placement differed; no causal explanation for the page regression was established.

The retained version shares the existing traversal and rejects eligibility when a complete interior tile cannot exist. The final comparison uses one configuration per process and places old/new measurements close together to address the unresolved Medium result. Because both integration and experimental setup changed, the difference between the first and final experiments cannot be attributed solely to the source refinement. The rejected first candidate retains source hashes, frozen local executables and measurements; its complete source snapshot was not archived. Its raw evidence is kept separately from the final measurements.

## Correctness and execution limits

Focused correctness ran before performance tests. The final three focused tests cover bit equality against the checked traversal across five geometries, bias/no-bias, fused channel addition/ReLU/residuals, batches, horizontal and vertical strides, dilation, rectangular kernels, non-vector channel counts, misaligned slices, guarded output bands and malformed storage. An independent brute-force geometry check validates eligibility.

The final release library test suite with profiling enabled passed **332 tests**, with 14 opt-in diagnostics ignored. The ordinary release example built successfully, and `cargo check --release --locked --no-default-features --lib` passed. Formatting checks on changed Rust files and `git diff --check` passed. A recursive formatting check also encountered a pre-existing formatting difference in the unchanged `src/runtime/nhwc/pointwise/large_channel_tests.rs`; the focused check uses `skip_children=true` and includes the new files explicitly.

Conservative process wall-time accounting, including setup, stayed below the 60-second performance limit per code version: **13.6961 s baseline**, **9.6184 s rejected first candidate**, **5.6992 s retained candidate**. Builds and correctness checks are excluded. The baseline includes batching selection, the first candidate's two extra control pages, one serial diagnosis and the final comparison; the rejected candidate includes kernels, the zero-inference allocation census, pages and serial diagnosis. The final candidate includes 16 paired kernel calls and 20 complete OCR calls across five cases. Each final comparison process had a 10-second timeout, with a 35-second allocation per variant. No automatic sample growth or retries were used.

This is focused Windows AVX2 validation. No C# qualification matrix, concurrent throughput test or ARM/macOS execution was performed. Small and Medium recognition remain dominated by large group-one pointwise convolutions; a further optimization needs a new mechanism with a demonstrated page-level benefit.
