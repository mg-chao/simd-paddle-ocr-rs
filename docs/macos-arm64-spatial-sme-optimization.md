# ARM64 spatial convolution acceleration with SME

The retained change accelerates eligible 3×3 convolutions by packing small receptive-field blocks for the existing SME matrix kernel. On the physical Apple M4, complete sparse-page OCR median latency improves **3.3% for Tiny, 7.9% for Small and 12.4% for Medium**, with disjoint before/after sample ranges. Dense and mixed-page ranges overlap: their precise effects are uncertain with three samples, including a **2.3% higher Tiny dense median**. The implementation is retained for its consistent sparse-page gains and bounded memory cost, without claiming a universal page speedup.

All nine page result projections match exactly, all three production-profile pairs preserve complete owned outputs, and **213/213 reference qualification requests pass**. Retained activation and weight capacities are unchanged. Reusable workspace increases by **384–1024 KiB per request** across these cases.

[Portable evidence](reports/macos-arm64-2026-10-18-spatial-sme-optimization.json) preserves source and executable hashes, both candidate experiments, raw samples, complete profile counters, memory, validation, qualification and reproduction scripts. The comparison baseline is the initial working tree with the [fused panel-store optimization](macos-arm64-panel-store-optimization.md), whose bottlenecks were recorded in the [preceding pipeline analysis](macos-panel-store-pipeline-analysis.md). It is not clean `ca0996a`. Existing changes were preserved.

**Implementation.** The new [spatial packer](../src/runtime/nhwc/dense/sme.rs) binds each receptive field's channel spans, substitutes positive zero for padding, and uses 4×4 NEON transposes to produce the existing K-major panel layout. It packs at most 64 output pixels at a time, preserving the original input-channel/kernel-row/kernel-column reduction order. The storage is independent of image area; no full-image im2col buffer is materialized.

The [shared SME group executor](../src/runtime/nhwc/pointwise/sme.rs) consumes those panels with the existing 32×32 tiles, 16×16 tails and sixteen-lane fused stores. Pointwise convolution uses the same extracted helper. Assembly, bias-first ascending-K FMA chains, thread-state guards and activation/residual order are unchanged. A declined SME guard or missing scratch executes the existing NEON dense kernel.

The [selector](../src/runtime/nhwc/dense.rs) applies only to native macOS ARM64 with available SME, packed NHWC dense 3×3 convolution, at least 32 input channels, 16 output channels and 32 output pixels. Grouped/depthwise convolution and other shapes keep their existing dispatch. Parallel execution borrows disjoint output bands and private scratch from the same bounded CPU context. [Runtime reservation](../src/runtime/optimized.rs) accounts for `64 × 9 × input_channels` floats per worker, takes the maximum required across selected layers, and reserves it before activation writes. Warm calls and smaller reshapes reuse the allocation. No public API, model, threshold, resolution or new dependency is introduced.

**Candidate selection.** The first candidate limited the crossover to at least 48 input and 32 output channels. One warm-up and three complete graph calls per shape gave:

| Graph / threads | Baseline median ms | Narrow crossover ms | Reduction |
| --- | --- | --- | --- |
| Medium DET 960×608 / 8 | 203.260 | 181.457 | 10.7% |
| Small REC width 128 / 1 | 5.676 | 5.527 | 2.6% |
| Medium REC width 640 / 1 | 62.970 | 60.694 | 3.6% |

The second candidate admitted the narrower 32-input/16-output channel boundary to include the measured Tiny and Small spatial heads. Three detector shapes, each with one warm-up and three calls, gave:

| Detector graph 960×608 / 8 threads | Baseline median ms | Broader crossover ms | Reduction |
| --- | --- | --- | --- |
| Tiny | 27.364 | 25.664 | 6.2% |
| Small | 51.216 | 49.778 | 2.8% |
| Medium | 203.260 | 174.799 | 14.0% |

Medium's unchanged baseline graph samples were reused from the first experiment; they were not rerun or pooled. The two new baseline shapes and candidate shapes used separate processes. Every warm-up and sampled output tensor hash matches. Graph inputs are deterministic synthetic tensors. Recognition graph probes include their full vocabulary head and softmax, so their timings are not production OCR's compact recognition path. These experiments selected the broader crossover; the final page comparison below supplies independent adoption evidence.

After selection, the dense module description and selection/accounting tests were completed. The final source was rebuilt and passed the full validation below before page measurement. No production source changed afterward.

**Matched complete OCR comparison.** Physical Apple M4, 10 cores, 16 GiB RAM, macOS 27.0.1, AC power, native release/default features and pinned `nightly-2026-09-08`. Eight shared workers, one request, automatic SIMD/layout/fusion and line ownership, batch size one, adaptive recognition widths, detector limit 960, classification enabled, alignment off. Model-specific thresholds and hash-verified inputs are unchanged. macOS scheduling is unpinned.

Sparse/dense/mixed are `imgs_1`, `imgs_00018069` and `imgs_00006737`. Each case runs before/after executables in separate sequential processes, alternating version order by case. The worker records its first call separately, performs one explicit warm-up, then measures three owned-result calls on predecoded BGR24 pixels. Timers exclude setup, decoding, serialization and service admission. Builds and correctness checks are idle during timing. Fresh baseline processes provide matched controls for this candidate; earlier page samples are not pooled into these results.

| Bundle / page | Before ms | After ms | Reduction | Before range ms | After range ms |
| --- | --- | --- | --- | --- | --- |
| Tiny / sparse | 36.65 | 35.45 | 3.3% | 36.21–38.95 | 34.98–35.63 |
| Tiny / dense | 51.97 | 53.19 | -2.3% | 51.73–52.23 | 49.96–54.88 |
| Tiny / mixed | 61.19 | 60.67 | 0.8% | 58.89–62.09 | 59.12–61.31 |
| Small / sparse | 72.80 | 67.02 | 7.9% | 72.59–73.93 | 66.33–69.07 |
| Small / dense | 115.47 | 113.17 | 2.0% | 112.92–116.17 | 107.59–116.04 |
| Small / mixed | 148.02 | 145.16 | 1.9% | 145.82–152.57 | 142.29–147.94 |
| Medium / sparse | 261.34 | 228.84 | 12.4% | 258.93–262.23 | 225.16–229.96 |
| Medium / dense | 356.17 | 341.17 | 4.2% | 349.86–364.92 | 330.51–357.38 |
| Medium / mixed | 493.96 | 456.58 | 7.6% | 465.41–503.19 | 446.90–465.87 |

Only the three sparse-page ranges are disjoint. Tiny dense has a higher median and a wider candidate range, 49.96–54.88 ms versus 51.73–52.23 ms. Its effect is inconclusive, and the observed median regression is retained in the evidence. Small dense/mixed and Medium dense/mixed have lower medians but overlapping ranges; those figures do not establish a reliable improvement magnitude or tail latency. No case was rerun to obtain cleaner results.

**Production-stage diagnosis, elapsed milliseconds.** Each before/after profile uses one warm-up and one instrumented production call. Summed worker work remains in the evidence and is not added to these sequential wall phases.

| Case / version | DET prep | DET network | DB | Crop | CLS | REC | Total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Tiny / sparse / before | 3.05 | 26.57 | 0.42 | 0.43 | 1.18 | 6.02 | 37.70 |
| Tiny / sparse / after | 2.82 | 25.17 | 0.40 | 0.37 | 1.19 | 5.92 | 35.88 |
| Small / dense / before | 0.52 | 13.45 | 0.30 | 0.69 | 16.46 | 82.02 | 113.46 |
| Small / dense / after | 0.54 | 12.73 | 0.28 | 0.65 | 16.83 | 84.44 | 115.49 |
| Medium / mixed / before | 2.86 | 168.30 | 0.56 | 1.63 | 9.31 | 288.80 | 471.46 |
| Medium / mixed / after | 2.79 | 137.30 | 0.62 | 1.55 | 8.55 | 299.55 | 450.38 |

Detector time falls from 26.57→25.17 ms for Tiny sparse, 13.45→12.73 ms for Small dense, and 168.30→137.30 ms for Medium mixed. The new path targets this measured cost. Recognition wall time rises in the two latter single profiles, despite lower medians in their complete-page comparisons. These single profiles cannot isolate scheduling effects from kernel or measurement variation and do not establish a consistent recognition-stage gain. The existing recognition ownership policy is unchanged; its imbalance remains a separate optimization question.

**Memory tradeoff.** Retained activations, normalized/cached weights and request count are identical before/after in all nine page cases. The workspace increase is entirely accounted for by retained kernel scratch.

| Bundle / page | Activation MiB unchanged | Workspace before MiB | Workspace after MiB | Added workspace KiB | RSS change MiB |
| --- | --- | --- | --- | --- | --- |
| Tiny / sparse | 39.06 | 6.338 | 6.838 | 512 | +1.08 |
| Tiny / dense | 12.42 | 10.343 | 10.843 | 512 | +0.69 |
| Tiny / mixed | 41.79 | 11.978 | 12.478 | 512 | +0.73 |
| Small / sparse | 64.29 | 8.927 | 9.302 | 384 | +0.62 |
| Small / dense | 26.85 | 20.036 | 20.411 | 384 | +0.88 |
| Small / mixed | 84.79 | 21.770 | 22.168 | 408 | +0.70 |
| Medium / sparse | 156.36 | 12.427 | 13.427 | 1024 | +1.33 |
| Medium / dense | 46.95 | 23.933 | 24.933 | 1024 | +1.39 |
| Medium / mixed | 175.57 | 25.814 | 26.814 | 1024 | -0.72 |

The maximum additional workspace is 1 MiB for Medium. Tiny adds 512 KiB; Small adds 384 KiB, or 408 KiB for the mixed page. Peak RSS includes allocator and process effects and is not a direct measure of scratch allocation. The negative Medium mixed RSS difference is therefore not claimed as a memory reduction. More simultaneously retained requests would each carry their own bounded scratch increase.

**Correctness and portability.** New coverage checks packed bits independently across padding, dilation, batch crossings, row tails and odd channel counts; compares ordered convolution against an independent oracle; validates indexed fused stores, offsets, missing-scratch fallback and parallel output bands; enforces platform/backend/shape selection; and checks scratch accounting and stable allocation addresses across shape changes. The expanded fused-store regression exercises four convolution shapes, three activations, two activation/add orderings and two input tensors: 48 exact comparisons. Existing SME tests continue to cover extreme values, subnormals, bias order and guarded thread state.

Final local checks passed:

- **410 default checks:** 360 library tests, 45 integration tests and five doctests.
- **348 serial library tests** with default features disabled.
- **365 profiling-enabled library tests**.
- Clippy for all targets/features with warnings denied.
- All-target/all-feature x86_64 macOS compilation on the physical M4. This is a compilation check, not native x86 execution.
- **213/213 local C# reference qualification requests**, including all 66 canonical bundle/fixture outputs and focused worker budgets, batching, alignment, pixel formats/strides, classification options, widths, dilation, reuse and recovery. Models, goldens and tolerances are unchanged.

The full qualifier is warranted because the new spatial path and shared group helper cover multiple stages, batches and shapes beyond the three performance inputs. It uses an external 600-second deadline. The initial launch found that the local .NET SDK was absent from the shell's PATH; loading the workspace's existing `env.sh` resolved it before qualification ran. No SDK or dependency was installed or changed.

**Execution budget and reproduction.** The narrow prototype used **1.808 seconds** of performance execution. The broader selection experiment used **4.128 seconds**. Final page/profile execution used **11.222 seconds**; conservatively charging the broader selection experiment to the retained algorithm gives **15.350 seconds**. The unchanged baseline's cumulative charge, including earlier analysis, is **39.190 seconds**. All remain below 60 seconds per version, excluding builds and correctness qualification. Performance processes have a 12-second timeout and the remaining aggregate budget. There are no adaptive repetitions or sampling retries.

Local artifacts are in `target/arm64-spatial-sme-20261018`. The portable JSON embeds both selection runners, the final comparison/profile runners, validation and qualification records, source patch and hashes. Representative correctness commands are:

```sh
cargo test --release --locked --lib sme -- --test-threads=1
cargo test --release --locked --lib neon_panel_stores -- --test-threads=1
cargo test --release --locked
cargo test --release --locked --no-default-features --lib
cargo test --release --locked --features profiling --lib
cargo clippy --release --locked --all-targets --all-features -- -D warnings
source ../env.sh
cargo run --release --locked --example cpu_qualify
```

Performance conclusions apply to the measured M4 configuration. Other supported hardware retains guarded fallbacks; Windows/x86 performance, concurrent-request throughput and statistical tail latency were not measured in this change.
