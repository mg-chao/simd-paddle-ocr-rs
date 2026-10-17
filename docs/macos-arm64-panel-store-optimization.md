# ARM64 fused convolution store optimization

The retained implementation reduces complete OCR median latency by **3.9–16.9%** across nine local comparisons against `ca0996a`. It applies fused activation/residual operations to a complete sixteen-channel convolution panel instead of four channels per call. This targets the remaining convolution costs identified in the [pipeline analysis after SME packing](macos-sme-packing-pipeline-analysis.md), without changing models, image dimensions, classifier policy or arithmetic reductions.

All nine measured result projections are exactly equal, all three production profile pairs preserve complete owned outputs, and **213/213 reference qualification requests pass**. Retained activations, workspace, kernel scratch, normalized/cached weights and request counts are unchanged in every page comparison. Eight of nine measured before/after ranges are disjoint; Tiny mixed ranges overlap, so its precise improvement is uncertain with three samples.

[Portable evidence](reports/macos-arm64-2026-10-18-panel-store-optimization.json) contains raw samples, outputs, source/executable hashes, candidate-selection graphs, stage/owner profiles, qualification, validation logs and reproduction scripts. These results concern the final measured source in this worktree, built from baseline `ca0996a` plus the included patch and new regression test.

**Implementation.** The [SME path](../src/runtime/nhwc/pointwise/sme.rs) already stages each accumulator row as sixteen contiguous channels. Its stores now consume that row as `Simd<f32, 16>`: two fused-store calls per 32-channel row instead of eight, and one call per 16-channel tail row instead of four. The [NEON pointwise path](../src/runtime/nhwc/pointwise/neon.rs) and [NEON dense spatial path](../src/runtime/nhwc/dense.rs) combine their four existing accumulator vectors into one sixteen-lane store value. The shared [masked store helper](../src/runtime/nhwc/pointwise.rs) retains the complete output index and channel index for residuals, channel broadcasts, worker offsets and partial panels.

This reduces repeated recipe dispatch and operand indexing in the [prepared fused store](../src/runtime/optimized/store.rs). It does not add 512-bit NEON hardware instructions: the compiler lowers the wide value to native vector operations. Convolution's bias-first ascending-K FMA chains, SME packing, tile assembly, streaming-state guards, crossover and fallbacks remain intact. Activation operation order is unchanged. The dense change is limited to the ARM64 fused four-vector instantiation; portable and x86 paths retain their existing stores. No public API or heap workspace is added.

**Candidate selection.** Two alternatives were tested: widening SME stores alone, then extending the same approach to narrow NEON pointwise and dense spatial convolution. Each uses three deterministic recognition input shapes, one thread, one warm-up and three single graph calls per shape. All warm-up and sampled output hashes match the unchanged baseline exactly. These are full `Session` recognition graph timings, including its vocabulary head and softmax, rather than page OCR's compact projection/CTC path.

| REC graph / width | Baseline ms | SME stores only ms | SME + NEON stores ms |
| --- | ---: | ---: | ---: |
| Tiny / 128 | 3.123 | 2.874 | 2.433 |
| Small / 128 | 7.002 | 6.136 | 5.960 |
| Medium / 640 | 80.422 | 67.334 | 67.146 |

The extension gives the largest additional benefit to Tiny, while Medium is nearly unchanged relative to the SME-only variant. The Small candidate ranges slightly overlap and Medium's ranges overlap; their small incremental differences are not established statistically. The broader variant was selected because it addresses the same repeated store work in narrow kernels and improves Tiny, then subjected to the independent full-page comparison below. Earlier graph samples are selection evidence and are not pooled into page samples.

Before final page measurement, a regression test and NEON golden coverage were added and the NEON call was formatted. The source difference is recorded explicitly. Formatting changes the executable hash, so the earlier graph executable is not presented as validation of the final executable. The final source was rebuilt, passed focused correctness, and received its own page and production-profile measurements. No production source changed afterward.

**Matched full-page comparison.** Physical Apple M4, 10 cores, 16 GiB RAM, macOS 27.0.1, AC power, native release/default features and pinned `nightly-2026-09-08`. Eight shared workers, one request, automatic SIMD/layout/fusion and line ownership, detector limit 960, adaptive recognition widths, batch size one, classification enabled, alignment off. The manifest's model-specific thresholds, classifier threshold and hash-verified model/pixel assets are unchanged. Scheduling is unpinned.

For each case, baseline and final executables run in separate sequential processes, alternating order by case. Each records its first call separately, performs one explicit warm-up and measures three owned-result calls on the same predecoded BGR24 input. Times exclude setup, image decoding, serialization and admission waiting. Builds and correctness work are idle during measurement. Sparse/dense/mixed are `imgs_1`, `imgs_00018069`, and `imgs_00006737` respectively; all values below are milliseconds.

| Bundle / page | Before ms | After ms | Reduction | Before range ms | After range ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tiny / sparse | 41.54 | 38.54 | 7.2% | 40.71–47.31 | 37.95–38.59 |
| Tiny / dense | 64.05 | 55.14 | 13.9% | 61.94–64.25 | 54.03–55.61 |
| Tiny / mixed | 73.97 | 65.41 | 11.6% | 73.13–75.76 | 65.21–75.37 |
| Small / sparse | 88.64 | 82.54 | 6.9% | 87.73–89.09 | 81.41–83.68 |
| Small / dense | 135.73 | 112.79 | 16.9% | 133.18–140.20 | 111.18–114.43 |
| Small / mixed | 172.38 | 152.11 | 11.8% | 161.55–174.75 | 151.82–152.91 |
| Medium / sparse | 279.15 | 252.41 | 9.6% | 269.51–280.46 | 240.71–263.39 |
| Medium / dense | 399.88 | 365.23 | 8.7% | 391.95–403.96 | 356.76–373.92 |
| Medium / mixed | 530.31 | 509.82 | 3.9% | 528.00–553.39 | 505.75–517.76 |

The strongest measured improvement is Small dense, **135.73→112.79 ms (16.9%)**. Tiny dense improves **64.05→55.14 ms (13.9%)**; Medium sparse improves **279.15→252.41 ms (9.6%)**. Medium mixed's 3.9% lower median has disjoint ranges in this run, but three samples do not establish tail latency or small-effect significance. Tiny mixed has one slower after sample at 75.37 ms; it is retained in the range rather than discarded or rerun.

**Production diagnosis.** Three before/after profile pairs use one warm-up and one instrumented call per version. These are elapsed phase times, not components of the ordinary page medians. Summed worker counters and owner finish records are preserved in the JSON and must not be added to these sequential wall phases.

| Case / version | DET prep | DET network | DB | Crop | CLS | REC | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Tiny / dense / before | 0.31 | 8.23 | 0.31 | 0.66 | 17.32 | 30.48 | 57.32 |
| Tiny / dense / after | 0.63 | 8.22 | 0.30 | 0.79 | 17.80 | 26.62 | 54.38 |
| Small / dense / before | 3.59 | 21.66 | 0.31 | 0.74 | 17.99 | 100.11 | 144.94 |
| Small / dense / after | 0.51 | 13.55 | 0.29 | 0.61 | 17.14 | 79.38 | 111.49 |
| Medium / mixed / before | 1.60 | 180.99 | 0.65 | 1.46 | 9.96 | 339.23 | 533.92 |
| Medium / mixed / after | 2.86 | 172.39 | 0.56 | 1.43 | 9.28 | 296.47 | 483.01 |

Recognition falls from 30.48→26.62 ms in Tiny dense, 100.11→79.38 ms in Small dense, and 339.23→296.47 ms in Medium mixed. Classification does not show a consistent gain. Small's unusually high baseline preprocessing/network times and Medium's higher candidate preprocessing time demonstrate the variation in single instrumented samples; those differences cannot all be attributed to the changed store loop. Use the paired ordinary samples for the adoption decision. The existing recognition ownership policy is unchanged, and scheduling imbalance remains a possible future target.

**Memory and correctness.** Every page's retained diagnostics are identical before/after, including kernel scratch and cached weights. Peak process RSS changes range from −1.28 to +0.19 MiB, which includes allocator/setup effects and is not evidence of a deliberate memory reduction. Existing stack-based accumulator storage remains in use; retained heap capacity does not measure stack frame size.

The new [fused-store regression test](../src/runtime/optimized/tests/neon_epilogue.rs) compares fused execution against independently executed graph operators for three convolution shapes, three activations, two activation/add orderings and two changed input tensors: 36 exact output comparisons. It covers narrow NEON pointwise, SME-eligible pointwise, spatial borders, full residuals, channel bias, two batches, short row groups and parallel output bands. The serial build uses one thread. Existing low-level SME/NEON tests cover odd channel tails, offsets, sentinel preservation, extreme values, ordered reductions, scratch reuse and guard fallback. The prepared-store golden test now explicitly includes NEON at scalar/4/8/16 lane widths, preserving its nonfused Erf polynomial even though convolution uses FMA.

Local final-source validation passed:

- **405 default checks:** 355 library tests, 45 integration tests, five doctests.
- **343 serial library tests** with default features disabled.
- **360 profiling-enabled library tests**.
- Clippy across all targets/features with warnings denied.
- All-target/all-feature x86_64 macOS compilation. This is a compilation check on the physical M4, not native x86 execution.
- **213/213 reference qualification requests**, including all 66 canonical bundle/fixture outputs, thread budgets, exact-width batches, alignment, pixel formats/strides, classification options, widths, dilation, reuse and recovery. Goldens and tolerances are unchanged.

The qualifier runs locally against the pinned C# checkout with an external 600-second deadline. Full qualification is warranted because the changed stores are shared by multiple stages, shapes and modes that the three page inputs cannot cover. The benchmark's nine before/after result projections match exactly; the three profile pairs additionally match hashes of complete owned outputs. Reference compatibility is distinct from ground-truth OCR accuracy.

**Execution budget and reproduction.** The final source uses **12.109 seconds** of performance execution: nine page cases and three production-profile cases, including startup/loading, first calls, warm-ups and measured calls. The expanded prototype used 1.571 seconds and the SME-only prototype 0.970 seconds. Conservatively charging the expanded selection probe to the retained algorithm gives **13.680 seconds**. The unchanged baseline's cumulative execution, including its earlier analysis and these comparisons, is **36.677 seconds**. Every source version remains below 60 seconds; build time and correctness qualification are excluded. Each performance subprocess has a 12-second cap and its remaining aggregate budget. There are no retries for cleaner numbers or growing sampling loops.

Local artifacts live in `target/arm64-sme-stores-20261017`. The portable JSON embeds drivers, validation logs, the tracked source patch and new regression-test text. Drivers reject overwritten measurement markers and verify source, model, input and executable hashes. Rebuild both variants in a fresh evidence directory for a new comparison and reset that version's run counts. Main final validation commands were:

```sh
cargo test --release --locked
cargo test --release --locked --no-default-features --lib
cargo test --release --locked --features profiling --lib
cargo clippy --release --locked --all-targets --all-features -- -D warnings
cargo check --release --locked --all-targets --all-features --target x86_64-apple-darwin
source ../env.sh
# Run the frozen final qualifier under the driver's 600-second deadline.
python3 target/arm64-sme-stores-20261017/qualify.py
```

The measured result supports adopting the broader fused-store change on this M4 workload. Other ARM64 CPUs, other SME implementations, native x86 performance, service throughput and p95/p99 are not measured. Further convolution, classifier and scheduling improvements remain possible; this change does not assume those bottlenecks have been exhausted.
