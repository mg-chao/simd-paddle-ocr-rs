# ARM64 streaming, spatial convolution and direct-store optimization

The retained combination reduces complete OCR median latency by **5.0–16.0% for Medium** and **3.6–4.5% for Small dense/mixed pages** on the physical Apple M4. These five cases have disjoint before/after sample ranges. Tiny sparse improves 1.3%; Tiny dense/mixed remain inconclusive. **Small sparse has a 5.7% higher median with overlapping, noisy ranges**, so this is not a universal speedup.

All nine page result projections match exactly, all three production-profile pairs preserve complete owned outputs, and **213/213 local reference qualification requests pass**. Retained activation, workspace and weight capacities are unchanged in all nine cases.

The changes address the convolution costs identified in the [preceding pipeline analysis](macos-spatial-sme-pipeline-analysis.md). [Portable evidence](reports/macos-arm64-2026-10-18-streaming-spatial-optimization.json) contains raw samples, profiles, source/executable hashes, the implementation patch, validation logs and reproduction scripts. The baseline production source matches commit `2316ab0beb3ef7a409c207350e73c50bea3e1891`; the existing analysis documentation was preserved.

**Implementation.** Three changes form the best-performing combination tested here:

- [Paired SME tiles](../src/runtime/nhwc/pointwise/sme.rs) execute two 32×32 row tiles within one guarded streaming interval. Groups of 49–64 rows with complete weight panels share the entry/exit overhead; smaller groups and column tails retain their existing paths. Bias initialization and ascending-K FMA order are preserved. The SVCR, TPIDR2, streaming-vector-length and FPCR guards and register clobbers remain intact. No call or unwind occurs inside the assembly interval.
- [Spatial SME selection](../src/runtime/nhwc/dense.rs) now covers dense square 2×2, 5×5 and 7×7 kernels as well as 3×3. It reuses the existing bounded receptive-field packer and matrix executor, preserving input-channel/kernel-row/kernel-column order. Existing gates remain: native macOS ARM64, available SME, at least 32 input channels, 16 output channels and 32 output pixels. Grouped/depthwise and other shapes keep their existing paths. Scratch is checked and reserved as `64 × kernel_height × kernel_width × input_channels` floats per worker, independent of image area.
- [Direct transpose stores](../src/runtime/nhwc/transpose.rs) reuse the native four-row NEON matrix reduction and write directly into interleaved 2×2 output positions. This removes the intermediate matrix write/read and scatter from eligible execution. The final macOS ARM64 gate requires at least 64 input and 64 output channels; narrower layers had no convincing measured benefit. Fused residual/activation stores and channel tails preserve their original arithmetic. Existing scratch reservations remain available to fallback execution.

The paired path uses an 8 KiB temporary result array instead of the one-tile path's 4 KiB array; this is stack storage, not additional retained heap workspace. The new spatial shapes fit within the existing per-request maximum workspace in all measured bundles. Other models may require a larger bounded maximum, which existing accounting and memory limits cover. Public APIs, models, thresholds, recognition ownership and preprocessing settings are unchanged.

**Candidate selection.** Focused correctness checks preceded each experiment. Each graph used one warm-up and three measured calls. The initial broad direct-transpose candidate reduced Medium detector graph latency from 179.741 to 167.805 ms (6.6%); Tiny/Small detector medians were inconclusive. This selected the 64-channel gate.

A four-step SME reduction-loop unroll was tested and discarded. Its three kernel medians were 0.132/1.509/1.414 ms versus the original 0.115/1.424/1.366 ms. Pairing tiles instead gave 0.113/1.065/1.255 ms on the same shapes. These microbenchmarks include input packing, use one thread and preserve output bits; their noisy ranges are retained in the evidence. The final code keeps the original reduction loop.

| Complete graph / threads | Baseline median ms | Direct transpose + paired SME ms | Also expanded spatial SME ms | Combined reduction |
| --- | --- | --- | --- | --- |
| Medium DET 960×608 / 8 | 179.741 | 165.091 | 142.211 | 20.9% |
| Small REC width 128 / 1 | 5.760 | 5.409 | 4.762 | 17.3% |
| Medium REC width 640 / 1 | 65.131 | 58.935 | 56.795 | 12.8% |

All warm-up and sampled graph output hashes match. Unchanged baseline samples were reused between these experiments, without pooling or rerunning. Inputs are deterministic synthetic tensors. The recognition graphs include the full vocabulary head and softmax; these are not production OCR's compact recognition-stage timings. After selection, tests and a module comment were completed, then the final source was rebuilt and fully validated before the independent page measurements below.

**Complete OCR measurements.** Physical Apple M4, 10 cores, 16 GiB RAM, macOS 27.0.1, AC power, native release builds and pinned `nightly-2026-09-08`. Eight shared workers, one request, automatic SIMD/layout/fusion and line ownership, batch size one, adaptive recognition widths, detector limit 960, classification on, alignment off. Model-specific thresholds and hash-verified inputs are identical. macOS scheduling is unpinned.

Sparse/dense/mixed mean `imgs_1`, `imgs_00018069` and `imgs_00006737`. Each before/after case uses separate sequential processes, alternating version order. The worker records its first inference separately, performs one explicit warm-up and measures three owned-result calls on predecoded BGR24. Setup, decoding, serialization and service admission are outside the sample timers. No build or correctness run overlaps timing. Earlier page samples are not pooled.

| Bundle / page | Before ms | After ms | Reduction | Before range ms | After range ms |
| --- | --- | --- | --- | --- | --- |
| Tiny / sparse | 34.94 | 34.49 | 1.3% | 34.93–35.09 | 34.47–34.51 |
| Tiny / dense | 51.78 | 49.78 | 3.9% | 48.38–52.23 | 49.50–50.27 |
| Tiny / mixed | 59.34 | 58.21 | 1.9% | 58.03–60.05 | 56.43–59.18 |
| Small / sparse | 65.95 | 69.73 | -5.7% | 65.64–66.80 | 64.94–72.65 |
| Small / dense | 105.52 | 101.73 | 3.6% | 104.03–106.26 | 100.91–102.87 |
| Small / mixed | 132.25 | 126.29 | 4.5% | 129.57–133.62 | 125.82–127.69 |
| Medium / sparse | 226.43 | 190.09 | 16.0% | 225.69–228.86 | 186.01–190.86 |
| Medium / dense | 358.13 | 340.26 | 5.0% | 357.86–360.64 | 336.63–342.62 |
| Medium / mixed | 468.36 | 417.11 | 10.9% | 449.68–471.96 | 402.15–419.99 |

Small sparse's candidate samples are 64.94, 69.73 and 72.65 ms; its baseline samples are 65.95, 66.80 and 65.64 ms. This does not establish either a reliable improvement or a reliable regression magnitude. Tiny dense/mixed also overlap. No unchanged case was rerun to seek cleaner results. Three samples do not characterize statistical tails or performance on other machines.

**Production-stage diagnosis, elapsed milliseconds.** Each pair has one warm-up and one instrumented production call. The following sequential wall phases exclude the separately reported owned-result conversion; small residual totals cover binding, reshape, result storage and timer overhead.

| Case / version | DET prep | DET network | DB | Crop | CLS | REC | Total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Tiny / dense / before | 0.55 | 7.47 | 0.28 | 0.68 | 16.24 | 23.22 | 48.45 |
| Tiny / dense / after | 0.51 | 7.23 | 0.32 | 0.67 | 16.27 | 26.90 | 51.91 |
| Small / dense / before | 0.53 | 12.64 | 0.30 | 0.69 | 16.20 | 77.41 | 107.78 |
| Small / dense / after | 0.52 | 12.60 | 0.29 | 0.66 | 16.15 | 71.62 | 101.86 |
| Medium / mixed / before | 2.74 | 144.28 | 0.53 | 1.12 | 9.87 | 301.09 | 459.65 |
| Medium / mixed / after | 2.77 | 118.17 | 0.50 | 1.63 | 9.64 | 299.01 | 431.74 |

Small dense recognition wall time decreases 7.5% in this profile; summed recognition backbone work decreases 478.68→455.32 ms. Medium mixed's detector decreases 18.1%, while recognition wall time changes only 0.7% despite backbone work falling 1564.63→1505.07 ms. Tiny dense recognition wall time rises despite lower summed backbone work (152.71→143.53 ms). Summed worker work overlaps and must not be added to wall phases. These single profiles support the targeted kernel improvements but cannot isolate scheduling variation or guarantee a corresponding recognition wall-time gain. Recognition ownership imbalance remains a separate optimization question.

**Memory.** All retained diagnostics, including activation, total workspace, kernel scratch, normalized weights, cached weights and request count, are identical before/after in all nine cases. Workspace remains 6.84–12.48 MiB for Tiny, 9.30–22.17 MiB for Small and 13.43–26.81 MiB for Medium. Peak RSS differences range from -0.30 to +0.31 MiB and are not evidence of a persistent memory change. Full byte counts and per-case RSS are preserved in the evidence.

**Validation.** Tests cover independent ordered convolution results, packed bits, padding/dilation/stride, batch and row-band crossings, odd channel counts, missing scratch, SME thread-state guards, bias order and subnormals. Direct transpose is compared bitwise against the old buffered GEMM/scatter oracle, with fused stores and destination sentinels. Expanded graph tests cover 84 convolution and 24 transpose activation/residual-order comparisons. Scratch tests check accounting and stable reuse across reshapes.

Final local release validation passed:

- 413 default checks: 363 library tests, 45 integration tests and five doctests.
- 351 serial library tests with default features disabled; 368 profiling-enabled library tests.
- Clippy for all targets/features with warnings denied.
- All-target/all-feature x86_64 macOS compilation on the M4; no native x86 execution claim.
- 213/213 C# reference requests over 16 configurations, including all 66 canonical bundle/fixture outputs, worker budgets, batching, alignment, formats/strides, classification options, widths, dilation, reuse and recovery. Goldens and tolerances are unchanged.

The full reference suite was needed because shared matrix and spatial execution affect multiple stages and shapes. It ran locally with the existing .NET SDK and an external 600-second deadline, completing in 159.72 seconds. No CI, remote runner or virtual machine was used.

**Budget and reproduction.** Cumulative performance execution is **36.594 seconds for the unchanged baseline**, including the preceding analysis, and **18.378 seconds for the retained candidate**, conservatively charging all retained prototype experiments. The discarded unroll version used 0.419 seconds. Builds and correctness checks are excluded; all process wall time, warm-ups and internal repetitions are included. Every version remains under 60 seconds. Timed subprocesses use a 12-second ceiling and the remaining aggregate budget. There is no adaptive sample growth.

Local artifacts and copied executables are under `target/arm64-direct-store-streaming-20261018`. The portable JSON embeds the final source patch, hash manifests, scripts, command records, logs and raw measurements. Intermediate experiments preserve source hashes and measurements; the discarded unroll source itself is not archived. Representative validation commands are:

```sh
cargo test --release --locked --lib sme -- --test-threads=1
cargo test --release --locked --lib direct_transpose -- --test-threads=1
cargo test --release --locked --lib neon_ -- --test-threads=1
cargo test --release --locked
cargo test --release --locked --no-default-features --lib
cargo test --release --locked --features profiling --lib
cargo clippy --release --locked --all-targets --all-features -- -D warnings
source ../env.sh
cargo run --release --locked --example cpu_qualify
```

Results apply to the measured M4 configuration. Windows/x86 performance, concurrent-request throughput and tail latency were not measured. Unsupported hardware and guarded thread states retain the existing fallback paths.
