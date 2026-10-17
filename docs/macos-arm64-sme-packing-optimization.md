# Apple M4 pointwise convolution packing and crossover optimization

The retained implementation lowers complete OCR median latency by **2.4–10.6%** across nine local comparisons against `83fc0c7`. It admits the measured smaller pointwise convolutions to SME and replaces scalar input packing with 4×4 NEON transposes. All nine paired OCR output projections match exactly, all three profile pairs preserve complete owned outputs, and the full **213-request reference qualification passes**.

Six of nine before/after measured ranges are disjoint. Small mixed, Medium dense and Medium mixed ranges overlap, so their precise gains remain uncertain with three samples. In particular, the 2.4–2.5% Medium dense/mixed improvements are lower medians in this run, not established small-effect significance. Tiny adds **1.08–1.31 MiB** of retained workspace; Small and Medium workspace and every bundle's retained activations and cached weights are unchanged.

The [preceding pipeline analysis](macos-sme-tile-pipeline-analysis.md) identified Small's 96→192 pointwise layers, Medium's larger recognition layers, and detector convolution as the main targets. [Portable evidence](reports/macos-arm64-2026-10-17-sme-packing-optimization.json) contains raw samples, complete comparison outputs, stage/owner profiles, kernel probes, source and executable hashes, validation logs and reproduction scripts. Model assets, decoding, score thresholds, dimensions and numerical tolerances are unchanged.

The [SME crossover](../src/runtime/nhwc/pointwise.rs) now requires at least 32 rows in an executing band, both channel dimensions at least 96 and one dimension at least 192. This includes the measured 96→192 recognition and 128→256 detector products, plus their reverse directions. The previous gate required dimensions of at least 192 and 384. Eligibility remains restricted to macOS ARM64, the NEON/FMA policy and detected SME support. Shorter bands, smaller products and other hardware retain their existing paths. Scratch reservation and execution share the eligibility function.

The [input packer](../src/runtime/nhwc/pointwise/sme.rs) now reads four adjacent K values from each of four input rows and transposes them into four contiguous groups of row lanes. It resolves duplicated tail-row references once per panel, handles the last one to three K values separately, and preserves the existing sixteen-row K-major layout. Its operations only load, permute and store bits. The 64-row request-local scratch size and immutable sixteen-column weight panels are unchanged. The existing 32×32/16×16 SME assembly, bias-first ascending-K FMA order, fused stores, thread-state checks and NEON fallbacks remain intact.

Candidate selection used three fixed representative layer shapes, one warm-up and three single calls per kernel, including packing and identity output stores. Complete output arrays match bit-for-bit. The original SME kernel was already 59.7% faster than NEON at rows/K/N = 384/96/192 and 63.6% faster at 4560/128/256, which supports widening the gate. The larger case represents an existing eligible recognition layer. Values below are per-build medians; the raw JSON preserves all samples and alternating within-build kernel order.

| Rows / K / N | NEON in retained build ms | Original packing SME ms | Retained packing SME ms | Rejected unroll SME ms |
| --- | ---: | ---: | ---: | ---: |
| 384/96/192 | 0.4647 | 0.1845 | 0.1692 | 0.1666 |
| 4560/128/256 | 4.4246 | 1.5970 | 1.4731 | 1.4918 |
| 960/512/1024 | 8.7070 | 1.3770 | 1.3917 | 1.3785 |

The retained packer's first two SME medians are 8.3% and 7.8% lower than the original packer's, while the large-layer median is 1.1% higher. The second and third shapes have wide ranges, so these separate-build microbenchmarks do not isolate a reliable small gain for every shape. Adoption relies on the complete paired OCR comparison below, not an assumption that every layer gets faster.

A further prototype unrolled the 32×32 tile's reduction by four while preserving ascending-K FMA order. Its SME medians differed from the retained implementation by +1.6%, −1.3% and +0.9% latency reduction across the three shapes. This was not a consistent benefit, so the assembly change was rejected before page benchmarking. Its code and raw results are preserved in the evidence; it is absent from production. The selected change keeps the measured broader crossover and simpler packing improvement.

The physical host is Apple M4, 10 cores, 16 GiB RAM, macOS 27.0.1, on AC power. Page comparisons use native release/default features and pinned `nightly-2026-09-08`, eight shared workers, one request, automatic SIMD/layout/fusion and line ownership, detector limit 960, adaptive recognition widths, batch size one, classification on and alignment off. Scheduling is unpinned; core placement and frequencies are not measured.

The three inputs are sparse `imgs_1`, dense `imgs_00018069` and mixed `imgs_00006737`. Each before/after pair uses fresh processes in alternating order, identical hash-verified predecoded BGR24, a separately recorded first call, one explicit warm-up and three measured owned-result calls. Builds and correctness work were idle during benchmarking. Timings include owned conversion and exclude image decoding, serialization and service admission waiting. The first call additionally warms request/process state. No samples from the earlier analysis or other implementations are pooled into these numbers.

| Bundle / page | Before ms | After ms | Reduction | Before range ms | After range ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tiny / sparse | 45.49 | 42.52 | 6.5% | 45.39–47.56 | 41.99–44.31 |
| Tiny / dense | 63.67 | 60.15 | 5.5% | 63.52–66.79 | 60.14–60.70 |
| Tiny / mixed | 76.34 | 68.24 | 10.6% | 74.62–78.70 | 67.67–71.02 |
| Small / sparse | 84.79 | 80.97 | 4.5% | 83.50–84.81 | 79.19–81.37 |
| Small / dense | 144.17 | 137.06 | 4.9% | 143.37–155.01 | 134.28–140.79 |
| Small / mixed | 184.60 | 165.72 | 10.2% | 174.26–188.19 | 163.98–178.98 |
| Medium / sparse | 287.22 | 259.08 | 9.8% | 275.76–301.68 | 254.07–264.80 |
| Medium / dense | 413.83 | 403.69 | 2.4% | 413.52–425.55 | 384.81–434.50 |
| Medium / mixed | 564.32 | 550.41 | 2.5% | 549.13–596.86 | 543.17–551.94 |

The shared broader gate can improve both detection and recognition; the result is not limited to the dense Small case that motivated the first new shape. Tiny sparse/dense/mixed medians improve 6.5%/5.5%/10.6%. Small improves 4.5%/4.9%/10.2%, and Medium 9.8%/2.4%/2.5%. The overlapping ranges above remain visible. This comparison does not resolve the earlier optimization's Tiny-versus-older-baseline timing limitation; it establishes a separate change against `83fc0c7`.

Three separate before/after production profile pairs diagnose representative paths. Each version has one warm-up and one instrumented call. Values are elapsed milliseconds. These profiles are not components of the ordinary medians, and nested summed worker work in the JSON must not be added to wall phases.

| Case / version | DET prep | DET infer | DB | Crop | CLS | REC | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Tiny / dense / before | 0.52 | 8.37 | 0.32 | 0.73 | 17.72 | 34.52 | 62.20 |
| Tiny / dense / after | 0.29 | 8.26 | 0.32 | 0.68 | 17.52 | 33.02 | 60.10 |
| Small / dense / before | 0.63 | 15.20 | 0.30 | 0.65 | 18.08 | 108.42 | 143.29 |
| Small / dense / after | 0.38 | 15.34 | 0.32 | 0.69 | 19.37 | 99.28 | 135.40 |
| Medium / mixed / before | 2.78 | 188.53 | 0.64 | 1.37 | 8.77 | 352.96 | 555.08 |
| Medium / mixed / after | 1.62 | 183.94 | 0.56 | 1.50 | 10.13 | 328.72 | 526.49 |

Tiny dense recognition falls from 34.52 to 33.02 ms in its profile; Small dense falls from 108.42 to 99.28 ms. Medium mixed records detector inference of 188.53→183.94 ms and recognition of 352.96→328.72 ms. Classification is not consistently faster, and its implementation was not changed. These individual profiles support where to investigate the gains, rather than proving independent per-stage speedups. Medium owner records remain available for the scheduling follow-up identified in the analysis.

Tiny's retained kernel scratch grows by 1.078/1.234/1.313 MiB on sparse/dense/mixed pages, respectively, because layers newly eligible for SME now reserve packing space. Its total retained workspace grows by the same amounts; no hidden additional weight format is cached. Small and Medium already reserve enough scratch for larger layers, so every measured retained memory counter remains unchanged. All nine cases preserve retained activations, normalized weights, cached packed weights and request counts. Process peak RSS increases by approximately 0.08–1.31 MiB across the pairs; these are process lifetime high-water marks and also include setup/allocator effects.

Validation ran locally on the physical machine. The final selected source passes **404 default checks** (354 library, 45 integration, five doctests), **342 serial library tests**, **359 profiling-enabled library tests**, clippy with warnings denied, and all-target/all-feature x86_64 macOS compilation. Cross-compilation does not constitute native x86 execution or performance validation. Eleven library diagnostics and one local integration check remain intentionally ignored in ordinary suites.

The new packing test exercises 132 combinations of row counts and K lengths, checks panel layout and duplicated tail rows, covers signed zeros, subnormals, infinities and quiet/signaling NaN payloads, and verifies surrounding sentinels. Ordered convolution tests cover the new channel directions and row tails. Dispatch tests cover the 31/32-row, 95/96-channel and 191/192-channel boundaries and noneligible backends/platforms. Session tests verify exact output, reservation, retained accounting, pointer reuse and reshapes for 96↔192 and 128↔256. Existing tests exercise fused stores, extreme reductions, parallel scratch ownership, missing scratch and live ZA/nondefault FPCR rejection.

The **213/213 reference qualification** covers all 66 canonical bundle/fixture outputs plus serial/worker budgets, exact-width batches, CTC alignment/token boxes, padded pixel formats, classification options, widths, dilation, reuse and recovery. It ran with a 600-second external timeout. Goldens and comparison tolerances are unchanged. Nine measured before/after result projections are exact; the three profile pairs additionally compare hashes of complete owned outputs.

Selected-version performance execution totals **14.066 seconds**: the fixed kernel probe, nine paired page cases on this version, and three profile cases. Original-baseline cumulative execution, including its earlier analysis and the new kernel/page/profile calls, is **43.843 seconds**. The rejected unroll prototype uses **0.422 seconds**. Each remains below 60 seconds. Page cases have three measured calls, one explicit warm-up and one first call; profiles have one warm-up and one measured call; each kernel has one warm-up and three single-call samples per shape. All subprocess loading and harness calls are included, with ten-second per-command caps and remaining aggregate budgets. Builds and correctness qualification are excluded. There were no retries for cleaner numbers or sampling-growth loops.

After rejecting unrolling, the selected source was restored. The only subsequent source change adds explicit special-value coverage to the packing test. All final default/profiling/qualification executables were rebuilt and match the measured selected executables byte-for-byte. The new test and all validation suites ran on the final source, whose hashes are recorded separately. No stale executable is used as evidence for an unbuilt production change.

Local artifacts and scripts are in `target/arm64-sme-packing-20261017`. The portable JSON embeds the measurement drivers, source patch, rejected candidate, final source hashes and validation logs. Reproduction requires a fresh evidence directory, separately built baseline/selected executables and the manifest's hash-verified model/pixel assets; scripts reject overwritten measurements and changed source. Relevant validation commands are:

```sh
cargo test --release --locked --lib sme -- --test-threads=1
cargo test --release --locked
cargo test --release --locked --no-default-features --lib
cargo test --release --locked --features profiling --lib
cargo clippy --release --locked --all-targets --all-features -- -D warnings
cargo check --release --locked --all-targets --all-features --target x86_64-apple-darwin
source ../env.sh
target/release/examples/cpu_qualify
```

This supports adopting the selected implementation on the measured M4 workload with the stated Tiny memory tradeoff. Other SME hardware/vector lengths, service throughput, ground-truth OCR accuracy and p95/p99 latency were not measured. Unsupported hardware retains the existing fallback; no general speedup claim is made for those platforms.
