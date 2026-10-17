# macOS ARM64 recognition acceleration with SME

The retained change accelerates large-channel recognition convolutions on the physical Apple M4 using ARM Scalable Matrix Extension (SME). Against revision `2e5287f`, complete page latency fell **4–7% for Small and 10–25% for Medium** in the final focused comparison. Tiny has no demonstrated gain: its dense-page median increased **3.4%**, while sparse and mixed medians decreased about 1%. Three samples cannot establish the cause of those small differences.

The [preceding stage analysis](macos-current-pipeline-analysis.md) identified large 1×1 recognition convolutions as the priority. The retained implementation preserves ordered F32 arithmetic and existing scheduling; it changes neither model weights nor OCR settings. All nine paired page outputs match exactly, and the full **213-request Rust/C# correctness qualification passes**. [Portable evidence](reports/macos-arm64-2026-10-17-sme-optimization.json) contains samples, profiles, source and binary hashes, the production patch and new source files, qualification records, drivers, and rejected experiments.

## Retained implementation

[SME execution](../src/runtime/nhwc/pointwise/sme.rs) packs at most 64 input rows into reusable request storage and computes 16×16 outer-product tiles. It reuses the existing immutable 16-column weight panels. Each output starts from bias and processes K in ascending order; fused epilogues use the same store interface, channel coordinates and global output indices as NEON. Row and column tails are bounded. Packing is included in page timings.

[Dispatch](../src/runtime/nhwc/pointwise.rs) is compiled only on macOS ARM64 and requires the NEON/FMA backend, at least 32 rows per executing band, and input/output channels of at least 384. Thus automatic and explicit `NeonFma` policies can select this specialization. Scalar, Portable4, x86, constant MatMul and smaller convolutions retain their existing kernels. There is no new dependency or public configuration field. The pinned nightly enables its existing experimental SME feature-detection macro.

Each tile checks runtime SME support through its caller and verifies a 64-byte streaming vector length, inactive streaming/ZA state, a null TPIDR2 save pointer, and default FPCR before entering streaming mode. A failed condition uses NEON. The assembly region contains no calls, allocations or unwinding paths; it exits streaming mode before returning and declares all vector and predicate clobbers. Leaving dormant ZA untouched satisfies the preservation requirements in [Arm AAPCS64](https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst). This implementation uses a normal non-streaming Rust interface; it does not expose a streaming-compatible API.

[Session reservation](../src/runtime/optimized.rs) allocates the largest required `64 × input_channels` float packing tile per existing worker before activation writes. Buffers survive warm calls and reshapes, and existing retained-scratch accounting includes them. Parallel workers receive exclusive scratch and output slices. The final version isolates this dispatch so unaffected kernels retain their original row-band closures.

## Paired page latency

Measurements used native release builds on Apple M4, macOS 27.0.1, eight workers, one request, automatic dispatch/layout/fusion, adaptive recognition widths, batch size one, classification on and alignment off. Both executables consumed the same hash-verified predecoded BGR24 images. Each case ran once for first-inference measurement, once for warm-up, and three times for ordinary measurement; the table reports the median and full measured range. Process order alternated by case. Builds and other inference were idle during measurements.

| Bundle / page | Before ms | After ms | Latency reduction | Before range ms | After range ms |
|---|---:|---:|---:|---:|---:|
| Tiny / sparse | 43.01 | 42.66 | +0.8% | 42.23–43.46 | 41.85–42.72 |
| Tiny / dense | 63.49 | 65.62 | -3.4% | 63.13–64.21 | 64.55–70.46 |
| Tiny / mixed | 74.59 | 73.81 | +1.1% | 73.07–75.26 | 73.63–75.55 |
| Small / sparse | 94.16 | 90.25 | +4.2% | 93.56–94.57 | 90.21–90.46 |
| Small / dense | 181.33 | 168.31 | +7.2% | 180.59–183.88 | 166.68–172.24 |
| Small / mixed | 219.26 | 207.49 | +5.4% | 217.87–220.45 | 202.68–207.85 |
| Medium / sparse | 367.82 | 329.55 | +10.4% | 366.20–371.59 | 321.03–336.11 |
| Medium / dense | 699.18 | 554.52 | +20.7% | 685.06–701.99 | 542.99–560.55 |
| Medium / mixed | 907.54 | 681.31 | +24.9% | 901.56–914.14 | 679.17–722.21 |

These are complete OCR calls including owned result conversion; image decoding, serialization and admission queueing are excluded. Negative reduction means a slower median. No unchanged-code reruns were made to improve these figures. The Tiny dense result remains visible despite its unfavorable median.

## Stages after optimization

Each row below is a separate instrumented run with one warm-up and one measured call, in milliseconds. It diagnoses the retained implementation and is not added to the ordinary medians above. Small unlisted bookkeeping is included in profile total; nested worker work counters overlap and must not be summed into wall time. Full nested counters, owner intervals, startup costs and memory are in the evidence.

| Bundle / page | DET prep | DET inference | DB boxes | Crop | CLS wall | REC wall | Profile total |
|---|---:|---:|---:|---:|---:|---:|---:|
| Tiny / sparse | 2.77 | 29.70 | 0.44 | 0.39 | 1.34 | 7.47 | 42.13 |
| Tiny / dense | 0.57 | 8.12 | 0.31 | 0.70 | 17.24 | 38.19 | 65.14 |
| Tiny / mixed | 2.07 | 23.89 | 0.53 | 1.33 | 8.32 | 36.76 | 72.91 |
| Small / sparse | 2.94 | 61.29 | 0.43 | 0.42 | 1.41 | 23.31 | 89.83 |
| Small / dense | 0.51 | 15.00 | 0.29 | 0.67 | 17.55 | 128.06 | 162.11 |
| Small / mixed | 2.03 | 54.19 | 0.57 | 1.39 | 9.08 | 140.48 | 207.75 |
| Medium / sparse | 4.13 | 260.18 | 0.50 | 0.59 | 1.49 | 70.18 | 337.09 |
| Medium / dense | 0.73 | 59.31 | 0.31 | 0.87 | 21.36 | 484.49 | 567.09 |
| Medium / mixed | 2.78 | 220.64 | 0.56 | 1.18 | 10.49 | 479.68 | 715.35 |

The optimization addresses recognition arithmetic. Detection and orientation remain substantial costs, especially on sparse and Tiny pages. A single stage sample supports identifying remaining bottlenecks, not attributing small differences or claiming a statistically established stage speedup.

## Memory and candidate selection

| Bundle / page | Added retained workspace MiB | Peak RSS before → after MiB |
|---|---:|---:|
| Tiny / sparse | 0.00 | 69.06 → 68.67 |
| Tiny / dense | 0.00 | 45.00 → 44.95 |
| Tiny / mixed | 0.00 | 77.16 → 76.81 |
| Small / sparse | 2.98 | 146.05 → 148.67 |
| Small / dense | 2.98 | 116.38 → 119.12 |
| Small / mixed | 3.16 | 177.23 → 180.16 |
| Medium / sparse | 6.44 | 449.33 → 454.89 |
| Medium / dense | 6.44 | 349.62 → 357.81 |
| Medium / mixed | 6.81 | 475.36 → 480.78 |

Small adds approximately 3.0–3.2 MiB and Medium 6.4–6.8 MiB of retained workspace for these single-request workloads. Tiny adds none. Shared packed weights and retained activation capacities are unchanged in every paired case. Peak RSS also includes allocator and process effects; it is not interchangeable with retained workspace. More simultaneous requests can multiply private scratch costs.

Two NEON 12×8 input-packing prototypes, including a four-K unrolled variant, failed to improve the three representative shapes and were rejected. The original six-row NEON loop already kept its accumulators in registers. The isolated SME prototype instead produced the following result, including packing but excluding initial workspace and weight allocation, with one warm-up and three single-call samples per variant:

| Rows / K / columns | NEON median ms | SME prototype median ms | Reduction |
|---|---:|---:|---:|
| 96 / 384 / 768 | 0.469 | 0.162 | 65.4% |
| 960 / 512 / 1024 | 8.915 | 3.343 | 62.5% |
| 480 / 1536 / 768 | 9.931 | 3.406 | 65.7% |

These prototype figures motivated integration; they are not measurements of the final page implementation. The first integrated version reduced Small dense/mixed medians by 5.4/6.5% and Medium by 15.5/19.8%, but all three Tiny medians were 1.5–2.0% slower. Workspace dispatch was then isolated from unaffected kernels and the final comparison above was run once. Results from these versions are kept separate. Isolated tile gains are diluted by other stages, packing, worker scheduling and concurrent kernel execution; no hardware-counter attribution was established.

## Local validation and reproduction

The retained source passed **402 default checks**: 352 library tests, 45 integration tests and five doctests. Serial library tests passed **340**, and profiling-enabled library tests passed **357**. Five new regression tests cover bias-first reduction order, cancellation across K=512, subnormal/extreme values, offsets and tails, indexed fused stores, one/two/four/eight worker budgets, insufficient-scratch fallback, live ZA and nondefault FPCR rejection, workspace reuse and retained accounting. Ten library diagnostics and one local integration test remain intentionally ignored by the ordinary suite.

Full reference qualification passed **213/213**, one execution per case, with a 600-second external timeout. It includes all 66 saved corpus/model results plus worker budgets, batching, alignment, rotations, pixel formats, classification settings, dilation, reuse and error recovery. Goldens, weights and tolerances are unchanged. The qualification is a correctness test and makes no performance claim. Nine page comparisons preserve every field of the benchmark's result projection exactly; nine profile warm-up/measured pairs also preserve complete owned output exactly.

Clippy with warnings denied and all targets/features passes. The serial build and all-target/all-feature x86_64 macOS cross-check pass; the latter is compilation evidence only. Changed files pass rustfmt and whitespace checks. Whole-tree formatting reports the pre-existing assertion layout in unchanged `src/runtime/nhwc/pointwise/large_channel_tests.rs`.

Retained performance execution totaled **21.353 seconds**: nine first calls, nine ordinary warm-ups, 27 ordinary samples, nine profile warm-ups and nine profiles, or 63 OCR executions. The baseline's cumulative execution across the preceding analysis and both integration comparisons is **53.399 seconds**. Both remain within 60 seconds per version, excluding build and correctness execution. Every benchmark/profile subprocess had at most ten seconds and an aggregate remaining-budget cap. Each rejected kernel prototype took less than 0.2 seconds; the superseded integration took 12.744 seconds.

Relevant commands are:

```sh
cargo test --release --locked
cargo test --release --locked --no-default-features --lib
cargo test --release --locked --features profiling --lib
cargo clippy --release --locked --all-targets --all-features -- -D warnings
cargo check --release --locked --all-targets --all-features --target x86_64-apple-darwin
cargo build --release --locked --example cpu_benchmark --example cpu_qualify
source ../env.sh
target/release/examples/cpu_qualify
cargo build --release --locked --features profiling --example cpu_profile
```

Frozen final executables and drivers are in `target/arm64-recognition-sme-final-20261017`; prototype records are in `target/arm64-recognition-kernel-20261017`; full correctness evidence is in `target/qualification/1792244044180-69411`. Reproduction uses a fresh directory at the same depth and the hash-checked jobs from the preceding analysis. The portable JSON embeds the drivers, patch, new files and provenance. Keep standard and profiling executables separate and do not overwrite existing measurements.

Timing scope is one physical M4, three representative inputs and one thread/concurrency configuration. Unsupported hardware/state uses the established fallback, but other SME processors, vector lengths, native x86 performance and concurrent throughput were not measured. The full correctness corpus establishes compatibility with the pinned reference, not ground-truth OCR accuracy.
