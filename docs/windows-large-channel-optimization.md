# Large-channel recognition optimization experiments

Implemented selective automatic recognition ownership: use sixteen recognition owners for dense pages on Windows x64 AVX2/FMA contexts with 24–31 logical CPUs and at least 32 recognition units. The previous policy used eight owners on these pages. Final automatic-policy measurements reduced median complete-page latency by **18.7% Tiny, 17.8% Small and 12.7% Medium** on the local 24-thread context. Alternative register tiles and fused-store specialization did not establish a broad improvement and were rejected.

The production change is confined to [`CpuBudgets::for_recognition_units`](../src/cpu.rs). Convolution arithmetic, SIMD dispatch, model settings, classification, batching defaults and explicit worker limits are unchanged. The final release library suite passed **327 tests, with 12 diagnostics ignored**.

The [raw evidence](reports/windows-x64-2026-10-17-large-channel-optimization.json) preserves all timing samples, CPU times, retained-memory figures, executable/source hashes, the rejected store patch and portable standalone tile sources. It does not rely solely on machine-local `target` files. The earlier [implementation analysis](windows-large-channel-convolution-analysis.md) explains the full-K reduction, packed layout and prior failed blocking/grouping experiments.

## Scope and selection

All measurements ran locally on the physical Windows machine reporting an AMD Ryzen 9 5950X, 12 cores and 24 logical processors. Rust used the pinned `nightly-2026-09-08` toolchain, release optimization, default parallel features and no profiling for page tests. Real-image comparisons use the same predecoded 430×267 dense fixture, `imgs_00018069`, the pinned model/dictionary assets and unchanged classification, preprocessing and adaptive-width settings. Each model detects 69 lines in the owner comparison.

The selected policy changes stable recognition-job ownership, without altering convolution arithmetic. At capacity 24, eight owners receive up to three nested kernel threads each. Sixteen owners receive one base thread each, reducing nested kernel scheduling and increasing independent line work. Those are plausible mechanisms; hardware counters and a full per-stage causal profile were not captured. Sixteen owners do not guarantee all 24 pool threads are busy at every moment.

The predicate is limited to automatic Windows x64 AVX2/FMA execution, context capacity 24–31, and at least 32 recognition units. The threshold requires at least two jobs per owner to amortize the extra workers; performance was measured at 69 units, not at every qualifying count. Other eligible pages retain the eight-owner expansion; ineligible pages retain their prior policy. Explicit owner overrides remain explicit. At capacities 25–31 the sixteen-owner policy still gives one base thread per owner; their inclusion is an inference from identical budget arithmetic, with boundary correctness checked but performance measured only at capacity 24. Contexts with 32 or more threads retain their existing policy. No result qualifies a performance change on AVX-512, NEON, macOS, sparse pages or concurrent requests.

## Final automatic-policy comparison

Both versions use automatic line scheduling, the same CPU24 context and `ffffff` affinity, sixteen classifiers, the same input/model assets and settings, and complete owned-result timing. The baseline executable was preserved before the production edit. The final executable was rebuilt after the CPU policy and integration checks were complete. Each automatic case has one warm-up and three measured calls; source and executable hashes were checked before and after each process. Models ran in order Medium, Small, Tiny for both versions.

| Model | Baseline automatic samples, ms | Final automatic samples, ms | Median before→after, ms | Lower latency |
|---|---|---|---|---:|
| Tiny | 57.0082, 55.8380, 57.1439 | 46.3330, 45.8605, 47.0898 | 57.0082→46.3330 | 18.73% |
| Small | 149.7829, 143.7806, 157.8740 | 123.1267, 123.9859, 121.9956 | 149.7829→123.1267 | 17.80% |
| Medium | 493.8240, 492.3828, 457.6391 | 416.5237, 438.9798, 429.9087 | 492.3828→429.9087 | 12.69% |

All three observed ranges are separate. These are short sequential measurements on one host and one dense fixture, not confidence intervals or a general corpus/throughput result. The larger explicit-setting gains below are separate selection evidence and are not substituted for this final comparison.

The final probe also runs explicit eight owners after automatic execution as a correctness control. Complete `OwnedOcr` field values match across both configurations in every warm-up and measured call; capture is disabled, so CTC spans are absent. This control changes classifier ownership from sixteen to eight and therefore cannot isolate recognition performance. The performance table compares automatic with automatic, where classification remains sixteen in both versions. The old owner probe did not export its control output or digest, so there is no independently serialized complete-output comparison across the two executables. The final probe now exports a control digest for future comparisons.

Final automatic retained activation and workspace capacities exactly match the prior explicit-sixteen setting. Additional retained request capacity is **6.76 / 20.72 / 23.62 MiB** for Tiny / Small / Medium, respectively; the table below gives both totals. Compiled weights and dictionary remain shared. Activation limits remain per session; there is no new aggregate memory cap or automatic owner-reduction fallback on allocation failure. A larger warmed request can exceed an engine's configured `max_idle_bytes` and be evicted when its lease returns. Explicit `line_workers: 8` remains available for memory-sensitive deployments.

## Baseline owner comparison

The baseline executable compares automatic, twelve and sixteen owners in that fixed order, with one shared `CpuContext::new(24)`, shared immutable models and a fresh mutable request per setting. Affinity is the first 24 allowed logical processors, `ffffff` on this host. Each setting receives one warm-up and three complete owned-result measurements. The table reports the median of the three samples; it does not pool models or configurations.

| Model | Automatic eight owners, ms | Twelve owners, ms | Sixteen owners, ms | Reduction, automatic→sixteen |
|---|---:|---:|---:|---:|
| Tiny | 57.0082 | 53.2611 | 43.7437 | 23.27% |
| Small | 149.7829 | 121.8982 | 114.7313 | 23.40% |
| Medium | 492.3828 | 447.3677 | 454.8122 | 7.63% |

All warm-up and measured outputs compare exactly as complete `OwnedOcr` values within each model's process, including ratios and line scores, with `capture=false`. CTC spans are absent in this measurement mode. No text or quality tradeoff was introduced in these comparisons.

Sixteen owners improves all three sampled model families and has the lowest Tiny and Small medians. Twelve owners is 1.64% faster than sixteen on the Medium median, but also changes classifier ownership from sixteen to twelve. Recognition's nested thread budget changes from three under automatic execution to two with twelve owners and one with sixteen. The twelve-owner result therefore cannot isolate a recognition-only effect. The sixteen-owner comparison keeps classification at sixteen workers and supports the broader candidate selection. It does not prove sixteen is the universal optimum.

Three samples are not a confidence interval. The fixed order can expose later configurations to different cache, frequency, temperature or allocator state. In particular, Medium automatic samples range from 457.6391 to 493.8240 ms, while sixteen-owner samples range from 452.7390 to 462.5644 ms; those ranges overlap. The gain needs a final automatic-policy comparison rather than another unchanged-code run seeking cleaner numbers.

More owners retain more request-private buffers:

| Model | Automatic activation + workspace, MiB | Sixteen-owner activation + workspace, MiB | Increase, MiB |
|---|---:|---:|---:|
| Tiny | 24.86 | 31.62 | 6.76 |
| Small | 46.04 | 66.75 | 20.72 |
| Medium | 67.83 | 91.45 | 23.62 |

These sums exclude immutable model weights and do not add kernel scratch again: scratch is a subset of workspace. Process `PeakPagefileUsage` is also recorded, but it is Windows peak private commit accumulated across configurations in that process. It is neither resident memory nor an isolated per-setting measurement. Use the retained-buffer counters for the direct comparison; final automatic counters equal the explicit-sixteen column.

## Rejected register tiles

A standalone native probe compared the existing six-row by sixteen-output-channel tile with eight-by-eight and four-by-twenty-four alternatives. Packing, native bias seeding and ascending-K FMA order were preserved. The eight-by-eight implementation handles each eight-column half of the packed sixteen-column panel separately. The four-by-twenty-four implementation uses three eight-lane accumulators per row.

The probe uses one allowed logical CPU, synthetic tensors and an identity final store. Each shape/variant receives one warm-up and three single-call measurements, in orders `[0,1,2]`, `[2,1,0]`, `[1,0,2]`. All measured outputs are bit-identical across variants. Its supplied correctness mode covers six geometries, bias present/absent, global fused-store indices, tails and guard sentinels.

| M,K,N | Existing median, ms | Eight-by-eight median, ms | Four-by-twenty-four median, ms |
|---|---:|---:|---:|
| 96,384,768 | 0.4258 | 0.5112 | 0.4670 |
| 960,512,1024 | 7.5793 | 9.3147 | 7.5717 |
| 480,1536,768 | 8.2977 | 9.9520 | 8.2619 |

Eight-by-eight regresses the medians by 19.94–22.90%. Four-by-twenty-four regresses the first shape by 9.68%; its approximately 0.10% and 0.43% lower medians on the larger shapes have overlapping ranges. Neither establishes a broad benefit. The existing production tile remains preferable on this evidence. This question was answered by the focused kernel probe, so those tile candidates were not promoted into further page runs.

The raw report embeds a portable source bundle and exact synthetic-data generators. Extract that bundle into one directory, compile `probe.rs` with `rustc --edition=2024 -O`, run its `check` mode, then run its `bench` mode under a ten-second external timeout. Only the three module paths were normalized when preserving those sources.

## Rejected fused-store specialization

The store candidate selected common channel Add, channel Add plus GELU, and channel Add plus full residual recipes once per six-row tile. It retained the full reduction and graph operation order, attempting to reduce repeated recipe selection and channel-operand loads during final stores.

Page tests used eight logical CPUs with affinity `ff`, independent baseline/candidate processes, and order Tiny AB, Small BA, Medium AB. Each process records one first inference, one warm-up and three measured owned-result calls.

| Model | Baseline median, ms | Store candidate median, ms | Candidate latency change |
|---|---:|---:|---:|
| Tiny | 72.0617 | 70.8020 | −1.75% |
| Small | 199.9039 | 198.8774 | −0.51% |
| Medium | 743.5534 | 755.1439 | +1.56% |

Tiny and Small sample ranges overlap. Medium regresses with nonoverlapping observed ranges. Retained activation/workspace diagnostics remain equal, and all serialized `CaseResult` fields compare exactly across versions. The worker also checks complete `OwnedOcr` stability internally, but its exported cross-version result omits ratios and optional spans. This is not a cross-version complete-owned-result proof. The store candidate was rejected and its production changes were removed.

Eight-CPU store results and 24-CPU owner results are different configurations and source versions. Their absolute latencies must not be combined to calculate a candidate speedup.

## Execution budget and validation

The standalone tile comparison took 0.2927207 seconds, covering three shapes, three variants, nine warm-ups and 27 measured kernel calls. The store page comparison used 6.0308965 seconds for baseline and 6.0480577 seconds for the rejected candidate, including worker startup, model loading, first inference and warm-up. These are separate source-version measurements.

The three baseline ownership processes took 9.3070598 seconds total, covering nine configuration warm-ups and 27 measured owned-result calls. Internal ownership deadlines are fifteen seconds, with bounded external process deadlines of at most sixteen seconds. The latest conservative baseline checkpoint is 35.8 seconds before the final Tiny owner process and 36.7556981 seconds after it. Earlier checkpoint values are cumulative; adding them together would double-count execution. Builds and correctness runs are excluded from performance execution accounting. The default limit is sixty seconds per code version.

Final-version performance execution took **6.9306653 seconds** across three model processes, including startup/loading and six configuration warm-ups plus eighteen measured calls. There was no additional performance sampling after these questions were answered. Every version remained below sixty seconds, excluding builds and correctness checks. There were no CI, remote or virtual-machine runs.

Focused checks ran first: fourteen CPU-policy tests and the new Windows AVX2 36-line OCR integration test passed. The integration test compares complete captured output to serial execution, exercises sixteen-owner warm storage under waiting demand, dense→sparse→dense reuse, same-width batching that reduces thirty-six lines to eighteen units and returns to eight owners, and result-budget failure followed by successful reuse with retained capacities unchanged. The full release library suite then passed 327 tests, with twelve timing/diagnostic tests ignored, in 3.83 seconds. Formatting of the affected Rust files and `git diff --check` passed.

```powershell
cargo test --release --locked --lib cpu::tests -- --test-threads=1
cargo test --release --locked --lib automatic_sixteen_recognition_owners -- --test-threads=1
cargo test --release --locked --lib -- --test-threads=1
cargo build --release --locked --example cpu_owner_probe
# Each process: fifteen-second internal and sixteen-second external deadline.
# Use the three pinned models, one fixture, one warm-up and three calls per setting.
target/release/examples/cpu_owner_probe.exe medium imgs_00018069 '0,8'
```

The raw report preserves final source/executable hashes, every sample, input identities, CPU and retained-memory counters, validation commands, and process execution accounting. The local bounded runner is `target/large-channel-adoption-20261017/run-owners.ps1`; the report contains the evidence needed after that temporary directory is removed.
