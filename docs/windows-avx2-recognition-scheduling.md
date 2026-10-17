# Windows AVX2/FMA recognition scheduling

Optimization 1 extends the existing automatic eight-worker recognition policy
to AVX2/FMA. On the physical Windows host, single-request dense-page latency
fell by **18.2% for Tiny, 12.0% for Small, and 7.6% for Medium** against Rust
commit `4c88abc2daaff3e93867c0fe7e099a304dafa039`. Tiny mixed-width latency fell
10.3%. All seven baseline/candidate serialized OCR outputs matched exactly.
The [evidence record](reports/windows-x64-2026-10-17-recognition-scheduling.json)
preserves samples, CPU times, memory, input identities, process order, validation
results, source fingerprints and executable hashes.

## Implementation

`CpuBudgets::for_recognition_units` in [`cpu.rs`](../src/cpu.rs) selects eight
recognition owners when all of these conditions hold:

- `CpuOptions.line_workers == 0` (automatic selection).
- The compiled recognizer uses AVX2/FMA or native NEON.
- The shared CPU context has capacity of at least eight.
- The page has at least eight independent recognition units after batching.

[`ocr.rs`](../src/ocr.rs) supplies the actual compiled backend after counting
units. An exact-width batch counts as one unit, regardless of its crop count.
At context capacity eight, this changes four owners with two nested kernel
workers into eight owners with one kernel worker each. The existing fixed CPU
pool still limits aggregate execution across simultaneous requests.

Selection depends on geometry and configuration, so transient concurrent demand
does not reassign lines or invalidate warmed worker caches. Demand can still
reduce nested parallelism. Each run resolves its base budgets before selecting
the recognition policy; a subsequent sparse page returns to the base selection.
Inactive buffers remain reusable under the existing request-retention policy.

This reuses the existing NEON policy and job assignment instead of adding an
extra scheduling mechanism. Classification, kernels, model arithmetic, crop
ordering and public APIs are unchanged. Positive worker limits still take
precedence. `line_workers: 4` is available when a workload favors fewer retained
buffers, but also caps classification at four. `Ocr::cpu_budgets()` reports the
selected budgets after inference; the benchmark's startup log prints the base
budgets before that selection occurs.

## Local measurements

2026-10-17, physical Windows x64, AMD Ryzen 9 5950X (Windows reports 12 enabled
cores / 24 logical processors), approximately 64 GiB RAM. Both revisions used
the same eight logical CPUs, affinity `0xff`; this is not a claim of eight
distinct physical cores. Standard release/default-feature builds used the
pinned Rust nightly, automatic SIMD/layout, fusion, adaptive recognition widths,
batch size one, classification enabled and alignment capture disabled.

Each case ran in a fresh process per revision, alternating baseline/candidate
order across cases. Model/manifest and predecoded input hashes were checked.
One first batch and one warm-up preceded exactly three measured batches. The
baseline executable was preserved before editing production code; the candidate
was rebuilt after the changes. Loading and input decoding are outside inference
timing; producing owned results is included. Prior analysis timings are not
pooled with this comparison.

Single-request median milliseconds; positive reduction means lower latency:

| Model / input | Baseline | Candidate | Reduction | Sample ranges overlap |
| --- | ---: | ---: | ---: | --- |
| Tiny dense, 69 lines | 83.666 | 68.428 | 18.2% | No |
| Small dense, 69 lines | 223.610 | 196.841 | 12.0% | No |
| Medium dense, 69 lines | 836.890 | 773.254 | 7.6% | No |
| Tiny mixed widths, 31 lines | 90.409 | 81.067 | 10.3% | No |
| Tiny sparse, 3 lines | 52.378 | 44.199 | 15.6% observed | No |

Dense is `imgs_00018069`, mixed is `imgs_00006737`, and sparse is `imgs_1`.
Sparse uses the unchanged three-active-owner selection, with identical retained
activation/workspace counts. Its observed improvement is **not attributed to
this scheduling change**. Medium baseline samples drifted upward
(800.386, 836.890, 906.724 ms); candidate samples were 755.990, 773.254 and
778.861 ms. Three samples and disjoint ranges do not establish statistical
significance, steady state, or a universal speedup.

Two concurrent requests shared the same eight-worker context. These are
two-image batch times, not individual-request latency:

| Model / dense input | Baseline batch (ms) | Candidate batch (ms) | Reduction | Sample ranges overlap |
| --- | ---: | ---: | ---: | --- |
| Tiny | 155.169 | 155.432 | -0.2% | Yes |
| Medium | 1556.988 | 1530.312 | 1.7% | No |

Tiny concurrent throughput is inconclusive. Medium's observed improvement is
small; this change primarily improves the measured single-request cases.

## Memory tradeoff

More recognition owners retain more private activations and workspace. Shared
normalized weights and cached packed weights remained identical between
revisions for every case.

| Model / input / requests | Baseline peak commit (MiB) | Candidate peak commit (MiB) | Increase |
| --- | ---: | ---: | ---: |
| Tiny dense / 1 | 36.14 | 40.28 | 11.5% |
| Small dense / 1 | 96.73 | 110.13 | 13.9% |
| Medium dense / 1 | 320.50 | 336.53 | 5.0% |
| Tiny mixed / 1 | 67.71 | 72.18 | 6.6% |
| Tiny dense / 2 | 50.56 | 60.08 | 18.8% |
| Medium dense / 2 | 365.87 | 397.82 | 8.7% |

For single dense requests, retained activation plus workspace capacity increased
by 4.14 MiB (Tiny), 12.36 MiB (Small), and 15.68 MiB (Medium). These counters
exclude shared weights and are distinct from process peak private commit.
Windows peak commit includes initialization, loading and first-use allocations;
it does not measure working set. Sparse peak commit was 64.66 versus 64.74 MiB.

## Validation and scope

Before benchmarking, 34 focused release tests passed across CPU scheduling,
OCR and engine modules. The expanded policy test covers eligible backends,
capacity/unit boundaries, explicit caps and concurrent CPU shares. A new OCR
integration test checks dense output equality including alignment capture,
stable ownership and buffer reuse across demand changes, dense-to-sparse
transitions, batch counting and an explicit worker override. Existing tests
cover shared-pool caps, concurrent requests, recovery and result ordering.

The build without default features and focused formatting checks passed. The
existing Windows `UNBOUND_AFFINITY` unused-import warning remains unrelated.
Each real-model worker checked cold, warm-up and measured owned results for
equality, including both concurrent requests. All seven cross-revision
serialized outputs matched exactly: text, scores, coordinates, classifications,
rotation, counts and recognition dimensions.

The seven cases used three distinct inputs and one CPU/affinity configuration:
seven first batches, seven warm-ups and 21 measured batches per revision,
equivalent to 27 measured images or 45 images including first/warm-up batches.
Execution took **16.285 seconds baseline and 15.619 seconds candidate**,
including process startup and model loading, excluding builds. Each process
had a 15-second timeout within a 60-second per-version budget. The baseline's
earlier original-harness execution was 7.16 seconds; combined use remains below
60 seconds. There were no retries or automatic sample increases.

This is a focused Rust-versus-Rust comparison on one physical AVX2/FMA host.
Other CPU capacities, ARM64/AVX-512 hardware, the full accuracy corpus and the
36-cell Rust/C# qualification were not run. Their performance is not established
by these samples.

Validation commands:

```powershell
cargo test --release --locked --lib -- cpu::tests ocr::tests engine::tests --test-threads=1
cargo check --locked --no-default-features --lib
rustfmt --check --edition 2024 --config skip_children=true src/cpu.rs src/ocr.rs examples/cpu_benchmark.rs
cargo build --release --locked --example cpu_benchmark
```

Local raw artifacts and the fixed-scope runner are under
`target/recognition-scheduling-20261017/`. The runner invokes each preserved
revision's `cpu_benchmark.exe _worker <job.json>` sequentially with the same job,
then checks output equality. Immutable jobs and predecoded inputs originated
from the earlier local benchmark. The evidence record preserves their paths,
hashes and settings; paths must be remapped when reproducing on another machine.
