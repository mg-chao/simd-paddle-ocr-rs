# Apple M4 recognition scheduling optimization

Automatic recognition scheduling now uses eight line workers on the NEON
backend when the CPU context has at least eight workers and the page provides
at least eight independent recognition units. This reduced measured dense-page
latency by **12.7–14.6%** against Rust commit
`f598697bed49295579d1f15d3a9f289bbaf02435` on the physical Apple M4.
The [portable evidence record](reports/macos-arm64-2026-10-17-recognition-scheduling.json)
contains raw samples, CPU times, memory, source and executable fingerprints,
process order, validation results and local artifact hashes.

This is a latency improvement for the measured dense input. Concurrent
throughput remains inconclusive, and Tiny sparse latency measured **4.8%
slower** despite keeping its existing worker selection. Additional recognition
buffers increased peak resident memory by up to **19.0%** in this selection.
These results do not establish a universal speedup or a new C# performance
qualification.

## Retained change

Previously, an eight-worker context used at most four automatic recognition
owners, usually giving each recognizer two nested kernel workers. The measured
dense pages benefit from distributing independent lines across eight owners
with one kernel worker each. The existing shared pool still caps aggregate
workers across simultaneous requests.

The policy in `src/cpu.rs` is selected after recognition batching is counted in
`src/ocr.rs`. A same-width batch is one unit. Selection depends on page geometry,
backend and configuration, so concurrent request timing does not reassign line
ownership. Existing demand tracking can still reduce nested kernel work.
Each OCR run resolves its base budgets before selecting recognition workers;
a subsequent sparse page returns to the base policy. Worker buffers remain
reusable subject to the existing retention policy.

Explicit positive `CpuOptions.line_workers` values continue to control both
classification and recognition. `line_workers: 4` selects the earlier
recognition limit when memory or a particular workload favors it. Other
backends, serial contexts and pages with fewer than eight units retain their
existing selection. `Ocr::cpu_budgets()` reports the selected budgets after a
run. Classification scheduling is unchanged by this patch.

## Final measurements

Physical Apple M4, 10 cores (four performance and six efficiency), 16 GiB,
macOS 27.0.1; native standard release builds, default features, automatic
SIMD/layout, fusion enabled, no custom Rust flags. Each case uses a shared
eight-worker context, identical pinned models and prepared input, one first
batch, one warm-up and three measured batches. Each version runs in a fresh
process per case; process order alternates across cases. CPU affinity is
unbound. The final comparison contains only the retained scheduling change.

Single-request medians in milliseconds; positive reduction means lower latency.
Dense input is `imgs_00018069`; sparse input is `imgs_1`.

| Model / input | Baseline | Retained | Latency reduction | Sample ranges overlap |
| --- | ---: | ---: | ---: | --- |
| Tiny dense | 75.319 | 64.337 | 14.6% | No |
| Small dense | 228.309 | 194.961 | 14.6% | No |
| Medium dense | 766.678 | 669.337 | 12.7% | No |
| Tiny sparse | 46.968 | 49.243 | -4.8% | No |

Tiny sparse has only three recognition units and does not select eight owners.
Its retained activation/workspace counts are identical. The regression's cause
is not established. First-batch/warm-up timings were also higher for that
candidate process (166.975/90.990 ms versus 53.704/47.005 ms); the bounded run
does not establish steady state or cold-start equivalence. No unchanged-code
reruns were made to obtain more favorable numbers.

Two concurrent requests share the same eight-worker context. These are median
batch times for two dense pages, not per-request latency:

| Model | Baseline batch (ms) | Retained batch (ms) | Sample ranges overlap |
| --- | ---: | ---: | --- |
| Tiny | 125.938 | 118.359 | Yes |
| Small | 377.332 | 383.783 | Yes |
| Medium | 1322.513 | 1332.217 | Yes |

All three concurrent sample ranges overlap; this selection does not establish
a throughput improvement. Peak resident memory increased from
40.42/103.33/335.98 MiB to 45.06/115.47/351.48 MiB for the respective
single-request dense cases. The largest relative increase was Small with two
requests: 129.16 to 153.73 MiB, or 19.0%. More recognition owners retain more
private activation and workspace buffers.

Complete owned OCR outputs matched exactly between baseline and retained Rust
in all seven cases. The selection comprises seven first batches, seven
warm-ups and 21 measured batches per version: 30 measured requests, or 50
requests including first and warm-up batches. Three samples and disjoint
observed ranges are not a statistical significance test.

Final worker execution took 15.389 seconds for the baseline and 15.477 seconds
for retained Rust, including startup and model loading. The unchanged baseline
was also used in earlier investigations; its aggregate 67.155 seconds stayed
within an explicitly expanded 75-second bound. The expansion isolated the
retained scheduling change from a rejected kernel candidate and checked dense
latency, concurrent requests and sparse Tiny. Sampling remained fixed, with no
retries or automatic growth. Retained Rust stayed below the default 60-second
performance budget. Correctness checks and builds are separate from these
performance measurements.

## Alternatives evaluated

- Pointwise cache groups of 48, 96 and 192 rows produced overlapping kernel
  timing ranges on representative shapes. The existing group size remains.
- Four-row versus six-row pointwise tiles did not show a consistent full-tile
  benefit.
- Exact-size pointwise tails improved the four-row, 512-input, 1024-output
  diagnostic from 0.054833 to 0.035833 ms. However, all twelve full OCR
  before/after sample ranges overlapped. Those kernel edits were removed.
- A paired recognition scheduling diagnostic showed that eight owners helped
  dense Tiny and Small, motivating the retained policy. A combined
  scheduling-plus-tail build was measured separately before removing the
  unproven kernel change.

The evidence record keeps each experiment's source and results separate.
Earlier mixed-width measurements included the rejected kernel and do not
validate the final source. No new pointwise kernel, unsafe operation, model,
accuracy tolerance or dependency change is retained.

## Validation

All checks ran locally on the physical machine with the final production
source:

- 334 default-feature library tests, 29 integration tests and five doctests
  passed (368 total); twelve library diagnostics and one explicit local
  integration test remained ignored by the normal suite.
- 324 library tests passed without default features; twelve diagnostics were
  ignored.
- The live C# qualification passed all 213 requests over all three models and
  22 fixtures, including serial/parallel settings, same-width batching,
  alignment, rotation, padded pixels, fixed widths, classification off, reuse
  and recovery. Zero differences were reported under the existing contract.
- Native all-target/all-feature compilation, x86_64 macOS cross-compilation,
  Clippy, formatting and whitespace checks passed. Clippy retained existing
  architecture-specific unused/dead-code and harness-style warnings.

The C# checkout remains clean at
`33ff8a81672452041d110e8ba52dcd2639e54228`. Toolchains were
`nightly-2026-09-08` (LLVM 23.1.1), .NET SDK 10.0.303 and runtime 10.0.11.
Qualification artifacts are under
`target/qualification/1792215985444-57043`.

Final benchmark fingerprints (the source manifest excludes documentation):

```text
Baseline source: 6b29c6f901b0082c8101d8779bb7c4e505c815ceaef6ca0374e3cd5e5c99740e
Retained source: ef7825aebf3e4ff8221a0edaadb6904af21779aa4e3c656ed6f1e1188600696f
Baseline binary: 182207b0e6633ab3df64a8ed71f85945d462e6c698a88b1daeed2df97116d624
Retained binary: 7ed87f25cfff28f2211dd67225655c11212f15f05110d7d7884a207bf847063e
```

## Reproduction and scope

The final comparison driver, frozen baseline binary/archive, jobs and raw worker
results are under `target/arm64-pointwise-optimization-20261017`. Run
`compare-scheduling.py` from the repository root with the local environment
loaded and a newly built release `cpu_benchmark`. Its seven fixed cases and
source manifests correspond to `scheduling-only-comparison/report.json`.
The portable evidence file preserves samples and fingerprints independently of
that ignored local directory.

With those local baseline artifacts present, a fresh comparison directory can
be produced with:

```sh
python3 target/arm64-pointwise-optimization-20261017/compare-scheduling.py \
  --candidate target/release/examples/cpu_benchmark \
  --candidate-source . --label scheduling-reproduction
```

Build and correctness commands:

```sh
cargo build --release --locked --example cpu_benchmark --example cpu_qualify
cargo test --release --locked
cargo test --release --locked --no-default-features --lib
cargo check --release --locked --all-targets --all-features
cargo check --release --locked --all-targets --all-features --target x86_64-apple-darwin
cargo clippy --release --locked --all-targets --all-features
cargo fmt --all -- --check
target/release/examples/cpu_qualify
git diff --check
```

Performance was measured only on this M4 with an eight-worker context and the
inputs above. Other ARM CPUs, other capacities, final-source mixed-width pages
and native x86 performance were not measured. x86 uses its existing scheduling
policy. The live C# run checks correctness; historical C# timings in the
[earlier ARM64 report](macos-arm64-optimization.md) are not pooled with this
Rust-to-Rust comparison.
