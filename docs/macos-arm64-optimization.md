# Apple M4 ARM64 optimization results

This record covers the ARM64 optimization after the
[original Portable4 comparison](macos-cpu-performance.md). Measurements use the
physical Apple M4, macOS 27.0.1, native release builds, and the unchanged pinned
C# reference, models, dictionary, corpus and accuracy contract. The
[machine-readable evidence](reports/macos-arm64-2026-10-17-arm64-optimization.json)
keeps each source version separate and preserves raw wall/CPU samples, memory,
process order, fingerprints, commands and local artifact paths.

The final Tiny-model warmed latency is **18.5% lower on sparse input,
32.8% lower on dense short lines, and 27.7% lower on mixed widths**
than the fresh clean Portable4 baseline measured in this session. Baseline Rust
medians were 64.581, 127.197 and 121.996 ms, respectively. Small and Medium have
no before/after pipeline baseline here; their rows compare the final Rust
version directly with C#.

The final fifteen-cell selection produced **10 passed, 3 inconclusive and
2 failed** performance cells. All fifteen output comparisons and resident-memory
gates passed. All nine eight-worker latency medians met the 110%-of-C#
threshold; Medium sparse and mixed-width sample ranges crossed that threshold.
Tiny and Medium concurrent throughput failed the 90%-of-C# gate; Small
throughput remained inconclusive. No performance parity qualification is claimed.

### Eight-worker single-request latency

Medians of three fixed samples, in milliseconds. `Rust/C#` is wall latency;
lower values are faster. Sparse/dense/mixed correspond to `imgs_1`,
`imgs_00018069` and `imgs_00006737`.

| Model | Input | Rust (ms) | C# (ms) | Rust/C# | Gate |
| --- | --- | ---: | ---: | ---: | --- |
| tiny | sparse | 52.661 | 52.038 | 1.012× | passed |
| tiny | dense | 85.421 | 98.907 | 0.864× | passed |
| tiny | mixed | 88.259 | 89.693 | 0.984× | passed |
| small | sparse | 122.817 | 117.793 | 1.043× | passed |
| small | dense | 230.934 | 242.696 | 0.952× | passed |
| small | mixed | 278.053 | 272.184 | 1.022× | passed |
| medium | sparse | 376.478 | 356.190 | 1.057× | inconclusive |
| medium | dense | 760.418 | 713.848 | 1.065× | passed |
| medium | mixed | 990.273 | 903.432 | 1.096× | inconclusive |

### Dense-input serial latency and concurrent throughput

Throughput is `2000 / median_batch_ms`: two simultaneous requests per batch.
The throughput ratio is Rust images/second divided by C# images/second.

| Mode | Model | Rust | C# | Ratio | Gate |
| --- | --- | ---: | ---: | ---: | --- |
| 1 worker, 1 request | tiny | 268.931 ms | 264.493 ms | 1.017× | passed |
| 1 worker, 1 request | small | 868.347 ms | 867.876 ms | 1.001× | passed |
| 1 worker, 1 request | medium | 3104.193 ms | 2943.384 ms | 1.055× | passed |
| 8 workers, 2 requests | tiny | 14.126 img/s | 16.651 img/s | 0.848× | failed |
| 8 workers, 2 requests | small | 5.358 img/s | 5.800 img/s | 0.924× | inconclusive |
| 8 workers, 2 requests | medium | 1.529 img/s | 1.780 img/s | 0.859× | failed |

Rust peak resident memory was 34.8–88.1% of C# across
these cells. The throughput gaps remain material; the concurrency-policy
limitation below prevents assigning them solely to kernel performance.

Final worker execution consumed **46.629 s Rust / 48.776 s C#**, including
startup, model loading, first inference, warm-up and fixed samples, within an
aggregate sixty-second budget per implementation. Budget accounting also
reserved 0.010 s for the final Rust kernel diagnostic, for a 46.639 s Rust total.
Across fifteen fresh workers
per language this was fifteen first batches, fifteen warm-ups and forty-five
measured batches: fifty-four measured OCR requests, or ninety total OCR requests
including first and warm-up batches per language.

All three scopes used one source and executable fingerprint:

```text
Rust source: 2dbd1392c8a6ab9b877dc64e7b50b1827ef27edd46b78932e594d1904b4b8797
Executable:  c28c6b2a27caef0ad562cd5ecca8a77d35400ff20f35d6d653cad48bdc2b1553
```

Source hashes cover production/benchmark sources and build configuration; docs
are excluded. The archived report retains each raw source-version fingerprint;
intermediate measurements are not pooled with the final source.

## Changes and causes

1. **Native ARM64 dot-product kernels.** Automatic dispatch previously selected
   Portable4, which uses separate multiply/add and generic tiles. It now selects
   `NeonFma`: six-row by sixteen-channel pointwise/dense convolution,
   four-row by sixteen-column matrix multiplication and compact CTC projection,
   and two-pixel by thirty-two-channel packed depthwise convolution. The
   register tiles reuse input and weight loads and keep independent ordered FMA
   chains. Compact projection bounds row and weight panels and retains the
   existing CTC path that avoids a full logits tensor. Existing fused
   activation/residual stores remain in use.
2. **Bounded partial matrix loads.** A variable-length slice copy generated
   repeated `memcpy` calls within attention's K loop, with call-clobbered vector
   registers saved to the stack. Explicit loads for zero through four valid
   lanes remove those calls while preserving lane bits and zero padding.
3. **Independent automatic classification capacity.** The staged classifier
   previously inherited the four recognition owners and left half an eight-CPU
   request budget unused. Automatic classifiers can now use up to the current
   request's share, capped at sixteen, while recognition ownership stays stable.
   Explicit positive `line_workers` still limits both stages. Classification/rotation
   finishes before recognition; the shared pool still caps aggregate workers.
4. **Initialized activation high-water span.** Narrowing a cached recognition
   shape shortened the arena's initialized length. Growing back within retained
   capacity then cleared the same tail again. Reshaping now retains that
   initialized span and exposes only the active plan's slices. No uninitialized
   allocation or new unsafe memory policy is involved; retained-byte limits and
   allocation replacement when limits tighten remain enforced.

The NEON backend is AArch64-gated and rejects forced use on unsupported CPUs.
Canonical NCHW specializations and activation programs keep their established
Portable4 arithmetic where selected, and diagnostics report the actual choice.
Bias placement, reduction order, scalar-tail rules, stable argmax ties and
nonfinite checks are preserved. FMA intentionally changes rounding relative to
Portable4, so OCR comparisons use the existing numerical tolerances.

## Focused evidence for the causes

These probes belong to the explicitly recorded intermediate/control versions;
they are separate from final pipeline qualification. Each kernel shape/policy uses one
warm-up and three single-call samples. Timings are medians.

| Probe | Control | Candidate | Observation |
| --- | ---: | ---: | --- |
| Pointwise, rows/K/channels = 96/384/768 | 0.954834 ms Portable4 | 0.490917 ms NEON | 1.94× faster |
| Pointwise, 36480/96/24 | 2.721666 ms Portable4 | 1.963166 ms NEON | 1.39× faster |
| Unpacked matrix, rows/K/columns = 32/64/8 | 0.007458 ms copied tails | 0.001458 ms bounded loads | 5.12× faster |
| Unpacked matrix, 64/64/63 | 0.041083 ms copied tails | 0.008708 ms bounded loads | 4.72× faster |
| Cached arena cycle, widths 320/32/320/32 | 19,959 ns | 334 ns | Avoided repeated clearing |

The arena probe binds a real TinyREC graph prefix, performs zero inference
runs, and measures three cached four-transition cycles after one warm-up cycle.
Retained allocation stays at 737,280 bytes; each wide transition previously
cleared 663,552 initialized bytes. The absolute saving is about **0.020 ms per
four transitions**. Timer granularity is about 42 ns; this is not a claim of a
comparable OCR speedup.

A separate one-warm-up/one-sample Tiny/dense stage profile changed the explicit
four-classifier limit to automatic capacity, keeping four recognition owners.
Classification wall time fell from 24.811 to 18.582 ms; recognition wall time
was 46.203 versus 46.600 ms. Classifier retained activations rose by 1,024,000
bytes. Summed worker elapsed time increased. These samples show shorter measured
stage latency; they do not establish reduced process CPU time. Intermediate C#
dense timings also drifted between
versions; do not attribute every pipeline timing difference to one edit.

Release assembly inspection confirms no calls in matrix K loops and no vector
spills in complete matrix/compact-projection K loops or the reviewed pointwise
and dense convolution loops. Partial four-row matrix tiles still spill four
accumulators. The general dense fallback for kernels above 49 taps retains two
vector spills; shipped
production convolutions use the specialized paths reviewed here.

## Correctness and portability

The final numerical source passed the following local checks:

- 329 default-feature library tests, 29 integration tests and five doctests
  (363 total); nine performance diagnostics and one explicit local integration
  test remained ignored by the normal suite.
- 319 serial library tests without default features, and 330 profiling-enabled
  library tests. Focused kernel/arena diagnostics were run separately as recorded
  above, with fixed counts and external deadlines.
- All 66 model/fixture combinations against the unchanged saved expectations,
  using serial execution.
- All 213 live C# configuration requests, including all three models and 22
  fixtures plus serial/parallel budgets, alignment, batching, rotations, padded
  pixels, fixed widths, classification off, reuse and recovery; zero differences.
- Native all-target/all-feature compilation and x86_64 macOS cross-compilation,
  Clippy, formatting and whitespace checks. Clippy succeeded with existing
  architecture-specific unused/dead-code and harness-style warnings.

The reference checkout stayed clean at
`33ff8a81672452041d110e8ba52dcd2639e54228`; the Rust base is
`af633772eb116789b6af99591890f11d9aa613fd` plus the recorded source changes.
No golden, tolerance, model, dictionary or dependency-lock changes were made.
Rust used `nightly-2026-09-08` (LLVM 23.1.1); C# used SDK 10.0.303 and runtime
10.0.11. Build settings were standard release/default features, automatic SIMD
and layout, fusion enabled, and no custom Rust flags for pipeline measurements.

## Reproduction and limits

From the prepared workspace, source `../env.sh`, then build the examples with
`cargo build --release --locked --examples`. Run the following fixed selections
once each, keeping builds and other inference idle during measurement:

```bash
target/release/examples/cpu_benchmark --model all --threads 8 \
  --fixture imgs_1,imgs_00018069,imgs_00006737 --workload latency \
  --runs 3 --budget-seconds 59 --case-timeout-seconds 20

target/release/examples/cpu_benchmark --model all --threads 8 \
  --fixture imgs_00018069 --workload throughput \
  --runs 3 --budget-seconds REMAINING --case-timeout-seconds 20

target/release/examples/cpu_benchmark --model all --threads 1 \
  --fixture imgs_00018069 --workload latency \
  --runs 3 --budget-seconds REMAINING --case-timeout-seconds 20
```

Replace `REMAINING` with the floor of the smaller remaining per-language budget;
subtract each report's elapsed worker time from an aggregate sixty seconds.
The archived commands contain the actual values used. Isolated adapter builds
are excluded. Each fresh worker performs one separately reported first batch,
one warm-up, then exactly three measured batches. Throughput batches contain
two concurrent owned requests sharing model weights. No retries, sample growth
or pooling across versions occurred.

Both implementations configure nominal eight-CPU stage budgets and four
recognition line workers. Rust's automatic classifier stage can use eight
workers; the pinned C# adapter uses four shared classification/recognition line
workers. Rust enforces an eight-worker aggregate pool cap across requests.
C# retains per-request nested budgets without a global admission cap: two dense
requests can each request four recognition workers with two intra-op workers.
On this ten-CPU Mac, unbound scheduling can therefore give C# more aggregate
CPU use. Process CPU measurements include runtime/JIT/GC as well as OCR, so
they cannot assign the whole throughput difference to this policy.

These measurements compare the retained concurrency policies as well as kernel
changes; the throughput cells do not isolate performance at equal enforced
aggregate CPU use. Rust explicit worker limits remain available. macOS scheduling
is unbound,
without a performance-core placement guarantee; AC power and Low Power Mode
off are recorded in provenance.

Fifteen selected cells do not complete the 36-cell platform matrix. One warm-up
and three samples do not establish steady state or statistical confidence.
The range-based gate preserves noisy results as inconclusive. Resident high-water
marks include shared pages and runtime initialization and are not private
commit. x86 macOS was compiled, but x86 execution and performance, other ARM
CPUs, and Windows performance after these changes remain unvalidated.

Automatic classification can retain more worker arenas, and request fairness
uses stage snapshots. Further tuning candidates include partial-column register
pressure and the remaining Portable4 fallback paths; they require measurements
on the affected workload before changing arithmetic or scheduling again.
