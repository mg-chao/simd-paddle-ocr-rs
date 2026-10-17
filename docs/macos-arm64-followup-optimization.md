# Apple M4 follow-up CPU optimization

This follow-up adopts three techniques from the SimdPaddleOCR comparison:
bounded ARM64 depthwise strips, shared graph-use indexes, and ARM bitmap mask
narrowing. On the physical Apple M4, the retained version reduced Medium sparse
OCR median latency from **391.072 to 368.471 ms (5.78%)**, with disjoint sampled
ranges. Tiny sparse improved by **0.84%** with disjoint ranges, a small absolute
change. Other latency ranges and all concurrent-throughput ranges overlap;
these measurements do **not** establish a general pipeline speedup.

The [portable evidence report](reports/macos-arm64-2026-10-17-followup-optimization.json)
preserves all stage samples and twelve final pipeline comparisons, including
first/warm-up timings, process CPU, memory, version order, and source/executable/
model/input hashes. Full owned OCR payloads remain in the linked local raw
artifacts; the portable report stores their canonical JSON hashes instead.
The earlier [ARM64 optimization record](macos-arm64-optimization.md) covers the
preceding native-kernel work and its separate C# comparison.

## Retained changes

1. [Bounded depthwise strips](../src/runtime/nhwc/neon_depthwise/strip.rs#L7)
   traverse one kernel row across up to 64 output pixels before advancing KY.
   The scratch block holds 32 channels, using **8,192 initialized stack bytes**
   per invocation. Each output retains ascending KY/KX FMA order, bias placement,
   scalar-tail arithmetic and its final fused store. Borders and partial bands
   continue through the checked readers. [Production dispatch](../src/runtime/nhwc/neon_depthwise.rs#L35)
   selects strips only for 9×9 kernels, at least 256 channels, unit stride and
   dilation, and output height and width of at least 64. The measured 7×7
   regression is excluded; variable-sized full-row heap scratch stays test-only.
2. [Shared graph-use indexes](../src/runtime.rs#L271) store compact consumer and
   producer indexes once in `CompiledModel`. Physical-plan binding reuses them
   instead of rebuilding semantic consumer maps and repeatedly scanning the
   graph for external uses. Prefix counts, duplicate operands, selected outputs,
   shape-specific legality and normalized trace identities remain relevant to
   each binding. [The regression test](../src/runtime/optimized/tests/semantic.rs#L134)
   checks the shared `Arc`, suffix branches and selected intermediates.
3. [ARM threshold narrowing](../src/pixel_kernels.rs#L584) keeps validity and
   nonempty masks in SIMD registers and converts comparison lanes directly to
   bitmap bytes. Strict threshold comparisons, finite-value rejection and
   scalar tails remain intact. Lookup-table expansion was also measured, but
   direct ARM narrowing was faster. The x86 production threshold path is
   unchanged.

## Focused measurements and rejected variants

These measurements belong to **diagnostic v1**, which forces each kernel
policy independently. They are not measurements of the final dispatch rule.
The retained production version was rebuilt and measured separately below.
Each shape/policy received one warm-up and three fixed single-call samples,
except physical binding, which used sixteen binds in each of three samples.
No automatic sample growth or retries occurred.

| Depthwise N/C/H/W; kernel | Existing tile (ms) | Full-row strip (ms) | 64-pixel strip (ms) | Bounded change |
| --- | ---: | ---: | ---: | --- |
| 1/96/152/240; 7×7 | 7.096 | 9.042 | 8.280 | 16.69% slower; rejected |
| 1/256/240/152; 9×9 | 45.046 | 34.532 | 35.302 | 21.63% lower |
| 1/256/120/76; 9×9 | 11.765 | 9.506 | 9.635 | 18.10% lower |

All three depthwise policies produced identical output float bits. Single
worker calls used synthetic finite inputs and packed weights; timing excludes
setup and output checks. Policy order was ABC/CBA/ABC. Full-row strips were
slightly faster on the two 9×9 shapes, but the retained policy bounds scratch
size and removes its heap allocation.

| Threshold map | Original (µs) | Deferred validity (µs) | Packed lookup (µs) | Selected ARM narrowing (µs) |
| --- | ---: | ---: | ---: | ---: |
| 320×320 | 29.291 | 26.708 | 16.375 | 13.125 |
| 960×960 | 256.458 | 233.625 | 143.959 | 117.375 |
| 1536×1537 | 659.083 | 609.416 | 369.458 | 293.833 |

Selected threshold medians were **54.23–55.42% lower**. Each sample rotated the
policy order; bitmap equality was checked before timing. This is a small
postprocessing stage, so the percentage is not an OCR latency prediction.

| Physical-plan bind | Baseline (µs/bind) | Shared indexes (µs/bind) | Reduction |
| --- | ---: | ---: | ---: |
| SmallREC, width 128, 333 nodes | 170.729 | 49.766 | 70.85% |
| MediumREC, width 640, 352 nodes | 135.724 | 53.901 | 60.29% |

The [binding diagnostic](../src/runtime/optimized/tests/performance.rs#L7)
prepares logical shapes and packed weights before timing. Each model/version
has one warm-up bind plus three samples of sixteen fresh physical binds;
there is no inference or activation allocation. Physical step counts, fused
counts and arena element counts match the baseline. The saving concerns new
sessions and physical shape-cache misses; cached execution avoids this work.

## Retained pipeline comparison

All rows compare the frozen Rust baseline at `577db45` with the retained source,
using native release/default-feature builds, automatic SIMD/layout, fusion,
and the same models, dictionary and prepared pixels. macOS 27.0.1 scheduling was
unbound on a ten-CPU, 16-GiB Apple M4. Each case ran in a fresh process with
eight workers, one separately recorded first batch, one warm-up, and exactly
three measured batches. Baseline/candidate process order alternated by case;
processes ran serially without concurrent builds or inference.

Sparse/dense/mixed are `imgs_1`, `imgs_00018069`, and `imgs_00006737`.
Latency rows contain one request per batch. Medians are milliseconds.

| Model | Input | Baseline | Retained | Sample ranges overlap |
| --- | --- | ---: | ---: | --- |
| Tiny | Sparse | 47.403 | 47.004 | No; 0.84% lower |
| Tiny | Dense | 75.776 | 75.239 | Yes |
| Tiny | Mixed | 87.232 | 86.151 | Yes |
| Small | Sparse | 124.874 | 122.482 | Yes |
| Small | Dense | 230.507 | 229.834 | Yes |
| Small | Mixed | 276.228 | 274.951 | Yes |
| Medium | Sparse | 391.072 | 368.471 | No; 5.78% lower |
| Medium | Dense | 762.380 | 766.491 | Yes |
| Medium | Mixed | 1002.549 | 985.560 | Yes |

Only the following three dense-input cells use two concurrent requests sharing
an aggregate eight-worker pool. Throughput is `2000 / median_batch_ms`.

| Model | Baseline (images/s) | Retained (images/s) | Sample ranges overlap |
| --- | ---: | ---: | --- |
| Tiny | 16.512 | 16.530 | Yes |
| Small | 5.369 | 5.309 | Yes |
| Medium | 1.521 | 1.495 | Yes |

All twelve complete parsed owned OCR results matched exactly between versions.
Retained activation/workspace diagnostics were identical in each matched pair;
peak resident memory ranged from 99.67% to 100.11% of baseline. The new strip's
stack scratch is not part of retained workspace accounting. Raw CPU samples
are preserved separately; no general process-CPU improvement is claimed.

The final workers consumed **25.907 s baseline / 26.089 s retained**, including
startup, load/compile, first inference, warm-up and measured batches. Per
version this is twelve first batches, twelve warm-ups and thirty-six measured
batches: **45 measured requests, 75 including first/warm-up**. The twelve-cell
selection includes nine latency cells and three throughput cells.

The baseline physical-binding process consumed another **0.931 s**, conservatively
including fingerprinting overhead, for a baseline total of **26.838 s**.
Diagnostic v1's three processes consumed **1.046 s** in total. Every source
version stayed within its aggregate sixty-second performance budget, excluding
builds. Native depthwise, threshold, physical binding, and pipeline workers had
external deadlines of 30, 10, 30, and at most 20 seconds respectively. The
depthwise diagnostic's internal assertion separately allows sixty seconds;
the runner's shorter external deadline governed this run.

```text
Baseline source:  fdf7b01ea9ab0116ee3501bd5210af4fea60c3c27aec8bfbf9d02fcd5fbdc46a
Diagnostic v1:   f058ef00973512ae0de5ff7f315c677ae2040df56d65cfb6f6e7aa6e53944b32
Retained source: b0fddcd1c454041574461f2aba725bea71676dd6c29ada00bf8727de2b57b09b
Baseline binary: 3b6212e4b0ed4b2e7e9d844f96a3f9e9cc26edc2039ec22a78e336c95527c72e
Retained binary: 182207b0e6633ab3df64a8ed71f85945d462e6c698a88b1daeed2df97116d624
```

Source manifests include production, tests/diagnostics within `src`, examples,
tools and build configuration; docs are excluded. Diagnostic samples are kept
under their own source fingerprint and are not pooled with retained timings.
The pinned C# reference remains `33ff8a81672452041d110e8ba52dcd2639e54228`;
this follow-up does not repeat its performance measurement.

## Validation and remaining scope

Completed local checks: **333 default-feature library tests**, **323 serial
library tests** without default features, **334 profiling-enabled library
tests**, **29 integration tests**, and **five doctests**. Normal suites retain
twelve ignored library diagnostics and one ignored integration test. Native
all-target/all-feature and x86_64 macOS cross-compilation, formatting and
whitespace checks passed. Clippy exited successfully with existing
architecture/harness warnings and three new benign diagnostic-style warnings;
the build is not warning-free. The measured production source was left frozen.

The final live C# correctness qualification **passed all 213/213 requests with
zero differences**, covering all three models, 22 default fixtures, worker
budgets, batching, alignment, padded pixel formats, rotations, fixed recognition
widths, classification settings, reuse and recovery. Its
[run record](../target/qualification/1792213146233-51112/run.json) and
[empty differences report](../target/qualification/1792213146233-51112/differences.json)
are linked and hashed in the portable evidence. Rust used
`nightly-2026-09-08`/LLVM 23.1.1; the reference used .NET SDK 10.0.303/runtime
10.0.11. Correctness qualification took 230.231 process seconds and is separate
from performance-budget accounting. Models, tolerances and saved expectations
were unchanged.

One warm-up and three samples do not establish statistical significance or
steady state. These twelve cells are not a full platform matrix. AVX-512 input
descriptors and tile variants remain deferred because this host cannot validate
their native execution. Arena base alignment was not adopted without evidence
of an incremental gain for the current native loads. x86 threshold production
code remains unchanged. Other ARM CPUs and native x86 performance need their
own measurements.

Exact commands, raw logs, source manifests and unabridged worker results remain
in the local [follow-up artifact directory](../target/cpu-adoption-20261017-followup/).
The [fixed comparison driver](../target/cpu-adoption-20261017-followup/compare.py)
and [final raw report](../target/cpu-adoption-20261017-followup/final-comparison/report.json)
record the prepared jobs and version order. The portable evidence report links
every raw worker output and records the hashes needed to identify this run.
