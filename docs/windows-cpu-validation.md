# Windows x64 CPU validation evidence

The historical validation on 2026-10-16 uses the pinned `nightly-2026-09-08` toolchain,
.NET SDK 10.0.303/runtime 10.0.11, the pinned C# reference
`33ff8a81672452041d110e8ba52dcd2639e54228`, and the unchanged baseline manifest
SHA-256 `0ecc7c2634c3da2d2bee2ee7021d6abe91f2685cc9b64fc34d8db6ebcf5e440e`.
The host is Windows x64, AMD Ryzen 9 5950X, with 24 available logical CPUs;
automatic inference selects AVX2/FMA. This report does not qualify AVX-512 or
another operating system or architecture.

The newer [NHWC concatenation and binary-alias adoption pass](#nhwc-concatenation-and-binary-alias-adoption-2026-10-17)
records the archived implementation's focused measurements and fresh 213-request
correctness qualification. Its scope remains separate from the historical matrices.
The subsequent [row-scalar correction](#row-scalar-in-place-follow-up-2026-10-17)
has separate focused validation.

The latest complete 36-case comparison belongs to the historical
`5a966e7` implementation: **26 passed, one failed and nine were inconclusive**.
All its output comparisons and process-memory gates passed; full performance
qualification did not. The earlier 21-pass/15-inconclusive matrix below measures
a different source version. Neither matrix qualifies later follow-up edits.

Follow-up work began on 2026-10-16 and completed on 2026-10-17, China time. Both
unconditional and capacity-aware CLS/REC overlap candidates were rejected.
The retained V4 staged pipeline passed release checks and all 213 real-model
configuration comparisons with zero differences. Its final Tiny/dense Rust/C#
comparison passed, covering **one of 36** performance cells. Final paired-kernel
sample ranges overlap; no stable global speedup or full V4 performance pass is
established. See [the follow-up section](#follow-up-optimization-and-scheduling-policy).
Correctness and packaged-consumer results in the historical sections identify
their recorded versions rather than the current working tree.

| Milestone | State |
| --- | --- |
| M1: independent validation | Complete: isolated C# builds/workers, process deadlines and canonical comparison |
| M2: CPU configurations | Complete for V4: all 213 real-model configuration/reuse/recovery comparisons passed with zero differences |
| M3: matched performance baseline | Tool and complete 36-case matrix implemented; performance exit gate unresolved |
| M4: measured optimization | Arena placement and targeted kernels implemented; global performance exit gate unresolved |
| M5: usable release | Historical package and all three packaged consumers verified; follow-up source/package validation remains separate |

## Historical correctness and distribution

| Check | Result |
| --- | --- |
| Serial canonical C# / golden / Rust comparison | Final runtime: 66/66 passed; zero differences |
| Configuration qualification | Final runtime: 213/213 passed across 16 profiles; zero differences |
| Default release tests | 310 passed; 4 ignored |
| No-default-feature release tests | 303 passed; 4 ignored |
| All-features release tests | 311 passed; 4 ignored |
| All-features library documentation | Built successfully; zero warnings after two documentation-comment corrections |
| Explicit concurrent C# build/live worker integration | Passed: two isolated builds and two simultaneously loaded workers |
| Build failures and process deadlines | Nonzero exits remain failures; timed-out process terminated and reaped |
| Formatting | `cargo fmt --all -- --check` passed |
| Package verification | `cargo package --allow-dirty --locked --offline` passed on the final runtime; both MIT texts included and byte hashes verified |
| Independent packaged consumer | Offline release build with only the packaged library dependency; Tiny, Small and Medium passed real-fixture OCR with captured alignment and owned results |
| OCR example | Text/classifier-disabled output and captured token boxes passed on a real fixture |

The explicit concurrency test is available with:

```powershell
cargo test --release --locked --test harness_contract -- --include-ignored
```

The final runtime test counts include the targeted pointwise, descriptor and
border-path changes. The four ignored cases are three explicitly invoked
physical performance probes and the live C# integration test. Two subsequent
comment-only edits fix a bracketed tensor-shape rustdoc link and a bare URL;
documentation was rebuilt afterward with zero warnings, without rerunning
runtime tests. Final real-model correctness and the complete uninstrumented
performance matrix are recorded below, with earlier versions retained separately.

The [packaged-consumer archive](reports/windows-x64-2026-10-16-packaged-consumer.json)
records the final runtime's package verification, license hashes, commands and
artifact hashes. All 82 Rust source files in the extracted crate match the
current source byte for byte. The independent offline release consumer has
one direct dependency, the extracted library, and runs the pinned `imgs_1`
BGR24 pixels through all three presets with the classifier, alignment capture
and owned results. Tiny returns three lines; Small and Medium return two each.
The inspected Rust dependency tree contains serde and rayon and their Rust
dependencies, and the consumer invokes no C# adapter. This verifies no .NET
runtime dependency in the consumer; .NET remained installed on the host.
The verified crate SHA-256 is
`8a30faf4e2c355f823cfacb02224f4e2330bf644d813dfd0266daf76ca044e1d`.
Final documentation/report additions and the validation-guard fix require a
repackage; the verified runtime source files and consumer manifest remain unchanged.
Its new archive hash must be recorded externally in
`target/package/final-package-check.json`, separately from this tested artifact.
The crate does not attempt to embed its own final archive hash.

The [final correctness archive](reports/windows-x64-2026-10-16-final-kernels-correctness.json)
records fresh canonical 66/66 and configuration 213/213 comparisons on the final
runtime, with zero differences. It includes all 16 configuration profiles,
six expected rejection/recovery cases per implementation, and 18 captured
alignment responses per implementation containing 2,909 spans and 2,909 character
boxes. The source fingerprint was rechecked at completion, and both runner
executable hashes remained unchanged. Canonical evidence is under
`target/baseline/1792155195977-verify-103952/`; configuration evidence is under
`target/qualification/1792155195893-60080/`. These are correctness runs;
their elapsed times do not establish performance.

After the allocator correction, both complete correctness suites were rerun
locally using the already-built release executables, concurrently with each
other. They completed with zero differences: canonical 66/66 in 360.152 seconds
and configuration 213/213 in 325.011 seconds. These are correctness durations,
not performance measurements. The runs precede subsequent targeted kernel
changes and establish correctness for the allocator version specifically.

Canonical evidence: `target/baseline/1792152238301-verify-86432/`.
Configuration evidence: `target/qualification/1792152238300-62740/`.
The [compact correctness archive](reports/windows-x64-2026-10-16-allocator-correctness.json)
preserves complete run metadata, selected profiles, completed counts, zero-difference
counts, artifact hashes and allocator-version provenance. It independently counts
66 Rust/C# response pairs for the canonical suite and 213 pairs for configuration,
including six expected stride/pixel-limit rejections per implementation.
Runner SHA-256 values were unchanged from launch to completion:

- Canonical: `a079b43a3206d2c3981a14c7375ceae07fb55964274a4cafc96a81d1a1f2d82a`.
- Configuration: `30734ac0a144ea48323df2d40b2104c4008934bfb7107c7d803c6d2e9b9a2d96`.

The allocator source file SHA-256 is
`bd0cb76ee0e1d87c7db6a8c323fc588f8a4b7eda2cb348bef4f165fece2d83ac`;
both runners were built after that allocator change. Production kernel changes
require new correctness runs and cannot inherit these results.
Raw generated artifacts stay outside the package; the checked-in manifest,
goldens, compact reports, test contracts and documented commands reproduce the checks. Network
package verification initially encountered an unavailable registry proxy;
offline verification succeeded with cached dependencies.

## Historical performance completion gate

The first focused tiny-model sparse-page comparison identified excess activation
memory. The old physical planner reused whole dead blocks, stranding unused
tails. The correction places descending-size alias groups into lifetime-compatible
gaps, preserving inclusive lifetimes, relative view offsets, aligned starts,
exact small budgets, overflow checks and transactional reshape. Six focused
regressions pass; the fragmentation reproducer drops from 176 to 113 floats.

On `tiny/imgs_1`, eight threads, one owned result, one first inference, one
warm-up and three measured runs per implementation:

| Metric | Before allocator correction | After correction |
| --- | ---: | ---: |
| Rust retained activations | 86,702,336 bytes | 40,960,000 bytes |
| Rust peak private commit | 113,602,560 bytes | 67,698,688 bytes |
| C# peak private commit | 96,788,480 bytes | 96,722,944 bytes |
| Rust median latency | 58.321 ms | 59.117 ms |
| C# median latency | 52.746 ms | 61.519 ms |
| Gate | Failed memory | Memory passes; timing inconclusive |

The memory correction reduces Rust peak commit by about 40% and retained
activations by about 53%. Timing noise crosses the 10% boundary, so these
samples do not establish a latency improvement or a complete performance pass.
Raw versioned evidence is in `target/cpu-benchmark/1792150345637-96800/` and
`target/cpu-benchmark/1792151426763-95040/`; measurements are not pooled across
the two source versions.

The complete post-allocator matrix is recorded in
`target/cpu-benchmark/1792151587227-70156/report.json` and `report.md`.
Its [compact machine-readable archive](reports/windows-x64-2026-10-16-allocator-performance.json)
is included with the crate: all 36 jobs, raw timing samples, medians, ratios,
peak memory, cold/load timing, diagnostics, differences, status, budgets and
provenance are preserved. Large OCR payloads are replaced by joined-text SHA-256
and line/detector counts. This archive labels the allocator version before
subsequent targeted kernel changes; the original report hash is retained.
It covers all 36 required model/fixture/thread/workload comparisons, with one
first inference, one warm-up, and three measured batches per implementation
per comparison. Throughput batches produce two concurrent owned results.

| Bundle | Passed | Failed | Inconclusive | Largest Rust/C# peak-memory ratio |
| --- | ---: | ---: | ---: | ---: |
| Tiny | 11 | 0 | 1 | 0.790 |
| Small | 2 | 3 | 7 | 0.887 |
| Medium | 1 | 7 | 4 | 0.956 |
| Total | 14 | 10 | 12 | 0.956 |

Every comparison has zero output differences and passes the peak private-memory
gate. Timing and throughput remain incomplete: small sparse-page serial latency
is 490.779 ms versus C# 412.329 ms (1.190 ratio); medium mixed-page eight-thread
latency is 1188.998 ms versus 1012.182 ms (1.175 ratio). Inconclusive comparisons
remain unresolved and are not counted as passes. The full run consumed 262.910
seconds for Rust and 261.186 seconds for C# within its explicit 600-second
budget per implementation, excluding build time.

This matrix identifies the allocator version with Rust source SHA-256
`d7190dc60f6fc13887789ef3dd903afe4a5cf050ef7ed5e3a0bb8971bca906be`
and benchmark executable SHA-256
`e224b1c3c834293b0ceef89bd93150556823a28498535a578d33a99a90022261`.
Later kernel changes require their own correctness and performance evidence;
samples from different source versions must not be pooled. See
[the protocol](cpu-performance.md) for sample, affinity, noise, and gate rules.

No full Windows CPU performance completion claim is made until every required
cell passes the agreed 10% gate. Missing or noisy evidence remains unresolved.

### Final runtime performance result

The final matrix completed all 36 required cases in
`target/cpu-benchmark/1792155564865-26452/`, with zero output differences, no
missing measurements and every peak private-memory comparison passing. The
[final compact performance archive](reports/windows-x64-2026-10-16-final-kernels-performance.json)
preserves every job, raw sample, load/first/warm-up timing, median, ratio, memory
diagnostic, difference list, status, source/binary/toolchain provenance and the
original report hash. Joined OCR text is replaced by SHA-256 and structural
counts. Its summary lists every outstanding case and observed timing range.

| Bundle | Passed | Failed | Inconclusive | Largest Rust/C# peak-memory ratio |
| --- | ---: | ---: | ---: | ---: |
| Tiny | 11 | 0 | 1 | 0.790 |
| Small | 6 | 0 | 6 | 0.876 |
| Medium | 4 | 0 | 8 | 0.948 |
| Total | 21 | 0 | 15 | 0.948 |

The run used one first inference, one warm-up and three measured batches per
implementation per case. It consumed 246.297 seconds for Rust and 270.658 seconds
for C# within the explicit 600-second budget per implementation; build time is
excluded. The run's exit code is 1 for its **inconclusive** result, and
`full_qualification_passed` is false. No unchanged-code rerun or sample growth
was performed to improve the outcome.

All 18 single-request latency medians are within 110% of C#, but six latency
cases have observed sample ranges crossing that gate: Small dense at eight
threads; Medium sparse/dense/mixed at one thread and dense/mixed at eight
threads. Nine two-request throughput cases are inconclusive. Seven of their
median throughputs fall below 90% of C#:

| Bundle | Fixture | Threads | Rust/C# median throughput |
| --- | --- | ---: | ---: |
| Tiny | `imgs_1` | 8 | 0.894 |
| Small | `imgs_00018069` | 1 | 0.851 |
| Small | `imgs_00018069` | 8 | 0.854 |
| Small | `imgs_00006737` | 8 | 0.853 |
| Medium | `imgs_1` | 8 | 0.862 |
| Medium | `imgs_00018069` | 8 | 0.826 |
| Medium | `imgs_00006737` | 8 | 0.864 |

The other two unresolved throughput cases are Small mixed at one thread and
Small sparse at eight threads. Their medians meet the target, but their observed
ranges cross it. The seven sub-target medians indicate an outstanding throughput
deficit to investigate; the fixed sample ranges prevent classifying them as
clear failures under the declared conservative gate. The remaining eight cases
need a separately justified uncertainty investigation. Neither group is a pass.

This final benchmark identifies Rust source SHA-256
`7da7dd8f87bd5ed16ef5fa752dbf144e36c421b8794d4ba0e1226d77dda35a3b`
and executable SHA-256
`b966f13769a9992f4fddf2eff8b52219fad102cf27ba9d626159a16a8b2e88bb`.
Its version is distinct from the historical allocator matrix. Correctness,
memory and distribution milestones are verified; the complete CPU performance
gate remains open until all 15 outstanding cases meet their acceptance criteria.

After qualification, a validation-tooling guard was tightened to reject
`CARGO_TARGET_*_RUSTFLAGS` overrides. No such override was present during the
recorded runs, and the change does not alter runtime source or the 82 source
files verified in the packaged consumer. The archived fingerprints and executable
hashes continue to identify the measured tooling version before that guard fix;
no measurement of a newly built benchmark binary is claimed.
The focused harness rerun passed four tests with the live integration case
ignored, and the target-specific override inventory was empty. This tooling-only
check does not replace or relabel the recorded runtime/performance runs.

The next investigation should focus on the seven median throughput deficits
under two simultaneous requests. Shared CPU scheduling and possible CLS/REC
overlap are hypotheses to test with bounded production-path diagnostics, not
established bottlenecks or grounds for changing the bounded architecture. The
other eight inconclusive cases need an explicit uncertainty investigation.
Keep one warm-up and three fixed measured samples for candidate comparisons;
expand a diagnostic only to answer a stated question with a defined deadline.
Do not automatically repeat unchanged comparisons for favorable samples, pool
different versions, or weaken correctness/memory limits to close the gate.

## Targeted optimization evidence

The following investigations explain costs found in the allocator-version matrix.
They are separate diagnostic code versions and do not update the correctness
counts or complete the performance gate above. The
[targeted diagnostic archive](reports/windows-x64-2026-10-16-targeted-diagnostics.json)
preserves raw fixed kernel samples, their version/scope labels, and the C# stage
and node reports with original artifact hashes.

The packed NCHW pointwise kernel previously computed eight output channels over
four spatial positions. A candidate computes four output channels over sixteen
spatial positions and tiles the input working set to approximately 192 KiB. A
focused AVX2/FMA probe used batch 1, 96 input channels, 24 output channels, CPU
affinity mask 1, fixed input/weight/bias patterns, one warm-up invocation, and
three samples of four invocations each. Allocation and packing precede timers;
reported times are medians in milliseconds per invocation. Focused NCHW
correctness checks passed before these measurements.

| Spatial shape | Before pointwise change | After pointwise change |
| --- | ---: | ---: |
| 19×30 | 0.132800 ms | 0.044125 ms |
| 76×120 | 2.195550 ms | 0.585275 ms |
| 152×240 | 8.907025 ms | 2.406275 ms |

The complete measured kernel work took 147.936 ms before and 41.431 ms after.
This candidate precedes the subsequent 3×3 reduction-isolation and border-path
changes; its samples cannot validate those later versions or an OCR latency gate.

The Medium detector also contains dense 7×7 convolutions. The NHWC dense kernel's
prevalidated receptive-field descriptors originally accepted at most 25 taps,
so a 49-tap convolution repeated general gather geometry inside its reduction.
Increasing that descriptor capacity to 49 admits the existing checked interior
path for 7×7 kernels. In the same synthetic landscape graph shape
`[1,3,608,960]`, the private physical-step timing for dense 32→32 node210 fell
from **338.9613 ms to 32.2219 ms**. These private probes execute isolated steps
with neutral buffers and include finite-output checking overhead. They identify
an implementation cost; they are not complete OCR timings or output-agreement
tests.

Shape identity is essential here. The first Rust physical-step probes used that
landscape shape, while the actual `imgs_1` fixture and the C# production-node
profiles use portrait input `[1,3,960,608]`. The resulting feature maps have
opposite height/width orientations. **Those Rust and C# node timings are not
matched comparisons; no exact cross-language ratio is established by them.**
The official benchmark and both production-stage profilers used the correct
fixture pixels and dimensions. The later private Rust helper corrects the
portrait binding; its supportive measurements are recorded below, separately
from the landscape investigation and the official performance gate.

For context, the correct-fixture C# stage-only samples report Small DET graph
244.8563 ms / REC graph 131.5795 ms and Medium DET graph 1251.3213 ms / REC graph
467.8936 ms. Each has one warm-up and one instrumented sample. C# DET-only
per-node instrumentation is a separate run with additional overhead; its totals
and differently oriented private Rust node probes must not be pooled with those
stage samples or the uninstrumented benchmark.

Assembly inspection found a further cause in the NCHW 3×3 production kernels:
inlining the prepared fused-output transformation caused four accumulators to
spill to stack inside the input reduction, whereas the identity-store body kept
them in vector registers. The targeted fix separates the ISA-specific reduction
into non-inlined helpers returning its accumulators and applies the output
transformation afterward. The existing layout policy and arithmetic order are
preserved. The subsequent border-path change shares duplicate input loads across
eight output channels. For the corrected portrait shape `[1,3,960,608]`, isolated
neutral-buffer probes use one warm-up and one sample per physical step:

| Small detector diagnostic | Reduction-isolation version | Border-sharing version |
| --- | ---: | ---: |
| Stride-2 node8, 48→24, 3×3 | 62.2755 ms | 29.6724 ms |
| Stride-1 node234, 96→24, 3×3 | 83.7439 ms | 21.9036 ms |
| Sum of isolated physical steps | 441.154 ms | 307.192 ms |

These before/after measurements share actual portrait geometry and the same
neutral-buffer/finite-check diagnostic method. They remain distinct source
versions and do not provide an OCR latency or throughput gate. For the
border-sharing version, the Medium isolated-step sum is 1357.926 ms, including
39.8796 ms for dense 7×7 node210; these are supportive single samples only.
Raw records and their hashes are preserved in the targeted diagnostic archive
from `target/physical-steps-reduce3.log` and
`target/physical-steps-border-final.log`. The final runtime test suites passed
as listed above, as did final real-model correctness. The matched uninstrumented
matrix above remains inconclusive. No completion claim follows from these
diagnostic improvements.

## Concurrency and kernel optimization pass

This subsequent pass starts from Rust commit
`5a5f89097c5f7b20a94b0354bb9fbc137d52593a` and compares against the same clean C#
commit `33ff8a81672452041d110e8ba52dcd2639e54228`. It retains these changes, in
investigation order:

1. OCR requests register demand before pool admission. Detector/preprocessing
   worker budgets and nested recognition threads use a stage-local share of
   the shared CPU capacity. Configured maxima, stable line-worker ownership,
   warmed buffers and the fixed pool's hard concurrency cap are preserved.
   RAII releases demand on errors and unwinding.
2. The recognition NCHW pad-end 2×2 kernel processes its spatial tails four
   positions at a time. It retains separate multiply/add, skipped padding taps
   and scalar output transformations on these tails.
3. The NCHW pointwise kernel validates input spans and packed panels before a
   separate ISA-specific reduction. NHWC large depthwise convolutions use
   checked interior spans and a separate two-pixel, 32-channel reduction;
   borders and partial channel blocks retain their existing fallback.
4. Thin LTO with one code-generation unit is documented as an optional
   application release setting. The repository's default profile is unchanged.

The [implementation archive](reports/windows-x64-2026-10-16-concurrency-and-kernels.json)
keeps the CPU-budget-only, kernel, build-setting and final OCR comparisons
separate. It includes raw samples, version fingerprints, qualification
metadata and limitations. No timings from different source versions are pooled.

### Correctness and build checks

- The full real-model qualification passed **213/213 cases**, with no C# output
  differences, at `target/qualification/1792161950895-67248/`. It covers all
  three model bundles, 22 fixtures and the configured serial, worker, batching,
  alignment, pixel-format, classification, width, threshold, dilation, reuse
  and recovery profiles.
- That full run preceded equivalent lint-only source edits. The rebuilt source
  passed a further **24/24 worker/recovery cases** with no differences at
  `target/qualification/1792163133530-75456/`. The final performance workers also
  compare owned OCR results against C#.
- Default-feature release unit tests passed **285 tests**; the final all-feature
  release suite passed **286 tests**, with five opt-in diagnostics ignored in
  each suite. Checks of all targets passed with all features and with default
  features disabled. Formatting and library all-feature Clippy with warnings
  denied passed. All-target Clippy reports existing style lints in
  `tools/baseline/mod.rs` and `tools/benchmark/mod.rs`.
- Tests exercise arithmetic order, border/tail handling, packed panels, output
  transforms and worker bands. The CPU-budget tests additionally cover waiting
  callers, context isolation, configured limits and unwind cleanup.

Validation ran on the physical Windows x64 host with AVX2/FMA. AVX-512 and
non-x64 native execution were unavailable. The C# checkout was not modified.

### Kernel and build diagnostics

The following measurements isolate mechanisms and are **not end-to-end OCR
speedups**. Kernel probes use one logical CPU, one warm-up and three fixed
samples; NCHW pointwise samples each contain four invocations. The recognition
backbone diagnostic uses one warm-up and one instrumented production-path pass
on a deterministic tensor at each of two widths.

| Changed operation | Before | After | Scope |
| --- | ---: | ---: | --- |
| Small REC node3, 48→24 pad-end 2×2 | 1.4826 ms | 0.2544 ms | One instrumented step, width 128 |
| NCHW pointwise 96→24, 19×30 | 0.034375 ms | 0.031775 ms | Median kernel call |
| NCHW pointwise 96→24, 76×120 | 0.550600 ms | 0.471925 ms | Median kernel call |
| NCHW pointwise 96→24, 152×240 | 2.579975 ms | 2.398825 ms | Median kernel call |
| NHWC depthwise 7×7, 96 channels, 152×240 | 21.0807 ms | 11.9028 ms | Median kernel call |

Fresh matched C# kernel probes and their raw ranges are in the archive. Their
warm samples vary substantially, especially for the large detector operations;
the kernel medians alone do not establish general cross-language parity.
The unchanged NHWC recognition pointwise kernel was already competitive on the
two matched shapes. A K=256 cache-blocking experiment passed its six focused
correctness tests but demonstrated no backbone improvement, so it was removed.
Instrumented backbone totals varied between runs, including unchanged steps;
no overall recognition speedup is inferred from the isolated node3 gain.

The separate Thin LTO build experiment used two fixtures, eight logical CPUs,
two simultaneous requests, alternating build order, one first inference, one
warm-up and three measured batches per variant. Both variants produced
identical owned OCR results and retained identical tensor/workspace sizes.

| Input | Standard release median | Thin LTO + one codegen unit median | Interpretation |
| --- | ---: | ---: | --- |
| Small / dense | 411.596 ms | 396.901 ms | 3.6% lower; sample ranges overlap |
| Medium / mixed | 3196.891 ms | 2217.498 ms | 30.6% lower; sample ranges do not overlap |

The build experiment consumed 18.052 seconds for standard release and 14.225
seconds for Thin LTO, below the 60-second budget per variant, excluding builds.
Thin LTO increased compilation time; these two cases do not qualify it across
all models or replace the standard-release C# comparison. Consumers can enable
the setting in their application's root manifest as shown in the
[usage guide](cpu-usage.md).

The initial CPU-budget-only Medium comparison overlapped an unrelated build,
so its timing failure does not isolate the scheduling change. Stable line
ownership and CLS/REC stage ordering remain unchanged. A larger pipeline
restructure was not justified by these measurements and is not claimed as an
implemented optimization.

### Standard-release comparison after the changes

The complete 36-case matrix used all three bundles and three fixtures, CPU
affinity masks `1` and `ff`, one or two requests, one first inference, one warm-up
and three measured batches per implementation/case. The explicit budget was
600 seconds per implementation with a 120-second case deadline, excluding
builds. Actual execution consumed **250.313 seconds for Rust** and
**302.265 seconds for C#**. The raw run is
`target/cpu-benchmark/1792163502784-22856/`; its full comparison data, source and
binary hashes, and report hash are preserved in the implementation archive.

| Bundle | Passed | Failed | Inconclusive | Largest Rust/C# peak-memory ratio |
| --- | ---: | ---: | ---: | ---: |
| Tiny | 9 | 1 | 2 | 0.792 |
| Small | 11 | 0 | 1 | 0.878 |
| Medium | 6 | 0 | 6 | 0.959 |
| Total | 26 | 1 | 9 | 0.959 |

All **36 output comparisons and memory gates passed**, with no missing
measurements, worker errors or source changes during measurement. The overall
performance status is **failed**, exit code 1, and
`full_qualification_passed` is false. The existing gates require single-request
latency within 110% of C# and two-request throughput at least 90% of C#, using
the observed ranges rather than medians alone. Fourteen of eighteen latency
medians and sixteen of eighteen throughput medians meet their respective
targets; overlapping ranges still leave the nine cases above inconclusive.

The clear failure is Tiny / `imgs_1` / one CPU / one request: Rust samples are
418.985, 370.043 and 350.262 ms versus C#'s 190.698, 166.317 and 170.181 ms.
Its cause is unresolved. The separate two-request case on one CPU passed,
with Rust's entire batch median 323.609 ms versus C#'s 339.204 ms; this
cross-case timing inconsistency warrants causal investigation, but does not
erase the single-request failure or justify an unchanged-code retry.

The remaining inconclusive cases are Tiny sparse at eight CPUs with one/two
requests; Small mixed at eight CPUs with one request; and Medium dense/mixed
at one CPU with one request, sparse at eight CPUs with two requests, dense at
eight CPUs with one request, and mixed at eight CPUs with one/two requests.
Tiny sparse and Medium sparse at eight CPUs are the only two-request medians
below the throughput target, at 0.856× and 0.890× C# respectively.

The eight-CPU, two-request measurements directly exercise the shared request
budgets. Times below are complete two-image batches, not per-image latency:

| Bundle / fixture | Rust median | C# median | Rust/C# throughput | Gate |
| --- | ---: | ---: | ---: | --- |
| Tiny / sparse | 126.919 ms | 108.625 ms | 0.856× | Inconclusive |
| Tiny / dense | 160.370 ms | 232.581 ms | 1.450× | Passed |
| Tiny / mixed | 180.021 ms | 213.538 ms | 1.186× | Passed |
| Small / sparse | 219.447 ms | 234.367 ms | 1.068× | Passed |
| Small / dense | 400.923 ms | 417.535 ms | 1.041× | Passed |
| Small / mixed | 501.401 ms | 514.056 ms | 1.025× | Passed |
| Medium / sparse | 835.737 ms | 744.044 ms | 0.890× | Inconclusive |
| Medium / dense | 1500.020 ms | 1463.470 ms | 0.976× | Passed |
| Medium / mixed | 2110.444 ms | 1919.074 ms | 0.909× | Inconclusive |

All six Small two-request configurations passed. This is encouraging evidence
for the combined changes, but does not isolate scheduling from the kernel
changes or eliminate host variation. The earlier 21-pass/15-inconclusive matrix
is a different source version and run, not a controlled speedup estimate.
No failed or inconclusive case was repeated to improve its classification.
Closing the remaining performance gaps requires a separate bounded
investigation; this pass does not establish maximum achievable performance.

## Follow-up optimization and scheduling policy

The [follow-up archive](reports/windows-x64-2026-10-16-followup-optimization.json)
preserves source snapshots, executable identities, raw samples and scope
limitations. The investigation uses these distinct versions:

| Version | Change and evidence |
| --- | --- |
| Clean baseline | Commit `5a966e7`; kernel/backbone probes and a focused Medium mixed-page C# comparison |
| V1 | Larger AVX pointwise cache groups, exact depthwise interior strips, exact mutable task bands, and the corrected benchmark harness |
| V2, rejected scheduling policy | Unconditional CLS/REC overlap at batch size one, together with compact reusable recognition-result storage; a three-case Rust variant comparison and correctness checks |
| V3, rejected scheduling policy | Capacity-aware overlap passed correctness but showed no consistent end-to-end gain against the staged control |
| V4, retained staged pipeline | Build/checks and all 213 real-model comparisons passed; final kernel timings remain uncertain and one focused C# performance cell passed |

No samples are pooled across these versions. Files named `final-*` in the
follow-up artifact directory were produced during candidate investigation;
their names do not establish that they qualify V4. V1, V2 and V3 each have an
84-entry runtime-source snapshot identifying their measured versions even after later
edits overwrite build paths. Missing historical binary hashes remain explicitly
unavailable rather than inferred from newer executables.

### Kernel and benchmark-harness revision

The V1 pointwise change reuses each full-K weight panel over 96 rows instead of
48 when input and output channels are at least 384. Depthwise strips split at
the exact horizontal interior boundaries so true border pixels do not force
interior pixels onto the checked path. Mutable preprocessing jobs use exact
floor partitions; nine items with an eight-job budget can occupy eight nonempty
jobs. These changes preserve per-output arithmetic order and exclusive output
ownership.

One warm-up and three fixed, single-call samples per shape, with allocation and
packing outside timers, produced these diagnostic medians:

| Operation | Clean baseline | V1 | Observed evidence |
| --- | ---: | ---: | --- |
| NHWC pointwise, rows 96 / inner 384 / columns 768 | 0.4638 ms | 0.4135 ms | 10.8% lower; sample ranges do not overlap |
| NHWC pointwise, rows 960 / inner 512 / columns 1024 | 7.5373 ms | 7.5453 ms | Neutral within observed variation |
| NHWC depthwise, 96 channels / 152×240 / 7×7 | 10.7298 ms | 9.7086 ms | 9.5% lower; sample ranges do not overlap |

These are focused kernel results. Medium's instrumented recognition backbone
at width 640 changed from 222.3449 to 216.1200 ms in one sample per version,
which supplies only supportive evidence. Small's 18.6018 to 9.5933 ms totals do
not establish a backbone speedup: its baseline has a 9.1832 ms pointwise step,
while another step with the same shape takes 0.6101 ms. Raw step names, shapes
and timings remain in the archive.

The original harness selected process order from case parity, making every
latency comparison Rust-first and every throughput comparison C#-first in the
full matrix. V1 replaces this with a recorded, canonical factorial ordering
independent of workload and filtering, plus explicit language-first modes.
Both workers now record process CPU time alongside wall time. Existing wall,
correctness and memory gates, one first batch, one warm-up and at most three
measured batches remain unchanged. These corrections improve future diagnosis;
they do not retroactively repair or relabel historical measurements.

The clean baseline's separate focused C# comparison measured Medium mixed
pages on eight CPUs: one-request medians were 1077.139 ms for Rust and
1107.285 ms for C# and passed the gate; two-request batches were 2521.391 ms
and 2167.913 ms and remained inconclusive. This is two of 36 required cases on
the clean baseline, not a V1 or later full qualification.

### Rejected unconditional overlap candidate

V2 performed classification, rotation and recognition in each stable line
worker without checking whether classification already occupied the request's
CPU share. It also reused compact recognition-result slots. The bounded Rust
variant experiment compared V1 control, V2 standard release and V2 Thin LTO
with one codegen unit. It used eight logical CPUs, three cases, a fixed
Latin-square process order, one first batch, one warm-up and three measured
batches per variant/case. Every retained owned result agreed exactly.

| Input and requests | V1 staged control | V2 standard release | V2 Thin LTO |
| --- | ---: | ---: | ---: |
| Small / dense / one | 322.741 ms | 255.195 ms | 257.149 ms |
| Tiny / sparse / one | 54.225 ms | 52.735 ms | 52.987 ms |
| Medium / mixed / two | 2125.325 ms | 2748.206 ms | 2487.574 ms |

The Small standard-release median is 20.9% lower with nonoverlapping observed
ranges. Tiny's ranges overlap. Medium's standard-release batch is 29.3%
slower, and all three standard samples exceed all three control samples. The
unconditional overlap policy was therefore rejected. This combined candidate
comparison does not isolate every effect of scheduling, result caching or host
variation; it does establish a reason to avoid applying overlap universally.

Thin LTO gave no observed benefit on the Small or Tiny cases. Its Medium median
was 9.5% below V2 standard release with overlapping ranges and remained slower
than V1 control. This candidate experiment supplies no basis for changing the
repository's default release profile. The older 30.6% LTO result above is a
separate experiment whose standard executable later ran faster without LTO.

Raw V2 evidence is retained under
`target/optimization-pass-20261016/variants-v2/`. The earlier `variants/`
directory contains the same experiment and must not be counted as another run.
The three variants consumed 14.787, 14.838 and 14.548 seconds respectively,
within the 60-second budget per variant, excluding builds. V2 also passed
213/213 real-model configuration checks and its focused correctness checks;
those checks do not validate subsequent policy edits or establish performance.

### Rejected capacity-aware overlap candidate

V3 limited overlap to untraced parallel OCR with classification, batch size
one, more than one line, and a current request CPU share greater than its active
line-worker count. It kept the stages separate when classification already
filled that share. A new bounded comparison against the preserved V1 staged
control used the same three cases, one first batch, one warm-up and three fixed
measured batches per case/variant, with recorded counterbalanced process order.
Every owned result agreed exactly.

| Input and requests | V1 staged control | V3 capacity-aware candidate | Interpretation |
| --- | ---: | ---: | --- |
| Small / dense / one | 231.9040 ms | 232.1322 ms | Neutral |
| Tiny / sparse / one | 49.1273 ms | 48.2357 ms | Ranges overlap |
| Medium / mixed / two | 2060.5973 ms | 2064.1235 ms | Neutral; ranges overlap |

The V2 Small-page improvement did not recur in this separate experiment. V3
supplied no consistent end-to-end benefit, so the overlap complexity was
removed entirely for V4. The comparison does not establish a current-source
speedup, and V2's 20.9% result must not be presented as a retained improvement.
Control and candidate execution consumed 12.917 and 13.100 seconds respectively,
below their separate 60-second budgets. Raw evidence is under
`target/optimization-pass-20261016/variants-v3/`.

V3 passed 213/213 real-model configuration comparisons with zero differences
at `target/qualification/1792168936384-29300/`. Its all-feature release checks
passed 328 tests, with seven opt-in tests ignored: 297 library tests, 26
integration tests and five documentation tests passed. Its no-default-feature
library checks passed 284 tests with six ignored. These results identify V3;
the later removal of overlap requires fresh V4 validation.

### Historical V3 paired kernel controls and retained cache correction

V3 also contains a same-binary comparison of the old and revised kernel
policies. Each policy received one warm-up followed by three fixed pairs in
`AB, BA, AB` order, on one logical CPU using AVX2/FMA. Allocation and packing
preceded timing, and every compared output matched bit for bit.

| Operation | Control median | Candidate median | Observed evidence |
| --- | ---: | ---: | --- |
| Depthwise 96 channels / 152×240 / 7×7, unsegmented versus exact interior strips | 9.0695 ms | 7.6096 ms | 16.1% lower; ranges do not overlap |
| Pointwise rows 96 / inner 384 / columns 768, group 48 versus adaptive group | 0.4242 ms | 0.4151 ms | 2.1% lower; ranges overlap |
| Pointwise rows 960 / inner 512 / columns 1024, group 48 versus adaptive group | 7.4787 ms | 7.3930 ms | 1.1% lower; ranges overlap |

This provides focused evidence for depthwise segmentation. The two pointwise
results do not establish clear gains and temper the earlier separate-build
estimate. The kernel probe is diagnostic evidence on the V3 snapshot, not V4
end-to-end qualification. Raw samples, executable identity and its 15-second
external deadline are retained in `v3-paired-kernels.log` and
`v3-paired-kernels-process.json` under the follow-up artifact directory; the
process completed in 0.578 seconds.

The result-cache correction is retained independently of overlap. An early V2
reuse test failed because retained workspace grew from 222,684 to 222,685 bytes
after warming. Shrinking the per-token owned-span vector dropped token strings;
regrowing it could allocate again. The corrected slot uses one reusable UTF-8
string and compact `CtcRun` offsets, and the slot vector retains inactive
high-water storage instead of truncating it. Unicode, shrinking pages,
capture toggles and repeated-error recovery are covered by focused checks.
The original failure log and its recorded test-executable hash are preserved;
its complete source snapshot was not captured. The passing V3 checks validate
that candidate's correction. Fresh V4 compact-cache and staged-cache checks and
its complete 213-request real-model qualification also passed.

### Final staged V4 correctness and focused measurements

V4 completes classification and rotation before recognition and removes both
overlap policies and their combined-stage profiler field. It retains compact
result storage, stable width-based worker ownership, bounded CPU demand sharing,
exact mutable task partitions and the kernel changes. Standalone extraction,
traced calls and same-width batches keep their established ordering. Separate
CLS/REC wall counters apply; summed worker counters remain work measurements.

The source was frozen before the final release build. Its 84-entry runtime
manifest is `target/optimization-pass-20261016/final-runtime-hashes.json`, SHA-256
`349b483af847b2041ee5513b389a31e7ed68241ed7bb926dfd61e3e348edef21`.
The standard benchmark executable SHA-256 is
`a5ace6c92710db45fc50742f925be13291c79addc049886bc4fab09099e758a8`.
The archive embeds this source manifest and records the separate unit-test
binary and validation-log hashes. Cargo release defaults remain unchanged.

| V4 check | Result |
| --- | --- |
| All-feature release checks | 324 passed, seven ignored: 293 library, 26 integration and five documentation tests passed |
| No-default-feature release library | 283 passed, six ignored |
| Focused cache and kernel-control checks | One compact Unicode-cache, one staged-cache and one paired-control correctness test passed |
| All-target checks | Passed with all features and with default features disabled |
| Library Clippy with warnings denied | Passed with all features and with default features disabled |
| Library documentation and formatting | Passed |

The final V4 qualification at `target/qualification/1792169929310-65804/`
passed all **213/213 requests with zero differences**. Its source fingerprint is
`f0511a2a262bbf3e4a58deee00774f2dc7a8cedb626e4341d133d8e805d4d2df`,
and its runner SHA-256 is
`6cd15fbff2e66107c1b0f53be2b0d8d1ed486296fb154246cf49ba7568828523`.
The runner reports 236.638 seconds for correctness execution; that duration is
not performance evidence. A final consistency check found all 84 runtime files
matching the frozen manifest and the C# checkout clean.

The final same-binary kernel probe used one warm-up per policy and three fixed
`AB, BA, AB` pairs. All compared outputs matched bit for bit. Unlike the earlier
V3 depthwise result, **all final V4 timing ranges overlap**:

| Operation | Control median | Candidate median | Final interpretation |
| --- | ---: | ---: | --- |
| Depthwise 96 channels / 152×240 / 7×7 | 8.7246 ms | 8.3651 ms | 4.1% lower median; timing inconclusive |
| Pointwise rows 96 / inner 384 / columns 768 | 0.4113 ms | 0.4124 ms | Neutral |
| Pointwise rows 960 / inner 512 / columns 1024 | 7.3254 ms | 7.3717 ms | Neutral |

The prior V3 16.1% depthwise observation remains historical and is not pooled
with these final samples. The small kernel changes are retained with their
earlier mechanism evidence and reduced repeated work, while stable timing
benefits remain uncertain. No unchanged-code retry was performed.

The final actual Rust/C# harness comparison used Tiny/dense
(`imgs_00018069`), eight logical CPUs, one owned request, explicit C#-first
process order, one first batch, one warm-up and three measured batches:

| Metric | Rust | C# |
| --- | ---: | ---: |
| Raw wall samples | 87.1770, 90.1467, 85.6526 ms | 124.6343, 119.1821, 115.9069 ms |
| Median wall time | 87.1770 ms | 119.1821 ms |
| Peak process private commit | 35,737,600 bytes | 69,611,520 bytes |

This cell passed: latency ratio **0.7315**, memory ratio **0.5134**, zero output
differences and planned/actual order both `[csharp, rust]`. Its raw report is
`target/cpu-benchmark/1792170207580-45044/report.json`. This establishes the
recorded comparison for one case. There is no matched pre-change Rust
measurement for that case, so the Rust/C# difference cannot be attributed to
these edits. Coverage is **one of 36**, with `full_qualification_passed=false`.

The final paired probe consumed 0.572 seconds of process wall time. Including
the harness workers, total final performance execution was 1.082 seconds for
Rust and 1.159 seconds for C#, below 60 seconds per implementation, excluding
builds. Full correctness execution is recorded separately. No further pipeline
variant run followed the overlap rejection, and no final Thin LTO comparison
was performed. Cargo release defaults remain unchanged.

The follow-up archive is complete for current-source correctness and these
focused measurements. Full 36-case performance qualification, a stable global
speedup and a new packaged-consumer verification remain unclaimed. Native
ARM, Wasm and AVX-512 were not validated. Non-x86 all-target strict Clippy was
also not run; a test-only helper may have unused arguments outside its x86
block. That unverified lint limitation does not identify a production defect,
and it was left outside the frozen Windows validation scope.

## NHWC concatenation and binary-alias adoption (2026-10-17)

This pass starts from Rust `59ed32aa0495027fd58cc6eccb3e4a9aa2372101` and the
same pinned C# reference, models and corpus. The physical host is the Ryzen 9
5950X above; all measurements use release defaults, AVX2/FMA and no custom Rust
flags. The [compact archive](reports/windows-x64-2026-10-17-nhwc-concat-and-alias.json)
preserves source manifests, executable hashes, every raw timing sample, output
digests, allocation/layout statistics and the fresh qualification metadata.

Two changes are retained:

- Rank-four channel concatenation can continue an NHWC segment. The bound
  descriptor retains the normalized logical axis, and exclusive pixel bands
  copy each operand's contiguous channels. Other axes, constant operands and
  canonical segments retain their established paths, including eligible
  zero-copy canonical concatenation.
- Elementwise placement can reuse either dying full-shaped binary operand,
  including vector-bias broadcasts and dead full-range alias groups. Constants,
  live sibling views and partial self-broadcast overlaps are excluded. The
  in-place executor preserves operand direction and the separate bias addition
  after MatMul accumulation.

The comparison freezes three versions: unchanged baseline, concat only, and
concat plus aliasing. Their runtime manifests contain 84, 86 and 88 files.
Build outputs are isolated per source snapshot. The graph probe uses public
production `Session::run`, deterministic raw NCHW inputs, one warm-up and three
measured calls per case on one allowed CPU. OCR workers use identical saved
jobs and decoded pixels, eight allowed CPUs, one first batch, one warm-up and
three measured batches. Variant order rotates across the three cases; each
case still has only one process order. The separate throughput case uses two
concurrent owned requests in combined, baseline, concat order.

| Workload | Baseline median | Combined median | Observed interpretation |
| --- | ---: | ---: | --- |
| Tiny DET graph, 320x320, one CPU | 20.1735 ms | 19.3853 ms | 3.91% lower; ranges do not overlap |
| Small REC graph, 48x128, one CPU | 11.3617 ms | 11.3570 ms | Ranges overlap |
| Medium REC graph, 48x320, one CPU | 112.2608 ms | 116.6902 ms | 3.95% higher; ranges do not overlap |
| Tiny dense OCR, eight CPUs | 104.5484 ms | 104.5795 ms | Ranges overlap |
| Small dense OCR, eight CPUs | 248.8878 ms | 245.9191 ms | Ranges overlap |
| Medium sparse OCR, eight CPUs | 438.5389 ms | 408.9173 ms | 6.75% lower; ranges do not overlap |
| Medium mixed OCR, two requests, eight CPUs | 2144.9722 ms/batch | 2130.1643 ms/batch | Ranges overlap; throughput inconclusive |

Concat alone reduces Tiny DET graph median time to 19.1316 ms (5.16% lower).
Graph layout conversions fall from 10 to 2 for Tiny DET, 6 to 3 for Small REC,
and 5 to 2 for Medium REC. The combined Medium sparse OCR median is also 3.91%
below concat alone, with nonoverlapping observed ranges. The raw recognition
graph executes its full dense output head; OCR uses compact projection, so the
graph regression remains a separate, recorded tradeoff.

All graph outputs and all compared owned OCR results match exactly across the
three versions. Retained activation capacities are unchanged in every measured
case; no general arena-memory reduction is established. Medium mixed throughput
is 0.932413 versus 0.938895 images/second, with overlapping ranges. Fixed samples,
one process order per case and bounded warm-up do not establish statistical
significance, steady state or a global speedup.

All 21 benchmark processes completed successfully. Aggregate execution,
including process startup, model setup, first/warm-up batches and output checks,
was 17.2684 seconds for baseline, 16.9114 for concat and 17.0210 for combined,
below 60 seconds per version. Builds and correctness inference were idle during
performance runs. No unchanged case was retried or expanded for cleaner timing.

Fresh validation of the retained production source:

| Check | Result |
| --- | --- |
| Default release library | 302 passed; six opt-in diagnostics ignored |
| Release integration contracts | 26 passed; one opt-in live harness test ignored |
| Release documentation tests | Five passed |
| No-default-feature release library | 293 passed; six diagnostics ignored |
| Full real-model C# qualification | 213/213 requests; zero differences |
| All-target/all-feature compilation | Passed |
| All-feature library Clippy, warnings denied | Passed |
| Formatting and diff whitespace | Passed |

The full qualification report is
`target/qualification/1792204343852-22884/run.json`, with `coverage=full`,
`status=passed` and an empty differences list. Its 271.3455-second correctness
duration is excluded from performance measurements. Frozen source snapshots,
runner scripts, raw outputs and logs are under `target/cpu-adoption-20261017/`;
the compact archive preserves the portable evidence.

AVX-512 and ARM64 kernel changes were deferred because the host cannot validate
them natively. Crop changes remain lower priority based on the prior stage-share
evidence. The complete 36-cell performance qualification remains open.

### Row-scalar in-place follow-up (2026-10-17)

Review found that expanded alias eligibility sent canonical row-scalar broadcasts
through the generic coordinate-division loop. The correction uses the bound
broadcast descriptor to select a row-wise SIMD path, preserving operand direction,
global row offsets across parallel bands and input-buffer reuse.

One synthetic Add graph, `[1,16,256,256] + [1,16,256,1]`, was compared on the same
physical host with release defaults, AVX2/FMA and one pinned CPU. Each version had
one warm-up and three measured calls. Pre-fix samples were 9.0018, 9.4938 and
9.1833 ms; fixed samples were 1.3970, 1.3569 and 1.2716 ms. The median fell from
9.1833 to 1.3569 ms. All output hashes matched exactly, and the fixed activation
plan still uses 4,194,304 bytes. Fixed-version process execution was 26.6872 ms;
the raw reports are under `target/review-row-broadcast/`.

Release planner validation passed 59 tests with two opt-in diagnostics ignored.
Coverage includes all four binary operations, both operand orders, SIMD tails,
unequal parallel row bands, mixed-layout fallback and repeated production-session
calls. Release library Clippy with warnings denied, formatting and diff whitespace
checks passed. This focused synthetic comparison does not repeat the archived
213-request OCR qualification or establish a general OCR speedup.
