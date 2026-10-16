# Windows x64 CPU validation evidence

Local validation on 2026-10-16 uses the pinned `nightly-2026-09-08` toolchain,
.NET SDK 10.0.303/runtime 10.0.11, the pinned C# reference
`33ff8a81672452041d110e8ba52dcd2639e54228`, and the unchanged baseline manifest
SHA-256 `0ecc7c2634c3da2d2bee2ee7021d6abe91f2685cc9b64fc34d8db6ebcf5e440e`.
The host is Windows x64, AMD Ryzen 9 5950X, with 24 available logical CPUs;
automatic inference selects AVX2/FMA. This report does not qualify AVX-512 or
another operating system or architecture.

The final runtime passes correctness and packaged-consumer checks. The complete
performance comparison remains **inconclusive: 21 of 36 cases passed and 15 are
unresolved**. Seven two-request throughput medians are below the agreed target;
the remaining work cannot be described solely as timing noise.

| Milestone | State |
| --- | --- |
| M1: independent validation | Complete: isolated C# builds/workers, process deadlines and canonical comparison |
| M2: CPU configurations | Complete: all 213 real-model configuration/reuse/recovery cases pass |
| M3: matched performance baseline | Tool and complete 36-case matrix implemented; performance exit gate unresolved |
| M4: measured optimization | Arena placement and targeted kernels implemented; global performance exit gate unresolved |
| M5: usable release | Example, documentation, license texts, package and all three packaged consumers verified; performance completion is not claimed |

## Correctness and distribution

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

## Performance completion gate

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
