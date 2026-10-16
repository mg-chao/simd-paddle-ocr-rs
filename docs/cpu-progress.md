# Windows x64 CPU rewrite progress

The active frozen Rust source is
`d4a381343691afd097fdaa436cb9306f522d71e996dccd4256aee86bf594adf5`
(83 files), against C# `33ff8a81672452041d110e8ba52dcd2639e54228` and
`nightly-2026-09-08` on Windows x64. **30 focused NCHW tests pass**, including
independent cancellation, complete spatial packet, worker-child family,
Portable4 dual-tile, shared-panel, exact-store and simulated ISA selection tests.
Attempt4 correctness, available-policy checks, scoped Debug/Release static
reviews and the preperformance package have passed for these exact source bytes.
The authorized performance round completed 37/72 targets, so M5 remains unmet.
Final package authority and capable-Windows AVX-512 execution remain separate
acceptance gates.

The exact source has passed **273/265 library tests (288/280 all-target
checks)** with all features and without default features respectively, six
documentation tests per mode, both all-target Clippy checks with warnings denied,
**113 tooling contracts**, and **273 Release-library tests**. Fresh model gates
passed seven exact imports and **447 comparisons**, **1,341 cases across nine
policy/layout/thread configurations**, nine real-batch comparisons, and explicit
unavailable-AVX-512 rejection. Current core and regression completion receipts
bind the actual logs, source snapshot and closed producing executable/PDB pairs.
These results establish correctness on available hardware, separately from speed
and capable-Windows AVX-512 acceptance.

The actual Debug static audit bounds the compiled fixed/logical kernel chain
at **1,776,512 bytes**, below the unchanged 2 MiB limit. It includes at most
11 logical visit levels and a 256 KiB outer/framework/caller allowance. That
entry allowance is an assumption; this proof does not establish a global bound
on physical Rayon helping, nested requests or callback recursion. Actual
bounded-worker and concurrency core tests passed separately; they do not make
this a global physical-stack theorem. The initial 2,162,480-byte sum across mutually exclusive worker action types remains
retained alongside the exhaustive 632-frame grouping and scoped posterior proof.

The prior `7f341...` attempt2 passed one-thread seven-model correctness, but
failed the exact Auto two-thread logical trace at medium DET `Conv.121`:
1,600 of 4,096 intermediate values differed, while the final tensor was exact.
Its old Packed3 path used four-position FMA on rows and tails where pinned C#
uses scalar separate multiply/add. The corrected eight-output family now uses
complete interior spatial packets of 4/8/16 lanes for Portable4/AVX/AVX-512,
with scalar borders and remainders. The paired16 family retains spatial octets.
Packed3 workers now use floor OC8 bands and reselect eligible native paired16
children. Packed stride-two has dedicated OC8/native OC16 bodies and Portable4
adjacent 4+4 spatial tiles; its worker split has no generic work threshold.
Independent regressions failed before each correction and pass afterwards.
RED and after-run GREEN captures remain explicitly distinguished.
No tolerance or thread-equality requirement changed.

Earlier Portable4 compact projection and column-shard fixes remain present:
complete packed 16-column reductions start at zero then add bias, scalar
remainders stay bias-first, and column shards own only complete tiles plus one
last scalar tail. Failed 1397 bounds evidence remains historical and retained.

M5 phase instrumentation now separates Rust `reshape_ns` work from binding.
The maintained C# exporter has ten passing model-free phase contracts and a
separate diagnostic phase process. C# returns owned results intrinsically;
external post-Run copying is zero by that API contract, while lazy text
publication is observed separately. The completed authorized paired round retains
and revalidates separate actual phase captures. Historical unmeasured fields
remain null; intrinsic zero-copy and lazy-text diagnostics remain distinct.

| Milestone | Implemented / observed | Remaining acceptance |
| --- | --- | --- |
| M1 reproducible validation | Maintained pinned seven-model harness; exact-source local core/full/variant/batch gates passed. Both named Windows checks passed for current published D4 head `fe639067...` in [draft PR #2](https://github.com/mg-chao/simd-paddle-ocr-rs/pull/2). Main requires both from Actions app 15368 with strict/admin enforcement verified. | Bind the final documentation publication and its corresponding CI run; retain final evidence. No earlier head substitutes for a later changed tree. |
| M2 CPU APIs | Character boxes, borrowed/owned results, model sets/builder/shared context, classifier limits, packed text/hash helpers and recognition batching; fresh behavioral/core/full/batch gates passed. | Preserve these contracts in the final package. |
| M3 optimization families | Active specialized source families, observable child selection, fused stores and reusable bounded storage; fresh integrated and selected Release static gates passed. | Retain documented spill/helper/fallback limits. Static coverage establishes no universal register-only or measured speed claim. |
| M4 ISA dispatch | Scalar, Portable4, AVX, AVX/FMA and AVX2/FMA available-policy checks passed; compiled AVX-512 and unavailable-policy rejection verified. | Actual capable-Windows AVX-512 execution remains explicitly user-deferred. Compilation does not qualify hardware execution. |
| M5 performance / release | Correctness/static prerequisites and the saved 131-member preperformance package passed. The fixed sixth round completed 37/72 targets (18/36 latency, 19/36 throughput); phase evidence is retained. | Latency/throughput acceptance remains unmet. All six rounds are consumed with no further measurement authorization. Bind the final documentation package and archive; hardware acceptance remains separate. |

The current Release static gate is closed by
`target/cpu-active-coverage-attempt4-production-codegen-20261016/complete.json`
(SHA-256 `33aa811584ef731d45dd22e3273d9e5d74f1011924bb9eac7844de0a78df695f`),
with 348 hash-bound evidence files. It binds the actual current executable/PDB,
source, native and Portable4 reductions, common-store/transpose bindings,
17 selected NCHW numerical tiles and 20 selected Gelu/store control-flow paths.
These are selected path proofs: canonical packed paths retain documented spills,
bounds checks and coefficient-tail helper overhead. They do not establish that
every kernel is register-only, all epilogue paths are call-free, or AVX-512 ran.

The preperformance package passed with **131 members**. Its saved
`preperformance-package.crate`, log and `preperformance-package-report.json`
under `target/core-validation-active-coverage-attempt4-20261016` preserve the
exact premeasurement documentation and member hashes. This is an immutable
prerequisite snapshot, not the final release package. Final package qualification
requires a new successful observed Cargo package command and
`package-report.json` binding the final documentation and shipped bytes.

The prior source
`f7bc2ce6761f4521e9492468fe6c201b6a96e5526b5963250d878ee898c2242c`
is **historical**. It passed 267/261 all-target tests (252/246 library), six
documentation tests per mode, Clippy both modes, 86 tooling contracts, 252
Release-library tests, seven exact imports/447 comparisons, 1,341 cases across
nine available policy/layout/thread configurations, nine real-batch comparisons,
unavailable-AVX-512 rejection, scoped Debug/Release audits and 117-member
packaging. Its evidence suffix is `canonical-stores-20261016`; these passes do
not qualify the changed active source. The historical Debug stack bound was
1,946,832 bytes. Its selected static proofs do not establish universal clean
loops or measured performance.

The last completed performance round, historical source `54a1e088...`, passed
**36/72 targets**: 17/36 latency and 19/36 throughput. Historical `a0bcf050...`
passed 65/72. Neither establishes the active source's speed. Fixed acceptance
remains warmed median Rust latency <=110% of C# and throughput >=90% for every
frozen model/workload at matched total CPU budgets 1/2/4/8. The first five
original-sample rounds remain consumed. The append-only grant
`42014eec-5933-488f-8879-41dc67a0da8b` authorizes one additional complete round,
raises the task cap to six and preserves all prior reservations.

The authorized sixth full 72-case performance round completed **37/72 targets**:
**18/36 latency** and **19/36 throughput**. **M5 remains unmet.** The fixed gates
remain warmed median Rust latency <=110% of C# and throughput >=90%, measured
under `normal-background-v3` with two warmups, five samples and one iteration
at matched total CPU budgets 1/2/4/8. All public output comparisons and same-kernel
thread determinism checks passed; measured repeated-shape Rust retained storage
had zero growth. The eight-partition report and separate phase evidence are
retained and revalidated, including every nonpassing result.

Actual completion is bound by
`target/cpu-sixth-round-attempt4-20261016/performance-completion.json`
(SHA-256 `07054326e6aca8d3e792a7ea8969913c399bcfdcfa51a338fecb943f1de5e9d0`),
round `c7db05b3-e51f-48ae-b20a-8d75d5be9b24` and frozen plan SHA-256
`43f9c5ed6c79ec8f24390f25ef949c95cfdbae387d8baa77e593da9c534a11bf`.
**All six authorized rounds are consumed; no further performance round is
authorized.** No threshold, sample schedule, tolerance or prior ledger entry was
changed to improve the reported outcome.

The numerical contract remains tensor atol=rtol=0.001, score error<=0.01 and
coordinates<=0.5px. Text, labels, counts, order, dimensions, CTC span boundaries,
frozen pixels and same-kernel thread-count results compare exactly. No tolerance,
workload, schedule or ledger entry is reset. Normal desktop load is allowed by
the already recorded `normal-background-v3` policy; it does not reinterpret
historical results. The waiting automation is paused.

Large evidence remains ignored or external. The existing archive at
`D:/simd-paddle-ocr-workspace/cpu-rewrite-evidence-20261015` retains historical
proofs. A new attempt4 archive must bind its own actual source, producers, logs,
reports and manifest. Failed attempt2, canceled attempt3, the Packed3/stride-two
RED and after-run GREEN captures, and the initial Debug overinclusive rejection
remain separate. The v2 wrapper's Python-resolution failure occurred before
Python, preflight or reservation; its preserved receipt does not count as a
performance run. The corrected v3 wrapper launched the sixth round, whose completed nonpassing
verdict is retained. Earlier passes do not qualify the corrected source.
