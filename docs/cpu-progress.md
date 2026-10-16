# Windows x64 CPU rewrite progress

The active frozen Rust source is
`7f341123f48d3d82ad22b970ac627d296b17d8ffda3a08fe9a6ca38f4149a077`,
against C# `33ff8a81672452041d110e8ba52dcd2639e54228` and
`nightly-2026-09-08` on Windows x64. Its completed local core checks passed
**279/272 all-target tests** (264/257 library), **six documentation tests in
each feature mode**, Clippy with warnings denied in both modes, **113 tooling
contracts**, and the Debug example. Its source/PE/PDB-bound Debug stack audit
also passed: the conservative maximum is **1,997,512 bytes** (compact projection
1,740,264), below the ordinary 2 MiB worker stack. Release-library, fresh all-model parity,
forced-policy variants, real batching, unavailable-AVX-512 rejection, scoped
Release static audits and packaging for these exact bytes are **pending**. Their
completion must be established by actual logs and source/producer-bound reports.
Core tests alone do not qualify this source for all-model or release acceptance.

This source adds a native single-thread paired 16-output 3x3 convolution family
for output counts >=64 and divisible by 16. Portable4 compact projection now
uses zero-initialized full 16-column reductions followed by bias, with scalar
remainder columns preserving bias-first separate multiply/add. A rejected
intermediate candidate (`1397` evidence) passed 261/263 library tests but failed
two column-shard bounds regressions. Its failed logs and producers remain
retained; the active core gate includes the corrected full-tile bounds.

M5 phase instrumentation now separates Rust `reshape_ns` work from binding.
The maintained C# exporter has ten passing model-free phase contracts and a
separate diagnostic phase process. C# returns owned results intrinsically;
external post-Run copying is zero by that API contract, while lazy text
publication is observed separately. These changes have **no new real-model
phase or performance measurement**. Historical unmeasured fields remain null.

| Milestone | Implemented / observed | Remaining acceptance |
| --- | --- | --- |
| M1 reproducible validation | Pinned seven-model harness, independent pixel/arithmetic oracles, import/logical/public/complete OCR comparisons, first-divergence reports, missing-evidence failures and Windows workflow | Finish exact-source all-model gates. As observed on 2026-10-16, remote main has no classic branch protection, repository rulesets are empty, and only the old Rust core workflow is registered. The new Windows workflow remains unpublished. Publish and pass both named checks, then require them on main. |
| M2 CPU APIs | Character boxes, borrowed/owned results, model sets/builder/shared context, classifier limits, packed text/hash helpers and recognition batching | Preserve behavior under fresh integrated and batching validation for the active source. |
| M3 optimization families | Specialized source families, observable selection, fused stores, bounded reusable storage, focused independent regressions and the actual Debug stack audit | Finish active-source integrated and scoped Release static gates. Historical PackedStride2 and generic PackedPointwise audits retain overhead; general >25-tap Portable4 geometry retains its explicit fallback. No universal register-only or performance claim. |
| M4 ISA dispatch | Scalar, Portable4, AVX, AVX/FMA, AVX2/FMA and compiled AVX-512 paths | Finish available-policy checks for this source. Actual capable-Windows AVX-512 execution remains explicitly user-deferred. Compilation does not qualify hardware execution. |
| M5 performance / release | Phase accounting and license/package tooling implemented | Current-source performance and packaging remain unqualified. Complete latency/throughput targets, actual phase evidence, hardware validation and exact-source packaging are still required. |

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
frozen model/workload at matched total CPU budgets 1/2/4/8. All five authorized
original-sample rounds (ordinary three plus the explicit two-round exception)
are consumed. The append-only one-round followup recorder is implemented, but
the new human grant is **pending and has not been recorded**. No further timing
run, including a real-model phase-only probe, is currently authorized.

The numerical contract remains tensor atol=rtol=0.001, score error<=0.01 and
coordinates<=0.5px. Text, labels, counts, order, dimensions, CTC span boundaries,
frozen pixels and same-kernel thread-count results compare exactly. No tolerance,
workload, schedule or ledger entry is reset. Normal desktop load is allowed by
the already recorded `normal-background-v3` policy; it does not reinterpret
historical results. The waiting automation is paused.

Large evidence remains ignored or external. The existing archive at
`D:/simd-paddle-ocr-workspace/cpu-rewrite-evidence-20261015` retains historical
proofs. A new attempt2 archive must bind its own actual source, producers, logs,
reports and manifest; it must not promote earlier evidence to current acceptance.
