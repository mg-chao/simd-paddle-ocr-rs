# Windows x64 CPU rewrite progress

The active frozen Rust source is
`d4a381343691afd097fdaa436cb9306f522d71e996dccd4256aee86bf594adf5`
(83 files), against C# `33ff8a81672452041d110e8ba52dcd2639e54228` and
`nightly-2026-09-08` on Windows x64. **30 focused NCHW tests pass**, including
independent cancellation, complete spatial packet, worker-child family,
Portable4 dual-tile, shared-panel, exact-store and simulated ISA selection tests.
The source is frozen for attempt4; full core and Release-library checks,
seven-model parity, policy/thread variants, real batching, unavailable-AVX-512
rejection, Release static audits and packaging are **pending**. Focused tests
do not substitute for those source/producer-bound gates.

The current core run has completed **273/265 library tests (288/280 all-target
checks)** with all features and without default features respectively, six documentation tests
per mode, both all-target Clippy checks with warnings denied, and **113 tooling
contracts**. Its final Release-example stage is still compiling, so the whole
core gate remains pending. These completed leaves do not qualify model parity,
Release-library, static, hardware or performance gates.

The actual Debug static audit bounds the compiled fixed/logical kernel chain
at **1,776,512 bytes**, below the unchanged 2 MiB limit. It includes at most
11 logical visit levels and a 256 KiB outer/framework/caller allowance. That
entry allowance is an assumption; this proof does not establish a global bound
on physical Rayon helping, nested requests or callback recursion. Actual
bounded-worker and concurrency tests are separate required core checks. The
initial 2,162,480-byte sum across mutually exclusive worker action types remains
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
The independent cancellation regression failed before this correction and passed
afterwards; RED and after-run GREEN captures remain explicitly distinguished.
No tolerance or thread-equality requirement changed.

Earlier Portable4 compact projection and column-shard fixes remain present:
complete packed 16-column reductions start at zero then add bias, scalar
remainders stay bias-first, and column shards own only complete tiles plus one
last scalar tail. Failed 1397 bounds evidence remains historical and retained.

M5 phase instrumentation now separates Rust `reshape_ns` work from binding.
The maintained C# exporter has ten passing model-free phase contracts and a
separate diagnostic phase process. C# returns owned results intrinsically;
external post-Run copying is zero by that API contract, while lazy text
publication is observed separately. These changes have **no new real-model
phase or performance measurement**. Historical unmeasured fields remain null.

| Milestone | Implemented / observed | Remaining acceptance |
| --- | --- | --- |
| M1 reproducible validation | Pinned seven-model harness and [draft PR #2](https://github.com/mg-chao/simd-paddle-ocr-rs/pull/2); both named Windows checks passed for published head `df04c959...` (historical ba0 source). Main now requires both exact checks from Actions app 15368, with strict and administrator enforcement verified. | Re-run CI for the corrected PR head and finish attempt4 exact-source local gates. Earlier CI success does not qualify changed source. |
| M2 CPU APIs | Character boxes, borrowed/owned results, model sets/builder/shared context, classifier limits, packed text/hash helpers and recognition batching | Preserve behavior under fresh integrated and batching validation for the active source. |
| M3 optimization families | Specialized source families, observable selection, fused stores, bounded reusable storage and independent focused regressions | Finish active-source integrated and Release static gates. Historical generic PackedStride2 findings do not qualify its replacement; generic PackedPointwise audits retain overhead; general >25-tap Portable4 geometry retains its explicit fallback. No universal register-only or performance claim. |
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
frozen model/workload at matched total CPU budgets 1/2/4/8. The first five
original-sample rounds (ordinary three plus the explicit two-round exception)
are consumed. The user authorized **one additional complete 72-case round**;
the append-only grant `42014eec-5933-488f-8879-41dc67a0da8b` raises the task cap
to six and preserves all prior reservations. The sixth round is **unreserved
and unmeasured**. It may run after current-source correctness, static and package
gates, under the fixed `normal-background-v3` policy, with the separate phase
evidence retained. Recording authorization does not establish any performance
result or consume the round.

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
remain separate; earlier passes do not qualify the corrected source.
