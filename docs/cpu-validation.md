# Windows x64 CPU validation

`tools/cpu_parity.py` builds the maintained C# friend-assembly exporter and Rust
`cpu_parity` example, captures both implementations, and compares their results.
It requires Python 3.10+, Git, .NET 10, and the repository-pinned
`nightly-2026-09-08` Rust toolchain. No image codec, host font, or Python package
is required. Tests never skip missing models or oracle outputs.

## Pinned inputs and commands

Use a clean C# checkout at revision
`33ff8a81672452041d110e8ba52dcd2639e54228`. `tools/reference-lock.json` pins all
seven ONNX files, both dictionaries, the original JPEG, and error budgets.
`tools/corpus-source/sample.json` pins the decoded BGR sample and its decoder
provenance. Corpus generation uses those checked bytes and an embedded bitmap
font; every generated file has recorded dimensions and SHA-256.

From the Rust repository, with the reference checkout alongside it:

```powershell
python -m unittest discover -s tools -p 'test_*.py' -v
python tools/cpu_parity.py --reference-dir ../SimdPaddleOCR --models-dir ../SimdPaddleOCR/models --output target/cpu-parity-quick --quick --threads 1 --correctness-only
```

The output directory must be new or empty. Quick validation has 67 cases per
configuration and all seven models at initial, larger, and repeated shapes.
Remove `--quick` for the 149-case pre-release corpus, which adds skewed and
rotated text, multiple lines, vertical text, long lines, noise, and narrow
images. `--correctness-only` executes each captured output once, plus required
logical traces. It omits warmups, timing repetitions, and extra production
profiles, and labels benchmark records `measured: false` without fake zero
samples. Load/reshape and diagnostic node timings remain observational.

```powershell
python tools/cpu_parity.py --reference-dir ../SimdPaddleOCR --models-dir ../SimdPaddleOCR/models --output target/cpu-parity-full --threads 1 --correctness-only
```

Reuse checked C# captures while testing a new Rust build:

```powershell
python tools/cpu_parity.py --replay target/cpu-parity-quick --models-dir ../SimdPaddleOCR/models --output target/cpu-parity-replay --quick --threads 1 --correctness-only
```

Replay requires the original lock, model hashes, corpus identity, workload, and
thread budget. Output must be outside the input bundle. Old bundles without
the seven import contracts fail; regenerate them with the maintained driver.

The new batching API also has a short all-three-recognizer gate. It submits
`sample,tiny,blank,sample` in one width-64 graph, checks input order, scores and
CTC spans, and requires each backend's batch results to match its standalone
results exactly:

```powershell
python tools/cpu_parity.py --reference-dir ../SimdPaddleOCR --models-dir ../SimdPaddleOCR/models --output target/cpu-batch --quick --batch-only --threads 1 --correctness-only
```

This also performs all seven import comparisons. A batch-only capture does not
replace the complete graph/public/OCR correctness gate. Independent 18-case C#
character-box fixtures under `tests/data/` run in core tests without models.
`--ocr-batch-size 4` also exercises grouping in the complete OCR gate (default
one). Replay may compare a new batch size against stored single-line reference
outputs, whose text/scores/spans must be invariant. Captures record the actual
batch size; performance fingerprints keep differing batching policies distinct.

The unary policy tests also retain independent binary32 goldens for Erf, GELU
and Sigmoid. Regenerate their source arithmetic oracle without running Rust or
model inference:

```powershell
python tools/generate_unary_policy_oracle.py --reference-source ../SimdPaddleOCR/src/Sdcb.SimdPaddleOCR/Kernels/Tensor.Avx.cs --output target/unary-policy-oracle.json
```

The generator checks the pinned reference content and rounds each primitive
directly from exact rational arithmetic, including single-rounding FMA. These
are arithmetic goldens, not captured C# execution. Forced portable/AVX policies
keep Erf multiplication and addition separate; FMA policies fuse its Horner
steps. Vector GELU uses the reference inverse-sqrt2 multiplication, and vector
Sigmoid uses nonfused exponential arithmetic. Optimized scalar tails retain the
selected policy so changing a worker partition preserves output bits. The
scalar diagnostic path and public CTC score computation remain independent.
The convolution GELU regression checks exact independent native values and the
fixed tensor contract against its unfused rounded-sqrt2 division graph, whose
rounded intermediates need not be bit-identical.

Portable4 softmax has separate exact goldens for the pinned generic vector
path, including its four-lane denominator and reciprocal multiplication:

```powershell
python tools/generate_portable_softmax_oracle.py --reference-dir ../SimdPaddleOCR/src/Sdcb.SimdPaddleOCR/Kernels --output-dir target/portable-softmax-oracle
```

This generator reuses the independent rational-rounding primitives and pins
both generic reference files. Its axes cover 3/4/5, full vectors and remainders;
the selected scalar tails equal their row maximum, so their exponential is
exactly `MathF.Exp(0)=1`. These are arithmetic oracles, with no C# or Rust model
execution. Tests also require exact results under bounded row partitions and
preserve the independent Scalar policy.

## Required contracts and diagnostics

Before inference, the driver compares raw and normalized imports for all seven
models. Raw contracts cover IR/opsets, interfaces, value metadata, initializers,
node order/edges, all attributes, and constant hashes. Normalized contracts
compare pre-layout topology, tensor dtypes/dimensions, all reachable constant
hashes, and the reference's effective parameter blocks. Generated tensor names
and tensor numbering are replaced by graph-input or producer-node identities.
Unknown/symbolic dimensions use the reference's `-1` convention. The C# Resize
block contains scales only; Reshape and Slice have no parameter blocks. Their
source attributes remain covered by raw contracts, and inference checks their
behavior. No reference source is patched. The adapter uses the pinned internal
reader/records and fails when that contract changes.

The three required inference comparisons are C# scalar versus Rust scalar,
C# optimized versus Rust optimized, and Rust scalar versus Rust optimized.
Capture files include graph outputs at three shapes, all logical nodes at the
initial shape, DET/CLS/REC preprocessing, perspective crops, standalone public
components, compact CLS/CTC heads, and complete tiny/small/medium OCR. The
original scalar process disables hardware intrinsics, layout conversion, and
fusion. Optimized C# enables NHWC/fusion and currently disables AVX-512 for the
AVX2 baseline; AVX-512 hardware acceptance is a separate milestone.

Missing cases, traces, payloads, malformed types, nonfinite numbers, changed
checksums, and shape mismatches fail. Tensor errors use
`atol=rtol=0.001`; public score error is at most `0.01`, coordinate error at
most `0.5 px`. Text, labels, counts, order, dimensions, and CTC span boundaries
are exact. Frozen tiny-image pixel oracles are bitwise exact and run without
models in `tests/pixel_oracle.rs`. Tolerances never widen automatically.

`import-report.json` reports the first import difference.
`report.json` reports the first differing logical node/tensor element and
numeric maxima. `operator-inventory.json` and `stage-profiles.json` are
diagnostic profiles from separate instrumented executions. Their timers include
instrumentation and bounded-worker dispatch, so they are not production latency.

`build-provenance.json` is written before inference. `run.json` records the
reference, model, dictionary, corpus, compiler, flags, OS/CPU, and thread budget.
Working-tree source hashes include uncommitted bytes and the import adapters.
The driver checks source stability during compilation, then copies the verified
Rust executable and complete C# runtime into `binaries/` inside the bundle.
Those frozen artifacts run even if development resumes in the checkout.
`--no-build` explicitly labels preexisting binaries unverified; it is unsuitable
for release acceptance. Preserve the complete bundle outside Git.

## Stages, threads, and performance

`--stages` adds SIMD/NCHW without fusion and NHWC without fusion.
`--backend scalar`, `portable4`, `avx`, `avx-fma`, `avx2-fma`, or `avx512` forces optimized Rust stages;
unsupported forced policies fail. Scalar numerical oracles always use one
thread; `--threads` sets the shared optimized CPU budget in both implementations.

After a full passing one-thread capture, `tools/cpu_variants.py` validates
intermediate stages, Portable4, and native budgets 2/4/8 against its frozen
scalar/native baseline. Native thread-count comparisons use zero numerical
tolerance. It requires verified unchanged sources/binaries, exact host identity,
and the complete 149-case corpus. Run `python tools/cpu_variants.py --help` for
the required bundle/model/output paths.

On an AVX2/FMA-capable Windows host, include explicit AVX without FMA, AVX/FMA,
and AVX2/FMA execution gates:

```powershell
python tools/cpu_variants.py --bundle target/cpu-parity-full --models-dir ../SimdPaddleOCR/models --output target/cpu-variants --backends avx,avx-fma,avx2-fma
```

Add `avx512` to `--backends` on a capable Windows host. Forced unavailable ISAs
fail and never silently fall back. AVX-512 release acceptance requires this real
execution report in addition to compilation and unsupported-policy tests.

`cpu_bench` is the separate interleaved Rust benchmark, with balanced variant
order, cached raw input, configurable warmups/batches, isolated CPU contexts,
allocation/memory statistics, and checkpointed output. Run
for example:

```powershell
cargo run --release --locked --example cpu_bench -- --models-dir ../SimdPaddleOCR/models --corpus target/cpu-parity-quick/corpus/corpus.json --output target/cpu-bench/stages.json --threads 1 --variants reference,simd,nhwc,auto --warmups 2 --batches 12 --iterations 1 --min-batch-ms 100
```

Its existing stage-speedup gates are diagnostic implementation milestones;
they do not claim the final C# latency/throughput targets have passed. Final
acceptance requires matched C# workloads and total budgets 1/2/4/8, warmed
median latency ≤110%, throughput ≥90%, and bounded retained memory. Keep
load/compile/reshape/cold/ownership costs separate and exclude tracing and
serialization. Follow `AGENTS.md`: at most three performance rounds per sample
for the entire task; start with representative small inputs.

## Paired release performance acceptance

`tools/cpu_acceptance.py` implements the separate M5 gate. It requires a passing
149-case bundle for the unchanged library sources, validates all seven imports
and model identities, freezes both executables, and records the schedule before
any official measurement. It is a manual Windows x64 command; CI only tests the
gate's contracts and does not assert timings on shared runners. An untimed
Windows process-load check runs before preflight/round reservation and every
paired budget. The user-authorized `normal-background-v3` policy permits up to
6.0 occupied logical cores in aggregate, 3.0 for any ordinary process, 3.0
unattributed, and 0.5 for each compiler/linker. These limits allow normal desktop
activity on this 32-thread host; the recorded samples still expose background
load. Exceeding a limit fails with a retained snapshot by default. Optional
`--wait-for-idle` waits with 30-second quiet gaps only before round reservation,
checks frozen inputs around every poll, and executes preflight once. Contention
after reservation fails immediately; timed rounds never repeat automatically.
The helper's policy and source identity are part of the immutable plan. It
records process names/PIDs and CPU deltas, without command lines. Hosts above
64 active logical processors currently fail this idle-proof check because
Windows system counters cover only the caller's processor group; the helper
does not claim to measure idle capacity across those groups.

The three fixed raw workloads are the original 320×320 sample, a dense 320×640
page made from two vertical copies of that multilingual sample, and a 928×128
page with three long lines. Each long line joins four unscaled copies of the
sample's fixed glyph rectangle `[15,250,240,270]` on a white background. Their
dimensions, bytes, derivation, and SHA-256 values
are frozen in `plan.json`. Before timing, both implementations must agree on all
nine model/workload public results. Dense output must contain at least 32 lines
with both Unicode and ASCII text; the long workload must actually exercise a REC
tensor width of at least 1280 and retain at least three lines. Failed
qualification stops before timing. The correctness corpus's synthetic 1024×48
line is excluded from performance acceptance: the tiny detector finds no text
on it, so it does not qualify as an all-model long-page workload.

Latency uses one persistent request with total CPU capacity 1, 2, 4, or 8.
Throughput separately uses N persistent serial requests with aggregate capacity
N: Rust forks share compiled models/packs and each receive a private capacity-one
context; C# uses one thread-safe pipeline with `DOTNET_PROCESSOR_COUNT=1` and
explicit DET/CLS/REC `MaxPooledSessions=N`. All inputs are cached raw BGR, CLS is
enabled, DET limit is 960, and recognition batching is one. Both return owned
public results. Rust also reports borrowed inference and owned conversion costs.

Each configuration uses two fixed warmup batches and five official batches by
default, with one request per worker per batch. Persistent workers are bound
before timing; ready/start/finish barriers delimit each batch. No decoding,
profiling, hashing, output checking, JSON writing, or progress writes occur in
the official timed region. Engine order alternates deterministically between
configurations. There is no adaptive calibration or automatic repetition.

The plan records the inherited `DOTNET_` and `COMPlus_` tiered compilation,
PGO, QuickJit, on-stack replacement and ReadyToRun controls, plus any related
JIT controls present in the environment. Explicit null entries mean unset,
not disabled. Each C# capture records its actual framework/version/architecture
and the same environment; the runtimeconfig bytes and properties are retained.
This records configuration without changing it. Two warmups do not prove that
JIT tiers or PGO have settled. Preserve all five samples and their fixed workload
order; within-case timing changes do not justify discarding samples, changing
thresholds, or automatically repeating a round.

For **every** model/workload at every CPU budget, warmed Rust median owned
latency must be ≤110% of C#, and Rust concurrent throughput must be ≥90% of C#.
Each engine must preserve its one-thread outputs exactly across budgets, and
Rust retained request capacities must not grow after warmup or a repeated shape.
No average can hide an individual failing workload.

New plans and reports record `retained_memory_policy` version 2,
`rust-bounded-reuse-reference-observation-v2`. Rust's activation, total workspace,
kernel scratch and remaining workspace capacities are all hard acceptance gates;
growth of any one field fails even if another field shrinks. Both engines must
still provide all three retained-memory snapshots and the existing memory and
numeric-pool evidence. C# observations include per-field deltas from warmup,
observed stability/growth, and `required_for_acceptance:false`; they do not gate
the Rust rewrite or establish an exact total C# memory bound.

The pinned C# reference assigns recognition lines through a shared worker cursor
and rents sessions by input-volume high water. Each session separately retains
up to eight shape plans plus grow-only resize arrays keyed by actual content
width. Repeating the same page can therefore populate a different session's
cache or enlarge its resize arrays after two fixed warmups, without growing
native activations. The existing captures contain no per-field owner snapshot,
so particular deltas cannot be attributed conclusively to those stores. This
reference behavior is reported without modifying the pinned implementation,
adding warmups, or changing numerical or performance thresholds. Earlier plans
and failed reports retain their original joint-engine memory contract; this
policy does not retroactively change their results.

Loading/normalization, compilation/request creation, and each input's first run
are recorded separately. Rust first-run profiles serialize `reshape_ns` as the
sum of actual DET and REC single/batch `Session.reshape` call work, including
no-op/cache-hit calls. `binding_ns` excludes those calls and includes the binding
work on either side. CLS uses its fixed constructor-bound input shape, so its
initial binding stays in `compile_requests_ns`. Parallel work sums are not
subtracted from wall time. First-run profiles include instrumentation overhead
and may reuse earlier model/shape caches; they are diagnostic.

New C# captures also run a separate phase diagnostic process with the same
frozen plan, source, workload and active round identity. Its wrappers measure
actual DET/CLS/REC reshape call work, including no-op calls; session/pool
initialization is recorded outside pipeline Run. The pinned API intrinsically
returns owned results, so external post-Run owned copying is zero under an
explicit contract, rather than an invented replacement for historical nulls.
Run includes construction of the returned result containers; accessing lazy
`Text` publishes the string later and is measured separately in this diagnostic.
Official paired timing and its fixed workloads, warmups and thresholds remain
unchanged. A real-model phase-only timing probe is performance work and requires
an authorized reserved round; it is never a free preliminary run. The ten
`--acceptance --phase-contracts` checks use fake clocks/sessions and no models.
Historical phase nulls remain unmeasured. Allocation volumes are collected in a
separate untimed batch. Rust
allocator tracking is disabled during official timing. C# reports managed byte
volume rather than allocation counts. Shared coefficients/packs, retained native
activations, retained kernel scratch, remaining request workspace, and process
peak working set are named separately. Rust's kernel scratch is an exact subset
of retained workspace; fixed stack tiles are excluded. C# workspace introspection
counts observed managed array payloads plus a Count-based estimate of collection
slots. Opaque framework collection backing capacities, including per-thread
`ConcurrentBag` queues, are excluded along with object headers and shared
models/dictionaries. Its stability observation covers those values and does
not prove total C# collection capacity stability. C# kernel scratch is explicitly null:
the pinned kernels use transient `ArrayPool` rentals and stack tiles that cannot
be attributed to retained session fields. A separate C# diagnostic process
observes all active primitive, vector, text, and CTC-run pool types after each
paired capture. A rent/return payload and element-size self-test is mandatory.
The event metric reports rented capacity and peak observed unreturned capacity;
it does not claim exact live memory or prove a leak. `PooledArrays` can discard a
bucket without a return event; the self-test covers `int[8193]` renting a
16384-element bucket and being discarded. Direct unpooled allocations and fixed
stack tiles are outside the event metric. Official timing and allocation-volume
processes never enable this observer, so its allocations cannot affect their
GC. Null means unmeasured, not zero; process peak working set includes committed
pool and stack pages. C# native activation bytes include the 64-float guard per
workspace, with usable and guard bytes also reported separately. These metrics
do not assert allocator equivalence between implementations.

```powershell
python tools/cpu_acceptance.py --reference-dir ../SimdPaddleOCR --models-dir ../SimdPaddleOCR/models --correctness-bundle target/cpu-parity-full --output target/cpu-acceptance-round1 --task-id same-codex-task
```

This command runs performance tests. One invocation counts as one round for
each fixed raw sample. The persistent `target/cpu-performance-ledger.json`
reserves that round before timing, counts failed/interrupted rounds, and rejects
a fourth round for the same task/sample. Keep the same task ID and ledger for
the entire task. Do not repeat runs to stabilize results or improve confidence.
`report.json` remains `complete:false` until the full matrix finishes; failed
reports and the immutable plan must be retained.

An explicit human instruction can authorize a two-round exception for an
existing task. Record it separately before another benchmark; this metadata-only
command performs no build, load sampling, inference, or round reservation:

```powershell
python tools/cpu_acceptance.py --record-round-extension --task-id same-codex-task --ledger target/cpu-performance-ledger.json --additional-rounds 2 --round-authorization "EXACT USER INSTRUCTION AUTHORIZING TWO ADDITIONAL ROUNDS"
```

The default cap remains three; the original task-specific two-round extension
and its prior reservations are immutable. An additional distinct human
instruction may be recorded as exactly one append-only followup:

```powershell
python tools/cpu_acceptance.py --record-followup-round-extension --task-id same-codex-task --ledger target/cpu-performance-ledger.json --additional-rounds 1 --authorization-id NEW_INSTRUCTION_UUID --prior-authorization-id CURRENT_LATEST_UUID --round-authorization "EXACT NEW USER INSTRUCTION AUTHORIZING ONE ADDITIONAL ROUND"
```

Use the existing task and ledger, the actual latest authorization ID, a distinct
new instruction ID and the exact new human instruction. The recorder copies and
hashes the complete prior ledger, preserves its prefix, appends the grant and
binds the prior-latest chain. It is idempotent only for the same latest ID, text
and prior ID. Conflicting IDs/text, stale chains, old replay and concurrent lost
updates fail. Plans/reports identify the latest cap and authorization. Neither
record-only command reserves a round; failed/interrupted timing rounds still
consume the stored allowance. Neither command grants permission or changes
`AGENTS.md`. All five currently authorized original-sample rounds are consumed;
the further human grant is pending and no followup grant has been recorded.

## CI

`.github/workflows/cpu.yml` defines the exact named checks
`Core / Windows / x86-64` and `Seven-model parity / Windows / x86-64`.
The first runs core, serial, documentation, lint, pixel and harness contracts.
The second checks out the exact C# revision, validates every asset, runs quick
seven-model correctness and real batching, executes the frozen friend assembly's
ten model-free phase contracts, and retains evidence on success or failure.
Missing assets or outputs fail. Push and PR runs for the same repository/head
share a concurrency group; distinct fork repositories remain separate.

The workflow is published in [draft PR #2](https://github.com/mg-chao/simd-paddle-ocr-rs/pull/2).
Actual PR run [38829351626](https://github.com/mg-chao/simd-paddle-ocr-rs/actions/runs/38829351626)
passed both exact named checks for head `10945d03...`, including 201 quick
comparisons, nine batching comparisons, ten model-free phase contracts and 113
tooling contracts. Main protection was applied and independently fetched again:
both checks are required from GitHub Actions app 15368, with strict and
administrator enforcement. The actual local receipt is
`target/cpu-ci-actual-20261016/required-ci-policy-complete.json`
(SHA-256 `9e666f926cef2ff1c85fae817db46c1c84def70e713d0aab1507ca10f5a3e466`).
The initial policy422/GET404 responses and earlier CI argv failure remain
historical evidence. No merge or hardware/performance acceptance is implied.

The corrected active source `ba0cd3bb...` must pass a new CI run and the local
attempt3 gates. Prior `7f341...` failed exact Auto2 logical-node equality at
medium DET `Conv.121`, despite an exact final tensor; the dedicated Packed3
spatial-packet correction and three new independent tests now pass 24 focused
NCHW tests. Full core/model/variant/static/package qualification remains pending.
The source receipt binds local working bytes separately from reviewed Git
newline normalization; it does not assert equal raw checkout hashes on CI.

Manual `workflow_dispatch` with `full=true` selects the full pre-release corpus;
GitHub requires that workflow file on the default branch before manual dispatch
is available. See [GitHub's dispatch requirement](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#workflow_dispatch).
Local full-corpus validation remains available before that publication step.
The pinned nightly date is intentional; its official distribution metadata was
available in the 2026-10-16 readiness check. Compiled AVX-512 is not hardware
qualification. No CI setting authorizes another performance round.
