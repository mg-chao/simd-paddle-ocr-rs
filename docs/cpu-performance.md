# CPU performance comparisons on Windows and macOS

Run this developer harness on a physical Windows x64 or 64-bit macOS host, using the pinned nightly and the exact C# reference and model/fixture hashes in `tests/baseline/manifest.json`. It starts Rust and C# in separate processes, records their execution order, and compares identical predecoded BGR24 bytes and owned OCR outputs. GPU backends, diagnostic node tracing, custom Rust flags and profiling builds are excluded. The C# adapter and every referenced project build into the invocation's isolated directory.

Start with a focused comparison:

```powershell
cargo run --release --locked --example cpu_benchmark -- --model tiny --threads 8 --fixture imgs_1 --workload latency --runs 3 --budget-seconds 60
```

For the prepared M4 workspace, source `../env.sh` from the Rust repository to
select the local native .NET SDK. A focused macOS comparison of all three
representative inputs is:

```bash
cargo run --release --locked --example cpu_benchmark -- \
  --model tiny --threads 8 --fixture imgs_1,imgs_00018069,imgs_00006737 \
  --workload latency --runs 3 --budget-seconds 60 --case-timeout-seconds 30
```

macOS jobs record `affinity: "unbound"` and reject `--affinity`; both processes
use normal macOS scheduling with matched OCR worker budgets and
`DOTNET_PROCESSOR_COUNT`. Neither CPU placement nor performance-core selection
is guaranteed. Peak resident bytes are reported separately from Windows
private commit, and the two memory metrics cannot be compared in one case.
No root privileges, scheduler changes, Rosetta, or GPU are required.

Defaults select tiny/small/medium, `imgs_1` (sparse), `imgs_00018069` (dense short lines), `imgs_00006737` (mixed widths), serial and eight-thread execution, latency and throughput. The default total budget is **60 seconds per implementation**, excluding build time. Worker startup, loading, first inference, warm-up and measured work all count. A timeout terminates the worker process tree and preserves partial reports. No automatic retry, repetition growth, or sample pooling occurs. The full matrix has 36 comparisons and needs an explicitly larger bounded budget, for example:

```powershell
cargo run --release --locked --example cpu_benchmark -- --runs 3 --budget-seconds 1800 --case-timeout-seconds 120
```

Expand only to answer a concrete outstanding qualification question. Rebuild after code changes and treat each invocation as a new version. Prior results do not validate changed code. Build time has its own 180-second timeout. A full matrix that exceeds the supplied budget is incomplete, even if its completed cases pass.

## Matched execution and measurements

- `--order balanced` is the default. Order uses the parity of `model_index + thread_index + fixture_index`, with zero-based canonical models tiny/small/medium, threads 1/8, and fixtures sparse/dense/mixed as listed above. Even parity runs Rust first and odd parity runs C# first. Both workloads in a group receive the same order; filtering a workload or changing the selected fixture order cannot change an individual case's order. The full matrix has nine Rust-first and nine C#-first cases **per workload**, three of each order per model and per fixture, and a minimal five/four split per thread count. A focused selection may be unbalanced, and a single case has only one process order. `--order rust-first` and `--order csharp-first` select an explicit order for a separately justified investigation; the harness does not duplicate cases or automatically repeat a comparison to counterbalance it. Provenance records the strategy/rule, and each case records planned order and successfully spawned workers, including workers that subsequently fail. Budget-skipped workers are absent from actual order.
- `--threads 1`: one line/preprocessing/detector worker. `--threads 8`: eight preprocessing workers and eight detector threads. Rust automatic recognition uses eight line workers on AVX2/FMA or native NEON when there are at least eight independent recognition units; otherwise its base maximum is four. The automatic classifier stage may use eight workers. The pinned C# adapter uses four shared CLS/REC line workers. C# receives matching `DOTNET_PROCESSOR_COUNT`; Rust uses one shared context of the same nominal capacity. Rust's fixed pool enforces an aggregate worker cap across concurrent requests. C# retains per-request nested budgets without a global admission cap, so two requests may use more OCR workers in total. Windows also pins both processes to the same one or eight allowed logical CPUs before model setup. macOS is unbound: these settings do not establish equal enforced aggregate CPU use, and runtime/JIT/GC threads and OS scheduling remain outside the Rust OCR cap.
- Rust completes classification and rotation before the recognition wave. Stable line ownership, configured thread maxima and the fixed shared CPU pool apply to this staged pipeline. Earlier unconditional and capacity-aware CLS/REC overlap candidates were rejected; their focused results do not qualify the retained source.
- On Windows, the automatic mask selects the first one or eight CPUs in the parent's current processor group. `--affinity HEX` pins an explicit mask whose bit count must equal the selected thread count; use a single `--threads` setting with this option. Processor topology, SMT and heterogeneous cores affect results; the exact mask or explicit unbound scheduling policy is stored with every job.
- `--workload latency` produces one owned OCR result per batch. `--workload throughput` starts two concurrent requests sharing model weights, and reports their combined batch wall time. Throughput in images/second is `2000 / median_ms`. Thread startup and synchronization count equally in both implementations.
- Every process performs one separately reported first batch, **one warm-up**, then exactly `--runs` batches (1–3; default 3). Model loading/compilation is timed separately; shape-dependent first-use packing remains in first inference. Image decoding, file input, output projection, diagnostics and JSON serialization are outside measured inference.
- Every first, warm-up and measured batch also records process CPU milliseconds alongside wall milliseconds. On Windows, Rust uses `GetProcessTimes` and C# uses `Process.TotalProcessorTime`, the same kernel-plus-user process counter. On macOS, both workers use `getrusage(RUSAGE_SELF)` user plus system time. The CPU values sum all process threads, include runtime/JIT/GC activity during the batch, and may exceed wall time with parallel workers. Counter granularity can produce zero for a short batch. The paired arrays preserve every sample; low CPU relative to wall can motivate a scheduling/load investigation, but does not itself identify the cause. CPU time is diagnostic and never replaces the wall-time performance gate. Older reports without CPU/order fields deserialize with unavailable diagnostics; they do not acquire inferred values.
- One first batch and one warm-up define a bounded warm measurement; they do not demonstrate steady state. Preserve sample drift and uncertainty from JIT, shape packing, caches, CPU frequency or host load. An uncertainty investigation needs a stated question and bounded scope; the harness does not extend warm-up or grow samples automatically.
- Windows `GetProcessMemoryInfo.PeakPagefileUsage` supplies peak process private commit for both processes, including runtime initialization, loading, first inference and retained requests. Capture occurs before diagnostic reflection or report generation. This measures process commitment rather than working set.
- macOS `getrusage(RUSAGE_SELF).ru_maxrss` supplies the lifetime resident high-water mark in **bytes** for both workers, captured at the same point before diagnostics. It includes shared resident pages and runtime/JIT allocations, and is not private commit. See [Apple's Darwin getrusage source](https://github.com/apple/darwin-xnu/blob/main/bsd/man/man2/getrusage.2). Schema 2 emits `peak_resident_bytes` on macOS and `peak_private_bytes` on Windows; missing, zero, ambiguous, or platform-mismatched memory measurements are rejected. Legacy Windows records still deserialize with their original private-commit semantics.
- Post-measurement diagnostics report Rust normalized/shared packed weights, retained activation/workspace/kernel scratch bytes and requests. C# source weight bytes and reflection-derived cached packs, native activation capacities, primitive request arrays, resize scratch and retained inference sessions are explicitly lower bounds; they exclude headers, TLS and shared `ArrayPool` capacity. Their accounting categories are not directly interchangeable and do not determine the memory gate.

## Gates and artifacts

Each model/fixture/thread/workload has its own gate: median single-image latency ≤110% of C#, throughput ≥90%, and platform-specific peak process memory ≤110% (private commit on Windows, resident bytes on macOS). Final samples also preserve the baseline correctness contract (text, ordering, dimensions, classifications and existing score/coordinate tolerances). Concurrent owned results must agree within each implementation.

A passing qualification requires three samples. If the observed min/max timing ratios cross the relevant boundary, the result is **inconclusive** even when its median passes. A clear timing failure, correctness difference or memory excess is **failed**. Missing measurements and timeouts remain inconclusive; neither is reported as success. Noisy evidence stays unresolved until a separately justified, bounded investigation addresses it.

Artifacts are written under `target/cpu-benchmark/<timestamp>-<pid>/`: `report.md`, `report.json`, immutable job JSON, predecoded input bytes, per-worker JSON/logs and isolated C# build output. Reports include raw wall/CPU samples, first/warm-up/load measurements, execution strategy and planned/actual process order, ratios, differences, budget consumption, model/fixture manifest hash, exact affinity or unbound scheduling, memory metric, toolchains, Git/worktree provenance and a content hash of Rust production/benchmark sources and build configuration. macOS also records its OS version, CPU model, available parallelism and power settings. A source change during execution invalidates the run. JSON coverage identifies a focused comparison or all 36 distinct required cases for one platform; only three-sample passes in all 36 set `full_qualification_passed` to true. Mixed scheduling platforms cannot qualify one matrix. Exit status is zero only when all selected cases pass; the selected subset must be identified when reporting results. A focused subset passing does not complete a full platform performance qualification.

Run `cargo run --release --locked --example cpu_benchmark -- --help` for all filters and path overrides. Production consumers do not need .NET or these development-only helpers.

## Diagnose a demonstrated gap

For separate production-stage instrumentation, use the
[Rust production profiler](../examples/cpu_profile.rs):

```powershell
cargo run --release --locked --features profiling --example cpu_profile -- tiny imgs_1 8
```

This helper uses the pinned local models/fixtures, one warm-up and one profiled
run. An optional fourth positional argument sets `CpuOptions.line_workers`:
`0` (default) keeps automatic classifiers independent of recognition ownership,
including the AVX2/FMA and native NEON eight-unit policy; a positive limit bounds both stages.
For example, compare
`tiny imgs_00018069 8 4` with `tiny imgs_00018069 8 0` to diagnose the classifier
budget; with the eight-unit policy this also changes recognition ownership,
so the comparison no longer isolates classifier scheduling. It reports fused production stage
counters and a retained activation breakdown. Worker-work counters can exceed wall time and are not an additive
decomposition of total latency. Instrumented timings are diagnostic evidence;
the uninstrumented benchmark determines the performance gate. Rebuild after
code changes and do not pool measurements from different versions.

`classification_wall_ns` and `recognition_wall_ns` report the separate line
stages. Individual classification, preprocessing, backbone and projection
counters remain summed worker work and may exceed wall time. Historical V2/V3
overlap diagnostics have a `combined_line_wall_ns` field for their combined
wave; that field belongs to those rejected candidate versions and is absent
from the retained staged profiler. Do not add worker counters to either wall
measurement or pool the different profiler versions.

The [follow-up record](windows-cpu-validation.md#follow-up-optimization-and-scheduling-policy)
keeps the clean baseline, kernel/harness revision, rejected unconditional
and capacity-aware overlap experiments, and retained V4 staged revision separate. Three-case
Rust variant diagnostics and focused C# comparisons do not complete the
historical 36-case Windows qualification matrix. A successful correctness run
on an earlier candidate likewise does not validate subsequent source edits.

Final V4 validation passed all 213 real-model correctness comparisons and one
Tiny/dense, eight-CPU, single-request Rust/C# performance cell with explicit
C#-first order. Its paired-kernel timing ranges overlap. These completed focused
checks do not establish a full 36-case pass or an end-to-end speedup caused by
the edits; the archived samples and source versions define their scope.

The C# companion profiler described below remains Windows-only; the macOS port
covers the matched benchmark workers, not that diagnostic profiler.

The [C# production profiler](../tools/csharp-profile/README.md) uses the same
serial benchmark job JSON, model/pixel hashes, options and CPU affinity. Its
default stage mode performs one warm-up and one instrumented sample. Separate
`--nodes` and `--kernels` modes expose DET production node counters and fixed
matched kernel probes, without disabling fusion or invoking a reference graph.
Use node names and actual tensor shapes to align implementations; layout
operations can change normalized indices. C# node counters have 512 slots, and
the tool reports that bound explicitly.

Keep builds and all other inference idle during diagnosis. A bounded two-model
investigation allows 25 seconds for stage runs, 10 seconds for the complete kernel
mode, and 25 seconds for node runs: 60 seconds total, excluding builds. Supply
external process deadlines, keep fixed samples, preserve raw JSON, and stop when
the demonstrated gap is explained. Diagnostic node/stage timers add overhead;
Rust `NodeTrace` also follows a separate diagnostic execution path. Neither
replaces the uninstrumented comparison gate.

## Comparing production graphs across source versions

`cpu_adoption_probe` measures public production `Session::run` calls without
profiling or diagnostic graph execution:

```powershell
cargo run --release --locked --example cpu_adoption_probe -- --case tiny-det --threads 1 --models-dir ../SimdPaddleOCR/models --label candidate --output target/adoption-candidate.json --dump-dir target/adoption-candidate-raw
```

Use `--case small-rec`, `--case medium-rec`, or omit the filter for all three
fixed graph shapes. `--model FILE --shape 1,3,H,W` selects a custom graph.
Each case gets one warm-up and three measured calls. Model/input/output hashes,
raw samples, affinity and execution statistics are recorded; checks and optional
F32 little-endian output dumps occur outside the inference timer. Use fresh
dump paths, since existing dump files are never replaced.

Build baseline and candidate in separate output directories and retain their
source manifests and executable hashes. Compare identical model/input hashes,
shapes, options and affinity before interpreting times. Rotate version order
across cases, enforce an external deadline, and keep aggregate execution below
the per-version budget. The helper's default 60-second budget also includes
setup and output checking. Overlapping fixed sample ranges remain inconclusive.

These raw-tensor graph measurements are separate from OCR latency. Recognition
graphs run the full dense head, while production OCR normally uses compact CTC
projection. The [2026-10-17 adoption record](windows-cpu-validation.md#nhwc-concatenation-and-binary-alias-adoption-2026-10-17)
preserves both graph and real OCR comparisons, including their differing results.

The [ARM64 follow-up record](macos-arm64-followup-optimization.md) covers selective
9x9 depthwise row strips, SIMD detector threshold output, and shared graph-use
indexes. It separates fixed-count stage diagnostics from a twelve-cell comparison
against the frozen Rust baseline and preserves overlapping timing ranges.

The [Windows stage analysis](windows-cpu-stage-analysis.md) records baseline
production stage timings on sparse, dense and mixed-width inputs, serial fused
recognition-kernel diagnostics, and a focused Tiny/dense scheduling comparison.
The accompanying `cpu_scheduling_probe` example compares automatic and eight-line
scheduling with one warm-up and three owned-result calls per configuration,
requiring exact output equality. Its single process order and workload do not
qualify a global scheduling-policy change; the report preserves its memory cost
and keeps these samples separate from the Rust/C# benchmark.

The [Windows AVX2/FMA adoption record](windows-avx2-recognition-scheduling.md)
validates the automatic eight-unit policy with a frozen baseline and a rebuilt
candidate across seven cases. On the adopted implementation, the scheduling
probe's automatic and explicit-eight configurations select the same recognition
policy for eligible pages; use separate revisions to reproduce the old/new
comparison.

The [detector upsampling record](windows-cpu-stage-analysis.md#optimization-3-detector-upsampling-on-windows-avx2fma)
documents direct packed transpose stores on Windows x64 AVX2/FMA. It preserves
per-step profiles, frozen-baseline graph and OCR comparisons, exact output
checks, and the inconclusive Small full-page result alongside the Tiny/Medium
improvements.
