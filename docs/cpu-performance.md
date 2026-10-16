# Windows x64 CPU performance qualification

Run this developer harness on the physical Windows x64 host, using the pinned nightly and the exact C# reference and model/fixture hashes in `tests/baseline/manifest.json`. It starts Rust and C# in separate processes, alternates their order, and compares identical predecoded BGR24 bytes and owned OCR outputs. GPU backends, diagnostic node tracing, custom Rust flags and profiling builds are excluded. The C# adapter and every referenced project build into the invocation's isolated directory.

Start with a focused comparison:

```powershell
cargo run --release --locked --example cpu_benchmark -- --model tiny --threads 8 --fixture imgs_1 --workload latency --runs 3 --budget-seconds 60
```

Defaults select tiny/small/medium, `imgs_1` (sparse), `imgs_00018069` (dense short lines), `imgs_00006737` (mixed widths), serial and eight-thread execution, latency and throughput. The default total budget is **60 seconds per implementation**, excluding build time. Worker startup, loading, first inference, warm-up and measured work all count. A timeout terminates the worker process tree and preserves partial reports. No automatic retry, repetition growth, or sample pooling occurs. The full matrix has 36 comparisons and needs an explicitly larger bounded budget, for example:

```powershell
cargo run --release --locked --example cpu_benchmark -- --runs 3 --budget-seconds 1800 --case-timeout-seconds 120
```

Expand only to answer a concrete outstanding qualification question. Rebuild after code changes and treat each invocation as a new version. Prior results do not validate changed code. Build time has its own 180-second timeout. A full matrix that exceeds the supplied budget is incomplete, even if its completed cases pass.

## Matched execution and measurements

- `--threads 1`: one allowed logical CPU; one line/preprocessing/detector thread. `--threads 8`: eight allowed logical CPUs; four line workers, eight preprocessing workers and eight detector threads. Both workers set process affinity before model setup. C# receives matching `DOTNET_PROCESSOR_COUNT`; Rust uses one shared context of the same capacity.
- The automatic mask selects the first one or eight CPUs in the parent's current processor group. `--affinity HEX` pins an explicit mask whose bit count must equal the selected thread count; use a single `--threads` setting with this option. Processor topology, SMT and heterogeneous cores affect results; the exact mask is stored with every job.
- `--workload latency` produces one owned OCR result per batch. `--workload throughput` starts two concurrent requests sharing model weights, and reports their combined batch wall time. Throughput in images/second is `2000 / median_ms`. Thread startup and synchronization count equally in both implementations.
- Every process performs one separately reported first batch, **one warm-up**, then exactly `--runs` batches (1–3; default 3). Model loading/compilation is timed separately; shape-dependent first-use packing remains in first inference. Image decoding, file input, output projection, diagnostics and JSON serialization are outside measured inference.
- Windows `GetProcessMemoryInfo.PeakPagefileUsage` supplies peak process private commit for both processes, including runtime initialization, loading, first inference and retained requests. Capture occurs before diagnostic reflection or report generation. This measures process commitment rather than working set.
- Post-measurement diagnostics report Rust normalized/shared packed weights, retained activation/workspace/kernel scratch bytes and requests. C# source weight bytes and reflection-derived cached packs, native activation capacities, primitive request arrays, resize scratch and retained inference sessions are explicitly lower bounds; they exclude headers, TLS and shared `ArrayPool` capacity. Their accounting categories are not directly interchangeable and do not determine the memory gate.

## Gates and artifacts

Each model/fixture/thread/workload has its own gate: median single-image latency ≤110% of C#, throughput ≥90%, and peak process private commit ≤110%. Final samples also preserve the baseline correctness contract (text, ordering, dimensions, classifications and existing score/coordinate tolerances). Concurrent owned results must agree within each implementation.

A passing qualification requires three samples. If the observed min/max timing ratios cross the relevant boundary, the result is **inconclusive** even when its median passes. A clear timing failure, correctness difference or memory excess is **failed**. Missing measurements and timeouts remain inconclusive; neither is reported as success. Noisy evidence stays unresolved until a separately justified, bounded investigation addresses it.

Artifacts are written under `target/cpu-benchmark/<timestamp>-<pid>/`: `report.md`, `report.json`, immutable job JSON, predecoded input bytes, per-worker JSON/logs and isolated C# build output. Reports include raw samples, first/warm-up/load measurements, ratios, differences, budget consumption, model/fixture manifest hash, exact affinity, toolchains, Git/worktree provenance and a content hash of Rust production/benchmark sources and build configuration. A source change during execution invalidates the run. JSON coverage identifies a focused comparison or all 36 distinct required cases; only three-sample passes in all 36 set `full_qualification_passed` to true. Exit status is zero only when all selected cases pass; the selected subset must be identified when reporting results. A focused subset passing does not complete the full Windows x64 milestone.

Run `cargo run --release --locked --example cpu_benchmark -- --help` for all filters and path overrides. Production consumers do not need .NET or these development-only helpers.

## Diagnose a demonstrated gap

For separate production-stage instrumentation, use the
[Rust production profiler](../examples/cpu_profile.rs):

```powershell
cargo run --release --locked --features profiling --example cpu_profile -- tiny imgs_1 8
```

This helper uses the pinned local models/fixtures, one warm-up and one profiled
run. It reports fused production stage counters and a retained activation
breakdown. Worker-work counters can exceed wall time and are not an additive
decomposition of total latency. Instrumented timings are diagnostic evidence;
the uninstrumented benchmark determines the performance gate. Rebuild after
code changes and do not pool measurements from different versions.

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
