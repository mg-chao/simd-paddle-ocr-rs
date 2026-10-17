# Focused Apple M4 CPU performance comparison

This is the historical Portable4 baseline before the ARM64 kernel optimization.
The implementation now selects NEON/FMA automatically; later measurements are
recorded in [ARM64 optimization results](macos-arm64-optimization.md) to preserve
source-version provenance.

The benchmark runner supports native 64-bit macOS in both the Rust and C#
workers. The first focused comparison on the physical Apple M4 found Rust's
warmed median latency **23–37% higher than C#**, while Rust's process peak
resident memory was **39–54% of C#**. All three OCR comparisons agreed within
the existing correctness contract. All three performance cells failed the
110% latency gate; the resident-memory gate passed in each cell.

## Configuration and bounded execution

Run on 2026-10-17 (Asia/Shanghai), macOS 27.0.1, native ARM64, AC power, Low
Power Mode off. Rust used `nightly-2026-09-08`, standard release/default
features, automatic `Portable4` SIMD/layout, fusion and owned results. C# used
SDK 10.0.303 / runtime 10.0.11 and the clean pinned reference
`33ff8a81672452041d110e8ba52dcd2639e54228`, with CPU backends explicitly selected.
Models, dictionaries and decoded BGR24 input hashes were checked against the
unchanged manifest.

One configuration: tiny model, an eight-worker OCR budget, four line workers,
eight preprocessing/detector workers, recognition batching set to one, one
request per latency batch. C# received `DOTNET_PROCESSOR_COUNT=8`; Rust used a
shared eight-worker context. Both processes used unbound macOS scheduling,
without a CPU mask or performance-core placement guarantee.

Each of the three representative fixtures ran once per implementation in a
fresh worker: one separately reported first inference, one warm-up, then
exactly three measured batches. This is nine measured batches, three warm-ups,
and three first batches per language. No automatic retries, additional warm-up,
or repeated sampling occurred. Worker startup/loading/inference used **1.664 s
for Rust and 1.958 s for C#**, each within its 60-second execution budget.
Isolated adapter build time was excluded. Other builds/inference were idle
during the measured worker runs.

```bash
source ../env.sh # prepared workspace; otherwise use installed native toolchains
cargo run --release --locked --example cpu_benchmark -- \
  --model tiny --threads 8 --fixture imgs_1,imgs_00018069,imgs_00006737 \
  --workload latency --runs 3 --budget-seconds 60 --case-timeout-seconds 30
```

The runner exited with status 1 because the selected latency gates failed;
all workers completed successfully and their correctness comparisons passed.

## Warm latency and memory

Wall times are medians of the three fixed samples. Memory is lifetime process
peak resident bytes, expressed here in MiB (1,048,576 bytes), captured before
diagnostic inspection. It includes runtime/JIT and shared resident pages.
It is not Windows private commit.

| Fixture | Content | Rust median (ms) | C# median (ms) | Rust/C# latency | Rust peak RSS (MiB) | C# peak RSS (MiB) | Process order |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `imgs_1` | sparse | 65.576 | 53.285 | 1.231× | 68.969 | 127.578 | C# → Rust |
| `imgs_00018069` | dense short lines | 129.075 | 93.964 | 1.374× | 38.641 | 100.000 | Rust → C# |
| `imgs_00006737` | mixed widths | 124.190 | 93.824 | 1.324× | 70.469 | 133.609 | C# → Rust |

| Fixture | Rust wall samples (ms) | C# wall samples (ms) |
| --- | --- | --- |
| sparse | 64.124, 65.576, 66.254 | 55.958, 53.285, 52.283 |
| dense | 129.075, 146.686, 127.236 | 99.272, 93.764, 93.964 |
| mixed | 125.926, 124.190, 122.893 | 93.824, 94.169, 91.452 |

Even the fastest Rust sample divided by the slowest C# sample exceeds 1.1 in
each cell, so these are clear latency-gate failures under the runner's observed
sample-range rule. Dense Rust samples are visibly variable; C# sparse samples
drift downward. One warm-up does not establish steady state, and these samples
do not identify the cause of the gap.

## Measurements and evidence

Both workers use `getrusage(RUSAGE_SELF)` user plus system CPU time and resident
high-water marks. Darwin reports `ru_maxrss` in bytes; see
[Apple's source](https://github.com/apple/darwin-xnu/blob/main/bsd/man/man2/getrusage.2).
Schema 2 preserves Windows `peak_private_bytes` and emits
`peak_resident_bytes` separately for macOS. The evaluator rejects missing,
zero, ambiguous or mismatched memory metrics. macOS rejects `--affinity`
instead of recording an ineffective CPU mask. Historical Windows reports keep
their original semantics.

Port validation passed all **13 benchmark contract tests**, release builds of
all examples, all-target compilation with all features, and the C# native
counter probe. The tests cover legacy records, Darwin counters, unavailable
affinity, matching memory metrics, report labels and mixed-platform matrix
rejection. The native C# probe confirmed the 144-byte Darwin ABI, positive
resident bytes and monotonic CPU counters. Existing architecture-specific
unused/dead-code warnings remain. Windows execution was not rerun on this Mac.

The [archived machine-readable report](reports/macos-arm64-2026-10-17-focused-performance.json)
preserves exact wall/CPU samples, cold/warm-up/load timings, resident peaks,
process order, output comparisons, budgets, power settings, and executable and
source hashes. Full local artifacts, including immutable jobs, worker logs,
outputs and isolated C# build files, are in
`target/cpu-benchmark/1792208746962-37408/`.

This covers three latency cells out of the 36-case platform matrix. It does not
qualify throughput, serial performance, small/medium models, other CPUs, Intel
macOS, or a general Rust/C# performance parity claim. The
[performance protocol](cpu-performance.md) defines the complete measurement
contract and separate gates.
