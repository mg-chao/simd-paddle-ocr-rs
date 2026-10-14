# Historical import and normalization measurements

These results belong to the earlier model-import milestone and its three-warmup,
30-measurement protocol. They are separate from the scalar detection and oriented
crop measurements documented in `step12-inference-performance.md`.

Measured locally on 2026-10-14 with Windows x86_64, Rust 1.97.1, and .NET 10.0.11.
Each import case used a fresh process, three warmups and thirty measured calls
with warm filesystem caches. The all-model case retains seven models until the
timer stops. No inspection, JSON export, archive writing or SHA-256 work occurs
inside the loading timer.

| Model | Rust median / p95 ms | C# median / p95 ms | Rust retained / peak heap MB |
|---|---:|---:|---:|
| CLS | 1.83 / 2.11 | 8.06 / 9.51 | 1.08 / 2.12 |
| Tiny DET | 2.78 / 3.40 | 12.45 / 16.39 | 1.98 / 2.40 |
| Tiny REC | 5.34 / 6.40 | 18.95 / 22.51 | 4.61 / 7.52 |
| Small DET | 10.92 / 14.33 | 37.48 / 44.75 | 10.08 / 11.66 |
| Small REC | 21.42 / 24.50 | 69.22 / 74.48 | 21.46 / 37.87 |
| Medium DET | 60.42 / 74.56 | 201.64 / 234.48 | 61.72 / 67.83 |
| Medium REC | 73.79 / 82.03 | 216.31 / 236.51 | 76.86 / 93.20 |
| All seven | 178.95 / 221.30 | 544.31 / 635.99 | 177.79 / 194.13 |

The source files total 176.89 MB and retained normalized weight payloads total
175.89 MB. The all-model Rust heap peak is about 16.34 MB above its retained
heap, consistent with tensor conversion and temporary graph rewriting rather
than a second retained model blob. MB here means 1,000,000 bytes.

C# computes its original full-source FNV1a checksum during loading; ordinary
Rust loading records source length without computing a checksum. This
bookkeeping difference explains part of the timing difference. These numbers
describe the measured import paths, not inference speed or a universal Rust
speedup.

Rust heap data comes from a separate feature-enabled benchmark executable that
forwards the standard System allocator and counts requested live/peak bytes.
The main library forbids unsafe code. Allocator forwarding is the only unsafe
code, confined to this optional instrumentation binary. Heap counters exclude
allocator bookkeeping and process memory mappings. Uninstrumented builds
provide latency; the profiling build supplies memory data.

Process memory has different accounting: Rust's parent samples the child's OS
high-water resident memory and can miss the final observation; C# reads its own
process high-water resident memory, including warmup and GC effects. Managed
allocation volume is not a C# heap peak. See the JSON record for these separate
metrics.

The full OCR comparator measured **432.46 ms before** and **228.71 ms after**
streaming, using the same parent process timer and exact tiny baseline/replay.
That wall time includes process startup. Both use three warmups and thirty
fresh-process measurements. The new comparator validates each binary
descriptor in one traversal without retaining decoded tensor arrays.

Machine-readable measurements (local `docs/step12-performance.json`) include all model
summaries, phase timings, process-memory observations and comparator samples.
Raw local import samples remain in `target/step12-measurements`.

## Reproduce

Use fresh output directories:

```powershell
./tools/model-import-benchmark.ps1 -Out target/measurements/csharp
./tools/benchmark-rust-imports.ps1 -Out target/measurements/rust
```

The Rust launcher builds separate normal and `heap-profiler` executables and
starts one process for each individual model and the sequential seven-model
case. SHA-256 inspection and archive validation remain separate operations.
