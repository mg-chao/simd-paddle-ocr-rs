GPU optimization results, 2026-10-24

The implemented changes reduce full-GPU OCR latency by 23–26% on the measured
Apple M4 workloads, corresponding to 1.29–1.36× throughput for one sequential
request. The SIMD CPU backend remains faster on these inputs. CPU remains the
default, and no model precision or recognition-quality requirement was relaxed.

The [baseline analysis](gpu-backend-analysis-2026-10-24.md) describes each stage
and the original measurements at commit `84f84b5`. This report describes the
subsequent working-tree implementation.

## Changes and rationale

1. **Batch orientation classification independently of recognition width.**
   Previously every crop ran a separate `[1, 3, 80, 160]` classifier graph. Full
   OCR now prepares up to four crops in recognition order and classifies them
   together. Each rotation and compact result reads its own winner row. The
   existing recognition units and two-wave submission bound are preserved, with
   at most three unused prepared crops retained for a later unit. The classifier
   shape stays fixed during queued work so parameter-buffer writes cannot change
   an earlier encoded graph. Tail rows are ignored. Fixed-batch-one models keep
   batch one; transactional `ResourceLimit` failures reduce optional capacity
   before recording work. Other errors propagate. Standalone classification
   remains single-image execution.
2. **Use tiled convolution for eligible dense 2×2 kernels.** The existing tiled
   shader already handles these kernel coordinates and padding, but the selector
   admitted only 1×1 and 3×3 kernels. It now includes square 2×2 kernels under the
   same channel, stride, dilation, device-limit and workgroup-count constraints.
   This avoids the generic implementation for expensive early layers in the
   small recognizer and medium detector.
3. **Use direct NCHW pointwise addressing for large Metal recognition layers.**
   FP32 1×1 convolutions with feature-map height at most six can use the existing
   direct kernel beyond 256 input channels. This removes repeated spatial
   division, remainder and bounds calculations. The extension is restricted to
   Metal because earlier AMD/Vulkan measurements found a regression for large
   reductions. No device-name heuristic, startup tuning or layout conversion is
   added. The portable large-channel policy on Vulkan/DX12 stays unchanged.

Source locations: `src/ocr/gpu.rs`, `src/runtime/gpu/heads.rs`, `vision.rs`,
`rotate.wgsl`, `operators.rs`, and `physical.rs`. Classifier batching needs no new
public setting and does not change the recognition batch-size setting.

The bounded pointwise experiment compared three shapes with one correctness
warm-up and three measured calls per candidate. For `[1,1536,3,240] → 768`,
median wall time fell from 5.173 ms to 3.715 ms with direct NCHW addressing.
Increasing the reduction tile from 16 to 32 did not improve it; NHWC plus
conversion took 15.964 ms. Neither rejected alternative was retained. The
experiment's wall timings include submission/readback and are distinct from the
unprofiled OCR latency measurements below.

## Local performance

Physical Mac mini, Apple M4, 10 GPU cores, 16 GiB unified memory, macOS 27.0.1,
Metal, pinned nightly toolchain, wgpu 30.0.1, release build with all features.
Identical local models, dictionaries, decoded pixels and settings were used:
full GPU, FP32, one request, one GPU slot, recognition batch size two, CTC
alignment enabled. Model loading, image decoding, JSON serialization and
profiling are outside the measured warm OCR interval.

Each row has one cold correctness/warm-up call and exactly three unprofiled
measured calls, followed by one separate profile call. The typical image is
500×500 with 16 lines; dense is its 1000×1000 repetition with 64 lines. These
are related fixtures, not a diverse document corpus.

| Workload | Baseline median ms | Optimized median ms | Latency reduction | Speedup |
|---|---:|---:|---:|---:|
| Tiny, typical | 94.30 | 71.65 | 24.0% | 1.32× |
| Small, typical | 293.56 | 224.00 | 23.7% | 1.31× |
| Medium, typical | 1206.18 | 887.36 | 26.4% | 1.36× |
| Tiny, dense | 304.58 | 235.24 | 22.8% | 1.29× |

Optimized raw warm samples in milliseconds:

- Tiny typical: 72.017, 71.645, 71.498.
- Small typical: 224.853, 224.002, 223.505.
- Medium typical: 886.997, 887.363, 887.514.
- Tiny dense: 235.343, 235.222, 235.236.

The final performance harnesses took 10.37 seconds total, including process
startup and cold/profile calls, below the 60-second per-version budget. The
earlier classifier-only implementation and rejected kernel experiments have
separate source snapshots and results; their samples are not pooled here.
The classifier-only version reduced tiny typical latency to 76.47 ms before
the convolution changes.

Separate device profiles explain the gains:

| Work on typical page | Baseline device ms | Optimized device ms |
|---|---:|---:|
| Tiny classifier | 23.90 | 6.06 |
| Small recognition 2×2 nodes 4 and 8, summed | 29.23 | 10.93 |
| Medium detector 2×2 nodes 2 and 4, summed | 32.46 | 10.07 |
| Medium recognition pointwise nodes 300, 309, 317, 326, summed | 171.08 | 114.30 |
| Medium recognizer, all available timestamps | 962.38 | 688.09 |

These are summed device durations from separate instrumented calls, not wall
latencies or exact additive contributions to the end-to-end improvement. Some
Metal timestamps are unavailable, so totals over available records can
undercount. Wall latency is the performance decision criterion.

Application-accounted peak device allocation rose from 40.75 to 41.87 MiB for
tiny, 90.17 to 91.35 MiB for small, and 246.17 to 247.29 MiB for medium on the
typical page. Dense tiny rose from 117.82 to 118.95 MiB. The additional roughly
1.1–1.2 MiB buys bounded classifier batching. These figures exclude opaque
driver allocations.

## Correctness and build validation

All validation ran on the physical machine; no CI or remote runner was used.

- 392 non-ignored release unit tests passed with all features. The 45
  hardware/benchmark tests remain opt-in, rather than being silently counted as
  executed by the unit suite.
- Sixteen distinct focused GPU tests passed, including classifier row/tail
  selection, pixel-exact rotation, reuse for standalone classification, fixed
  batch one, activation-budget rollback, asymmetric 2×2 padding, channel tails,
  multiple batches, fused/unfused comparisons, mixed storage, resident lifetime
  and transfer contracts, and query-page rollover. The new 2×2 test covers
  three shapes in FP32 and mixed-F16; it needs FP16-capable hardware.
- Eighteen full-OCR comparisons passed: three presets × typical/vertical/dense
  pages × FP32/mixed context, one invocation per GPU case with its CPU oracle.
  Checks cover text, counts, classification and rotation, recognition shape,
  CTC token/span boundaries, score tolerances, coordinates and residency.
  Full mixed mode retains FP32 detection/recognition as before. No CPU compute
  stage or diagnostic tensor readback was accepted in the full path.
- Eighteen resident shader variants loaded successfully; eight two-FP32
  floor/round boundary cases passed. Cooperative-matrix WGSL passed Naga
  validation, but this M4 does not expose that execution capability.
- Strict Clippy passed for all targets with all features and with no default
  features. Formatting and `git diff --check` passed.

Validation also repaired three existing test/tool defects: stale request-field
references in the ARM parallel test, a Rust attribute accidentally embedded in
the shader-check WGSL string, and an unused NEON test import. The pointwise
experiment now records missing timestamps as null and reports wall samples
without changing its numerical correctness gate.

The final mixed-F16 test addition and unused-import cleanup changed only test
sources after the OCR measurements. Release targets were rebuilt and focused
tests/lint rerun. SHA-256 verification confirms the rebuilt benchmark and OCR
executables are byte-for-byte identical to the measured/qualified executables;
no additional performance runs were made for these test-only edits.

## Evidence and limits

Ignored local evidence is under `reports/gpu-optimization-2026-10-24/`:
`summary.json`, `final-typical/manifest.json`, `final-dense/manifest.json`,
`qualification-f32/manifest.json`, `qualification-mixed-f16/manifest.json`,
`verified-tests.json`, `final-tests.json`, shader logs and candidate logs.
Performance manifests preserve exact source archives, executable, model,
dictionary and input hashes, commands, settings and raw samples. Baseline
artifacts remain under `reports/gpu-analysis-2026-10-24/`.

Only this M4/Metal configuration is performance-qualified. Three warm samples
establish a local improvement, not a confidence interval or a universal speedup.
Vulkan/DX12 correctness and the portable 2×2 change still require validation on
their physical hardware. The local query-page test passed in the final suite,
but intermittent zero timestamp pairs were observed in earlier isolated runs;
an extra submission boundary did not fix them and was not retained. Final
profiles contain 10/4/4 missing records for tiny/small/medium typical and 21 for
dense tiny. No speed claim depends on treating those records as measured zero.

Recognition convolution remains the largest cost, particularly on medium;
generic 5×5/7×7 dense layers and shape-cache churn remain candidates for future
focused work. Increasing concurrency, changing precision or adopting NHWC is
not justified as a new default by these measurements.
