# Native Metal validation history and reproduction

Target: physical Mac mini, Apple M4 (10 CPU / 10 GPU cores), 16 GiB, macOS 27.0.1.
Build tools: pinned Rust nightly-2026-09-08, Xcode 27.1 (27A9275), Metal Toolchain
27A266a, Apple Metal compiler 32023.921. C# production reference is unchanged at
`33ff8a81672452041d110e8ba52dcd2639e54228`, using .NET SDK 10.0.303.

Final qualification is complete and optimization stopped at the user's five-round
limit. **The performance objective remains unmet: v49 passes 33 of 48 gates.**
All cold-start and CER gates pass; fifteen warm/first-shape gates fail. The final
rebuilt source passes 69 physical GPU checks, all 54 OCR fixture comparisons and
383 default CPU tests (three explicitly ignored). The complete campaign contains
162 launches and 468 outputs. See the [final report](metal-final-report.md) for
measurements, the five-round decisions, source/binary integrity and limitations.

The retained runtime is v44 (matrix Add fusion with preserved half rounding and
exact convolution epilogue fast paths), including the v42 incremental-build fix.
Versions 45–47 were removed. The attempted v48 campaign used a stale executable
and is explicitly invalid; v49 was freshly rebuilt and audited. No earlier
measurement is pooled into the final gate count.

The following sections preserve historical development evidence. Their versioned
measurements describe those builds, not the final build.

## Earlier validation evidence

- 61 physical GPU correctness checks passed after native submission, calibrated
  counters, matrix projection, and specialized depthwise implementation. Checks
  cover synthetic OCR, explicit/automatic targets, concurrency, quotas,
  ownership, parameter offsets above 64 KiB, tensor operations, FP16 IEEE edges,
  geometry, crops, classifier batches, winner ties, softmax and Unicode CTC.
- The archive hit check uses `FailOnBinaryArchiveMiss` on the physical M4.
- 30 affected kernel/graph checks passed after direct device-fragment matrix loads.
- All 27 full-precision OCR fixtures passed (nine images, all three presets),
  using 27 corresponding CPU outputs and the unchanged strict contract.
- The detector precision correction also passed all 27 mixed-precision fixtures
  in v9. Both campaigns preserve the existing assertions.
- Version 14 passed all 61 physical GPU checks with relaxed neural arithmetic
  that preserves infinities and NaNs. Image and geometry shaders retain safe
  arithmetic. Its small-sample profiled output passed the existing CPU contract.
- Default CPU build: 383 passed, 3 explicitly ignored.
- Benchmark readiness and acceptance policy: 8 tests passed.
- Tiny sample text matches C#; the current mixed-precision sample passes the
  unchanged Rust geometry, metadata and score contract against Rust CPU.

Rust's established tall-crop transform rotates counterclockwise before
classification. The pinned C# implementation rotates clockwise. Their classifier
labels and applied-rotation metadata can therefore differ on the same tall box
while producing identical text. This change preserves Rust's existing contract;
it does not alter validators or rewrite prediction metadata.

Local raw artifacts are under `target/metal-rebuild/`. The frozen C# baseline has
162 launches and 468 outputs, completed in 259.32 seconds. Its XFUND spatial CER
is 2.3501% (tiny), 2.3275% (small), and 2.4760% (medium), identical across cold,
first-shape and warm phases. These are custom spatial OCR metrics, not the
official XFUND semantic entity/relationship scores.

The complete Rust v29 report also contains 162 launches and 468 outputs. Every
CER gate passes: tiny 2.3468%, small 2.3468%, medium 2.4534%, identical across
cold, first-shape and warm phases. The allowed increase is 0.05 percentage
points. All 27 fixtures pass in full precision and all 27 pass in mixed
precision (`fixtures-v29-f32` and `fixtures-v29-f16`).

| Preset | C# sample cold median ms | Rust v29 | C# sample warm median ms | Rust v29 | C# corpus warm mean ms | Rust v29 |
|---|---:|---:|---:|---:|---:|---:|
| Tiny | 337.40 | 104.42 | 30.85 | 24.62 | 60.94 | 74.93 |
| Small | 464.94 | 159.81 | 53.35 | 66.22 | 132.11 | 193.06 |
| Medium | 834.01 | 370.32 | 175.05 | 204.62 | 634.84 | 534.19 |

Sample cold p95 is 567.46 / 162.35 / 370.84 ms. Corpus cold means are
168.77 / 300.36 / 738.44 ms. Every cold gate passes, and medium passes every
corpus latency gate. Remaining failures are tiny/small corpus first-shape means
and medians, tiny/small corpus warm means/medians/p95, and small/medium sample
warm medians/p95. The complete comparison is
`target/metal-rebuild/comparison-v29.json` (34 passed, 14 failed).

The report at `target/metal-rebuild/rust-v29-all` combines the original three
sample launches per preset with the missing cold and corpus phases. No sample
measurement was repeated. Combination required identical executable and source
archive hashes, every source hash, model/input hashes, hardware and protocol
settings. Its `composition` field retains the original manifest hashes and
paths; all 162 raw stdout/stderr logs are copied into the combined directory.
The source reports are `rust-v29-sample`, `rust-v29-cold`, and `rust-v29-corpus`.
Separate frozen paths and all their launch costs are retained. Earlier v9 data
remains in `rust-v9-all` and `comparison-v9.json`; results are never pooled across
code versions.

Version 29 passes 63 physical GPU correctness checks. New regressions cover
2,049 profiled dispatches across multiple Metal counter buffers and matrix
column tails in reused allocations, including NaN-poisoned unused capacity in
both storage precisions. Native half conversion handles finite normal values;
explicit IEEE edge handling and both 1,153-case oracle checks remain. Convolution
staging processes 32 reduction elements per barrier and shares address
calculations across four adjacent output columns. Direct matrix column tails
are admitted only when binding-time range checks prove the complete read stays
inside an initialized allocation. Other tails retain guarded staging.

A bounded detector diagnostic used two uniform gray RGB inputs (512 and 960
square), one first-shape and one warm call per input, with no detected boxes.
It includes normalization and empty DB processing, not just neural kernel time:

| Preset | C# warm 512 ms | Rust warm 512 ms | C# warm 960 ms | Rust warm 960 ms |
|---|---:|---:|---:|---:|
| Tiny | 6.73 | 6.68 | 18.03 | 20.55 |
| Small | 8.80 | 13.90 | 25.23 | 46.94 |
| Medium | 26.26 | 47.04 | 85.05 | 159.40 |

These single observations locate a substantial small/medium detector cost under
Rust's FP32 policy; they are not acceptance measurements. Tiny's corpus gap is
mostly elsewhere. Raw inputs, commands and all 24 outputs are in
`target/metal-rebuild/detector-empty-v29`. Corpus detected-line counts and summed
recognition widths differ by less than 0.1%, so materially different amounts of
recognized text do not explain the latency gap.

Experiments adding convolution packing, fragment-level conditional matrix loads,
smaller output staging, native MPS pointwise multiplication, dedicated pointwise
scratch, or serial recognition waves did not improve end-to-end latency and
were removed. Versions 16 and 17 passed all 27 mixed fixtures; their source and
outputs remain frozen. Version 17 also passed a dedicated MPS odd-dimension,
batch and bias check.

Versions 20–23 explored channel-contiguous regions. All 27 OCR fixtures and a
focused residual/broadcast/tail test passed. Profiling revealed conversions for
unused bias bindings; fixing those cut sample conversions from 660 to 218.
Warm medians remained 36.48 / 98.95 / 305.91 ms, so the implementation was
removed. The experiment ledger is `target/metal-rebuild/optimization-ledger.json`.

Version 31's rectangular 16×64 convolution tiles passed 65 GPU checks and all
27 mixed OCR fixtures, but sample medians (24.94 / 65.80 / 205.17 ms) did not
improve on the retained implementation. The experiment was removed. Version
32 isolated detector-storage sensitivity: all 18 small/medium fixtures passed
with half storage. Version 33 replaced that temporary override with the explicit
Metal context option and verified all precision-policy combinations. Its 65
physical GPU checks and exact-policy fixture reports are `hardware-v33.log`,
`fixtures-v33-tiny-f32-detector`, and `fixtures-v33-f16-detector`.

Version 30 adds the missing 7×7 depthwise specialization on macOS. The four
such layers in the small detector previously used generic convolution. The
new path passes a focused FP32/mixed-storage regression covering borders,
asymmetric padding, both strides and batches, plus all nine small mixed OCR
fixtures. Three fresh sample processes produced a warm median of 64.93 ms
(v29: 66.22 ms; C#: 53.35 ms). The first launch was 2,148.09 ms and remains in
the report; this version does not establish a cold-tail pass. Raw evidence is
`rust-v30-sample-small` and `fixtures-v30-f16-small`. This focused result is not
pooled into the v29 complete campaign.

A separate three-operator diagnostic is retained in `kernel-probe-v29b`.
Each implementation used one correctness/warmup call and three unprofiled
measurements for dense 3×3, pointwise and depthwise convolution. All numerical
checks passed. C# feature-map output is physically NHWC; its oracle conversion
is outside timing. Rust returns canonical NCHW. Measurements include public
tensor input/output transfers, including Rust's host FP16 conversion, and
therefore do not isolate resident OCR kernel cost. The original adapter's
layout-mismatch failure is retained in `kernel-probe-v29`; no performance
measurement was made after that failed correctness check.

Version 34 tested the reference's direct-dot approach for short pointwise
reductions in canonical NCHW storage. It passed the exact-policy OCR fixtures,
but sample warm medians (25.54 / 64.72 / 203.56 ms) were slower than v33, so the
new kernel and its selector were removed. Version 35 retains an independent
correctness correction found during that work: both matrix fast paths now
require unchanged spatial extents for 1×1 convolution. Bottom/right padding
otherwise makes the output plane an invalid input stride. The regression
compares optimized/generic FP32 and mixed GPU output with the scalar oracle on
batched, asymmetric and reduction-tail cases. This fix does not relax gates.
No fresh performance run is made solely to replace the removed experiment with
more favorable timings.

Version 36 tested a bounded native buffer pool (64 MiB, at most 512 idle
handles), returning buffers only after their last completion/upload owner and
zeroing every logical checkout. It passed 68 GPU checks, all 27 fixtures and
the complete 50-page first-shape/warm accuracy campaign. Sample warm medians
were 24.84 / 63.88 / 202.30 ms; corpus warm means were 73.96 / 187.03 /
527.95 ms. Against v33, corpus changes ranged from a 1.3% tiny improvement to
a 0.06% small regression. This does not establish native allocation as the main
bottleneck, so the pool was removed. Its first sample cold launch was
1,053.21 ms, exceeding the C# tiny p95; that observation remains in the report.
Evidence: `rust-v36-sample`, `rust-v36-corpus`, `hardware-v36.log`.

Version 37 computed exact resize coefficients once per column and row on the
GPU. All 66 GPU checks and 27 fixtures passed, including unchanged byte-for-byte
resize comparisons. Sample warm medians were 24.04 / 65.52 / 210.81 ms: tiny
improved, while small and medium regressed, so this experiment was also removed.
One instrumented small-sample call passed the CPU contract. Combined resize
dispatch intervals were 1.30 ms, compared with 1.74 ms in the historical v29
profile; different detector precision and sampling effects mean this is not an
isolated causal comparison. Recognition convolution remained dominant. No
unchanged sample rerun or corpus expansion followed. Evidence:
`rust-v37-sample` and `profile-v37-small-sample.json`.

Version 38 used 64×64 dense-convolution tiles with at least 32 output channels,
following the reference's larger tiles for narrow channel counts. It passed 67
GPU checks and 27 fixtures, but sample warm medians of 24.82 / 65.39 / 208.99 ms
did not improve the retained implementation; it was removed. A separate C#
diagnostic enabled its existing `SIMD_OCR_GPU_TIME` logging for one small sample
first/warm pair. Both outputs exactly matched the pinned baseline. Warm command
recording/submission was 0.08 ms for detection, 0.04 ms for classification, and
0.47 ms for recognition (13 units, three waves, 1,222 records). Those diagnostic
timings are not acceptance data. Evidence: `rust-v38-sample`,
`csharp-stage-timing-v38`, and `hardware-v38.log`.

Version 39's recipe inspection found channel-bias additions throughout the hot
convolution paths. Physical lowering guarantees that each fused operation keeps
the original convolution shape, so its already-computed output channel is the
same index that the generic epilogue reconstructed using division and remainder.
Tiled, depthwise, generic and transposed Metal convolutions now pass that index
directly. The generic path remains for other kernels. The new regression covers
batched pointwise, depthwise and grouped convolutions with ordered channel-bias,
activation and full-tensor residual additions, including spatial tile tails.
The 50-page corpus uses the unchanged accuracy protocol; first-shape and warm
outputs have identical CER. See `hardware-v39.log`,
`fixtures-v39-tiny-f32-detector`, and `fixtures-v39-f16-detector`.
Full-FP32 fixture reports are `fixtures-v39-f32` (tiny) and
`fixtures-v39-f32-small-medium`. The final source and executable audit is
`retained-v39-audit.json`; all 243 runtime/build source hashes match the v39
fixture manifest. The pinned production C# checkout remains clean.

The remaining work is to reduce resident neural execution cost, particularly
recognition convolution and the small detector, while preserving the existing
geometry/text contract. Allocation reuse, detector storage alone, and resize
coefficient sharing have not closed the gap. Any new approach must pass the
focused numerical checks and full fixture contract before a new frozen latency
campaign. The requested all-preset performance guarantee is still unsupported.

The first launch of changed executables is substantially slower than subsequent
fresh processes. A focused macOS `syspolicyd` log query confirmed an XProtect
scan of the frozen v12 benchmark executable before launch. The unaccounted
pre-setup delay varies between versions and remains included in cold latency;
no security setting is changed, no first result is discarded, and binaries are
not pre-launched to hide it. Three observations provide limited tail precision.

Profiling v14's small sample still points to recognition matrix work: 38.85 ms
in 64x64 convolution dispatch intervals and 13.64 ms in 32x32 intervals, versus
11.76 ms in depthwise convolution. These are instrumented intervals, not an
additive end-to-end latency estimate: counter sampling changes scheduling.

Version 8's expanded mixed-precision qualification found a border-fixture
failure: quantizing the detector changed its probability map, moving a thin
box from 11 to 14 pixels wide and changing the recognized text. Replaying the
same map through CPU, precise GPU, and native-trig GPU geometry produced the
same boxes, locating the change before geometry. The Metal resident detector
retains FP32 weights/tensors by default; mixed classification/recognition remain
explicit. The later `metal_detector_precision` opt-in is qualified separately
per model and does not change this default. The fixture gate has not been relaxed. The correction passed all 27 mixed
fixtures and the complete 50-page corpus accuracy campaign in v9.

The classifier-backbone tensor oracle also passed against C# CPU and Metal
(maximum absolute errors 1.09e-7 and 8.08e-4 respectively). The reference's
standalone generic Metal graph cannot execute its final 1280-to-2 classifier
MatMul, so this comparison uses the original ONNX boundary
`p2o.pd_op.flatten.0.0`. End-to-end C# OCR uses its specialized classifier path.

Version 40 adapted the reference's four-channel shared depthwise tile to NCHW.
It passed 69 GPU checks, all 54 fixture comparisons and the 50-page accuracy
campaign. Sample medians improved to 23.09 / 59.33 / 192.66 ms, but corpus warm
means regressed to 74.60 / 178.94 / 517.17 ms. Version 41 reduced the shared
allocation to one channel and kept four adjacent outputs per thread. It passed
69 GPU checks and all 27 exact-policy fixtures, but sample warm medians were
24.70 / 62.36 / 201.43 ms. Both experiments were removed; their raw reports and
source snapshots remain under `rust-v40-*`, `rust-v41-sample`, `source-v40`, and
`source-v41`. No unchanged reruns or further v41 corpus expansion were made.

Version 42 corrects a build dependency found during that rollback. Removing a
catalog entry left every surviving shader unchanged, so the old incremental
build skipped portable-library relinking and embedded the removed kernels.
The ordered object inventory now invalidates both linking and native archive
generation; a missing portable library does the same. An isolated test uses the
real Metal tools with two shaders and verifies removal, re-addition using cached
objects, a no-op build with identical output hashes, and missing-library recovery.
Production portable IR shrank from 11,339,673 to 9,686,583 bytes, and its compressed
payload is identical to the one embedded in frozen v39. This establishes correct
artifact contents, not a new latency result. Validation and source/executable
hashes are in `retained-v42-audit.json`; raw build evidence is in
`incremental-build-v42`. Version 42 fixture reports are `fixtures-v42-f32`,
`fixtures-v42-tiny-f32-detector`, and `fixtures-v42-f16-detector`.

## Reproduction

Run on physical Apple Silicon, with the local model directory and prepared
50-image XFUND Chinese validation split. No CI, remote runner, model download or
virtual machine is used. Source changes require rebuilding before measurements.

```sh
source ../env.sh
SIMD_OCR_TEST_METAL_BUILD=1 python3 -m unittest discover -s tools -p test_metal_build.py
cargo test --lib
cargo test --release --features gpu,profiling --lib gpu_ -- --ignored --nocapture --test-threads=1
cargo build --release --locked --features gpu,profiling --examples
dotnet build tools/reference/Reference.csproj -c Release --nologo
python3 -m unittest discover -s tools -p 'test_metal_*.py'

# Validate full precision and the benchmark's explicit detector policies.
target/metal-tools/bin/python tools/full_gpu_validation.py --all-cases --presets tiny small medium --precision f32 --matrix f32 --wave 16 --classification-batch 16 --report target/metal-fixtures-f32 --budget-seconds 300
target/metal-tools/bin/python tools/full_gpu_validation.py --all-cases --presets tiny --precision mixed-f16 --matrix f16 --wave 16 --classification-batch 16 --report target/metal-fixtures-tiny --budget-seconds 180
target/metal-tools/bin/python tools/full_gpu_validation.py --all-cases --presets small medium --precision mixed-f16 --matrix f16 --metal-detector-storage f16 --wave 16 --classification-batch 16 --report target/metal-fixtures-small-medium --budget-seconds 300

# Preparation needs Pillow; run/analysis use only the Python standard library.
target/metal-tools/bin/python tools/metal_benchmark.py prepare --output target/metal-inputs
python3 tools/metal_benchmark.py run --implementation csharp --inputs target/metal-inputs/inputs.json --report target/csharp-metal --scope all --budget-seconds 900
python3 tools/metal_benchmark.py run --implementation rust --inputs target/metal-inputs/inputs.json --report target/rust-metal --scope all --budget-seconds 900
python3 tools/metal_accuracy.py --report target/csharp-metal
python3 tools/metal_accuracy.py --report target/rust-metal
python3 tools/metal_comparison.py --csharp target/csharp-metal --rust target/rust-metal --output target/metal-comparison.json
```

Each report must use a new directory. The harness freezes executables and source
contents, hashes models and inputs, records the actual backend, and rejects
stale builds, missing readiness markers, fallback, nondeterministic repeated
outputs and timeouts. It does not retry or increase sample counts automatically.

Cold is parent process launch through the first flushed `RESULT_READY` marker,
after an owned OCR result exists. It includes raw RGB input read, runtime/model
setup and first inference. Warm intervals include inference and the owned result
copy, excluding input IO, correctness checks and JSON serialization. A corpus
process makes two fixed-order passes over all 50 images; there is no extra
warm-up. OS filesystem and driver caches are not purged. Profiling is disabled.

Acceptance requires at least 5% lower sample cold/warm medians and corpus
cold/first-shape/warm means and medians, with no p95 regression, for every preset.
XFUND CER may exceed C# by at most 0.05 percentage points; existing Rust fixture
gates remain unchanged. Partial or noisy runs do not establish completion.
