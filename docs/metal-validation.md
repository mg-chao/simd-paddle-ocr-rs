# Native Metal 4 validation on M4

Validation ran locally on a physical Apple M4 (`Mac16,10`, 16 GiB), macOS
27.0.1 (26A434), using `nightly-2026-09-08` / rustc 1.100.0-nightly
(`cea272fa3`). No CI, remote runner, VM, or software GPU supplied these results.
The runtime targets Apple Silicon with macOS 26+; this machine does not establish
compatibility or performance on other Apple GPU generations or Vulkan hardware.

## Frozen identities

The original release baseline was captured before replacement, at revision
`0e019536801f3da7a22884e5dbe07fa227fd4050`. Its initial hardware-context test
established the starting environment only. Both performance reports preserve
their executable, source archive, source hashes, input hashes, model hashes,
dictionary hashes, configuration, and raw results under the local `target/` tree.

| Artifact | SHA-256 |
| --- | --- |
| Baseline executable | `f1e75a89191aed0da32468030b2766de58466a169170c9773f53e2bcaca0de19` |
| Baseline source archive | `5487f532c9ba077322d5db4aefa11f9426a7dfe865e803f90b65d2d0053bdc92` |
| Native executable | `b49c78bc1e7061c211d8399814aa333c7579250ae7196825e0b7b89a06258918` |
| Native source archive | `7e5299b9ed7d1d22597bde68f26d6dc11f74b42909bc14dc74444c8247df8c90` |

The reports are `target/metal4-baseline/performance/manifest.json` and
`target/metal4-native/performance/manifest.json`. Historical artifacts are local
validation evidence and are excluded from the package.

## Correctness

The final frozen native executable passed **36 GPU OCR cases**: tiny, small, and
medium presets on typical, dense, and vertical images, once for each combination:

| Tensor storage | Matrix operands | Cases passed |
| --- | --- | --- |
| F32 | F32 | 9/9 |
| MixedF16 | F32 | 9/9 |
| F32 | F16 | 9/9 |
| MixedF16 | F16 | 9/9 |

These use all seven local models: `cls`, `det`, `rec`, `small_det`, `small_rec`,
`medium_det`, and `medium_rec`. Nine CPU references were captured once and reused
only after checking identical input bytes. Final reports are
`target/metal4-native/verified-{f32,storage-f16,matrix-f16,f16}/manifest.json`.
Each configuration had a 180-second correctness timeout and no benchmark loops.
The existing precision-specific validators were unchanged: text, ordering,
orientation, score, geometry, and span contracts all passed. Every result also
passed the GPU-residency checks, including zero diagnostic readback and no CPU
compute stages.

Focused release checks passed:

- 149 runtime/kernel checks, including 89 explicitly selected physical-GPU
  tests. These exercise upload/clear/compute/readback ordering, parameter
  snapshots, allocation aliases, residency, foreign contexts, concurrent sessions,
  quotas, completion, timestamp counters, and bounded host-side failure injection.
- Seven OCR integration tests and one hardware statistics test.
- CPU numerical oracles for convolution, matrix tails, depthwise operations,
  reductions, normalization, attention, image preparation, crops, projection,
  geometry, orientation, and CTC, including padding and offset guards.
- IEEE binary16 conversion cases, including signed NaNs, subnormals, ties,
  overflow, and tensor stores.

Qualification exposed and corrected three numerical-policy issues. Geometry now
uses ties-to-even for polygon vertices and source-coordinate mapping; a focused
half-integer regression failed before the correction and passed afterward.
Strict resident OCR now applies the same F32 policy to classification as to
detection and recognition, fixing a confidence failure also reproduced with the
baseline. Half-storage kernels explicitly retain their native transcendental
function policy: changing those functions crossed a half-rounding boundary in
the small vertical detector and changed reading order. F32 kernels use precise
functions; both policies retain safe arithmetic and F32 accumulation.

The CPU/GPU, parallel/serial, and profiling feature matrix passed. Final GPU
library configurations passed 449, 444, 437, and 432 non-hardware tests,
respectively, with 100 hardware/diagnostic tests left explicitly opt-in. CPU-only
configurations passed 387, 382, 375, and 370 tests. Example builds, seven doctests
in each all-features/serial configuration, documentation, formatting, and Clippy
with warnings denied passed. The all-target dependency graph and package source
audit contain none of the retired translation or execution stack.

## Bounded performance comparison

Both versions consumed the same decoded 500×500 typical image and CPU reference,
with the tiny preset, F32 storage and matrix operands, one request, one slot,
recognition batch 2, wave size 1, and classification batch 4. Model, input, RGB,
and reference hashes match. There was one cold warm-up and two uninstrumented
measurements per version. No adaptive sampling or favorable-result reruns were
used.

| Measurement | Baseline | Native Metal 4 |
| --- | --- | --- |
| OCR latency, two samples | 80.728, 80.055 ms | 58.847, 58.585 ms |
| Mean OCR latency | 80.391 ms | 58.716 ms |
| Process CPU time, two samples | 6.950, 6.600 ms | 4.360, 4.167 ms |
| Submissions per request | 15 | 15 |

Observed latency decreased **27.0%**, and process CPU time decreased **37.1%**.
The latency difference exceeds the variation in these samples, but two samples
on one machine do not establish a broad performance distribution.

One separate instrumented call per version examined native execution overhead.
The baseline profiling scope was expanded with one cold preparation call and one
instrumented call, bounded by a 15-second timeout, to answer that attribution
question; no additional latency samples were collected. Encoding measured
1.809 ms versus 2.024 ms, and queue submission measured 0.155 ms versus 0.032 ms
(79% lower in this single observation). These instrumented observations are not
pooled with the latency samples. Waiting and readback move between host timing
scopes in the two runtimes and should not be compared independently.

The native profile resolved all dispatch timestamps: 57.346 ms summed device
duration, including 37.648 ms recognition, 9.583 ms detection, and 5.890 ms
classification. The baseline has no comparable dispatch device timestamps.
Host durations and device durations remain separate. Profiling-disabled calls
allocate no counter heap.

Aggregate performance harness execution was 1.41 seconds for the baseline
(including the extra profile scope) and 1.96 seconds for the native version,
excluding builds and correctness runs. Both remained below 60 seconds.

## Reproduction

Use the local models in `../SimdPaddleOCR/models` and install Pillow for the
fixture-generation harness. Run focused checks first, following
[local GPU validation](gpu-validation.md). Build the release examples:

```sh
cargo build --release --locked --features gpu,profiling --examples
python3 tools/full_gpu_validation.py --report target/metal-check/f32 \
  --presets tiny small medium --cases typical dense vertical --budget-seconds 180
```

Repeat qualification in fresh report directories with `--precision mixed-f16`,
`--matrix f16`, and both options. Use `--cpu-reference-dir target/metal-check/f32`
to reuse the verified CPU references. Each selected case executes once.

```sh
python3 tools/gpu_performance.py --corpus target/metal-check/f32/corpus \
  --report target/metal-check/performance --presets tiny --cases typical \
  --configs 1:1:2 --measurements 2 --profile --budget-seconds 60
```

Rebuild after source changes; never combine samples from different executables
or configurations. The harness refuses a stale executable and preserves the
tested source and binary. Native ownership, pointer, residency, and visibility
invariants are documented in [Metal architecture](metal-architecture.md).
