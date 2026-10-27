# Windows Vulkan migration: local results

These results describe the final native `ash` implementation on one physical
Windows machine, measured on 2026-10-25. They do not establish performance on
other devices or a general end-to-end OCR speedup.

## Hardware and reproducibility

- AMD Radeon RX 6750 XT; AMD driver 26.10.1; Vulkan 1.4.315.
- 64-lane default subgroups, 32–64 supported subgroup range, 32 KiB shared memory,
  64 timestamp bits, 10 ns timestamp period; FP16 and KHR cooperative matrices.
- Release builds with `gpu,profiling`, the pinned repository toolchain, default
  GPU context/scheduling, deterministic input seed 1. Validation layers disabled
  only for performance measurements.
- Original baseline source revision:
  `d164e663c6f4435b1fc5e89a3a43f9752528336c`. The exact revision is recorded
  in the local baseline manifest described below.

The original source archive, executable, SHA-256 identity, configuration and raw
samples are retained outside production code in `reports/ash-migration/baseline`.
Final measurements and per-source/executable SHA-256 identities are in
`reports/ash-migration/performance-v5/manifest.json`. Those local directories are
ignored by Git and excluded from the published crate. Earlier `performance-v3`
measurements are a separate experiment and are not pooled with final samples.

## Bounded model comparisons

Two representative neural inputs were used: classifier `[1, 3, 80, 160]` and
recognition `[1, 3, 48, 320]`, using the same local `cls.onnx` and `rec.onnx` as the
baseline. Each model/precision combination used one CPU-oracle correctness
invocation as its only GPU warm-up, then exactly three measured calls. F32 and
mixed-F16 storage were separate campaigns; both used F32 accumulation and the
same scheduling configuration. Timings include the profiled public session call
and readback, excluding model loading and compilation.

All values below are milliseconds; the reduction column compares medians.

| Case | Original baseline samples | Native ash samples | Median, original → native | Reduction |
| --- | --- | --- | --- | --- |
| Classifier F32 | 4.5457, 3.1115, 3.1265 | 1.1527, 1.2442, 1.1900 | 3.1265 → 1.1900 | 61.9% |
| Classifier mixed F16 | 4.5951, 3.2353, 3.1499 | 1.1427, 1.0502, 1.0360 | 3.2353 → 1.0502 | 67.5% |
| Recognition F32 | 8.5039, 5.3477, 2.0164 | 2.0005, 2.2008, 1.9666 | 5.3477 → 2.0005 | 62.6% |
| Recognition mixed F16 | 9.4280, 6.9685, 2.4313 | 2.3303, 2.3712, 2.5171 | 6.9685 → 2.3712 | 66.0% |

Classifier samples are separated between versions. Recognition baseline samples
vary by more than 4× and overlap the new ranges, so their median reductions are
observations, not reliable estimates of a sustained speedup. No unchanged run was
repeated to obtain cleaner samples. Mixed F16 remains slower than F32 on the
recognition case in the final run; smaller storage does not guarantee lower
latency on this workload.

All four final CPU-oracle comparisons passed the existing mode-specific
tolerances. Maximum absolute errors were 4.17e-7 (classifier F32), 0.004807
(classifier mixed), 4.97e-5 (recognition F32), and 0.048943 (recognition mixed).

## Projection kernel experiments

One shape `[rows=40, inner=120, classes=6906]`, one warm-up and three dispatches per
variant, with device timestamps and numerical checks of winners, maxima,
probability sums and invalid flags. Inputs are exactly representable in half, so
this experiment isolates implementation cost; independent tail/half-oracle tests
and complete OCR comparisons cover rounding behavior.

| Kernel | Device samples (µs) | Median (µs) |
| --- | --- | --- |
| Heads4 | 93.32, 93.56, 93.68 | 93.56 |
| Heads4Subgroup | 63.76, 63.16, 63.04 | 63.16 |
| ProjectionMatrixF16 | 47.84, 47.60, 48.12 | 47.84 |

The subgroup F32 median is 32.5% lower than the portable F32 kernel. Cooperative
F16 is 24.3% lower than subgroup F32 and 48.9% lower than portable F32. These paths
are selected by capabilities, with cooperative F16 additionally requiring the
explicit matrix-F16 policy. Unsupported devices retain the portable kernels.

The earlier cooperative implementation had a 65.04 µs median on this shape.
Skipping wholly padded row tiles and their reductions reduced that to 47.84 µs
(26.4%). The input shape and bounded measurement protocol were unchanged; the
versions remain separate in their manifests.

The final model and kernel campaigns together consumed 1.600 seconds of executable
wall time, including setup, correctness warm-ups, harness loops and all measured
calls. The original baseline consumed 3.781 seconds; the earlier native kernel
experiment consumed 1.187 seconds. Each version stayed below its 60-second budget.
No adaptive repetitions or full-pipeline performance campaign was used.

## Final correctness and build checks

- Native release unit suite: 415 passed in 1.55 s. Its 71 normally ignored tests
  contain the separately selected hardware checks and bounded performance probes.
- Physical GPU correctness with Khronos synchronization validation: 62 passed
  (five native lifetime/binding/transfer tests, one diagnostic statistics test,
  and 56 numerical, resident, scheduling and resource tests), approximately
  8.52 s of test execution. No validation error was reported.
- The two half conversion/store tests each exercise 1,153 IEEE edge cases.
  Cooperative-matrix correctness includes three tail shapes. Other tests cover
  convolution tails, matrix shapes, long reductions, attention, geometry, CTC,
  projection ties, concurrent sessions, quotas, aliases and reshape rollback.
- Complete OCR: three representative images (typical, perspective, sparse), each
  compared once on CPU and once on GPU in three modes: F32, mixed-F16 storage
  with F32 matrices, and mixed-F16 storage with cooperative F16 matrices.
  All nine comparisons passed text, order, orientation, spans and the existing
  mode-specific geometry contracts under synchronization validation. Outputs
  contained 16, 16 and 2 lines respectively. Configuration: tiny model preset,
  one resident wave setting, classifier batch 4. Local reports:
  `ocr-f32-v5`, `ocr-mixed-v5`, `ocr-cooperative-v5` under the migration directory.
- One additional F32 profile check preserved the typical-image output and
  returned device durations for all 1,084 dispatches. Host encoding and queue
  submission totals matched the public stage profile separately. This was a
  correctness check, not a performance comparison.
- CPU release tests: 347 passed without default features (1.88 s), 359 passed
  with default features (1.50 s). CPU-only example build passed.
- Release Clippy with warnings denied passed for GPU/profiling and CPU-only
  all-target configurations. GPU examples, formatting and package construction
  passed. Embedded interfaces for all 855 declared precision variants are
  checked against the shader catalog during the Cargo build.
- The GPU library also passed a release check without profiling or default
  features. The final crate contains 311 files, including native shaders,
  build metadata, preserved macOS shaders and this report.
- The Windows normal/build dependency tree contains only its native GPU stack.
  CPU-only dependencies also exclude `ash`,
  `gpu-allocator`, `shaderc` and `shaderc-sys`.
- The preserved macOS backend passed an `aarch64-apple-darwin` GPU test-target
  cross-check from Windows. That Windows qualification did not validate macOS shaders.

Khronos layers were obtained locally from the project's official Windows nightly
archive, SHA-256
`4f74f9ffbb76b3ccbb36de9353260392dd9f402390082d295c7206b511101da0`.
The manifest and library are retained under `target/vulkan-validation/layers`.
The physical tests enabled `SIMD_OCR_VULKAN_VALIDATION=1` and synchronization
validation; missing layers are an initialization error.

## Limits and remaining costs

NVIDIA, Intel and macOS hardware remain unverified. Capability checks and portable
fallbacks are implemented, but one Radeon cannot validate other drivers.
Non-coherent flush/invalidate alignment is checked in code and tests; this
machine's selected coherent host allocations do not exercise non-coherent memory
hardware. Device-loss propagation is tested with an injected Vulkan error and
allocation failure with quota rejection; actual driver loss and physical memory
exhaustion were not induced. Timeout lifetime behavior uses an unsignalled
timeline value followed by safe retirement of the actual queued work.

Host encoding, queue submission, transfer setup and numerous small dispatches
remain visible costs; the isolated kernel improvements should not be extrapolated
to complete OCR latency. Mixed-F16 storage can introduce conversion overhead.
The first GPU build also compiles bundled shaderc and all shader variants; deployed
applications need only the Vulkan loader and driver. CPU-only builds avoid that
toolchain.

Package construction reports an existing yanked `tokio 1.54.0` lockfile entry from
an inactive WebAssembly dependency branch. It is absent from the active Windows
GPU and CPU dependency trees; no macOS dependency update was introduced to silence
that unrelated warning.
