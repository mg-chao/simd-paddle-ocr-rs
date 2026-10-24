# GPU matrix coverage extension — 2026-10-24

Extending the existing FP32 cooperative pointwise kernel to eligible canonical
32×32 pointwise recipes reduces Medium Full GPU OCR warm latency by **7.4%**
on the physical Apple M4. Tiny and Small remain within observed timing noise.
FP32 precision, activation layouts, shared shader storage, model files, batching,
and accounted device memory are unchanged.

## Implementation

`physical::select_cooperative_pointwise` now considers `PointwiseNchw32` in
addition to `PointwisePacked32` and `PointwiseBlocked32`. The previous selector
excluded compatible layers simply because the earlier lowering pass had chosen
the canonical tiled kernel. This affected Medium recognition's 256→512 H12
pointwise layers, among others.

The existing eligibility guards remain: capable Metal adapter, canonical FP32
bound tensors, at least 128 input channels, 64 output channels and 64 spatial
elements, and all three matrix dimensions divisible by eight. Partial 32×32
tiles remain masked. Prescaled kernels, spatial convolutions, small direct16
recipes and generic MatMul retain their previous selection. Existing fused
bias, residual, activation and output-offset behavior is retained.

The selected change reuses the existing 4 KiB direct matrix shader without
changing its arithmetic or workgroup geometry. The default
`cooperative_pointwise: true` enables it; the existing opt-out and portable
fallbacks remain available. No startup benchmark or new option is introduced.

## Measurements

Physical Mac mini, Apple M4, ten GPU cores, 16 GiB unified memory, Metal,
wgpu 30.0.1, pinned nightly-2026-09-08, locked release/all-features build.
The sample is one 500×500 image with sixteen lines, one request, one GPU slot,
recognition batch two, FP32 Full placement, Logit scores and CTC spans.

The baseline is the frozen, source-verified matrix implementation at `da9e3a8`.
The new run uses identical RGB pixels, models, dictionaries, settings and CPU
reference output. Each preset/version has one cold warm-up, three ordinary
measurements and one separate diagnostic profile. The unchanged baseline was
not rerun. Measurement sessions are separate; three samples are not a confidence
interval. Wall time excludes decoding, construction, owned-result conversion,
validation, serialization and profiling.

| Preset | Before median, ms | After median, ms | After samples, ms |
|---|---:|---:|---|
| Tiny | 61.875 | 62.220 | 64.276, 62.220, 61.565 |
| Small | 158.170 | 158.935 | 158.935, 156.704, 161.442 |
| Medium | 451.450 | 417.931 | 418.334, 417.931, 416.722 |

Medium's three new samples are below all three baseline samples. Tiny and
Small medians are approximately 0.6% and 0.5% higher, respectively, with
overlapping sample ranges; no latency improvement is established for them.

The separate Medium profiles show recognizer device work changing from
288.587 to 262.770 ms and pointwise work from 178.128 to 149.289 ms. Original
recognition nodes 74 and 103 change from 13.157/13.375 ms to 4.263/4.389 ms.
Recognizer matrix dispatches increase from 286 to 364; eleven detector
dispatches also move to the matrix kernel. Detector device duration changes
from 121.878 to 114.509 ms.

Profiling changes pass scheduling. These durations are not additive components
of ordinary wall latency, and no hardware counters establish an occupancy or
bandwidth bottleneck. New Tiny/Small/Medium profiles have 4/5/7 unavailable
timestamps; missing durations are omitted, never treated as zero.

Focused comparisons use three shapes and three kernel candidates, with one
correctness warm-up and three measured calls per kernel/shape. On
`[1,256,12,200]→512`, the previous canonical kernel's wall median is 3912.167 μs,
versus 1542.792 μs for the matrix kernel. Calls include allocation, submission,
completion and readback. Samples are noisy; all three available device queries
also favor the matrix kernel. The other two shapes cover a smaller H12 map
and an existing large-reduction matrix case. These experiments do not establish
a new speedup for layers already using the unchanged matrix kernel.

Peak and retained device byte counts match the baseline exactly for every
preset. Upload/readback payloads, submission and dispatch counts are unchanged.
All normal samples retain GPU placement, no CPU compute stages, no diagnostic
tensor downloads and zero layout conversions.

## Rejected experiment

A separate prototype grouped four matrix fragments in scratch, reducing
explicit initialization/output barriers from forty to ten per tile phase.
It preserved FP32 arithmetic and passed the focused correctness checks, but
increased shared storage from 4 KiB to 16 KiB and was slower on all three
focused shapes. It was discarded before OCR measurement. No claim is made
about the precise hardware cause of that regression, and none of its shader
or scratch changes remain in the selected implementation.

The rejected version consumed 0.306 seconds of performance execution. The
selected version consumed **5.333 seconds**, including the focused harness,
warm-ups, nine ordinary OCR samples and three separate profiles. Builds and
standalone correctness checks are excluded. Each version stayed below the
sixty-second budget; their samples are preserved separately and not pooled.

## Local validation and reproduction

- Release library tests: **394 passed**, 67 explicitly ignored.
- Six focused checks passed: two selection/fallback checks and four physical
  GPU tests covering tails, subgroup workgroups, bias, fused residual/activation,
  output offsets and existing register-policy boundaries. Exact test filters
  each selected one test. Scalar tolerances were unchanged.
- Nine FP32 CPU/Full OCR comparisons passed: Tiny/Small/Medium on typical,
  dense and rotated fixtures, one invocation per placement. Text, line counts,
  order, orientation, recognition dimensions, CTC spans and score gates pass;
  maximum observed box-coordinate difference is zero. These related fixtures
  are not a diverse document corpus.
- All three timing configurations passed their CPU reference, warm-output
  repeatability, profile-output and residency checks.
- Strict all-features/all-targets Clippy, release example builds, formatting
  and whitespace checks passed. No CI, VM or remote runner was used.

Only Apple M4/Metal was executed. Other Metal adapters, Vulkan, DX12,
concurrent requests and dense-document latency are unmeasured here.

Local evidence is under `reports/gpu-matrix-extension-2026-10-24/` (ignored):
`performance/` freezes the executable, sources, model/input hashes and commands;
`kernel-results.json`, `gpu_tests` and `kernel-source.zip` preserve the focused
experiment; `qualification/` and the test logs preserve correctness outputs;
`rejected-staging16k/` preserves the rejected version. `summary.json` and
`summarize.py` audit all 165 current performance inputs, baseline artifacts,
settings and accuracy records without executing workloads again.

Build with `cargo build --release --locked --all-features --example ocr --example gpu_ocr_bench`.
Qualification uses `tools/full_gpu_validation.py --backend metal --precision f32
--presets tiny small medium --cases typical dense vertical --modes cpu full`
with an explicit new report directory. Timings use `tools/gpu_performance.py`
with the resulting corpus, `--configs 1:1:2 --measurements 3 --alignment --profile`
and an explicit remaining execution budget. The bounded ignored kernel test is
`runtime::gpu::tests::gpu_pointwise_cooperative_long_map_candidates`.
