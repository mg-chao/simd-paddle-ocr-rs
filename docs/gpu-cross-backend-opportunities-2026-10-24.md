GPU backend comparison and optimization opportunities — 2026-10-24

The strongest opportunity is to apply cooperative matrix operations to the
recognizer's convolutions. SimdPaddleOCR implements this with native Metal
matrix shaders, FP16 operands, FP32 accumulation and a layout designed for
those shaders. Rust's current Full GPU path uses FP32 conventional arithmetic
for the same expensive convolution families. Kernel choice, precision, layout
and scheduling all differ; the existing measurements do not isolate their
individual contributions to the performance gap.

There is also a newly verified selection limitation: **this physical M4
advertises cooperative matrices through the locked wgpu 30.0.1 dependency,
but Rust rejects them because the advertised minimum subgroup size is four.**
A native Metal rewrite is therefore not a prerequisite for investigating
matrix acceleration. Start with the existing wgpu path and establish its
correctness and throughput on actual convolution shapes.

This analysis examines the current Rust working tree, including its uncommitted
pointwise optimizations, and sibling SimdPaddleOCR revision
`33ff8a81672452041d110e8ba52dcd2639e54228`. It changes no production code.

**Evidence and comparison limits.** I rechecked the saved local comparison
against all 164 Rust and 234 C# source/build-input hashes, seven model files,
two dictionaries, nine frozen C# build artifacts, the frozen Rust executable,
source archive and image/RGB hashes. All match. Consequently, another timing
run would repeat the same configuration without answering a new question.

These are the existing ordinary warm medians on the physical Apple M4, one
500×500 image with sixteen detected lines, recognition batch two, one request,
one Rust GPU slot and CTC spans enabled. Each configuration has one cold
warm-up, three measured runs and one separate diagnostic profile.

| Preset | Rust Full GPU, ms | C# Metal, ms | Rust/C# latency |
|---|---:|---:|---:|
| Tiny | 63.925 | 19.967 | 3.20× |
| Small | 181.491 | 43.492 | 4.17× |
| Medium | 598.185 | 162.961 | 3.67× |

This compares the implementations' current precision and execution policies,
not equal-precision kernels. Text and line counts agree, but the full output
contracts do not: one line has different classification/rotation metadata
even between the CPU implementations, and Small C# Metal has a CTC span
difference from C# CPU. C# timings use explicitly recorded self references;
their `qualified` flags do not prove equivalence to Rust. See the
[original comparison](csharp-comparison-2026-10-24.md) for samples, settings and
the complete differences. The Rust performance records report no CPU fallback,
no diagnostic tensor downloads and zero layout conversions.

Reaggregating only the profiles from that same comparison gives the following
device durations. These are separate instrumented calls, **not an additive
breakdown of the ordinary wall times above**. Metal profiling splits neural
dispatches into individual passes; normal execution groups consecutive
dispatches. Missing image/geometry/decoding timestamps number 6/4/6 for
Tiny/Small/Medium; they are excluded, not treated as measured zero.

| Device work, ms | Tiny | Small | Medium |
|---|---:|---:|---:|
| Detector graph | 10.919 | 27.104 | 121.553 |
| Classifier graph | 6.040 | 6.073 | 6.085 |
| Recognizer graph | 35.232 | 124.272 | 437.538 |
| Recognizer pointwise subset | 19.801 | 66.589 | 326.794 |
| Vocabulary and classifier projection | 6.257 | 20.301 | 33.404 |
| DB geometry, available records | 2.466 | 2.356 | 2.300 |
| CTC, available records | 0.117 | 0.461 | 0.474 |

Pointwise layers account for 56.2% / 53.6% / 74.7% of recognizer device time.
Medium's `pointwise_packed32` alone accounts for 262.001 ms. Its original
recognition nodes 300/309/317/326 together account for 63.555 ms, including
their fused epilogues across recognition units. No occupancy or memory-bandwidth
counters were collected, so these durations locate expensive work without
establishing whether it is limited by arithmetic, memory or occupancy.

**Implementation differences that matter.**

| Area | SimdPaddleOCR | Current Rust | Implication |
|---|---|---|---|
| Matrix arithmetic | `gemm.metal` uses `simdgroup_half8x8` operands and `simdgroup_float8x8` accumulators, composing a 64×64 workgroup tile. `EmitMm` selects it after a live correctness probe; eligible plain matrices use direct fragment loads. | `pointwise_blocked.wgsl` uses a 32×32 tile, K=16, 64 threads and four-by-four FP32 register accumulation with shared-memory barriers. | Matrix convolution is a larger architectural opportunity than another small weight-loader adjustment. Matrix instructions alone do not establish an achievable speedup. |
| Precision | FP16 weights and activation arenas, with FP32 accumulation and selected FP32 boundaries. | Full detector and recognizer explicitly call `with_full_precision()`. Mixed storage elsewhere still uses FP32 arithmetic. | `shader_f16` or `MixedF16` alone cannot reproduce the C# path. |
| Layout and packs | Channel-contiguous NHWC activation slabs; tap-major spatial-convolution weights, transposed MatMul weights and vector depthwise weights. | Full FP32 graphs use canonical NCHW/row-major storage and already share packed pointwise weights. | Layout should be designed with the selected kernel and retained across a region. Existing zero-conversion counts are not evidence that NCHW is intrinsically optimal. |
| Recognition scheduling | `RunMany` interleaves independent shapes by dispatch level using a concurrent Metal encoder, and streams a few waves. | With one slot, each recognition unit becomes a wave; at most two waves remain pending. Complete graphs are encoded in sequence. | There is an opportunity to expose independent work without padding different widths to the same tensor. More slots alone do not implement C#'s level interleaving. |
| Plans and buffers | Shared model cache of up to 512 metadata schedules; session buffers grow without changing schedule offsets. Metal buffers use shared mappings, with host input/output copies still present. | Eight previous shape plans plus the current plan; graph bindings depend on buffer generations. Resident image/head dispatches create fresh parameter buffers and bind groups. | Shared immutable shape schedules and per-wave parameter arenas could reduce rebuilding and allocation. |
| CPU/GPU division | CPU preprocessing, geometry and vocabulary projection/CTC; CPU projection for early waves overlaps later GPU work. | Full mode keeps these stages on the GPU and reads compact results. | Full residency saves transfers but does not guarantee lower latency. Compare individual stages and overlap, rather than moving the whole pipeline back to CPU. |

Primary source entry points are C#
[MetalGraph.cs](../../SimdPaddleOCR/src/Sdcb.SimdPaddleOCR/Backends/Metal/MetalGraph.cs)
(`ProbeSimdMatrix`, `EmitMm`, `ConstTapMajor`, `ConstGemmW`),
[gemm.metal](../../SimdPaddleOCR/src/Sdcb.SimdPaddleOCR/Backends/Metal/Shaders/gemm.metal),
[MetalRunner.cs](../../SimdPaddleOCR/src/Sdcb.SimdPaddleOCR/Backends/Metal/MetalRunner.cs)
(`RunCore`, `StreamCuts`, `RecordOne`) and
[PaddleOcrRecognizer.cs](../../SimdPaddleOCR/src/Sdcb.SimdPaddleOCR/PaddleOcrRecognizer.cs)
(`TryRecognizeUnitsBatched`). The corresponding Rust code is
[gpu.rs](../src/runtime/gpu.rs), [physical.rs](../src/runtime/gpu/physical.rs),
[pointwise_blocked.wgsl](../src/runtime/gpu/pointwise_blocked.wgsl),
[ocr/gpu.rs](../src/ocr/gpu.rs) and [resident.rs](../src/runtime/gpu/resident.rs).

**Recommended work order.**

1. **Make matrix convolution a qualified, selectable kernel family.**
   `GpuContext::new_with_options` requires `subgroup_min_size >= 8`, while
   the physical M4 reports `4..=64`. A fresh probe confirms advertised 8×8×8
   configurations for F32→F32, F16→F16 and F16→F32, and successfully creates
   a device requesting cooperative matrices, subgroups and shader f16.
   The existing hardware test passes using the portable fallback and prints
   `Cooperative physical support: false`; that label describes the requested
   device feature set, not the adapter's actual support.

   Do not merely delete the guard. `cooperative.wgsl` reserves 512 floats
   per scratch array, enough for eight subgroup tiles. A 64-thread workgroup
   with four-lane subgroups could require sixteen tiles. Make subgroup storage,
   collective control flow, tail handling and dispatch geometry valid for the
   supported execution range, or establish a narrower supported geometry.
   The existing host launch also allocates one workgroup per tile although
   each workgroup processes `num_subgroups` tiles, causing excess masked work
   when there is more than one subgroup. Neither issue contributes to the
   measured production runs because this experimental path is disabled.

   More importantly, current replacement in `GpuSession::bind` only targets
   `MatMul16/32`: it does not target convolution recipes or resident vocabulary
   heads. Medium's ordinary recognizer MatMul kernels total only 4.133 ms,
   compared with 326.794 ms of pointwise work. Enabling the existing option
   alone therefore misses the primary bottleneck.

   First implement FP32 matrix-backed pointwise convolution, preserving the
   storage policy to separate kernel effects from quantization. NCHW pointwise
   naturally permits `W[Cout,Cin] × X[Cin,HW]`; a whole-graph NHWC conversion
   is not required for this first experiment. Compose multiple matrix fragments
   per workgroup and retain fused bias/residual/activation epilogues. Start
   with actual `[1,1536,3,240]→768` and `[1,1024,6,240]→512` shapes, plus one
   smaller/batched case. These come from the earlier focused shape experiments.

   The locked library already exposes the native experimental
   [cooperative-matrix feature](https://docs.rs/wgpu/30.0.1/wgpu/struct.Features.html#associatedconstant.EXPERIMENTAL_COOPERATIVE_MATRIX)
   and [operand/accumulator properties](https://docs.rs/wgpu/30.0.1/wgpu/struct.CooperativeMatrixProperties.html).
   Investigate a native Metal shader/backend only if a focused experiment
   demonstrates a specific wgpu/Naga limitation or a material throughput gap.

2. **Evaluate mixed precision and physical layouts together, under an explicit accuracy policy.**
   The second matrix variant should use FP16 operands with FP32 accumulation.
   This is distinct from storing halves and converting every scalar load to
   FP32. `Detector::new` and `Recognition::new` currently force FP32 because
   earlier half-storage experiments changed discrete box/CTC decisions.
   Even FP32 matrix operations can change summation order; neither variant
   should bypass numerical and end-to-end semantic checks.

   If half precision is useful but cannot satisfy the current contract, keep
   a strict path and evaluate an opt-in policy with documented acceptance
   criteria. Keep sensitive boundaries in FP32 where evidence supports that
   choice. Extend padded NHWC or another matrix-friendly layout over compatible
   convolution/depthwise/SE regions, with conversions only at boundaries.
   Merely changing the final layer back to FP32 cannot recover information
   already lost upstream. Do not infer accuracy from matching text on one page.

3. **Schedule multiple recognition shapes per submission with stable resources.**
   Current typical runs make sixteen submissions and 1,102 / 2,012 / 2,100
   dispatches. Recognition has thirteen warm plan-cache misses and zero hits
   per call for each preset. C# instead shares shape schedules and interleaves
   independent units in a few streamed waves. Prototype a bounded multi-shape
   wave with disjoint activation regions and immutable per-dispatch parameter
   slices; preserve equal-width batching and reading order.

   Rust's comments correctly warn that reshaping can overwrite parameter
   storage. Refactor parameter ownership before accumulating more unsubmitted
   graphs. To reproduce C#'s overlap, verify the backend's encoder ordering
   rather than assuming that multiple queued graphs run concurrently. A larger
   plan cache is a smaller experiment, not a substitute for this design; prior
   cache/slot experiments did not establish a general latency gain. Measure
   uninstrumented wall time, allocation counts and retained/peak memory.

4. **Extend matrix/spatial specialization after the pointwise prototype.**
   Small REC node 19 still costs 9.822 ms in `conv_implicit16`; its earlier
   spatial nodes 4/8 add 6.100/4.917 ms. Medium DET node 268 costs 11.600 ms
   in `conv_implicit32`, while node 172 costs 6.382 ms in generic 9×9
   depthwise convolution. C# has implicit matrix spatial convolution and a
   tiled depthwise path supporting 9×9. Rust's depthwise specialization only
   covers 3×3 and 5×5. Test one actual shape for the chosen family, including
   padding, stride, channel tails and fused output placement. The C# source
   itself records that a larger K tile did not win universally; copying tile
   constants without a shape-specific comparison is not justified.

5. **Improve vocabulary projection, then assess CPU overlap as a separate option.**
   Rust already avoids the full logits readback and reduces winners on the GPU.
   `heads4.wgsl` reuses weights across four rows but still loops over the entire
   feature dimension with ordinary arithmetic. Test a tiled matrix projection
   with on-device partial winner reduction, retaining FP32 where needed.
   Preserve lowest-index ties, invalid-value handling, CTC spans and Probability
   mode's full-vocabulary normalization. The default Logit fast path already
   skips probability work; adding it again is not an opportunity.

   C#'s batched CPU ArgMax/CTC overlaps later GPU waves. It is worth a bounded
   optional comparison after the graph is faster, including feature readback
   and CPU usage. C#'s `rec_graph` profile includes this streamed CPU tail,
   while Rust's device `recognizer` category excludes projection, so those
   two profile numbers are not equivalent stage measurements.

6. **Reduce host overhead after measuring its share of ordinary execution.**
   Reuse resident uniform/bind resources, coalesce compatible resident passes,
   combine crop metadata generation with box output, and fold orientation
   into recognition sampling. These target allocations and synchronization
   boundaries. Neural graph pipelines, weights, bindings and grouped passes
   already have reuse; do not treat every neural dispatch as a fresh pipeline
   or allocation. Current per-image payloads are 750,000 uploaded bytes and
   only about 26–29 KiB of output/control data. This is not evidence of a
   bandwidth-bound transfer path. Completion wait overlaps GPU execution and
   must not be added to device time as independent overhead. Startup pipeline
   compilation is a separate objective from the warm gap.

The earlier packed-weight, H12/H1 selection, RGB stem, convolution epilogue,
SE/attention/layer-normalization, concat-placement and Logit-head changes are
already present. They should be baselines, not proposed as new fixes. CTC and
attention are comparatively small in the current profiles. With Medium
pointwise work at 53.7% of available device time, even a hypothetical 4×
pointwise speedup gives only about 1.67× improvement in that device-time sum;
this is an Amdahl illustration, not a forecast of ordinary wall latency.

**Vulkan scope.** C# also has a capability-specific Vulkan matrix design:
`VkDevice.CoopGemm` selects supported subgroup/matrix configurations, and
`conv1x1_cm_sg32.comp` implements FP16→FP32 matrix pointwise/spatial kernels
with staged/direct variants. Rust's current experimental gate instead asks
specifically for 8×8×8 F32 operands/results. Supporting Vulkan effectively
requires selecting advertised shapes/types rather than hard-coding the Metal
geometry. No Vulkan/DX12 performance or numerical execution was performed in
this analysis; the M4 ratios must not be generalized to those devices.

**Validation performed and next acceptance gates.** One fresh release hardware
correctness test passed on the physical M4:

```sh
SIMD_OCR_GPU_BACKEND=metal cargo test --release --locked --all-features --lib \
  runtime::gpu::tests::gpu_hardware_resident_fusions_and_experimental_selection \
  -- --ignored --exact --nocapture --test-threads=1
```

The initial unqualified `--exact` filter selected zero tests and is not counted
as validation. The subsequent fully qualified invocation ran one test. Its
matrix case validates the fallback, not matrix execution. The separate adapter
probe ran once and validates capability reporting/device creation only; it
does not prove matrix shader correctness or performance. Initial direct-rustc
probe compilation failed due to split dependency artifacts; the final isolated
Cargo probe built successfully. No production files were edited and no new
performance runs were made. No CI, remote runner or VM was used.

For the first implementation, require an assertion that the intended matrix
kernel actually ran, focused numerical comparisons on at most three hot shapes
with tails/fusions, then one warm-up and three release measurements per shape.
Keep cumulative performance execution within sixty seconds per code version,
including internal repetitions, and rebuild after each change. Follow with a
bounded OCR contract check for text, boxes, orientation, scores and spans before
attributing any speedup to a deployable optimization.

Reproducible analysis artifacts are under the ignored
`reports/gpu-cross-backend-analysis-2026-10-24/`: `analysis.json` and
`analyze_saved_runs.py` preserve provenance checks and extracted costs;
`adapter_probe.rs`, its isolated `Cargo.toml`/lockfile and `adapter-probe.log`
preserve the live capability check. The original timing samples remain under
`reports/csharp-comparison-2026-10-24/`. The evidence covers one physical GPU,
one image and one request/batch configuration, and contains no controlled
ablation assigning the complete cross-implementation gap to one mechanism.
