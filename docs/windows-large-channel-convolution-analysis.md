# Large channel recognition convolution paths and optimization analysis

Large-channel recognition convolutions already use packed weights, channels-last activation storage, SIMD FMA reductions and fused final stores. The dominant Windows AVX2 path is a six-row by sixteen-output-channel full-K kernel. A new 96-versus-192-row grouping comparison found **no established improvement** on two actual Medium-model shapes. Earlier generic K blocking, input packing and large-K inlining also failed to establish a broad improvement. Further gains remain possible, but need a different mechanism or a more selective shape policy.

This analysis follows production revision `061c33b` on the physical Windows Ryzen 9 5950X host. Source edits add only a `cfg(test)` grouping control and reproducible diagnostics; production dispatch and arithmetic remain unchanged. The [pipeline analysis](windows-current-pipeline-analysis.md) establishes the page-stage bottlenecks. This report examines the convolution implementations underneath them, preserving the distinction between measured gains and optimization hypotheses.

## Shapes and measured importance

For group-one 1×1 convolution, the channels-last activation is a matrix product: `M = batch × output height × output width`, `K = input channels`, and `N = output channels`. Each reduction starts with its native Conv bias, or zero if absent, and accumulates in ascending input-channel order. Fused graph Add, activation and residual operations follow in graph order. A graph bias Add after reduction must not be moved into the accumulator seed; that changes rounding. The new synthetic probe explicitly uses a native Conv bias.

The [current production-prefix profiles](reports/windows-x64-2026-10-17-current-pipeline-analysis.json) use one warm-up and one instrumented pass per model, deterministic synthetic tensors and one logical CPU. They exclude the final vocabulary projection.

| Prefix shape | Whole backbone physical steps, ms | All 1×1 convolutions, ms | 1×1 share |
|---|---:|---:|---:|
| Small `[1,3,48,128]` | 9.4067 | 7.4926 | 79.7% |
| Medium `[1,3,48,640]` | 212.3399 | 191.1623 | 90.0% |

The following large products are identifiable in those profiles. This table groups the listed layers only, not every convolution with a large input-channel count.

| Model | M | K→N | Listed calls | Summed physical-step time, ms |
|---|---:|---|---:|---:|
| Small | 96 | 384→768 | 2 | 1.1147 |
| Small | 96 | 768→384 | 2 | 0.8982 |
| Medium | 960 | 512→1024 | 6 | 58.9528 |
| Medium | 960 | 1024→512 | 6 | 48.7517 |
| Medium | 480 | 768→1536 | 2 | 19.8609 |
| Medium | 480 | 1536→768 | 2 | 17.2453 |

The four listed Small calls consume **21.4%** of that backbone sample. The sixteen listed Medium calls consume **68.2%** of its backbone and **75.8%** of its pointwise work. Small also has substantial smaller-channel pointwise work; optimizing only the listed largest products cannot remove all of its 79.7% pointwise share. These serial percentages are localization evidence, not eight-owner page-latency shares.

## Layout selection and kernel dispatch

The physical planner in [`optimized.rs`](../src/runtime/optimized.rs:194) accepts group-one convolution into NHWC when weights are constant rank-four tensors, an optional base bias is constant rank one, dilation is one, and output channels are at least sixteen and divisible by sixteen. It propagates NHWC through compatible convolutions, depthwise operations, pooling, activations and residuals. It memoizes conversions at incompatible boundaries; recognition preprocessing can write the first input directly into the chosen layout.

All the listed large products qualify for NHWC. Their steady path is:

`NHWC physical plan → packed 1×1 eligibility → pointwise owner/band wrapper → ISA dispatch → row group → 16-column panel → six-row reduction → fused final store`.

| Path | Selection and implementation | Optimization assessment |
|---|---|---|
| NHWC direct pointwise | [`nhwc.rs:129`](../src/runtime/nhwc.rs:129): packed group-one 1×1, stride one, no top/left padding, unchanged spatial dimensions. [`nhwc.rs:383`](../src/runtime/nhwc.rs:383) dispatches it before general convolution. | This is the active path for the dominant large products. It reads existing NHWC rows without im2row or input transpose. |
| AVX2 full-K | [`pointwise.rs:96`](../src/runtime/nhwc/pointwise.rs:96), [`pointwise.rs:199`](../src/runtime/nhwc/pointwise.rs:199), [`pointwise.rs:278`](../src/runtime/nhwc/pointwise.rs:278): six rows ×16 outputs, two eight-lane accumulators per row, complete K reduction. | Main current-host target. SIMD, packed columns, independent accumulator chains and direct fused output stores are already present. |
| Windows short-K inline AVX2 | [`pointwise.rs:22`](../src/runtime/nhwc/pointwise.rs:22) and [`pointwise/avx2.rs`](../src/runtime/nhwc/pointwise/avx2.rs): input channels 96–192 and output channels at least 128. Inline assembly avoids the per-tile function/return boundary. | It covers some Small intermediate layers, but excludes the 384–1536-channel reductions. Extending it indiscriminately has already been tested without a clear large-product benefit. |
| AVX and AVX plus FMA | [`pointwise.rs:139`](../src/runtime/nhwc/pointwise.rs:139): same six-by-sixteen grouped traversal. AVX uses separate multiply/add; the FMA backend uses fused accumulation. | Share the cache/traversal issues, but preserve their distinct arithmetic contracts. They were not independently timed in this investigation. |
| AVX-512 | [`pointwise.rs:506`](../src/runtime/nhwc/pointwise.rs:506): six rows ×16 outputs with one sixteen-lane accumulator per row and fixed 48-row groups. | It does not inherit AVX2's 96-row policy. Separate tile/group tuning needs native hardware; this host does not expose AVX-512. |
| Portable4 | [`pointwise.rs:451`](../src/runtime/nhwc/pointwise.rs:451): two six-by-eight halves per sixteen-column panel, fixed 48-row groups, nonfused arithmetic. | A maintained fallback with different instruction and arithmetic behavior. Large AVX2 measurements do not validate changing it. |
| Scalar | [`pointwise.rs:506`](../src/runtime/nhwc/pointwise.rs:506) has a packed one-lane multiply/add body, but [`runtime.rs:460`](../src/runtime.rs:460) normally skips pack preparation for the scalar backend. | An already-packed scalar call/test can use that body. Ordinary scalar NHWC execution normally falls through to the generic scalar convolution instead. Neither is active on this AVX2 host. |
| ARM64 NEON | [`pointwise/neon.rs:28`](../src/runtime/nhwc/pointwise/neon.rs:28): six rows ×16 outputs, 24 accumulators, four K values loaded per input row and consumed with lane FMA, fixed 48-row groups. | Already a native implementation. Windows grouping and ABI conclusions cannot be transferred to it. |
| Nonpointwise NHWC dense | [`dense.rs:13`](../src/runtime/nhwc/dense.rs:13), [`dense.rs:127`](../src/runtime/nhwc/dense.rs:127): six-row packed tiles, 48-row groups, K blocks up to 512, fixed stack partials, direct checked input readers. | Already uses ordered blocking where spatial gathers make it useful. Its extra partial traffic is unnecessary for many direct 1×1 shapes; do not route all pointwise work through it without evidence. |
| NHWC depthwise | [`nhwc.rs:962`](../src/runtime/nhwc.rs:962): one input channel per group; x86 vector pixel kernels/strip paths and separate native NEON paths. | Spatial depthwise work remains relevant to overall REC, but is a different operation from group-one channel mixing. A depthwise kernel cannot replace these large pointwise products. |
| Other grouped convolution | [`nhwc.rs:841`](../src/runtime/nhwc.rs:841): falls through to scalar/vector spatial convolution after packed dense and depthwise checks. | Not the dominant listed REC route. No group-one pack is silently rebuilt inside this fallback. |
| im2row reference | [`nhwc.rs:904`](../src/runtime/nhwc.rs:904), `#[cfg(test)]`. | Test-only. There is no production im2row allocation here to eliminate. |

## Canonical NCHW alternatives

[`nchw.rs:135`](../src/runtime/nchw.rs:135) retains a different path when canonical layout is requested or geometry cannot stay in NHWC. Its pointwise choices include output-channel-major kernels for spatial planes below 48; a direct single-pixel variant; packed pointwise for output channels divisible by four; and spatial-vector fallbacks.

After the short-plane output-channel-major branch, the AVX2 canonical packed path selects a blocked implementation when input channels are at least 32, output channels are at least sixteen and divisible by sixteen, and per-batch work reaches four million work units. Its local predicate requires a spatial plane of at least 32, but the preceding branch already consumes matching planes below 48, making the effective minimum 48 for these shapes. It uses spatial tiles of 128, or 64 for planes above 4096. [`nchw.rs:2656`](../src/runtime/nchw.rs:2656) blocks K by 32, stages a transposed input tile and reloads/stores raw output partials, applying the epilogue only after the final K block.

This is an implemented alternative, not a missing optimization. Forcing it into the automatic REC backbone would add staging/partial traffic and can disrupt existing NHWC continuity. Compare complete physical plans only if there is evidence that its smaller working set compensates for those costs. The current large-channel measurements do not establish that.

## Weight packing and the full K loop

[`packing.rs:112`](../src/runtime/packing.rs:112) stores weights as `[output block][K][16]`. A model-owned cache shares immutable packed allocations through `Arc` across owners, widths and requests. The hot pointwise call receives an existing slice; it does not repack weights or lock the weight cache inside the reduction. The main channel sizes are divisible by sixteen, so padded output columns are not wasted work for the listed shapes.

The AVX full-K loop order is:

1. Choose a spatial group: 96 rows when both K and N are at least 384, otherwise 48.
2. Choose one full-K panel for sixteen output channels.
3. Reduce six activation rows against that panel.
4. Store the twelve vector accumulators through the fused output transform.
5. Continue with the next six rows, then the next output panel.

The large-channel policy also has boundaries: `384→192` stays at group 48, while `384→768`, `512↔1024` and `768↔1536` use 96. The selected group changes cache traversal, not the per-output reduction order or number of multiply-accumulates.

The archived `tile_fma<8>` assembly in `target/recognition-kernel-analysis-20261017/pointwise-tile.asm.txt` matches the unchanged full-K reduction source. Each K step has **twelve vector FMAs, six scalar broadcasts and two vector weight loads**, with twelve live accumulators. It has no accumulator spills or bounds checks inside K. The Windows function boundary saves/restores nine XMM registers and returns a 384-byte tile outside the loop. The caller then loads the returned tile and applies final stores. This is archived assembly evidence, not a newly generated disassembly of the diagnostic binary.

There is tangible ABI/store overhead, but the prior full-K inline experiment found no clear gain on large reductions. At large K, this fixed overhead is amortized over hundreds or thousands of FMA steps. Increasing rows at the same sixteen-output width would require more accumulators and risks spills; a larger tile cannot be assumed faster.

## Cache working sets

A full-K sixteen-output panel occupies `64 × K` bytes. The six input rows consumed by one reduction add `24 × K` bytes:

| K | Weight panel, KiB | Six input rows, KiB |
|---:|---:|---:|
| 384 | 24 | 9 |
| 512 | 32 | 12 |
| 768 | 48 | 18 |
| 1024 | 64 | 24 |
| 1536 | 96 | 36 |

The prior local host probe reports 32 KiB L1 data and 512 KiB L2. These byte counts justify investigating reuse; they do not prove which cache level or execution unit limits the kernel. Hardware miss, bandwidth, frequency and execution-port counters were not captured.

Increasing a group from 96 to 192 can reduce revisits to the complete matrix while enlarging the activation span reused between output panels:

| Shape M,K,N | Full weights | Group count 96→192 | Group activation span 96→192 | Nominal full-matrix walks 96→192 |
|---|---:|---:|---:|---:|
| 960,512,1024 | 2 MiB | 10→5 | 192→384 KiB | 20→10 MiB |
| 480,1536,768 | 4.5 MiB | 5→3 | 576→1152 KiB | 22.5→13.5 MiB |

Those nominal walks are **not DRAM traffic**. The same full panel is still loaded by every six-row leaf call: both policies execute about 320 MiB and 360 MiB of vector weight-load bytes, respectively, for these two shapes. Cache hits can satisfy those loads. A larger group changes where they are satisfied, as well as activation/output locality; it does not halve the instruction-level weight loads. For M=96, both grouping policies would process the same single group, so that shape cannot test added row reuse.

## Existing experiments and what they rule out

The [earlier pointwise report](reports/windows-x64-2026-10-17-pointwise-optimization.json) already measured several plausible changes. Its samples belong to those source versions and are not pooled with the new grouping probe.

| Earlier change | Recorded large-shape result | Interpretation |
|---|---|---|
| Ordered K=256 blocking, group 96 | `960,512,1024`: 7.3864→7.6660 ms, **3.79% slower**. `480,1536,768`: 8.2432→8.4715 ms, **2.77% slower**. | Better panel residency did not repay the partial-state/loop overhead on these products. The smaller `96,768,384` case improved 2.13%; that does not justify broad dispatch. |
| Input packing plus K=256, group 48 | Three products regressed **4.24–7.73%**. | Additional activation packing is not a free improvement over direct NHWC rows. |
| Inline full-K large reductions | Approximately 0.10–0.81% lower medians, all observed ranges overlapping. | Removing the isolated tile boundary alone did not establish a gain. |
| Explicit row stores with large reductions | Approximately 0.42–0.70% lower medians, overlapping ranges. | No established large-product improvement. |
| Two-step K unrolling | Large-product ranges overlapped; medians included regressions. | More unrolling has not demonstrated a useful benefit. |
| Extending inline dispatch through K=256 | Medium page median regressed 2.41%. | The retained short-K predicate is intentionally narrower. |
| Old 48→96 grouping comparison | Final paired medians 0.4113→0.4124 ms and 7.3254→7.3717 ms, overlapping ranges. | The existing 96-row policy is not a proven universal win. Earlier source versions had small differences in the opposite direction; do not pool them. |

The final grouping comparison is recorded in [the Windows follow-up evidence](reports/windows-x64-2026-10-16-followup-optimization.json). The original isolated 0.4251/7.4918 ms kernel diagnostic overlapped assembly compilation and is not used as a clean performance baseline here.

## Parallelism and fused output paths

[`nhwc.rs:24`](../src/runtime/nhwc.rs:24) begins pointwise parallel work at two million multiply-accumulates, balances complete six-row tiles, and uses recursive Rayon stack jobs with disjoint output bands. It does not allocate a heap vector of band jobs. The actual worker count is bounded by requested threads, CPU capacity and reserved worker workspaces.

Dense-page AVX2 automatic scheduling already chooses eight independent recognition owners when the page has enough units. With an eight-thread CPU context, those kernels receive one thread each. More nested jobs would therefore not fix a missing dense-page parallel path. Sparse pages can receive nested kernel threads; changing their thresholds is a separate question. Shape-aware group tuning on one logical CPU does not establish the result with eight owners contending for cache resources.

[`pointwise.rs:175`](../src/runtime/nhwc/pointwise.rs:175) calls `StoreTransform::vector_channel`, so channel-index recovery does not require division at each NHWC store. [`optimized/store.rs`](../src/runtime/optimized/store.rs) binds common operand/activation recipes before execution and applies them while writing the final convolution output. This already avoids separate intermediate activation/residual passes.

The prepared recipe still has a per-vector selection in its implementation. Hoisting a frequent recipe into an operator-level typed specialization is a concrete secondary candidate. Its ceiling should first be bounded by timing the actual fused layer against an identity-store control. Preserve Add/activation ordering, channel operands and full residual indexing. The current store code deliberately shares activation call sites to control code and stack-frame growth; indiscriminate specialization could lose that benefit.

## Remaining optimization candidates

The remaining opportunities are hypotheses until a matched experiment establishes their value. The new grouping results below further narrow the search:

1. **Test one alternative register tile.** A four-by-sixteen or eight-by-eight tile changes register pressure, input broadcasts and weight reuse. Six-by-sixteen is already close to its named-register budget; increasing the row count without reducing the output width is particularly risky. Preserve ordered accumulation and fused stores, inspect the generated inner loop, and measure the actual expansion/contraction shapes. This is a more substantive next experiment than merely expanding the full-K inline predicate again.
2. **Measure frequent fused-store recipes before specializing them.** Existing fusion is valuable; any gain would come from reducing dispatch/index work inside the final stores. Expect a smaller ceiling than changing the dominant reduction, and control code-size growth.
3. **Inspect real-model packed-weight alignment before changing allocation.** The current `Box<[f32]>` packing code has no explicit 32- or 64-byte alignment, but both new synthetic test allocations were already 64-byte aligned. Alignment is therefore not an observed defect in these measurements. If real-model addresses expose a repeating unfavorable offset, compare deliberately aligned storage with the same values, layout and FMA order. This is not a claim of doubled memory traffic: sequential loads can reuse adjacent cache lines. Include ownership/deallocation and retained-capacity accounting if allocation changes.
4. **Treat batching and shared-cache behavior as page-level questions.** Exact-width batch size two can improve weight reuse, but reduces independent recognition units and serializes per-batch preprocessing. This dense page still has enough units for eight owners. A kernel-local gain must survive complete-result and memory checks under the production owner policy.
5. **Tune other ISAs independently.** AVX-512 and NEON retain distinct tiles and 48-row policies. Their performance needs physical native-host measurements; no result here validates a change to them.

Generic K blocking, full-K inlining or activation packing should be revisited only with a mechanism materially different from the tested variants. A dominant stage is evidence of opportunity, not proof that a particular rewrite will improve it.

## New row grouping comparison

The test-only [grouping probe](../src/runtime/nhwc/pointwise/large_channel_tests.rs) compares the production adaptive group (96 on both chosen shapes) with a fixed 192-row group. Both use the same packed allocation, input, bias, six-row FMA leaf and identity final store. The only intended difference is the grouping implementation. It runs in a release test executable with the profiling feature present; no stage timers execute in these direct kernel calls.

Each variant receives one warm-up and three single-call measurements, in fixed AB/BA/AB order, on one logical CPU selected from the allowed process mask. Output equality is checked outside timed calls. There is no internal repetition, automatic sampling growth or retry. Builds and other agent work finished before timing.

| M,K,N | Production 96-row samples, ms | 192-row samples, ms | Median 96→192, ms | Median latency change |
|---|---|---|---|---:|
| 960,512,1024 | 7.4695, 7.3626, 7.3720 | 7.3710, 7.3655, 7.3924 | 7.3720→7.3710 | −0.014% |
| 480,1536,768 | 8.2408, 8.2467, 8.3060 | 8.2057, 8.2982, 8.5851 | 8.2467→8.2982 | +0.624% |

Both observed ranges overlap. Neither shape establishes a benefit; the second does not establish a reliable regression either. Keep production's 96-row policy. Doubling the group reduced nominal matrix revisits but did not establish an acceleration of the tested full-K kernel. The larger sample in the second 192-row set is retained as observed.

Both packed-weight pointers and input pointers had address modulo 64 equal to zero. That establishes alignment only for these allocations, not every model tensor or allocator run. The two variants have separate output allocations, so output placement remains a source of noise. Three samples are not a confidence interval. These are single-owner synthetic identity-store measurements; no fused-layer, eight-owner throughput or full-page improvement is claimed. No page benchmark was needed after this candidate failed to demonstrate a kernel gain.

The [raw evidence](reports/windows-x64-2026-10-17-large-channel-analysis.json) preserves every sample, allocation offset, executable/source hash, exact order, output checks, prior-report hashes and runtime diff. The fixed 192-row option remains test-only. Current runtime dispatch and model settings are unchanged.

## Validation and reproduction

Before performance measurement, **22 focused pointwise checks passed**, with four timing diagnostics ignored. They cover canonical and NHWC pointwise paths, worker bands, bias-first FMA behavior, nonfused fallbacks, fused epilogues and guards. The new scalar-FMA oracle case crosses a 192-row boundary, uses reduction/channel tails, tests bias present and absent, and verifies an unaligned output slice with guard sentinels. Both warm-ups and all twelve measured kernel calls produced bit-identical outputs between group policies.

The new performance process took **0.243 seconds**, including initialization and comparisons, with a ten-second external deadline and a ten-second internal bound. There were sixteen total kernel calls: four warm-ups and twelve measured calls across two shapes and two variants. No full pipeline, larger sample sweep, remote runner or VM was used. Even conservatively adding the prior current-runtime analysis execution (20.274 seconds) leaves this investigation below 60 seconds, excluding build and correctness time. Earlier experiments remain separate source-version evidence.

```powershell
cargo test --release --locked --features profiling --lib pointwise -- --test-threads=1
# Run only after correctness/build activity finishes; enforce a 10-second deadline.
cargo test --release --locked --features profiling --lib runtime::nhwc::pointwise::large_channel_tests::large_channel_group_comparison -- --exact --ignored --nocapture --test-threads=1
```

The exact bounded runner is retained at `target/large-channel-analysis-20261017/run.ps1`; its machine-local executable path comes from the focused build. `git diff --check` and formatting checks for the changed Rust modules passed. Formatting traversal of older child test files reports existing differences; those unrelated files were left unchanged.
