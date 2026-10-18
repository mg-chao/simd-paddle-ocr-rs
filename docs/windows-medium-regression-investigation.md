# Windows Medium slowdown investigation — 2026-10-18

The earlier **10.6% Medium slowdown did not reproduce** in a bounded alternating comparison. The fresh baseline/current medians are **434.7797 / 435.0426 ms (+0.06%)**, and the means are **447.5210 / 446.9569 ms (−0.13%)**. Both versions have a slow sample. Combined with the dispatch audit below, the evidence supports benchmark variation rather than a repeatable regression caused by the PadEnd2 change. This is not a statistical guarantee of zero effect on every workload.

**Decision.** Retain the existing PadEnd2 optimization. No additional runtime, quality-setting, batching, or worker-policy change is justified. Tiny sparse remains faster in all three matched pairs. The correction is to the comparison method and the earlier interpretation: the [owner probe](../examples/cpu_owner_probe.rs) now supports externally sequenced paired calls, and the [plan probe](../examples/cpu_plan_probe.rs) can inspect kernel dispatch without inference.

The [portable evidence](reports/windows-x64-2026-10-18-medium-regression.json) retains every measurement, execution order, output digest, plan census, source/executable hash, build/test log, controller and diagnostic source. The [original adoption report](windows-avx2-pad-end-optimization.md) remains a separate experiment; no samples are pooled.

**Why the original result was inconclusive.** Its candidate Medium samples were 424.1598, 487.1084 and 508.0666 ms; its earlier baseline samples were 440.3400, 454.7224 and 434.4181 ms. A ratio of these non-interleaved three-sample medians conflates version and execution time. The candidate range already included a sample faster than every baseline sample. That observation warranted investigation, not a declaration of either regression or noise.

**Fresh controlled comparison.** Both release versions use the identical new harness. Baseline runtime sources come from `b30436a`; current runtime hashes match the previously validated optimization. Baseline was rebuilt in a separate target directory, and the two frozen executable hashes differ. Both processes read the same verified assets, manifest and predecoded images through `SIMD_OCR_PROBE_ROOT`.

Runs remain local on the physical Windows Ryzen 9 5950X host, with CPU24 affinity `ffffff`, automatic owners, batch one, automatic AVX2/FMA/layout/fusion, detection limit 960 and enabled orientation classification. Each process retains its own model/request/pool; both remain resident but inference never overlaps. Setup and stdin/JSON handling are outside each timed call. The timed boundary is unchanged: OCR from BGR24 pixels plus complete owned-result conversion.

The predeclared scope was exactly Medium dense `imgs_00018069` and Tiny sparse `imgs_1`, one warm-up and three measured calls per version/case, with a twenty-second experiment limit. Medium uses B/C, C/B, B/C order; Tiny reverses that order. There are sixteen OCR calls including four warm-ups, with no extra sampling or performance retries.

| Case | Baseline samples, ms | Current samples, ms | Median baseline → current, ms | Median change |
| --- | --- | --- | --- | --- |
| medium / imgs_00018069 | 478.1500, 429.6334, 434.7797 | 435.0426, 420.3893, 485.4387 | 434.7797 → 435.0426 | +0.06% |
| tiny / imgs_1 | 47.0818, 46.5998, 51.5710 | 43.0677, 45.6796, 45.4777 | 47.0818 → 45.4777 | -3.41% |

Medium’s matched changes switch direction: **−9.02%, −2.15%, +11.65%**. Its baseline slow call is 478.1500 ms; the current slow call is 485.4387 ms. The fresh process-CPU samples also vary in both versions, rather than showing a stable extra cost in current. This directly demonstrates run variation, although it does not identify whether scheduling, cache residency, boost behavior or another host effect caused it. No unsupported attribution to a particular background process is made.

Tiny’s current call is faster in every pair. Its median falls **3.41%**, from 47.0818 to 45.4777 ms. That supports preservation of the improvement without treating the earlier 10.4% page change as a guaranteed gain. The earlier 46–51% affected-convolution result remains separate evidence; it was not rerun.

**Causal dispatch audit.** The new parallel branch in [nchw.rs](../src/runtime/nchw.rs) requires eight output channels, eight SIMD lanes, stride-one 2×2 pad-end geometry and row-shard eligibility. The refactor also changes the shared serial canonical PadEnd2 writer, so inspecting only that new branch would be insufficient.

The raw and normalized Medium models have 64→32 and 32→64 2×2 layers. Their constant-weight shapes, unit dilation and output channels make them NHWC-capable under [conv_nhwc_capable](../src/runtime/optimized.rs); automatic layout selects NHWC. No changed canonical PadEnd2 writer is selected. Classification contains no 2×2 convolution at all. The read-only plan census confirms:

| Bound plan | Shape / budget | Selected canonical PadEnd2 steps |
| --- | --- | --- |
| Medium detector, dense page | `[1,3,256,416]`, 16 threads | 0 |
| Medium recognition, every observed dense width | `[1,3,48,W]`, W = 64, 96, 128, 160, 192, 224; 1 thread | 0 in all six plans |
| Shared orientation classifier | `[1,3,80,160]`, 1 thread | 0 |
| Tiny detector, sparse page | `[1,3,960,608]`, 16 threads | 1 |

These nine successful checks only compile/bind full graphs and execute zero inference calls. Recognition/classification production execute prefixes; early convolution eligibility is determined locally from the same normalized nodes and bound shapes, independently of the unused head. The earlier actual Medium detector profile also records its two stem convolutions as NHWC. Indirect compiler/code-layout effects cannot be excluded solely by dispatch inspection, which is why the paired measurements are necessary.

A detail in the earlier report was corrected: Small is not an unaffected control, because its canonical 2×2 layers can use the refactored serial writer. This follow-up does not make a new Small performance claim.

**Validation and limits.** All eight complete output digests per case match across baseline/current and the preceding experiment. Text, scores, boxes and order are preserved; retained activation/workspace/kernel-scratch sizes are unchanged. Before measurement, the frozen release test binaries passed six baseline and seven current focused pad-end checks. The new protocol rejects multiple owner configurations before inference; the actual paired run validates ready/warm-up/sample/completion order, all three sample numbers, process exits and exact outputs. Both release harnesses and the new plan probe build successfully.

Fresh experiment wall time was **6.030 s**. Conservative per-version process time, including time waiting for the other version, was **5.610 / 5.149 s**. Even adding the previous experiments gives **23.508 / 12.010 s**, below sixty seconds per runtime version. Build time and zero-inference correctness/plan checks are excluded.

Three samples cannot establish confidence intervals, tail latency or behavior on all hosts. Two processes resident together may affect caches/memory pressure compared with a standalone application; this exposure is shared and explicit. No production fix is made for an unconfirmed regression. The measured result warrants withdrawing the 10.6% regression interpretation while keeping the Tiny optimization.

**Reproducing the method.** Build `cpu_owner_probe` from each source snapshot using separate target directories. Set `SIMD_OCR_PROBE_ROOT` to the common repository containing `tests/baseline` and the adjacent models directory. Start each executable with `--paired medium imgs_00018069 0 1`; after both JSON `ready` events, issue one `warmup` command to each, then alternate three `run` commands per process in the recorded order. Read every event and the final `complete` report, verify hashes, enforce an external deadline, and avoid concurrent inference. The exact PowerShell controller is included in the evidence JSON.

Example plan-only inspection:

```powershell
cargo run --release --locked --example cpu_plan_probe -- ../SimdPaddleOCR/models/medium_rec.onnx 1,3,48,128 1
```
