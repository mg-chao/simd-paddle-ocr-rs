# GPU performance phase: objectives 1–4

Scope: fair C#/Rust comparisons and bottleneck evidence; bounded recognition
scheduling; FP32 kernel/layout efficiency; measured dense-page geometry work.
The starting tree contains the full-GPU implementation and qualification reports
dated 2026-10-23 on base commit 277b11c. Preserve that work and its historical
reports. The four objectives are implemented and qualified within the hardware
and corpus limits recorded below; stretch performance targets remain open.

## Acceptance and evidence

1. Compare identical model/input/options/output contracts across Rust CPU,
   hybrid, Full, and C#. Separate construction, first use, warmed performance,
   and instrumented runs. Control requests, GPU slots, and recognition batch
   size independently. Pin source/build/model/input identities; preserve failed
   equivalence checks. Rank the three largest measured costs per preset.
2. Reduce avoidable waits between recognition waves using bounded ownership of
   pending resources. Preserve output order, error handling, quotas, and session
   lifetime rules. Measure single-request latency and concurrent throughput;
   close the Full/hybrid gap where possible and explain remaining limitations.
3. Evaluate FP32 pointwise/layout optimization on production shapes, then select
   additional kernels only from measured costs. Require an end-to-end benefit,
   not just isolated timing improvements.
4. Profile dense geometry, optimize justified costs, and validate canonical
   ordering, candidate limits, holes, borders, perspective and rounding.

All changes preserve exact text, orientation, count, reading order and CTC spans;
FP32 score/tensor gates remain 1e-3 + 1e-3 * abs(reference), coordinates <=0.5
source pixels. Normal Full calls have no CPU compute or diagnostic transfers.
C# GPU REC failures remain explicit; passing tensor comparisons alone are not
OCR semantic qualification. Include real local documents beyond generated
fixtures in final correctness coverage where available.

Provisional targets on RX 6750 XT/Vulkan/tiny/FP32: typical single <=41 ms,
dense single <=120 ms; first match hybrid throughput (26.86 and 7.99 images/s
for two typical/dense requests), stretch 30 and 8.8. These are targets, not
promised results or universal device requirements.

## Work log

- Source inspection: resident batches currently omit device timestamp profiles;
  recognition reads each wave synchronously; NHWC pointwise selection is mixed
  precision only. The benchmark couples GPU slots to request count and checks
  joined text only. Instrumentation and comparison contracts are the first work.
- Validation policy: physical local machine, release builds; focused correctness
  before performance. Default at most one warmup and three measurements, 60 s
  performance execution per version excluding build. Expanded experiments need
  a concrete question, declared bounded scope, and separate raw evidence.
- Added resident device timestamp profiles (including neural graph dispatches),
  host encoding/admission counters, dispatch/layout/allocation counts and separate
  timestamp readback accounting. Physical Vulkan query-page rollover test passed
  (260 dispatches, profiled/unprofiled outputs identical). Graph profiles use
  batch-owned query ranges, so repeated classifier calls cannot overwrite earlier
  timestamps.
- Added `tools/gpu_performance.py` and the C# `--ocr-bench` reference adapter.
  Both consume identical raw RGB bytes and explicit CPU options (4 line workers,
  8 detector threads, 16 preprocessing workers), recognition batch and capture.
  Rust request count and GPU slot count are independent. Cold public outputs are
  checked before warm measurements; full outputs, failed comparisons, frozen
  binaries and source/model/input hashes are preserved. C# public results own
  their storage; Rust owned conversion is outside the timed inference interval.
- Initial cross-preset comparison: `reports/gpu-performance/phase1-baseline`,
  15 configurations (3 presets x Rust CPU/hybrid/Full + C# CPU/GPU), one typical
  page, one cold/one measured call plus a separate profiling call for successful
  configurations. Declared total phase baseline budget 120 s, first run 28.44 s.
  All nine Rust configurations passed; all six C# comparisons failed semantic
  equivalence and correctly produced no qualified warm performance result.
- C# mismatch causal chain: the same tall box (tiny line 7, x414–429/y233–302)
  enters C# `Kernels/Warp.cs:89` / `Warp.Scalar.cs:45`, which map source (x,y) to
  (height-1-y,x), a clockwise rotation. Rust `crop.rs` intentionally maps a
  counterclockwise rotation (commit 997a01e). Classification consequently returns
  1/180 degrees in C# versus 0/0 degrees in Rust; final text can still agree after
  classification rotates it. The adapter neither swaps channels nor changes
  orientation. Preserve these reference differences, including small/medium
  confidence differences and additional GPU coordinate/text failures; do not
  weaken gates or modify the reference to manufacture parity.
- Device-time ranking (separate profiled calls, not latency): tiny recognizer
  40.02 ms, classifier 16.79 ms, detector 5.29 ms; small recognizer 93.79 ms,
  detector 10.58 ms, classifier 9.09 ms; medium recognizer 334.02 ms, detector
  53.10 ms, projection 10.89 ms. Medium's costliest recorded nodes
  210/300/309/317/326 are all ungrouped stride-one 1x1 convolutions. The generic
  implicit kernel still computes spatial kernel coordinates for each K tile.
  Evaluate a direct NCHW pointwise specialization as well as NHWC before choosing
  the production path; the original NHWC plane>=1024 threshold misses many costly
  recognition shapes (for example 3x240).
- Density baseline: `phase1-density`, 18 configurations, 51.55 s. All six
  Rust Full cases passed. C# CPU sparse passed all presets; C# GPU sparse passed
  tiny and medium. Dense comparisons failed the recorded semantic gates.
  Dense geometry totals were 5.90–6.09 ms: labels 4.09–4.42 ms, geometry
  1.49–1.66 ms, reading order 0.14–0.19 ms. Ordered label timestamp records
  identify phase 5 (serial block prefix scan) as 2.82–3.02 ms of labeling.
- Concurrency baseline: `phase1-concurrency`, tiny, three warm samples per
  configuration. Requests/slots 2/2: hybrid/Full typical 75.28/81.87 ms and
  dense 244.74/260.43 ms per two-image group. Requests/slots 1/2:
  87.70/53.39 ms typical, 245.63/165.99 ms dense. These use explicit CPU options
  and the new inference-return timing boundary; do not pool historical numbers.
- Pointwise experiment: one correctness warmup and three device timestamp samples
  for three shapes and three candidates, 6.20 s total on Vulkan. All candidates
  passed FP32 CPU checks. Median original/direct NCHW/NHWC (including conversion):
  [1,128,32,33]→128: 351/263/433 us; [1,1536,3,240]→768:
  2278/3753/3051 us (high variance); [2,192,6,136]→1024:
  2683/1791/4335 us. Select direct addressing for ungrouped, unpadded, stride-one
  FP32 pointwise kernels with input channels <=256; keep the larger reduction
  and existing mixed-precision policy unchanged. End-to-end validation pending.
- Implemented bounded recognition lookahead: submit the next wave before draining
  the previous one, with two pending waves at most. Queue submissions precede
  subsequent parameter writes; each pending result retains graph buffers,
  compact readback, timestamps, and the admission guard. Early errors drain
  submitted work before releasing reservations; completion failures poison the
  context. Admission guards cannot move between threads.
- Implemented two-level component prefix scan: 64 block totals per workgroup,
  followed by a serial scan of one total per 4096 pixels. Integer prefix order,
  foreground/background offsets and candidate truncation stay unchanged.
  Profiles now identify individual labeling phases.
- Focused physical checks passed on both Vulkan and DX12: 10 resident tests,
  21 hardware operator/lifecycle tests, and two OCR integration/quota tests per
  backend (66 test executions). Coverage includes timestamp page rollover,
  pending readback order and abandonment, allocation rollback, context failures,
  fused/prescaled convolution, CTC, perspective bytes, equal-key sorting, holes,
  dilation and candidate truncation. The non-ignored release suite passed
  365 tests (45 explicit hardware/performance tests remain ignored by default).
  Historical 27 Vulkan Full outputs also pass the strengthened width/time-step
  and CTC-score checks.

## Measured results (2026-10-23)

RX 6750 XT, Vulkan, FP32, identical RGB/model/dictionary hashes. Final runs are
`reports/gpu-performance/phase2-concurrency` and `phase2-presets`; execution took
27.92 + 28.01 = 55.94 seconds, including construction and cold invocations.
Each configuration has one cold call and three warm samples. Preset profiles
come from a separate instrumented call. Frozen binaries, source archives, raw
outputs and full settings are in those directories.

Single request, one GPU slot, recognition batch 2 (milliseconds):

| Preset | Page | Initial warm sample | Final warm median | Reduction |
|---|---|---:|---:|---:|
| tiny | typical | 67.01 | 40.47 | 39.6% |
| tiny | dense | 189.57 | 118.11 | 37.7% |
| small | typical | 140.93 | 114.24 | 18.9% |
| small | dense | 443.57 | 368.48 | 16.9% |
| medium | typical | 428.18 | 403.19 | 5.8% |
| medium | dense | 1481.61 | 1415.03 | 4.5% |

The initial preset matrix used one warm sample; percentages above are directional
observations, not confidence intervals or isolated attribution to one change.
Tiny's final ranges were 39.00–41.56 ms typical and 117.20–119.58 ms dense.
Both median single-request targets (41/120 ms) were reached in this configuration.
Cold-start cost remains material: the first final preset process recorded
5765.61 ms construction and 3934.48 ms first inference. Later processes were much
faster; driver/cache state was not controlled, so no cold-start improvement is
claimed. Construction, cold calls and instrumented calls never enter warm medians.

Tiny comparisons with three warm samples on both versions:

| Page | Requests / slots | Full before → after, group ms | Reduction | Final hybrid, group ms |
|---|---|---:|---:|---:|
| typical | 1 / 2 | 53.39 → 38.62 | 27.7% | 109.44 |
| typical | 2 / 1 | 97.17 → 77.35 | 20.4% | 86.65 |
| typical | 2 / 2 | 81.87 → 75.10 | 8.3% | 73.16 |
| dense | 1 / 2 | 165.99 → 157.11 | 5.3% | 288.22 |
| dense | 2 / 1 | 292.56 → 233.06 | 20.3% | 278.95 |
| dense | 2 / 2 | 260.43 → 241.94 | 7.1% | 245.67 |

At two requests/two slots, Full is 26.63 images/s typical and 8.27 dense,
versus hybrid 27.34 and 8.14. Typical remains 2.6% behind hybrid; dense leads
by 1.5%, within the uncertainty of three samples. One slot gave the best measured
dense throughput (8.58 images/s) and single-request latency (118.11 ms).
The existing one-slot default therefore remains appropriate; more slots are not
an automatic speedup. Stretch targets of 30 typical / 8.8 dense images/s remain
unmet. No additional sampling was used to select cleaner numbers.

Dense geometry fell from 5.90–6.09 ms to 3.13–3.29 ms. The two prefix phases
together cost 0.059–0.061 ms versus 2.82–3.02 ms for the old serial scan.
Label roots, integer prefix order, holes and candidate selection are unchanged.
The remaining largest costs are recognizer/classifier/detector for tiny,
recognizer/detector/classifier for small, and recognizer/detector/projection for
medium on typical pages. Medium recognition remains about 333 ms typical and
1185 ms dense; its large channel reductions are the next kernel target.

Full memory reservations remain bounded. Accounted peak buffers in the preset
matrix are 42.42/122.03 MiB (tiny typical/dense), 92.11/187.02 MiB (small), and
248.56/490.37 MiB (medium). Compared with the starting matrix, tiny is effectively
unchanged and small/medium increase by at most 2.04 MiB. This is library-accounted
buffer memory, not total driver VRAM. C# memory remains explicitly unavailable.

## Final qualification and limits

All 108 Full GPU cases passed: three presets × nine fixtures × Vulkan/DX12 ×
FP32/mixed contexts, each checked against its CPU oracle with CTC alignment
enabled. The final saved-output audit also requires the FP32 score gate for
DET/REC in mixed contexts; only the mixed classifier uses the mixed score gate.
Text, orientation, count, order, recognition widths/time steps and CTC spans
match. Maximum coordinate difference across these cases is **0 pixels**.
Normal Full results have no CPU compute stages or diagnostic tensor transfers.
Build-source and output identities are verified in
`reports/gpu-performance/summary.json` with links to the raw manifests.

The first correctness run reached its 360-second limit after 19 DX12 FP32 cases.
Only the remaining eight FP32 and 27 mixed cases continued under a separate
720-second limit, finishing in 508.72 seconds. This was correctness qualification,
not additional performance sampling. Completed comparisons were not repeated.

The corpus uses the only local document, `SimdPaddleOCR/examples/sample.jpg`, plus
derived dense, vertical, wide, perspective and border images and synthetic
tiny/holes cases. Independent real documents, Metal and other GPU vendors remain
unverified. C# qualified five of 18 configurations (horizontal sparse pages);
the other results retain their semantic failures and are excluded from speed
claims. The reference implementation was not modified to force agreement.

Next work should prioritize large-channel recognition kernels for medium,
the remaining typical-page concurrency gap, controlled cold-start measurements,
and independent document/device coverage. The current evidence does not justify
enabling FP32 NHWC conversion globally or claiming a universal C# speedup.

## Acceptance status

- [x] Objective 1: instrumentation, configurable comparison harness, C# adapter,
      physical baselines and ranked costs for all three presets.
- [x] Objective 2: bounded scheduling change, lifecycle/semantic checks, measured
      latency/throughput comparison.
- [x] Objective 3: measured FP32 specialization, affected-kernel correctness,
      full OCR qualification and performance.
- [x] Objective 4: measured geometry work, semantic regressions and performance.
- [x] Final audit: source-matched reports, strict correctness gates, actual
      placement/memory, broad affected tests, user-facing results and limitations.
