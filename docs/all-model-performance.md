# All-model scalar performance

The refined scalar Rust implementation is faster than canonical scalar C# in all 36 paired public comparisons on this frozen corpus. Optimized single-worker C# remains faster in all 36 of its pairs and is the later performance target. These comparisons measure the small frozen pages described below; they do not establish throughput for arbitrary documents.

All 180 timing processes and six untimed ordinary public semantic preflights passed. Six strict summaries and the final frozen-cohort validator passed, preserving every sample, process and tail without exclusions. Later documentation commits do not change the measured source, `a121646c56b3884c3a2c924a465455fbdf3b4052`.

## Frozen scope and controls

Every headline run selects `image-000`: BGR24 page `00-page.png`, 160 × 58 pixels, stride 480, 27,840 bytes, SHA-256 `c98940aac665ea80f28762b74421f24c9fc3602e402b9924786451d12658f39c`. Each family produces two lines and 16 dictionary tokens. REC recognizes two already oriented family-specific crops and returns owned text/CTC spans. Its corpus value sums matching sample indexes from two separately timed contiguous line blocks; it is a two-line aggregate, not batched page throughput. OCR measures the full page through detection, crops, classification/orientation, recognition and owned results.

| Family | REC crops, width × height | Resized / tensor widths; time steps | DET box threshold |
| --- | --- | --- | ---: |
| tiny | 154 × 23; 159 × 28 | 322 / 352; 44; 273 / 288; 36 | 0.4 |
| small | 156 × 24; 157 × 26 | 312 / 320; 40; 290 / 320; 40 | 0.45 |
| medium | 155 × 24; 158 × 30 | 310 / 320; 40; 253 / 256; 32 | 0.45 |

Models, dictionaries, all effective options and artifact hashes are frozen per family. Tiny uses 6,906 CTC classes; small/medium use 18,710. The complete three compact reference bundles are preserved with this evidence. Whole-bundle integrity is mandatory before case selection. First-call and after-30-warmup native checks require exact text, fields, order, token counts, dimensions, labels, rotations and CTC bounds; only scores and box coordinates use `abs(candidate-reference) <= 1e-5 + 1e-4 * abs(reference)`. Public scores are raw maxima and may exceed one. Unicode checksum length units differ between C# and Rust; checksum stability is a within-engine guard, while public JSON certifies cross-engine semantics.

Each engine/scope has six fresh processes, 30 warmups and 100 samples, alternating C# first on odd rounds and Rust first on even rounds. Windows affinity mask 1 pins logical CPU0. The host reports an AMD Ryzen 9 5950X, 12 cores / 24 logical processors, using the Balanced power plan. Each group records a full 30-second idle window, actual processes, RSS sampling, source/toolchain/flags and raw stdout/stderr. Observed CPU0 background means span 4.27–6.10%; the host is not claimed to have zero background. Canonical C# disables intrinsics, fusion, NHWC, tiered compilation and PGO; the CPU profile enables them. Both use one processor and DET/line/preprocess worker. Loading, preparation, heap observers and stage timers are excluded from warmed headline latency.

## Public latency and variation

Cells are **minimum / median / maximum** across six process p50 or p95 values, in milliseconds. `canonical` and optimized `cpu` are independent, separately labeled native comparisons.

| Family | C# profile | Scope | C# p50 | Rust p50 | C# p95 | Rust p95 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| tiny | canonical | REC aggregate | 97.342 / 99.676 / 101.301 | 49.242 / 50.388 / 54.514 | 102.486 / 106.919 / 111.045 | 54.429 / 56.081 / 59.434 |
| tiny | canonical | Owned OCR | 128.594 / 131.533 / 135.243 | 68.606 / 69.562 / 71.528 | 140.309 / 141.741 / 146.152 | 73.675 / 76.129 / 79.636 |
| tiny | cpu | REC aggregate | 12.075 / 12.232 / 12.391 | 49.461 / 52.437 / 55.766 | 12.410 / 12.586 / 12.770 | 53.581 / 58.112 / 63.836 |
| tiny | cpu | Owned OCR | 22.414 / 26.929 / 28.642 | 68.664 / 71.124 / 72.869 | 28.219 / 29.602 / 39.432 | 74.219 / 76.819 / 81.230 |
| small | canonical | REC aggregate | 564.509 / 567.212 / 570.257 | 264.778 / 265.547 / 269.571 | 582.742 / 585.168 / 586.370 | 275.694 / 278.426 / 281.345 |
| small | canonical | Owned OCR | 620.794 / 621.720 / 631.745 | 296.709 / 298.282 / 303.273 | 637.678 / 641.704 / 651.473 | 309.911 / 316.215 / 322.578 |
| small | cpu | REC aggregate | 47.779 / 50.527 / 55.810 | 266.349 / 270.597 / 273.820 | 63.000 / 66.996 / 69.321 | 283.693 / 286.527 / 296.668 |
| small | cpu | Owned OCR | 58.645 / 59.696 / 60.713 | 301.054 / 302.634 / 307.612 | 88.620 / 94.075 / 98.411 | 314.306 / 320.782 / 330.587 |
| medium | canonical | REC aggregate | 3115.432 / 3123.003 / 3144.396 | 1285.535 / 1291.937 / 1304.576 | 3175.814 / 3188.519 / 3210.641 | 1313.885 / 1318.669 / 1364.715 |
| medium | canonical | Owned OCR | 3468.914 / 3484.645 / 3504.313 | 1449.144 / 1469.305 / 1537.179 | 3532.301 / 3555.367 / 3583.831 | 1477.398 / 1503.014 / 1575.558 |
| medium | cpu | REC aggregate | 190.248 / 192.463 / 194.326 | 1289.366 / 1301.312 / 1392.784 | 197.667 / 202.805 / 203.930 | 1308.818 / 1336.926 / 1432.526 |
| medium | cpu | Owned OCR | 209.084 / 216.372 / 229.031 | 1454.457 / 1459.300 / 1511.747 | 219.682 / 228.133 / 239.565 | 1474.984 / 1488.638 / 1566.296 |

All canonical p50 paired C#/Rust ratios are 1.82–2.43×; all optimized CPU ratios are 0.140–0.402×, favoring optimized C#. Both percentile directions agree in every pair. The structured report retains all six p50/p95 ratios per scope, sample CVs and descriptive Tukey 1.5-IQR flags. Flags prompt inspection and remove no data. Tiny CPU C# OCR round 6 has the lowest p50 (22.414 ms) and highest p95 (39.432 ms); higher medium Rust rounds also remain included. No new CV or hosted performance threshold is inferred.

Rust-only `owned_copies` independently measures copying, observing and dropping the two recognition results. The canonical/CPU labels are separate Rust repetitions, not C# pairs or a C# profile effect. These small values approach timer resolution.

| Family | Cohort label | Copy p50 min / median / max (ms) | Copy p95 min / median / max (ms) |
| --- | --- | ---: | ---: |
| tiny | canonical | 0.000900 / 0.000900 / 0.001000 | 0.000900 / 0.001000 / 0.001100 |
| tiny | cpu | 0.000900 / 0.000900 / 0.001000 | 0.000900 / 0.001000 / 0.001100 |
| small | canonical | 0.000900 / 0.000900 / 0.001100 | 0.001000 / 0.001000 / 0.001200 |
| small | cpu | 0.000900 / 0.000900 / 0.001000 | 0.001000 / 0.001000 / 0.001000 |
| medium | canonical | 0.000900 / 0.000900 / 0.001000 | 0.001000 / 0.001000 / 0.001000 |
| medium | cpu | 0.000800 / 0.000900 / 0.000900 | 0.000900 / 0.000900 / 0.001105 |

## Production stage observations

Separate timer-enabled production-path profiling passed all 12 case contracts: one fresh pinned process per family, four contiguous case blocks, one first call plus three warmups and ten samples per case. It observes compact ordinary OCR with borrowed CTC results. These means are diagnostic observations, not substitutes for the official owned-output p50/p95 values or a measurement of observer overhead. Text cases emit two lines per call (20 across the ten sampled calls); blank cases emit zero.

| Family | Case | Total mean ms | DET inference | CLS/orientation | REC backbone | Projection |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| tiny | image-000 | 71.448 | 9.307 | 8.647 | 48.142 | 4.464 |
| tiny | image-001 | 70.003 | 9.332 | 9.517 | 46.431 | 3.853 |
| tiny | image-002 | 0.987 | 0.977 | 0.000 | 0.000 | 0.000 |
| tiny | image-003 | 1.008 | 0.999 | 0.000 | 0.000 | 0.000 |
| small | image-000 | 305.588 | 22.225 | 9.762 | 255.788 | 16.971 |
| small | image-001 | 301.504 | 23.555 | 9.065 | 251.872 | 16.028 |
| small | image-002 | 2.693 | 2.683 | 0.000 | 0.000 | 0.000 |
| small | image-003 | 2.649 | 2.639 | 0.000 | 0.000 | 0.000 |
| medium | image-000 | 1581.425 | 160.094 | 8.819 | 1377.174 | 34.266 |
| medium | image-001 | 1480.965 | 159.048 | 8.994 | 1278.318 | 33.349 |
| medium | image-002 | 18.197 | 18.170 | 0.000 | 0.000 | 0.000 |
| medium | image-003 | 17.661 | 17.650 | 0.000 | 0.000 | 0.000 |

On text pages, REC backbone accounts for 66–87% of observed total time, projection 2–6%; it is the clearest next profiling target. Tiny CLS/orientation is about 12–14%, while medium DET inference is about 10–11%. Blank cases spend about 99% in DET inference. The complete stage vectors include binding, all preprocessing/postprocessing, crop, CTC, result storage and unassigned time. Per-operator evidence should guide the next optimization; these stage totals alone do not identify a particular kernel as the remaining bottleneck. Explicit SIMD and threading remain outside this scalar iteration.

## Memory and candidate retention

The separate frozen untuned/refined Rust selection cohort passed all 36 pairs with p50 and p95 improvements, six compact REC/OCR comparisons at zero tolerance, and 24 heap processes. The retained changes are a four-pixel × four-output Conv1x1 scalar tile, contiguous trailing-axis ReduceMean, and same-rank row-scalar broadcasting. Each Conv output preserves its bias-first ascending-input-channel F32 chain. All saved expanded Windows v2/v3 descriptors and hashes also match: 68,459 artifacts / 47,046,536,714 bytes. This establishes saved-corpus identity, not a claim about all possible inputs.

For the selected `image-000` state, active storage, retained capacity and workspace matched the untuned baseline. Warmed borrowed REC/OCR allocated zero bytes with zero retained/peak heap growth. Owned results allocate their copies and retain zero growth after release.

| Family | Scope | Active bytes | Retained bytes | Workspace bytes |
| --- | --- | ---: | ---: | ---: |
| tiny | REC | 1,327,104 | 1,622,016 | 203,867 |
| tiny | OCR | 2,660,864 | 2,955,776 | 691,712 |
| small | REC | 3,756,800 | 3,756,800 | 407,899 |
| small | OCR | 5,784,320 | 5,784,320 | 895,400 |
| medium | REC | 6,348,800 | 7,946,240 | 430,443 |
| medium | OCR | 10,915,840 | 12,513,280 | 918,758 |

Normalized constant tensor payload bytes (including integer metadata) are tiny REC 4,417,232 / DET+CLS+REC 7,096,348; small 21,066,724 / 31,845,632; medium 76,454,256 / 138,874,476. This static inventory excludes dictionary/object overhead, activation/workspace/result storage and RSS. It is not resident memory. Profiling's final active/storage values follow all four shape blocks and remain separately recorded; they are not the selected-page working state above.

The following ranges are sampled whole-process working-set high-water observations across six processes, in MiB, including validation/loading and each engine's own session/setup components. The process helper polls about once per second; it may miss shorter peaks. This differs from active tensor bytes or allocator-tracked heap growth. An unsampled zero is unavailable, not zero memory.

| Family | C# profile | C# REC RSS | Rust REC RSS | C# OCR RSS | Rust OCR RSS |
| --- | --- | ---: | ---: | ---: | ---: |
| tiny | canonical | 64.41–67.72 | 16.18–16.21 | 79.12–124.70 | 21.59–21.65 |
| tiny | cpu | 60.00–60.67 | 16.18–16.24 | 72.03–78.02 | 21.54–21.71 |
| small | canonical | 112.26–152.56 | 36.51–36.56 | 161.22–171.43 | 50.43–50.44 |
| small | cpu | 103.75–118.07 | 36.54–36.55 | 133.09–161.74 | 50.23–50.47 |
| medium | canonical | 266.12–271.05 | 94.05–94.11 | 527.09–535.96 | 159.91–160.03 |
| medium | cpu | 214.09–220.77 | 94.08–94.15 | 374.14–378.06 | 159.91–159.92 |

## Reproducibility and durable evidence

| Identity | SHA-256 |
| --- | --- |
| Measured production/tool source | `12e7ea543fbadab761eb449c521ae891ad2486f9e8adbf08d1e8d8a4ce82669d` |
| Native source | `a6e40ec506fcde1736cf0e058d33e3a0af1149f30af687dbd43730ba68054966` |
| Prepared build manifest | `37f309f7bc61180733e79b4528f1d3a3a8d04c6104a6357321f69e38a79c4954` |
| Official timing executable | `9ac50d3407e3bb957d15a361aaf44de0b19bdf467c115873cb9340867212b8ec` |
| Prepared heap executable | `ec2f8a9c1ddd76dd9c8e016bd25e7afc10b7b057d3228d25c009f031ba4958f5` |
| Native exporter | `5791650e6f58e04ff93853aa932ae69a9b6a25a057c56d221344e9f228f46fa2` |
| Native C# library | `766d6fc6a83d856a931bc4894ce91fb7b41e07f745fba4ea048e86f22b944cd0` |
| Production profiling executable | `2fd999b89753aecafb9332c22a53322a3281f0b7f07b2b73d899407c2fdc2801` |
| Strict official report | `0b369b544b31178892a8ad29a6ca4bd27c677bb06b325492784fb412c3754ae1` |
| Production profile summary | `61866004459060e84c873207f0a904795f0b6d9d516ffb0b9f9ffc3f86bfc595` |

The immutable official/profile archive is `reference-archives/evidence/all-model-public-latency-profile-20261015-v2.zip` in the workspace sibling directory. Its `.verification.json` and `.sha256` sidecars record archive bytes/SHA and full CRC, per-entry byte-count and SHA readback. The archive includes complete captured source and frozen binaries, all three compact references, 180+6 raw process records/controls/idle observations, prepared metadata, strict JSON and report scripts, every production profile record/ledger/stage summary, and this report. Detailed paired ratios/flags are in `measurements/all-model-official-public-report-v2.json`; model/dictionary/options/input identities are also in `measurements/all-model-official-public-review-v2.json`.

The prior v1 attempt remains explicitly interrupted with 45 timing reports / 53 raw process records and no deletion. Its disclosed 0.3-second PE inspection is preserved as background activity. Post-idle analysis isolated repeated feature-switch relinking differences to COFF/debug timestamps and CodeView GUID metadata; final v2 reused identical prepared bytes instead. v1, disposal/reuse smokes, allocator measurements and timer-observed profiling are excluded from the official latency cohort.

Separate immutable dependencies remain distinct:

| Evidence archive | Bytes | SHA-256 |
| --- | ---: | --- |
| all-model-untuned-baseline-20261015-v1.zip | 1,650,073 | `75d3663c5273ba87fc1bc824449722c11a7ee0a14e577a1a405f3a446912aaac` |
| all-model-scalar-conv-selection-20261015-v1.zip | 5,959,614 | `a3ed23f0f4a88025f492602fd491ab0c712663a0a6afbdad02fc81c53e09fc01` |
| all-model-windows-expanded-final-20261015-v1.zip | 6,888,131 | `4a16ba914b712c4c4db3ca4e5a029cbc133a471c32743f336b2dfe645e427380` |
| all-model-linux-container-expanded-final-20261015-v1.zip | 6,829,726 | `ae269f18d3050dd189c49d82466d83d794ac11c921936d7a6d40a8b6be48a389` |

See [all-model validation](all-model-validation.md) for native/API/cross-platform and hosted CI evidence tied to the same production source. The report and archive are documentation/evidence work after both collection barriers; they introduce no additional timing samples or kernel changes.
