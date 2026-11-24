# RapidOCR parity on Apple M4

The release archived in `parity-m4/matrix` matched ordered raw text on **768 of 768** image/preset/backend comparisons. Across 34,652 matched line pairs, 0 quadrilaterals differed in any source coordinate. These corpus results and timings predate the register-shuffle Metal spatial mean, detector resize tables and SME interior packing changes.

## Optimization follow-up

The three changes were built successively on the accuracy-fixed release. The first implementation passed three focused image checks. A full-corpus check then caught a missing line on medium image 34. The reference score for the line was 0.50001 against a publication threshold of 0.5. The regression appeared with the Metal spatial mean. The final shader preserves the previous left-to-right horizontal fold for ordinary reductions that fit in one 1,024-element staging tile. Longer reductions use register shuffles, and GlobalAveragePool keeps its own pairwise order. The line returned with a score of 0.50000 and its original coordinates. The OCR publication threshold stayed at 0.5.

The final release passed **768 of 768** public OCR comparisons on the 128 images in `/Users/chao/Desktop/test-ocr-imgs`. All 34,652 line comparisons had identical text, line order and all four source-image corners. Every confidence was within the requested absolute tolerance of 0.0001. The input hashes, frozen public Python results and effective configurations were rechecked before inference.

| Preset | CPU and Metal passing images, each | Lines per backend | Maximum CPU score error | Maximum Metal score error |
|---|---:|---:|---:|---:|
| Tiny | 128 / 128 | 5,501 | 0.00004998 | 0.00004998 |
| Small | 128 / 128 | 5,292 | 0.00004999 | 0.00006998 |
| Medium | 128 / 128 | 6,533 | 0.00006997 | 0.00008998 |

The final release library suite passed 435 tests, with 78 opt-in tests skipped by default. Three opt-in Metal tests also passed on the M4. They cover short-plane reduction ordering, long-plane and pooling order, and resident pixel preprocessing with guarded outputs. The SME packing regression passed in the library suite.

The local evidence directory `parity-m4/optimization-builds.0Asi8K/final` contains the six captures, release executable hashes, a source hash manifest, raw timings and `verified-corpus.json`. Running `verify_corpus.py final` from its parent rechecks each ordered line against the frozen Python result.

### Kernel and stage checks

Each timing below used one captured input, one warm-up and two measured calls per version on the physical Apple M4. Values are individual measurements in milliseconds.

| Boundary | Before | Optimized | Correctness check |
|---|---:|---:|---|
| Metal ReduceMean on image 108, including input upload and output readback | 8.30, 8.47 | 6.21, 7.28 | Final CPU and Metal outputs both match all 32 Python FP32 values bit for bit |
| Resident Metal detector resize on image 3, from host encoding through GPU completion | 2.08, 1.93 | 1.39, 1.39 | Both paths match all 4,310,016 FP32 detector input values from Python |
| CPU detector prefix on image 108, including the first four convolutions | 39.46, 38.97 | 37.87, 34.96 | SME interior and boundary packing checked against the original packing order |

The resize timer includes allocation of the per-axis coefficient buffer and a four-byte readback to wait for completion. The complete output was compared in the warm-up. The resize and CPU prefix comparisons used consecutive builds differing only in the optimization under test.

### End-to-end latency

The comparisons used eight CPU threads and the same release harness, models and decoded image for each version. Each table cell contains two measured calls in milliseconds after one warm-up. The earlier release is commit `4c45ae3`. The accuracy-fixed column uses the uncommitted version before the three optimizations.

| Image and backend | Earlier release | Accuracy-fixed | Optimized |
|---|---:|---:|---:|
| Tiny image 3, CPU | 149.91, 152.10 | 133.10, 132.66 | 136.40, 142.57 |
| Tiny image 3, Metal | 126.24, 116.37 | 111.36, 107.66 | 113.58, 105.66 |
| Medium image 57, CPU | 3435.42, 3105.77 | 3123.95, 3167.76 | 3138.88, 3276.86 |
| Medium image 57, Metal | 2581.53, 2562.53 | 2654.32, 2676.53 | 2683.64, 2594.32 |

Against the earlier release, mean latency changed by −7.6% and −9.6% on tiny CPU and Metal, and by −1.9% and +2.6% on medium CPU and Metal. Against the accuracy-fixed build, the changes were +5.0%, +0.1%, +2.0% and −1.0%, respectively. The medium Metal run returned 223 lines in the earlier release and 222 in both accuracy-fixed and optimized builds. The latency comparison covers these two pages and varies between the two measured calls.

## Corpus and execution

The reference contains 128 images for each of the Chinese V6 tiny, small and medium presets. The starting Rust release was `4c45ae3ba642b015e0f42fe87dd952a42185ce4e`. Its source matches the frozen local checkout named `after`. All comparisons used the same physical Apple M4 Mac mini, macOS 27.2.1 (26B131), arm64, Rust 1.100.0-nightly (cea272fa3), and eight CPU threads. Rust used release builds with default parallelism and the GPU feature. This validation covers native CPU FP32 and Metal FP32. Vulkan, x86 and mixed precision have no corpus numbers in this report.

The input manifest covers every image in the directory, including 76 PNGs, 49 JPEGs and three WebPs. Both engines received the same PNG conversions for the WebPs. An audit compared decoded mode, dimensions and every pixel against the originals. The manifest SHA-256 is `069a7e1b9923c506a1f378127ccae547368bd0afbe68a0bd910da019bf3cafb5`.

The Python reference came from `RapidOCR(params=...)(input_path)` at commit `28490093dfb75dc4cd0f5d4f1d2259aeb44f5351`, using Python 3.12.16, ONNX Runtime 1.24.4 (2d924974ef), NumPy 2.2.6, OpenCV 4.12.0.88 and Pillow 12.3.0. Recreating each configuration verified the actual CPUExecutionProvider for all three sessions, eight intra-op threads, one inter-op thread, the complete effective configuration, the model hashes and every dictionary entry including the appended space. The classifier input was 3 × 80 × 160. Detector threshold was 0.3, box threshold 0.5 and unclip ratio 1.6, with dilation enabled. Classification and recognition used batches of six. Minimum published text confidence was 0.5.

Each Rust capture called `OcrModelSet::from_directory(...).builder()`, selected the requested execution target and ran `request.run(image, RunOptions::default()).try_to_owned()`. Every CPU row reported CPU execution. Every Metal row reported resident execution on Apple M4. Per-image placement records identify the host perspective-coordinate calculation where used and the device stages for detector geometry, projection and CTC decoding. Input decoding and output materialization use the same boundaries for baseline and candidate.

## Corpus results

An exact image has the same ordered array of raw line strings, including case, whitespace and punctuation. CER is Unicode Levenshtein distance on newline-joined strings, divided by the total reference character count. The unchanged comparator greedily matches convex quadrilaterals at IoU ≥ 0.5. Misses, extras and order inversions refer to those matches. Corner displacement averages the Euclidean distance between corresponding corners in source pixels. Confidence errors are absolute differences between public line scores.

All six comparisons cover 128/128 images before and after. Both versions have zero missing captures and zero execution errors.

| Preset | Backend | Exact images | Character edits | CER | Missing / extra lines | Order inversions (affected images) |
|---|---|---:|---:|---:|---:|---:|
| tiny | CPU | 113 → 128 | 40 → 0 | 0.02973% → 0.00000% | 0 / 0 → 0 / 0 | 0 (0) → 0 (0) |
| small | CPU | 125 → 128 | 6 → 0 | 0.00436% → 0.00000% | 0 / 0 → 0 / 0 | 0 (0) → 0 (0) |
| medium | CPU | 117 → 128 | 153 → 0 | 0.09324% → 0.00000% | 0 / 0 → 0 / 0 | 0 (0) → 0 (0) |
| tiny | Metal | 30 → 128 | 10637 → 0 | 7.90655% → 0.00000% | 34 / 46 → 0 / 0 | 1,494 (43) → 0 (0) |
| small | Metal | 35 → 128 | 10210 → 0 | 7.42454% → 0.00000% | 20 / 40 → 0 / 0 | 1,621 (42) → 0 (0) |
| medium | Metal | 28 → 128 | 12663 → 0 | 7.71706% → 0.00000% | 29 / 31 → 0 / 0 | 2,569 (47) → 0 (0) |

| Preset | Backend | Mean quad IoU | Mean corner displacement (px) | Mean absolute confidence error | Max absolute confidence error |
|---|---|---:|---:|---:|---:|
| tiny | CPU | 0.9999574 → 1 | 0.0009811005 → 0 | 0.0001464361 → 6.737343e-07 | 0.14212 → 4.997698e-05 |
| small | CPU | 0.999994 → 1 | 9.448224e-05 → 0 | 6.111791e-05 → 4.150951e-07 | 0.0204 → 4.999086e-05 |
| medium | CPU | 0.9999965 → 1 | 0.000153069 → 0 | 0.0001986483 → 4.227154e-07 | 0.10238 → 6.997372e-05 |
| tiny | Metal | 0.9811335 → 1 | 0.4409029 → 0 | 0.009684092 → 8.698933e-07 | 0.40602 → 4.997698e-05 |
| small | Metal | 0.981962 → 1 | 0.4323927 → 0 | 0.005267724 → 3.158114e-06 | 0.38082 → 6.998114e-05 |
| medium | Metal | 0.9833786 → 1 | 0.3697534 → 0 | 0.004724954 → 5.431559e-06 | 0.38222 → 8.997946e-05 |

| Preset | Reference lines | Reference characters | Reference empty | CPU empty, before → after | Metal empty, before → after |
|---|---:|---:|---:|---:|---:|
| tiny | 5,501 | 134,534 | 11 | 11 → 11 | 11 → 11 |
| small | 5,292 | 137,517 | 10 | 10 → 10 | 10 → 10 |
| medium | 6,533 | 164,091 | 0 | 0 → 0 | 0 → 0 |

Every image/preset/backend combination that agreed with Python at baseline still agreed in the accuracy-fixed `matrix` build.

## How the discrepancies were isolated

- On tiny image 3, the baseline Metal detector and Python detector each produced 66 identical quads when Python processed their maps. Passing the same frozen Python map through baseline Metal geometry reproduced only 33 of 66. The Metal implementation now quantizes the complete resize coordinate, enumerates outer contours and holes in reverse raster order, and applies the native hull, rotating calipers, mask scoring, polygon expansion and reading-order rules. Replaying the frozen map returned all 66 quads in order.
- On tiny image 39, the source pixels, preprocessed tensor and threshold mask agreed. Three coordinates still differed by up to two pixels. On image 77, the quads agreed, but a fixed perspective crop differed in 1,543 channel values. Replaying the crop and recognizer tensor isolated the coordinate arithmetic. ARM calipers, quad corners, the perspective solve and pixel interpolation now use the multiply-add order of the local OpenCV build.
- On tiny image 3, two crops with equal aspect ratios straddled a recognition batch boundary. Reproducing NumPy's AArch64 float64 argsort permutation restored the reference canvas widths. The ARM implementation includes NumPy's license and tied-ratio regression coverage.
- On tiny CPU images 108 and 112, Python postprocessing of the native map reproduced the differing boxes. Isolated ARM ReduceMean and GlobalAveragePool inputs identified distinct four-float accumulation orders. The native implementation reproduced all 32 and 64 reference values on the two sampled layers. On image 108 this reduced maximum detector-map error from 0.001258 to 0.000264. Substituting just the first RGB convolution still left a differing box on both images. Substituting the two early 3×3 spatial convolutions restored every box on both. On fixed activations, tap-first, channel-second fused multiply-adds reproduced the ORT 3×3 results exactly. The native NEON and SME executors now consume a matching packed panel.
- On medium Metal image 57, three map values crossed the 0.3 threshold. Native and Python geometry agreed on each individual map, but one of 233 boxes shifted by 1.39 source pixels and its confidence changed by 0.02484. Preserving the ARM spatial reduction order on Metal reduced maximum map error from 0.000110 to 0.0000121 in isolation. Applying the ARM 3×3 traversal and bias seed to Metal matrix fragments as well restored tiny image 108, where changing the reductions alone left one raw character different. The final maps for medium image 57 and tiny image 108 had zero threshold crossings relative to Python, maximum errors of 0.0000144 and 0.0000272, and all 233 and 62 quads identical, respectively.

The index values above are zero-based rows of the frozen manifest. Model files, images, tensor dumps and per-image raw OCR text stay in the local evidence directory.

## Residual confidence differences

The table identifies the largest public confidence difference in each capture by zero-based image and output line. Per-image details retain every matched score delta.

| Preset | Backend | Image | Line | Maximum absolute error |
|---|---|---:|---:|---:|
| tiny | CPU | 63 | 94 | 0.00004998 |
| small | CPU | 42 | 5 | 0.00004999 |
| medium | CPU | 95 | 80 | 0.00006997 |
| tiny | Metal | 63 | 94 | 0.00004998 |
| small | Metal | 69 | 33 | 0.00006998 |
| medium | Metal | 95 | 107 | 0.00008998 |

Casting the Python public score to FP32 accounts for 29,595 unequal serialized scores. A further 4,474 line pairs have different FP32 scores. Both APIs round CTC confidence to five decimal places. Different native FP32 posterior values can cross one of those boundaries.

To isolate inference, four actual RapidOCR recognition batches were replayed unchanged through both Rust backends, using tiny image 119 and small and medium image 105. All eight replays retained every CTC time-step winner. Maximum posterior differences ranged from 0.00000268 to 0.0000758. Reapplying the Python score calculation to those posteriors gave rounded line-score differences up to 0.00004. The native CPU and Metal matrix, activation and softmax kernels have their own FP32 arithmetic. A bit-exact score match would require reproducing more of ORT’s shape- and hardware-specific arithmetic. These sampled controls isolate inference as a cause of confidence drift.

## Local performance of the accuracy-fixed build

Times are milliseconds. Each stage used one captured input from tiny and one release configuration per version. Detector-prefix timing covers the first four convolutions and their intervening operations, from session entry through the returned tensor. The Metal reduction includes the same input upload and output readback in both executables. DB timing covers postprocessing a frozen probability map through compact box readback. The end-to-end boundary starts after decoding and model loading and ends after copying the public OCR result.

| Boundary and input | Warm-ups per version | Starting release, individual runs | Accuracy-fixed build (`matrix`), individual runs |
|---|---:|---:|---:|
| CPU detector prefix, image 108, 1984 × 1568 tensor | 1 | 37.36, 36.49 | 36.31, 38.88 |
| Metal detector prefix, same tensor | 1 | 30.05, 32.44 | 32.40, 30.32 |
| CPU GlobalAveragePool, 64 channels × 3,038 pixels | 0 | 0.1170, 0.1200, 0.0912 | 0.0380, 0.0333, 0.0332 |
| Metal ReduceMean, 32 channels × 194,432 pixels | 1 | 6.24, 6.51 | 9.20, 8.63 |
| Metal DB, image 3, 736 × 1952 map, 66 boxes | 1 | 11.44, 9.99 | 8.05, 7.61 |
| CPU end to end, tiny image 3 | 1 | 137.34 | 139.27, 140.64 |
| Metal end to end, same image | 1 | 100.65 | 114.00, 107.97 |

The Metal spatial mean retains the reference accumulation order. Cooperative loading reduced its two warm timings from 11.45 and 10.71 ms in the first ordered implementation to 9.20 and 8.63 ms in the `matrix` build. The baseline parallel tree completes that stage faster. On the CPU detector prefix, moving traversal selection outside the NEON accumulation loop reduced the first candidate’s 39.16 and 41.71 ms to 36.31 and 38.88 ms. The `matrix` end-to-end spot timings were 1.4–2.4% above baseline on CPU and 7.3–13.3% above baseline on Metal. These end-to-end comparisons have one baseline observation per backend on one image. They give the magnitude seen here and leave corpus-wide latency uncertain.

Timed processes in the `matrix` comparison took 2.62 seconds in total. Reused baseline timings and the added baseline stage checks also stayed below the 60-second budget. Full-corpus captures measured correctness separately.

## Correctness checks

The accuracy-fixed `matrix` build passed 434 GPU-feature library tests and skipped 77 opt-in tests. A separate physical-device run passed 23 of the opt-in checks, covering 18 resident OCR cases, spatial pooling, geometry boundary and tie handling, convolution, tiled reductions and fused convolution epilogues. The remaining 54 opt-in checks were outside this local run. The full-corpus captures used the same build through the public OCR API. The tensor controls exercised the underlying native executors directly.

## Local reproduction

Run from the Rust repository root on the physical Mac. The commands below use the retained checkout, models, inputs and original capture drivers. The external evidence directory keeps all raw OCR and tensors off Git.

```sh
D="$(cd ../.local/rapidocr-macos-2026-11-24.qe9dHD && pwd)"
E="$D/parity-m4"
PY="$D/venv/bin/python"
export PYTHONPATH="$PWD/tools/rapidocr_parity"
"$PY" "$E/audit.py" > "$E/rechecked-audit.json"
cmp "$E/audit.json" "$E/rechecked-audit.json"

cp "$D/after/examples/macos_parity.rs" examples/
cargo build --locked --release --features gpu --example macos_parity \
  --example rapidocr_parity_probe --example rapidocr_tensor_probe
for preset in tiny small medium; do
  for backend in cpu gpu; do
    "$PY" "$E/capture.py" "$preset" "$backend" reproduction
  done
done
"$PY" "$E/metrics.py" reproduction
```

The capture driver binds each JSONL to its binary SHA-256, verifies input hashes before use and resumes missing rows. Use a fresh output name for another binary. For a focused reproduction, append manifest indices, for example `tiny cpu focused-reproduction 108 112`. For the independent Metal detector replay, run the following.

```sh
"$PY" "$E/replay.py" gpu medium 57 "$E/reproduced-medium-57"
"$PY" "$E/scores-replay.py" reproduced-score-tensors
cargo test --locked --release --features gpu --lib -- --test-threads=1
cargo test --locked --release --features gpu --lib gpu_resident -- --ignored --test-threads=1
for test in gpu_spatial_mean_keeps_packet_alignment_and_pool_order \
  gpu_native_trig_geometry_metadata_ties_and_boundary_guards \
  gpu_hardware_convolution_matches_scalar \
  gpu_hardware_specialized_tiles_tails_and_long_reductions \
  gpu_metal_common_convolution_epilogues_match_scalar; do
  cargo test --locked --release --features gpu --lib "$test" -- --ignored --test-threads=1
done
```

For the optimization follow-up, the following replay checks all 32 spatial means from the saved RapidOCR detector activation on tiny image 108. It uses the release probe built above.

```sh
replay_dir=$(mktemp -d)
probe=target/release/examples/rapidocr_tensor_probe
"$probe" "$E/mean-stage.onnx" "$E/mean-input.f32" "$replay_dir/cpu.f32" 1x32x496x392
"$probe" "$E/mean-stage.onnx" "$E/mean-input.f32" "$replay_dir/gpu.f32" 1x32x496x392 --gpu
"$PY" - "$E/cpu-det-108/ReduceMean.npz" "$replay_dir" <<'PYTHON'
from pathlib import Path
import sys
import numpy as np
with np.load(sys.argv[1]) as capture:
    expected = capture["expected"].astype("<f4").tobytes()
for backend in ("cpu", "gpu"):
    assert (Path(sys.argv[2]) / f"{backend}.f32").read_bytes() == expected, backend
PYTHON
```

For a focused OCR comparison, tiny images 3 and 108 cover the measured resize and reduction inputs. Medium image 57 covers a threshold-sensitive box. Compare ordered text, every corner and confidence deltas to the frozen Python outputs on both backends. Compare warm tiny-image-3 latency with the same timing boundary and eight threads used in the local performance table.

The frozen references reside in `$D/reference/{tiny,small,medium}`, beside their complete effective configurations. To regenerate them through public Python inference in a fresh directory, keep the original manifest and RapidOCR checkout.

```sh
"$PY" - "$D" <<'PYTHON'
from pathlib import Path
import shutil, sys, tempfile, collect
root = Path(sys.argv[1])
collect.RAPID = root / "RapidOCR/python"
collect.OUT = Path(tempfile.mkdtemp(prefix="reference-replay-", dir=root))
shutil.copyfile(root / "inputs.json", collect.OUT / "inputs.json")
for preset in collect.FILES:
    collect.capture("reference", preset, None, None)
print(collect.OUT)
PYTHON
```

The baseline captures are `$D/accuracy-{preset}-{cpu,gpu}.jsonl`. The baseline and accuracy-fixed per-image metrics are in `$E/start` and `$E/matrix`. The latter also contains the frozen accuracy-fixed executable, one input-checked capture per backend and preset, and `metrics-*.json` with full-precision aggregate values. `$E/audit.json` lists the original model-configuration, input-manifest, baseline-executable, capture and per-reference hashes. `$E/delivery.json` identifies the `matrix` build and its six output hashes. `$E/scores-replay.py`, `$E/three-cut.py` and `$E/layer-order.py` reproduce the numerical controls. The retained diagnostic subgraphs and input dumps supply the inputs to these controls. The ONNX subgraph diagnostics additionally use `$E/diagnostic-packages` on `PYTHONPATH`.

For the same timing boundaries, restore the three local example sources and build a release. Use the saved timing logs for historical comparisons.

```sh
cp "$D/after/examples/macos_perf.rs" examples/
cp "$E/macos_db_perf.rs" "$E/macos_stage_perf.rs" examples/
cargo build --locked --release --features gpu --example macos_perf \
  --example macos_db_perf --example macos_stage_perf
cp src/runtime/spatial_mean.rs "$E/spatial_mean.rs"
rustc --edition=2024 -C opt-level=3 "$E/pool-bench.rs" -o "$E/pool-bench"
"$PY" "$E/perf-candidate.py" reproduction
target/release/examples/macos_stage_perf "$E/early-spatial.onnx" \
  "$E/cpu-det-108/det-input.f32" unused 1x3x1984x1568 --gpu
```
