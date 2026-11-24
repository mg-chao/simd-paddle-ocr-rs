# RapidOCR parity

[Resident Vulkan GPU results and reproduction](GPU_RESULTS.md) · [CPU operator-level results and release timings](OPERATOR_RESULTS.md) · [First-pass results and residual cases](RESULTS.md)

Run the commands from the repository root in PowerShell. The capture uses the default native CPU kernels and the public Rust builder through the `ocr` example. Python imports the checkout at `E:\RapidOCR\python`. Both read weights from `..\SimdPaddleOCR\models`. The Python environment used here contains Python 3.12.10, ONNX Runtime 1.24.4, OpenCV 4.12.0, NumPy 2.2.6, Shapely 2.1.2 and Pillow 12.3.0.

| Preset | Detection | Recognition | Dictionary |
| --- | --- | --- | --- |
| tiny | det.onnx | rec.onnx | ppocr_keys.txt |
| small | small_det.onnx | small_rec.onnx | rec_keys.txt |
| medium | medium_det.onnx | medium_rec.onnx | rec_keys.txt |

All presets load `cls.onnx`. Its static input shape is 3 × 80 × 160, which RapidOCR selects through its PP-OCRv5 classifier configuration. The scripts verify the classifier input shape, the complete effective recognition vocabulary (including the CTC blank and trailing space), CPU providers, image hashes and executable hashes. Each reference directory contains its full effective YAML, SHA-256 for each weight and dictionary, Python package versions, Git revision, and the resolved classifier shape.

## Collect

```powershell
$root = (Get-Location).Path
$out = Join-Path $root 'reports/rapidocr-parity'
$presets = @('tiny','small','medium')
python tools/rapidocr_parity/collect.py prepare --images "$env:USERPROFILE/Desktop/test-ocr-imgs"

# Use the saved reference directory to reproduce the comparison immediately.
# To collect it from scratch, use the recorded checkout and an empty output directory.
foreach ($p in $presets) {
    python tools/rapidocr_parity/collect.py reference $p --indices 2 3 4
    python tools/rapidocr_parity/collect.py reference $p
}
```

The prepare step records relative paths and original file hashes. It creates one lossless PNG for each WebP and records the converted input hash. Each successful entry contains a list of lines, including an empty list when appropriate. Each failed invocation records its exception. A capture resumes completed files only when its binary or reference configuration matches.

```powershell
# Rebuild the original release in an independent checkout if the saved binary is unavailable.
$old = Join-Path (Split-Path $root) 'rapidocr-parity-baseline'
git worktree add --detach $old 60217e7f68a90b98d2d6714a78a1d12665f80af5
cargo build --manifest-path "$old/Cargo.toml" --locked --release --example ocr
$baseline = "$old/target/release/examples/ocr.exe"

cargo build --locked --release --example ocr --example rapidocr_parity_latency
$final = "$root/target/release/examples/ocr.exe"
foreach ($p in $presets) {
    python tools/rapidocr_parity/collect.py rerun-before $p --exe $baseline
    python tools/rapidocr_parity/collect.py rerun-after $p --exe $final
}
python tools/rapidocr_parity/report.py rerun-before rerun-after
```

The archived acceptance can be recomputed without inference. Extract `reports/rapidocr-parity/reference-frozen.zip` and `reports/rapidocr-parity/acceptance.zip` into `$out`, then run `python tools/rapidocr_parity/report.py baseline validated`. To inspect one image, use its zero-based manifest index with the probe. Trace and same-tensor replay outputs are diagnostic.

```powershell
$c = Get-Content "$out/inputs.json" -Raw | ConvertFrom-Json
$i = 6
$p = 'medium'
cargo build --locked --release --example rapidocr_parity_probe
& "$root/target/release/examples/rapidocr_parity_probe.exe" $p $c[$i].input "$out/replay-$p-$i"
python tools/rapidocr_parity/diagnose.py $p $i "$out/replay-$p-$i"
python tools/rapidocr_parity/diagnose.py $p $i "$out/replay-$p-$i" --downstream-only

# Vary just the detector, holding the Python preprocessing and downstream fixed.
python tools/rapidocr_parity/diagnose.py small 76 "$out/replay-map-small-76" --map-only

# Compare a saved recognition tensor from a full trace on both optimized executors.
& "$root/target/release/examples/rapidocr_parity_probe.exe" medium $c[59].input "$out/replay-medium-59"
python tools/rapidocr_parity/diagnose.py medium 59 "$out/replay-medium-59" --rec-line 2
```

## Measurements

Exact match compares the complete ordered arrays of line text. CER is summed Unicode code-point Levenshtein distance divided by total reference characters, with a literal newline between consecutive lines. Case, whitespace and punctuation are significant. Geometry uses a deterministic greedy one-to-one match in descending convex-quadrilateral IoU, with an inclusive 0.5 cutoff and index tie-breakers. Missing and extra lines are unmatched reference and Rust lines respectively. Reading order counts inversions among matched line indices. Box displacement is the mean Euclidean distance of corresponding corners in source pixels. Confidence MAE uses matched line recognition scores. Errors and successful empty outputs are counted separately.

The release latency harness times one public request and its owned-result conversion per preset on manifest image 2, with decoded pixels, initialized model requests and eight CPU threads. Only tiny receives a warm-up call, for one warm-up total. Model construction and image decoding are outside the measured region. Use the identical harness source in the baseline checkout for a fresh comparison. Run with the repository root as the working directory so the model path resolves correctly.

```powershell
$c = Get-Content "$out/inputs.json" -Raw | ConvertFrom-Json
Copy-Item "$root/examples/rapidocr_parity_latency.rs" "$old/examples/rapidocr_parity_latency.rs"
cargo build --manifest-path "$old/Cargo.toml" --locked --release --example rapidocr_parity_latency
& "$old/target/release/examples/rapidocr_parity_latency.exe" $c[2].input
& "$root/target/release/examples/rapidocr_parity_latency.exe" $c[2].input
```

The separate instrumented small-preset capture uses one request and no warm-up. Build the same example with `--features profiling` and pass `--profile-small` following the image argument. These stage wall times use a different executable and are recorded separately from the three uninstrumented measurements.

The Rust crop and rectangle implementations follow [OpenCV 4.12.0](https://github.com/opencv/opencv/tree/4.12.0/modules/imgproc/src). Equal aspect ratios at a batch boundary follow the float64 index permutation observed in this reference build. NumPy selects its sorting implementation at runtime. [NumPy 2.2.6 sort dispatch](https://github.com/numpy/numpy/blob/v2.2.6/numpy/_core/src/npysort/x86_simd_argsort.dispatch.cpp)

## Operator replay

The follow-up capture is named `packet`. The intermediate `mean` capture has the same arithmetic with a scalar fallback for misaligned planes. Use the following command to recompute per-image geometry, scores, CER, and regressions.

```powershell
python tools/rapidocr_parity/report.py validated packet
```

The operator probe takes one ONNX graph and a raw little-endian F32 input. The Python driver also requires `onnx` (1.23.2 on this workstation). For example, build and trace the same medium line as in the report, then run the isolated spatial reduction, full graph and first-convolution swap.

```powershell
cargo build --locked --release --example rapidocr_tensor_probe --example rapidocr_parity_probe
$c = Get-Content "$out/inputs.json" -Raw | ConvertFrom-Json
& "$root/target/release/examples/rapidocr_parity_probe.exe" medium $c[59].input "$out/replay-medium-59"
python tools/rapidocr_parity/operators.py medium "$out/replay-medium-59" "$out/replay-operators-medium" --rec-line 2
```

For a detector, use its trace directory and omit `--rec-line`. The small and tiny operator examples in the report use manifest indices 76 and 106. Add `--before "$out/layers/before-tensor.exe"` to compare the saved pre-fix executor on the same inputs. The tool writes the exact model boundaries, arithmetic comparisons and final-output errors to `operators.json`. Raw tensors and extracted ONNX files stay in its diagnostic directory.

For the focused CPU kernel measurement, extract `operator-acceptance.zip` into `$out` and use the archived real odd-sized recognition activation. The harness converts to NHWC before timing and measures one kernel call. Run it from the repository root.

```powershell
$l = Join-Path $out 'layers'
rustc --edition=2024 -C opt-level=3 -C target-feature=+avx2,+fma "$l/mean_kernel_bench.rs" -o "$l/replay-mean-kernel.exe"
& "$l/replay-mean-kernel.exe" "$l/rec-odd-input.f32" 768 before
& "$l/replay-mean-kernel.exe" "$l/rec-odd-input.f32" 768 after
```
