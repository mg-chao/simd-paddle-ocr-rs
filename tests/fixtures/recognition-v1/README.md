# Native tiny REC preprocessing and operator goldens

These small committed bundles were captured from the original C# implementation with hardware intrinsics disabled. They are independent of Rust output and run in ordinary tests. Each reader verifies the complete manifest's declared byte counts and SHA-256 hashes before comparing results; manifests also retain source snapshots and the C# assembly identity.

`manifest.json` contains 56 preprocessing cases: 14 width/mode configurations in BGR24, RGB24, BGRA32 and RGBA32. Each has a configuration JSON, source pixels and normalized NCHW REC input (168 artifacts total). Adaptive width cases cross natural-width boundaries at 1, 31, 32, 33, 48, 62, 63, 64, 65 and 72; fixed-width cases exercise padding and clamping at 32, 64 and 320. Strides include 13 padding bytes, the final row omits trailing padding, alpha is ignored, and varied channel bytes exercise byte-domain interpolation. The Rust test compares all output F32 bits in two traversal orders (112 resize comparisons), reusing a destination filled with a sentinel to check refreshed right-side zero padding. The preprocessing manifest SHA-256 is `0874e5e88a250bbab6ee17a31a9c4a13957b7ed3d0a08cc2675ceb0b7e59b5c0`.

`operators/manifest.json` contains eight independently generated tiny ONNX graphs and captures from the native `InferenceSession`: inference BatchNormalization; 3×2 AveragePool with 3×2 stride and no padding; rank-three MatMul with a shared rank-two right operand; rank-three last-axis Softmax; negative-axis ReduceMean and Squeeze; positive-axis Unsqueeze; and rank-three Transpose. Each case includes fixture JSON, ONNX bytes, input, node trace and output (40 artifacts total). Shapes and trace coverage are exact. MatMul, Squeeze, Unsqueeze and Transpose values compare bit for bit; other values use `atol=1e-5`, `rtol=1e-4`. This proves the shipped operator subset. C# negative-axis Unsqueeze has a legacy clamping interpretation and is outside this native golden scope. The operator manifest SHA-256 is `b2ec95ceceead0a2daf3c4eb35ace1bbca3a2f470e4b54601d5fb6aacf88f211`.

Run the readers without downloaded models:

```powershell
cargo test --locked --test recognizer independently_captured_native_rec_preprocessing_is_bit_exact
cargo test --locked --test rec_operator_goldens tiny_rec_subset_matches_independent_native_nodes_and_outputs
```

Capture new goldens into new directories from the Rust crate root, with the original reference exporter already built:

```powershell
$env:DOTNET_EnableHWIntrinsic = '0'
$env:DOTNET_PROCESSOR_COUNT = '1'
$env:PPOCR_NHWC = '0'
$env:SIMD_OCR_NOFUSE = '1'
dotnet tools/reference-export/bin/Release/net10.0/SimdPaddleOCR.ReferenceExport.dll --export-rec-fixtures --out target/new-rec-goldens --reference-repo ../SimdPaddleOCR --exporter-source tools/reference-export
dotnet tools/reference-export/bin/Release/net10.0/SimdPaddleOCR.ReferenceExport.dll --export-rec-ops --out target/new-rec-operator-goldens
```

Capture refuses an existing destination. Normal tests and CI consume the committed oracle; they do not regenerate it. Both manifests pin the native assembly SHA-256 `766d6fc6a83d856a931bc4894ce91fb7b41e07f745fba4ea048e86f22b944cd0` from .NET 10.0.11. The source and captured outputs derive from the Apache-2.0 SimdPaddleOCR reference. Larger frozen REC and complete OCR selections are documented in [step12](../step12/README.md); milestone coverage and its limits are recorded in recognition parity (local `docs/rec-parity.json`) and tiny OCR parity (local `docs/tiny-ocr-parity.json`).

The exporter also has a model-free decode-contract selftest: `dotnet tools/reference-export/bin/Release/net10.0/SimdPaddleOCR.ReferenceExport.dll --verify-rec-contracts`. Its 46 form checks accept different dense/public decodes, emitted empty dictionary tokens, uncalibrated public scores and positive fixed widths, while rejecting malformed dimensions/counts/spans and nonfinite scores. The separate harness hardening proof (local `docs/rec-harness-hardening.json`) records its native before/after checks and the final Windows/Linux revalidation. This command does not regenerate a fixture and is separate from the Rust/Python CI gates.
