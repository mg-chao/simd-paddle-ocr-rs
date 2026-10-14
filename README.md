# simd-paddle-ocr-rs

The Rust rewrite includes checked tensors, streaming ONNX import, graph normalization, and a reproducible reference harness. All seven shipped models have exact raw/normalized import parity with the C# reference, including folded and merged weights. Rust executes tiny, small, and medium DET plus the shared CLS network, and accepts interleaved pixels to produce ordered, oriented BGR text crops. Execution uses serial F32/NCHW kernels, reusable workspaces, and synchronous tracing. Recognition preprocessing and REC execution are the next pipeline stages.

The library and Rust commands require the pinned Rust 1.97.1 toolchain. Reference export and paired C# benchmarks additionally require the .NET 10 SDK, PowerShell 7, and Git. Keep the two repositories adjacent, or supply `-ReferenceRepo` to the exporter. The C# checkout includes this workspace's opt-in trace and model-inspection entry points.

## Detection and oriented crops

```powershell
./tools/export-reference.ps1 -CaptureScope step12 -DetectorModel tiny -Nodes all -MaxTracedLines 0 -Out reference-bundles/tiny-step12-new
cargo run --release --locked --offline -- run-det reference-bundles/tiny-step12-new --out reference-bundles/rust-det-new
cargo run --release --locked --offline -- compare-det reference-bundles/tiny-step12-new reference-bundles/rust-det-new --json
cargo run --release --locked --offline -- run-step12 reference-bundles/tiny-step12-new --out reference-bundles/rust-step12-new
cargo run --release --locked --offline -- compare-step12 reference-bundles/tiny-step12-new reference-bundles/rust-step12-new --json
```

Choose `small` or `medium` for the other detector variants. `run-det` starts
from frozen normalized tensors; `run-step12` independently computes all stages
from frozen pixels. Both commands report explicit partial OCR coverage.
[The image pipeline API and contracts](docs/step12-inference.md) describe the
borrowed results, four pixel formats, reusable memory, and public CLS behavior.
[Windows/Linux parity evidence](docs/step12-parity.json) and
[the performance protocol and results](docs/step12-inference-performance.md)
record this milestone separately from historical import and CLS measurements.

Compact real-model goldens are committed. Prepare them from hash-pinned models
or the adjacent checkout, then run the same explicit gate used in CI:

```powershell
python tools/prepare-step12-fixtures.py --models ../SimdPaddleOCR/models
cargo test --release --locked --offline --test step12_harness compact_real_models_match_all_detector_and_pixel_stages -- --ignored
```

The preparation script uses pinned public model downloads when `--models` is
omitted. It verifies model and archive hashes and never regenerates the oracle.

## Scalar CLS inference

```powershell
# Preserve every node and every detected line in a fresh canonical reference.
./tools/export-reference.ps1 -ReplayBundle reference-bundles/tiny-reference-v1 -Profile canonical -Nodes all -MaxTracedLines 0 -Out reference-bundles/tiny-canonical-allnodes-v1
cargo run --release --offline -- run-cls reference-bundles/tiny-canonical-allnodes-v1 --out reference-bundles/rust-cls-v1
cargo run --release --offline -- compare-cls reference-bundles/tiny-canonical-allnodes-v1 reference-bundles/rust-cls-v1 --json
```

`run-cls` consumes frozen normalized input tensors, not images. It writes only
CLS inputs, reference-selected node tensors, and independently computed
probabilities. `compare-cls` checks all those stages in one traversal and reports
explicitly partial coverage; DET, REC, preprocessing, and public OCR results are
not implemented by this command. All declared files on both sides are still
validated. Existing output directories are never overwritten.

[CLS inference](docs/cls-inference.md) documents the library API, supported
operators, shape evaluation, borrowed trace/output lifetimes, parity gates,
and reproducible allocation and latency measurements.

## Import references and local archives

```powershell
./tools/export-model-contracts.ps1 -Out reference-bundles/import-reference-v1
cargo run --release --offline -- validate-import reference-bundles/import-reference-v1
cargo run --release --offline -- compare-import reference-bundles/import-reference-v1 medium_rec
cargo run --release --offline -- inspect-model ../SimdPaddleOCR/models/cls.onnx
./tools/archive-reference.ps1 -Bundle reference-bundles/import-reference-v1 -BaselineId import-reference-v1
```

The archive tool preserves verified ZIPs and SHA-256 sidecars in the adjacent
`reference-archives` directory, extracts each archive into a fresh verification
directory, and records identities in [the baseline catalog](docs/baselines.json).
ZIP entries have deterministic order and timestamps. Bundles and archives are
never overwritten. Keep those archives when cleaning build outputs.

[Model import](docs/model-import.md) documents ownership, supported types,
normalization, inspection, and the all-model parity gate.
[Iteration measurements](docs/step12-performance.md) records local latency and
memory evidence with reproducible commands.

## Validate or compare one implemented stage

```powershell
cargo run --release --offline -- validate reference-bundles/tiny-reference-v1 --json
cargo run --release --offline -- compare-stage reference-bundles/tiny-reference-v1 reference-bundles/tiny-replay-v1 --case sample --artifact det.input --atol 0 --rtol 0
```

Stage comparison explicitly reports partial coverage. A candidate can contain
only its implemented stages, while every declared model, artifact, and indexed
auxiliary file is still validated. The typed Rust bundle writer supports staged
candidate generation without copying unimplemented reference outputs.

## Export a reference

Run from this repository:

```powershell
./tools/export-reference.ps1 -Out reference-bundles/tiny-reference-v1
```

The default corpus freezes `examples/sample.jpg`, its 180-degree rotation, a white 32x32 image, and a single BGR pixel. The sample includes Chinese/English text and variable line widths. Output directories must be new; an existing baseline is never overwritten.

The default `canonical` profile uses CPU only, one effective processor, scalar kernels, NCHW storage, and disabled graph fusions. The launcher sets these controls before .NET starts and restores the caller's environment afterward. It removes inherited graph/kernel debug-dump variables so measured runs are untraced.

For a fixed external corpus, including the existing generated benchmark dataset:

```powershell
./tools/export-reference.ps1 -InputPath ../SimdPaddleOCR/dataset -Count 20 -Out reference-bundles/corpus-20
```

Input can be one image or a directory of JPG/JPEG/PNG/BMP/WebP images. Directory inputs are selected in ordinal filename order. Encoded originals and decoded BGR24 buffers are frozen, so subsequent runs can use the saved pixels without relying on fonts or a decoder. Ground-truth scoring from dataset metadata is not implemented in this step.

## Replay the frozen inputs and models

```powershell
./tools/export-reference.ps1 -ReplayBundle reference-bundles/tiny-reference-v1 -Out reference-bundles/tiny-replay-v1
cargo run --release --offline -- compare reference-bundles/tiny-reference-v1 reference-bundles/tiny-replay-v1 --atol 0 --rtol 0
```

Replay validates the input/model hashes, uses the bundle's original case IDs and raw pixel buffers, and inherits its profile and trace settings unless explicitly overridden. The compiled C# implementation comes from the current reference checkout; its provenance is captured again. This makes replay useful for detecting changes to the reference implementation as well as checking reproducibility.

`--offline` works when dependencies are already cached; omit it for the first dependency download on a normally connected machine. Dependency versions are pinned in `Cargo.lock` and the exporter's `packages.lock.json`.

## Capture more detail

```powershell
# Capture every executed node and every text line. This can produce large bundles.
./tools/export-reference.ps1 -Nodes all -MaxTracedLines 0 -Out reference-bundles/tiny-all-nodes

# Separate optimized CPU baseline, still one effective processor.
./tools/export-reference.ps1 -Profile cpu -Out reference-bundles/tiny-cpu
```

Default checkpoints capture the first, middle, penultimate, and last node indexes, mapped to the execution sink when nodes are fused. Every detected line has its crop pixels saved. Only the first three lines per image have detailed CLS/REC traces by default; `-MaxTracedLines 0` captures all lines. Both modes capture complete public OCR results, including full quadrilateral boxes and CTC alignment.

CPU and canonical bundles have different contracts and cannot be compared directly by the CLI. Compare each implementation with the matching profile and trace configuration. Node IDs refer to the C# loader's canonicalized graph, not raw ONNX node indexes; `graph-inventory.json` records the mapping, node names, operators, inputs, outputs, and opsets.

## Compare a Rust candidate

For a future complete Rust OCR pipeline, write the same bundle schema using
the frozen model and input files. The current CLS implementation uses
`run-cls` and `compare-cls` above. Full pipeline comparison will use:

```powershell
cargo run --release --offline -- compare reference-bundles/tiny-reference-v1 reference-bundles/rust-candidate --atol 1e-5 --rtol 1e-4
```

The comparator validates every model and case artifact against its own recorded hash and length before comparing. It requires matching contracts, model identities, case/artifact order, data types, shapes, and layouts. Source snapshots, assembly metadata, and timing reports are provenance and are not parity requirements.

Float tensors use `abs(candidate - reference) <= atol + rtol * abs(reference)`. Nonfinite values are rejected. Pixel bytes, text, keys, array order, and pairs of JSON integers compare exactly. Floating-point JSON numbers, including an integer paired with a floating-point number, use the numeric tolerance. Preserve integer types for structural fields such as widths, labels, counts, and span boundaries.

The first differing captured stage is reported. Tensor diagnostics include the first differing element, mismatch count, maximum absolute error, and maximum relative error. Use all-node capture to narrow a mismatch to the first executed node. Exit codes are `0` for a match, `1` for a parity difference, and `2` for invalid arguments or an invalid bundle.

See [the bundle schema](docs/reference-bundle.md) for the format and score contracts.

## Validation

```powershell
cargo test --offline
cargo clippy --offline --all-targets -- -D warnings
dotnet test ../SimdPaddleOCR/test/Sdcb.SimdPaddleOCR.UnitTests/Sdcb.SimdPaddleOCR.UnitTests.csproj -c Release --filter FullyQualifiedName~InferenceSessionTraceTests
```

The exporter uses the C# repository's pinned ImageSharp 3.1.11 decoder to preserve its existing fixture decoding. NuGet reports the existing advisories for that version; use controlled fixture images. The reference checkout documents its deferred upgrade in `docs/imagesharp-3.2-upgrade.md`. Frozen-pixel replay skips decoding.

Generated bundles and build outputs are ignored by Git. Keep any baseline you rely on in durable artifact storage together with its manifest; source code and the small synthetic comparator fixtures remain in Git.
