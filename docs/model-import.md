# Structural model import

The Rust importer loads all seven shipped networks: shared `cls`, tiny `det` and
`rec`, `small_det`, `small_rec`, `medium_det`, and `medium_rec`. This milestone
imports graph structure, operator attributes, constants, and shape metadata. It
does not execute a network or allocate activation buffers.

`Model::load_path`, `Model::load_reader<R: BufRead>`, and `Model::load_bytes` use
default limits. Their `_with_options` counterparts accept `LoadOptions`.
`LoadLimits` defaults to a 1 GiB source, nesting depth 64, and rank 8. The loader
requires one graph input and one output, rejects unresolved/non-topological
retained inputs, and reports unsupported retained operators or tensor types.
Models own constant buffers once. Accessors expose shared borrows; a caller can
share the complete immutable model with `Arc<Model>` later.

The parser streams bounded protobuf messages through buffered reads. It does not
retain the source ONNX blob. Initializer bytes become typed F32/I32/I64/U8
storage; scalar `[]`, empty `[0]`, unknown rank, and symbolic dimensions remain
distinct. Symbol strings are opaque metadata, including expression-looking
symbols in the tiny recognizer. Optional node slots retain their positions.

Normalization follows the C# importer: Constant promotion, Identity aliases,
GlobalAveragePool lowering, constant axes/scales, stride-one SAME_UPPER padding,
Conv+BatchNormalization weight folding, centered convolution-branch merging,
and dead-node/constant removal. Generated float weights use the same individual
F32 operations and accumulation order as the reference. Ordinary loading does
not hash tensors or serialize inspection snapshots.

`ShapeRecipe` records stripped dynamic Reshape relationships. `shape_metadata()`
retains the deduplicated original metadata nodes, attributes, and necessary
integer constants even when diagnostic capture is disabled. This supplies the
future shape evaluator without retaining a second model or executing integer
shape operators now. No NHWC rewrite, fusion, weight packing, kernel, worker
pool, session, or GPU path is implemented at this milestone.

## Inspect and compare

From the Rust project directory:

```powershell
cargo run --release -- inspect-model ../SimdPaddleOCR/models/det.onnx
cargo run --release -- inspect-model ../SimdPaddleOCR/models/det.onnx --out reference-bundles/det-inspection --json
cargo run --release -- validate-import reference-bundles/import-reference-v1 --json
cargo run --release -- compare-import reference-bundles/import-reference-v1 det --json
```

`inspect-model --out` writes `raw.json`, `normalized.json`, and the normalized
weight payloads in `weights/`. This standalone inspection is not an authoritative
multi-model reference archive. `validate-import` and `compare-import` accept the
C# export archive whose `manifest.json` has `kind: model_import_contracts`.

Comparison validates every source model, dictionary, structural contract,
provenance file, and normalized weight declared by the archive before comparing
the requested model. It resolves the source through the manifest. Archive paths
must remain inside the archive, including after resolving filesystem links.
Invalid references return an error rather than a successful comparison.

Canonical tensor keys are `original:<name>` or deterministic generated fold/
merge keys. Surviving node keys retain the original source index, while
`source_indices` records the sorted merged lineage. Numeric tensor IDs and
weight file paths are implementation details. Comparison requires exact graph
connectivity, rich shape metadata, attribute float bits, shape recipes and
integer operands, byte lengths, and SHA256 payload hashes. Float tolerances used
by OCR execution reference comparisons do not apply to this structural gate.
The pinned Windows C# oracle and current Rust importer produce identical folded
and merged F32 weight bytes for all shipped models.

Effective Softmax attributes resolve the missing-axis default from the imported
opset: 1 through opset 12, and -1 from opset 13. Every shipped Softmax node has an
explicit axis, so their C# reference parity is unchanged. Raw inspection preserves
the original opset and attribute absence. The loader covers the shipped operator
subset rather than claiming general ONNX operator-schema coverage.

## Validation

Small ONNX fixtures exercise aliases/cycles, unresolved inputs, dead empty
initializers, scalar constants, optional input positions, weight folds, branch
lineage, SAME_UPPER padding, checked overflow, and retained shape programs.
Authoritative model payloads remain in local archives. To run the complete gate:

```powershell
$env:SIMDOCR_IMPORT_REFERENCE = 'D:\simd-paddle-ocr-workspace\simd-paddle-ocr-rs\reference-bundles\import-reference-v1'
cargo test --release --test model_import all_shipped_models_match_authoritative_imports -- --ignored
```

Normal tests do not require the large local archive. The ignored gate requires
an explicit archive path and fails when it is missing. `measure_timings` records
parse and normalization separately, with diagnostic hashing excluded from both.
Measure production loading with `capture_raw_snapshot: false`; inspection and
archive validation perform extra hashing and output work by design.
