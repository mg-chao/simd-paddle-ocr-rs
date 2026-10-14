# Reference bundle v1

`manifest.json` is written last. Its presence marks a successfully completed export. A failed export may leave a partial directory without a manifest; choose a fresh output path for the next run. The Rust `bundle::BundleWriter` writes and syncs a temporary manifest before publishing it, creates a fresh directory, and refuses to overwrite existing files. It accepts only artifacts produced by that writer and requires every written artifact to be attached to a case before completion.

## Manifest

The comparator consumes these fields:

```json
{
  "schema_version": 1,
  "contract": { "model": "tiny", "profile": "canonical" },
  "models": [
    {
      "id": "rec",
      "path": "models/rec.onnx",
      "sha256": "<64 hexadecimal digits>",
      "bytes": 4462639
    }
  ],
  "cases": [
    {
      "id": "sample",
      "kind": "ocr",
      "artifacts": [
        {
          "id": "line.000.rec.input",
          "path": "cases/sample/line.000.rec/input.f32.bin",
          "dtype": "f32",
          "shape": [1, 3, 48, 320],
          "layout": "nchw",
          "sha256": "<64 hexadecimal digits>",
          "bytes": 184320
        }
      ]
    }
  ]
}
```

This is a format example. Actual OCR manifests contain the full options object, all four model/dictionary assets, complete cases, and provenance. Candidate writers should copy the reference contract and frozen model/input identity fields and emit their own output descriptors and hashes. `bundle::BundleWriter` provides `copy_model`, `write_bytes`, `write_floats`, `write_json`, `write_auxiliary`, `copy_auxiliary`, `add_case`, and `finish` for this purpose. Unknown top-level fields are accepted for evolution.

Paths are portable relative paths using `/`; absolute paths, traversal, Windows device names, and paths escaping the bundle through symlinks/junctions are rejected by the comparator. Models, cases, and artifacts require unique IDs. Case and artifact arrays are ordered; public OCR line and CTC arrays also preserve reading order.

## Integrity validation and partial comparisons

```powershell
cargo run --release --offline -- validate reference-bundles/tiny-reference-v1
cargo run --release --offline -- compare-stage reference-bundles/tiny-reference-v1 reference-bundles/rust-stage --case sample --artifact line.000.cls.node.0000
```

The library APIs are `validate_bundle(directory)`, `compare_bundles(reference, candidate, tolerance)`, `compare_stage(reference, candidate, case_id, artifact_id, tolerance)`, and `compare_cls(reference, candidate, tolerance)`. Existing full comparisons retain their v1 identity, ordering, metadata, and numeric-tolerance requirements.

A stage comparison selects one exact case/artifact pair by ID and reports `PARTIAL MATCH` or `PARTIAL DIFFERENCE`. It permits unrelated candidate cases/artifacts to be absent or reordered. Both bundles must still have the same contract and model identities, and the selected case kind, dtype, shape, and layout must match. A missing reference selector is an invalid request; a valid candidate missing the selected stage is a parity difference. A partial match establishes parity only for the selected artifact. Copying unimplemented outputs from the reference does not establish Rust inference parity.

`compare-cls` selects every frozen CLS input, captured node, and probability
tensor declared by a canonical CPU/NCHW reference. It compares that complete
selection in reference order, validates every declared file on both sides,
and reports partial classifier coverage with required line/artifact counts.
It rejects a reference with no captured CLS inputs. A candidate cannot omit
a required classifier artifact and still pass. Public preprocessing metadata,
DET, REC, and complete OCR results remain outside this comparison scope.
The batch traversal avoids repeatedly hashing the whole bundle for each node.

Every declared model, artifact, and auxiliary file in both bundles is validated before a parity result is returned, including files outside the selected stage and extra candidate cases. An invalid file takes precedence over an earlier parity difference. Each declaration is streamed once per side; files declared by multiple entries are checked for each declaration. Numeric and byte payloads use 64 KiB buffers, without materializing whole tensors. The comparator retains the first parity difference in reference stage order and completes the remaining integrity checks. Float diagnostics cover all elements of the first differing tensor and preserve its global element indexes across stream blocks.

Manifests and individual JSON artifacts are limited to 64 MiB each. JSON documents are parsed in memory; numeric and byte payload memory is independent of their file size. Float payloads must contain finite values, including in stages that are not selected for comparison.

`ComparisonReport` retains `checked_cases`, `checked_artifacts`, and `first_difference`, and adds an explicit `scope`, per-side `reference_validation`/`candidate_validation` statistics, and overall `elapsed_ms`. Parity coverage stops at the first difference; validation statistics count every present declaration. `ValidationReport` records checked model/case/artifact/auxiliary counts and bytes hashed, excluding the manifest. Its `elapsed_ms` is populated for standalone validation and is `null` when timing is shared by a paired comparison. Auxiliary contents and provenance are validated for integrity when indexed, but are not output-parity requirements.

## Optional auxiliary files

Version 1 manifests may additionally index supporting files:

```json
{
  "auxiliary_files": [
    {
      "path": "graph-inventory.json",
      "sha256": "<64 hexadecimal digits>",
      "bytes": 128844,
      "role": "graph_inventory"
    }
  ]
}
```

Every auxiliary descriptor requires a nonempty role, portable relative path, SHA-256, and byte count. Auxiliary paths must be unique and cannot refer to the manifest itself. Roles describe the file; auxiliary files are checked as opaque bytes. Model-import commands separately validate the meaning of graph/model contracts. Locally copied provenance snapshots, graph contracts and performance reports can be indexed here. Original absolute assembly/source locations in provenance are informational and are never followed by the validator.

Old v1 manifests without `auxiliary_files` remain supported. Their old `diagnostics` paths and unknown provenance fields remain informational: a successful check verifies the declared model/artifact files, not every unindexed file in the directory. This is a backward-compatible addition to the OCR bundle format. Model-import contracts have a separately versioned schema; they do not change OCR score semantics or require an OCR bundle schema bump.

## Artifact encoding

| dtype | Encoding | Shape |
|---|---|---|
| `f32` | Finite IEEE 754 binary32, little endian | Product of dimensions times 4 must equal bytes |
| `u8` | Raw bytes | Product of dimensions must equal bytes |
| `json` | UTF-8 JSON | `[]`, with `layout: "json"` |

For numeric tensors `shape: []` represents one scalar. A dimension of zero represents an empty tensor. Tensor shapes are logical shapes. Rank-4 node tensors may have `layout: "nhwc"` while their logical shape remains `[N,C,H,W]`; lower-rank contiguous outputs use `row_major`. Preprocessed inputs are explicitly exported as logical NCHW even under the CPU profile.

`input.pixels` has shape `[height, stride]` and layout `bgr24_rows`. Crop buffers have shape `[height, width, 3]` and layout `bgr24`. The metadata artifact records the source width, height, stride, pixel format, and fixture origin.

The exporter preserves these stages in order:

1. Input metadata, optional encoded original, and exact BGR pixel bytes.
2. Detector normalized input, selected node outputs, probabilities, and complete ordered boxes with resize ratios.
3. Every line's pre-classification perspective crop.
4. For traced lines: classifier input/nodes/probabilities/result, rotated recognition pixels, recognizer input/nodes/probabilities, dense decoded result, and public compact result.
5. Complete public OCR result, with four-corner boxes, text, scores, classification, applied rotation, recognition widths/timesteps, and token alignment.

Node IDs such as `det.node.0121` are canonicalized model indexes. Under fusion the callback observes the final node of the fused step before workspace reuse. Earlier transient activations cannot be recovered by reading the arena after inference. The opt-in trace API avoids changing the normal inference hot path.

## Score contracts

The reference currently has two recognition score behaviors, and this exporter records both explicitly:

- `result.public` and `line.*.rec.public` preserve the existing compact MatMul/ArgMax maxima. They are not recalibrated and can exceed 1.
- `line.*.rec.decoded_dense` consumes the graph's final Softmax output and averages probabilities at the first frame of each emitted token.

Both paths must agree on text, emitted count, content/tensor widths, time steps, and token span boundaries for the export to complete. Their scores are intentionally kept separate. Changing the public score API should be an explicit later compatibility decision with a new baseline contract.

Token classes are blank at 0, dictionary tokens at 1 through N, and space at N+1. Tokens are strings, not necessarily single Unicode characters. CTC spans are half-open timestep intervals; repeated frames stay inside a span, and blanks end it.

## Provenance and measurements

Provenance records the reference Git HEAD, dirty status, byte-preserved tracked patch, snapshots of C#/project/build inputs (including untracked files), exporter source/lockfile/launcher snapshots, aggregate source hashes, SDK/runtime/OS/architecture, actual CPU feature flags, named runtime switches, and assembly versions/hashes. Thus the new internal trace method is captured even before it is committed. Replaying uses the current C# checkout; restore the recorded revision and patch/source inputs when reproducing the original implementation.

`performance.json` contains model load time, each fixture's first call, warm samples, p50/p95, mean throughput, allocated bytes, and process working set/peak. Timed calls use the complete public pipeline with alignment enabled, before diagnostic sessions and artifact writes. First calls share a process: only the first fixture is process-cold. Working set includes the runtime, decoding, and provenance capture; it is process memory rather than tensor memory alone.

Performance and provenance are excluded from parity comparison. Their locally bundled files may be integrity checked through `auxiliary_files`. Compare performance on the same machine with matching profile/options and sufficient repetitions; the default three samples are a smoke baseline, not a performance regression gate. An explicit measurement can use `-Warmup 3 -Repeat 30`, keeping diagnostic tracing and file writing outside timed calls. Harness timing measures validation/comparison rather than OCR inference; distinguish fresh-process runs, warm filesystem-cache runs, total bytes verified, and process peak working set. Peak working set includes manifests/JSON documents/runtime overhead and is not tensor allocation alone. Encoded originals and raw buffers are both preserved, but replay uses the raw buffers and frozen model assets.

Generated bundles remain outside Git. Preserve a local archive together with an index recording archive and manifest SHA-256, all frozen model/dictionary hashes, source-tree digest, runtime profile/options, and restore/validation commands. Small synthetic fixtures and contract metadata can remain in Git. Restore into a fresh directory and validate before using a recovered baseline; do not regenerate a baseline in place when testing a changed reference checkout.
