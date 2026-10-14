# Reference bundle v1

`manifest.json` is written last. Its presence marks a successfully completed export. A failed export may leave a partial directory without a manifest; choose a fresh output path for the next run.

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

This is a format example. Actual manifests contain the full options object, all four model/dictionary assets, complete cases, and provenance. Future candidate writers should copy the reference contract and frozen model/input identity fields and emit their own output descriptors and hashes. Unknown top-level fields are accepted for evolution.

Paths are portable relative paths using `/`; absolute paths, traversal, Windows device names, and paths escaping the bundle through symlinks/junctions are rejected by the comparator. Models, cases, and artifacts require unique IDs. Case and artifact arrays are ordered; public OCR line and CTC arrays also preserve reading order.

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

Performance and provenance are excluded from parity comparison. Compare performance on the same machine with matching profile/options and sufficient repetitions; the default three samples are a smoke baseline, not a performance regression gate. Encoded originals and raw buffers are both preserved, but replay uses the raw buffers and frozen model assets.
