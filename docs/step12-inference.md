# Scalar detection and oriented text crops

The Rust pipeline runs tiny, small and medium DET and shared two-class CLS.
Its endpoint is an ordered list of packed BGR crops after perspective extraction,
intrinsic tall-crop normalization, and the optional 180-degree classification
rotation. It accepts raw pixels. Image decoding and recognition are separate
stages.

Graph tensors and arithmetic use F32. The scalar exponential reproduces the
native Windows primitive's internal table, F64 cubic and FMA order, then returns
F32 before Erf, Sigmoid or softmax arithmetic. Host `expf` rounding differed on
Linux and amplified through DET; using a wider standard exponential also changed
some Windows results. The shared implementation follows the independently
captured native behavior and is covered by compact exponential goldens.

## Library API

```rust,no_run
use std::sync::Arc;
use simd_paddle_ocr_rs::{
    detector::{Detector, DetectorOptions},
    image::{ImageView, PixelFormat},
    lines::{LineOptions, OrientedLineExtractor},
    model::Model,
    runtime::{CompiledModel, ExecutionLimits},
};

fn prepare_lines(pixels: &[u8], width: usize, height: usize)
    -> Result<(), Box<dyn std::error::Error>>
{
    let det = Arc::new(CompiledModel::new(Arc::new(Model::load_path("models/det.onnx")?))?);
    let cls = Arc::new(CompiledModel::new(Arc::new(Model::load_path("models/cls.onnx")?))?);
    let detector = Detector::new(det, DetectorOptions::default(), ExecutionLimits::default())?;
    let mut extractor = OrientedLineExtractor::new(detector, Some(cls), LineOptions::default())?;
    let image = ImageView::new(pixels, width, height, 0, PixelFormat::Bgr24)?;
    let result = extractor.extract(image)?;
    for line in result.iter() {
        println!("{}x{} rotation={} score={:?}", line.image.width(), line.image.height(),
            line.metadata.applied_rotation_degrees,
            line.metadata.classification.map(|c| c.score));
        // line.image.pixels() is packed BGR ready for a future recognizer.
    }
    let saved = result.try_to_owned()?;
    // saved remains valid across subsequent extractor calls.
    assert_eq!(saved.lines.len(), saved.as_view().iter().count());
    Ok(())
}
```

Immutable models/compiled instructions may be shared with `Arc`. Each mutable
detector or extractor owns its own sessions and workspaces. Borrowed detections
and crops remain valid until the next mutable use; Rust enforces this lifetime.
`try_to_owned()` performs explicit fallible copies. Passing `None` for the
classifier disables classification and returns `None` classification metadata.

`ImageView` supports BGR24, RGB24, BGRA32 and RGBA32, ignoring alpha and reading
the original layout directly. Stride zero means packed rows; a positive stride
must fit each row. Required bytes are `(height-1)*stride + width*bytes_per_pixel`,
so the last row may omit trailing padding. Crops always use packed BGR24.

## Numerical behavior

Detector preprocessing pads very small sources with zeros, computes the native
multiple-of-32 dimensions, performs byte-domain fixed-point resize, then
normalizes. DET preserves native BGR channel associations. CLS retains aspect,
uses the left four-times-height source window for very long lines, and pads
unused columns with -1. Native interpolation coefficients, operation order and
rounding are preserved.

DB postprocessing includes eight-connected foreground and four-connected holes,
candidate accounting, polygon scoring, round-join unclip, minimum rectangles,
source coordinate rounding, and native reading order. The initial sort includes
.NET's equal-key permutations; attribution is in `NOTICE` and the MIT license
under `licenses/`. Cropping uses F64 cubic perspective sampling with replicated
borders and native byte rounding. Intrinsic tall-crop rotation is recorded
separately from the classifier's applied 180 degrees.

Public CLS matches the C# endpoint: its terminal dot products begin with bias,
labels are chosen from logits, and the selected score uses stable Softmax.
Production executes a private backbone prefix and the compatible two-class
head. Full diagnostic execution preserves every original graph node and dense
probability tensor, while computing the public head when its feature is available.
Unequal logits can round to equal probabilities; dense probabilities must not
replace the public decision. Rotation requires label 1 and score strictly greater
than `classifier_threshold`; exact equality does not rotate.

The added runtime subset is Div, Erf, Sigmoid, MaxPool, Concat, nearest Resize,
and ConvTranspose for the shipped detector parameter combinations. Unsupported
combinations return errors. The original CLS graph's operation and accumulation
order are preserved.

## Memory and failures

`Session::run_with_input` and `run_with_input_and_trace` fill the active input
directly, eliminating a separate normalized tensor and copy. The callback must
overwrite its full slice on every execution: the arena can reuse input storage.
Existing `run`/`run_with_trace` retain copying semantics and finite validation.

Rebinding prepares a checked plan and fallible replacements before committing,
reuses sufficient capacities, and compacts when retained buffers would exceed
the activation limit. `activation_bytes()` reports the current planned storage;
`retained_activation_bytes()` reports buffer capacities. Transient staging can
temporarily retain the old and replacement buffers and is measured separately.
There is no unbounded cache of shape-specific sessions.

The image pipeline reuses resize rows, DB maps/scanline work, geometry buffers,
crop transforms, line metadata, and one BGR slab. Repeated same-shape calls after
observed contour/crop capacities are warmed allocate nothing. Changed shapes,
new contour maxima, output growth, and explicit owned copies have separate costs.
Workspace capacities, shared weights, planned activations, requested heap, and
process resident memory are distinct metrics.

Native defaults: detector side 960 (32–4096), 3000 candidates (1–10000), bitmap
threshold 0.2, unclip ratio 1.4, dilation off. An unspecified box threshold uses
0.4 for first-convolution output channels at most 16 and 0.45 otherwise; an explicit
threshold wins. Source/CLS limits default to 40 million pixels, individual crops
to 16 million pixels, and total crop bytes to `i32::MAX`. The existing 1 GiB runtime
activation budget is independent of image workspace and shared model weights.

Invalid image views/options fail before execution. Rebinding failures retain
the valid binding. Crop geometry/offsets/budgets are preflighted before slab
writing. Observer or processing errors expose no partial result; scratch may
change, and a later valid call can reuse the instance.

## Parity and measurements

`run-det` consumes frozen DET input tensors and writes independent node outputs
and probabilities. `run-step12` starts from frozen pixels and computes detection,
boxes, crops, CLS and oriented pixels without reference outputs seeding execution.
Its reference must capture every detected line. All-node references require
complete inventory coverage; checkpoint references report their narrower scope.

Float tensors, geometry and scores use `atol=1e-5`, `rtol=1e-4`. Shapes, layouts,
line ordering, integer structural fields, labels, rotations and bytes are exact.
Passing detector float tolerance does not waive a downstream crop-byte or decision
difference. Every declared asset/artifact/auxiliary file is validated, including
extra/unselected files and files following an earlier parity difference.

```powershell
./tools/benchmark-step12.ps1 -Reference reference-bundles/medium-step12-compact-v1 -Out target/measurements/step12-medium-new
```

Defaults are six alternating fresh-process timing pairs, 30 warmups and 100
measurements; three allocator-instrumented processes follow separately per mode.
Modes cover copy-inclusive DET graphs, frozen isolated stages, the complete
pixel pipeline, and one-session mixed shapes with more than eight distinct
bindings. The mixed sequence uses compact 32–128 pixel spatial bindings,
including large→small→large revisits; full-page shapes have separate parity and
latency cases. Isolated `cls_graph` measures every canonical node from a copied
frozen tensor; `cls_public` includes preprocessing and the production prefix/head
from frozen crop pixels. Timed blocks are per case/stage; aggregate samples sum matching indexes
and are not corpus wall sweeps or throughput measurements. C# explicitly uses
its span overload for matched-copy graph timing. Pixel pipelines fill session
input directly. C# allocation volume covers the current thread's managed
allocations; Rust counts its System allocator. Instrumented timings are excluded.

Rust-only `node_profile` observes node intervals in a separate run with
preallocated counters; its timing is not a production latency measurement.
The first node includes input validation/copy. Benchmark reports preserve model,
manifest, executable identities, raw samples, first-call/setup timing, requested
allocations, capacity counters and sampled process high-water memory.
The Rust-only `owned_copies` mode extracts once before timing and measures
`try_to_owned()` separately. Empty results and results containing crops are
reported individually; their allocations are excluded from borrowed-call gates.

Committed compact goldens include all nodes for all three detectors at rectangular
small bindings and two distinct CLS inputs, plus 56 native geometry/preprocessing
cases. CI runs the explicit real-model gate on Windows/Linux after hash-verified
model preparation. Large references and ZIP restorations remain immutable local
archives; their identities are recorded in `baselines.json`.
