# Tiny scalar recognition

Tiny REC is implemented independently from frozen normalized tensors and from
oriented crop pixels. Small/medium recognizers, SIMD, threading and GPU execution
are separate milestones. Shared model import and DET/CLS contracts remain intact.

## Public recognition

```rust,no_run
use std::sync::Arc;
use simd_paddle_ocr_rs::{image::{ImageView, PixelFormat}, model::Model,
    recognizer::{Dictionary, Recognizer, RecognizerOptions}, runtime::{CompiledModel, ExecutionLimits}};
fn recognize(pixels: &[u8], width: usize, height: usize) -> Result<(), Box<dyn std::error::Error>> {
    let model = Arc::new(CompiledModel::new(Arc::new(Model::load_path("models/rec.onnx")?))?);
    let dictionary = Arc::new(Dictionary::from_utf8(&std::fs::read("models/ppocr_keys.txt")?)?);
    let mut recognizer = Recognizer::new(model, dictionary, RecognizerOptions::default(), ExecutionLimits::default())?;
    let image = ImageView::new(pixels, width, height, 0, PixelFormat::Bgr24)?;
    let result = recognizer.recognize(image, true)?;
    println!("{} score={} width={}/{}", result.text, result.score, result.resized_width, result.tensor_width);
    if let Some(spans) = result.ctc_spans { for span in spans.iter() { println!("{} [{}..{})", span.text, span.start_column, span.end_column); } }
    let saved = result.try_to_owned()?;
    assert_eq!(saved.as_view().text, result.text);
    Ok(())
}
```

Immutable models and dictionaries are shared through `Arc`. Each mutable
recognizer owns one prefix session, resize scratch, compact winners and result
storage. Results borrow this instance; owned conversion copies explicitly and
survives reuse. Trace values are borrowed only during the synchronous callback.
Dense diagnostic storage is initialized lazily and has separate counters.
Callbacks can observe completed stages before a later error; the call returns a
result view only after the full requested execution succeeds.

Recognition resizes the complete source to height 48. Natural width is
`ceil(48*w/h)`. Adaptive tensor width rounds up to a multiple of 32 with minimum
32 and no 320 cap; fixed mode uses the configured width, default 320. Content
width is capped by tensor width. Existing byte-domain interpolation writes
normalized BGR planes directly into input. Right padding is normalized zero.
All four interleaved formats and the existing stride rules are accepted.

The dictionary requires strict UTF-8. Its initial BOM, one final LF-created
empty entry, and one terminal CR per entry are removed. Interior empty entries,
duplicates and multi-character tokens remain. Classes are blank 0, entries 1..N,
then space N+1. Construction checks the graph vocabulary against N+2.

CTC emits on nonblank class transitions; blanks reset repetitions. Scores use
the first frame of each emission, accumulate in F64 and return a F32 mean.
Spans are half-open, extend over repeats and exclude blanks. Alignment disabled
returns `None`; a captured empty decode returns an empty span list. Token count
is independent from character count or UTF-8 byte length.

Public compact scores preserve bias-first projection maxima and can be negative
or exceed one. Dense diagnostic decoding averages Softmax probabilities. These
remain separate contracts, even when their text and alignment are identical.

## Runtime and memory

The added execution subset includes shipped 3x2 AveragePool, inference
BatchNormalization, Squeeze/Unsqueeze, checked Transpose, rank-two-or-higher
left MatMul with shared rank-two right weights, and last-axis Softmax. Shape
symbols are opaque; actual shapes and timesteps are inferred from operators.
Unsupported combinations return errors. There is no attention MatMul support.

The production prefix ends before vocabulary MatMul. It plans no dense logits
or probability buffers. Compact scalar projection tiles independent rows and
columns while preserving bias-first, ascending-inner accumulation and first-index
ties. Every projected value must be finite. Diagnostic MatMul retains its
zero-first accumulation and separate Add.

Active/staging shape plans and binding scratch retain capacity. Shape changes
are transactional; sufficient activation capacities are reused without allocating.
No width-specific session cache exists. Warm allocation guarantees require prior
growth of both plans, resize, winner and actual decoded-output capacities.
New content can require additional text/span capacity even at a familiar width.

The default activation limit is 1 GiB, excluding shared weights. Recognition
`max_result_bytes` defaults to `i32::MAX` and bounds active text/span records.
Retained capacities, planning storage, diagnostic storage and transient growth
are reported separately. Errors return no partial view; a subsequent valid call
can reuse the instance.
`retained_workspace_bytes()` includes diagnostic planning and decoded-result
storage when tracing has initialized them. `diagnostic_retained_workspace_bytes()`
reports that subset, so adding both would count it twice. Activation limits apply
per session; reshaping can temporarily retain the old and replacement buffers.
Active result limits exclude retained spare capacity, owned copies and observer
allocations. Fallible reserves report recoverable allocation errors; ordinary
`Arc`/initial metadata allocation follows Rust's process allocation behavior.

Final REC correctness evidence (local `docs/rec-parity.json`) records both platform gates.
Compact projection (local `docs/tiny-rec-projection-tuning.json`) and
dense MatMul tuning (local `docs/tiny-rec-tuning.json`) preserve their separate score orders.

## Frozen recognition gate

```powershell
cargo run --release --locked --offline -- run-rec reference-bundles/tiny-canonical-allnodes-v1 --out reference-bundles/rust-rec-new
cargo run --release --locked --offline -- compare-rec reference-bundles/tiny-canonical-allnodes-v1 reference-bundles/rust-rec-new --json
```

This scope is explicitly partial: input/nodes/probabilities, independently
decoded dense results and compatible public results, not full image OCR.
Every declared source/model/artifact/provenance file is validated. Content width
comes from frozen crop pixels and preprocessing rules, not expected output JSON.

Committed native preprocessing and operator goldens run in normal tests.
Required Windows/Linux CI runs complete hosted tiny/small/medium OCR fixtures
through both REC and OCR, using the hash-pinned fixture index. The older
`tiny-rec.zip` and `tiny-ocr.zip` captures remain optional local references.
To explicitly include those historical captures:

```powershell
python tools/prepare-step12-fixtures.py --include-local --models ../SimdPaddleOCR/models
$env:SIMDOCR_REC_REFERENCE = 'target/step12-fixtures/tiny-rec'
cargo test --release --locked --offline --test rec_harness all_frozen_tiny_rec_lines_match_authoritative_native_reference -- --ignored
```

All-node numerical tensors use `atol=1e-5`, `rtol=1e-4`. Shapes, layouts, text,
token counts, widths, timesteps and span boundaries are exact. C# negative-axis
Unsqueeze has a distinct legacy interpretation; native operator goldens exercise
the shipped positive-axis subset and make no general negative-axis parity claim.
