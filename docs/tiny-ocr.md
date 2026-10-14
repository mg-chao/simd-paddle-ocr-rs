# Complete scalar tiny OCR

`Ocr` composes an existing `OrientedLineExtractor` and `Recognizer`. Its serial
raw-pixel pipeline independently computes DET, ordered boxes, perspective crops,
optional CLS rotation and tiny REC/CTC results. Image decoding is caller-owned.

Construct stages with shared `Arc<CompiledModel>` values and a shared dictionary,
then `Ocr::new(extractor, recognizer, OcrOptions::default())?`. Call
`run(image, return_ctc_alignment)` or the synchronous trace variant.
`OcrView` exposes aggregate newline-joined `text`, detector resized dimensions,
detected count, ratios and `get`/`iter` line views. Each line includes quadrilateral
box/score, text, recognition/classification scores and label, applied rotation,
emitted token count, content/tensor width, timesteps and optional CTC spans.

One result is retained for every detection, including empty text. Reading order
matches DET. Disabled CLS returns label/score/applied rotation zero; intrinsic
tall-crop handling remains separate. An empty detection preserves resized dimensions.
CTC capture disabled is `None`; captured empty output is an empty list.

Crop pixels are consumed directly. Borrowed results use one reusable joined-text
slab, line metadata and span records. Completed recognition text/runs are copied
into this result storage before reusing the single recognizer. No crop copy,
worker pool, per-width session cache or batching is introduced.

`try_to_owned()` performs explicit fallible copies. The owned result exposes
`as_view()` and remains valid after later runs. Owned results store joined text
and ergonomic per-line strings; their copy cost is measured separately.
`max_total_result_bytes` defaults to `i32::MAX` and checks active joined text,
line metadata and optional span records. Result/workspace retained capacities
are separate from activation budgets and process resident memory.
The result limit bounds active text and records; it excludes spare capacity,
owned copies and observer allocations. Stage callbacks can run before a later
error, while a final result view is returned only on success.

Final complete correctness evidence (local `docs/tiny-ocr-parity.json`) records the full
Windows/Linux gates and distinguishes the larger references from compact selections.

## Complete parity

```powershell
./tools/export-reference.ps1 -Nodes all -MaxTracedLines 0 -Out reference-bundles/tiny-ocr-new
cargo run --release --locked --offline -- run-ocr reference-bundles/tiny-ocr-new --out reference-bundles/rust-ocr-new
cargo run --release --locked --offline -- compare-ocr reference-bundles/tiny-ocr-new reference-bundles/rust-ocr-new --json
```

New full references carry semantic normalized graph inventories. The strict
complete comparator derives node sets, shapes and vocabulary from validated
models/dictionary, requires every detected line/stage and complete public output,
and rejects missing/extra cases or output artifacts. Older numeric-inventory
archives remain immutable and are useful for the explicitly partial REC gate.

Candidates start from source pixels, models and dictionary. Reference outputs
never seed computation. Generated descriptors are ordered to match the reference
manifest. All declared files are validated even after an earlier parity mismatch.
Exit codes remain 0 match, 1 difference and 2 invalid arguments/data.

The optional local gate restores a real two-line page with different recognition
widths plus blank/tiny cases. Supply the ignored `tiny-rec.zip` and `tiny-ocr.zip`
archives in `tests/fixtures/step12` before preparing them. Large original-page
and rotation references also remain local archives.

```powershell
python tools/prepare-step12-fixtures.py --include-local --models ../SimdPaddleOCR/models
$env:SIMDOCR_OCR_REFERENCE = 'target/step12-fixtures/tiny-ocr'
cargo test --release --locked --offline --test ocr_harness -- --ignored
```

The local archive commands above preserve the historical tiny selection. Current
Windows/Linux CI also downloads the immutable complete four-case OCR fixtures
and requires all-node REC, complete OCR and instrumented borrowed/allocation
gates for tiny, small and medium. See [the fixture guide](../tests/fixtures/step12/README.md)
and [all-model inference](all-model-inference.md) for the current required gates.
