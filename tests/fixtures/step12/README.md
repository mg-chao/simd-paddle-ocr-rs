# Canonical compact model goldens

Five compact immutable archives are committed for CI. Two larger REC/OCR trace archives remain local and are ignored by Git. All seven contain C# reference outputs, inventories and source provenance. Model payloads are supplied separately through the exact byte counts and SHA-256 hashes in `index.json`; the two larger entries are marked `local_only`.

- `tiny.zip`, `small.zip`, `medium.zip`: complete pixel-to-oriented-crop references at 64×32 and 32×64, including every detector node. Medium's portrait case also contains a complete classifier line.
- `cls.zip`: two distinct frozen crops from the existing canonical sample and its 180-degree rotation, including every classifier node. This isolated classifier oracle excludes image preprocessing from its comparison scope.
- `imports.zip`: frozen C# raw/normalized graph contracts for all seven models, plus normalized constant byte counts and exact SHA-256 digests. Its explicit `model_import_hash_contracts` kind validates source models, dictionaries, semantic JSON and digest declarations; normalized weight bytes are outside this compact archive. The original `model_import_contracts` archive retains its stricter physical-weight validation.
- `tiny-rec.zip` (local only, about 93 MB): three selected frozen recognition lines across two cases and tensor widths 288, 320 and 352. Its 492 artifacts retain every tiny REC node (477 node tensors), final Softmax probabilities, dense decode and compact public decode. Frozen normalized inputs are the execution boundary; this gate has partial OCR scope.
- `tiny-ocr.zip` (local only, about 84 MB): complete source-pixel-to-public-OCR capture for an upright 160×58 two-line page, a 32×32 blank image and a 1×1 image. Its 1,295 artifacts include 726 DET, 212 CLS and 318 REC node tensors, oriented crops, both recognition score contracts and 16 public CTC spans. The full local milestone references additionally cover rotated pages and 32 recognition lines.

Restore with Python 3.12 or later:

```powershell
python tools/prepare-step12-fixtures.py
$env:SIMDOCR_IMPORT_REFERENCE = 'target/step12-fixtures/imports'
cargo test --release --locked --test model_import all_shipped_models_match_authoritative_imports -- --ignored
cargo test --release --locked --test step12_harness compact_real_models_match_all_detector_and_pixel_stages -- --ignored
cargo test --release --locked --test line_pipeline -- --ignored
```

For full REC/OCR checks, supply the hash-pinned `tiny-rec.zip` and `tiny-ocr.zip` files in this directory, or pass `--goldens` pointing at a directory containing `index.json` and all seven archives. Existing local files are preserved during Git cleanup. These reference outputs cannot be regenerated from the downloaded model payloads alone. Explicit local preparation fails if a requested archive is missing or differs from its pinned identity.

```powershell
python tools/prepare-step12-fixtures.py --include-local
$env:SIMDOCR_REC_REFERENCE = 'target/step12-fixtures/tiny-rec'
cargo test --release --locked --test rec_harness all_frozen_tiny_rec_lines_match_authoritative_native_reference -- --ignored
$env:SIMDOCR_OCR_REFERENCE = 'target/step12-fixtures/tiny-ocr'
cargo test --release --locked --test ocr_harness -- --ignored
cargo run --release --locked --features heap-profiler --bin ocr-benchmark -- --reference target/step12-fixtures/tiny-ocr --mode rec_public_borrowed --warmup 3 --samples 3
cargo run --release --locked --features heap-profiler --bin ocr-benchmark -- --reference target/step12-fixtures/tiny-ocr --mode ocr_borrowed --warmup 3 --samples 3
cargo run --release --locked --features heap-profiler --bin ocr-benchmark -- --reference target/step12-fixtures/tiny-ocr --mode mixed_widths --warmup 3 --samples 3
```

For offline preparation, pass `--models ../SimdPaddleOCR/models`. The preparation script verifies existing caches, archive hashes and downloaded model hashes. Requested model acquisition or validation failures stop the gate. The anonymous model URLs use the pinned C# commit recorded in the index. CI runs the committed compact import, DET, CLS and line-pipeline comparisons on Windows and Linux. Normal tests include native REC preprocessing and operator goldens; full REC/OCR and allocation smoke modes use the optional local archives. Each comparator first validates every declared asset, artifact and auxiliary file, including unselected captures, before checking the required semantic stage coverage; outputs are computed from the authorized input boundary.

The original Windows import oracle captured the two dictionaries with CRLF line endings. Their public Git blobs use LF. The index pins both downloaded LF identities and the final oracle identities; preparation reconstructs CRLF deterministically and verifies both hashes on either operating system.

The uncompressed restored bundles live in `target/step12-fixtures`. Each archive retains its source manifest hash in `fixture_selection`; the original local reference bundles and milestone archives remain immutable. `tools/pack-step12-fixtures.py` is the explicit packaging tool used for this selection, and is outside the CI execution path.

Step 1 recognition parity (local `docs/rec-parity.json`) and Step 2 tiny OCR parity (local `docs/tiny-ocr-parity.json`) pin historical Windows and Linux executable/source identities and distinguish full and compact capture scopes. These reports remain local evidence. The three-line REC and two-line OCR selections now require the optional local archives; they are not default CI gates. Independently captured pixel-format and operator goldens live in [recognition-v1](../recognition-v1/README.md) and run in ordinary tests.

The C# source and test outputs derive from the Apache-2.0 SimdPaddleOCR reference. Frozen model assets retain their original provenance and identities.
