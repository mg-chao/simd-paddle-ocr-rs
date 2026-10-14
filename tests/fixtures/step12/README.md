# Canonical compact model goldens

Five compact immutable archives are committed for CI. Versioned complete OCR archives for all three model families run both REC and OCR gates: tiny is acquired from the public `fixtures-v1` GitHub release, and small/medium from `fixtures-v2`. Two historical REC/OCR trace archives remain local and are ignored by Git. All references contain C# outputs, inventories and source provenance. Model payloads are supplied separately through the exact byte counts and SHA-256 hashes in `index.json`; the two historical entries remain marked `local_only`.

- `tiny.zip`, `small.zip`, `medium.zip`: complete pixel-to-oriented-crop references at 64×32 and 32×64, including every detector node. Medium's portrait case also contains a complete classifier line.
- `cls.zip`: two distinct frozen crops from the existing canonical sample and its 180-degree rotation, including every classifier node. This isolated classifier oracle excludes image preprocessing from its comparison scope.
- `imports.zip`: frozen C# raw/normalized graph contracts for all seven models, plus normalized constant byte counts and exact SHA-256 digests. Its explicit `model_import_hash_contracts` kind validates source models, dictionaries, semantic JSON and digest declarations; normalized weight bytes are outside this compact archive. The original `model_import_contracts` archive retains its stricter physical-weight validation.
- `tiny-ocr-ci-v1.zip` (hosted, 163,787,330 bytes): complete source-pixel-to-public-OCR capture for upright and 180-degree rotated 160×58 two-line pages, a 32×32 blank image and a 1×1 image. All 2,092 artifacts are retained, including every executed DET/CLS/REC node, dense/public recognition forms and public CTC spans. Four recognized lines exercise tensor widths 288, 320 and 352 and applied classifier rotation. The same bundle supplies the frozen-input REC gate; a separate duplicate REC archive is unnecessary.
- `small-ocr-ci-v1.zip` (hosted, 439,183,643 bytes): the same four frozen source cases, independently executed with small DET/REC and shared CLS. All 2,788 artifacts are retained; four recognition lines exercise widths 288, 320 and 352, including native attention execution and upright/rotated public OCR.
- `medium-ocr-ci-v1.zip` (hosted, 790,136,013 bytes): the same four frozen source cases, independently executed with medium DET/REC and shared CLS. All 2,808 artifacts are retained; four recognition lines exercise widths 256, 288 and 320. Wide-line numerical parity belongs to expanded full-reference gates; the CI mixed-width allocation sequence additionally exercises widths through 992 without implying native numerical coverage for those synthetic inputs.
- `tiny-rec.zip` (local only, about 93 MB): three selected frozen recognition lines across two cases and tensor widths 288, 320 and 352. Its 492 artifacts retain every tiny REC node (477 node tensors), final Softmax probabilities, dense decode and compact public decode. Frozen normalized inputs are the execution boundary; this gate has partial OCR scope.
- `tiny-ocr.zip` (local only, about 84 MB): complete source-pixel-to-public-OCR capture for an upright 160×58 two-line page, a 32×32 blank image and a 1×1 image. Its 1,295 artifacts include 726 DET, 212 CLS and 318 REC node tensors, oriented crops, both recognition score contracts and 16 public CTC spans. The full local milestone references additionally cover rotated pages and 32 recognition lines.

Restore with Python 3.12 or later:

```powershell
python tools/prepare-step12-fixtures.py
$env:SIMDOCR_IMPORT_REFERENCE = 'target/step12-fixtures/imports'
cargo test --release --locked --test model_import all_shipped_models_match_authoritative_imports -- --ignored
cargo test --release --locked --test step12_harness compact_real_models_match_all_detector_and_pixel_stages -- --ignored
cargo test --release --locked --test line_pipeline -- --ignored
$env:SIMDOCR_SMALL_OCR_REFERENCE = 'target/step12-fixtures/small-ocr-ci-v1'
$env:SIMDOCR_MEDIUM_OCR_REFERENCE = 'target/step12-fixtures/medium-ocr-ci-v1'
foreach ($family in 'tiny', 'small', 'medium') {
    $env:SIMDOCR_REC_REFERENCE = "target/step12-fixtures/$family-ocr-ci-v1"
    $env:SIMDOCR_OCR_REFERENCE = $env:SIMDOCR_REC_REFERENCE
    cargo test --release --locked --test rec_harness "all_frozen_${family}_rec_lines_match_authoritative_native_reference" -- --ignored --exact
    cargo test --release --locked --test ocr_harness "complete_${family}_ocr_matches_every_native_stage_and_public_result" -- --ignored --exact
    cargo test --release --locked --test ocr_harness complete_ocr_rejects_ -- --ignored
}
```

Hosted acquisition prefers a supplied archive in `--goldens`, then a verified content-addressed cache, then the fixture's pinned `url`. Supplied corrupt files fail even if a valid cache exists. Every local, downloaded and cached archive is checked by exact byte count and SHA-256; every declared archive and prepared payload is checked before reuse. Preparation stages a complete validated bundle before replacing an older prepared directory. ZIP duplicate names, aliases, symlinks and escaping paths are rejected. CI caches payloads under `target/reference-models`, uploads gate logs/allocation JSON and supports `workflow_dispatch` with `cold_cache=true` to bypass cache restoration.

Select one or several fixtures with repeatable `--fixture` directory identities. `--offline` prohibits network access; supply archives in `--goldens` or pre-populate the verified cache, and supply `--models` or cached models. For example:

```powershell
python tools/prepare-step12-fixtures.py --fixture tiny-ocr-ci-v1 --offline --models ../SimdPaddleOCR/models
```

For historical local selections, supply the hash-pinned `tiny-rec.zip` and `tiny-ocr.zip` files in this directory, or pass `--goldens` pointing at a directory containing `index.json` and the requested archives. Existing local files are preserved. These reference outputs cannot be regenerated from downloaded model payloads alone. Explicit preparation fails if a requested archive is missing or differs from its pinned identity.

```powershell
python tools/prepare-step12-fixtures.py --fixture tiny-rec --fixture tiny-ocr --offline --models ../SimdPaddleOCR/models
$env:SIMDOCR_REC_REFERENCE = 'target/step12-fixtures/tiny-rec'
cargo test --release --locked --test rec_harness all_frozen_tiny_rec_lines_match_authoritative_native_reference -- --ignored
$env:SIMDOCR_OCR_REFERENCE = 'target/step12-fixtures/tiny-ocr'
cargo test --release --locked --test ocr_harness complete_tiny_ocr_matches_every_native_stage_and_public_result -- --ignored --exact
cargo test --release --locked --test ocr_harness complete_ocr_rejects_ -- --ignored
cargo run --release --locked --features heap-profiler --bin ocr-benchmark -- --reference target/step12-fixtures/tiny-ocr --mode rec_public_borrowed --warmup 3 --samples 3
cargo run --release --locked --features heap-profiler --bin ocr-benchmark -- --reference target/step12-fixtures/tiny-ocr --mode ocr_borrowed --warmup 3 --samples 3
cargo run --release --locked --features heap-profiler --bin ocr-benchmark -- --reference target/step12-fixtures/tiny-ocr --mode mixed_widths --warmup 3 --samples 3
```

Requested acquisition or validation failures stop the gate. The anonymous model URLs use the pinned C# commit recorded in the index. CI runs compact import, DET, CLS, line-pipeline and complete REC/OCR comparisons for tiny, small and medium on Windows and Linux. Each family also runs completeness/structural corruption rejection gates. Required named tests must appear in the compiled test listing before execution, preventing a stale filter from silently passing with zero tests. Normal tests include native REC preprocessing and operator goldens. Each family's instrumented `rec_public_borrowed`, `ocr_borrowed` and `mixed_widths` modes use three warmups and three samples; `tools/assert-allocation-smoke.py` requires all four warmed allocation counters to be measured and zero, matching executable/reference identities and complete input coverage. Timings are smoke observations, without CI latency thresholds. Each comparator validates every declared asset, artifact and auxiliary file before checking semantic coverage; outputs are computed from source inputs.

The original Windows import oracle captured the two dictionaries with CRLF line endings. Their public Git blobs use LF. The index pins both downloaded LF identities and the final oracle identities; preparation reconstructs CRLF deterministically and verifies both hashes on either operating system.

The uncompressed restored bundles live in `target/step12-fixtures`. Each archive retains its source manifest hash in `fixture_selection`; the original local reference bundles and milestone archives remain immutable. `tools/pack-step12-fixtures.py` is the explicit packaging tool used for this selection, and is outside the CI execution path.

Step 1 recognition parity (local `docs/rec-parity.json`) and Step 2 tiny OCR parity (local `docs/tiny-ocr-parity.json`) pin historical Windows/Linux identities and capture scopes; their historical CI labels do not replace actual remote workflow evidence. The older three-line REC and upright two-line OCR selections remain optional local references. Larger 32-line original-page references remain immutable; new hosted full-case shards must each stay below 2 GiB and run separately from default compact CI. `tools/pack-step12-fixtures.py --bundle-source SOURCE --fixture-name NAME --variant tiny --case CASE --out OUTPUT --url URL` preserves all stages of selected cases without model payloads. Omit `--case` for a complete compact corpus; repeat it for multi-case full shards. Independently captured pixel-format/operator goldens live in [recognition-v1](../recognition-v1/README.md) and run in ordinary tests.

The C# source and test outputs derive from the Apache-2.0 SimdPaddleOCR reference. Frozen model assets retain their original provenance and identities.
