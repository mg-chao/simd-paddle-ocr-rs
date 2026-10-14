# Canonical compact model goldens

These five immutable archives contain C# reference outputs, inventories and source provenance. Model payloads are supplied separately through the exact byte counts and SHA-256 hashes in `index.json`.

- `tiny.zip`, `small.zip`, `medium.zip`: complete pixel-to-oriented-crop references at 64×32 and 32×64, including every detector node. Medium's portrait case also contains a complete classifier line.
- `cls.zip`: two distinct frozen crops from the existing canonical sample and its 180-degree rotation, including every classifier node. This isolated classifier oracle excludes image preprocessing from its comparison scope.
- `imports.zip`: frozen C# raw/normalized graph contracts for all seven models, plus normalized constant byte counts and exact SHA-256 digests. Its explicit `model_import_hash_contracts` kind validates source models, dictionaries, semantic JSON and digest declarations; normalized weight bytes are outside this compact archive. The original `model_import_contracts` archive retains its stricter physical-weight validation.

Restore with Python 3.12 or later:

```powershell
python tools/prepare-step12-fixtures.py
$env:SIMDOCR_IMPORT_REFERENCE = 'target/step12-fixtures/imports'
cargo test --release --locked --test model_import all_shipped_models_match_authoritative_imports -- --ignored
cargo test --release --locked --test step12_harness compact_real_models_match_all_detector_and_pixel_stages -- --ignored
cargo test --release --locked --test line_pipeline -- --ignored
```

For offline preparation, pass `--models ../SimdPaddleOCR/models`. The preparation script verifies existing caches, archive hashes and downloaded model hashes. Requested model acquisition or validation failures stop the gate. The anonymous model URLs use the pinned C# commit recorded in the index. CI runs the same frozen-output comparisons on Windows and Linux.

The original Windows import oracle captured the two dictionaries with CRLF line endings. Their public Git blobs use LF. The index pins both downloaded LF identities and the final oracle identities; preparation reconstructs CRLF deterministically and verifies both hashes on either operating system.

The uncompressed restored bundles live in `target/step12-fixtures`. Each archive retains its source manifest hash in `fixture_selection`; the original local reference bundles and milestone archives remain immutable. `tools/pack-step12-fixtures.py` is the explicit packaging tool used for this selection, and is outside the CI execution path.

The C# source and test outputs derive from the Apache-2.0 SimdPaddleOCR reference. Frozen model assets retain their original provenance and identities.
