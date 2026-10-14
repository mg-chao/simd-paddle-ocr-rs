#!/usr/bin/env bash
# Native Linux verification; run from the repository root with pinned Rust and
# prepared model fixtures. Candidate directories and logs are never overwritten.
set -euo pipefail
output=${1:-target/validation/linux-tiny-ocr-20261014-v1}
test ! -e "$output"
mkdir -p "$output"
output=$(realpath "$output")
export RUSTUP_TOOLCHAIN=1.97.1
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-target/linux-ocr-validation}
export SIMDOCR_IMPORT_REFERENCE=target/step12-fixtures/imports
export SIMDOCR_STEP12_FIXTURES=target/step12-fixtures
reference_full=reference-bundles/tiny-ocr-allnodes-v1
reference_compact=reference-bundles/tiny-ocr-compact-v1
for bundle in "$reference_full" "$reference_compact"; do test -f "$bundle/manifest.json"; done
rustc -Vv > "$output/rustc.txt"
cargo -V > "$output/cargo.txt"
uname -a > "$output/kernel.txt"
cat /etc/os-release > "$output/os-release.txt"
test "$(rustc -V | cut -d ' ' -f 2)" = 1.97.1
python3 - "$output/source-start.json" <<'PY'
import hashlib, json, pathlib, sys
files = sorted([pathlib.Path(p) for p in ("Cargo.toml", "Cargo.lock", "rust-toolchain.toml")] + list(pathlib.Path("src").rglob("*.rs")) + list(pathlib.Path("tests").rglob("*.rs")))
json.dump({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}, open(sys.argv[1], "w"), indent=2)
PY
run_gate() {
    name=$1
    shift
    printf 'RUN %s\n' "$name"
    printf '%q ' "$@" > "$output/$name.command.txt"
    printf '\n' >> "$output/$name.command.txt"
    if "$@" > "$output/$name.log" 2>&1; then
        printf 'PASS %s\n' "$name"
    else
        status=$?
        cat "$output/$name.log"
        printf 'FAIL %s (%s)\n' "$name" "$status"
        return "$status"
    fi
}
run_gate normal-tests cargo test --locked --offline --all-features
run_gate clippy cargo clippy --locked --offline --all-targets --all-features -- -D warnings
run_gate python-tests python3 -m unittest discover -s tools/tests
run_gate imports cargo test --release --locked --offline --test model_import all_shipped_models_match_authoritative_imports -- --ignored
run_gate detector-classifier-regression cargo test --release --locked --offline --test step12_harness compact_real_models_match_all_detector_and_pixel_stages -- --ignored
run_gate line-api-regression cargo test --release --locked --offline --test line_pipeline -- --ignored
run_gate cli-build cargo build --release --locked --offline --bin simd-paddle-ocr-rs
binary="$CARGO_TARGET_DIR/release/simd-paddle-ocr-rs"
for scope in full compact; do
    reference=$reference_full
    if [[ "$scope" = compact ]]; then reference=$reference_compact; fi
    run_gate "$scope-rec-run" "$binary" run-rec "$reference" --out "$output/$scope-rec-candidate" --json
    run_gate "$scope-rec-compare" "$binary" compare-rec "$reference" "$output/$scope-rec-candidate" --json
    run_gate "$scope-ocr-run" "$binary" run-ocr "$reference" --out "$output/$scope-ocr-candidate" --json
    run_gate "$scope-ocr-compare" "$binary" compare-ocr "$reference" "$output/$scope-ocr-candidate" --json
done
run_gate rec-explicit-gate env SIMDOCR_REC_REFERENCE="$reference_compact" cargo test --release --locked --offline --test rec_harness all_frozen_tiny_rec_lines_match_authoritative_native_reference -- --ignored
run_gate ocr-explicit-gates env SIMDOCR_OCR_REFERENCE="$reference_compact" cargo test --release --locked --offline --test ocr_harness -- --ignored
python3 - "$output" "$binary" <<'PY'
import hashlib, json, pathlib, sys
out, binary = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
files = sorted([pathlib.Path(p) for p in ("Cargo.toml", "Cargo.lock", "rust-toolchain.toml")] + list(pathlib.Path("src").rglob("*.rs")) + list(pathlib.Path("tests").rglob("*.rs")))
source = {str(p):sha(p) for p in files}
start = json.loads((out / "source-start.json").read_text())
references = []
for label in ("allnodes", "compact"):
    root = pathlib.Path(f"reference-bundles/tiny-ocr-{label}-v1")
    manifest = json.loads((root / "manifest.json").read_text())
    references.append({"path":str(root), "manifest_sha256":sha(root / "manifest.json"), "models":manifest["models"], "case_count":len(manifest["cases"]), "artifact_count":sum(len(c["artifacts"]) for c in manifest["cases"])})
comparisons = {p.stem.removesuffix(".log"):json.loads(p.read_text()) for p in out.glob("*-compare.log")}
proof = {"schema_version":1, "status":"all_native_linux_gates_passed", "runtime":"native x86_64-unknown-linux-gnu", "toolchain":(out/"rustc.txt").read_text(), "cargo":(out/"cargo.txt").read_text(), "kernel":(out/"kernel.txt").read_text(), "os_release":(out/"os-release.txt").read_text(), "executable_sha256":sha(binary), "references":references, "comparisons":comparisons, "source_at_completion":source, "source_changed_during_validation":start != source, "files":{p.name:sha(p) for p in sorted(out.glob("*.log"))}, "commands":{p.name:p.read_text().strip() for p in sorted(out.glob("*.command.txt"))}}
(out / "proof.json").write_text(json.dumps(proof, indent=2)+"\n")
print(json.dumps({"status":proof["status"], "proof":str(out/"proof.json"), "source_changed_during_validation":proof["source_changed_during_validation"]}))
PY
