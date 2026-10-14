#!/usr/bin/env bash
# Native Linux allocation gates matching CI. Timings are smoke observations.
# Run only after any official latency protocol has finished.
set -euo pipefail
output=${1:?usage: verify-ocr-linux-allocation.sh NEW_OUTPUT_DIRECTORY [REFERENCE]}
reference=${2:-target/step12-fixtures/tiny-ocr}
rec_reference=${3:-target/step12-fixtures/tiny-rec}
test ! -e "$output"
test -f "$reference/manifest.json"
test -f "$rec_reference/manifest.json"
mkdir -p "$output"
output=$(realpath "$output")
reference=$(realpath "$reference")
rec_reference=$(realpath "$rec_reference")
export RUSTUP_TOOLCHAIN=1.97.1
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-target/linux-ocr-validation}
rustc -Vv > "$output/rustc.txt"
cargo -V > "$output/cargo.txt"
uname -a > "$output/kernel.txt"
test "$(rustc -V | cut -d ' ' -f 2)" = 1.97.1
python3 - "$output/source-start.json" <<'PY'
import hashlib, json, pathlib, sys
files = sorted([pathlib.Path(p) for p in ("Cargo.toml", "Cargo.lock", "rust-toolchain.toml")] + list(pathlib.Path("src").rglob("*.rs")) + list(pathlib.Path("tests").rglob("*.rs")))
json.dump({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}, open(sys.argv[1], "w"), indent=2)
PY
printf '%q ' env SIMDOCR_REC_REFERENCE="$rec_reference" cargo test --release --locked --offline --test rec_harness all_frozen_tiny_rec_lines_match_authoritative_native_reference -- --ignored > "$output/committed-rec-selection.command.txt"
printf '\n' >> "$output/committed-rec-selection.command.txt"
env SIMDOCR_REC_REFERENCE="$rec_reference" cargo test --release --locked --offline --test rec_harness all_frozen_tiny_rec_lines_match_authoritative_native_reference -- --ignored > "$output/committed-rec-selection.log" 2>&1
printf '%s\n' 'PASS committed-rec-selection'
printf '%q ' env SIMDOCR_OCR_REFERENCE="$reference" cargo test --release --locked --offline --test ocr_harness -- --ignored > "$output/committed-ocr-selection.command.txt"
printf '\n' >> "$output/committed-ocr-selection.command.txt"
env SIMDOCR_OCR_REFERENCE="$reference" cargo test --release --locked --offline --test ocr_harness -- --ignored > "$output/committed-ocr-selection.log" 2>&1
printf '%s\n' 'PASS committed-ocr-selection'
printf '%s\n' 'cargo build --release --locked --offline --features heap-profiler --bin ocr-benchmark' > "$output/build.command.txt"
cargo build --release --locked --offline --features heap-profiler --bin ocr-benchmark > "$output/build.log" 2>&1
cp "$CARGO_TARGET_DIR/release/ocr-benchmark" "$output/ocr-benchmark"
for mode in rec_public_borrowed ocr_borrowed mixed_widths; do
    printf 'RUN %s\n' "$mode"
    printf '%q ' "$output/ocr-benchmark" --reference "$reference" --mode "$mode" --warmup 3 --samples 3 > "$output/$mode.command.txt"
    printf '\n' >> "$output/$mode.command.txt"
    "$output/ocr-benchmark" --reference "$reference" --mode "$mode" --warmup 3 --samples 3 > "$output/$mode.json" 2> "$output/$mode.stderr.log"
    printf 'PASS %s\n' "$mode"
done
python3 - "$output" "$reference" "$rec_reference" <<'PY'
import datetime, hashlib, json, pathlib, sys
out, reference, rec_reference = map(pathlib.Path, sys.argv[1:])
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
files = sorted([pathlib.Path(p) for p in ("Cargo.toml", "Cargo.lock", "rust-toolchain.toml")] + list(pathlib.Path("src").rglob("*.rs")) + list(pathlib.Path("tests").rglob("*.rs")))
source = {str(p):sha(p) for p in files}
assert source == json.loads((out / "source-start.json").read_text()), "source changed during allocation gates"
executable = sha(out / "ocr-benchmark")
reports = {}
for mode in ("rec_public_borrowed", "ocr_borrowed", "mixed_widths"):
    report = json.loads((out / f"{mode}.json").read_text())
    assert report["instrumented"] is True and report["runtime"] == "rust"
    assert report["mode"] == mode and report["warmup"] == 3 and report["samples"] == 3
    assert report["executable_sha256"] == executable
    assert report["manifest_sha256"] == sha(reference / "manifest.json")
    assert report["inputs"], "allocation gate has no input rows"
    for row in report["inputs"]:
        for counter in ("warm_allocation_count", "warm_allocation_bytes", "warm_retained_heap_bytes", "warm_peak_heap_growth_bytes"):
            assert row[counter] == 0, (mode, row["case_id"], row["artifact_id"], counter, row[counter])
    if mode == "mixed_widths":
        assert len({tuple(s) for s in report["shape_sequence"]}) > 8
    reports[mode] = report
proof = {
    "schema_version":1,
    "status":"all_native_linux_allocation_gates_passed",
    "completed_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "runtime":"native x86_64-unknown-linux-gnu",
    "scope":"CI allocation gates; reported milliseconds are smoke observations, not an official Linux latency protocol",
    "toolchain":(out / "rustc.txt").read_text(),
    "cargo":(out / "cargo.txt").read_text(),
    "kernel":(out / "kernel.txt").read_text(),
    "executable_sha256":executable,
    "reference":{"path":str(reference), "manifest_sha256":sha(reference / "manifest.json")},
    "committed_selection_gates":{
        "rec":{"reference_path":str(rec_reference), "manifest_sha256":sha(rec_reference / "manifest.json"), "log_sha256":sha(out / "committed-rec-selection.log")},
        "ocr":{"reference_path":str(reference), "manifest_sha256":sha(reference / "manifest.json"), "log_sha256":sha(out / "committed-ocr-selection.log")},
    },
    "source_at_completion":source,
    "source_changed_during_validation":False,
    "reports":reports,
    "files":{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()},
    "commands":{p.name:p.read_text().strip() for p in sorted(out.glob("*.command.txt"))},
}
(out / "proof.json").write_text(json.dumps(proof, indent=2)+"\n")
print(json.dumps({"status":proof["status"], "proof":str(out / "proof.json"), "executable_sha256":executable}))
PY
