"""Assert instrumented borrowed-inference allocation gates; timings are observations only."""
import argparse
import hashlib
import json
from pathlib import Path

COUNTERS = ("warm_allocation_count", "warm_allocation_bytes",
            "warm_retained_heap_bytes", "warm_peak_heap_growth_bytes")
MODES = ("rec_public_borrowed", "ocr_borrowed", "mixed_widths")


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(65536), b""):
            value.update(block)
    return value.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(report, reference, mode, executable, warmup=3, samples=3):
    require(mode in MODES, "Unsupported allocation smoke mode")
    require(report.get("instrumented") is True and report.get("runtime") == "rust",
            "Allocation smoke requires an instrumented Rust executable")
    require(report.get("mode") == mode and report.get("warmup") == warmup
            and report.get("samples") == samples, "Allocation smoke protocol differs")
    manifest_path = reference / "manifest.json"
    require(report.get("manifest_sha256") == sha(manifest_path), "Reference manifest identity differs")
    require(report.get("executable_sha256") == sha(executable), "Executable identity differs")
    rows = report.get("inputs")
    require(isinstance(rows, list) and bool(rows), "Allocation smoke has no input rows")
    for row in rows:
        require(row.get("stage") == mode, "Allocation smoke row stage differs")
        require(isinstance(row.get("samples_ms"), list) and len(row["samples_ms"]) == samples,
                "Allocation smoke sample count differs")
        for counter in COUNTERS:
            require(type(row.get(counter)) is int and row[counter] == 0,
                    f"Allocation gate failed: {mode}/{row.get('case_id')}/{row.get('artifact_id')}/{counter}={row.get(counter)}")
    if mode == "mixed_widths":
        sequence = report.get("shape_sequence", [])
        require(isinstance(sequence, list) and len(sequence) > 8,
                "Mixed-width allocation gate has insufficient shapes")
        require(all(isinstance(shape, list) and len(shape) == 4 and shape[:3] == [1, 3, 48]
                    and type(shape[3]) is int and shape[3] > 0 for shape in sequence),
                "Mixed-width allocation shape is malformed")
        require(len({tuple(shape) for shape in sequence}) > 8
                and min(shape[3] for shape in sequence) <= 32
                and max(shape[3] for shape in sequence) > 320,
                "Mixed-width allocation gate must cross narrow/wide shape boundaries")
        require(len(rows) == 1 and rows[0].get("case_id") == "mixed-widths"
                and rows[0].get("artifact_id") == "rec.pixels", "Mixed-width row identity differs")
    else:
        manifest = json.loads(manifest_path.read_bytes())
        expected = [(case["id"], artifact["id"], artifact["sha256"], artifact["shape"])
                    for case in manifest["cases"] for artifact in case["artifacts"]
                    if (artifact["id"].endswith(".rec.pixels") if mode == "rec_public_borrowed"
                        else artifact["id"] == "input.pixels")]
        observed = [(row.get("case_id"), row.get("artifact_id"), row.get("sha256"), row.get("shape"))
                    for row in rows]
        require(bool(expected) and observed == expected, "Allocation smoke input coverage differs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()
    validate(json.loads(args.report.read_bytes()), args.reference, args.mode, args.executable,
             args.warmup, args.samples)
    print("PASS instrumented allocation smoke: " + args.mode)


if __name__ == "__main__":
    main()
