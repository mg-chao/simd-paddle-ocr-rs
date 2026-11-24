"""Check public capture versus the CLI, including F32 fields and GPU stage placement."""
import argparse
import ctypes
import json
from pathlib import Path
import subprocess

from collect import OUT, ROOT, digest, save
from compare import assess


def typed(value):
    if isinstance(value, dict):
        return {k: typed(v) for k, v in value.items()}
    if isinstance(value, list):
        return [typed(v) for v in value]
    if isinstance(value, float):
        return ctypes.c_float(value).value
    return value


def verify(exe, capture, dest, mixed):
    rows = json.loads((OUT / "inputs.json").read_text("utf-8"))
    for p, i in [("tiny", 2), ("small", 76), ("medium", 59)]:
        row = rows[i]
        assert digest(Path(row["input"])) == row.get("input_sha256", row["sha256"])
        command = [exe, "--preset", p, "--execution", "gpu", "--threads", "8",
                   "--gpu-precision", "mixed-f16" if mixed else "f32",
                   "--gpu-matrix", "f16" if mixed else "f32",
                   "--statistics", "--json", row["input"]]
        run = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, timeout=90)
        raw = json.loads(run.stdout)
        placement, _ = json.JSONDecoder().raw_decode(run.stderr.decode("utf-8").lstrip())
        assert placement["execution"] == "Resident"
        expected_precision = ("F16 storage; F16 matrix operands/F32 accumulation" if mixed else "F32")
        for stage in ("detector", "classifier", "recognizer"):
            entries = placement["stages"][stage]
            assert entries and all(s["execution_backend"] == "Gpu" and s["fallback"] is None
                                   and s["effective_precision"] == expected_precision for s in entries)
        name = row["path"] + ".json"
        if not mixed:
            resident = json.loads((OUT / capture / p / name).read_text("utf-8"))
            assert resident["status"] == "ok" and typed(resident["raw"]) == typed(raw)
        lines = [dict(text=x["text"], box=x["detection"]["points"], score=x["recognition_score"])
                 for x in raw["lines"]]
        reference = json.loads((OUT / "reference" / p / name).read_text("utf-8"))
        result = assess(reference["lines"], lines)
        save(dest / (p+".json"), dict(path=row["path"], result=result, placement=placement, raw=raw))
        print(p, result["exact"], result["edits"], result["missing"], result["extra"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exe", type=Path)
    parser.add_argument("capture")
    parser.add_argument("dest", type=Path)
    parser.add_argument("--mixed", action="store_true")
    args = parser.parse_args()
    verify(args.exe.resolve(strict=True), args.capture, args.dest, args.mixed)
