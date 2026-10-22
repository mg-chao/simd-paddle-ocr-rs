"""Fixed public sample: exact text/orientation agreement, one run per configuration.

This is correctness validation, with no timing or benchmark loops. Requires the
release OCR example and sibling repository's public examples/sample.jpg.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/native-wgpu/ocr-final"
REPORT.mkdir(parents=True, exist_ok=True)
IMAGE = ROOT.parent / "SimdPaddleOCR/examples/sample.jpg"
EXE = ROOT / ("target/release/examples/ocr.exe" if os.name == "nt" else "target/release/examples/ocr")
results = []
for preset in ["tiny", "small", "medium"]:
    reference = None
    for backend, precision in [("cpu", "f32"), ("vulkan", "f32"), ("dx12", "f32"), ("vulkan", "mixed-f16"), ("dx12", "mixed-f16")]:
        label = f"{preset}-{backend}-{precision}"
        command = [str(EXE), "--preset", preset, "--json", "--statistics", "--execution", "cpu" if backend == "cpu" else "gpu"]
        if backend != "cpu":
            command += ["--gpu-backend", backend, "--gpu-precision", precision]
        run = subprocess.run([*command, str(IMAGE)], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=120)
        (REPORT / f"{label}.json").write_text(run.stdout, encoding="utf-8")
        (REPORT / f"{label}.stderr").write_text(run.stderr, encoding="utf-8")
        result = dict(preset=preset, backend=backend, precision=precision, returncode=run.returncode,
                      image_sha256=hashlib.sha256(IMAGE.read_bytes()).hexdigest(),
                      executable_sha256=hashlib.sha256(EXE.read_bytes()).hexdigest())
        if run.returncode == 0:
            output = json.loads(run.stdout)
            semantic = [(line["text"], line["classification_label"], line["applied_rotation_degrees"]) for line in output["lines"]]
            if backend == "cpu":
                reference = semantic
            result.update(lines=len(semantic), matching_text_orientation=semantic == reference)
            if semantic != reference:
                result["differences"] = {"reference": reference, "actual": semantic}
        results.append(result)
        print(label, "PASS" if run.returncode == 0 and result["matching_text_orientation"] else "FAIL", flush=True)
        (REPORT / "summary.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
if any(r["returncode"] or not r["matching_text_orientation"] for r in results):
    raise SystemExit(1)
