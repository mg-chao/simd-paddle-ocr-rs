"""Physical-device full OCR qualification. No downloads, remote execution, or autotuning.

Generate deterministic fixtures with Pillow; run the release OCR example once
per selected case/placement. Numerical and semantic failures remain failures.
Performance measurements belong to gpu_ocr_bench, not this correctness harness.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def output(command):
    return subprocess.check_output(command, cwd=ROOT, text=True, encoding="utf-8").strip()


def fixtures(directory, sample):
    from PIL import Image, ImageDraw, __version__
    directory.mkdir(parents=True, exist_ok=True)
    source = Image.open(sample).convert("RGB")
    w, h = source.size
    cases = {"typical": source, "tiny": Image.new("RGB", (1, 27), "white"),
             "sparse": source.crop((0, 0, w, max(32, h // 5))),
             "vertical": source.transpose(Image.Transpose.ROTATE_90),
             "wide": source.crop((0, 0, w, max(32, h // 5))).resize((w * 2, max(32, h // 5)), Image.Resampling.BICUBIC),
             "perspective": source.transform((w, h), Image.Transform.PERSPECTIVE,
                 (0.96, 0.04, 1.25, -0.015, 0.98, 3.5, -0.00006, 0.00004), Image.Resampling.BICUBIC, fillcolor="white")}
    dense = Image.new("RGB", (w * 2, h * 2), "white")
    for y in (0, h):
        for x in (0, w):
            dense.paste(source, (x, y))
    cases["dense"] = dense
    border = Image.new("RGB", (w + 19, h + 13), "black")
    border.paste(source, (7, 3))
    cases["border"] = border
    holes = Image.new("RGB", (192, 128), "white")
    draw = ImageDraw.Draw(holes)
    for x in range(5, 180, 30):
        draw.rectangle((x, 9, x + 22, 101), fill="black")
        draw.rectangle((x + 5, 15, x + 17, 90), fill="white")
    cases["holes"] = holes
    manifest = {"generator": digest(Path(__file__)), "pillow": __version__, "source_sha256": digest(sample), "cases": {}}
    for name, image in cases.items():
        path = directory / f"{name}.png"
        image.save(path, compress_level=9)
        manifest["cases"][name] = {"sha256": digest(path), "size": list(image.size)}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def check(actual, reference, mixed, full=False):
    errors = []
    absolute, relative = (0.05, 0.01) if mixed else (0.001, 0.001)
    def close(a, b, fp32=False):
        atol, rtol = (0.001, 0.001) if fp32 else (absolute, relative)
        return (isinstance(a, (float, int)) and isinstance(b, (float, int))
                and math.isfinite(a) and math.isfinite(b)
                and abs(a - b) <= atol + rtol * abs(b))
    if actual["text"] != reference["text"]:
        errors.append("joined text")
    if actual["detected_count"] != reference["detected_count"] or len(actual["lines"]) != len(reference["lines"]):
        errors.append("count")
    for field in ("detector_resized_width", "detector_resized_height"):
        if actual[field] != reference[field]:
            errors.append(field)
    coordinate_error = 0.0
    for i, (a, b) in enumerate(zip(actual["lines"], reference["lines"])):
        for field in ("text", "classification_label", "applied_rotation_degrees",
                      "recognition_content_width", "recognition_tensor_width", "recognition_time_steps", "emitted_count"):
            if a[field] != b[field]:
                errors.append(f"line {i}: {field}")
        for field in ("recognition_score", "classification_score"):
            if not close(a[field], b[field], full and field == "recognition_score"):
                errors.append(f"line {i}: {field}: {a[field]} vs {b[field]}")
        if not close(a["detection"]["score"], b["detection"]["score"], full):
            errors.append(f"line {i}: detection score")
        for p, q in zip(a["detection"]["points"], b["detection"]["points"]):
            if any(not isinstance(x, (float, int)) or not math.isfinite(x) for x in [*p, *q]):
                errors.append(f"line {i}: nonfinite coordinate")
                continue
            coordinate_error = max(coordinate_error, *(abs(x - y) for x, y in zip(p, q)))
        for field in ("ctc_spans",):
            aa, bb = a.get(field), b.get(field)
            if not isinstance(aa, list) or not isinstance(bb, list):
                errors.append(f"line {i}: missing CTC spans")
            else:
                if [(x["text"], x["start_column"], x["end_column"]) for x in aa] != [(x["text"], x["start_column"], x["end_column"]) for x in bb]:
                    errors.append(f"line {i}: CTC spans")
                if any(not close(x["score"], y["score"], full) for x, y in zip(aa, bb)):
                    errors.append(f"line {i}: CTC score")
    if coordinate_error > 0.5:
        errors.append(f"coordinates: {coordinate_error}")
    return dict(errors=errors, max_coordinate_error=coordinate_error)


def json_stream(text):
    decoder = json.JSONDecoder()
    while text.strip():
        value, end = decoder.raw_decode(text.lstrip())
        yield value
        text = text.lstrip()[end:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["vulkan", "dx12", "metal"], required=True)
    parser.add_argument("--precision", choices=["f32", "mixed-f16"], default="f32")
    parser.add_argument("--presets", nargs="+", choices=["tiny", "small", "medium"], default=["tiny"])
    parser.add_argument("--cases", nargs="+", default=["typical", "sparse", "dense"])
    parser.add_argument("--all-cases", action="store_true")
    parser.add_argument("--modes", nargs="+", choices=["cpu", "inference-only", "full"],
                        default=["cpu", "inference-only", "full"])
    parser.add_argument("--report", type=Path, default=ROOT / "reports/full-gpu/qualification")
    parser.add_argument("--sample", type=Path, default=ROOT.parent / "SimdPaddleOCR/examples/sample.jpg")
    parser.add_argument("--models", type=Path, default=ROOT.parent / "SimdPaddleOCR/models")
    parser.add_argument("--executable", type=Path, default=ROOT / ("target/release/examples/ocr.exe" if os.name == "nt" else "target/release/examples/ocr"))
    args = parser.parse_args()
    if not args.modes or args.modes[0] != "cpu" or len(set(args.modes)) != len(args.modes):
        parser.error("--modes must start with the CPU oracle and contain no duplicates")
    build_inputs = [*filter(Path.is_file, (ROOT / "src").rglob("*")), ROOT / "Cargo.toml", ROOT / "Cargo.lock", ROOT / "rust-toolchain.toml", ROOT / "examples/ocr.rs"]
    if max(p.stat().st_mtime for p in build_inputs) > args.executable.stat().st_mtime:
        raise RuntimeError("release executable predates source changes; rebuild before qualification")
    args.report.mkdir(parents=True, exist_ok=True)
    corpus = fixtures(args.report / "corpus", args.sample)
    cases = list(corpus["cases"]) if args.all_cases else args.cases
    manifest = dict(platform=platform.platform(), compiler=output(["rustc", "-Vv"]),
                    cargo=output(["cargo", "-V"]), revision=output(["git", "rev-parse", "HEAD"]),
                    source_sha256={str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT / "src").rglob("*")) if p.is_file()},
                    build_inputs_sha256={str(p.relative_to(ROOT)): digest(p) for p in build_inputs},
                    lock_sha256=digest(ROOT / "Cargo.lock"), toolchain_sha256=digest(ROOT / "rust-toolchain.toml"),
                    executable_sha256=digest(args.executable), backend=args.backend, precision=args.precision,
                    models={p.name: digest(p) for p in sorted(args.models.glob("*.onnx"))}, corpus=corpus,
                    qualification="physical-device run; other adapters unverified", results=[])
    for preset in args.presets:
        for case in cases:
            reference = None
            for mode in args.modes:
                label = f"{preset}-{case}-{mode}"
                command = [str(args.executable), "--models-dir", str(args.models), "--preset", preset, "--alignment", "--statistics",
                           "--execution", "cpu" if mode == "cpu" else "gpu", "--gpu-backend", args.backend,
                           "--gpu-precision", args.precision, "--gpu-pipeline", "full" if mode == "cpu" else mode,
                           str(args.report / "corpus" / f"{case}.png")]
                run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=120)
                (args.report / f"{label}.json").write_text(run.stdout, encoding="utf-8")
                (args.report / f"{label}.log").write_text(run.stderr, encoding="utf-8")
                result = dict(label=label, command=command, returncode=run.returncode, errors=[])
                if run.returncode:
                    result["errors"].append(run.stderr)
                else:
                    actual = json.loads(run.stdout)
                    if mode == "cpu":
                        reference = actual
                    if reference is None:
                        result["errors"].append("CPU reference failed")
                    else:
                        # Full retains FP32 DET/REC even in a mixed context.
                        result.update(check(actual, reference, args.precision == "mixed-f16", mode == "full"))
                    statistics = list(json_stream(run.stderr))
                    result["statistics"] = statistics
                    if mode == "full":
                        stats = statistics[-1]
                        if stats.get("mode") != "Full" or stats.get("cpu_compute_stages") != [] or stats.get("diagnostic_readback_bytes") != 0:
                            result["errors"].append("residency contract")
                manifest["results"].append(result)
                (args.report / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
                print(label, "FAIL" if result["errors"] else "PASS", flush=True)
    return int(any(result["errors"] for result in manifest["results"]))


if __name__ == "__main__":
    sys.exit(main())
