"""Physical-device full OCR qualification. No downloads, remote execution, or autotuning.

Generate deterministic fixtures with Pillow; run the release OCR example once
per selected case/target. Numerical and semantic failures remain failures.
Validation uses the same Rust comparison contract as gpu_ocr_bench; no benchmark repetitions run.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", choices=["f32", "mixed-f16"], default="f32")
    parser.add_argument("--matrix", choices=["f32", "f16"], default="f32")
    parser.add_argument("--presets", nargs="+", choices=["tiny", "small", "medium"], default=["tiny"])
    parser.add_argument("--cases", nargs="+", default=["typical", "dense", "vertical"])
    parser.add_argument("--all-cases", action="store_true")
    parser.add_argument("--report", type=Path, default=ROOT / "target/gpu-validation")
    parser.add_argument("--cpu-reference-dir", type=Path, help="reuse frozen CPU references only when corpus bytes match")
    parser.add_argument("--sample", type=Path, default=ROOT.parent / "SimdPaddleOCR/examples/sample.jpg")
    parser.add_argument("--models", type=Path, default=ROOT.parent / "SimdPaddleOCR/models")
    parser.add_argument("--executable", type=Path, default=ROOT / ("target/release/examples/gpu_ocr_bench.exe" if os.name == "nt" else "target/release/examples/gpu_ocr_bench"))
    parser.add_argument("--wave", type=int, choices=range(1, 17), default=1)
    parser.add_argument("--classification-batch", type=int, choices=range(1, 17), default=4)
    parser.add_argument("--budget-seconds", type=float, default=60)
    args = parser.parse_args()
    if sys.platform == "darwin":
        parser.error("GPU execution is not supported on macOS")
    if args.budget_seconds <= 0:
        parser.error("positive execution budget required")
    build_inputs = [*filter(Path.is_file, (ROOT / "src").rglob("*")), ROOT / "Cargo.toml", ROOT / "Cargo.lock",
                    ROOT / "build.rs", ROOT / "rust-toolchain.toml", ROOT / "examples/gpu_ocr_bench.rs", ROOT / "examples/support/ocr_compare.rs"]
    if max(p.stat().st_mtime for p in build_inputs) > args.executable.stat().st_mtime:
        raise RuntimeError("release executable predates source changes; rebuild before qualification")
    args.report.mkdir(parents=True, exist_ok=False)
    corpus = fixtures(args.report / "corpus", args.sample)
    cases = list(corpus["cases"]) if args.all_cases else args.cases
    if not set(cases) <= set(corpus["cases"]):
        parser.error("unknown corpus case")
    executor = "Vulkan"
    manifest = dict(schema=2, platform=platform.platform(), compiler=output(["rustc", "-Vv"]),
                    cargo=output(["cargo", "-V"]), executor=executor, precision=args.precision, matrix=args.matrix,
                    wave=args.wave, classification_batch=args.classification_batch,
                    source_sha256={str(p.relative_to(ROOT)): digest(p) for p in build_inputs},
                    executable_sha256=digest(args.executable), models={p.name: digest(p) for p in sorted(args.models.glob("*.onnx"))},
                    corpus=corpus, measurements=0, qualification=f"physical {executor}; unchanged Rust accuracy contracts", results=[])
    start = time.monotonic()
    for preset in args.presets:
        for case in cases:
            reference = args.report / f"{preset}-{case}-cpu.json"
            targets = ["cpu", "gpu"]
            if args.cpu_reference_dir is not None:
                original = args.cpu_reference_dir / f"{preset}-{case}-cpu.json"
                original_input = args.cpu_reference_dir / "corpus" / f"{case}.png"
                if digest(original_input) != digest(args.report / "corpus" / f"{case}.png"):
                    raise RuntimeError("CPU reference input differs from the current corpus")
                reference.write_bytes(original.read_bytes())
                manifest.setdefault("cpu_reference_sha256", {})[f"{preset}-{case}"] = digest(original)
                targets = ["gpu"]
            for target in targets:
                remaining = args.budget_seconds - (time.monotonic() - start)
                if remaining <= 0:
                    raise RuntimeError("declared qualification budget exhausted")
                label = f"{preset}-{case}-{target}"
                command = [str(args.executable.resolve()), target, str((args.report / "corpus" / f"{case}.png").resolve()), "1", "1", "2", preset]
                env = dict(os.environ, SIMD_OCR_BENCH_MEASUREMENTS="0", SIMD_OCR_BENCH_CAPTURE="1", SIMD_OCR_BENCH_PROFILE="0",
                           SIMD_OCR_BENCH_MODELS=str(args.models.resolve()), SIMD_OCR_BENCH_WAVE=str(args.wave),
                           SIMD_OCR_BENCH_CLASSIFICATION_BATCH=str(args.classification_batch),
                           SIMD_OCR_BENCH_MATRIX=args.matrix if target == "gpu" else "f32",
                           SIMD_OCR_BENCH_STORAGE="f16" if target == "gpu" and args.precision == "mixed-f16" else "f32")
                for key in ["SIMD_OCR_BENCH_REFERENCE", "SIMD_OCR_BENCH_RGB", "SIMD_OCR_BENCH_WIDTH", "SIMD_OCR_BENCH_HEIGHT"]:
                    env.pop(key, None)
                if target == "gpu": env["SIMD_OCR_BENCH_REFERENCE"] = str(reference.resolve())
                run = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", timeout=remaining)
                (args.report / f"{label}-run.json").write_text(run.stdout, encoding="utf-8")
                (args.report / f"{label}.log").write_text(run.stderr, encoding="utf-8")
                record = dict(label=label, command=command, returncode=run.returncode, errors=[])
                if run.returncode:
                    record["errors"].append(run.stderr)
                else:
                    result = json.loads(run.stdout)
                    record["result"] = result
                    record["errors"].extend(result.get("errors", []))
                    if not result.get("qualified"): record["errors"].append("accuracy qualification failed")
                    if target == "cpu":
                        reference.write_text(json.dumps(result["outputs"][0], indent=2, ensure_ascii=False), encoding="utf-8")
                    else:
                        for stats in result["cold_statistics"]:
                            if (stats is None or stats.get("backend") != executor or stats.get("cpu_compute_stages") != []
                                    or stats.get("diagnostic_readback_bytes") != 0
                                    or stats.get("dispatches", 0) == 0
                                    or stats.get("native_graph_dispatches", 0) + stats.get("native_resident_dispatches", 0) != stats.get("dispatches")):
                                record["errors"].append(f"{executor} residency contract")
                manifest["results"].append(record)
                (args.report / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
                print(label, "FAIL" if record["errors"] else "PASS", flush=True)
                if record["errors"]: return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
