"""Frozen, bounded physical-device OCR comparisons.

All implementations consume identical decoded RGB bytes. Cold outputs must pass
the public OCR contract before warm measurements. Instrumented calls are separate.
Build Rust and (when selected) tools/reference/Reference.csproj before running.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT.parent / "SimdPaddleOCR"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hashes(paths, base):
    return {str(p.relative_to(base)): digest(p) for p in sorted(paths) if p.is_file()}


def config(value):
    try:
        requests, slots, batch = map(int, value.split(":"))
        if not (1 <= requests <= 16 and 1 <= slots <= 16 and 1 <= batch <= 64):
            raise ValueError()
        return requests, slots, batch
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected REQUESTS:SLOTS:BATCH within 1..16:1..16:1..64") from error


def costs(result):
    stages, kernels = defaultdict(int), defaultdict(int)
    missing = 0
    for profile in result.get("profiles", []):
        pipeline = profile.get("pipeline") or {}
        for record in (pipeline.get("profile") or {}).get("kernels", []):
            ns = record.get("device_ns")
            if ns is None:
                missing += 1
                continue
            stages[record["stage"]] += ns
            kernels[(record["stage"], record["kernel"])] += ns
    return {"stage_ms": sorted(({"stage": s, "ms": ns / 1e6} for s, ns in stages.items()), key=lambda x: -x["ms"]),
            "kernel_ms": sorted(({"stage": s, "kernel": k, "ms": ns / 1e6} for (s, k), ns in kernels.items()), key=lambda x: -x["ms"]),
            "missing_timestamp_records": missing,
            "scope": "separate instrumented calls, summed device durations; not concurrent wall latency"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--presets", nargs="+", choices=["tiny", "small", "medium"], default=["tiny"])
    parser.add_argument("--cases", nargs="+", default=["typical"])
    parser.add_argument("--configs", nargs="+", type=config, default=[(1, 1, 2)])
    parser.add_argument("--targets", nargs="+", choices=["cpu", "gpu"], default=["gpu"])
    parser.add_argument("--csharp", nargs="*", choices=["cpu", "gpu"], default=[])
    parser.add_argument("--csharp-reference-dir", type=Path,
                        help="Canonical PRESET-CASE-cpu|gpu.json C# references; cross-implementation equivalence must be checked separately")
    parser.add_argument("--matrix", choices=["f32", "f16"], default="f32")
    parser.add_argument("--storage", choices=["f32", "f16"], default="f32")
    parser.add_argument("--wave", type=int, choices=range(1, 17), default=1)
    parser.add_argument("--classification-batch", type=int, choices=range(1, 17), default=4)
    parser.add_argument("--measurements", type=int, choices=range(4), default=3)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--alignment", action="store_true")
    parser.add_argument("--budget-seconds", type=float, default=60)
    args = parser.parse_args()
    if args.budget_seconds <= 0:
        parser.error("positive execution budget required")
    executable = ROOT / ("target/release/examples/gpu_ocr_bench.exe" if os.name == "nt" else "target/release/examples/gpu_ocr_bench")
    inputs = [*filter(Path.is_file, (ROOT / "src").rglob("*")), ROOT / "Cargo.toml", ROOT / "Cargo.lock", ROOT / "build.rs", ROOT / "rust-toolchain.toml",
              ROOT / "examples/gpu_ocr_bench.rs", *filter(Path.is_file, (ROOT / "examples/support").rglob("*"))]
    if max(p.stat().st_mtime for p in inputs) > executable.stat().st_mtime:
        raise RuntimeError("release executable predates source changes; rebuild")
    args.report.mkdir(parents=True, exist_ok=False)
    frozen = args.report / executable.name
    shutil.copy2(executable, frozen)
    manifest = {"schema": 1, "configuration": vars(args) | {"corpus": str(args.corpus), "report": str(args.report),
                "csharp_reference_dir": str(args.csharp_reference_dir) if args.csharp_reference_dir else None},
                "executable_sha256": digest(frozen), "source_sha256": hashes(inputs, ROOT),
                "compiler": subprocess.check_output(["rustc", "-Vv"], text=True),
                "cargo": subprocess.check_output(["cargo", "-V"], text=True),
                "harness_sha256": digest(Path(__file__)), "runs": [],
                "warmup": "one cold invocation per configuration", "measurements": args.measurements,
                "profile_calls_per_configuration": "one separate instrumented call per request after uninstrumented measurements; same native schedule; dispatch device timestamps and separate host timings" if args.profile else 0,
                "model_sha256": hashes((REFERENCE / "models").glob("*.onnx"), REFERENCE),
                "dictionary_sha256": hashes((REFERENCE / "models").glob("*.txt"), REFERENCE)}
    if args.csharp:
        build = ROOT / "tools/reference/bin/Release/net10.0"
        reference_dll = build / "Sdcb.SimdPaddleOCR.GpuBench.dll"
        cs_inputs = [*filter(Path.is_file, (ROOT / "tools/reference").glob("*.cs*")),
                     *[p for p in (REFERENCE / "src").rglob("*") if p.is_file() and p.suffix in (".cs", ".csproj", ".props", ".targets", ".spv") and "obj" not in p.parts and "bin" not in p.parts],
                     *[p for directory in (REFERENCE, REFERENCE.parent) for p in directory.iterdir()
                       if p.is_file() and (p.suffix in (".props", ".targets") or p.name in ("global.json", "NuGet.Config", "nuget.config"))]]
        if max(p.stat().st_mtime for p in cs_inputs) > reference_dll.stat().st_mtime:
            raise RuntimeError("C# adapter predates source changes; rebuild")
        shutil.copytree(build, args.report / "csharp")
        manifest["csharp_source_sha256"] = hashes(cs_inputs, ROOT.parent)
        manifest["csharp_binary_sha256"] = hashes(build.rglob("*"), build)
        manifest["dotnet"] = subprocess.check_output(["dotnet", "--info"], text=True, encoding="utf-8")
    # The checkout may contain uncommitted implementation work. Hashes pin its
    # identity; the archive also preserves the actual sources for reproduction.
    archive_inputs = inputs + [Path(__file__), ROOT / "tools/full_gpu_validation.py"]
    with zipfile.ZipFile(args.report / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(archive_inputs)):
            archive.write(path, path.relative_to(ROOT.parent))
    manifest["source_archive_sha256"] = digest(args.report / "source.zip")
    def save():
        (args.report / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    save()
    from PIL import Image
    start = time.monotonic()
    seen_cs = set()
    for preset in args.presets:
        for case in args.cases:
            path = args.corpus / f"{case}.png"
            reference = args.corpus.parent / f"{preset}-{case}-cpu.json"
            if not reference.exists():
                raise RuntimeError(f"qualify CPU output first: {reference}")
            image = Image.open(path).convert("RGB")
            rgb = args.report / f"{case}.rgb"
            rgb.write_bytes(image.tobytes())
            for requests, slots, batch in args.configs:
                jobs = [("rust", mode) for mode in args.targets]
                for mode in args.csharp:
                    key = (preset, case, requests, batch, mode)
                    if key not in seen_cs:
                        jobs.append(("csharp", mode))
                        seen_cs.add(key)
                for implementation, mode in jobs:
                    remaining = args.budget_seconds - (time.monotonic() - start)
                    if remaining <= 0:
                        manifest["budget_exhausted"] = True
                        save()
                        raise RuntimeError("declared execution budget exhausted")
                    label = f"{preset}-{case}-{implementation}-{mode}-r{requests}-s{slots}-b{batch}"
                    env = dict(os.environ, SIMD_OCR_BENCH_REFERENCE=str(reference.resolve()),
                               SIMD_OCR_BENCH_RGB=str(rgb.resolve()), SIMD_OCR_BENCH_WIDTH=str(image.width), SIMD_OCR_BENCH_HEIGHT=str(image.height),
                               SIMD_OCR_BENCH_CAPTURE=str(int(args.alignment)), SIMD_OCR_BENCH_PROFILE=str(int(args.profile)),
                               SIMD_OCR_BENCH_MEASUREMENTS=str(args.measurements),
                               DOTNET_TieredCompilation="0")
                    if implementation == "rust":
                        env.update(SIMD_OCR_BENCH_MATRIX=args.matrix if mode == "gpu" else "f32",
                                   SIMD_OCR_BENCH_STORAGE=args.storage if mode == "gpu" else "f32",
                                   SIMD_OCR_BENCH_WAVE=str(args.wave),
                                   SIMD_OCR_BENCH_CLASSIFICATION_BATCH=str(args.classification_batch))
                        command = [str(frozen.resolve()), mode, str(path.resolve()), str(requests), str(slots), str(batch), preset]
                    else:
                        command = ["dotnet", str((args.report / "csharp/Sdcb.SimdPaddleOCR.GpuBench.dll").resolve()), "--ocr-bench", str((REFERENCE / "models").resolve()), preset, mode, str(rgb.resolve()), str(image.width), str(image.height), str(requests), str(batch)]
                    selected_reference = reference
                    reference_scope = "rust-cpu"
                    if implementation == "csharp" and args.csharp_reference_dir:
                        selected_reference = args.csharp_reference_dir / f"{preset}-{case}-{mode}.json"
                        reference_scope = "implementation-cold-output; cross-implementation differences checked separately"
                    env["SIMD_OCR_BENCH_REFERENCE"] = str(selected_reference.resolve())
                    env["SIMD_OCR_BENCH_REFERENCE_SCOPE"] = reference_scope
                    record = {"label": label, "implementation": implementation, "command": command,
                              "input_sha256": digest(path), "rgb_sha256": digest(rgb), "reference_sha256": digest(selected_reference),
                              "reference_scope": reference_scope}
                    try:
                        run = subprocess.run(command, cwd=ROOT, env=env, text=True, encoding="utf-8", capture_output=True, timeout=remaining)
                        (args.report / f"{label}.json").write_text(run.stdout, encoding="utf-8")
                        (args.report / f"{label}.log").write_text(run.stderr, encoding="utf-8")
                        record["returncode"] = run.returncode
                        if run.returncode:
                            record["errors"] = [run.stderr]
                        else:
                            record["result"] = json.loads(run.stdout)
                            record["errors"] = record["result"].get("errors", [])
                            if not record["result"].get("qualified"):
                                record["errors"].append("accuracy qualification failed")
                            if implementation == "rust" and args.profile:
                                record["costs"] = costs(record["result"])
                    except subprocess.TimeoutExpired as error:
                        record["errors"] = [f"execution exceeded remaining {remaining:.2f}s budget"]
                        record["timeout"] = True
                        (args.report / f"{label}.log").write_bytes((error.stderr or b"") if isinstance(error.stderr, bytes) else (error.stderr or "").encode())
                    manifest["runs"].append(record)
                    manifest["elapsed_seconds"] = time.monotonic() - start
                    save()
                    result = record.get("result", {})
                    print(label, "FAIL" if record["errors"] else "PASS", result.get("median_group_ms"), flush=True)
                    if record.get("timeout"):
                        raise RuntimeError("declared execution budget exhausted")
                    if record["errors"]:
                        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
