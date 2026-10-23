"""Bounded physical-device comparison; one cold run and three warm runs per configuration."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["vulkan", "dx12", "metal"], required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=["cpu", "inference-only", "full"], default=["inference-only", "full"])
    args = parser.parse_args()
    executable = ROOT / ("target/release/examples/gpu_ocr_bench.exe" if os.name == "nt" else "target/release/examples/gpu_ocr_bench")
    inputs = [*filter(Path.is_file, (ROOT / "src").rglob("*")), ROOT / "Cargo.toml", ROOT / "Cargo.lock", ROOT / "rust-toolchain.toml", ROOT / "examples/gpu_ocr_bench.rs"]
    if max(p.stat().st_mtime for p in inputs) > executable.stat().st_mtime:
        raise RuntimeError("release executable predates source changes; rebuild")
    args.report.mkdir(parents=True, exist_ok=True)
    frozen = args.report / executable.name
    if frozen.exists():
        raise RuntimeError("use a fresh report directory; never overwrite measured versions")
    shutil.copy2(executable, frozen)
    manifest = {"executable_sha256": digest(executable), "compiler": subprocess.check_output(["rustc", "-Vv"],text=True),
                "build_inputs_sha256": {str(p.relative_to(ROOT)):digest(p) for p in inputs},
                "runs": [], "budget_seconds": 60, "warmup": 1, "measurements": 3}
    start = time.monotonic()
    for case in ["sparse", "typical", "dense"]:
        image = args.corpus / (case + ".png")
        for mode in args.modes:
            for requests in [1, 2]:
                remaining = 60 - (time.monotonic() - start)
                if remaining <= 0: raise RuntimeError("performance budget exhausted")
                label = f"{case}-{mode}-{requests}"
                command = [str(frozen.resolve()), args.backend, mode, str(image.resolve()), str(requests), str(requests), "2", "tiny"]
                reference = args.corpus.parent / f"tiny-{case}-cpu.json"
                if not reference.exists(): raise RuntimeError("qualify the tiny CPU corpus before benchmarking")
                env = dict(os.environ, SIMD_OCR_BENCH_REFERENCE=str(reference.resolve()))
                run = subprocess.run(command, cwd=ROOT, text=True, encoding="utf-8",capture_output=True,timeout=remaining,env=env)
                (args.report / (label + ".json")).write_text(run.stdout,encoding="utf-8")
                (args.report / (label + ".log")).write_text(run.stderr,encoding="utf-8")
                if run.returncode: raise RuntimeError(run.stderr)
                result = json.loads(run.stdout)
                manifest["runs"].append({"label":label,"command":command,"input_sha256":digest(image),"cpu_reference_sha256":digest(reference),"result":result})
                manifest["elapsed_seconds"] = time.monotonic() - start
                (args.report / "manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
                print(label, f"{result['median_group_ms']:.2f} ms; {result['images_per_second']:.2f} images/s",flush=True)
if __name__ == "__main__":
    main()
