"""Bounded physical-Mac comparison with parent-measured cold start.

prepare needs Pillow; measurement and analysis use only the Python standard library.
No automatic builds, warmups, retries, downloads, or sampling growth.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import selectors
import shutil
import signal
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT.parent / "SimdPaddleOCR"
CSHARP_REVISION = "33ff8a81672452041d110e8ba52dcd2639e54228"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def git(directory, *args):
    return subprocess.check_output(["git", "-C", str(directory), *args], text=True).strip()


def prepare(args):
    from PIL import Image, __version__
    args.output.mkdir(parents=True, exist_ok=False)
    dataset = json.loads((args.corpus / "manifest.json").read_text())
    if len(dataset) != 50 or len({row["id"] for row in dataset}) != 50:
        raise ValueError("the agreed XFUND split must have exactly 50 unique pages")
    # Preserve official numerical image order, independent of directory iteration.
    dataset.sort(key=lambda row: int(row["id"].rsplit("_", 1)[1]))
    image = Image.open(args.sample).convert("RGB")
    rgb = args.output / "sample.rgb"
    rgb.write_bytes(image.tobytes())
    entries = [dict(id="sample", image=str(args.sample.resolve()), rgb=str(rgb.resolve()),
                    width=image.width, height=image.height), *dataset]
    for entry in entries:
        raw = Path(entry["rgb"])
        if raw.stat().st_size != entry["width"] * entry["height"] * 3:
            raise ValueError(f"invalid RGB size: {raw}")
        entry["image_sha256"] = sha(entry["image"])
        entry["rgb_sha256"] = sha(raw)
    save(args.output / "inputs.json", entries)
    save(args.output / "provenance.json", dict(pillow=__version__,
         annotations=str((args.corpus / "zh.val.json").resolve()),
         annotations_sha256=sha(args.corpus / "zh.val.json"),
         dataset_manifest_sha256=sha(args.corpus / "manifest.json")))


def invoke(command, env, report, label, timeout):
    """Read readiness from an unbuffered pipe, independent of result serialization."""
    started = time.perf_counter_ns()
    first_ready_ms = None
    records, markers = [], []
    with (report / f"{label}.stderr.log").open("wb") as errors:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors, env=env,
                                   cwd=ROOT, start_new_session=True, bufsize=0)
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        pending = b""
        try:
            with (report / f"{label}.stdout.log").open("wb") as raw:
                while selector.get_map():
                    remaining = timeout - (time.perf_counter_ns() - started) / 1e9
                    if remaining <= 0:
                        raise TimeoutError(f"{label}: exceeded {timeout:.1f}s")
                    for key, _ in selector.select(min(remaining, 1)):
                        chunk = os.read(key.fd, 65536)
                        received = time.perf_counter_ns()
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        raw.write(chunk)
                        pending += chunk
                        while b"\n" in pending:
                            line, pending = pending.split(b"\n", 1)
                            if line.startswith(b"RESULT_READY "):
                                _, image_id, phase, ms = line.decode().split()
                                markers.append(dict(id=image_id, phase=phase, ms=float(ms)))
                                if first_ready_ms is None:
                                    first_ready_ms = (received - started) / 1e6
                            elif line.strip():
                                records.append(json.loads(line))
                process.wait(timeout=max(.001, timeout - (time.perf_counter_ns() - started) / 1e9))
            if process.returncode or pending:
                raise RuntimeError(f"{label}: exit {process.returncode}, trailing bytes {len(pending)}; see stderr")
        finally:
            selector.close()
            process.stdout.close()
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    images = [row for row in records if row["kind"] == "image"]
    summaries = [row for row in records if row["kind"] == "summary"]
    if first_ready_ms is None or len(summaries) != 1 or not summaries[0]["backend_verified"]:
        raise RuntimeError(f"{label}: missing ready signal or verified Metal summary")
    if len(images) != len(markers) or len(images) != summaries[0]["calls"]:
        raise RuntimeError(f"{label}: output count differs from readiness markers")
    for row, marker in zip(images, markers):
        if any(row[k] != marker[k] for k in ("id", "phase", "ms")) or not math.isfinite(row["ms"]) or row["ms"] <= 0:
            raise RuntimeError(f"{label}: mismatched or invalid measurement")
    stderr = (report / f"{label}.stderr.log").read_text()
    if "fallback" in stderr.lower():
        raise RuntimeError(f"{label}: fallback reported on stderr")
    return dict(label=label, command=command, launch_to_result_ms=first_ready_ms,
                elapsed_seconds=(time.perf_counter_ns() - started) / 1e9,
                summary=summaries[0], images=images)


def run(args):
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("physical Apple Silicon macOS required")
    if args.budget_seconds <= 0:
        raise ValueError("positive execution budget required")
    if git(REFERENCE, "rev-parse", "HEAD") != CSHARP_REVISION or git(REFERENCE, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("the pinned C# reference must be unchanged")
    entries = json.loads(args.inputs.read_text())
    if len(entries) != 51 or entries[0]["id"] != "sample":
        raise ValueError("run prepare first: sample + 50 XFUND pages required")
    for entry in entries:
        if sha(entry["rgb"]) != entry["rgb_sha256"] or sha(entry["image"]) != entry["image_sha256"]:
            raise RuntimeError("input changed: " + entry["id"])
    if args.implementation == "rust":
        build_inputs = [p for p in (ROOT / "src").rglob("*") if p.is_file()]
        build_inputs += [ROOT / p for p in ("Cargo.toml", "Cargo.lock", "build.rs", "rust-toolchain.toml", "examples/metal_ocr_bench.rs")]
        executable = args.rust_executable
    else:
        build_inputs = [p for directory in (ROOT / "tools/reference", REFERENCE / "src")
                        for p in directory.rglob("*") if p.is_file() and p.suffix in {".cs", ".csproj", ".props", ".targets", ".metal"}
                        and not {"obj", "bin"}.intersection(p.parts)]
        executable = ROOT / "tools/reference/bin/Release/net10.0/Sdcb.SimdPaddleOCR.GpuBench.dll"
    if max(p.stat().st_mtime_ns for p in build_inputs) > executable.stat().st_mtime_ns:
        raise RuntimeError("benchmark executable predates source changes; rebuild before measuring")
    args.report.mkdir(parents=True, exist_ok=False)
    frozen = args.report / "frozen"
    frozen.mkdir()
    if args.implementation == "csharp":
        source = ROOT / "tools/reference/bin/Release/net10.0"
        shutil.copytree(source, frozen / "csharp")
        command = [str(args.dotnet.resolve()), str((frozen / "csharp/Sdcb.SimdPaddleOCR.GpuBench.dll").resolve()), "--metal-bench"]
    else:
        shutil.copy2(args.rust_executable, frozen / "metal_ocr_bench")
        command = [str((frozen / "metal_ocr_bench").resolve())]
    sources = [ROOT / "Cargo.toml", ROOT / "Cargo.lock", ROOT / "build.rs", ROOT / "rust-toolchain.toml", Path(__file__), ROOT / "tools/metal_accuracy.py", ROOT / "tools/metal_comparison.py"]
    for directory in (ROOT / "src", ROOT / "examples", ROOT / "tools/reference", REFERENCE / "src"):
        sources.extend(p for p in directory.rglob("*") if p.is_file() and not {"obj", "bin"}.intersection(p.parts)
                       and p.suffix in {".rs", ".cs", ".csproj", ".props", ".targets", ".metal", ".json", ".comp", ".glsl"})
    # Freeze source contents as well as hashes; later edits cannot change this baseline.
    import zipfile
    with zipfile.ZipFile(frozen / "sources.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(set(sources)):
            archive.write(p, p.relative_to(ROOT.parent))
    binary_hashes = {str(p.relative_to(frozen)): sha(p) for p in frozen.rglob("*") if p.is_file()}
    manifest = dict(schema=1, implementation=args.implementation, csharp_revision=CSHARP_REVISION,
                    rust_revision=git(ROOT, "rev-parse", "HEAD"), rust_status=git(ROOT, "status", "--porcelain"),
                    platform=platform.platform(), machine=platform.machine(),
                    hardware=subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
                    binary_sha256=binary_hashes, sources_sha256={str(p.relative_to(ROOT.parent)): sha(p) for p in sorted(set(sources))},
                    inputs_sha256=sha(args.inputs), inputs=entries,
                    models_sha256={p.name: sha(p) for p in sorted(args.models.glob("*")) if p.is_file()},
                    protocol=dict(scope=args.scope, presets=args.presets, sample_launches=3, corpus_passes=2,
                                  requests=1, slots=1, recognition_batch=2, line_workers=4, detector_threads=8,
                                  preprocess_workers=16, rust_wave=16, rust_classification_batch=16,
                                  budget_seconds=args.budget_seconds, dotnet_runtime="default JIT settings",
                                  cold="parent process launch through first owned result readiness; includes raw input read, runtime, model and GPU setup",
                                  warm="decoded RGB through owned OCR result; excludes input IO, validation and serialization",
                                  cache_policy="fresh processes; no application persistent cache; OS filesystem and driver caches not purged"),
                    runs=[], complete=False)
    save(args.report / "manifest.json", manifest)
    sample_path, corpus_path = args.report / "sample.json", args.report / "corpus.json"
    save(sample_path, entries[:1])
    save(corpus_path, entries[1:])
    jobs = []
    for preset in args.presets:
        if args.scope == "preflight":
            jobs.append((preset, "cold", sample_path, 0, "preflight"))
        if args.scope in ("all", "sample"):
            jobs.extend((preset, "sample", sample_path, 0, f"sample-{i}") for i in range(3))
        if args.scope in ("all", "cold"):
            jobs.extend((preset, "cold", corpus_path, i, f"cold-{entries[i+1]['id']}") for i in range(50))
        if args.scope in ("all", "corpus"):
            jobs.append((preset, "corpus", corpus_path, 0, "corpus"))
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SIMD_OCR_", "OCR_DATASET_", "DOTNET_", "COMPlus_", "MTL_"))}
    env.update(DOTNET_ROOT=str(args.dotnet.resolve().parent), DOTNET_NOLOGO="1", DOTNET_CLI_TELEMETRY_OPTOUT="1")
    began = time.monotonic()
    try:
        for preset, mode, inputs, index, suffix in jobs:
            remaining = args.budget_seconds - (time.monotonic() - began)
            if remaining <= 0:
                raise TimeoutError("declared total execution budget exhausted")
            label = f"{preset}-{suffix}"
            call = [*command, str(args.models.resolve()), preset, str(inputs.resolve()), mode, str(index)]
            result = invoke(call, env, args.report, label, min(remaining, 240 if mode == "corpus" else 60))
            manifest["runs"].append(result)
            manifest["elapsed_seconds"] = time.monotonic() - began
            save(args.report / "manifest.json", manifest)
            print(f"{args.implementation} {label}: ready {result['launch_to_result_ms']:.2f} ms; {len(result['images'])} outputs", flush=True)
        manifest["complete"] = True
    except BaseException as error:
        manifest["failure"] = str(error)
        raise
    finally:
        manifest["elapsed_seconds"] = time.monotonic() - began
        save(args.report / "manifest.json", manifest)
    summarize(args.report)


def distribution(values):
    ordered = sorted(values)
    return dict(n=len(values), mean_ms=statistics.mean(values), p50_ms=statistics.median(values),
                p95_ms=ordered[math.ceil(.95 * len(ordered)) - 1])


def summary_rows(manifest):
    groups = {}
    for run in manifest["runs"]:
        preset = run["summary"]["preset"]
        mode = run["summary"]["mode"]
        if mode in ("cold", "sample"):
            group = "sample_cold" if run["images"][0]["id"] == "sample" else "corpus_cold"
            groups.setdefault((preset, group), []).append(run["launch_to_result_ms"])
        for row in run["images"]:
            if row["phase"] != "cold":
                group = ("sample_" if row["id"] == "sample" else "corpus_") + row["phase"]
                groups.setdefault((preset, group), []).append(row["ms"])
    return [dict(preset=preset, metric=metric, **distribution(values)) for (preset, metric), values in groups.items()]


def summarize(report):
    manifest = json.loads((report / "manifest.json").read_text())
    result = dict(implementation=manifest["implementation"], complete=manifest["complete"], rows=summary_rows(manifest))
    save(report / "summary.json", result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--corpus", type=Path, default=ROOT.parent / ".local/datasets/xfund-zh-val")
    prep.add_argument("--sample", type=Path, default=REFERENCE / "examples/sample.jpg")
    prep.add_argument("--output", type=Path, required=True)
    measure = sub.add_parser("run")
    measure.add_argument("--implementation", choices=["csharp", "rust"], required=True)
    measure.add_argument("--scope", choices=["preflight", "sample", "cold", "corpus", "all"], default="all")
    measure.add_argument("--presets", nargs="+", choices=["tiny", "small", "medium"], default=["tiny", "small", "medium"])
    measure.add_argument("--inputs", type=Path, required=True)
    measure.add_argument("--report", type=Path, required=True)
    measure.add_argument("--models", type=Path, default=REFERENCE / "models")
    measure.add_argument("--dotnet", type=Path, default=ROOT.parent / ".tools/dotnet/dotnet")
    measure.add_argument("--rust-executable", type=Path, default=ROOT / "target/release/examples/metal_ocr_bench")
    measure.add_argument("--budget-seconds", type=float, default=900)
    summary = sub.add_parser("summarize")
    summary.add_argument("report", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args)
    elif args.action == "run":
        run(args)
    else:
        summarize(args.report)


if __name__ == "__main__":
    main()
