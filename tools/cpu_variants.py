#!/usr/bin/env python3
"""Validate intermediate stages, Portable4 and multicore outputs from a full bundle.

No C# or Rust scalar inference is repeated. The original verified 149-case
bundle, raw inputs, new captures and first-divergence reports are preserved.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import cpu_parity as gate


VARIANTS = (("simd-c1", "simd", "auto", 1), ("nhwc-c1", "nhwc", "auto", 1),
            ("portable4-c1", "auto", "portable4", 1), ("auto-c2", "auto", "auto", 2),
            ("auto-c4", "auto", "auto", 4), ("auto-c8", "auto", "auto", 8))


def build_matrix(backends: str) -> tuple:
    selected = [] if not backends else backends.split(",")
    allowed = {"avx", "avx-fma", "avx2-fma", "avx512"}
    if len(set(selected)) != len(selected) or any(value not in allowed for value in selected):
        raise ValueError("--backends must be a unique comma-separated selection of avx,avx-fma,avx2-fma,avx512")
    return VARIANTS + tuple((f"{backend}-c1", "auto", backend, 1) for backend in selected)


def current_host() -> dict:
    processor = os.environ.get("PROCESSOR_IDENTIFIER") or platform.processor()
    if not processor and Path("/proc/cpuinfo").is_file():
        processor = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                          if line.startswith("model name")), "unknown")
    return {"host_name": platform.node(), "cpu_identity": processor,
            "operating_system": platform.platform(), "architecture": platform.machine()}


def verify_host(environment: dict) -> dict:
    host = current_host()
    if any(environment.get(name) != value for name, value in host.items()):
        gate.fail("Variant validation must use the original host/CPU/OS/architecture identity.")
    return host


def numerical_tolerances(locked: dict, exact: bool) -> dict:
    result = locked.copy()
    if exact:
        for name in ("tensor_atol", "tensor_rtol", "score_atol", "coordinate_atol"):
            result[name] = 0.0
    return result


def verify_current(executable: Path, provenance: dict) -> None:
    if provenance.get("provenance") != "built_by_driver":
        gate.fail("The baseline must identify an executable built by the maintained driver.")
    if gate.sha256(executable) != provenance["executable_sha256"]:
        gate.fail("The built parity executable changed; create a new verified full bundle.")
    if gate.rust_sources() != provenance["sources"]:
        gate.fail("Rust sources changed after the verified bundle; rebuild and repeat its full gate.")


def read_identity(bundle: Path, lock: dict) -> dict:
    metadata = gate.read_json(bundle / "run.json")
    report = gate.read_json(bundle / "report.json")
    if metadata.get("schema_version") != 1 or report.get("schema_version") != 1 or report.get("passed") is not True:
        gate.fail("A successful schema-1 full parity bundle is required.")
    if metadata.get("quick") is not False or metadata.get("threads") != 1 or metadata.get("backend") not in (None, "auto"):
        gate.fail("The baseline must be the full 149-case native Auto gate at one thread.")
    if metadata.get("batch_only", False) or metadata.get("ocr_batch_size", 1) != 1:
        gate.fail("The variant baseline must retain the default single-line OCR batching policy.")
    if metadata.get("reference_lock_sha256") != gate.sha256(gate.LOCK) or metadata.get("reference_revision") != lock["revision"]:
        gate.fail("The original bundle's pinned reference identity changed.")
    if metadata.get("model_hashes") != lock["models"]:
        gate.fail("The original bundle's model/dictionary identities changed.")
    exporter = metadata["exporter_build"]
    if exporter.get("provenance") != "built_by_driver" or exporter.get("reference_revision") != lock["revision"]:
        gate.fail("The baseline requires verified pinned C# exporter provenance.")
    gate.verify_frozen_exporter(bundle, exporter, lock)
    corpus = gate.read_json(bundle / "corpus" / "corpus.json")
    required = {f"graph/{name}/{index}" for name in gate.GRAPH_NODE_COUNTS for index in range(3)}
    for image in corpus["images"]:
        id_ = image["id"]
        required.update(f"preprocess/{kind}/{id_}" for kind in ("det", "cls", "rec"))
        required.add(f"crop/{id_}")
        if image["public_case"]:
            required.add(f"classify/{id_}")
            for rec, det in (("rec", "det"), ("small_rec", "small_det"), ("medium_rec", "medium_det")):
                required.update((f"recognize/{rec}/{id_}", f"detect/{det}/{id_}"))
                if image["e2e_case"]:
                    required.add(f"ocr/{rec}/{id_}")
    if len(required) != 149:
        gate.fail("The original corpus does not describe the reviewed full 149-case workload.")
    for name in ("csharp-reference", "csharp-auto", "rust-reference", "rust-auto"):
        capture = gate.read_json(bundle / name / "capture.json")
        if capture.get("schema_version") != 1 or len(capture.get("cases", [])) != 149:
            gate.fail(f"The full baseline is missing required cases: {name}; expected 149.")
        if {case["id"] for case in capture["cases"]} != required:
            gate.fail(f"The baseline case IDs differ from the required complete corpus: {name}.")
        if capture["metadata"]["threads"] != 1:
            gate.fail(f"The baseline has a different effective thread budget: {name}.")
        engine, mode = name.split("-", 1)
        details = capture["metadata"]
        if capture.get("engine") != engine or capture.get("mode") != mode or details.get("corpus_sha256") != metadata["corpus_sha256"]:
            gate.fail(f"Capture-level engine/mode/corpus identity differs from the full bundle: {name}.")
        identity = "rust_build" if engine == "rust" else "exporter_build"
        if details.get(identity) != metadata[identity]:
            gate.fail(f"Capture-level build provenance differs from the full bundle: {name}.")
        if engine == "rust" and details.get("benchmark_environment") != metadata["benchmark_environment"]:
            gate.fail(f"Capture-level host/compiler/budget identity differs from the full bundle: {name}.")
        if engine == "csharp":
            policy = (details.get("hardware_intrinsics"), details.get("nhwc"), details.get("no_fuse"))
            expected = ("0", "0", "1") if mode == "reference" else ("1", "1", "0")
            if policy != expected:
                gate.fail(f"Capture-level C# CPU policies differ from the declared mode: {name}.")
    auto = gate.read_json(bundle / "rust-auto" / "capture.json")
    if auto["metadata"]["compile_options"] != {"kernels": "auto", "layout": "Auto", "fusion": True}:
        gate.fail("The baseline Rust Auto configuration is not the optimized default.")
    reference = gate.read_json(bundle / "rust-reference" / "capture.json")
    if reference["metadata"]["compile_options"] != {"kernels": "scalar", "layout": "Nchw", "fusion": False}:
        gate.fail("The baseline Rust reference configuration is not the scalar oracle.")
    return metadata


def verify_inputs(models: Path, corpus: Path, lock: dict, expected_corpus: str) -> None:
    for name, expected in lock["models"].items():
        if gate.sha256(models / name) != expected:
            gate.fail(f"Changed or missing variant model/dictionary: {name}.")
    gate.validate_corpus(corpus)
    if gate.sha256(corpus / "corpus.json") != expected_corpus:
        gate.fail("The original raw corpus manifest changed.")


def verify_original(bundle: Path, tolerances: dict) -> dict:
    pairs = (("csharp-reference", "rust-reference"), ("csharp-auto", "rust-auto"),
             ("rust-reference", "rust-auto"))
    comparisons = {f"{left}_vs_{right}": gate.compare_bundles(bundle / left, bundle / right, tolerances)
                   for left, right in pairs}
    imports = gate.compare_imports(bundle / "csharp-reference", bundle / "rust-reference")
    return {"passed": imports["passed"] and all(value["passed"] for value in comparisons.values()), "imports": imports, "comparisons": comparisons}


def verify_produced_capture(capture: dict, mode: str, backend: str, threads: int, provenance: dict,
                            host: dict, source_run: str, command: list[str], require_old_block: bool) -> None:
    metadata = capture["metadata"]
    expected = {"correctness_only": True, "ocr_batch_size": 1, "requested_backend": backend,
                "os": "windows", "architecture": "x86_64", "profiling_enabled": True,
                "images_decoded_inside_timing": False}
    if any(metadata.get(key) != value for key, value in expected.items()):
        gate.fail("Captured raw producer controls differ from the verified Windows correctness configuration")
    block = metadata.get("variant_validation")
    expected_block = {"mode": mode, "requested_backend": backend, "threads": threads, "rust_build": provenance,
                      "source_run_sha256": source_run, "command": command, "validation_host": host,
                      "effective_threads": threads, "timings_are_diagnostic": True}
    if (require_old_block and block is None) or (block is not None and block != expected_block):
        gate.fail("Retained original producer validation differs from its producing report")


def verify_producing_command(previous: dict, expected: list[str], directory: Path) -> None:
    if (previous.get("command") != expected or Path(previous["capture_directory"]).resolve() != directory
            or previous.get("inference_executed_in_this_run") is False):
        gate.fail("Retained producing command/input paths/controls differ; use the original producing run")


def capture_variant(command: list[str], directory: Path, baseline: Path, mode: str, backend: str, threads: int,
                    provenance: dict, host: dict, source_run_sha256: str, lock: dict, producing_proof: dict | None = None) -> dict:
    if producing_proof is None:
        gate.run(command)
    capture_path = directory / "capture.json"
    capture = gate.read_json(capture_path)
    producing_command = producing_proof["command"] if producing_proof else command
    verify_produced_capture(capture, mode, backend, threads, provenance, host, source_run_sha256,
                            producing_command, bool(producing_proof and producing_proof["reported_status"] == "passed"))
    expected_options = {"kernels": backend.replace("-", "_"), "layout": "Nchw" if mode == "simd" else "Auto", "fusion": mode == "auto"}
    if (capture.get("schema_version") != 1 or capture.get("engine") != "rust" or capture.get("mode") != mode
            or len(capture.get("cases", [])) != 149 or capture["metadata"]["threads"] != threads
            or capture["metadata"]["compile_options"] != expected_options
            or capture["metadata"]["corpus_sha256"] != gate.sha256(baseline / "corpus" / "corpus.json")):
        gate.fail(f"Variant did not honor its complete workload/backend/budget/corpus: {directory.name}.")
    if producing_proof is not None:
        shutil.copy2(capture_path, directory / "produced-capture.json")
    validation = {
        "mode": mode, "requested_backend": backend, "threads": threads, "rust_build": provenance,
        "source_run_sha256": source_run_sha256, "command": command,
        "validation_host": host, "effective_threads": threads, "timings_are_diagnostic": True}
    if producing_proof is not None:
        validation.update({"revalidation_only": True, "producing_proof": producing_proof,
            "command": producing_proof["command"], "revalidated_by_driver_sha256": gate.sha256(Path(__file__))})
    capture["metadata"]["variant_revalidation" if producing_proof else "variant_validation"] = validation
    gate.write_json(capture_path, capture)
    exact = mode == "auto" and backend == "auto" and threads > 1
    tolerances = numerical_tolerances(lock["tolerances"], exact=exact)
    comparisons = {"native_auto_c1": gate.compare_bundles(baseline / "rust-auto", directory, tolerances)}
    if not exact:
        comparisons["scalar_oracle"] = gate.compare_bundles(baseline / "rust-reference", directory, lock["tolerances"])
    return {"cases": 149, "tolerances": tolerances,
            "passed": all(value["passed"] for value in comparisons.values()), "comparisons": comparisons}


def verify_reuse_identity(prior: dict, original: dict, provenance: dict, host: dict, source_run: str, lock: dict) -> None:
    if (prior.get("schema_version") != 1 or prior.get("source_run_sha256") != source_run
            or prior.get("rust_build") != provenance or prior.get("validation_host") != host
            or prior.get("model_hashes") != lock["models"] or prior.get("corpus_sha256") != original["corpus_sha256"]
            or prior.get("reference_lock_sha256") != gate.sha256(gate.LOCK)
            or prior.get("comparison_helper_sha256") != gate.sha256(Path(gate.__file__))):
        gate.fail("Retained variants have different executable/source/host/model/corpus/comparator provenance")


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True, help="Successful fresh full one-thread cpu_parity.py output")
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=gate.ROOT / "target" / "cpu-variants")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--backends", default="", help="Additional forced ISA gates: avx,avx-fma,avx2-fma,avx512. Unavailable ISAs fail; request only on eligible hardware.")
    parser.add_argument("--reuse-captures", type=Path, help="Revalidate retained complete captures from the same verified build; missing matrix entries are executed. Original producer provenance is preserved.")
    args = parser.parse_args(argv)
    if args.iterations < 1:
        parser.error("iterations must be positive")
    try:
        matrix = build_matrix(args.backends)
    except ValueError as error:
        parser.error(str(error))
    source, out, models = args.bundle.resolve(), args.output.resolve(), args.models_dir.resolve()
    tooling = {"driver_source_sha256": gate.sha256(Path(__file__)),
               "comparison_helper_sha256": gate.sha256(Path(gate.__file__))}
    if source == out or source.is_relative_to(out) or out.is_relative_to(source):
        gate.fail("Variant output must be outside the original bundle, with no ancestor overlap.")
    if out.exists() and any(out.iterdir()):
        gate.fail("Variant output must be empty; preserve previous runs in their own directories.")
    lock = gate.read_json(gate.LOCK)
    original = read_identity(source, lock)
    host = verify_host(original["benchmark_environment"])
    provenance = original["rust_build"]
    reuse = args.reuse_captures.resolve() if args.reuse_captures else None
    prior = None
    prior_hash = None
    if reuse is not None:
        if reuse == out or reuse.is_relative_to(out) or out.is_relative_to(reuse):
            gate.fail("Retained variants and fresh output must not overlap")
        prior = gate.read_json(reuse / "report.json")
        prior_hash = gate.sha256(reuse / "report.json")
        verify_reuse_identity(prior, original, provenance, host, gate.sha256(source / "run.json"), lock)
    executable = gate.ROOT / "target" / "release" / "examples" / ("cpu_parity.exe" if os.name == "nt" else "cpu_parity")
    verify_current(executable, provenance)
    verify_inputs(models, source / "corpus", lock, original["corpus_sha256"])
    out.mkdir(parents=True, exist_ok=True)
    baseline = out / "baseline"
    print(f"Preserving complete replay baseline: {baseline}", flush=True)
    shutil.copytree(source, baseline)
    gate.archive_sources(gate.source_snapshot(gate.ROOT, [Path(__file__).resolve(), Path(gate.__file__).resolve(), gate.LOCK]), out / "sources" / "driver")
    if reuse is not None:
        (out / "reused-run").mkdir()
        shutil.copy2(reuse / "report.json", out / "reused-run" / "report.json")
        archive = reuse / "sources" / "driver"
        snapshot = gate.read_json(archive / "snapshot.json")
        if (snapshot.get("files", {}).get("tools/cpu_variants.py") != prior["driver_source_sha256"]
                or snapshot["files"].get("tools/cpu_parity.py") != prior["comparison_helper_sha256"]):
            gate.fail("Retained producing driver archive differs from its report")
        gate.verify_source_archive(snapshot, archive)
        shutil.copytree(archive, out / "reused-run" / "sources" / "driver")
        gate.verify_source_archive(snapshot, out / "reused-run" / "sources" / "driver")
    verification = verify_original(baseline, lock["tolerances"])
    verification.update({**tooling, "required_payload_schema": "pinned full case IDs, typed public/tensor payloads, complete initial logical traces"})
    gate.write_json(out / "source-verification.json", verification)
    if not verification["passed"]:
        gate.fail(f"The preserved original bundle no longer passes: {out / 'source-verification.json'}")
    started = time.time()
    report = {"schema_version": 1, "complete": False, "passed": False, "variants": [],
              "source_bundle": str(source), "source_run_sha256": gate.sha256(baseline / "run.json"),
              "reference_lock_sha256": gate.sha256(gate.LOCK), "rust_build": provenance,
              **tooling,
              "model_hashes": lock["models"], "corpus_sha256": original["corpus_sha256"],
              "validation_host": host, "baseline_environment": original["benchmark_environment"],
              "timings_are_diagnostic": True,
              "numerical_contract": "Native Auto parallelism uses zero numeric tolerances; intermediate SIMD/NCHW, unfused NHWC and Portable4 use the reviewed reference lock against scalar and default Auto. Text, spans, order and shapes are exact."}
    if reuse is not None:
        report["retained_producing_run"] = {"path": str(reuse), "report_sha256": prior_hash,
                                           "driver_source_sha256": prior["driver_source_sha256"]}
    gate.write_json(out / "report.json", report)
    for name, mode, backend, threads in matrix:
        command = [str(executable), "--models-dir", str(models), "--corpus", str(baseline / "corpus"),
                   "--output", str(out / name), "--mode", mode, "--backend", backend,
                   "--threads", str(threads), "--iterations", str(args.iterations), "--correctness-only"]
        result = {"name": name, "mode": mode, "backend": backend, "threads": threads, "command": command, "capture_directory": str(out / name),
                  "rust_build": provenance, "status": "running", "passed": False}
        report["variants"].append(result)
        gate.write_json(out / "report.json", report)
        print(f"Validating {name}: all 149 cases", flush=True)
        try:
            verify_host(original["benchmark_environment"])
            verify_current(executable, provenance)
            verify_inputs(models, baseline / "corpus", lock, original["corpus_sha256"])
            if (gate.sha256(gate.LOCK) != report["reference_lock_sha256"]
                    or gate.sha256(Path(__file__)) != report["driver_source_sha256"]
                    or gate.sha256(Path(gate.__file__)) != report["comparison_helper_sha256"]):
                gate.fail("Variant driver/comparator/reference lock changed during validation.")
            proof = None
            previous = next((entry for entry in prior["variants"] if entry["name"] == name), None) if prior else None
            if previous is not None:
                if (gate.sha256(reuse / "report.json") != prior_hash or previous.get("rust_build") != provenance
                        or (previous.get("mode"), previous.get("backend"), previous.get("threads")) != (mode, backend, threads)):
                    gate.fail("Retained variant entry/provenance changed")
                old_command = previous["command"]
                expected_command = [str(executable), "--models-dir", str(models), "--corpus", str(reuse / "baseline" / "corpus"),
                    "--output", str(reuse / name), "--mode", mode, "--backend", backend, "--threads", str(threads),
                    "--iterations", str(args.iterations), "--correctness-only"]
                verify_producing_command(previous, expected_command, reuse / name)
                if gate.sha256(reuse / "baseline" / "run.json") != report["source_run_sha256"]:
                    gate.fail("Retained producer's original baseline changed")
                verify_inputs(models, reuse / "baseline" / "corpus", lock, original["corpus_sha256"])
                old_capture = reuse / name / "capture.json"
                proof = {"source_capture": str(old_capture), "capture_sha256": gate.sha256(old_capture),
                         "producing_run_sha256": prior_hash, "producing_driver_source_sha256": prior["driver_source_sha256"],
                         "reported_status": previous["status"], "reported_error": previous.get("error"), "command": old_command}
                shutil.copytree(reuse / name, out / name)
                if gate.sha256(out / name / "capture.json") != proof["capture_sha256"]:
                    gate.fail("Retained capture changed while copying")
                result.update({"capture_origin": "retained_revalidated", "producing_proof": proof,
                               "command": old_command, "inference_executed_in_this_run": False})
            else:
                result.update({"capture_origin": "executed", "inference_executed_in_this_run": True})
            result.update(capture_variant(command, out / name, baseline, mode, backend, threads, provenance,
                                          host, report["source_run_sha256"], lock, proof))
            verify_current(executable, provenance)
            verify_inputs(models, baseline / "corpus", lock, original["corpus_sha256"])
            if (gate.sha256(gate.LOCK) != report["reference_lock_sha256"]
                    or gate.sha256(Path(__file__)) != report["driver_source_sha256"]
                    or gate.sha256(Path(gate.__file__)) != report["comparison_helper_sha256"]):
                gate.fail("Variant driver/comparator/reference lock changed during validation.")
            result["status"] = "passed" if result["passed"] else "failed"
        except (RuntimeError, OSError, subprocess.CalledProcessError, ValueError, KeyError, TypeError, AttributeError, OverflowError) as error:
            result.update({"status": "failed", "passed": False, "error": str(error)})
            report["elapsed_seconds"] = time.time() - started
            gate.write_json(out / "report.json", report)
            raise
        report["elapsed_seconds"] = time.time() - started
        gate.write_json(out / "report.json", report)
        print(f"{'PASS' if result['passed'] else 'FAIL'}: {name}, 149 cases", flush=True)
        if not result["passed"]:
            for label, comparison in result["comparisons"].items():
                for case in comparison["cases"]:
                    if not case["passed"]:
                        print(name, label, json.dumps(case, ensure_ascii=False), flush=True)
            return 1
    report["complete"] = True
    report["passed"] = True
    report["elapsed_seconds"] = time.time() - started
    gate.write_json(out / "report.json", report)
    print(f"PASS: {149 * len(matrix)} variant cases; report: {out / 'report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, subprocess.CalledProcessError, ValueError, KeyError, TypeError, AttributeError, OverflowError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
