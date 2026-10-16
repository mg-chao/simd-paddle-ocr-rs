#!/usr/bin/env python3
"""Build, capture and compare the pinned C# CPU oracle and the Rust implementation.

Only Python's standard library is required. A frozen decoded source and a
deterministic generator give both inference executables identical raw BGR bytes.
Every missing prerequisite, missing case, nonfinite value and mismatch is fatal.
"""
from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "tools" / "reference-lock.json"
# Normalized logical graph sizes at the pinned reference/model identities. A
# missing trace in both bundles must fail rather than compare equal by omission.
GRAPH_NODE_COUNTS = {"cls": 106, "det": 242, "small_det": 242, "medium_det": 228,
                     "rec": 159, "small_rec": 333, "medium_rec": 352}
RAW_GRAPH_NODE_COUNTS = {"cls": 143, "det": 242, "small_det": 242, "medium_det": 276,
                         "rec": 219, "small_rec": 481, "medium_rec": 508}


def fail(message: str) -> None:
    raise RuntimeError(message)


def read_json(path: Path):
    with path.open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_snapshot(root: Path, paths: list[Path]) -> dict:
    """Identify actual working-tree bytes, including uncommitted changes."""
    files = {path.relative_to(root).as_posix(): sha256(path) for path in sorted(set(paths))}
    if not files:
        fail("Cannot fingerprint an empty source set.")
    return {"sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(), "files": files}


def archive_sources(snapshot: dict, destination: Path) -> None:
    """Retain the actual dirty working-tree bytes that produced an executable."""
    destination.mkdir(parents=True)
    for name, expected in snapshot["files"].items():
        source = checked_child(ROOT, name)
        target = checked_child(destination, name)
        if sha256(source) != expected:
            fail(f"Source changed before archival: {name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if sha256(target) != expected or sha256(source) != expected:
            fail(f"Source changed while archiving: {name}")
    write_json(destination / "snapshot.json", snapshot)


def verify_source_archive(snapshot: dict, directory: Path) -> None:
    if read_json(directory / "snapshot.json") != snapshot:
        fail("Archived source manifest differs from build provenance")
    for name, expected in snapshot["files"].items():
        if sha256(checked_child(directory, name)) != expected:
            fail(f"Archived source file changed: {name}")


def verify_frozen_exporter(bundle: Path, provenance: dict, lock: dict) -> None:
    """A replay consumes historical binaries, not current exporter source files."""
    files = provenance.get("runtime_files")
    assembly = "Sdcb.SimdPaddleOCR.Tests.dll"
    if (provenance.get("provenance") != "built_by_driver" or provenance.get("reference_revision") != lock["revision"]
            or not provenance.get("sources", {}).get("files") or not isinstance(files, dict) or not files
            or files.get(assembly) != provenance.get("executable_sha256")):
        fail("Replay requires verified pinned C# exporter build and frozen runtime provenance")
    directory = bundle / "binaries" / "csharp"
    if not directory.is_dir() or {path.name for path in directory.iterdir() if path.is_file()} != set(files):
        fail("Frozen C# runtime is missing files or contains unrecorded files")
    for name, expected in files.items():
        if Path(name).name != name or sha256(checked_child(directory, name)) != expected:
            fail(f"Frozen C# runtime file changed: {name}")


def rust_sources() -> dict:
    files = [ROOT / name for name in ("Cargo.toml", "Cargo.lock", "rust-toolchain.toml", "README.md")]
    files.extend((ROOT / "examples").rglob("*.rs"))
    files.extend((ROOT / "src").rglob("*.rs"))
    files.extend(path for path in (ROOT / ".cargo").glob("*") if path.is_file())
    if (ROOT / "build.rs").is_file():
        files.append(ROOT / "build.rs")
    return source_snapshot(ROOT, files)


def exporter_sources() -> dict:
    files = [ROOT / name for name in ("tools/reference-exporter/ReferenceExporter.csproj",
        "tools/corpus-source/sample.bgr", "tools/corpus-source/sample.json", "tools/reference-lock.json")]
    files.extend((ROOT / "tools/reference-exporter").glob("*.cs"))
    return source_snapshot(ROOT, files)


def build_flags() -> dict:
    # Include encoded flags and profile overrides as well as the common RUSTFLAGS.
    names = {name for name in os.environ if name.startswith("CARGO_PROFILE_RELEASE_")}
    names.update(("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_BUILD_RUSTFLAGS", "CARGO_BUILD_TARGET"))
    flags = {name: os.environ.get(name, "") for name in sorted(names)}
    flags["manifest_sha256"] = sha256(ROOT / "Cargo.toml")
    flags["lock_sha256"] = sha256(ROOT / "Cargo.lock")
    flags["cargo_features"] = ["profiling"]
    cargo_home = Path(os.environ.get("CARGO_HOME", str(Path.home() / ".cargo")))
    configs = [directory / ".cargo" / name for directory in (ROOT, *ROOT.parents) for name in ("config", "config.toml")]
    configs.extend(cargo_home / name for name in ("config", "config.toml"))
    flags["cargo_configs"] = {str(path): sha256(path) for path in sorted(set(configs)) if path.is_file()}
    return flags


def run(command: list[str], *, env=None, capture=False) -> str:
    print("+ " + subprocess.list2cmdline(list(map(str, command))), flush=True)
    result = subprocess.run(command, cwd=ROOT, env=env, text=True,
                            encoding="utf-8",
                            stdout=subprocess.PIPE if capture else None,
                            stderr=subprocess.STDOUT if capture else None, check=True)
    return (result.stdout or "").strip()


def require_tool(name: str) -> str:
    found = shutil.which(name)
    if found is None:
        fail(f"Required executable '{name}' is missing from PATH; no parity tests were run.")
    return found


def reference_environment(mode: str, threads: int) -> dict:
    # Reference tuning/debug switches must not leak in from the developer's
    # shell. These are the pinned source's switch families and CLR ISA controls.
    prefixes = ("PPOCR_", "SIMD_OCR_", "DOTNET_ENABLE", "COMPLUS_ENABLE")
    env = {name: value for name, value in os.environ.items() if not name.upper().startswith(prefixes)}
    env.update({"PPOCR_NHWC": "0" if mode == "reference" else "1",
                "SIMD_OCR_NOFUSE": "1" if mode == "reference" else "0",
                "SIMD_OCR_BACKEND": "cpu",
                "DOTNET_EnableHWIntrinsic": "0" if mode == "reference" else "1",
                "DOTNET_PROCESSOR_COUNT": str(threads),
                "DOTNET_EnableAVX512F": "0", "DOTNET_EnableAVX512BW": "0"})
    return env


def checked_child(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        fail(f"Capture path escapes its bundle: {relative}")
    return path


def validate_corpus(root: Path) -> None:
    manifest = read_json(root / "corpus.json")
    images = manifest.get("images", [])
    if manifest.get("schema_version") != 1 or manifest.get("pixel_format") != "bgr24" or not images:
        fail("Unsupported or empty raw corpus.")
    if len({image["id"] for image in images}) != len(images):
        fail("Duplicate raw-corpus image IDs.")
    for image in images:
        path = checked_child(root, image["file"])
        if image["width"] <= 0 or image["height"] <= 0 or path.stat().st_size != image["width"] * image["height"] * 3:
            fail(f"Invalid raw BGR dimensions/length: {path}")
        if sha256(path) != image["sha256"]:
            fail(f"Raw corpus checksum mismatch: {path}")


def validate_inputs(reference: Path, models: Path, lock: dict) -> None:
    if not (reference / "src" / "Sdcb.SimdPaddleOCR" / "Sdcb.SimdPaddleOCR.csproj").is_file():
        fail(f"Reference source checkout is missing: {reference}")
    actual = run([require_tool("git"), "-C", str(reference), "rev-parse", "HEAD"], capture=True)
    if actual != lock["revision"]:
        fail(f"Reference revision is {actual}, expected {lock['revision']}.")
    dirty = run(["git", "-C", str(reference), "status", "--porcelain"], capture=True)
    if dirty:
        fail("Reference checkout has source changes; use a clean pinned checkout.\n" + dirty)
    for name, expected in lock["models"].items():
        path = models / name
        if not path.is_file():
            fail(f"Required model/dictionary is missing: {path}")
        if sha256(path) != expected:
            fail(f"SHA-256 mismatch for {path}; reference model identity is not negotiable.")
    sample = reference / lock["sample"]["path"]
    if not sample.is_file() or sha256(sample) != lock["sample"]["sha256"]:
        fail(f"Pinned sample image is missing or changed: {sample}")


def load_f32(root: Path, entry: dict) -> array.array:
    path = checked_child(root, entry["file"])
    raw = path.read_bytes()
    if len(raw) % 4:
        fail(f"Truncated little-endian F32 tensor: {path}")
    if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
        fail(f"Tensor checksum mismatch: {path}")
    data = array.array("f")
    data.frombytes(raw)
    if sys.byteorder != "little":
        data.byteswap()
    expected = math.prod(entry["shape"])
    if len(data) != expected:
        fail(f"Tensor shape/length mismatch: {path}: {len(data)} != {expected}")
    return data


def compare_tensor(a_root: Path, a: dict, b_root: Path, b: dict, tolerances: dict) -> dict:
    if a["shape"] != b["shape"]:
        return {"passed": False, "reason": "shape", "expected": a["shape"], "actual": b["shape"]}
    av, bv = load_f32(a_root, a), load_f32(b_root, b)
    if a["sha256"] == b["sha256"]:
        for index, value in enumerate(av):
            if not math.isfinite(value):
                return {"passed": False, "reason": "nonfinite", "index": index,
                        "expected": str(value), "actual": str(value)}
        return {"passed": True, "elements": len(av), "unequal": 0,
                "max_abs": 0.0, "max_relative": 0.0, "first_difference": None}
    first = None
    unequal = 0
    max_abs = 0.0
    max_relative = 0.0
    atol, rtol = tolerances["tensor_atol"], tolerances["tensor_rtol"]
    exact = atol == 0 and rtol == 0
    a_bits = memoryview(av).cast("B").cast("I") if exact else None
    b_bits = memoryview(bv).cast("B").cast("I") if exact else None
    for index, (x, y) in enumerate(zip(av, bv)):
        if not math.isfinite(x) or not math.isfinite(y):
            return {"passed": False, "reason": "nonfinite", "index": index,
                    "expected": str(x), "actual": str(y)}
        difference = abs(x - y)
        max_abs = max(max_abs, difference)
        max_relative = max(max_relative, difference / max(abs(x), abs(y), 1e-30))
        different = a_bits[index] != b_bits[index] if exact else x != y
        unequal += different
        if first is None and (different if exact else difference > atol + rtol * abs(x)):
            first = {"index": index, "expected": x, "actual": y, "abs_error": difference}
            if exact:
                first.update(expected_bits=f"0x{a_bits[index]:08x}", actual_bits=f"0x{b_bits[index]:08x}")
    return {"passed": first is None, "elements": len(av), "unequal": unequal,
            "max_abs": max_abs, "max_relative": max_relative, "first_difference": first}


def compare_result(expected, actual, tolerances: dict, path="result") -> list[dict]:
    errors = []
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(expected) != set(actual):
            return [{"path": path, "reason": "object keys", "expected": sorted(expected),
                     "actual": sorted(actual) if isinstance(actual, dict) else actual}]
        for key in expected:
            errors.extend(compare_result(expected[key], actual[key], tolerances, path + "." + key))
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            return [{"path": path, "reason": "length", "expected": len(expected),
                     "actual": len(actual) if isinstance(actual, list) else actual}]
        for index, (a, b) in enumerate(zip(expected, actual)):
            errors.extend(compare_result(a, b, tolerances, f"{path}[{index}]"))
    elif isinstance(expected, float) or (isinstance(expected, int) and not isinstance(expected, bool)
                                       and (".points[" in path or path.endswith("score") or path.endswith("ratio"))):
        coordinate = ".points[" in path
        tolerance = tolerances["coordinate_atol"] if coordinate else tolerances["score_atol"]
        if not isinstance(actual, (int, float)) or not math.isfinite(expected) or not math.isfinite(actual) or abs(expected - actual) > tolerance:
            errors.append({"path": path, "expected": expected, "actual": actual, "atol": tolerance})
    elif expected != actual:
        errors.append({"path": path, "expected": expected, "actual": actual})
    return errors


def first_exact_difference(expected, actual, path="import"):
    """Import structure and constant hashes have no numerical tolerance."""
    if type(expected) is not type(actual):
        return {"path": path, "expected": expected, "actual": actual}
    if isinstance(expected, dict):
        if set(expected) != set(actual):
            return {"path": path, "reason": "keys", "expected": sorted(expected), "actual": sorted(actual)}
        for key in sorted(expected):
            difference = first_exact_difference(expected[key], actual[key], path + "." + key)
            if difference:
                return difference
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            return {"path": path, "reason": "length", "expected": len(expected), "actual": len(actual)}
        for index, (left, right) in enumerate(zip(expected, actual)):
            difference = first_exact_difference(left, right, f"{path}[{index}]")
            if difference:
                return difference
    elif expected != actual:
        return {"path": path, "expected": expected, "actual": actual}
    return None


def compare_imports(reference: Path, actual: Path) -> dict:
    expected_files = {name + ".json" for name in GRAPH_NODE_COUNTS}
    for directory in (reference, actual):
        found = {path.name for path in (directory / "imports").glob("*.json")}
        if found != expected_files:
            fail(f"All seven raw/normalized import contracts are required: {directory}; found {sorted(found)}")
    models = []
    for name, nodes in GRAPH_NODE_COUNTS.items():
        left = read_json(reference / "imports" / (name + ".json"))
        right = read_json(actual / "imports" / (name + ".json"))
        for contract in (left, right):
            if (contract.get("schema_version") != 1 or set(contract) != {"schema_version", "raw", "normalized"}
                    or len(contract["normalized"].get("nodes", [])) != nodes
                    or len(contract["raw"].get("nodes", [])) != RAW_GRAPH_NODE_COUNTS[name]
                    or any(not contract[stage].get(field) for stage in ("raw", "normalized") for field in ("inputs", "outputs"))):
                fail(f"Incomplete import contract for {name}")
        difference = first_exact_difference(left, right, name)
        models.append({"model": name, "passed": difference is None, "first_difference": difference,
                       "reference_sha256": sha256(reference / "imports" / (name + ".json")),
                       "actual_sha256": sha256(actual / "imports" / (name + ".json"))})
    return {"passed": all(model["passed"] for model in models), "models": models}


def validate_case_payload(capture: dict, case: dict) -> None:
    id_ = case["id"]
    parts = id_.split("/")

    def tensor(value):
        if (not isinstance(value, dict) or not isinstance(value.get("file"), str) or not value["file"]
                or not isinstance(value.get("shape"), list) or not value["shape"]
                or not all(type(size) is int and size > 0 for size in value["shape"])
                or not isinstance(value.get("sha256"), str) or len(value["sha256"]) != 64):
            fail(f"Missing or invalid required tensor payload: {id_}.")

    def require(condition, field):
        if not condition:
            fail(f"Missing or invalid public field type: {id_}, {field}.")

    def number(value):
        try:
            return type(value) in (int, float) and math.isfinite(value)
        except OverflowError:
            return False

    def fields(value, strings=(), integers=(), numbers=()):
        for name in strings:
            require(isinstance(value.get(name), str), name)
        for name in integers:
            require(type(value.get(name)) is int and value[name] >= 0, name)
        for name in numbers:
            require(number(value.get(name)), name)

    def spans(values):
        require(isinstance(values, list), "ctc_spans")
        for span in values:
            require(isinstance(span, dict) and set(span) == {"text", "score", "start_column", "end_column"}, "ctc_spans[]")
            fields(span, strings=("text",), integers=("start_column", "end_column"), numbers=("score",))

    def box(value):
        require(isinstance(value, dict) and set(value) == {"points", "score"}, "box")
        require(number(value["score"]), "box.score")
        points = value["points"]
        require(isinstance(points, list) and len(points) == 4, "box.points")
        require(all(isinstance(point, list) and len(point) == 2 and all(number(coordinate) for coordinate in point)
                    for point in points), "box.points[]")

    if len(parts) == 3 and parts[0] == "graph" and parts[1] in GRAPH_NODE_COUNTS and parts[2] in ("0", "1", "2"):
        tensor(case.get("tensor"))
        if parts[2] == "0" and (capture.get("engine"), capture.get("mode")) != ("csharp", "auto"):
            nodes = case.get("nodes")
            if not isinstance(nodes, list) or len(nodes) != GRAPH_NODE_COUNTS[parts[1]]:
                fail(f"Missing or incomplete required logical graph trace: {id_}.")
            for index, node in enumerate(nodes):
                if (not isinstance(node, dict) or node.get("index") != index
                        or not isinstance(node.get("name"), str) or not isinstance(node.get("op_type"), str)
                        or not node["op_type"]):
                    fail(f"Missing or invalid logical node identity: {id_}, index {index}.")
                tensor(node.get("tensor"))
    elif ((len(parts) == 3 and parts[0] == "preprocess" and parts[1] in ("det", "cls", "rec"))
          or (len(parts) == 2 and parts[0] == "crop")):
        tensor(case.get("tensor"))
    else:
        result_keys = {
            "classify": {"label", "score", "resized_width"},
            "recognize": {"text", "score", "emitted_count", "resized_width", "tensor_width", "time_steps", "ctc_spans"},
            "detect": {"boxes", "resized_width", "resized_height", "width_ratio", "height_ratio"},
            "ocr": {"text", "detected_count", "detector_resized_width", "detector_resized_height", "lines"},
        }
        valid_id = ((len(parts) == 2 and parts[0] == "classify")
                    or (len(parts) == 3 and parts[0] in ("recognize", "ocr") and parts[1] in ("rec", "small_rec", "medium_rec"))
                    or (len(parts) == 3 and parts[0] == "detect" and parts[1] in ("det", "small_det", "medium_det")))
        if not valid_id or not parts[-1]:
            fail(f"Unknown required capture case schema: {id_}.")
        result = case.get("result")
        if not isinstance(result, dict) or set(result) != result_keys[parts[0]]:
            fail(f"Missing or invalid required public result payload: {id_}.")
        if parts[0] == "classify":
            fields(result, integers=("label", "resized_width"), numbers=("score",))
        elif parts[0] == "recognize":
            fields(result, strings=("text",), integers=("emitted_count", "resized_width", "tensor_width", "time_steps"), numbers=("score",))
            spans(result["ctc_spans"])
        elif parts[0] == "detect":
            fields(result, integers=("resized_width", "resized_height"), numbers=("width_ratio", "height_ratio"))
            require(isinstance(result["boxes"], list), "boxes")
            for value in result["boxes"]:
                box(value)
        else:
            fields(result, strings=("text",), integers=("detected_count", "detector_resized_width", "detector_resized_height"))
            require(isinstance(result["lines"], list), "lines")
            line_keys = {"detection", "text", "recognition_score", "classification_score", "classification_label",
                         "applied_rotation_degrees", "emitted_count", "recognition_content_width", "recognition_tensor_width",
                         "recognition_time_steps", "ctc_spans"}
            for line in result["lines"]:
                require(isinstance(line, dict) and set(line) == line_keys, "lines[]")
                fields(line, strings=("text",), integers=("classification_label", "applied_rotation_degrees", "emitted_count",
                       "recognition_content_width", "recognition_tensor_width", "recognition_time_steps"),
                       numbers=("recognition_score", "classification_score"))
                box(line["detection"])
                spans(line["ctc_spans"])


def required_case_ids(corpus: Path, quick: bool) -> set[str]:
    ids = {f"graph/{name}/{index}" for name in GRAPH_NODE_COUNTS for index in range(3)}
    for image in read_json(corpus / "corpus.json")["images"]:
        if quick and not image["quick"]:
            continue
        image_id = image["id"]
        ids.update(f"preprocess/{kind}/{image_id}" for kind in ("det", "cls", "rec"))
        ids.add(f"crop/{image_id}")
        if image["public_case"]:
            ids.add(f"classify/{image_id}")
            for rec, det in (("rec", "det"), ("small_rec", "small_det"), ("medium_rec", "medium_det")):
                ids.update((f"recognize/{rec}/{image_id}", f"detect/{det}/{image_id}"))
                if image["e2e_case"]:
                    ids.add(f"ocr/{rec}/{image_id}")
    if len(ids) != (67 if quick else 149):
        fail("Corpus no longer describes the reviewed complete quick/full workload")
    return ids


def compare_bundles(reference: Path, actual: Path, tolerances: dict, required_ids=None) -> dict:
    ref, got = read_json(reference / "capture.json"), read_json(actual / "capture.json")
    for capture in (ref, got):
        if (not isinstance(capture, dict) or capture.get("schema_version") != 1
                or not isinstance(capture.get("cases"), list)
                or not all(isinstance(case, dict) and isinstance(case.get("id"), str) and case["id"] for case in capture["cases"])):
            fail("Unsupported or malformed capture schema.")
    expected = {case["id"]: case for case in ref["cases"]}
    observed = {case["id"]: case for case in got["cases"]}
    if len(expected) != len(ref["cases"]) or len(observed) != len(got["cases"]):
        fail("Duplicate case IDs in capture.")
    if not expected or set(expected) != set(observed):
        fail(f"Capture case sets differ: missing={sorted(expected.keys() - observed.keys())}, extra={sorted(observed.keys() - expected.keys())}")
    if required_ids is not None and set(expected) != set(required_ids):
        fail(f"Missing or extra mandatory corpus cases: missing={sorted(required_ids - expected.keys())}, extra={sorted(expected.keys() - required_ids)}")
    for capture in (ref, got):
        for case in capture["cases"]:
            validate_case_payload(capture, case)
    required_graphs = {f"graph/{name}/{index}" for name in GRAPH_NODE_COUNTS for index in range(3)}
    if not required_graphs.issubset(expected):
        fail(f"Capture is missing required graph cases: {sorted(required_graphs - expected.keys())}")
    rows = []
    for id_, case in expected.items():
        other = observed[id_]
        row = {"id": id_, "passed": True}
        if "tensor" in case:
            row["tensor"] = compare_tensor(reference, case["tensor"], actual, other["tensor"], tolerances)
            row["passed"] &= row["tensor"]["passed"]
        if "result" in case:
            row["result_errors"] = compare_result(case["result"], other.get("result"), tolerances)
            row["passed"] &= not row["result_errors"]
        # The default C# graph is physically transformed; only scalar captures
        # carry logical node traces. A missing Rust diagnostic trace is fatal.
        if case.get("nodes"):
            nodes = case["nodes"]
            other_nodes = other.get("nodes", [])
            if len(nodes) != len(other_nodes):
                row["passed"] = False
                row["first_divergence"] = {"reason": "node count", "expected": len(nodes), "actual": len(other_nodes)}
            else:
                for node, other_node in zip(nodes, other_nodes):
                    if (node["index"], node["op_type"]) != (other_node["index"], other_node["op_type"]):
                        row["passed"] = False
                        row["first_divergence"] = {"reason": "node identity", "expected": node, "actual": other_node}
                        break
                    difference = compare_tensor(reference, node["tensor"], actual, other_node["tensor"], tolerances)
                    if not difference["passed"]:
                        row["passed"] = False
                        row["first_divergence"] = {"node_index": node["index"], "name": node["name"], "op_type": node["op_type"], **difference}
                        break
        rows.append(row)
    return {"passed": all(row["passed"] for row in rows), "cases": rows}


def compare_batch_bundles(reference: Path, actual: Path, tolerances: dict) -> dict:
    captures = [read_json(directory / "capture.json") for directory in (reference, actual)]
    required = {f"batch/{name}" for name in ("rec", "small_rec", "medium_rec")}
    rows = []
    typed = []
    for capture in captures:
        cases = capture.get("cases", [])
        if capture.get("schema_version") != 1 or len(cases) != 3 or {case.get("id") for case in cases} != required:
            fail("Batch gate requires all three recognition models")
        mapped = {}
        for case in cases:
            if set(case) != {"id", "image_ids", "results", "standalone"} or case["image_ids"] != ["sample", "tiny", "blank", "sample"]:
                fail("Missing batch payload or input ordering")
            for field in ("results", "standalone"):
                if not isinstance(case[field], list) or len(case[field]) != 4:
                    fail("Batch gate requires four independently decoded inputs")
                for result in case[field]:
                    validate_case_payload(capture, {"id": f"recognize/{case['id'].split('/')[1]}/batch-input", "result": result})
            mapped[case["id"]] = case
        typed.append(mapped)
    exact = {name: 0.0 for name in tolerances}
    for id_ in sorted(required):
        left, right = (capture[id_] for capture in typed)
        errors = compare_result(left["results"], right["results"], tolerances)
        # Each implementation must preserve its own single-image result exactly.
        independence = [compare_result(case["standalone"], case["results"], exact) for case in (left, right)]
        rows.append({"id": id_, "passed": not errors and not any(independence), "result_errors": errors,
                     "standalone_vs_batch_errors": independence})
    return {"passed": all(row["passed"] for row in rows), "cases": rows}


def speedups(previous: dict, current: dict, minimum: float) -> dict:
    if any(case.get("benchmark", {}).get("measured") is False
           for capture in (previous, current) for case in capture.get("cases", [])):
        fail("Correctness-only captures contain no inference timing samples; use a separate performance run")
    required = {"host_name", "cpu_identity", "operating_system", "architecture", "threads", "corpus_sha256", "rustc", "rustflags", "build_flags", "reference_lock_sha256"}
    before_environment = previous.get("metadata", {}).get("benchmark_environment", {})
    after_environment = current.get("metadata", {}).get("benchmark_environment", {})
    if not required.issubset(before_environment) or not required.issubset(after_environment):
        fail("Performance baseline lacks a complete benchmark_environment fingerprint; capture it with the maintained driver.")
    if before_environment != after_environment:
        changed = sorted(key for key in before_environment.keys() | after_environment.keys() if before_environment.get(key) != after_environment.get(key))
        fail(f"Performance baseline changes {', '.join(changed)}; compare the same host/compiler/workload budget.")
    before = {case["id"]: case for case in previous["cases"] if case.get("benchmark")}
    after = {case["id"]: case for case in current["cases"] if case.get("benchmark")}
    if set(before) != set(after) or not before:
        fail("Performance baseline has a different or empty timed workload.")
    ratios = []
    rows = []
    regressions = []
    for key in before:
        old = statistics.median(before[key]["benchmark"]["warm_ns"])
        new = statistics.median(after[key]["benchmark"]["warm_ns"])
        ratio = old / max(new, 1)
        ratios.append(ratio)
        rows.append({"id": key, "baseline_ns": old, "current_ns": new, "speedup": ratio})
        if key.startswith("ocr/") and new > old * 1.03:
            regressions.append({"id": key, "baseline_ns": old, "current_ns": new, "regression_percent": (new / old - 1) * 100})
    geometric = math.exp(sum(map(math.log, ratios)) / len(ratios))
    return {"passed": geometric >= minimum and not regressions, "minimum": minimum, "geometric_mean_speedup": geometric,
            "maximum_e2e_regression_percent": 3, "e2e_regressions": regressions, "cases": rows}


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, help="Clean checkout at tools/reference-lock.json revision")
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "target" / "cpu-parity")
    parser.add_argument("--replay", type=Path, help="Reuse a previously captured bundle, including its raw corpus")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1, help="Shared CPU budget, within 1..=16")
    parser.add_argument("--ocr-batch-size", type=int, default=1, help="Maximum same-width lines per complete OCR recognition batch")
    parser.add_argument("--stages", action="store_true", help="Also capture portable/intrinsic SIMD NCHW and unfused NHWC stages")
    parser.add_argument("--backend", choices=("auto", "scalar", "portable4", "avx", "avx-fma", "avx2-fma", "avx512"), help="Force the Rust optimized stages' kernel policy; reference remains scalar, unsupported policies fail")
    parser.add_argument("--quick", action="store_true", help="Smaller image workload; all seven graph models remain mandatory")
    parser.add_argument("--batch-only", action="store_true", help="Check all three real recognizers' four-input batching, without detector/OCR inference")
    parser.add_argument("--correctness-only", action="store_true", help="One inference per captured output plus required traces; no warmup/timing repetitions or production profiles")
    parser.add_argument("--baseline", type=Path, help="Prior Rust auto capture.json for measured milestone comparison")
    parser.add_argument("--min-speedup", type=float, default=1.0)
    parser.add_argument("--no-build", action="store_true", help="Require existing executables; never skips execution")
    args = parser.parse_args(argv)
    if args.iterations < 1 or args.min_speedup <= 0 or not 1 <= args.threads <= 16 or not 1 <= args.ocr_batch_size <= 2**31 - 1:
        parser.error("iterations/min-speedup must be positive; threads must be within 1..=16; ocr-batch-size within 1..=i32::MAX")
    if args.batch_only and args.baseline:
        parser.error("batch-only captures are correctness checks and have no performance baseline")
    if args.correctness_only and args.baseline:
        parser.error("correctness-only captures have no performance baseline")
    lock = read_json(LOCK)
    out = args.output.resolve()
    models = args.models_dir.resolve()
    if out.exists() and any(out.iterdir()):
        fail(f"Output directory is not empty; preserve previous evidence and choose a fresh path: {out}")
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    rust = require_tool("cargo")
    corpus = out / "corpus"
    driver_source = source_snapshot(ROOT, [Path(__file__).resolve()])
    native_provenance = None
    pending_native = []
    if args.replay:
        replay = args.replay.resolve()
        metadata = read_json(replay / "run.json")
        if metadata["reference_lock_sha256"] != sha256(LOCK) or metadata["quick"] != args.quick or metadata["threads"] != args.threads or metadata.get("batch_only", False) != args.batch_only:
            fail("Replay identity or workload differs from the pinned lock/current --quick choice.")
        if replay == out or out.is_relative_to(replay):
            fail("Replay output must be outside the source bundle.")
        for name in lock["models"]:
            if sha256(models / name) != lock["models"][name]:
                fail(f"Changed replay model: {name}")
        for dirname in ("corpus", "csharp-reference", "csharp-auto"):
            shutil.copytree(replay / dirname, out / dirname, dirs_exist_ok=True)
        if read_json(out / "csharp-reference" / "capture.json")["metadata"]["threads"] != 1:
            fail("Replay scalar oracle must have an effective thread budget of one.")
        native_provenance = metadata.get("exporter_build", {"provenance": "legacy_capture_without_source_fingerprint"})
        verify_frozen_exporter(replay, native_provenance, lock)
        shutil.copytree(replay / "binaries" / "csharp", out / "binaries" / "csharp")
        archived_exporter = replay / "sources" / "exporter"
        if archived_exporter.is_dir():
            verify_source_archive(native_provenance["sources"], archived_exporter)
            shutil.copytree(archived_exporter, out / "sources" / "exporter")
            verify_source_archive(native_provenance["sources"], out / "sources" / "exporter")
    else:
        if args.reference_dir is None:
            parser.error("--reference-dir is required unless --replay is supplied")
        reference = args.reference_dir.resolve()
        validate_inputs(reference, models, lock)
        dotnet = require_tool("dotnet")
        project = ROOT / "tools" / "reference-exporter" / "ReferenceExporter.csproj"
        native_provenance = {"sources": exporter_sources(), "dotnet": run([dotnet, "--info"], capture=True),
                             "provenance": "preexisting_executable_unverified" if args.no_build else "built_by_driver",
                             "reference_revision": lock["revision"]}
        if not args.no_build:
            run([dotnet, "build", str(project), "-c", "Release", f"-p:ReferenceRoot={reference}", "--nologo"])
            if native_provenance["sources"] != exporter_sources():
                fail("Exporter sources changed during compilation; repeat with stable source files.")
        exporter = project.parent / "bin" / "Release" / "net10.0" / "Sdcb.SimdPaddleOCR.Tests.dll"
        if not exporter.is_file():
            fail(f"Exporter executable is missing: {exporter}")
        native_provenance["executable_sha256"] = sha256(exporter)
        native_provenance["runtime_files"] = {path.name: sha256(path) for path in sorted(exporter.parent.iterdir()) if path.is_file()}
        frozen_exporter = out / "binaries" / "csharp"
        shutil.copytree(exporter.parent, frozen_exporter)
        if any(sha256(frozen_exporter / name) != digest for name, digest in native_provenance["runtime_files"].items()):
            fail("C# exporter runtime changed while freezing the baseline")
        exporter = frozen_exporter / exporter.name
        archive_sources(native_provenance["sources"], out / "sources" / "exporter")
        run([dotnet, str(exporter), "--generate-corpus", "--sample", str(ROOT / "tools" / "corpus-source"), "--corpus", str(corpus)])
        validate_corpus(corpus)
        for mode in ("reference", "auto"):
            effective_threads = 1 if mode == "reference" else args.threads
            env = reference_environment(mode, effective_threads)
            command = [dotnet, str(exporter), "--models-dir", str(models), "--corpus", str(corpus),
                       "--output", str(out / f"csharp-{mode}"), "--mode", mode, "--iterations", str(args.iterations), "--threads", str(effective_threads), "--ocr-batch-size", str(args.ocr_batch_size)]
            if args.quick:
                command.append("--quick")
            if args.batch_only:
                command.append("--batch-only")
            if args.correctness_only:
                command.append("--correctness-only")
            capture_file = out / f"csharp-{mode}" / "capture.json"
            pending_native.append((command, env, capture_file))
    validate_corpus(corpus)
    rust_provenance = {"sources": rust_sources(), "provenance": "preexisting_executable_unverified" if args.no_build else "built_by_driver"}
    if not args.no_build:
        run([rust, "build", "--release", "--example", "cpu_parity", "--features", "profiling", "--locked"])
        if rust_provenance["sources"] != rust_sources():
            fail("Rust sources changed during compilation; repeat with stable source files.")
    executable = ROOT / "target" / "release" / "examples" / ("cpu_parity.exe" if os.name == "nt" else "cpu_parity")
    if not executable.is_file():
        fail(f"Rust parity executable is missing: {executable}")
    rust_provenance["executable_sha256"] = sha256(executable)
    frozen_executable = out / "binaries" / "rust" / executable.name
    frozen_executable.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(executable, frozen_executable)
    if sha256(frozen_executable) != rust_provenance["executable_sha256"]:
        fail("Rust executable changed while freezing the baseline")
    archive_sources(rust_provenance["sources"], out / "sources" / "rust")
    archive_sources(driver_source, out / "sources" / "driver")
    executable = frozen_executable
    write_json(out / "build-provenance.json", {"rust_build": rust_provenance, "exporter_build": native_provenance,
               "driver_source": driver_source, "rustc": run([require_tool("rustc"), "--version", "--verbose"], capture=True)})
    print(f"Frozen Rust baseline: {executable}; source sha256={rust_provenance['sources']['sha256']}", flush=True)
    if not args.replay:
        import_env = reference_environment("reference", 1)
        run([dotnet, str(exporter), "--imports-only", "--models-dir", str(models), "--output", str(out / "csharp-reference")], env=import_env)
    run([str(executable), "--imports-only", "--models-dir", str(models), "--output", str(out / "rust-reference")])
    import_preflight = compare_imports(out / "csharp-reference", out / "rust-reference")
    write_json(out / "import-report.json", import_preflight)
    if not import_preflight["passed"]:
        fail(f"Import contracts differ; see {out / 'import-report.json'}")
    for command, env, capture_file in pending_native:
        run(command, env=env)
        capture = read_json(capture_file)
        capture["metadata"]["exporter_build"] = native_provenance
        write_json(capture_file, capture)
    processor = os.environ.get("PROCESSOR_IDENTIFIER") or platform.processor()
    if not processor and Path("/proc/cpuinfo").is_file():
        processor = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                          if line.startswith("model name")), "unknown")
    compiler = run([require_tool("rustc"), "--version", "--verbose"], capture=True)
    environment = {"host_name": platform.node(), "cpu_identity": processor, "operating_system": platform.platform(),
                   "architecture": platform.machine(), "threads": args.threads, "corpus_sha256": sha256(corpus / "corpus.json"),
                   "rustc": compiler, "rustflags": os.environ.get("RUSTFLAGS", ""), "build_flags": build_flags(), "reference_lock_sha256": sha256(LOCK), "ocr_batch_size": args.ocr_batch_size}
    modes = ("reference", "simd", "nhwc", "auto") if args.stages else ("reference", "auto")
    inventories = {}
    stage_profiles = {}
    for mode in modes:
        effective_threads = 1 if mode == "reference" else args.threads
        command = [str(executable), "--models-dir", str(models), "--corpus", str(corpus),
                   "--output", str(out / f"rust-{mode}"), "--mode", mode, "--iterations", str(args.iterations), "--threads", str(effective_threads), "--ocr-batch-size", str(args.ocr_batch_size)]
        if args.quick:
            command.append("--quick")
        if args.batch_only:
            command.append("--batch-only")
        if args.correctness_only:
            command.append("--correctness-only")
        if args.backend is not None and mode != "reference":
            command.extend(("--backend", args.backend))
        run(command)
        capture_file = out / f"rust-{mode}" / "capture.json"
        capture = read_json(capture_file)
        capture["metadata"]["benchmark_environment"] = environment
        capture["metadata"]["benchmark_fingerprint"] = hashlib.sha256(json.dumps(environment, sort_keys=True).encode()).hexdigest()
        capture["metadata"]["rust_build"] = rust_provenance
        capture["metadata"]["driver_source"] = driver_source
        capture["metadata"]["requested_threads"] = args.threads
        capture["metadata"]["effective_threads"] = effective_threads
        write_json(capture_file, capture)
        inventories[mode] = [{"case_id": case["id"], "input_shape": case["input_shape"], **operator}
                             for case in capture["cases"] for operator in case.get("diagnostic_operator_inventory", [])]
        stage_profiles[mode] = [{"case_id": case["id"], **case["diagnostic_production_stage_profile"]}
                               for case in capture["cases"] if case.get("diagnostic_production_stage_profile")]
    write_json(out / "operator-inventory.json", {"schema_version": 1,
               "timing_contract": "Unfused diagnostic node kernel wall time, including bounded-pool dispatch and excluding observer serialization; not production latency.",
               "stages": inventories})
    write_json(out / "stage-profiles.json", {"schema_version": 1,
               "timing_contract": "One separately instrumented production OCR run after official timing; phase wall time and worker sums include timer overhead.",
               "stages": stage_profiles})
    comparisons = {}
    imports = compare_imports(out / "csharp-reference", out / "rust-reference")
    pairs = [("csharp-reference", "rust-reference"), ("csharp-auto", "rust-auto"), ("rust-reference", "rust-auto")]
    if args.stages:
        pairs.extend(("csharp-reference", f"rust-{mode}") for mode in ("simd", "nhwc"))
    for left, right in pairs:
        comparisons[f"{left}_vs_{right}"] = (compare_batch_bundles(out / left, out / right, lock["tolerances"]) if args.batch_only
            else compare_bundles(out / left, out / right, lock["tolerances"], required_case_ids(corpus, args.quick)))
    performance = None
    if args.baseline:
        performance = speedups(read_json(args.baseline), read_json(out / "rust-auto" / "capture.json"), args.min_speedup)
    stage_performance = {}
    if args.stages and not args.batch_only and not args.correctness_only:
        for left, right in (("reference", "simd"), ("simd", "nhwc"), ("nhwc", "auto")):
            stage_performance[f"{left}_to_{right}"] = speedups(read_json(out / f"rust-{left}" / "capture.json"), read_json(out / f"rust-{right}" / "capture.json"), 1.0)
    report = {"schema_version": 1, "passed": imports["passed"] and all(c["passed"] for c in comparisons.values()) and (performance is None or performance["passed"]),
              "imports": imports,
              "comparisons": comparisons, "performance": performance, "stage_performance": stage_performance,
              "elapsed_seconds": time.time() - start}
    write_json(out / "report.json", report)
    metadata = {"schema_version": 1, "reference_lock_sha256": sha256(LOCK), "reference_revision": lock["revision"],
                "model_hashes": lock["models"], "quick": args.quick, "batch_only":args.batch_only, "iterations": args.iterations,"threads":args.threads,
                "platform": platform.platform(), "machine": platform.machine(), "processor": processor,
                "python": sys.version, "rustc": compiler, "benchmark_environment": environment,
                "rust_build": rust_provenance, "exporter_build": native_provenance, "driver_source": driver_source,
                "backend": args.backend, "stages": args.stages,
                "ocr_batch_size":args.ocr_batch_size,
                "correctness_only":args.correctness_only,
                "reference_effective_threads": 1,
                "rust_revision": run(["git", "rev-parse", "HEAD"], capture=True),
                "rustflags": os.environ.get("RUSTFLAGS", ""), "corpus_sha256": sha256(corpus / "corpus.json")}
    write_json(out / "run.json", metadata)
    print(f"{'PASS' if report['passed'] else 'FAIL'}: {sum(len(c['cases']) for c in comparisons.values())} comparisons; report: {out / 'report.json'}")
    for pair, comparison in comparisons.items():
        for case in comparison["cases"]:
            if not case["passed"]:
                print(pair, json.dumps(case, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, subprocess.CalledProcessError, ValueError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
