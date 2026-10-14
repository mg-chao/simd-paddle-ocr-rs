#!/usr/bin/env python3
"""Validate and summarize the frozen Step 1/2 performance protocol (stdlib only).

Latency uses six uninstrumented processes, each with 30 warmups and 100
samples. Three separate Rust heap processes supply allocation evidence.
Nothing in this report establishes a speedup outside the recorded corpus.
"""

import argparse
import hashlib
import json
import math
import re
import statistics
import struct
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MODES = ("det_graph", "pipeline", "stages", "mixed_shapes")
ENGINES = ("csharp", "rust-timing", "rust-heap")
COUNTS = ("warm_allocation_count", "warm_allocation_bytes",
          "warm_retained_heap_bytes", "warm_peak_heap_growth_bytes")
MEMORY = ("activation_bytes", "retained_activation_bytes",
          "retained_workspace_bytes", "crop_capacity_bytes", "line_capacity_bytes",
          "observed_process_peak_working_set_bytes")
CONTROLS = {
    "DOTNET_PROCESSOR_COUNT": "1", "DOTNET_EnableHWIntrinsic": "0",
    "DOTNET_TieredCompilation": "0", "DOTNET_TieredPGO": "0",
    "PPOCR_NHWC": "0", "SIMD_OCR_NOFUSE": "1", "PPOCR_DET_THREADS": "1",
    "PPOCR_DUMP_CONV": None, "PPOCR_DUMP_NODES": None,
    "PPOCR_DUMP_PLAN": None, "PPOCR_DUMP_GRAPH": None,
}
MIXED_SHAPES = [[32, 64], [64, 32], [64, 128], [128, 64], [32, 64],
                [64, 128], [32, 32], [32, 96], [96, 32], [64, 64],
                [64, 96], [96, 64], [96, 96], [128, 32]]
SHA = re.compile(r"[0-9a-f]{64}\Z")
RUN_NAME = re.compile(
    r"(det_graph|pipeline|stages|mixed_shapes|owned_copies)-(csharp|rust-timing|rust-heap)-round-(\d+)\.json\Z")


class InvalidReport(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise InvalidReport(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise InvalidReport(f"non-finite JSON number: {value}")


def parse_float(value):
    result = float(value)
    require(math.isfinite(result), f"non-finite JSON number: {value}")
    return result


def read_json(path):
    try:
        with path.open(encoding="utf-8-sig") as source:
            return json.load(source, object_pairs_hook=unique_object,
                             parse_constant=reject_constant, parse_float=parse_float)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise InvalidReport(f"{path}: {error}") from error


def file_identity(path):
    path = path.resolve(strict=True)
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            size += len(block)
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": size}


def hash_value(value, label):
    require(isinstance(value, str) and SHA.fullmatch(value), f"invalid SHA256: {label}")
    return value


def finite_number(value, label):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0, f"invalid nonnegative number: {label}")
    return value


def integer(value, label):
    require(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
            f"invalid nonnegative integer: {label}")
    return value


def shape(value, label):
    require(isinstance(value, list), f"invalid shape: {label}")
    for extent in value:
        require(integer(extent, label) > 0, f"zero shape dimension: {label}")
    return value


def percentile(values, quantile):
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    left, right = math.floor(position), math.ceil(position)
    return ordered[left] + (ordered[right] - ordered[left]) * (position - left)


def stats(values):
    present = [v for v in values if v is not None]
    if not present:
        return {"available": False}
    return {"available": True, "count": len(present), "median": statistics.median(present),
            "min": min(present), "max": max(present), "values_in_round_order": values}


def sample_stats(values):
    return {"count": len(values), "p50_ms": percentile(values, .5),
            "p95_ms": percentile(values, .95), "min_ms": min(values), "max_ms": max(values)}


def samples(row, label):
    values = row.get("samples_ms")
    require(isinstance(values, list) and len(values) == 100,
            f"{label}: expected 100 samples")
    for value in values:
        finite_number(value, f"{label} samples_ms")
    for field, expected in (("p50_ms", percentile(values, .5)),
                            ("p95_ms", percentile(values, .95)),
                            ("mean_ms", sum(values) / len(values))):
        actual = finite_number(row.get(field), f"{label} {field}")
        require(math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-8),
                f"{label}: {field} does not agree with raw samples")
    return values


def counts(value, label, measured):
    require(isinstance(value, dict), f"{label}: missing allocation counters")
    for field in COUNTS:
        item = value.get(field)
        if measured:
            integer(item, f"{label} {field}")
        else:
            require(item is None, f"{label}: timing run contains heap instrumentation")


def row_identity(row):
    require(isinstance(row, dict), "invalid input row")
    for key in ("case_id", "artifact_id", "stage"):
        require(isinstance(row.get(key), str) and row[key], f"missing row {key}")
    return {key: row[key] for key in ("case_id", "artifact_id", "stage")} | {
        "shape": shape(row.get("shape"), "input row"),
        "sha256": hash_value(row.get("sha256"), "input row")}


def resolve_reference(value):
    require(isinstance(value, str) and value, "missing reference directory")
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve(strict=True)


def bundle_file(root, asset, identities):
    relative = Path(asset.get("path", ""))
    require(str(relative) not in ("", ".") and not relative.is_absolute(), "invalid bundle path")
    path = (root / relative).resolve(strict=True)
    require(path.is_relative_to(root), f"bundle path escapes reference: {relative}")
    identity = file_identity(path)
    require(identity["sha256"] == hash_value(asset.get("sha256"), str(path)),
            f"bundle file hash differs: {path}")
    require(identity["bytes"] == integer(asset.get("bytes"), str(path)),
            f"bundle file size differs: {path}")
    identities[str(path)] = identity
    return path


def load_dataset(reference, selected, identities):
    manifest_path = reference / "manifest.json"
    manifest_identity = file_identity(manifest_path)
    identities[str(manifest_path)] = manifest_identity
    manifest = read_json(manifest_path)
    contract = manifest.get("contract", {})
    require(manifest.get("schema_version") == 1 and contract.get("profile") == "canonical"
            and contract.get("backend") == "cpu" and contract.get("capture_scope") == "step12",
            f"{reference}: requires canonical CPU Step 1/2 reference")
    models = {item["id"]: item for item in manifest.get("models", [])}
    require("det" in models and "cls" in models, "reference requires DET and CLS models")
    for asset in models.values():
        bundle_file(reference, asset, identities)
    all_cases = manifest.get("cases")
    require(isinstance(all_cases, list) and all_cases, "reference has no cases")
    require(len({c["id"] for c in all_cases}) == len(all_cases), "duplicate reference case")
    cases = [case for case in all_cases if selected is None or case["id"] == selected]
    require(cases, "selected case does not exist")
    artifacts, descriptions = {}, []
    for case in cases:
        rows = case.get("artifacts", [])
        inventory = {a["id"]: a for a in rows}
        require(len(inventory) == len(rows), f"duplicate artifact in {case['id']}")
        require(all(key in inventory for key in ("input.metadata", "input.pixels", "step12.public")),
                f"{case['id']}: missing image/public metadata")
        meta = read_json(bundle_file(reference, inventory["input.metadata"], identities))
        public = read_json(bundle_file(reference, inventory["step12.public"], identities))
        lines = public.get("lines")
        require(isinstance(lines, list), f"{case['id']}: missing public lines")
        require(integer(public.get("detected_count"), "detected_count") == len(lines),
                f"{case['id']}: public line count differs")
        descriptions.append({"case_id": case["id"], "kind": case.get("kind"),
                             "input": meta, "line_count": len(lines),
                             "detector_input_shape": inventory["det.input"]["shape"],
                             "input_pixels_sha256": inventory["input.pixels"]["sha256"],
                             "rotated_line_count": sum(line.get("applied_rotation_degrees") == 180
                                                       for line in lines)})
        artifacts[case["id"]] = inventory
    result = {"reference": str(reference), "manifest": manifest_identity,
              "model": contract.get("model"), "contract": contract,
              "models": [{key: models[name][key] for key in ("id", "sha256", "bytes")}
                         for name in ("det", "cls")],
              "selected_case": selected, "manifest_case_ids": [c["id"] for c in all_cases],
              "case_ids": [c["id"] for c in cases], "cases": descriptions,
              "total_line_count": sum(c["line_count"] for c in descriptions),
              "reference_git_head": manifest.get("provenance", {}).get("git", {}).get("head")}
    return result, artifacts


def mixed_sha256():
    digest = hashlib.sha256()
    for height, width in MIXED_SHAPES:
        for index in range(3 * height * width):
            digest.update(struct.pack("<f", ((index * 17 % 251) - 125) / 125))
    return digest.hexdigest()


def expected_rows(mode, dataset, artifacts):
    if mode == "mixed_shapes":
        return [{"case_id": "mixed-shapes", "artifact_id": "det.input", "shape": [],
                 "sha256": mixed_sha256(), "stage": "mixed_shapes"}]
    result = []
    def add(case, artifact_id, stage):
        require(artifact_id in artifacts[case], f"reference missing {case}/{artifact_id}")
        artifact = artifacts[case][artifact_id]
        result.append({"case_id": case, "artifact_id": artifact_id,
                       "shape": artifact["shape"], "sha256": artifact["sha256"], "stage": stage})
    for case in dataset["cases"]:
        name = case["case_id"]
        if mode == "det_graph":
            add(name, "det.input", mode)
        elif mode in ("pipeline", "owned_copies"):
            add(name, "input.pixels", "owned_result_copy" if mode == "owned_copies" else mode)
        else:
            add(name, "det.input", "det_preprocess")
            add(name, "det.probabilities", "db_postprocess")
            for index in range(case["line_count"]):
                prefix = f"line.{index:03}"
                for suffix, stage in (("crop", "crop"), ("cls.input", "cls_preprocess"),
                                      ("cls.probabilities", "cls_graph"),
                                      ("cls.public", "cls_public"), ("rec.pixels", "rotate180")):
                    add(name, f"{prefix}.{suffix}", stage)
    return result


def validate_run(report, path, mode, engine, round_number, dataset, expected):
    label = str(path)
    require(report.get("schema_version") == 1 and report.get("profile") == "canonical"
            and report.get("mode") == mode, f"{label}: invalid report schema/profile/mode")
    require(report.get("runtime") == ("csharp" if engine == "csharp" else "rust"),
            f"{label}: wrong runtime")
    require(report.get("engine", engine) == engine and report.get("round", round_number) == round_number,
            f"{label}: engine/round differs from filename")
    require(report.get("warmup") == 30 and report.get("samples") == 100,
            f"{label}: requires 30 warmup / 100 samples")
    require(resolve_reference(report.get("reference")) == Path(dataset["reference"]),
            f"{label}: reference directory differs")
    model_hashes = {m["id"]: m["sha256"] for m in dataset["models"]}
    for key, expected_hash in (("model_sha256", model_hashes["det"]),
                               ("cls_model_sha256", model_hashes["cls"]),
                               ("manifest_sha256", dataset["manifest"]["sha256"])):
        require(report.get(key) == expected_hash, f"{label}: {key} differs from reference")
    rows = report.get("inputs")
    require(isinstance(rows, list) and [row_identity(row) for row in rows] == expected,
            f"{label}: input case/artifact/stage/shape/hash identities differ")
    heap = engine == "rust-heap"
    if engine != "csharp":
        require(report.get("instrumentation") == (
            "separate allocator-instrumented executable; exclude these timings" if heap else
            "standard allocator; uninstrumented latency"), f"{label}: wrong Rust instrumentation")
        counts(report.get("load_allocations"), f"{label} load", heap)
        counts(report.get("prepare_allocations"), f"{label} prepare", heap)
    else:
        require(report.get("instrumentation") == "Current-thread managed allocation volume; native allocations and retained heap are not measured."
                and report.get("warm_allocation_count") is None,
                f"{label}: wrong C# timing instrumentation")
        for field in ("load_allocation_bytes", "prepare_allocation_bytes", "warm_allocation_bytes"):
            integer(report.get(field), f"{label} {field}")
    for field in ("load_ms", "prepare_ms"):
        finite_number(report.get(field), f"{label} {field}")
    for field in MEMORY:
        if report.get(field) is not None:
            integer(report[field], f"{label} {field}")
    for index, row in enumerate(rows):
        samples(row, f"{label} row {index}")
        finite_number(row.get("first_call_ms"), f"{label} first_call_ms")
        if engine != "csharp":
            counts(row, f"{label} row {index} warm", heap)
            counts(row.get("first_call_allocations"), f"{label} row {index} first", heap)
        else:
            integer(row.get("warm_allocation_bytes"), f"{label} row {index} managed bytes")
    corpus = samples(report.get("corpus", {}), f"{label} corpus")
    for index, value in enumerate(corpus):
        require(math.isclose(value, sum(row["samples_ms"][index] for row in rows),
                             rel_tol=1e-10, abs_tol=1e-8), f"{label}: corpus sum differs")
    if heap:
        for field in COUNTS[:2]:
            require(report.get(field) == sum(row[field] for row in rows),
                    f"{label}: total {field} differs from rows")
        if mode not in ("mixed_shapes", "owned_copies"):
            require(all(row["warm_allocation_count"] == 0 and row["warm_allocation_bytes"] == 0
                        for row in rows), f"{label}: fixed-scope Rust zero-allocation gate failed")
    elif engine == "rust-timing":
        require(all(report.get(field) is None for field in COUNTS[:2]),
                f"{label}: instrumented run mislabeled as latency")
    if mode == "mixed_shapes":
        require(report.get("shape_sequence") == MIXED_SHAPES, f"{label}: mixed-shape sequence differs")
        values = report.get("reshape_samples_ms")
        require(isinstance(values, list) and len(values) == 100, f"{label}: missing reshape samples")
        for value, whole in zip(values, corpus):
            finite_number(value, f"{label} reshape timing")
            require(value <= whole + 1e-8, f"{label}: reshape timing exceeds whole call")
    else:
        require(report.get("shape_sequence") is None and report.get("reshape_samples_ms", []) == [],
                f"{label}: unexpected shape-change samples")
    preparations = report.get("session_preparations", [])
    require(isinstance(preparations, list), f"{label}: invalid session preparations")
    for preparation in preparations:
        require(isinstance(preparation, dict) and isinstance(preparation.get("case_id"), str)
                and isinstance(preparation.get("stage"), str), f"{label}: invalid preparation identity")
        finite_number(preparation.get("elapsed_ms"), f"{label} session preparation")
        for field in ("activation_bytes", "retained_activation_bytes"):
            if engine != "csharp" or preparation.get(field) is not None:
                integer(preparation.get(field), f"{label} session preparation {field}")
        if engine != "csharp":
            counts(preparation.get("allocations"), f"{label} session preparation", heap)


def latency_summary(rows):
    return {"process_p50_ms": stats([r["p50_ms"] for r in rows]),
            "process_p95_ms": stats([r["p95_ms"] for r in rows]),
            "pooled_600_samples_ms": sample_stats([v for r in rows for v in r["samples_ms"]])}


def allocation_summary(values):
    return {field.removeprefix("warm_"): stats([value.get(field) for value in values])
            for field in COUNTS}


def preparation_summary(reports, heap):
    arrays = [r.get("session_preparations", []) for r in reports]
    identities = [[{k: p.get(k) for k in ("case_id", "stage")} for p in array] for array in arrays]
    require(all(identity == identities[0] for identity in identities),
            "session preparation identities differ between processes")
    result = []
    for index, identity in enumerate(identities[0]):
        phases = [array[index] for array in arrays]
        row = identity | {"elapsed_ms": stats([p["elapsed_ms"] for p in phases]),
                          "activation_bytes": stats([p["activation_bytes"] for p in phases]),
                          "retained_activation_bytes": stats([p["retained_activation_bytes"] for p in phases])}
        if heap:
            row["allocations"] = allocation_summary([p["allocations"] for p in phases])
        # Keep additions to the phase schema reviewable without confusing their timing units.
        extras = [{k: v for k, v in p.items() if k not in (
            "case_id", "stage", "elapsed_ms", "activation_bytes", "retained_activation_bytes", "allocations")}
                  for p in phases]
        if any(extras):
            row["additional_metadata_in_round_order"] = extras
        result.append(row)
    return result


def engine_summary(reports):
    result = {"process_count": len(reports), "corpus": latency_summary([r["corpus"] for r in reports]),
              "cold_phases_ms": {field: stats([r[field] for r in reports])
                                  for field in ("load_ms", "prepare_ms")},
              "sum_of_per_row_first_calls_ms": stats([sum(row["first_call_ms"] for row in r["inputs"])
                                                      for r in reports]),
              "memory_bytes": {field: stats([r.get(field) for r in reports]) for field in MEMORY},
              "measurement_semantics": sorted({r["measurement"] for r in reports}),
              "process_memory_semantics": sorted({r["process_memory_semantics"] for r in reports
                                                    if "process_memory_semantics" in r}),
              "additional_memory_semantics": {
                  key: sorted({r[key] for r in reports if isinstance(r.get(key), str)})
                  for key in sorted({key for r in reports for key in r
                                     if key.endswith("_semantics") and key != "process_memory_semantics"})},
              "session_preparations": preparation_summary(reports, False), "rows": []}
    for index, first in enumerate(reports[0]["inputs"]):
        rows = [r["inputs"][index] for r in reports]
        result["rows"].append(row_identity(first) | {"latency": latency_summary(rows),
                             "first_call_ms": stats([r["first_call_ms"] for r in rows])})
    if reports[0]["runtime"] == "csharp":
        result["managed_allocation_bytes"] = {
            "load": stats([r["load_allocation_bytes"] for r in reports]),
            "prepare": stats([r["prepare_allocation_bytes"] for r in reports]),
            "warm": stats([r["warm_allocation_bytes"] for r in reports])}
        for index, row in enumerate(result["rows"]):
            row["managed_warm_allocation_bytes"] = stats([r["inputs"][index]["warm_allocation_bytes"]
                                                         for r in reports])
    if reports[0]["mode"] == "mixed_shapes":
        values = [r["reshape_samples_ms"] for r in reports]
        result["reshape_only"] = latency_summary([{
            "samples_ms": v, "p50_ms": percentile(v, .5), "p95_ms": percentile(v, .95)} for v in values])
    return result


def heap_summary(reports, mode):
    result = {"process_count": len(reports), "timings_excluded_from_latency": True,
              "instrumentation": reports[0]["instrumentation"],
              "load": allocation_summary([r["load_allocations"] for r in reports]),
              "prepare": allocation_summary([r["prepare_allocations"] for r in reports]),
              "warm_total": {field.removeprefix("warm_"): stats([r[field] for r in reports])
                             for field in COUNTS[:2]},
              "memory_bytes": {field: stats([r.get(field) for r in reports]) for field in MEMORY},
              "session_preparations": preparation_summary(reports, True), "rows": [],
              "zero_allocation_gate": {"applicable": mode not in ("mixed_shapes", "owned_copies"),
                                       "passed": True if mode not in ("mixed_shapes", "owned_copies") else None,
                                       "scope": "Each prepared fixed-input row's 100 measured calls, after first call and 30 warmups."}}
    for index, first in enumerate(reports[0]["inputs"]):
        rows = [r["inputs"][index] for r in reports]
        result["rows"].append(row_identity(first) | {
            "first_call": allocation_summary([r["first_call_allocations"] for r in rows]),
            "warm": allocation_summary(rows)})
    return result


def source_inventory(directory, environment, identities):
    path = directory / "source-files.json"
    inventory = read_json(path)
    require(isinstance(inventory, list) and inventory, "missing source inventory")
    paths = set()
    for item in inventory:
        require(isinstance(item, dict) and isinstance(item.get("path"), str) and item["path"]
                and item["path"] not in paths, "invalid/duplicate source path")
        paths.add(item["path"])
        hash_value(item.get("sha256"), "source inventory")
    encoded = "\n".join(f"{item['path']} {item['sha256']}" for item in inventory).encode("utf-8")
    require(hashlib.sha256(encoded).hexdigest() == environment.get("source_files_sha256"),
            "source inventory fingerprint differs from summary")
    identities[str(path)] = file_identity(path)
    for filename, key in (("step12-timing.exe", "rust_timing_sha256"),
                          ("step12-heap.exe", "rust_heap_sha256")):
        identity = file_identity(directory / filename)
        require(identity["sha256"] == environment.get(key), f"snapshot executable hash differs: {filename}")
        identities[identity["path"]] = identity


def load_measurements(directory, identities):
    directory = directory.resolve(strict=True)
    summary_path = directory / "summary.json"
    summary = read_json(summary_path)
    identities[str(summary_path)] = file_identity(summary_path)
    require(summary.get("schema_version") == 1, "invalid summary schema")
    for key, value in (("rounds", 6), ("heap_rounds", 3), ("warmup", 30), ("samples", 100)):
        require(summary.get(key) == value, f"{directory}: protocol requires {key}={value}")
    modes = summary.get("modes")
    require(isinstance(modes, list) and modes and len(set(modes)) == len(modes)
            and all(mode in MODES for mode in modes), "invalid/duplicate declared modes")
    environment = summary.get("environment")
    require(isinstance(environment, dict) and environment.get("runtime_controls") == CONTROLS,
            "canonical scalar runtime controls differ")
    for key in ("source_files_sha256", "rust_timing_sha256", "rust_heap_sha256",
                "csharp_exporter_sha256", "csharp_native_sha256"):
        hash_value(environment.get(key), key)
    source_inventory(directory, environment, identities)
    dataset, artifacts = load_dataset(resolve_reference(summary.get("reference")),
                                      summary.get("case") or None, identities)
    require(dataset["reference_git_head"] == environment.get("reference_git_head"),
            "native benchmark source HEAD differs from reference provenance")
    expected_orders = [{"mode": mode, "round": number,
                        "order": ["csharp", "rust-timing"] if number % 2 else ["rust-timing", "csharp"]}
                       for mode in modes for number in range(1, 7)]
    require(summary.get("process_orders") == expected_orders, "fresh-process alternating order differs")
    indexed_summary = {}
    for item in summary.get("runs", []):
        key = (item.get("mode"), item.get("engine"), item.get("round"))
        require(key not in indexed_summary, "duplicate summary run")
        indexed_summary[key] = item
    expected_keys = {(mode, engine, number) for mode in modes for engine in ENGINES
                     for number in range(1, 4 if engine == "rust-heap" else 7)}
    require(set(indexed_summary) == expected_keys, "summary is missing or adds protocol runs")
    raw_paths = {}
    for path in directory.glob("*.json"):
        match = RUN_NAME.fullmatch(path.name)
        if match:
            key = (match[1], match[2], int(match[3]))
            require(key not in raw_paths, f"duplicate raw report round: {path}")
            raw_paths[key] = path
    require(set(raw_paths) == expected_keys, "raw report files are missing or add undeclared runs")
    measurements = []
    for mode in modes:
        grouped = {engine: [] for engine in ENGINES}
        expected = expected_rows(mode, dataset, artifacts)
        for engine in ENGINES:
            for number in range(1, 4 if engine == "rust-heap" else 7):
                key = (mode, engine, number)
                path = raw_paths[key]
                report = read_json(path)
                validate_run(report, path, mode, engine, number, dataset, expected)
                item = indexed_summary[key]
                for field in ("load_ms", "prepare_ms", "warm_allocation_count", "warm_allocation_bytes",
                              "activation_bytes", "retained_activation_bytes", "observed_process_peak_working_set_bytes"):
                    require(item.get(field) == report.get(field), f"{path}: summary/raw {field} differs")
                for field in ("p50_ms", "p95_ms"):
                    require(item.get(field) == report["corpus"][field], f"{path}: summary/raw {field} differs")
                identities[str(path)] = file_identity(path)
                grouped[engine].append(report)
        measurements.append({"mode": mode, "shape_sequence": MIXED_SHAPES if mode == "mixed_shapes" else None,
                             "csharp": engine_summary(grouped["csharp"]),
                             "rust": engine_summary(grouped["rust-timing"]),
                             "rust_allocations": heap_summary(grouped["rust-heap"], mode)})
    return {"directory": str(directory), "dataset": dataset, "measurements": measurements}, environment, artifacts


def load_owned(directory, measurement_sets, environment, identities):
    directory = directory.resolve(strict=True)
    raw_paths = {}
    for path in directory.glob("*.json"):
        match = RUN_NAME.fullmatch(path.name)
        if match:
            key = (match[1], match[2], int(match[3]))
            require(key not in raw_paths, "duplicate owned-copy run")
            raw_paths[key] = path
    expected_keys = {("owned_copies", engine, number) for engine in ("rust-timing", "rust-heap")
                     for number in range(1, 4 if engine == "rust-heap" else 7)}
    require(set(raw_paths) == expected_keys, "owned copies require six timing and three heap runs")
    first = read_json(raw_paths[("owned_copies", "rust-timing", 1)])
    matches = [(dataset, artifacts) for dataset, artifacts in measurement_sets
               if dataset["manifest"]["sha256"] == first.get("manifest_sha256")
               and dataset["reference"] == str(resolve_reference(first.get("reference")))]
    require(len(matches) == 1, "owned-copy reference must match exactly one measurement set")
    dataset, artifacts = matches[0]
    expected = expected_rows("owned_copies", dataset, artifacts)
    sidecar_path = directory / "identity.json"
    sidecar = read_json(sidecar_path) if sidecar_path.exists() else None
    if sidecar is not None:
        require(isinstance(sidecar, dict), "invalid owned identity sidecar")
        for key in ("rust_timing_sha256", "rust_heap_sha256"):
            require(sidecar.get(key) == environment.get(key), f"owned copy {key} differs from main snapshot")
        if "source_files_sha256" in sidecar:
            require(sidecar["source_files_sha256"] == environment["source_files_sha256"],
                    "owned-copy source identity differs")
        if "environment" in sidecar:
            require(sidecar["environment"] == environment, "owned-copy environment differs")
        identities[str(sidecar_path)] = file_identity(sidecar_path)
        for filename, key in (("step12-timing.exe", "rust_timing_sha256"),
                              ("step12-heap.exe", "rust_heap_sha256")):
            path = directory / filename
            if path.exists():
                identity = file_identity(path)
                require(identity["sha256"] == sidecar[key], f"owned snapshot hash differs: {filename}")
                identities[identity["path"]] = identity
    grouped = {engine: [] for engine in ("rust-timing", "rust-heap")}
    for (mode, engine, number), path in sorted(raw_paths.items()):
        report = read_json(path)
        validate_run(report, path, mode, engine, number, dataset, expected)
        identities[str(path)] = file_identity(path)
        grouped[engine].append(report)
    return {"directory": str(directory), "manifest_sha256": dataset["manifest"]["sha256"],
            "identity": sidecar, "executable_identity_verified": sidecar is not None,
            "identity_verification": "Sidecar hashes match SHA-verified main executables; owned executable files are also hashed when present." if sidecar is not None else "No owned identity sidecar; model/manifest/row identities verified, executable provenance unavailable.",
            "scope": "Rust persistent-result copying only; extraction is outside the timed interval. No C# pair.",
            "rust": engine_summary(grouped["rust-timing"]),
            "rust_allocations": heap_summary(grouped["rust-heap"], "owned_copies")}


def build_report(directories, owned=None):
    identities, results, dataset_pairs = {}, [], []
    environment = None
    require(len({p.resolve() for p in directories}) == len(directories), "duplicate measurement directory")
    for directory in directories:
        result, current_environment, artifacts = load_measurements(directory, identities)
        if environment is None:
            environment = current_environment
        else:
            require(current_environment == environment, "measurement source/executable/environment identities differ")
        results.append(result)
        dataset_pairs.append((result["dataset"], artifacts))
    report = {
        "schema_version": 1,
        "protocol": {"timing_processes_per_engine_per_mode": 6, "heap_processes_per_mode": 3,
                     "warmup_calls_per_row": 30, "samples_per_process_per_row": 100,
                     "pooled_samples_per_engine_per_row": 600, "process_order": "alternating C#/Rust fresh-process pairs"},
        "semantics": {
            "process_p50_ms": "median/min/max of six process medians; values are in process-round order",
            "process_p95_ms": "median/min/max of six process p95 values; this is not pooled p95",
            "pooled_600_samples_ms": "linear-interpolated p50/p95 of all 600 uninstrumented samples, ignoring process grouping",
            "corpus": "Sum of corresponding per-row sample indexes; not wall time of a manifest-order corpus sweep",
            "cold_phases": "Fresh-process load/setup phases as defined by each engine; setup work differs. Session preparations are separate and may occur later.",
            "first_call": "One first measured call before 30 warmups, per row. Later rows may reuse buffers from earlier rows.",
            "rust_allocations": "Three separate allocator-instrumented processes. Successful allocation/reallocation count and requested bytes; retained/peak heap growth are relative to each phase baseline. Peak includes transient staging.",
            "csharp_allocations": "Current-thread managed allocation volume only; no native allocation count or retained/transient heap peak measurement",
            "activation": "Recorded planned/retained F32 activation buffers, excluding model weights; phase and whole-process residency are separate metrics",
            "rss": "Parent-sampled OS high-water resident memory, including startup/validation/corpus/model/runtime; can miss the last observation. Not activation bytes or allocator peak.",
            "mixed_shapes": "One session follows the recorded 14-shape sequence each call. Total latency includes reshape and graph work; reshape-only samples are reported separately.",
            "conclusion_scope": "Recorded corpus, build and environment only; no universal speedup claim"},
        "environment_and_source": environment, "measurement_sets": results,
        "summarizer": file_identity(Path(__file__)),
    }
    if owned is not None:
        report["owned_copies"] = load_owned(owned, dataset_pairs, environment, identities)
    report["raw_files"] = [identities[key] for key in sorted(identities)]
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", action="append", type=Path, required=True,
                        help="Completed benchmark-step12.ps1 output directory; repeat for each dataset")
    parser.add_argument("--owned", type=Path, help="Optional raw-only Rust owned-copies directory")
    parser.add_argument("--out", type=Path, required=True, help="New JSON report; existing files are refused")
    args = parser.parse_args(argv)
    try:
        require(not args.out.exists(), f"output already exists: {args.out}")
        report = build_report(args.measurements, args.owned)
        # Exclusive creation also handles a file created after the initial check.
        with args.out.open("x", encoding="utf-8", newline="\n") as destination:
            json.dump(report, destination, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            destination.write("\n")
    except (InvalidReport, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"error: {error}\n")
    print(f"Saved validated performance report: {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
