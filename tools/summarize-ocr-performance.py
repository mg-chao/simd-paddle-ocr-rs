"""Validate frozen REC/OCR identities and independent measurement scopes."""
import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import statistics

MODES = {"rec_graph", "rec_public_owned", "rec_public_borrowed", "ocr_owned",
         "ocr_borrowed", "mixed_widths", "owned_copies", "node_profile", "public_stage_profile"}
PAIRED = {"rec_graph", "rec_public_owned", "ocr_owned"}
REUSED = {"rec_graph", "rec_public_borrowed", "ocr_borrowed", "mixed_widths", "node_profile"}
COUNTERS = ("warm_allocation_count", "warm_allocation_bytes",
            "warm_retained_heap_bytes", "warm_peak_heap_growth_bytes")
WIDTHS = [32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 32, 384]
WIDTHS_V2 = [32, 64, 96, 160, 224, 320, 480, 576, 992, 32, 992]
CONTROLS = {
    "DOTNET_PROCESSOR_COUNT": "1", "DOTNET_EnableHWIntrinsic": "0",
    "DOTNET_TieredCompilation": "0", "DOTNET_TieredPGO": "0",
    "PPOCR_NHWC": "0", "SIMD_OCR_NOFUSE": "1", "PPOCR_DET_THREADS": "1",
    "PPOCR_DUMP_CONV": None, "PPOCR_DUMP_NODES": None,
    "PPOCR_DUMP_PLAN": None, "PPOCR_DUMP_GRAPH": None,
}
SHA = re.compile(r"[0-9a-f]{64}\Z")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError(f"Nonfinite JSON number: {value}")


def parse_float(value):
    result = float(value)
    require(math.isfinite(result), f"Nonfinite JSON number: {value}")
    return result


def read_json(path):
    with path.open(encoding="utf-8-sig") as source:
        return json.load(source, object_pairs_hook=unique_object,
                         parse_constant=reject_constant, parse_float=parse_float)


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(65536), b""):
            hasher.update(block)
    return hasher.hexdigest()


def integer(value, label, minimum=0):
    require(isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
            f"Invalid integer: {label}")
    return value


def finite(value, label, nonnegative=True):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and (not nonnegative or value >= 0),
            f"Invalid finite number: {label}")
    return value


def text(value, label):
    require(isinstance(value, str) and bool(value), f"Missing text: {label}")
    return value


def hash_value(value, label):
    require(isinstance(value, str) and SHA.fullmatch(value), f"Invalid SHA256: {label}")
    return value


def shape(value, label):
    require(isinstance(value, list), f"Invalid shape: {label}")
    for dimension in value:
        integer(dimension, label)
    return value


def checked(root, entry):
    raw = text(entry.get("path"), "identity path")
    relative = PurePosixPath(raw)
    require(not relative.is_absolute() and ".." not in relative.parts and "\\" not in raw
            and relative.as_posix() == raw and ":" not in raw,
            f"Unsafe identity path: {raw}")
    path = root.joinpath(*relative.parts)
    require(path.resolve().is_relative_to(root.resolve()) and path.is_file(),
            f"Missing or escaping identity: {path}")
    require(path.stat().st_size == integer(entry.get("bytes"), raw)
            and digest(path) == hash_value(entry.get("sha256"), raw),
            f"Snapshot identity differs: {path}")
    return path


def inventory(root, entries, label, key="path", nonempty=True):
    require(isinstance(entries, list) and (entries or not nonempty), f"Missing {label} inventory")
    identities = {}
    paths = set()
    for entry in entries:
        require(isinstance(entry, dict), f"Invalid {label} entry")
        identifier = text(entry.get(key), label)
        require(identifier not in identities and entry.get("path") not in paths,
                f"Duplicate {label} identity: {identifier}")
        checked(root, entry)
        identities[identifier] = entry
        paths.add(entry["path"])
    return identities


def source_inventory(folder, snapshot, prefix, field, hash_field):
    entries = snapshot.get(field)
    identities = inventory(folder / prefix, entries, field)
    encoded = "".join(f"{entry['path']}\0{entry['sha256']}\n" for entry in entries).encode()
    require(hashlib.sha256(encoded).hexdigest() == hash_value(snapshot.get(hash_field), hash_field),
            f"Source inventory digest differs: {field}")
    return identities


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    left, right = math.floor(position), math.ceil(position)
    return ordered[left] + (ordered[right] - ordered[left]) * (position - left)


def sample_stats(row, count, label):
    require(isinstance(row, dict), f"{label}: invalid statistic row")
    values = row.get("samples_ms")
    require(isinstance(values, list) and len(values) == count, f"{label}: invalid sample count")
    for value in values:
        finite(value, label + " sample")
    for key, expected in (("p50_ms", percentile(values, .5)),
                          ("p95_ms", percentile(values, .95)), ("mean_ms", sum(values) / count)):
        actual = finite(row.get(key), label + " " + key)
        require(math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-8),
                f"{label}: {key} differs from raw samples")
    return values


def allocation_counts(value, measured, label, native=False):
    require(isinstance(value, dict), f"{label}: missing allocation counters")
    for key in COUNTERS:
        if measured or (native and key == "warm_allocation_bytes"):
            integer(value.get(key), label + " " + key)
        else:
            require(value.get(key) is None, f"{label}: uninstrumented allocation counter {key}")


def expected_rows(cases, mode, schema_version=1):
    if mode == "mixed_widths":
        hasher = hashlib.sha256()
        for width in WIDTHS if schema_version == 1 else WIDTHS_V2:
            hasher.update(bytes((i * 17 + width) % 251 for i in range(width * 48 * 3)))
        return [("mixed-widths", "rec.pixels", [], hasher.hexdigest(), mode)]
    result = []
    for case in cases:
        for artifact in case["artifacts"]:
            identifier = artifact["id"]
            selected = (identifier.endswith(".rec.input") if mode in ("rec_graph", "node_profile") else
                        identifier.endswith(".rec.pixels") if mode in ("rec_public_owned", "rec_public_borrowed", "owned_copies") else
                        identifier == "input.pixels")
            if selected:
                result.append((case["id"], identifier, artifact["shape"], artifact["sha256"], mode))
    require(bool(result), f"{mode}: selected corpus has no measured rows")
    return result


def summarize(folder):
    folder = folder.resolve()
    snapshot_path = folder / "snapshot.json"
    snapshot = read_json(snapshot_path)
    require(isinstance(snapshot, dict), "Invalid measurement snapshot object")
    schema_version = integer(snapshot.get("schema_version"), "snapshot schema", 1)
    require((schema_version, snapshot.get("kind")) in ((1, "tiny_ocr_measurement_snapshot"), (2, "ocr_measurement_snapshot")),
            "Unknown measurement snapshot")
    native_profile = snapshot.get("csharp_profile", "canonical")
    require(native_profile in ("canonical", "cpu"), "Invalid C# execution profile")
    if schema_version == 1:
        require(native_profile == "canonical", "Legacy snapshots require canonical execution")
    else:
        integer(snapshot.get("processor_affinity_mask"), "processor_affinity_mask", 1)
    rounds = integer(snapshot.get("rounds"), "rounds", 1)
    heap_rounds = integer(snapshot.get("heap_rounds"), "heap_rounds")
    warmup = integer(snapshot.get("warmup"), "warmup")
    samples = integer(snapshot.get("samples"), "samples", 1)
    for field in ("os", "architecture"):
        text(snapshot.get(field), field)
    integer(snapshot.get("processor_count"), "processor_count", 1)
    expected_controls = dict(CONTROLS)
    if native_profile == "cpu":
        expected_controls.update(DOTNET_EnableHWIntrinsic="1", DOTNET_TieredCompilation="1", DOTNET_TieredPGO="1",
                                 PPOCR_NHWC="1", SIMD_OCR_NOFUSE="0")
    require(snapshot.get("runtime_controls") == expected_controls, "Execution runtime controls differ")
    selected_case = snapshot.get("case")
    require(selected_case is None or isinstance(selected_case, str), "Invalid selected case")
    selected_case = selected_case or None
    requested_modes = snapshot.get("modes")
    require(isinstance(requested_modes, list) and bool(requested_modes)
            and all(isinstance(mode, str) and mode in MODES for mode in requested_modes)
            and len(set(requested_modes)) == len(requested_modes), "Unknown or duplicate modes")
    binaries = inventory(folder, snapshot.get("executables"), "executable")
    required_binaries = {"ocr-timing.exe", "ocr-heap.exe", "csharp/SimdPaddleOCR.ReferenceExport.dll",
                         "csharp/Sdcb.SimdPaddleOCR.dll"}
    require(required_binaries <= binaries.keys(), "Missing required binary snapshots")
    sources = source_inventory(folder, snapshot, "source", "source_files", "source_sha256")
    reference_sources = source_inventory(folder, snapshot, "reference-source", "reference_source_files", "reference_source_sha256")
    require({"Cargo.toml", "Cargo.lock"} <= sources.keys() and any(path.endswith(".rs") for path in sources),
            "Missing Rust build/source snapshots")
    require(any(path.endswith(".cs") for path in reference_sources)
            and any(path.endswith(".csproj") for path in reference_sources),
            "Missing native C# build/source snapshots")
    reference = Path(text(snapshot.get("reference"), "reference")).resolve()
    manifest_path = reference / "manifest.json"
    manifest_hash = hash_value(snapshot.get("manifest_sha256"), "manifest_sha256")
    require(digest(manifest_path) == manifest_hash, "Reference manifest identity changed")
    manifest = read_json(manifest_path)
    require(isinstance(manifest, dict) and isinstance(manifest.get("contract"), dict), "Invalid reference manifest object")
    family = manifest.get("contract", {}).get("model")
    require(integer(manifest.get("schema_version"), "manifest schema", 1) == 1
            and manifest.get("contract", {}).get("profile") == "canonical"
            and family in (("tiny",) if schema_version == 1 else ("tiny", "small", "medium")), "Shipped-model canonical reference required")
    if schema_version == 2:
        require(snapshot.get("model_family") == family, "Snapshot model family differs")
    assets = inventory(reference, manifest.get("models"), "model", "id")
    require({"rec", "dictionary"} <= assets.keys(), "Missing recognition model/dictionary")
    if any(mode.startswith("ocr_") or mode == "public_stage_profile" for mode in requested_modes):
        require({"det", "cls"} <= assets.keys(), "Missing OCR detector/classifier models")
    inventory(reference, manifest.get("auxiliary_files", []), "auxiliary", nonempty=False)
    all_cases = manifest.get("cases")
    require(isinstance(all_cases, list) and bool(all_cases), "Missing reference cases")
    case_ids = set()
    for case in all_cases:
        require(isinstance(case, dict), "Invalid reference case object")
        identifier = text(case.get("id"), "case id")
        require(identifier not in case_ids, "Duplicate case identity")
        case_ids.add(identifier)
        artifacts = inventory(reference, case.get("artifacts"), "artifact", "id")
        for artifact in artifacts.values():
            dimensions = shape(artifact.get("shape"), "artifact shape")
            dtype = artifact.get("dtype")
            text(artifact.get("layout"), "artifact layout")
            require(dtype in ("f32", "u8", "json"), "Invalid artifact dtype")
            if dtype == "json":
                require(dimensions == [] and artifact["layout"] == "json", "Invalid JSON artifact metadata")
            else:
                require(math.prod(dimensions) * (4 if dtype == "f32" else 1) == artifact["bytes"],
                        "Artifact shape/length differs")
    require(selected_case is None or selected_case in case_ids, "Unknown selected case")
    cases = [case for case in all_cases if selected_case is None or case["id"] == selected_case]
    modes = {}
    expected_files = {"snapshot.json"}
    for mode in requested_modes:
        identities = expected_rows(cases, mode, schema_version)
        groups = {}
        for engine, count in (("rust-timing", rounds), ("csharp", rounds if mode in PAIRED else 0), ("rust-heap", heap_rounds)):
            reports = []
            for round_id in range(1, count + 1):
                name = f"{mode}-{engine}-r{round_id}.json"
                expected_files.add(name)
                path = folder / name
                report = read_json(path)
                require(isinstance(report, dict), f"{name}: invalid report object")
                for key, expected in (("kind", "tiny_ocr_benchmark" if schema_version == 1 else "ocr_benchmark"), ("schema_version", schema_version), ("profile", native_profile if engine == "csharp" else "canonical"),
                                      ("mode", mode), ("runtime", "csharp" if engine == "csharp" else "rust"),
                                      ("round", round_id), ("executable_kind", engine), ("warmup", warmup),
                                      ("samples", samples), ("manifest_sha256", manifest_hash),
                                      ("model_sha256", assets["rec"]["sha256"]),
                                      ("dictionary_sha256", assets["dictionary"]["sha256"]),
                                      ("instrumented", engine == "rust-heap")):
                    require(type(report.get(key)) is type(expected) and report[key] == expected, f"{name}: {key} differs")
                if schema_version == 2:
                    require(report.get("model_family") == family, f"{name}: model family differs")
                if mode.startswith("ocr_") or mode == "public_stage_profile":
                    for key, asset in (("det_model_sha256", "det"), ("cls_model_sha256", "cls")):
                        require(report.get(key) == assets[asset]["sha256"], f"{name}: {key} differs")
                executable = ("csharp/SimdPaddleOCR.ReferenceExport.dll" if engine == "csharp" else
                              "ocr-heap.exe" if engine == "rust-heap" else "ocr-timing.exe")
                require(report.get("executable_sha256") == binaries[executable]["sha256"], f"{name}: binary digest differs")
                if engine == "csharp":
                    require(report.get("csharp_native_sha256") == binaries["csharp/Sdcb.SimdPaddleOCR.dll"]["sha256"],
                            f"{name}: native assembly digest differs")
                for key in ("load_ms", "prepare_ms"):
                    finite(report.get(key), name + " " + key)
                finite(report.get("checksum"), name + " checksum", nonnegative=False)
                integer(report.get("process_resident_high_water_bytes"), name + " resident high water")
                if engine != "csharp":
                    for key in ("activation_bytes", "retained_activation_bytes", "workspace_bytes"):
                        integer(report.get(key), name + " " + key)
                    require(report["retained_activation_bytes"] >= report["activation_bytes"], f"{name}: retained activations below active bytes")
                    allocation_counts(report.get("load_allocations"), engine == "rust-heap", name + " load")
                    allocation_counts(report.get("prepare_allocations"), engine == "rust-heap", name + " preparation")
                rows = report.get("inputs")
                require(isinstance(rows, list), f"{name}: missing input rows")
                actual = []
                for row in rows:
                    require(isinstance(row, dict), f"{name}: invalid row")
                    actual.append((text(row.get("case_id"), name), text(row.get("artifact_id"), name),
                                   shape(row.get("shape"), name), hash_value(row.get("sha256"), name), text(row.get("stage"), name)))
                    sample_stats(row, samples, name + " row")
                    finite(row.get("first_call_ms"), name + " first call")
                    allocation_counts(row, engine == "rust-heap", name + " warm", native=engine == "csharp")
                    if engine != "csharp":
                        allocation_counts(row.get("first_call_allocations"), engine == "rust-heap", name + " first call")
                    if engine == "rust-heap" and mode in REUSED:
                        require(all(row[key] == 0 for key in COUNTERS), f"{name}: warmed reuse gate failed")
                require(actual == identities, f"{name}: row identities/order differ")
                profiles = report.get("stage_profile", [])
                if mode == "public_stage_profile":
                    require(schema_version == 2 and isinstance(profiles, list) and len(profiles) == len(rows),
                            f"{name}: missing production stage profiles")
                    for profile, row in zip(profiles, rows):
                        require(isinstance(profile, dict) and profile.get("case_id") == row["case_id"]
                                and profile.get("artifact_id") == row["artifact_id"] and profile.get("samples") == samples,
                                f"{name}: production profile identity differs")
                        totals = profile.get("totals")
                        require(isinstance(totals, dict), f"{name}: missing stage totals")
                        for key in ("total_ns", "binding_ns", "det_preprocess_ns", "det_inference_ns", "det_postprocess_ns",
                                    "crop_ns", "cls_orientation_ns", "rec_preprocess_ns", "rec_backbone_ns", "projection_ns",
                                    "ctc_ns", "result_storage_ns", "recognized_lines"):
                            integer(totals.get(key), f"{name}: stage {key}")
                        require(totals["total_ns"] > 0 and totals["det_inference_ns"] > 0,
                                f"{name}: empty production timings")
                else:
                    require(profiles == [], f"{name}: unexpected production profiles")
                sequence = report.get("shape_sequence", [])
                if mode == "mixed_widths":
                    require(sequence == [[1, 3, 48, width] for width in (WIDTHS if schema_version == 1 else WIDTHS_V2)]
                            and all(all(type(n) is int for n in dims) for dims in sequence),
                            f"{name}: mixed-width sequence differs")
                else:
                    require(sequence == [], f"{name}: unexpected shape sequence")
                total = [sum(row["samples_ms"][i] for row in rows) for i in range(samples)]
                corpus_values = sample_stats(report.get("corpus", {}), samples, name + " corpus")
                require(all(math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-8) for a, b in zip(total, corpus_values)),
                        f"{name}: aggregate samples differ")
                reports.append({"path": name, "bytes": path.stat().st_size, "sha256": digest(path),
                                "corpus": report["corpus"], "load_ms": report["load_ms"], "prepare_ms": report["prepare_ms"],
                                "activation_bytes": report.get("activation_bytes"), "retained_activation_bytes": report.get("retained_activation_bytes"),
                                "workspace_bytes": report.get("workspace_bytes"), "process_resident_high_water_bytes": report.get("process_resident_high_water_bytes"),
                                "allocations": [{key: row.get(key) for key in COUNTERS} for row in rows]})
            if reports:
                timings = [value for report in reports for value in report["corpus"]["samples_ms"]]
                groups[engine] = {"reports": reports,
                    "median_process_p50_ms": statistics.median(report["corpus"]["p50_ms"] for report in reports),
                    "median_process_p95_ms": statistics.median(report["corpus"]["p95_ms"] for report in reports),
                    "process_p50_range_ms": [min(report["corpus"]["p50_ms"] for report in reports), max(report["corpus"]["p50_ms"] for report in reports)],
                    "pooled_p50_ms": percentile(timings, .5), "pooled_p95_ms": percentile(timings, .95)}
        modes[mode] = {"paired_owned_or_copied_scope": mode in PAIRED,
                       "observer_instrumented_diagnostic": mode in ("node_profile", "public_stage_profile"), "groups": groups}
    require({path.name for path in folder.glob("*.json")} == expected_files, "Unexpected or missing raw report files")
    return {"schema_version": schema_version, "kind": "tiny_ocr_performance" if schema_version == 1 else "ocr_performance",
            "model_family": family, "csharp_profile": native_profile,
            "snapshot": {"path": str(snapshot_path), "bytes": snapshot_path.stat().st_size, "sha256": digest(snapshot_path)},
            "source_sha256": snapshot["source_sha256"], "reference_source_sha256": snapshot["reference_source_sha256"],
            "protocol": {key: snapshot[key] for key in ("rounds", "heap_rounds", "warmup", "samples", "case", "os", "architecture", "processor_count")},
            "scope": "Scalar Rust; C# execution profile is recorded separately. Copied full graphs and owning public calls are paired. Borrowed, mixed-width, owned-copy and observed timings are Rust-only. Heap/profiling timings excluded from latency claims. Cold boundaries and RSS differ by engine.",
            "modes": modes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.measurements)
    with args.out.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")
    print("Saved validated OCR report:", args.out)


if __name__ == "__main__":
    main()
