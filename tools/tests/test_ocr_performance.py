"""Synthetic measurement identities and failure gates; no inference or processes."""
import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "summarize-ocr-performance.py"
SPEC = importlib.util.spec_from_file_location("ocr_summary", SCRIPT)
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)
LEGACY_MODES = ("rec_graph", "rec_public_owned", "rec_public_borrowed", "ocr_owned",
                "ocr_borrowed", "mixed_widths", "owned_copies", "node_profile")
MODES = (*LEGACY_MODES, "public_stage_profile")
PAIRED = {"rec_graph", "rec_public_owned", "ocr_owned"}
WIDTHS = [32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 32, 384]
CONTROLS = {
    "DOTNET_PROCESSOR_COUNT": "1", "DOTNET_EnableHWIntrinsic": "0",
    "DOTNET_TieredCompilation": "0", "DOTNET_TieredPGO": "0",
    "PPOCR_NHWC": "0", "SIMD_OCR_NOFUSE": "1", "PPOCR_DET_THREADS": "1",
    "PPOCR_DUMP_CONV": None, "PPOCR_DUMP_NODES": None,
    "PPOCR_DUMP_PLAN": None, "PPOCR_DUMP_GRAPH": None,
}


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def counters(measured, calls=0, retained=0, peak=0):
    return {"warm_allocation_count": calls if measured else None,
            "warm_allocation_bytes": calls * 64 if measured else None,
            "warm_retained_heap_bytes": retained if measured else None,
            "warm_peak_heap_growth_bytes": peak if measured else None}


class OcrPerformanceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.reference = self.root / "reference"
        self.reference.mkdir()
        self.measurements = self.root / "measurements"
        self.measurements.mkdir()

        def saved(root, path, contents, **extra):
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(contents)
            return {"path": path, "sha256": sha(contents), "bytes": len(contents)} | extra

        models = [saved(self.reference, f"models/{name}.bin", name.encode(), id=name)
                  for name in ("det", "cls", "rec", "dictionary")]
        cases = []
        for case_id, lines in (("sample", 2), ("blank", 0)):
            artifacts = [
                saved(self.reference, f"{case_id}/metadata.json", json.dumps({"width": 2,
                    "height": 2, "stride": 6, "pixel_format": "bgr24"}).encode(),
                    id="input.metadata", dtype="json", shape=[], layout="json"),
                saved(self.reference, f"{case_id}/pixels.bin", bytes(range(12)),
                    id="input.pixels", dtype="u8", shape=[2, 6], layout="bgr24_rows"),
            ]
            for line in range(lines):
                stage = f"line.{line:03}"
                artifacts.extend([
                    saved(self.reference, f"{case_id}/{stage}/rec-input.bin", b"\0" * (3 * 48 * 32 * 4),
                          id=stage + ".rec.input", dtype="f32", shape=[1, 3, 48, 32], layout="nchw"),
                    saved(self.reference, f"{case_id}/{stage}/rec-pixels.bin", bytes(range(63)),
                          id=stage + ".rec.pixels", dtype="u8", shape=[3, 7, 3], layout="bgr24"),
                ])
            artifacts.append(saved(self.reference, f"{case_id}/result.json",
                                   json.dumps({"detected_count": lines, "lines": [{}] * lines}).encode(),
                                   id="result.public", dtype="json", shape=[], layout="json"))
            cases.append({"id": case_id, "kind": "ocr", "artifacts": artifacts})
        self.manifest = {"schema_version": 1, "contract": {"profile": "canonical", "backend": "cpu",
                         "model": "tiny", "capture_scope": "ocr"}, "models": models,
                         "cases": cases, "auxiliary_files": []}
        write_json(self.reference / "manifest.json", self.manifest)
        executables = [saved(self.measurements, name, name.encode()) for name in
                       ("ocr-timing.exe", "ocr-heap.exe", "csharp/SimdPaddleOCR.ReferenceExport.dll",
                        "csharp/Sdcb.SimdPaddleOCR.dll")]
        source_files = [saved(self.measurements / "source", name, name.encode()) for name in
                        ("Cargo.lock", "Cargo.toml", "src/runtime.rs",
                         "tools/reference-export/Program.cs", *sorted(SUMMARY.PROFILE_SOURCE_FILES))]
        reference_source_files = [saved(self.measurements / "reference-source", name, name.encode()) for name in
                                  ("src/Sdcb.SimdPaddleOCR/PaddleOcrRecognizer.cs",
                                   "src/Sdcb.SimdPaddleOCR/Sdcb.SimdPaddleOCR.csproj")]
        self.snapshot = {"schema_version": 1, "kind": "tiny_ocr_measurement_snapshot",
                         "reference": str(self.reference), "manifest_sha256": sha((self.reference / "manifest.json").read_bytes()),
                         "rounds": 2, "heap_rounds": 1, "warmup": 2, "samples": 3,
                         "case": None, "modes": list(LEGACY_MODES), "os": "synthetic", "architecture": "X64",
                         "processor_count": 1, "runtime_controls": copy.deepcopy(CONTROLS),
                         "executables": executables, "source_files": source_files,
                         "reference_source_files": reference_source_files,
                         "reference_source_sha256": sha("".join(f"{e['path']}\0{e['sha256']}\n" for e in reference_source_files).encode())}
        self.refresh_source_identity()
        self.build_reports()

    def refresh_source_identity(self):
        self.snapshot["source_sha256"] = sha("".join(
            f"{entry['path']}\0{entry['sha256']}\n" for entry in self.snapshot["source_files"]).encode())

    def expected_rows(self, mode):
        if mode == "mixed_widths":
            widths = WIDTHS if self.snapshot["schema_version"] == 1 else SUMMARY.WIDTHS_V2
            data = b"".join(bytes((index * 17 + width) % 251 for index in range(width * 48 * 3))
                            for width in widths)
            return [{"case_id": "mixed-widths", "artifact_id": "rec.pixels", "shape": [],
                     "sha256": sha(data), "stage": mode}]
        rows = []
        for case in self.manifest["cases"]:
            if self.snapshot.get("case") and self.snapshot["case"] != case["id"]:
                continue
            for artifact in case["artifacts"]:
                identifier = artifact["id"]
                selected = (identifier.endswith(".rec.input") if mode in ("rec_graph", "node_profile") else
                            identifier.endswith(".rec.pixels") if mode in ("rec_public_owned", "rec_public_borrowed", "owned_copies") else
                            identifier == "input.pixels")
                if selected:
                    rows.append({"case_id": case["id"], "artifact_id": identifier,
                                 "shape": artifact["shape"], "sha256": artifact["sha256"], "stage": mode})
        return rows

    def raw_report(self, mode, engine, round_id):
        heap = engine == "rust-heap"
        rust = engine != "csharp"
        assets = {item["id"]: item for item in self.manifest["models"]}
        binaries = {item["path"]: item for item in self.snapshot["executables"]}
        rows = []
        for ordinal, identity in enumerate(self.expected_rows(mode)):
            values = [1. + round_id + ordinal + index / 10 for index in range(self.snapshot["samples"])]
            row = copy.deepcopy(identity) | {"first_call_ms": 20., "samples_ms": values,
                    "p50_ms": SUMMARY.percentile(values, .5), "p95_ms": SUMMARY.percentile(values, .95),
                    "mean_ms": sum(values) / len(values)}
            row.update(counters(heap, calls=1 if mode in ("owned_copies", "rec_public_owned", "ocr_owned") else 0))
            if rust:
                row["first_call_allocations"] = counters(heap, calls=2, peak=128)
            else:
                row["warm_allocation_bytes"] = 256
            rows.append(row)
        total = [sum(row["samples_ms"][index] for row in rows) for index in range(self.snapshot["samples"])]
        executable = "csharp/SimdPaddleOCR.ReferenceExport.dll" if not rust else "ocr-heap.exe" if heap else "ocr-timing.exe"
        schema = self.snapshot["schema_version"]
        widths = WIDTHS if schema == 1 else SUMMARY.WIDTHS_V2
        result = {"schema_version": schema, "kind": "tiny_ocr_benchmark" if schema == 1 else "ocr_benchmark", "runtime": "rust" if rust else "csharp",
                  "profile": "canonical" if rust else self.snapshot.get("csharp_profile", "canonical"), "mode": mode, "round": round_id, "executable_kind": engine,
                  "executable_sha256": binaries[executable]["sha256"],
                  "csharp_native_sha256": binaries["csharp/Sdcb.SimdPaddleOCR.dll"]["sha256"],
                  "warmup": self.snapshot["warmup"], "samples": self.snapshot["samples"],
                  "model_sha256": assets["rec"]["sha256"], "dictionary_sha256": assets["dictionary"]["sha256"],
                  "det_model_sha256": assets["det"]["sha256"], "cls_model_sha256": assets["cls"]["sha256"],
                  "manifest_sha256": self.snapshot["manifest_sha256"], "instrumented": heap,
                  "inputs": rows, "load_ms": 10., "prepare_ms": 20., "checksum": 4.,
                  "activation_bytes": 1024 if rust else None, "retained_activation_bytes": 2048 if rust else None,
                  "workspace_bytes": 4096 if rust else None, "process_resident_high_water_bytes": 10000,
                  "shape_sequence": [[1, 3, 48, width] for width in widths] if mode == "mixed_widths" else [],
                  "node_profile": [], "stage_profile": [],
                  "corpus": {"samples_ms": total, "p50_ms": SUMMARY.percentile(total, .5),
                             "p95_ms": SUMMARY.percentile(total, .95), "mean_ms": sum(total) / len(total)}}
        if schema == 2:
            result["model_family"] = self.snapshot["model_family"]
        if rust:
            result["load_allocations"] = counters(heap, calls=3, peak=256)
            result["prepare_allocations"] = counters(heap, calls=4, peak=512)
            result["session_preparations"] = []
        if mode == "public_stage_profile":
            for row in rows:
                case = next(case for case in self.manifest["cases"] if case["id"] == row["case_id"])
                public = next(artifact for artifact in case["artifacts"] if artifact["id"] == "result.public")
                lines = json.loads((self.reference / public["path"]).read_bytes())["detected_count"]
                totals = {key: 1000 for key in SUMMARY.PROFILE_STAGES}
                if lines == 0:
                    for key in SUMMARY.PROFILE_RECOGNITION_STAGES:
                        totals[key] = 0
                totals.update(total_ns=int(sum(row["samples_ms"]) * 1e6) - 1000,
                              recognized_lines=lines * self.snapshot["samples"])
                result["stage_profile"].append({"case_id": row["case_id"], "artifact_id": row["artifact_id"],
                                                "samples": self.snapshot["samples"], "totals": totals})
        return result

    def build_reports(self):
        write_json(self.measurements / "snapshot.json", self.snapshot)
        for mode in self.snapshot["modes"]:
            for engine, rounds in (("rust-timing", self.snapshot["rounds"]),
                                   ("csharp", self.snapshot["rounds"] if mode in PAIRED else 0),
                                   ("rust-heap", self.snapshot["heap_rounds"])):
                for round_id in range(1, rounds + 1):
                    write_json(self.measurements / f"{mode}-{engine}-r{round_id}.json", self.raw_report(mode, engine, round_id))

    def change_report(self, change, mode="rec_graph", engine="rust-timing", round_id=1):
        path = self.measurements / f"{mode}-{engine}-r{round_id}.json"
        report = json.loads(path.read_bytes())
        change(report)
        write_json(path, report)

    def rejected(self):
        with self.assertRaises((ValueError, OSError, KeyError, TypeError)):
            SUMMARY.summarize(self.measurements)

    def test_all_modes_separate_paired_scopes_and_heap_latencies(self):
        self.upgrade_v2()
        result = SUMMARY.summarize(self.measurements)
        self.assertEqual(set(result["modes"]), set(MODES))
        for mode, value in result["modes"].items():
            self.assertEqual(value["paired_owned_or_copied_scope"], mode in PAIRED)
            self.assertEqual("csharp" in value["groups"], mode in PAIRED)
            self.assertEqual(len(value["groups"]["rust-timing"]["reports"]), 2)
            self.assertEqual(len(value["groups"]["rust-heap"]["reports"]), 1)
        timing = result["modes"]["rec_graph"]["groups"]["rust-timing"]
        self.assertNotEqual(timing["median_process_p95_ms"], timing["pooled_p95_ms"])

    def upgrade_v2(self, profile="canonical"):
        self.manifest["contract"]["model"] = "small"
        write_json(self.reference / "manifest.json", self.manifest)
        self.snapshot.update(schema_version=2, kind="ocr_measurement_snapshot", model_family="small",
                             csharp_profile=profile, processor_affinity_mask=1,
                             manifest_sha256=sha((self.reference / "manifest.json").read_bytes()))
        if "public_stage_profile" not in self.snapshot["modes"]:
            self.snapshot["modes"].append("public_stage_profile")
        if profile == "cpu":
            self.snapshot["runtime_controls"].update(DOTNET_EnableHWIntrinsic="1", DOTNET_TieredCompilation="1",
                DOTNET_TieredPGO="1", PPOCR_NHWC="1", SIMD_OCR_NOFUSE="0")
        self.build_reports()
        for path in self.measurements.glob("*-r*.json"):
            report = json.loads(path.read_bytes())
            report.update(schema_version=2, kind="ocr_benchmark", model_family="small",
                          profile=profile if report["runtime"] == "csharp" else "canonical")
            if report["mode"] == "mixed_widths":
                report["inputs"][0]["sha256"] = SUMMARY.expected_rows(self.manifest["cases"], "mixed_widths", 2)[0][3]
                report["shape_sequence"] = [[1, 3, 48, width] for width in SUMMARY.WIDTHS_V2]
            write_json(path, report)

    def test_v2_all_model_and_optimized_csharp_profiles_remain_separate(self):
        self.upgrade_v2("cpu")
        result = SUMMARY.summarize(self.measurements)
        self.assertEqual(result["schema_version"], 2)
        self.assertEqual(result["model_family"], "small")
        self.assertEqual(result["csharp_profile"], "cpu")
        self.change_report(lambda r: r.update(profile="cpu"))
        self.rejected()

    def test_v2_rejects_family_control_and_shape_sequence_drift(self):
        self.upgrade_v2()
        self.change_report(lambda r: r.update(model_family="medium"))
        self.rejected()
        self.upgrade_v2()
        self.change_report(lambda r: r.update(shape_sequence=[[1, 3, 48, 32]]), mode="mixed_widths")
        self.rejected()

    def test_file_and_source_digest_tampering_rejected(self):
        for base, path in ((self.measurements, "ocr-heap.exe"),
                           (self.measurements / "source", "src/runtime.rs"),
                           (self.measurements / "reference-source", "src/Sdcb.SimdPaddleOCR/PaddleOcrRecognizer.cs"),
                           (self.reference, "models/rec.bin"),
                           (self.reference, "sample/line.000/rec-pixels.bin")):
            with self.subTest(path=path):
                target = base / path
                original = target.read_bytes()
                target.write_bytes(original + b"changed")
                self.rejected()
                target.write_bytes(original)
        self.snapshot["source_sha256"] = sha(b"wrong source inventory")
        write_json(self.measurements / "snapshot.json", self.snapshot)
        self.rejected()

    def test_report_model_dictionary_and_binary_identities_rejected(self):
        for field in ("manifest_sha256", "model_sha256", "dictionary_sha256", "executable_sha256"):
            with self.subTest(field=field):
                self.change_report(lambda r: r.update({field: sha(b"wrong identity")}))
                self.rejected()
                self.build_reports()

    def test_csharp_report_is_bound_to_exporter_and_native_snapshots(self):
        for field in ("executable_sha256", "csharp_native_sha256"):
            with self.subTest(field=field):
                self.change_report(lambda r: r.update({field: sha(b"wrong native binary")}), engine="csharp")
                self.rejected()
                self.build_reports()

    def test_ocr_report_includes_detector_and_classifier_identities(self):
        for field in ("det_model_sha256", "cls_model_sha256"):
            for engine in ("rust-timing", "csharp"):
                with self.subTest(field=field, engine=engine):
                    self.change_report(lambda r: r.update({field: sha(b"wrong graph")}), mode="ocr_owned", engine=engine)
                    self.rejected()
                    self.build_reports()

    def test_row_identity_shape_order_and_missing_row_rejected(self):
        for change in (lambda r: r["inputs"][0].update({"shape": [1, 3, 48, 64]}),
                       lambda r: r["inputs"][0].update({"sha256": sha(b"wrong pixels")}),
                       lambda r: r["inputs"].reverse(), lambda r: r["inputs"].pop()):
            change_report = change
            self.change_report(change_report)
            self.rejected()
            self.build_reports()

    def test_missing_extra_round_and_unknown_report_rejected(self):
        target = self.measurements / "rec_graph-rust-timing-r2.json"
        target.unlink()
        self.rejected()
        self.build_reports()
        (self.measurements / "rec_graph-rust-timing-r3.json").write_bytes(target.read_bytes())
        self.rejected()
        (self.measurements / "rec_graph-rust-timing-r3.json").unlink()
        write_json(self.measurements / "unknown-rust-timing-r1.json", {})
        self.rejected()

    def test_unknown_and_duplicate_modes_rejected(self):
        original = copy.deepcopy(self.snapshot)
        self.snapshot["modes"] = list(MODES) + ["unknown"]
        self.build_reports()
        self.rejected()
        for path in self.measurements.glob("unknown-*.json"):
            path.unlink()
        self.snapshot = original
        self.snapshot["modes"].append("rec_graph")
        self.build_reports()
        self.rejected()

    def test_protocol_round_warmup_samples_and_engine_mismatch_rejected(self):
        for field, value in (("round", 2), ("warmup", 999), ("samples", 999),
                             ("runtime", "csharp"), ("executable_kind", "csharp"),
                             ("instrumented", True), ("profile", "cpu")):
            self.change_report(lambda r: r.update({field: value}))
            self.rejected()
            self.build_reports()

    def test_zero_and_negative_protocol_counts_rejected(self):
        # Remove now-unexpected report files so filename rejection cannot mask
        # an invalid snapshot count being accepted without measurements.
        for field, value in (("rounds", 0), ("rounds", -1), ("heap_rounds", -1)):
            original = copy.deepcopy(self.snapshot)
            self.snapshot[field] = value
            for path in self.measurements.glob("*-r*.json"):
                path.unlink()
            self.build_reports()
            self.rejected()
            self.snapshot = original
            self.build_reports()

    def test_canonical_runtime_controls_rejected(self):
        self.snapshot["runtime_controls"]["DOTNET_EnableHWIntrinsic"] = "1"
        write_json(self.measurements / "snapshot.json", self.snapshot)
        self.rejected()

    def test_unknown_case_and_empty_rec_corpus_rejected(self):
        for selected in ("missing", "blank"):
            original = copy.deepcopy(self.snapshot)
            self.snapshot["case"] = selected
            self.build_reports()
            self.rejected()
            self.snapshot = original
        self.build_reports()

    def test_duplicate_manifest_and_snapshot_identities_rejected(self):
        for field in ("models", "cases"):
            original = copy.deepcopy(self.manifest)
            self.manifest[field].append(copy.deepcopy(self.manifest[field][0]))
            write_json(self.reference / "manifest.json", self.manifest)
            self.snapshot["manifest_sha256"] = sha((self.reference / "manifest.json").read_bytes())
            self.build_reports()
            self.rejected()
            self.manifest = original
            write_json(self.reference / "manifest.json", self.manifest)
            self.snapshot["manifest_sha256"] = sha((self.reference / "manifest.json").read_bytes())
        for field in ("executables", "source_files"):
            original = copy.deepcopy(self.snapshot)
            self.snapshot[field].append(copy.deepcopy(self.snapshot[field][0]))
            self.refresh_source_identity()
            self.build_reports()
            self.rejected()
            self.snapshot = original
        self.build_reports()

    def test_duplicate_artifact_identity_rejected(self):
        case = self.manifest["cases"][0]
        case["artifacts"].append(copy.deepcopy(case["artifacts"][-1]))
        write_json(self.reference / "manifest.json", self.manifest)
        self.snapshot["manifest_sha256"] = sha((self.reference / "manifest.json").read_bytes())
        self.build_reports()
        self.rejected()

    def test_wrong_reference_model_contract_rejected(self):
        self.manifest["contract"]["model"] = "small"
        write_json(self.reference / "manifest.json", self.manifest)
        self.snapshot["manifest_sha256"] = sha((self.reference / "manifest.json").read_bytes())
        self.build_reports()
        self.rejected()

    def test_raw_statistics_counts_and_samples_recomputed(self):
        for change in (lambda r: r["inputs"][0].update({"p50_ms": 10.}),
                       lambda r: r["inputs"][0].update({"mean_ms": 10.}),
                       lambda r: r["corpus"].update({"p95_ms": 10.}),
                       lambda r: r["corpus"]["samples_ms"].__setitem__(0, 10.),
                       lambda r: r["inputs"][0]["samples_ms"].pop()):
            self.change_report(change)
            self.rejected()
            self.build_reports()

    def test_nonfinite_numbers_and_boolean_numeric_values_rejected(self):
        for change in (lambda r: r["inputs"][0].update({"p50_ms": float("nan")}),
                       lambda r: r["corpus"].update({"p95_ms": float("nan")}),
                       lambda r: r["corpus"]["samples_ms"].__setitem__(0, float("nan")),
                       lambda r: r.update({"load_ms": float("inf")}),
                       lambda r: r.update({"prepare_ms": -1.}),
                       lambda r: r["inputs"][0].update({"first_call_ms": -1.}),
                       lambda r: r["inputs"][0]["shape"].__setitem__(0, True)):
            self.change_report(change)
            self.rejected()
            self.build_reports()

    def test_duplicate_json_keys_and_overflowing_float_rejected(self):
        path = self.measurements / "rec_graph-rust-timing-r1.json"
        original = path.read_text()
        path.write_text(original.replace('"profile": "canonical"', '"profile": "bad", "profile": "canonical"'))
        self.rejected()
        path.write_text(original.replace('"load_ms": 10.0', '"load_ms": 1e999'))
        self.rejected()

    def test_zero_allocation_and_retention_failure_gates(self):
        for field in ("warm_allocation_count", "warm_allocation_bytes", "warm_retained_heap_bytes", "warm_peak_heap_growth_bytes"):
            self.change_report(lambda r: r["inputs"][0].update({field: 1}), engine="rust-heap")
            self.rejected()
            self.build_reports()

    def test_owned_heap_and_uninstrumented_counter_types_rejected(self):
        self.change_report(lambda r: r["inputs"][0].update({"warm_allocation_count": None}), mode="owned_copies", engine="rust-heap")
        self.rejected()
        self.build_reports()
        self.change_report(lambda r: r["inputs"][0].update({"warm_allocation_count": 1}))
        self.rejected()

    def test_mixed_width_sequence_and_synthetic_identity_rejected(self):
        self.change_report(lambda r: r["shape_sequence"].reverse(), mode="mixed_widths")
        self.rejected()
        self.build_reports()
        self.change_report(lambda r: r["inputs"][0].update({"sha256": sha(b"wrong widths")}), mode="mixed_widths")
        self.rejected()

    def test_missing_required_binary_and_source_inventory_rejected(self):
        self.snapshot["executables"] = [e for e in self.snapshot["executables"] if not e["path"].startswith("csharp/")]
        write_json(self.measurements / "snapshot.json", self.snapshot)
        self.rejected()

    def test_missing_source_inventories_and_reference_fingerprint_rejected(self):
        for field in ("source_files", "reference_source_files"):
            original = copy.deepcopy(self.snapshot)
            self.snapshot[field] = []
            write_json(self.measurements / "snapshot.json", self.snapshot)
            self.rejected()
            self.snapshot = original
        self.snapshot["reference_source_sha256"] = sha(b"wrong reference source tree")
        write_json(self.measurements / "snapshot.json", self.snapshot)
        self.rejected()

    def test_boolean_samples_and_protocol_counts_rejected(self):
        self.change_report(lambda r: r["inputs"][0]["samples_ms"].__setitem__(0, True))
        self.rejected()
        self.build_reports()
        self.snapshot["rounds"] = True
        write_json(self.measurements / "snapshot.json", self.snapshot)
        self.rejected()

    def test_single_sample_probe_and_no_heap_rounds_are_supported(self):
        self.snapshot.update(rounds=1, heap_rounds=0, warmup=0, samples=1)
        for path in self.measurements.glob("*-r*.json"):
            path.unlink()
        self.build_reports()
        result = SUMMARY.summarize(self.measurements)
        self.assertEqual(result["protocol"]["samples"], 1)
        self.assertNotIn("rust-heap", result["modes"]["rec_graph"]["groups"])

    def test_cli_writes_valid_new_output_and_refuses_overwrite(self):
        output = self.root / "report.json"
        arguments = [str(SCRIPT), "--measurements", str(self.measurements), "--out", str(output)]
        with patch("sys.argv", arguments):
            SUMMARY.main()
        original = output.read_bytes()
        self.assertEqual(json.loads(original)["kind"], "tiny_ocr_performance")
        with patch("sys.argv", arguments), self.assertRaises(FileExistsError):
            SUMMARY.main()
        self.assertEqual(output.read_bytes(), original)

    def test_source_inventory_requires_actual_code_and_build_inputs(self):
        original = copy.deepcopy(self.snapshot)
        self.snapshot["source_files"] = [entry for entry in self.snapshot["source_files"] if not entry["path"].endswith(".rs")]
        self.refresh_source_identity()
        write_json(self.measurements / "snapshot.json", self.snapshot)
        self.rejected()
        self.snapshot = original
        self.snapshot["reference_source_files"] = [entry for entry in self.snapshot["reference_source_files"] if not entry["path"].endswith(".csproj")]
        self.snapshot["reference_source_sha256"] = sha("".join(f"{e['path']}\0{e['sha256']}\n" for e in self.snapshot["reference_source_files"]).encode())
        write_json(self.measurements / "snapshot.json", self.snapshot)
        self.rejected()

    def test_unsafe_identity_paths_rejected(self):
        self.snapshot["source_files"][0]["path"] = "../outside.rs"
        self.refresh_source_identity()
        write_json(self.measurements / "snapshot.json", self.snapshot)
        self.rejected()

    def test_profile_summary_preserves_sampled_stages_and_residuals(self):
        self.upgrade_v2()
        result = SUMMARY.summarize(self.measurements)
        mode = result["modes"]["public_stage_profile"]
        self.assertTrue(mode["observer_instrumented_diagnostic"])
        self.assertFalse(mode["paired_owned_or_copied_scope"])
        self.assertNotIn("csharp", mode["groups"])
        for engine in ("rust-timing", "rust-heap"):
            profiles = mode["groups"][engine]["reports"][0]["stage_profile"]
            self.assertEqual([p["case_id"] for p in profiles], ["sample", "blank"])
            self.assertEqual([p["totals"]["recognized_lines"] for p in profiles], [6, 0])
            self.assertEqual(profiles[0]["mean_ms"]["projection"], 1000 / 3e6)
            self.assertGreater(profiles[0]["unassigned_mean_ms"], 0)
            self.assertGreater(profiles[0]["outer_wrapper_mean_ms"], 0)

    def test_profile_requires_native_integer_sample_count(self):
        self.upgrade_v2()
        self.snapshot["samples"] = 1
        self.build_reports()
        SUMMARY.summarize(self.measurements)
        for value in (True, 1., 0, 2):
            with self.subTest(value=value):
                self.change_report(lambda r: r["stage_profile"][0].update(samples=value), mode="public_stage_profile")
                with self.assertRaisesRegex(ValueError, "profile samples|profile sample count"):
                    SUMMARY.summarize(self.measurements)
                self.build_reports()

    def test_profile_identity_count_and_order_rejected(self):
        self.upgrade_v2()
        for change in (lambda r: r["stage_profile"][0].update(case_id="blank"),
                       lambda r: r["stage_profile"][0].update(artifact_id="line.000.rec.pixels"),
                       lambda r: r["stage_profile"].pop(), lambda r: r["stage_profile"].reverse()):
            self.change_report(change, mode="public_stage_profile")
            with self.assertRaisesRegex(ValueError, "production profile identity|production stage profiles"):
                SUMMARY.summarize(self.measurements)
            self.build_reports()

    def test_profile_fields_and_native_counter_types_rejected(self):
        self.upgrade_v2()
        for change in (lambda t: t.pop("crop_ns"), lambda t: t.update(owned_copy_ns=1),
                       lambda t: t.update(ctc_ns=True), lambda t: t.update(ctc_ns=1.),
                       lambda t: t.update(ctc_ns=-1), lambda t: t.update(ctc_ns=1 << 64)):
            self.change_report(lambda r: change(r["stage_profile"][0]["totals"]), mode="public_stage_profile")
            with self.assertRaisesRegex(ValueError, "stage fields|stage ctc_ns"):
                SUMMARY.summarize(self.measurements)
            self.build_reports()

    def test_profile_disjoint_stages_and_outer_sample_bounds_rejected(self):
        self.upgrade_v2()
        changes = (
            (lambda r: r["stage_profile"][0]["totals"].update(
                projection_ns=r["stage_profile"][0]["totals"]["total_ns"] + 1), "disjoint stages"),
            (lambda r: r["stage_profile"][0]["totals"].update(
                total_ns=int(sum(r["inputs"][0]["samples_ms"]) * 1e6) + 1), "outer sampled calls"),
        )
        for change, message in changes:
            self.change_report(change, mode="public_stage_profile")
            with self.assertRaisesRegex(ValueError, message):
                SUMMARY.summarize(self.measurements)
            self.build_reports()

    def test_profile_equal_timer_bounds_and_zero_residual_supported(self):
        self.upgrade_v2()
        def equal_bound(report):
            row, profile = report["inputs"][0], report["stage_profile"][0]
            row.update(samples_ms=[2.] * 3, p50_ms=2., p95_ms=2., mean_ms=2.)
            profile["totals"].update({key: 0 for key in SUMMARY.PROFILE_STAGES})
            profile["totals"].update(det_inference_ns=6_000_000, total_ns=6_000_000)
            total = [sum(r["samples_ms"][i] for r in report["inputs"]) for i in range(3)]
            report["corpus"].update(samples_ms=total, p50_ms=SUMMARY.percentile(total, .5),
                                    p95_ms=SUMMARY.percentile(total, .95), mean_ms=sum(total) / 3)
        self.change_report(equal_bound, mode="public_stage_profile")
        result = SUMMARY.summarize(self.measurements)
        profile = result["modes"]["public_stage_profile"]["groups"]["rust-timing"]["reports"][0]["stage_profile"][0]
        self.assertEqual(profile["unassigned_mean_ms"], 0)
        self.assertEqual(profile["outer_wrapper_mean_ms"], 0)

    def test_profile_rejects_first_call_warmups_in_line_totals(self):
        self.upgrade_v2()
        self.change_report(lambda r: r["stage_profile"][0]["totals"].update(
            recognized_lines=2 * (1 + self.snapshot["warmup"] + self.snapshot["samples"])), mode="public_stage_profile")
        with self.assertRaisesRegex(ValueError, "recognized line totals"):
            SUMMARY.summarize(self.measurements)

    def test_profile_empty_case_cannot_execute_rec_or_cls(self):
        self.upgrade_v2()
        for key in SUMMARY.PROFILE_RECOGNITION_STAGES:
            self.change_report(lambda r: r["stage_profile"][1]["totals"].update({key: 1}), mode="public_stage_profile")
            with self.assertRaisesRegex(ValueError, "empty ordinary OCR case"):
                SUMMARY.summarize(self.measurements)
            self.build_reports()

    def test_profile_rejects_dense_traces_and_unexpected_profile_on_plain_mode(self):
        self.upgrade_v2()
        self.change_report(lambda r: r.update(node_profile=[{"index": 0}]), mode="public_stage_profile")
        with self.assertRaisesRegex(ValueError, "dense node tracing"):
            SUMMARY.summarize(self.measurements)
        self.build_reports()
        self.change_report(lambda r: r.update(stage_profile=[{}]), mode="ocr_borrowed")
        with self.assertRaisesRegex(ValueError, "unexpected production profiles"):
            SUMMARY.summarize(self.measurements)

    def test_profile_requires_every_frozen_profiling_source_and_digest(self):
        self.upgrade_v2()
        original = copy.deepcopy(self.snapshot)
        for source in sorted(SUMMARY.PROFILE_SOURCE_FILES):
            self.snapshot["source_files"] = [entry for entry in original["source_files"] if entry["path"] != source]
            self.refresh_source_identity()
            write_json(self.measurements / "snapshot.json", self.snapshot)
            with self.assertRaisesRegex(ValueError, "profiling source snapshots"):
                SUMMARY.summarize(self.measurements)
        self.snapshot = original
        write_json(self.measurements / "snapshot.json", self.snapshot)
        target = self.measurements / "source/src/profile.rs"
        target.write_bytes(target.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "Snapshot identity differs"):
            SUMMARY.summarize(self.measurements)

    def test_profile_requires_matching_ordinary_public_line_count(self):
        self.upgrade_v2()
        case = self.manifest["cases"][0]
        artifact = next(a for a in case["artifacts"] if a["id"] == "result.public")
        target = self.reference / artifact["path"]
        target.write_bytes(json.dumps({"detected_count": 2, "lines": [{}]}).encode())
        artifact.update(bytes=target.stat().st_size, sha256=sha(target.read_bytes()))
        write_json(self.reference / "manifest.json", self.manifest)
        self.snapshot["manifest_sha256"] = sha((self.reference / "manifest.json").read_bytes())
        self.build_reports()
        with self.assertRaisesRegex(ValueError, "public line count differs"):
            SUMMARY.summarize(self.measurements)

    def test_profile_requires_public_result_artifact_and_ocr_capture_scope(self):
        self.upgrade_v2()
        original = copy.deepcopy(self.manifest)
        for change in (lambda m: m["contract"].update(capture_scope="rec"),
                       lambda m: m["cases"][0]["artifacts"].__setitem__(
                           -1, m["cases"][0]["artifacts"][-1] | {"id": "result.dense"})):
            self.manifest = copy.deepcopy(original)
            change(self.manifest)
            write_json(self.reference / "manifest.json", self.manifest)
            self.snapshot["manifest_sha256"] = sha((self.reference / "manifest.json").read_bytes())
            write_json(self.measurements / "snapshot.json", self.snapshot)
            for path in self.measurements.glob("*-r*.json"):
                report = json.loads(path.read_bytes())
                report["manifest_sha256"] = self.snapshot["manifest_sha256"]
                write_json(path, report)
            with self.assertRaisesRegex(ValueError, "ordinary OCR public reference"):
                SUMMARY.summarize(self.measurements)

    def test_profile_selected_empty_case_with_one_sample_and_no_heap_supported(self):
        self.snapshot.update(modes=["public_stage_profile"], case="blank", rounds=1,
                             heap_rounds=0, samples=1, warmup=0)
        for path in self.measurements.glob("*-r*.json"):
            path.unlink()
        self.upgrade_v2()
        result = SUMMARY.summarize(self.measurements)
        mode = result["modes"]["public_stage_profile"]
        self.assertEqual(set(mode["groups"]), {"rust-timing"})
        profiles = mode["groups"]["rust-timing"]["reports"][0]["stage_profile"]
        self.assertEqual(len(profiles), 1)
        self.assertEqual(profiles[0]["case_id"], "blank")
        self.assertEqual(profiles[0]["samples"], 1)
        self.assertEqual(profiles[0]["totals"]["recognized_lines"], 0)
        self.assertEqual(profiles[0]["mean_ms"]["projection"], 0)

    def test_legacy_schema_preserves_modes_without_profiling_sources(self):
        self.snapshot["source_files"] = [entry for entry in self.snapshot["source_files"]
                                         if entry["path"] not in SUMMARY.PROFILE_SOURCE_FILES]
        self.refresh_source_identity()
        write_json(self.measurements / "snapshot.json", self.snapshot)
        result = SUMMARY.summarize(self.measurements)
        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(set(result["modes"]), set(LEGACY_MODES))
        self.assertTrue(all("stage_profile" not in report for mode in result["modes"].values()
                            for group in mode["groups"].values() for report in group["reports"]))

    def test_legacy_schema_rejects_new_stage_mode(self):
        self.snapshot["modes"].append("public_stage_profile")
        self.build_reports()
        with self.assertRaisesRegex(ValueError, "require schema 2"):
            SUMMARY.summarize(self.measurements)


if __name__ == "__main__":
    unittest.main()
