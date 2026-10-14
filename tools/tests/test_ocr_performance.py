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
MODES = ("rec_graph", "rec_public_owned", "rec_public_borrowed", "ocr_owned",
         "ocr_borrowed", "mixed_widths", "owned_copies", "node_profile")
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
                         "tools/reference-export/Program.cs")]
        reference_source_files = [saved(self.measurements / "reference-source", name, name.encode()) for name in
                                  ("src/Sdcb.SimdPaddleOCR/PaddleOcrRecognizer.cs",
                                   "src/Sdcb.SimdPaddleOCR/Sdcb.SimdPaddleOCR.csproj")]
        self.snapshot = {"schema_version": 1, "kind": "tiny_ocr_measurement_snapshot",
                         "reference": str(self.reference), "manifest_sha256": sha((self.reference / "manifest.json").read_bytes()),
                         "rounds": 2, "heap_rounds": 1, "warmup": 2, "samples": 3,
                         "case": None, "modes": list(MODES), "os": "synthetic", "architecture": "X64",
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
            data = b"".join(bytes((index * 17 + width) % 251 for index in range(width * 48 * 3))
                            for width in WIDTHS)
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
        result = {"schema_version": 1, "kind": "tiny_ocr_benchmark", "runtime": "rust" if rust else "csharp",
                  "profile": "canonical", "mode": mode, "round": round_id, "executable_kind": engine,
                  "executable_sha256": binaries[executable]["sha256"],
                  "csharp_native_sha256": binaries["csharp/Sdcb.SimdPaddleOCR.dll"]["sha256"],
                  "warmup": self.snapshot["warmup"], "samples": self.snapshot["samples"],
                  "model_sha256": assets["rec"]["sha256"], "dictionary_sha256": assets["dictionary"]["sha256"],
                  "det_model_sha256": assets["det"]["sha256"], "cls_model_sha256": assets["cls"]["sha256"],
                  "manifest_sha256": self.snapshot["manifest_sha256"], "instrumented": heap,
                  "inputs": rows, "load_ms": 10., "prepare_ms": 20., "checksum": 4.,
                  "activation_bytes": 1024 if rust else None, "retained_activation_bytes": 2048 if rust else None,
                  "workspace_bytes": 4096 if rust else None, "process_resident_high_water_bytes": 10000,
                  "shape_sequence": [[1, 3, 48, width] for width in WIDTHS] if mode == "mixed_widths" else [],
                  "corpus": {"samples_ms": total, "p50_ms": SUMMARY.percentile(total, .5),
                             "p95_ms": SUMMARY.percentile(total, .95), "mean_ms": sum(total) / len(total)}}
        if rust:
            result["load_allocations"] = counters(heap, calls=3, peak=256)
            result["prepare_allocations"] = counters(heap, calls=4, peak=512)
            result["session_preparations"] = []
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
        result = SUMMARY.summarize(self.measurements)
        self.assertEqual(set(result["modes"]), set(MODES))
        for mode, value in result["modes"].items():
            self.assertEqual(value["paired_owned_or_copied_scope"], mode in PAIRED)
            self.assertEqual("csharp" in value["groups"], mode in PAIRED)
            self.assertEqual(len(value["groups"]["rust-timing"]["reports"]), 2)
            self.assertEqual(len(value["groups"]["rust-heap"]["reports"]), 1)
        timing = result["modes"]["rec_graph"]["groups"]["rust-timing"]
        self.assertNotEqual(timing["median_process_p95_ms"], timing["pooled_p95_ms"])

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


if __name__ == "__main__":
    unittest.main()
