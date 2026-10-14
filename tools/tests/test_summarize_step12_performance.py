"""Small synthetic fixtures; no model inference or benchmark processes."""
import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "summarize-step12-performance.py"
SPEC = importlib.util.spec_from_file_location("step12_summary", SCRIPT)
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def counters(measured, calls=0, retained=0, peak=0):
    return {"warm_allocation_count": calls if measured else None,
            "warm_allocation_bytes": calls * 64 if measured else None,
            "warm_retained_heap_bytes": retained if measured else None,
            "warm_peak_heap_growth_bytes": peak if measured else None}


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.reference = self.root / "reference"
        self.reference.mkdir()
        self.measurements = self.root / "measurements"
        self.measurements.mkdir()
        artifacts = []
        def asset(identifier, dimensions, contents=None):
            value = {"id": identifier, "path": identifier + ".json", "bytes": 1,
                     "shape": dimensions, "sha256": sha(identifier.encode())}
            if contents is not None:
                write_json(self.reference / value["path"], contents)
                value.update(SUMMARY.file_identity(self.reference / value["path"]))
                value["path"] = identifier + ".json"
            artifacts.append(value)
        asset("input.metadata", [], {"width": 2, "height": 2, "stride": 6,
                                    "origin": "synthetic test pixels", "pixel_format": "bgr24"})
        asset("input.pixels", [2, 6])
        asset("step12.public", [], {"detected_count": 1, "lines": [{"applied_rotation_degrees": 180}]})
        for identifier, dimensions in (("det.input", [1, 3, 32, 32]),
                                       ("det.probabilities", [1, 1, 32, 32]),
                                       ("line.000.crop", [2, 2, 3]),
                                       ("line.000.cls.input", [1, 3, 80, 160]),
                                       ("line.000.cls.probabilities", [1, 2]),
                                       ("line.000.cls.public", []),
                                       ("line.000.rec.pixels", [2, 2, 3])):
            asset(identifier, dimensions)
        models = []
        for identifier in ("det", "cls"):
            path = self.reference / (identifier + ".onnx")
            path.write_bytes(identifier.encode())
            models.append({"id": identifier, "path": path.name,
                           "bytes": path.stat().st_size, "sha256": sha(path.read_bytes())})
        manifest = {"schema_version": 1,
                    "contract": {"profile": "canonical", "backend": "cpu",
                                 "capture_scope": "step12", "model": "tiny"},
                    "provenance": {"git": {"head": "native-head"}},
                    "models": models, "cases": [{"id": "sample", "kind": "step12", "artifacts": artifacts}]}
        write_json(self.reference / "manifest.json", manifest)
        inventory = [{"path": "src/runtime.rs", "sha256": sha(b"source")},
                     {"path": "Cargo.lock", "sha256": sha(b"lock") }]
        write_json(self.measurements / "source-files.json", inventory)
        for filename in ("step12-timing.exe", "step12-heap.exe"):
            (self.measurements / filename).write_bytes(filename.encode())
        self.environment = {"runtime_controls": SUMMARY.CONTROLS,
                            "source_git_head": "rust-head", "reference_git_head": "native-head",
                            "source_files_sha256": sha("\n".join(
                                f"{a['path']} {a['sha256']}" for a in inventory).encode()),
                            "rust_timing_sha256": sha(b"step12-timing.exe"),
                            "rust_heap_sha256": sha(b"step12-heap.exe"),
                            "csharp_exporter_sha256": sha(b"exporter"),
                            "csharp_native_sha256": sha(b"native"), "os": "synthetic"}
        self.dataset, self.artifacts = SUMMARY.load_dataset(self.reference, None, {})
        self.summary = {"schema_version": 1, "reference": str(self.reference), "case": None,
                        "rounds": 6, "heap_rounds": 3, "warmup": 30, "samples": 100,
                        "modes": list(SUMMARY.MODES), "environment": self.environment,
                        "process_orders": [], "runs": []}
        for mode in SUMMARY.MODES:
            for number in range(1, 7):
                self.summary["process_orders"].append({"mode": mode, "round": number,
                    "order": ["csharp", "rust-timing"] if number % 2 else ["rust-timing", "csharp"]})
            for engine in SUMMARY.ENGINES:
                for number in range(1, 4 if engine == "rust-heap" else 7):
                    report = self.raw_report(mode, engine, number)
                    write_json(self.measurements / f"{mode}-{engine}-round-{number:02}.json", report)
                    self.summary["runs"].append({"mode": mode, "engine": engine, "round": number,
                        **{key: report.get(key) for key in ("load_ms", "prepare_ms", "warm_allocation_count",
                            "warm_allocation_bytes", "activation_bytes", "retained_activation_bytes",
                            "observed_process_peak_working_set_bytes")},
                        "p50_ms": report["corpus"]["p50_ms"], "p95_ms": report["corpus"]["p95_ms"]})
        write_json(self.measurements / "summary.json", self.summary)

    def raw_report(self, mode, engine, number):
        heap = engine == "rust-heap"
        rust = engine != "csharp"
        rows = []
        for identity in SUMMARY.expected_rows(mode, self.dataset, self.artifacts):
            values = [number + index / 1000 for index in range(100)]
            row = copy.deepcopy(identity) | {"first_call_ms": number + 5., "samples_ms": values,
                "p50_ms": SUMMARY.percentile(values, .5), "p95_ms": SUMMARY.percentile(values, .95),
                "mean_ms": sum(values) / 100}
            row.update(counters(heap, calls=1 if mode in ("mixed_shapes", "owned_copies") else 0))
            if rust:
                row["first_call_allocations"] = counters(heap, calls=2, peak=256)
            else:
                row["warm_allocation_bytes"] = 300
            rows.append(row)
        corpus = [sum(row["samples_ms"][index] for row in rows) for index in range(100)]
        report = {"schema_version": 1, "runtime": "rust" if rust else "csharp", "profile": "canonical",
            "reference": str(self.reference), "mode": mode, "engine": engine, "round": number,
            "warmup": 30, "samples": 100, "shape_sequence": SUMMARY.MIXED_SHAPES if mode == "mixed_shapes" else None,
            "reshape_samples_ms": [v / 10 for v in corpus] if mode == "mixed_shapes" else [],
            "manifest_sha256": self.dataset["manifest"]["sha256"],
            "model_sha256": self.dataset["models"][0]["sha256"],
            "cls_model_sha256": self.dataset["models"][1]["sha256"],
            "load_ms": 10 + number, "prepare_ms": 20 + number,
            "load_allocations": counters(heap, calls=3, retained=128, peak=192),
            "prepare_allocations": counters(heap, calls=4, retained=256, peak=512),
            "activation_bytes": 1024 if rust else None, "retained_activation_bytes": 2048 if rust else None,
            "retained_workspace_bytes": 4096 if rust else None,
            "crop_capacity_bytes": 8 if rust else None, "line_capacity_bytes": 2 if rust else None,
            "observed_process_peak_working_set_bytes": 100000,
            "process_memory_semantics": "synthetic RSS", "activation_memory_semantics": "synthetic buffers",
            "warm_allocation_count": sum(r["warm_allocation_count"] for r in rows) if heap else None,
            "warm_allocation_bytes": sum(r["warm_allocation_bytes"] for r in rows) if heap or not rust else None,
            "measurement": "synthetic contiguous row blocks", "inputs": rows,
            "corpus": {"samples_ms": corpus, "p50_ms": SUMMARY.percentile(corpus, .5),
                       "p95_ms": SUMMARY.percentile(corpus, .95), "mean_ms": sum(corpus) / 100}}
        if rust:
            report["instrumentation"] = "separate allocator-instrumented executable; exclude these timings" if heap else "standard allocator; uninstrumented latency"
            report["session_preparations"] = [{"case_id": "sample", "stage": "det_graph", "elapsed_ms": .1,
                "activation_bytes": 1024, "retained_activation_bytes": 2048,
                "allocations": counters(heap, calls=1, peak=2048), "shape": [1, 3, 32, 32]}]
        else:
            report.update({"load_allocation_bytes": 10, "prepare_allocation_bytes": 20,
                "instrumentation": "Current-thread managed allocation volume; native allocations and retained heap are not measured."})
        return report

    def change_run(self, mode, engine, number, change):
        path = self.measurements / f"{mode}-{engine}-round-{number:02}.json"
        report = SUMMARY.read_json(path)
        change(report)
        write_json(path, report)

    def test_all_scopes_and_statistics(self):
        result = SUMMARY.build_report([self.measurements])
        modes = {m["mode"]: m for m in result["measurement_sets"][0]["measurements"]}
        self.assertEqual(result["measurement_sets"][0]["dataset"]["total_line_count"], 1)
        self.assertEqual(modes["det_graph"]["rust"]["corpus"]["process_p50_ms"]["median"], 3.5495)
        self.assertEqual(modes["det_graph"]["rust"]["corpus"]["pooled_600_samples_ms"]["count"], 600)
        self.assertTrue(modes["stages"]["rust_allocations"]["zero_allocation_gate"]["passed"])
        self.assertFalse(modes["mixed_shapes"]["rust_allocations"]["zero_allocation_gate"]["applicable"])
        self.assertEqual(modes["mixed_shapes"]["rust"]["reshape_only"]["pooled_600_samples_ms"]["count"], 600)
        self.assertEqual(modes["pipeline"]["rust_allocations"]["load"]["peak_heap_growth_bytes"]["median"], 192)
        self.assertEqual(len(modes["stages"]["rust"]["rows"]), 7)
        self.assertEqual(modes["det_graph"]["rust"]["session_preparations"][0]["additional_metadata_in_round_order"][0]["shape"], [1, 3, 32, 32])
        self.assertTrue(all("sha256" in f and "bytes" in f for f in result["raw_files"]))

    def test_output_refuses_overwrite(self):
        output = self.root / "report.json"
        output.write_text("keep", encoding="utf-8")
        with self.assertRaises(SystemExit) as error:
            SUMMARY.main(["--measurements", str(self.measurements), "--out", str(output)])
        self.assertEqual(error.exception.code, 2)
        self.assertEqual(output.read_text(), "keep")

    def test_cli_writes_new_valid_json(self):
        output = self.root / "report.json"
        self.assertEqual(SUMMARY.main(["--measurements", str(self.measurements), "--out", str(output)]), 0)
        report = SUMMARY.read_json(output)
        self.assertEqual(report["protocol"]["timing_processes_per_engine_per_mode"], 6)
        self.assertEqual(report["summarizer"]["sha256"], sha(SCRIPT.read_bytes()))

    def test_missing_round_and_duplicate_filename_rejected(self):
        path = self.measurements / "det_graph-rust-timing-round-06.json"
        path.rename(self.measurements / "det_graph-rust-timing-round-6.json")
        duplicate = self.measurements / "det_graph-rust-timing-round-01.json"
        (self.measurements / "det_graph-rust-timing-round-1.json").write_bytes(duplicate.read_bytes())
        with self.assertRaisesRegex(SUMMARY.InvalidReport, "duplicate raw"):
            SUMMARY.build_report([self.measurements])
        duplicate.unlink()
        (self.measurements / "det_graph-rust-timing-round-6.json").unlink()
        with self.assertRaisesRegex(SUMMARY.InvalidReport, "raw report files"):
            SUMMARY.build_report([self.measurements])

    def test_identity_tampering_rejected(self):
        self.change_run("stages", "csharp", 1, lambda r: r["inputs"][0].update({"shape": [1, 3, 64, 32]}))
        with self.assertRaisesRegex(SUMMARY.InvalidReport, "identities differ"):
            SUMMARY.build_report([self.measurements])

    def test_source_and_snapshot_hashes_rejected(self):
        (self.measurements / "step12-heap.exe").write_bytes(b"tampered")
        with self.assertRaisesRegex(SUMMARY.InvalidReport, "executable hash"):
            SUMMARY.build_report([self.measurements])

    def test_source_fingerprint_and_protocol_rejected(self):
        path = self.measurements / "source-files.json"
        inventory = SUMMARY.read_json(path)
        inventory[0]["sha256"] = sha(b"different source")
        write_json(path, inventory)
        with self.assertRaisesRegex(SUMMARY.InvalidReport, "fingerprint differs"):
            SUMMARY.build_report([self.measurements])
        self.summary["warmup"] = 3
        write_json(self.measurements / "summary.json", self.summary)
        with self.assertRaisesRegex(SUMMARY.InvalidReport, "warmup=30"):
            SUMMARY.build_report([self.measurements])

    def test_fixed_allocations_and_shape_sequence_rejected(self):
        self.change_run("det_graph", "rust-heap", 1, lambda r: r["inputs"][0].update(counters(True, calls=1)))
        with self.assertRaisesRegex(SUMMARY.InvalidReport, "total warm_allocation"):
            SUMMARY.build_report([self.measurements])
        self.change_run("det_graph", "rust-heap", 1, lambda r: r.update({"warm_allocation_count": 1, "warm_allocation_bytes": 64}))
        with self.assertRaisesRegex(SUMMARY.InvalidReport, "zero-allocation gate failed"):
            SUMMARY.build_report([self.measurements])
        self.change_run("det_graph", "rust-heap", 1, lambda r: r.update({"warm_allocation_count": 0, "warm_allocation_bytes": 0}))
        self.change_run("det_graph", "rust-heap", 1, lambda r: r["inputs"][0].update(counters(True)))
        self.change_run("mixed_shapes", "rust-timing", 1, lambda r: r["shape_sequence"].reverse())
        with self.assertRaisesRegex(SUMMARY.InvalidReport, "sequence differs"):
            SUMMARY.build_report([self.measurements])

    def test_pooled_samples_are_not_process_p95(self):
        result = SUMMARY.build_report([self.measurements])
        corpus = result["measurement_sets"][0]["measurements"][0]["rust"]["corpus"]
        self.assertNotEqual(corpus["process_p95_ms"]["median"], corpus["pooled_600_samples_ms"]["p95_ms"])

    def test_raw_only_owned_copies(self):
        owned = self.root / "owned"
        owned.mkdir()
        write_json(owned / "identity.json", {key: self.environment[key] for key in (
            "rust_timing_sha256", "rust_heap_sha256", "source_files_sha256")})
        for engine in ("rust-timing", "rust-heap"):
            for number in range(1, 4 if engine == "rust-heap" else 7):
                report = self.raw_report("owned_copies", engine, number)
                del report["engine"]
                del report["round"]
                write_json(owned / f"owned_copies-{engine}-round-{number:02}.json", report)
        result = SUMMARY.build_report([self.measurements], owned)
        self.assertTrue(result["owned_copies"]["executable_identity_verified"])
        self.assertNotIn("csharp", result["owned_copies"])
        self.assertFalse(result["owned_copies"]["rust_allocations"]["zero_allocation_gate"]["applicable"])
        self.assertEqual(result["owned_copies"]["rust_allocations"]["rows"][0]["warm"]["allocation_count"]["median"], 1)

    def test_json_duplicate_keys_and_nonfinite_rejected(self):
        path = self.root / "invalid.json"
        path.write_text('{"mode": "a", "mode": "b"}')
        with self.assertRaisesRegex(SUMMARY.InvalidReport, "duplicate JSON"):
            SUMMARY.read_json(path)
        path.write_text('{"p50_ms": NaN}')
        with self.assertRaisesRegex(SUMMARY.InvalidReport, "non-finite"):
            SUMMARY.read_json(path)


if __name__ == "__main__":
    unittest.main()
