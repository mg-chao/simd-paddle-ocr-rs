"""Allocation gate failures must never become uninstrumented or partial successes."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "assert-allocation-smoke.py"
SPEC = importlib.util.spec_from_file_location("allocation_smoke", SCRIPT)
SMOKE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SMOKE)


class AllocationSmokeTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.executable = self.root / "benchmark"
        self.executable.write_bytes(b"instrumented binary identity")
        self.manifest = {"cases": [{"id": "page", "artifacts": [
            {"id": "input.pixels", "sha256": "a" * 64, "shape": [2, 6]},
            {"id": "line.000.rec.pixels", "sha256": "b" * 64, "shape": [48, 33, 3]},
        ]}]}
        (self.root / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def report(self, mode):
        artifact = self.manifest["cases"][0]["artifacts"][mode == "rec_public_borrowed"]
        row = {"case_id": "page", "artifact_id": artifact["id"], "sha256": artifact["sha256"],
               "shape": artifact["shape"], "stage": mode, "samples_ms": [1., 2., 3.]}
        row.update({counter: 0 for counter in SMOKE.COUNTERS})
        if mode == "mixed_widths":
            row.update(case_id="mixed-widths", artifact_id="rec.pixels", shape=[])
        return {"instrumented": True, "runtime": "rust", "mode": mode, "warmup": 3, "samples": 3,
                "manifest_sha256": SMOKE.sha(self.root / "manifest.json"),
                "executable_sha256": SMOKE.sha(self.executable), "inputs": [row],
                "shape_sequence": [[1, 3, 48, width] for width in range(32, 385, 32)] + [[1, 3, 48, 32]]}

    def validate(self, report, mode):
        SMOKE.validate(report, self.root, mode, self.executable)

    def test_all_borrowed_modes_accept_complete_instrumented_zero_counters(self):
        for mode in SMOKE.MODES:
            with self.subTest(mode=mode):
                self.validate(self.report(mode), mode)

    def test_each_counter_rejects_growth_null_boolean_and_missing_values(self):
        for counter in SMOKE.COUNTERS:
            for value in (1, None, False, "0"):
                with self.subTest(counter=counter, value=value):
                    report = self.report("ocr_borrowed")
                    report["inputs"][0][counter] = value
                    with self.assertRaises(ValueError):
                        self.validate(report, "ocr_borrowed")
            report = self.report("ocr_borrowed")
            del report["inputs"][0][counter]
            with self.assertRaises(ValueError):
                self.validate(report, "ocr_borrowed")

    def test_uninstrumented_empty_or_omitted_rows_fail(self):
        for edit in ({"instrumented": False}, {"inputs": []}, {"inputs": None}):
            with self.subTest(edit=edit):
                report = self.report("ocr_borrowed") | edit
                with self.assertRaises(ValueError):
                    self.validate(report, "ocr_borrowed")

    def test_report_identities_protocol_and_input_coverage_are_required(self):
        for field, value in (("runtime", "csharp"), ("manifest_sha256", "0" * 64),
                             ("executable_sha256", "0" * 64), ("mode", "mixed_widths"),
                             ("warmup", 0), ("samples", 2)):
            report = self.report("ocr_borrowed")
            report[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate(report, "ocr_borrowed")
        for edit in ({"case_id": "different"}, {"sha256": "0" * 64}, {"shape": [1]}, {"samples_ms": [1.]}):
            report = self.report("ocr_borrowed")
            report["inputs"][0].update(edit)
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                self.validate(report, "ocr_borrowed")

    def test_mixed_widths_require_distinct_narrow_wide_shapes(self):
        for sequence in ([], [[1, 3, 48, 32]] * 14, [[1, 3, 48, width] for width in range(64, 385, 32)],
                         [[1, 3, 48, width] for width in range(32, 321, 32)]):
            report = self.report("mixed_widths")
            report["shape_sequence"] = sequence
            with self.subTest(sequence=sequence), self.assertRaises(ValueError):
                self.validate(report, "mixed_widths")

    def test_non_numeric_nonfinite_or_negative_measurements_fail(self):
        for value in (None, True, "1", float("nan"), float("inf"), -1):
            report = self.report("ocr_borrowed")
            report["inputs"][0]["samples_ms"][1] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.validate(report, "ocr_borrowed")


if __name__ == "__main__":
    unittest.main()
