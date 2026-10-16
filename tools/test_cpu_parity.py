"""Small tests for the oracle gate itself; run without models or .NET."""
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

import cpu_parity as gate


class GateContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.tolerances = {"tensor_atol": 1e-4, "tensor_rtol": 1e-4,
                           "score_atol": .01, "coordinate_atol": .5}

    def tearDown(self):
        self.temporary.cleanup()

    def tensor(self, directory, name, values, shape=None):
        root = self.root / directory
        root.mkdir(exist_ok=True)
        data = struct.pack("<" + "f" * len(values), *values)
        (root / name).write_bytes(data)
        return {"file": name, "shape": shape or [len(values)], "sha256": hashlib.sha256(data).hexdigest()}

    @staticmethod
    def measured(cases):
        return {"metadata": {"benchmark_environment": {
            "host_name": "test-host", "cpu_identity": "test-cpu", "operating_system": "test-os", "architecture": "x86_64",
            "threads": 1, "corpus_sha256": "abc", "rustc": "test-rustc", "rustflags": "", "build_flags": {}, "reference_lock_sha256": "def",
        }}, "cases": cases}

    def test_reports_first_numerical_divergence_and_error_magnitude(self):
        a = self.tensor("a", "x.f32", [1., 2., 3.])
        b = self.tensor("b", "x.f32", [1., 2.1, 4.])
        result = gate.compare_tensor(self.root / "a", a, self.root / "b", b, self.tolerances)
        self.assertFalse(result["passed"])
        self.assertEqual(result["first_difference"]["index"], 1)
        self.assertEqual(result["max_abs"], 1.)

    def test_reference_process_uses_clean_pinned_cpu_controls(self):
        inherited = {"PATH": "tools", "PPOCR_DUMP_NODES": "1", "PPOCR_DET_THREADS": "99",
                     "SIMD_OCR_BACKEND": "gpu", "SIMD_OCR_NOFUSE": "1", "COMPlus_EnableAVX": "0",
                     "DOTNET_EnableAVX2": "0"}
        with mock.patch.dict(os.environ, inherited, clear=True):
            optimized = gate.reference_environment("auto", 4)
            scalar = gate.reference_environment("reference", 1)
        self.assertEqual(optimized["PATH"], "tools")
        self.assertEqual(optimized["DOTNET_PROCESSOR_COUNT"], "4")
        self.assertEqual(optimized["SIMD_OCR_BACKEND"], "cpu")
        self.assertEqual(optimized["PPOCR_NHWC"], "1")
        self.assertEqual(optimized["SIMD_OCR_NOFUSE"], "0")
        for name in ("PPOCR_DUMP_NODES", "PPOCR_DET_THREADS", "COMPlus_EnableAVX", "DOTNET_EnableAVX2"):
            self.assertNotIn(name, optimized)
        self.assertEqual(scalar["DOTNET_EnableHWIntrinsic"], "0")
        self.assertEqual(scalar["PPOCR_NHWC"], "0")

    def test_import_contracts_require_all_seven_models_and_compare_parameters_exactly(self):
        with self.assertRaisesRegex(RuntimeError, "All seven"):
            gate.compare_imports(self.root / "reference", self.root / "actual")
        for name, count in gate.GRAPH_NODE_COUNTS.items():
            contract = {"schema_version": 1, "raw": {"inputs": ["x"], "outputs": ["y"], "nodes": ["raw"] * gate.RAW_GRAPH_NODE_COUNTS[name]},
                        "normalized": {"inputs": ["x"], "outputs": ["y"], "nodes": [{"axis": 1}] * count}}
            for side in ("reference", "actual"):
                gate.write_json(self.root / side / "imports" / (name + ".json"), contract)
        self.assertTrue(gate.compare_imports(self.root / "reference", self.root / "actual")["passed"])
        path = self.root / "actual/imports/cls.json"
        contract = gate.read_json(path)
        contract["normalized"]["nodes"][0]["axis"] = 2
        gate.write_json(path, contract)
        report = gate.compare_imports(self.root / "reference", self.root / "actual")
        self.assertFalse(report["passed"])
        self.assertEqual(report["models"][0]["first_difference"]["path"], "cls.normalized.nodes[0].axis")

    def test_nonfinite_is_never_equal_even_for_two_nan_tensors(self):
        a = self.tensor("a", "x.f32", [float("nan")])
        b = self.tensor("b", "x.f32", [float("nan")])
        result = gate.compare_tensor(self.root / "a", a, self.root / "b", b, self.tolerances)
        self.assertEqual(result["reason"], "nonfinite")

    def test_zero_tolerances_require_finite_f32_bits_including_the_sign_of_zero(self):
        a = self.tensor("a", "x.f32", [0., 1.])
        b = self.tensor("b", "x.f32", [-0., 1.])
        exact = {**self.tolerances, "tensor_atol": 0., "tensor_rtol": 0.}
        self.assertTrue(gate.compare_tensor(self.root / "a", a, self.root / "b", b, self.tolerances)["passed"])
        result = gate.compare_tensor(self.root / "a", a, self.root / "b", b, exact)
        self.assertFalse(result["passed"])
        self.assertEqual(result["unequal"], 1)
        self.assertEqual(result["first_difference"]["index"], 0)
        self.assertEqual(result["first_difference"]["expected_bits"], "0x00000000")
        self.assertEqual(result["first_difference"]["actual_bits"], "0x80000000")
        a = self.tensor("a", "nan.f32", [float("nan")])
        b = self.tensor("b", "nan.f32", [float("nan")])
        self.assertEqual(gate.compare_tensor(self.root / "a", a, self.root / "b", b, exact)["reason"], "nonfinite")

    def test_source_archive_keeps_verified_bytes_and_rejects_drift(self):
        source = self.root / "source"
        source.mkdir()
        (source / "dirty.rs").write_text("modified working tree", encoding="utf-8")
        snapshot = gate.source_snapshot(source, [source / "dirty.rs"])
        with mock.patch.object(gate, "ROOT", source):
            gate.archive_sources(snapshot, self.root / "archived")
            self.assertEqual((self.root / "archived/dirty.rs").read_bytes(), (source / "dirty.rs").read_bytes())
            self.assertEqual(gate.read_json(self.root / "archived/snapshot.json"), snapshot)
            gate.verify_source_archive(snapshot, self.root / "archived")
            (self.root / "archived/dirty.rs").write_text("corrupt archived source", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Archived source file changed"):
                gate.verify_source_archive(snapshot, self.root / "archived")
            (source / "dirty.rs").write_text("changed after compile", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "changed before archival"):
                gate.archive_sources(snapshot, self.root / "rejected")

    def test_corruption_shape_mismatch_and_path_escape_are_errors(self):
        a = self.tensor("a", "x.f32", [1., 2.])
        (self.root / "a" / "x.f32").write_bytes(b"bad")
        with self.assertRaises(RuntimeError):
            gate.load_f32(self.root / "a", a)
        with self.assertRaises(RuntimeError):
            gate.checked_child(self.root / "a", "../secret.f32")

    def test_text_and_alignment_are_exact_but_scores_and_coordinates_have_explicit_budgets(self):
        expected = {"text": "abc", "score": .5, "points": [[1., 2.]], "start_column": 1}
        close = {"text": "abc", "score": .505, "points": [[1.4, 2.]], "start_column": 1}
        self.assertFalse(gate.compare_result(expected, close, self.tolerances))
        close["text"] = "abd"
        self.assertEqual(gate.compare_result(expected, close, self.tolerances)[0]["path"], "result.text")

    def test_missing_case_is_fatal_instead_of_a_skip(self):
        for name, cases in (("a", [{"id": "required"}]), ("b", [])):
            directory = self.root / name
            directory.mkdir()
            (directory / "capture.json").write_text(json.dumps({"schema_version": 1, "cases": cases}))
        with self.assertRaises(RuntimeError):
            gate.compare_bundles(self.root / "a", self.root / "b", self.tolerances)

    def test_matching_omissions_cannot_remove_mandatory_public_cases(self):
        for name in ("a", "b"):
            gate.write_json(self.root / name / "capture.json", {"schema_version": 1, "cases": [{"id": "classify/sample"}]})
        with self.assertRaisesRegex(RuntimeError, "mandatory corpus"):
            gate.compare_bundles(self.root / "a", self.root / "b", self.tolerances, {"classify/sample", "ocr/rec/sample"})

    def test_real_model_batch_gate_requires_complete_independent_exact_results(self):
        result = {"text": "中", "score": .5, "emitted_count": 1, "resized_width": 48, "tensor_width": 64,
                  "time_steps": 8, "ctc_spans": [{"text": "中", "score": .5, "start_column": 1, "end_column": 3}]}
        cases = [{"id": "batch/" + name, "image_ids": ["sample", "tiny", "blank", "sample"],
                  "results": [result] * 4, "standalone": [result] * 4} for name in ("rec", "small_rec", "medium_rec")]
        for side in ("a", "b"):
            gate.write_json(self.root / side / "capture.json", {"schema_version": 1, "cases": cases})
        self.assertTrue(gate.compare_batch_bundles(self.root / "a", self.root / "b", self.tolerances)["passed"])
        capture = gate.read_json(self.root / "b/capture.json")
        capture["cases"][0]["results"][0]["score"] += 0.000001
        gate.write_json(self.root / "b/capture.json", capture)
        report = gate.compare_batch_bundles(self.root / "a", self.root / "b", self.tolerances)
        self.assertFalse(report["passed"])
        changed = next(case for case in report["cases"] if case["id"] == "batch/rec")
        self.assertTrue(changed["standalone_vs_batch_errors"][1])
        capture["cases"].pop()
        gate.write_json(self.root / "b/capture.json", capture)
        with self.assertRaisesRegex(RuntimeError, "all three"):
            gate.compare_batch_bundles(self.root / "a", self.root / "b", self.tolerances)

    def test_matching_stripped_payloads_and_graph_traces_are_never_a_pass(self):
        entry = self.tensor("a", "x.f32", [1.])
        invalid = [
            {"id": "preprocess/det/sample"},
            {"id": "classify/sample", "result": {}},
            {"id": "graph/cls/0", "tensor": entry, "nodes": []},
            {"id": "graph/cls/0", "tensor": entry, "nodes": [
                {"index": 0, "name": "first", "op_type": "Conv", "tensor": entry}]},
        ]
        for case in invalid:
            with self.subTest(id=case["id"], payload=sorted(case)):
                for name in ("a", "b"):
                    (self.root / name).mkdir(exist_ok=True)
                    gate.write_json(self.root / name / "capture.json", {
                        "schema_version": 1, "engine": "rust", "mode": "auto", "cases": [case]})
                with self.assertRaises(RuntimeError):
                    gate.compare_bundles(self.root / "a", self.root / "b", self.tolerances)

    def test_public_field_types_reject_bool_labels_and_missing_numeric_scores(self):
        capture = {"engine": "rust", "mode": "auto"}
        valid = {"id": "classify/sample", "result": {"label": 0, "score": .5, "resized_width": 160}}
        gate.validate_case_payload(capture, valid)
        for label, score in ((False, .5), (0, None), (0, "0.5"), (0, float("nan"))):
            with self.subTest(label=label, score=score):
                invalid = {"id": valid["id"], "result": {"label": label, "score": score, "resized_width": 160}}
                with self.assertRaises(RuntimeError):
                    gate.validate_case_payload(capture, invalid)

    def test_performance_uses_median_and_requires_matching_workloads(self):
        before = self.measured([{"id": "x", "benchmark": {"warm_ns": [10, 100, 10]}}])
        after = self.measured([{"id": "x", "benchmark": {"warm_ns": [5, 5, 900]}}])
        result = gate.speedups(before, after, 1.1)
        self.assertTrue(result["passed"])
        self.assertEqual(result["geometric_mean_speedup"], 2.)
        with self.assertRaises(RuntimeError):
            gate.speedups(before, self.measured([]), 1.)

    def test_correctness_only_captures_cannot_become_performance_claims(self):
        capture = {"cases": [{"id": "graph/rec/0", "benchmark": {"measured": False, "reason": "correctness-only"}}]}
        with self.assertRaisesRegex(RuntimeError, "no inference timing samples"):
            gate.speedups(capture, capture, 1.0)

    def test_e2e_regression_cannot_hide_behind_a_faster_geometric_mean(self):
        before = self.measured([{"id": "graph/fast", "benchmark": {"warm_ns": [100]}},
                                {"id": "ocr/page", "benchmark": {"warm_ns": [100]}}])
        after = self.measured([{"id": "graph/fast", "benchmark": {"warm_ns": [10]}},
                               {"id": "ocr/page", "benchmark": {"warm_ns": [104]}}])
        result = gate.speedups(before, after, 1.05)
        self.assertGreater(result["geometric_mean_speedup"], 1.05)
        self.assertFalse(result["passed"])
        self.assertEqual(result["e2e_regressions"][0]["id"], "ocr/page")

    def test_host_compiler_budget_and_corpus_are_required_for_performance_claims(self):
        before = self.measured([{"id": "x", "benchmark": {"warm_ns": [100]}}])
        for key in before["metadata"]["benchmark_environment"]:
            after = self.measured([{"id": "x", "benchmark": {"warm_ns": [50]}}])
            after["metadata"]["benchmark_environment"][key] = "changed"
            with self.assertRaises(RuntimeError, msg=key):
                gate.speedups(before, after, 1.)
        with self.assertRaises(RuntimeError):
            gate.speedups({"cases": []}, {"cases": []}, 1.)

    def test_source_fingerprint_includes_uncommitted_bytes_and_has_stable_order(self):
        first, second = self.root / "first.rs", self.root / "second.cs"
        first.write_text("original")
        second.write_text("other")
        before = gate.source_snapshot(self.root, [second, first])
        self.assertEqual(before, gate.source_snapshot(self.root, [first, second, first]))
        first.write_text("uncommitted change")
        after = gate.source_snapshot(self.root, [first, second])
        self.assertNotEqual(before["sha256"], after["sha256"])
        self.assertEqual(before["files"]["second.cs"], after["files"]["second.cs"])
        with self.assertRaises(RuntimeError):
            gate.source_snapshot(self.root, [])

    def test_replay_checks_all_raw_corpus_bytes_before_inference(self):
        pixels = bytes(range(6))
        (self.root / "image.bgr").write_bytes(pixels)
        gate.write_json(self.root / "corpus.json", {"schema_version": 1, "pixel_format": "bgr24", "images": [
            {"id": "image", "file": "image.bgr", "width": 2, "height": 1, "sha256": hashlib.sha256(pixels).hexdigest()}]})
        gate.validate_corpus(self.root)
        (self.root / "image.bgr").write_bytes(bytes(reversed(pixels)))
        with self.assertRaises(RuntimeError):
            gate.validate_corpus(self.root)


if __name__ == "__main__":
    unittest.main()
