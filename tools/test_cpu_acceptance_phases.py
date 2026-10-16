"""Disposable phase receipts only; no models, timing samples or real reservations."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import uuid

import cpu_acceptance as acceptance
import cpu_parity as gate


class PhaseAccountingContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.ledger = self.root / "fake-ledger.json"
        self.output = self.root / "fake-round"
        self.output.mkdir()
        gate.write_json(self.output / "plan.json", {"diagnostic_schedule": "fake only"})
        self.plan_sha = gate.sha256(self.output / "plan.json")
        self.round_id = str(uuid.uuid4())
        self.policy = acceptance.read_round_policy(self.ledger, "fake-task")
        self.report = {"complete": False, "frozen_plan_sha256": self.plan_sha}
        self.reference, self.capture = self.fixture()

    def tearDown(self):
        self.temporary.cleanup()

    def fixture(self, mode="latency", budget=1):
        workers = budget if mode == "throughput" else 1
        reference = {}
        cases = []
        # Literal stage names are pinned PipelineProfiler.cs's independent 21 slots.
        names = ["det_preprocess", "det_graph", "det_postprocess", "crop", "cls_preprocess", "cls_graph", "rec_preprocess",
            "rec_graph", "rec_postprocess", "lines_wall", "crop_setup", "rec_acquire", "cls_acquire", "rec_decode",
            "rec_release", "cls_release", "rec_cache_get", "rec_rent", "rec_pool", "rec_reshape", "det_unclip"]
        for family in ("rec", "small_rec", "medium_rec"):
            for image in ("sample", "dense-multilingual", "text-long"):
                id_ = f"{family}/{image}"
                result = {"text": "中国\nOCR", "lines": [{"text": "中国"}, {"text": "OCR"}]}
                reference[id_] = {"id": id_, "result": copy.deepcopy(result)}
                cases.append({"id": id_, "result": result, "line_count": 2, "model_load_ns": 3, "compile_requests_ns": 4,
                    "phase": {"first_run_wall_ns": 5, "includes_reshape_and_instrumentation": True,
                        "cold_contract": "first-run-of-this-input-after-explicit-diagnostic-session-creation; earlier-shape-caches-may-exist",
                        "diagnostic_session_creation_ns": 6, "diagnostic_session_creation_calls": budget * 3,
                        "reshape_ns": 60, "reshape_stages": [
                            {"stage": "det", "nanoseconds": 10, "calls": workers, "failed_calls": 0},
                            {"stage": "cls", "nanoseconds": 20, "calls": workers * 2, "failed_calls": 0},
                            {"stage": "rec", "nanoseconds": 30, "calls": workers * 2, "failed_calls": 0}],
                        "reshape_time_kind": "work-sum; may-overlap-across-workers; never-subtract-from-wall-time",
                        "pinned_pipeline_stages": [{"stage": stage, "nanoseconds": 0, "calls": 0} for stage in names],
                        "post_run_owned_copy_ns": 0, "ownership_contract": "pinned-api-intrinsically-owned-no-post-run-copy-v1",
                        "ownership_explanation": "The pinned API allocates owned output internally and needs no external copy",
                        "post_run_text_publication_ns": 7,
                        "text_publication_contract": "actual-PaddleOcrResult.Text-string.Join-once-per-worker-after-Run; separate-diagnostic-work-sum",
                        "official_timing_changed": False}})
        capture = {"schema_version": 1, "complete": True, "engine": "csharp", "phase_only": True, "official_samples": False,
            "mode": mode, "budget": budget, "workers": workers, "capacity_per_request": 1 if mode == "throughput" else budget,
            "phase_round_id": self.round_id, "phase_plan_sha256": self.plan_sha,
            "reshape_contract": "summed-full-ocr-session-reshape-calls-including-noops-v1",
            "initialization_contract": "diagnostic-pools-precreated-outside-run-including-default-session-bindings-v1",
            "runtime_identity": {"framework_description": ".NET test", "environment_version": "test", "process_architecture": "X64",
                "os_description": "Windows test", "jit_environment": acceptance.jit_environment({}), "contract": "test-only receipt"},
            "cases": cases}
        return reference, capture

    def attach(self, capture=None, reference=None, mode="latency", budget=1):
        acceptance.attach_phase_capture(self.reference if reference is None else reference,
            self.capture if capture is None else capture, mode, budget, self.round_id, self.plan_sha)

    def reserve_fake(self):
        acceptance.reserve_round(self.ledger, "fake-task", list(acceptance.WORKLOAD_HASHES.values()), self.round_id, self.output, self.policy)
        self.report["round_id"] = self.round_id

    def arguments(self):
        return acceptance.phase_diagnostic_arguments(self.ledger, "fake-task", self.report, self.output, self.policy, self.plan_sha)

    def test_separate_work_sum_may_exceed_wall_without_subtraction(self):
        self.attach()
        phase = self.reference["rec/sample"]["phase_diagnostic"]
        self.assertEqual(phase["reshape_ns"], 60)
        self.assertEqual(phase["first_run_wall_ns"], 5)
        self.assertEqual(phase["post_run_owned_copy_ns"], 0)
        self.assertEqual(phase["post_run_text_publication_ns"], 7)
        self.assertEqual(phase["diagnostic_compile_requests_ns"], 4)

    def test_all_cpu_partitions_require_actual_det_cls_rec_call_counts(self):
        for mode in ("latency", "throughput"):
            for budget in (1, 2, 4, 8):
                reference, capture = self.fixture(mode, budget)
                self.attach(capture, reference, mode, budget)
                self.assertEqual(len(reference), 9)
                capture["cases"][0]["phase"]["reshape_stages"][1]["calls"] -= 1
                with self.assertRaises(RuntimeError):
                    self.attach(capture, reference, mode, budget)

    def test_missing_duplicate_or_changed_cases_fail_before_attachment(self):
        for mutate in (lambda c: c["cases"].pop(), lambda c: c["cases"].__setitem__(1, copy.deepcopy(c["cases"][0])),
                       lambda c: c["cases"][0]["result"].update(text="changed")):
            capture = copy.deepcopy(self.capture); mutate(capture)
            with self.assertRaises(RuntimeError): self.attach(capture)
            self.assertTrue(all("phase_diagnostic" not in case for case in self.reference.values()))

    def test_null_is_not_reinterpreted_as_zero_and_contract_is_required(self):
        for field, value in [("post_run_owned_copy_ns", None), ("post_run_owned_copy_ns", True),
                             ("post_run_owned_copy_ns", 1), ("ownership_contract", ""),
                             ("post_run_text_publication_ns", None), ("post_run_text_publication_ns", float("nan"))]:
            capture = copy.deepcopy(self.capture); capture["cases"][0]["phase"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(RuntimeError): self.attach(capture)

    def test_false_totals_missing_stages_failures_and_instrumented_official_claims_fail(self):
        mutations = [lambda p: p.update(reshape_ns=59), lambda p: p["reshape_stages"].pop(),
            lambda p: p["reshape_stages"][0].update(failed_calls=1), lambda p: p.update(official_timing_changed=True),
            lambda p: p.update(diagnostic_session_creation_calls=0), lambda p: p["pinned_pipeline_stages"].pop(),
            lambda p: p.update(cold_contract="fresh process means cold"), lambda p: p.update(reshape_time_kind="wall time")]
        for mutate in mutations:
            capture = copy.deepcopy(self.capture); mutate(capture["cases"][0]["phase"])
            with self.subTest(mutate=mutate), self.assertRaises(RuntimeError): self.attach(capture)

    def test_foreign_plan_round_runtime_and_cpu_identity_fail(self):
        for field, value in [("phase_round_id", str(uuid.uuid4())), ("phase_plan_sha256", "f" * 64),
                             ("capacity_per_request", 2), ("official_samples", True), ("phase_only", False)]:
            capture = copy.deepcopy(self.capture); capture[field] = value
            with self.subTest(field=field), self.assertRaises(RuntimeError): self.attach(capture)
        capture = copy.deepcopy(self.capture); del capture["runtime_identity"]
        with self.assertRaises(RuntimeError): self.attach(capture)

    def test_malformed_nested_receipts_fail_closed_without_partial_attachment(self):
        for mutate in (lambda c: c.update(cases=None), lambda c: c["cases"].__setitem__(0, None),
                       lambda c: c["cases"][0].update(phase=None),
                       lambda c: c["cases"][0]["phase"].update(reshape_stages=[None]),
                       lambda c: c["cases"][0]["phase"]["pinned_pipeline_stages"].__setitem__(0, None),
                       lambda c: c["cases"][0]["phase"].update(post_run_text_publication_ns=2 ** 100000)):
            capture = copy.deepcopy(self.capture); mutate(capture)
            with self.subTest(mutate=mutate), self.assertRaises(RuntimeError): self.attach(capture)
            self.assertTrue(all("phase_diagnostic" not in row for row in self.reference.values()))

    def test_rust_profile_requires_binding_and_reshape_per_request(self):
        acceptance.validate_rust_phase_profiles([{"binding_ns": 1, "reshape_ns": 2}], 1)
        for profiles, workers in [([{"binding_ns": 1}], 1), ([{"reshape_ns": 2}], 1),
                                  ([{"binding_ns": True, "reshape_ns": 2}], 1),
                                  ([{"binding_ns": 1, "reshape_ns": -1}], 1),
                                  ([{"binding_ns": 1, "reshape_ns": 2}], 2), ([], 1)]:
            with self.subTest(profiles=profiles), self.assertRaises(RuntimeError):
                acceptance.validate_rust_phase_profiles(profiles, workers)

    def test_phase_process_cannot_start_before_the_same_reserved_round(self):
        before = self.ledger.read_bytes()
        with self.assertRaisesRegex(RuntimeError, "active reserved round"): self.arguments()
        self.assertEqual(self.ledger.read_bytes(), before)
        self.reserve_fake()
        args = self.arguments()
        self.assertEqual(args, ["--phase-only", "--phase-round-id", self.round_id, "--phase-plan-sha256", self.plan_sha])
        self.assertEqual(len(gate.read_json(self.ledger)["tasks"]["fake-task"][next(iter(acceptance.WORKLOAD_HASHES.values()))]), 1)

    def test_closed_error_changed_plan_and_missing_workload_reservation_fail(self):
        self.reserve_fake()
        for field, value in (("complete", True), ("error", "failed"), ("round_id", str(uuid.uuid4()))):
            saved = dict(self.report); self.report[field] = value
            with self.subTest(field=field), self.assertRaises(RuntimeError): self.arguments()
            self.report = saved
        state = gate.read_json(self.ledger)
        state["tasks"]["fake-task"][next(iter(acceptance.WORKLOAD_HASHES.values()))].clear()
        gate.write_json(self.ledger, state)
        with self.assertRaises(RuntimeError): self.arguments()
        (self.output / "plan.json").write_text("changed")
        with self.assertRaises(RuntimeError): self.arguments()


if __name__ == "__main__":
    unittest.main()
