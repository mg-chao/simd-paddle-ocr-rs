"""Idle scheduling contracts only; no sleeps, model runs or performance rounds."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import cpu_acceptance as acceptance
import cpu_parity as gate


class IdleWaitContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "report.json"
        self.report = {"complete": False, "passed": False, "host_load_checks": []}
        self.verify = mock.Mock()

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def load(idle):
        return {"idle": idle, "reasons": [] if idle else ["external CPU activity"], "policy": {"unchanged": True}}

    def test_waits_before_reserving_and_preflight_runs_once_across_both_waits(self):
        guard = acceptance.IdleGate(self.report, self.path, self.verify, wait_for_idle=True)
        inference = mock.Mock(return_value={"qualified": True})
        reserve = mock.Mock(side_effect=lambda: self.report.update(round_id="reserved-once"))
        calls_at_gaps = []
        def quiet_gap(seconds):
            self.assertEqual(seconds, 30)
            reserve.assert_not_called()
            self.assertNotIn("round_id", self.report)
            calls_at_gaps.append(inference.call_count)
            persisted = gate.read_json(self.path)
            self.assertEqual(persisted["host_load_state"]["status"], "waiting_for_idle")
        loads = [self.load(False), self.load(True), self.load(False), self.load(False), self.load(True)]
        with (mock.patch.object(acceptance.cpu_load, "sample_load", side_effect=loads) as sample,
              mock.patch.object(acceptance.time, "sleep", side_effect=quiet_gap) as sleep):
            self.assertEqual(guard.preflight_and_reserve(inference, reserve), {"qualified": True})
        self.assertEqual(calls_at_gaps, [0, 1, 1])
        inference.assert_called_once_with()
        reserve.assert_called_once_with()
        self.assertEqual(sample.call_count, 5)
        self.assertEqual(sleep.call_args_list, [mock.call(30)] * 3)
        self.assertEqual(self.verify.call_count, 10)
        persisted = gate.read_json(self.path)
        self.assertEqual([row["idle"] for row in persisted["host_load_checks"]], [False, True, False, False, True])
        self.assertEqual([row["phase"] for row in persisted["host_load_checks"]],
                         [acceptance.PRE_RESERVATION_PHASES[0]] * 2 + [acceptance.PRE_RESERVATION_PHASES[1]] * 3)

    def test_default_still_fails_before_any_preflight_or_reservation(self):
        guard = acceptance.IdleGate(self.report, self.path, self.verify)
        inference, reserve = mock.Mock(), mock.Mock()
        with (mock.patch.object(acceptance.cpu_load, "sample_load", return_value=self.load(False)) as sample,
              mock.patch.object(acceptance.time, "sleep") as sleep):
            with self.assertRaisesRegex(RuntimeError, "CPU contention"):
                guard.preflight_and_reserve(inference, reserve)
        inference.assert_not_called()
        reserve.assert_not_called()
        sleep.assert_not_called()
        sample.assert_called_once_with(interval_seconds=1)

    def test_post_reservation_contention_never_waits_or_repeats_inference(self):
        guard = acceptance.IdleGate(self.report, self.path, self.verify, wait_for_idle=True)
        preflight = mock.Mock(return_value="qualified")
        reserve = mock.Mock(side_effect=lambda: self.report.update(round_id="one-round"))
        benchmark = mock.Mock()
        with (mock.patch.object(acceptance.cpu_load, "sample_load", side_effect=[self.load(True), self.load(True), self.load(False)]) as sample,
              mock.patch.object(acceptance.time, "sleep") as sleep):
            guard.preflight_and_reserve(preflight, reserve)
            benchmark()
            with self.assertRaisesRegex(RuntimeError, "CPU contention"):
                guard.check("after-latency-1-before-scratch-diagnostic")
        preflight.assert_called_once_with()
        reserve.assert_called_once_with()
        benchmark.assert_called_once_with()
        sleep.assert_not_called()
        self.assertEqual(sample.call_count, 3)
        self.assertEqual(self.report["host_load_state"]["status"], "contended")
        self.assertTrue(self.report["host_load_state"]["round_reserved"])

    def test_reserved_round_disables_wait_even_for_a_pre_reservation_phase_name(self):
        self.report["round_id"] = "already-reserved"
        guard = acceptance.IdleGate(self.report, self.path, self.verify, wait_for_idle=True)
        with (mock.patch.object(acceptance.cpu_load, "sample_load", return_value=self.load(False)),
              mock.patch.object(acceptance.time, "sleep") as sleep):
            with self.assertRaisesRegex(RuntimeError, "CPU contention"):
                guard.check(acceptance.PRE_RESERVATION_PHASES[0])
        sleep.assert_not_called()

    def test_changed_frozen_inputs_abort_during_wait_without_repeating_preflight(self):
        self.verify.side_effect = [None, None, RuntimeError("Frozen inputs changed")]
        guard = acceptance.IdleGate(self.report, self.path, self.verify, wait_for_idle=True)
        inference, reserve = mock.Mock(), mock.Mock()
        with (mock.patch.object(acceptance.cpu_load, "sample_load", side_effect=[self.load(False), self.load(True)]) as sample,
              mock.patch.object(acceptance.time, "sleep") as sleep):
            with self.assertRaisesRegex(RuntimeError, "Frozen inputs changed"):
                guard.preflight_and_reserve(inference, reserve)
        inference.assert_not_called()
        reserve.assert_not_called()
        sample.assert_called_once_with(interval_seconds=1)
        sleep.assert_called_once_with(30)
        self.assertEqual(len(gate.read_json(self.path)["host_load_checks"]), 1)

    def test_consistently_replaced_corpus_aborts_wait_before_sampling_or_preflight(self):
        corpus = self.path.parent / "corpus"
        corpus.mkdir()
        raw = corpus / "sample.bgr"
        manifest = corpus / "corpus.json"

        def replace(value):
            raw.write_bytes(value)
            gate.write_json(manifest, {"schema_version": 1, "pixel_format": "bgr24",
                "images": [{"id": "sample", "file": "sample.bgr", "width": 1,
                            "height": 1, "sha256": gate.sha256(raw)}]})

        replace(bytes([1, 2, 3]))
        frozen_sha256 = gate.sha256(manifest)
        guard = acceptance.IdleGate(self.report, self.path,
            lambda: acceptance.verify_frozen_corpus(corpus, frozen_sha256), wait_for_idle=True)
        preflight, reserve = mock.Mock(), mock.Mock()

        def quiet_gap(seconds):
            self.assertEqual(seconds, 30)
            replace(bytes([4, 5, 6]))
            # The replacement is internally consistent, so checking only the
            # latest manifest would incorrectly accept these changed inputs.
            gate.validate_corpus(corpus)

        with (mock.patch.object(acceptance.cpu_load, "sample_load", return_value=self.load(False)) as sample,
              mock.patch.object(acceptance.time, "sleep", side_effect=quiet_gap) as sleep,
              mock.patch.object(gate, "validate_corpus", wraps=gate.validate_corpus) as validate):
            with self.assertRaisesRegex(RuntimeError, "Frozen acceptance corpus manifest changed"):
                guard.preflight_and_reserve(preflight, reserve)
        preflight.assert_not_called()
        reserve.assert_not_called()
        sample.assert_called_once_with(interval_seconds=1)
        sleep.assert_called_once_with(30)
        # The failing verification checks the frozen SHA before raw validation.
        self.assertEqual(validate.call_count, 3)
        self.assertNotIn("round_id", self.report)


if __name__ == "__main__":
    unittest.main()
