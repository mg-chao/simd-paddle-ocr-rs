"""Explicit cap authorization contracts; no inference or benchmark samples."""
import contextlib
import copy
import hashlib
import io
import msvcrt
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import cpu_acceptance as acceptance
import cpu_parity as gate


class RoundAuthorizationContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.ledger = self.root / "ledger.json"
        for number in range(3):
            acceptance.reserve_round(self.ledger, "same-task", ["sample", "dense", "long"], str(number), self.root / str(number))
        self.original = self.ledger.read_bytes()
        self.original_tasks = gate.read_json(self.ledger)["tasks"]
        self.authorization = "The user explicitly authorizes two additional performance rounds in this existing task."

    def tearDown(self):
        self.temporary.cleanup()

    def grant(self):
        return acceptance.record_round_extension(self.ledger, "same-task", 2, self.authorization)

    def test_extension_preserves_all_previous_rounds_and_allows_only_two_more(self):
        policy = self.grant()
        self.assertEqual(policy["maximum_rounds"], 5)
        self.assertEqual(policy["base_round_cap"], 3)
        self.assertEqual(policy["additional_rounds"], 2)
        self.assertEqual(policy["authorization"]["task_id"], "same-task")
        self.assertEqual(policy["authorization"]["text"], self.authorization)
        self.assertTrue(policy["authorization"]["recorded_at_utc"].endswith("+00:00"))
        self.assertEqual(Path(policy["authorization"]["prior_ledger_backup"]).read_bytes(), self.original)
        self.assertEqual(policy["authorization"]["prior_ledger_sha256"], hashlib.sha256(self.original).hexdigest())
        self.assertEqual(gate.read_json(self.ledger)["tasks"], self.original_tasks)
        for number in (3, 4):
            acceptance.reserve_round(self.ledger, "same-task", ["sample", "dense", "long"], str(number), self.root / str(number), policy)
        saved = self.ledger.read_bytes()
        with self.assertRaisesRegex(RuntimeError, "five authorized"):
            acceptance.reserve_round(self.ledger, "same-task", ["sample"], "sixth", self.root / "sixth", policy)
        self.assertEqual(self.ledger.read_bytes(), saved)
        rows = gate.read_json(self.ledger)["tasks"]["same-task"]["sample"]
        self.assertEqual(rows[:3], self.original_tasks["same-task"]["sample"])
        self.assertEqual(rows[-1]["authorization_id"], policy["authorization"]["id"])

    def test_repeated_authorization_is_idempotent_and_cannot_stack_or_replace(self):
        policy = self.grant()
        saved = self.ledger.read_bytes()
        self.assertEqual(self.grant(), policy)
        self.assertEqual(self.ledger.read_bytes(), saved)
        for additional, text in ((4, self.authorization), (2, "different replacement instruction"), (2, "")):
            with self.assertRaises(RuntimeError):
                acceptance.record_round_extension(self.ledger, "same-task", additional, text)
        self.assertEqual(self.ledger.read_bytes(), saved)

    def test_new_task_does_not_inherit_or_reset_the_authorized_task_cap(self):
        self.grant()
        with self.assertRaisesRegex(RuntimeError, "existing task ID"):
            acceptance.record_round_extension(self.ledger, "new-task", 2, self.authorization)
        for number in range(3):
            acceptance.reserve_round(self.ledger, "another-task", ["sample"], str(number), self.root / str(number))
        with self.assertRaisesRegex(RuntimeError, "three performance rounds"):
            acceptance.reserve_round(self.ledger, "another-task", ["sample"], "fourth", self.root / "fourth")

    def test_policy_change_after_plan_and_malformed_extension_fail_before_reservation(self):
        original_policy = acceptance.read_round_policy(self.ledger, "same-task")
        policy = self.grant()
        with self.assertRaisesRegex(RuntimeError, "changed after"):
            acceptance.reserve_round(self.ledger, "same-task", ["sample"], "fourth", self.root / "fourth", original_policy)
        self.assertEqual(len(gate.read_json(self.ledger)["tasks"]["same-task"]["sample"]), 3)
        for mutate in (lambda row: row.update({"maximum_rounds": 6}),
                       lambda row: row["authorization"].update({"task_id": "foreign-task"}),
                       lambda row: row["authorization"].update({"recorded_at_utc": "missing"})):
            state = gate.read_json(self.ledger)
            state["round_cap_extensions"]["same-task"] = copy.deepcopy(policy)
            mutate(state["round_cap_extensions"]["same-task"])
            gate.write_json(self.ledger, state)
            saved = self.ledger.read_bytes()
            with self.assertRaisesRegex(RuntimeError, "Malformed"):
                acceptance.reserve_round(self.ledger, "same-task", ["sample"], "fourth", self.root / "fourth")
            self.assertEqual(self.ledger.read_bytes(), saved)

    def test_record_only_cli_never_builds_samples_load_or_reserves_a_round(self):
        with (mock.patch.object(gate, "require_tool", side_effect=AssertionError("must not build")),
              mock.patch.object(acceptance.cpu_load, "sample_load", side_effect=AssertionError("must not sample")),
              contextlib.redirect_stdout(io.StringIO())):
            result = acceptance.main(["--record-round-extension", "--task-id", "same-task", "--ledger", str(self.ledger),
                                      "--additional-rounds", "2", "--round-authorization", self.authorization])
        self.assertEqual(result, 0)
        self.assertEqual(gate.read_json(self.ledger)["tasks"], self.original_tasks)

    def test_original_reservations_and_verified_backup_cannot_be_reset(self):
        policy = self.grant()
        state = gate.read_json(self.ledger)
        state["tasks"]["same-task"]["sample"].pop(0)
        gate.write_json(self.ledger, state)
        with self.assertRaisesRegex(RuntimeError, "Original task reservations changed"):
            acceptance.read_round_policy(self.ledger, "same-task")
        state["tasks"] = copy.deepcopy(self.original_tasks)
        gate.write_json(self.ledger, state)
        Path(policy["authorization"]["prior_ledger_backup"]).write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "backup is missing or changed"):
            acceptance.read_round_policy(self.ledger, "same-task")

    def test_existing_empty_ledger_fails_closed_and_new_ledger_is_initialized(self):
        self.ledger.write_bytes(b" \r\n")
        saved = self.ledger.read_bytes()
        with self.assertRaisesRegex(RuntimeError, "ledger is empty"):
            acceptance.read_round_policy(self.ledger, "same-task")
        with self.assertRaisesRegex(RuntimeError, "ledger is empty"):
            acceptance.reserve_round(self.ledger, "same-task", ["sample"], "reset", self.root / "reset")
        self.assertEqual(self.ledger.read_bytes(), saved)
        fresh = self.root / "fresh.json"
        self.assertEqual(acceptance.read_round_policy(fresh, "same-task")["maximum_rounds"], 3)
        self.assertEqual(gate.read_json(fresh), {"schema_version": 1, "tasks": {}})

    def test_failed_write_flush_or_replace_keeps_every_existing_reservation(self):
        policy = self.grant()
        saved = self.ledger.read_bytes()
        real_open = Path.open

        class FailingWrite:
            def __init__(self, stream):
                self.stream = stream

            def __enter__(self):
                self.stream.__enter__()
                return self

            def __exit__(self, *args):
                return self.stream.__exit__(*args)

            def write(self, data):
                raise OSError("injected pending write failure")

        def open_with_failed_pending_write(path, *args, **kwargs):
            stream = real_open(path, *args, **kwargs)
            if path.parent == self.root and path.name.startswith(".ledger.json.") and path.suffix == ".tmp":
                return FailingWrite(stream)
            return stream

        faults = (mock.patch.object(Path, "open", open_with_failed_pending_write),
                  mock.patch.object(acceptance.os, "fsync", side_effect=OSError("injected flush failure")),
                  mock.patch.object(acceptance.os, "replace", side_effect=OSError("injected replace failure")))
        for fault in faults:
            with self.subTest(fault=fault), fault:
                with self.assertRaisesRegex(OSError, "injected"):
                    acceptance.reserve_round(self.ledger, "same-task", ["sample"], "fourth", self.root / "fourth", policy)
            self.assertEqual(self.ledger.read_bytes(), saved)
            self.assertEqual(gate.read_json(self.ledger)["tasks"], self.original_tasks)
            self.assertEqual(list(self.root.glob(".ledger.json.*.tmp")), [])

    def test_failure_after_committed_replace_does_not_roll_back_consumed_round(self):
        policy = self.grant()
        real_replace = acceptance.os.replace

        def committed_then_interrupted(source, destination):
            real_replace(source, destination)
            raise OSError("interrupted after commit")

        with mock.patch.object(acceptance.os, "replace", side_effect=committed_then_interrupted):
            with self.assertRaisesRegex(OSError, "interrupted after commit"):
                acceptance.reserve_round(self.ledger, "same-task", ["sample"], "fourth", self.root / "fourth", policy)
        rows = gate.read_json(self.ledger)["tasks"]["same-task"]["sample"]
        self.assertEqual(rows[:3], self.original_tasks["same-task"]["sample"])
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[3]["round_id"], "fourth")
        acceptance.reserve_round(self.ledger, "same-task", ["sample"], "fifth", self.root / "fifth", policy)
        with self.assertRaisesRegex(RuntimeError, "five authorized"):
            acceptance.reserve_round(self.ledger, "same-task", ["sample"], "sixth", self.root / "sixth", policy)

    def test_stable_sidecar_lock_rejects_concurrent_writer_without_unlocking_owner(self):
        active = False

        def locking(descriptor, mode, count):
            nonlocal active
            self.assertEqual(count, 1)
            if mode == msvcrt.LK_NBLCK:
                if active:
                    raise OSError("already locked")
                active = True
            elif mode == msvcrt.LK_UNLCK:
                self.assertTrue(active)
                active = False

        saved = self.ledger.read_bytes()
        with mock.patch.object(msvcrt, "locking", side_effect=locking):
            with acceptance.locked_ledger(self.ledger):
                self.assertTrue(active)
                with self.assertRaisesRegex(RuntimeError, "Another acceptance runner"):
                    acceptance.reserve_round(self.ledger, "same-task", ["sample"], "blocked", self.root / "blocked")
                self.assertTrue(active)
            self.assertFalse(active)
        self.assertEqual(self.ledger.read_bytes(), saved)
        self.assertTrue(self.ledger.with_name(self.ledger.name + ".lock").is_file())


if __name__ == "__main__":
    unittest.main()
