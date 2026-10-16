"""Append-only explicit one-round authorization contracts on disposable ledgers."""
import contextlib
import copy
import hashlib
import io
import json
import msvcrt
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import uuid

import cpu_acceptance as acceptance
import cpu_parity as gate


class FollowupAuthorizationContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.ledger = self.root / "ledger.json"
        self.task = "same-task"
        for number in range(3):
            acceptance.reserve_round(self.ledger, self.task, ["sample", "dense", "long"], str(number), self.root / str(number))
        self.base = acceptance.record_round_extension(self.ledger, self.task, 2, "Explicit original two-round authorization")
        for number in (3, 4):
            acceptance.reserve_round(self.ledger, self.task, ["sample", "dense", "long"], str(number), self.root / str(number), self.base)
        # Full-ledger preservation covers unrelated existing tasks as well.
        acceptance.reserve_round(self.ledger, "unrelated-task", ["other-sample"], "unrelated", self.root / "unrelated")
        self.original = self.ledger.read_bytes()
        self.state = gate.read_json(self.ledger)
        self.new_id = str(uuid.uuid4())
        self.prior_id = self.base["authorization"]["id"]
        self.text = "The user explicitly permits ONE further full performance round."

    def tearDown(self):
        self.temporary.cleanup()

    def grant(self, **kwargs):
        return acceptance.record_followup_round_extension(
            self.ledger, self.task, kwargs.get("additional_rounds", 1),
            kwargs.get("authorization_id", self.new_id), kwargs.get("prior_id", self.prior_id),
            kwargs.get("text", self.text))

    def test_preserves_full_ledger_and_original_two_round_extension(self):
        policy = self.grant()
        state = gate.read_json(self.ledger)
        self.assertEqual(state["tasks"], self.state["tasks"])
        self.assertEqual(state["round_cap_extensions"], self.state["round_cap_extensions"])
        self.assertEqual(policy["maximum_rounds"], 6)
        self.assertEqual(policy["additional_rounds"], 3)
        self.assertEqual(policy["base_round_cap"], 3)
        self.assertEqual(policy["authorization"]["id"], self.new_id)
        self.assertEqual(policy["authorization_chain"], [self.prior_id, self.new_id])
        records = state["round_cap_followups"][self.task]
        self.assertEqual(len(records), 1)
        entry = records[0]
        self.assertEqual((entry["additional_rounds"], entry["previous_maximum_rounds"], entry["maximum_rounds"]), (1, 5, 6))
        auth = entry["authorization"]
        self.assertEqual(auth["prior_authorization_id"], self.prior_id)
        self.assertEqual(auth["task_id"], self.task)
        self.assertEqual(auth["text"], self.text)
        self.assertEqual(Path(auth["prior_ledger_backup"]).read_bytes(), self.original)
        self.assertEqual(auth["prior_ledger_sha256"], hashlib.sha256(self.original).hexdigest())

    def test_sixth_reservation_binds_latest_cap_id_and_seventh_fails(self):
        policy = self.grant()
        acceptance.reserve_round(self.ledger, self.task, ["sample", "dense", "long"], "sixth", self.root / "sixth", policy)
        current = gate.read_json(self.ledger)
        for name in ("sample", "dense", "long"):
            rows = current["tasks"][self.task][name]
            self.assertEqual(rows[:5], self.state["tasks"][self.task][name])
            self.assertEqual(rows[5]["authorization_id"], self.new_id)
            self.assertEqual(rows[5]["maximum_rounds"], 6)
        saved = self.ledger.read_bytes()
        with self.assertRaisesRegex(RuntimeError, "six authorized"):
            acceptance.reserve_round(self.ledger, self.task, ["sample"], "seventh", self.root / "seventh", policy)
        self.assertEqual(self.ledger.read_bytes(), saved)

    def test_latest_idempotency_even_after_reservation_but_conflicts_fail(self):
        policy = self.grant()
        saved = self.ledger.read_bytes()
        self.assertEqual(self.grant(), policy)
        self.assertEqual(self.ledger.read_bytes(), saved)
        acceptance.reserve_round(self.ledger, self.task, ["sample"], "sixth", self.root / "sixth", policy)
        saved = self.ledger.read_bytes()
        self.assertEqual(self.grant(), policy)
        for kwargs in ({"text": "replacement"}, {"prior_id": str(uuid.uuid4())}, {"additional_rounds": 2}):
            with self.subTest(kwargs=kwargs), self.assertRaises(RuntimeError):
                self.grant(**kwargs)
            self.assertEqual(self.ledger.read_bytes(), saved)

    def test_stale_prior_and_reused_base_or_foreign_ids_fail_without_changes(self):
        self.grant()
        saved = self.ledger.read_bytes()
        for kwargs in ({"authorization_id": str(uuid.uuid4())}, {"authorization_id": self.prior_id},
                       {"authorization_id": str(uuid.uuid4()), "prior_id": str(uuid.uuid4())}):
            with self.subTest(kwargs=kwargs), self.assertRaises(RuntimeError):
                self.grant(**kwargs)
            self.assertEqual(self.ledger.read_bytes(), saved)

    def test_distinct_followups_append_chain_and_old_authorization_cannot_replay(self):
        first = self.grant()
        next_id = str(uuid.uuid4())
        second = self.grant(authorization_id=next_id, prior_id=self.new_id,
                            text="A separate explicit instruction grants ONE later round")
        self.assertEqual(second["maximum_rounds"], 7)
        self.assertEqual(second["authorization_chain"], [self.prior_id, self.new_id, next_id])
        self.assertEqual(second["followup_authorizations"][:1], first["followup_authorizations"])
        saved = self.ledger.read_bytes()
        with self.assertRaisesRegex(RuntimeError, "replayed"):
            self.grant()
        self.assertEqual(self.ledger.read_bytes(), saved)
        for policy in (first, self.base):
            with self.assertRaisesRegex(RuntimeError, "changed after"):
                acceptance.reserve_round(self.ledger, self.task, ["sample"], "stale", self.root / "stale", policy)
        self.assertEqual(self.ledger.read_bytes(), saved)

    def test_authorization_id_cannot_be_reused_across_tasks(self):
        for n in range(3):
            acceptance.reserve_round(self.ledger, "other-authorized", ["other"], str(n), self.root / str(n))
        other = acceptance.record_round_extension(self.ledger, "other-authorized", 2, "Distinct two-round instruction")
        saved = self.ledger.read_bytes()
        with self.assertRaisesRegex(RuntimeError, "replayed"):
            self.grant(authorization_id=other["authorization"]["id"])
        self.assertEqual(self.ledger.read_bytes(), saved)

    def test_new_task_or_unapproved_syntax_does_not_create_authorization(self):
        saved = self.ledger.read_bytes()
        for kwargs in ({"additional_rounds": 2}, {"additional_rounds": True}, {"text": " "},
                       {"authorization_id": "not-a-uuid"}, {"prior_id": "not-a-uuid"},
                       {"authorization_id": self.prior_id}):
            with self.subTest(kwargs=kwargs), self.assertRaises(RuntimeError):
                self.grant(**kwargs)
            self.assertEqual(self.ledger.read_bytes(), saved)
        with self.assertRaises(RuntimeError):
            acceptance.record_followup_round_extension(self.ledger, "new-task", 1, self.new_id, self.prior_id, self.text)
        self.assertEqual(self.ledger.read_bytes(), saved)

    def test_default_three_and_original_five_policy_are_unchanged(self):
        self.assertEqual(acceptance.read_round_policy(self.ledger, self.task), self.base)
        self.assertEqual(acceptance.read_round_policy(self.ledger, "unrelated-task"),
                         {"base_round_cap": 3, "additional_rounds": 0, "maximum_rounds": 3, "authorization": None})
        with self.assertRaises(RuntimeError):
            acceptance.record_followup_round_extension(self.ledger, "unrelated-task", 1, self.new_id, self.prior_id, self.text)

    def test_prior_reservations_other_tasks_and_original_authorizations_cannot_be_edited(self):
        self.grant()
        valid = gate.read_json(self.ledger)
        mutations = [
            lambda s: s["tasks"][self.task]["sample"].pop(),
            lambda s: s["tasks"]["unrelated-task"]["other-sample"].clear(),
            lambda s: s["round_cap_extensions"][self.task]["authorization"].update(text="changed"),
            lambda s: s["round_cap_followups"][self.task][0].update(additional_rounds=2),
            lambda s: s["round_cap_followups"][self.task][0]["authorization"].update(prior_authorization_id=str(uuid.uuid4())),
            lambda s: s["round_cap_followups"][self.task][0]["authorization"].update(recorded_at_utc="invalid"),
        ]
        for mutate in mutations:
            state = copy.deepcopy(valid); mutate(state); gate.write_json(self.ledger, state)
            saved = self.ledger.read_bytes()
            with self.subTest(mutate=mutate), self.assertRaises(RuntimeError):
                acceptance.read_round_policy(self.ledger, self.task)
            self.assertEqual(self.ledger.read_bytes(), saved)
        gate.write_json(self.ledger, valid)

    def test_changed_backup_rejected_and_existing_rounds_survive(self):
        policy = self.grant()
        backup = Path(policy["authorization"]["prior_ledger_backup"])
        backup.write_bytes(b"changed")
        saved = self.ledger.read_bytes()
        with self.assertRaisesRegex(RuntimeError, "backup.*missing or changed"):
            acceptance.read_round_policy(self.ledger, self.task)
        self.assertEqual(self.ledger.read_bytes(), saved)

    def test_stale_immutable_plan_is_rejected_after_authorization(self):
        self.grant()
        saved = self.ledger.read_bytes()
        with self.assertRaisesRegex(RuntimeError, "changed after"):
            acceptance.reserve_round(self.ledger, self.task, ["sample"], "sixth", self.root / "sixth", self.base)
        self.assertEqual(self.ledger.read_bytes(), saved)

    def test_record_only_cli_cannot_build_load_sample_or_reserve(self):
        with (mock.patch.object(gate, "require_tool", side_effect=AssertionError("must not build")),
              mock.patch.object(acceptance.cpu_load, "sample_load", side_effect=AssertionError("must not sample")),
              mock.patch.object(acceptance, "reserve_round", side_effect=AssertionError("must not reserve")),
              contextlib.redirect_stdout(io.StringIO()) as output):
            result = acceptance.main(["--record-followup-round-extension", "--task-id", self.task, "--ledger", str(self.ledger),
                "--additional-rounds", "1", "--authorization-id", self.new_id,
                "--prior-authorization-id", self.prior_id, "--round-authorization", self.text])
        self.assertEqual(result, 0)
        self.assertFalse(json.loads(output.getvalue())["performance_round_reserved"])
        self.assertEqual(gate.read_json(self.ledger)["tasks"], self.state["tasks"])

    def test_cli_rejects_incomplete_or_mixed_authorization_modes(self):
        saved = self.ledger.read_bytes()
        for argv in (["--record-followup-round-extension"],
                     ["--record-round-extension", "--record-followup-round-extension"],
                     ["--authorization-id", self.new_id],
                     ["--record-followup-round-extension", "--additional-rounds", "2"]):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                acceptance.main(["--task-id", self.task, "--ledger", str(self.ledger), *argv])
            self.assertEqual(self.ledger.read_bytes(), saved)

    def test_shared_sidecar_lock_prevents_competing_followup_and_reservation(self):
        active = False
        def locking(_descriptor, mode, _count):
            nonlocal active
            if mode == msvcrt.LK_NBLCK:
                if active: raise OSError("already locked")
                active = True
            else:
                self.assertTrue(active); active = False
        with mock.patch.object(msvcrt, "locking", side_effect=locking):
            with acceptance.locked_ledger(self.ledger):
                with self.assertRaisesRegex(RuntimeError, "Another acceptance runner"):
                    self.grant()
                self.assertTrue(active)
        self.assertEqual(self.ledger.read_bytes(), self.original)

    def test_real_windows_sidecar_lock_rejects_a_competing_process(self):
        script = ("from pathlib import Path; import sys; import cpu_acceptance as a; "
                  "guard=a.locked_ledger(Path(sys.argv[1])); guard.__enter__(); "
                  "print('LOCKED',flush=True); sys.stdin.readline(); guard.__exit__(None,None,None)")
        child = subprocess.Popen([sys.executable, "-c", script, str(self.ledger)],
            cwd=Path(acceptance.__file__).parent, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            self.assertEqual(child.stdout.readline().strip(), "LOCKED")
            with self.assertRaisesRegex(RuntimeError, "Another acceptance runner"):
                self.grant()
            self.assertEqual(self.ledger.read_bytes(), self.original)
            _stdout, stderr = child.communicate("release\n", timeout=5)
            self.assertEqual(child.returncode, 0, stderr)
        finally:
            if child.poll() is None:
                child.kill(); child.communicate(timeout=5)

    def test_lost_update_and_pending_replace_failure_preserve_original_ledger(self):
        with mock.patch.object(acceptance.os, "replace", side_effect=OSError("injected replace failure")):
            with self.assertRaisesRegex(OSError, "injected"):
                self.grant()
        self.assertEqual(self.ledger.read_bytes(), self.original)
        self.assertEqual(list(self.root.glob(".ledger.json.*.tmp")), [])
        # A second cooperating grant cannot reuse an old prior ID after commit.
        first = self.grant()
        saved = self.ledger.read_bytes()
        with self.assertRaisesRegex(RuntimeError, "latest"):
            self.grant(authorization_id=str(uuid.uuid4()))
        self.assertEqual(self.ledger.read_bytes(), saved)

    def test_writer_ignoring_sidecar_is_detected_before_replacement(self):
        changed = None
        with self.assertRaisesRegex(RuntimeError, "lost update"):
            with acceptance.locked_ledger(self.ledger) as (state, _):
                state["tasks"][self.task]["sample"].append({"round_id": "would-overwrite"})
                external = gate.read_json(self.ledger)
                external["tasks"][self.task]["sample"].append({"round_id": "external"})
                gate.write_json(self.ledger, external)
                changed = self.ledger.read_bytes()
        self.assertEqual(self.ledger.read_bytes(), changed)


if __name__ == "__main__":
    unittest.main()
