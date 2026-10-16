"""Contention-policy contracts, without benchmarking or sleeping."""
import unittest

import windows_cpu_load as load


def snapshot(rows=(), busy=0):
    return {"busy_seconds": busy, "processes": {pid: {"pid": pid, "name": name,
            "created": created, "cpu_seconds": cpu} for pid, name, created, cpu in rows}}


class CpuLoadContracts(unittest.TestCase):
    def test_compiler_and_external_limits_are_per_core_not_host_percent(self):
        for name, seconds in (("rustc.exe", .51), ("ordinary.exe", 3.01)):
            before = snapshot([(10, name, 1, 2)])
            after = snapshot([(10, name, 1, 2 + seconds)], seconds)
            result = load._evaluate(before, after, 1, 99)
            self.assertFalse(result["idle"])
            self.assertEqual(result["processes"][0]["name"], name)

    def test_current_runner_is_excluded_and_rows_contain_no_command_lines(self):
        before = snapshot([(99, "python.exe", 1, 4)])
        after = snapshot([(99, "python.exe", 1, 5)], 1)
        before["own_cpu_after_system"], after["own_cpu_before_system"] = 4, 5
        result = load._evaluate(before, after, 1, 99)
        self.assertTrue(result["idle"])
        self.assertEqual(result["processes"], [])
        self.assertEqual(result["external_cpu_cores"], 0)

    def test_new_or_reused_pid_is_not_subtracted_from_previous_process(self):
        before = snapshot([(10, "ordinary.exe", 1, 500)])
        after = snapshot([(10, "rustc.exe", 2, .6)], .6)
        self.assertFalse(load._evaluate(before, after, 1, 99)["idle"])

    def test_many_small_processes_and_unseen_short_lived_work_are_caught(self):
        rows = [(pid, "ordinary.exe", pid, 1.3) for pid in range(10, 15)]
        self.assertFalse(load._evaluate(snapshot(), snapshot(rows, 6.5), 1, 99)["idle"])
        result = load._evaluate(snapshot(), snapshot(busy=3.1), 1, 99)
        self.assertFalse(result["idle"])
        self.assertTrue(any("short-lived" in reason for reason in result["reasons"]))

    def test_exited_compiler_and_aggregate_process_lower_bound_fail(self):
        before = snapshot([(10, "rustc.exe", 1, 2)])
        self.assertFalse(load._evaluate(before, snapshot(busy=.2), 1, 99)["idle"])
        rows = [(pid, "ordinary.exe", pid, 1.3) for pid in range(10, 15)]
        result = load._evaluate(snapshot(), snapshot(rows, 0), 1, 99)
        self.assertFalse(result["idle"])
        self.assertEqual(result["external_cpu_cores"], 6.5)

    def test_slow_enumeration_does_not_dilute_counter_sampling_windows(self):
        before, after = snapshot(), snapshot(busy=3.2)
        before["captured_at"], after["captured_at"] = 10, 11
        result = load._evaluate(before, after, 2, 99)
        self.assertFalse(result["idle"])
        self.assertEqual(result["elapsed_seconds"], 1)
        self.assertEqual(result["external_cpu_cores"], 3.2)
        before = snapshot([(10, "rustc.exe", 1, 2)])
        after = snapshot([(10, "rustc.exe", 1, 2.6)], .6)
        before["processes"][10]["captured_at"] = 10
        after["processes"][10]["captured_at"] = 11
        self.assertFalse(load._evaluate(before, after, 2, 99)["idle"])

    def test_sampler_cpu_outside_system_window_cannot_mask_external_work(self):
        before = snapshot([(99, "python.exe", 1, 4)])
        after = snapshot([(99, "python.exe", 1, 7.1)], 3.1)
        before["captured_at"], after["captured_at"] = 10, 11
        before["processes"][99]["captured_at"] = 10.1
        after["processes"][99]["captured_at"] = 12
        for contained in (False, True):
            if contained:
                before["own_cpu_after_system"], after["own_cpu_before_system"] = 4, 4
            result = load._evaluate(before, after, 2, 99)
            self.assertFalse(result["idle"])
            self.assertEqual(result["external_cpu_cores"], 3.1)

    def test_normal_background_policy_passes_representative_desktop_load(self):
        rows = [(10, "msedgewebview2.exe", 1, .388),
                (11, "ChatGPT.exe", 1, .273), (12, "ordinary.exe", 1, .149)]
        result = load._evaluate(snapshot(), snapshot(rows, 1.343), 1, 99)
        self.assertTrue(result["idle"], result["reasons"])
        self.assertAlmostEqual(result["unattributed_cpu_cores"], .533)
        self.assertEqual(result["policy"], load.POLICY)
        self.assertEqual(result["policy"]["revision"], "normal-background-v3")
        self.assertEqual({key: result["policy"][key] for key in (
            "compiler_core_limit", "external_process_core_limit",
            "aggregate_external_core_limit", "unattributed_core_limit")}, {
                "compiler_core_limit": .50, "external_process_core_limit": 3.00,
                "aggregate_external_core_limit": 6.00, "unattributed_core_limit": 3.00})

    def test_normal_desktop_burst_passes_without_excluding_app_processes(self):
        rows = [(10, "ChatGPT.exe", 1, 1.773),
                (11, "codex.exe", 1, .197), (12, "ordinary.exe", 1, .530)]
        result = load._evaluate(snapshot(), snapshot(rows, 4.392), 1, 99)
        self.assertTrue(result["idle"], result["reasons"])
        self.assertAlmostEqual(result["external_cpu_cores"], 4.392)
        self.assertAlmostEqual(result["unattributed_cpu_cores"], 1.892)
        self.assertEqual(len(result["processes"]), 3)

    def test_unreadable_compiler_fails_and_light_activity_passes(self):
        unreadable = snapshot([(10, "link.exe", None, None)])
        self.assertFalse(load._evaluate(snapshot(), unreadable, 1, 99)["idle"])
        result = load._evaluate(snapshot(), snapshot([(10, "ordinary.exe", 1, .1)], .15), 1, 99)
        self.assertTrue(result["idle"])
        self.assertEqual(set(result["processes"][0]), {"pid", "name", "cpu_seconds", "cpu_cores"})

    def test_group_scoped_system_counters_never_claim_whole_host_idle(self):
        before, after = snapshot(), snapshot()
        after["active_logical_processors"] = 128
        self.assertFalse(load._evaluate(before, after, 1, 99)["idle"])

    def test_invalid_intervals_and_negative_counters_fail_explicitly(self):
        for interval in (0, -.1, float("nan"), 61):
            with self.assertRaises(ValueError):
                load.sample_load(interval)
        with self.assertRaises(RuntimeError):
            load._evaluate(snapshot(busy=2), snapshot(busy=1), 1, 99)


if __name__ == "__main__":
    unittest.main()
