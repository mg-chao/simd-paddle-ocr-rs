"""Gate tests only: no model inference or performance samples are executed."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

import cpu_acceptance as acceptance
import cpu_parity as gate
import test_cpu_acceptance_prerequisite as prerequisite_tests


class CompilerConfigurationContracts(unittest.TestCase):
    """A source-equivalent library still needs its tested compiler/options."""

    def setUp(self):
        self.fixture = prerequisite_tests.AcceptancePrerequisiteContracts()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.fixture.current_rustc = 'rustc 1.100.0-nightly\nhost: x86_64-pc-windows-msvc\nLLVM version: 23.1.1'
        self.fixture.current_build_flags = {'RUSTFLAGS': '', 'CARGO_ENCODED_RUSTFLAGS': '',
            'CARGO_BUILD_RUSTFLAGS': '', 'CARGO_BUILD_TARGET': '', 'manifest_sha256': 'manifest',
            'lock_sha256': 'lock', 'cargo_features': ['profiling'], 'cargo_configs': {}}
        self.fixture.metadata['benchmark_environment'].update({
            'rustc': self.fixture.current_rustc, 'build_flags': copy.deepcopy(self.fixture.current_build_flags)})
        self.sync_metadata()

    def sync_metadata(self):
        gate.write_json(self.fixture.bundle / 'run.json', self.fixture.metadata)
        for mode in ('reference', 'auto'):
            path = self.fixture.bundle / ('rust-' + mode) / 'capture.json'
            capture = gate.read_json(path)
            capture['metadata']['benchmark_environment'] = copy.deepcopy(self.fixture.metadata['benchmark_environment'])
            gate.write_json(path, capture)

    def reject_before_preparation(self):
        try:
            self.fixture.invoke()
        except prerequisite_tests.PreparationReached:
            self.fail('Compiler/build configuration drift reached output preparation without a fresh full gate')
        except RuntimeError as error:
            self.assertRegex(str(error), 'compiler|build configuration')
        else:
            self.fail('Missing compiler/build configuration rejection')
        self.assertFalse(self.fixture.output.exists(), 'Configuration drift must fail before output creation')

    def test_rustc_drift_cannot_reuse_a_source_equivalent_full_gate(self):
        self.fixture.current_rustc += '\nchanged compiler'
        self.reject_before_preparation()

    def test_release_flags_and_cargo_config_drift_require_a_new_full_gate(self):
        baseline = copy.deepcopy(self.fixture.current_build_flags)
        changes = {'RUSTFLAGS': '-C target-cpu=native', 'CARGO_ENCODED_RUSTFLAGS': '-C\x1ftarget-feature=+fma',
            'CARGO_BUILD_RUSTFLAGS': '-C opt-level=2', 'CARGO_BUILD_TARGET': 'another-target',
            'CARGO_PROFILE_RELEASE_LTO': 'false', 'CARGO_PROFILE_RELEASE_OPT_LEVEL': '2',
            'cargo_configs': {'C:/Users/example/.cargo/config.toml': 'changed-config'}}
        for name, value in changes.items():
            with self.subTest(name=name):
                self.fixture.current_build_flags = {**baseline, name: value}
                self.reject_before_preparation()

    def test_missing_compiler_or_build_evidence_cannot_qualify(self):
        baseline = copy.deepcopy(self.fixture.metadata['benchmark_environment'])
        for name in ('rustc', 'build_flags'):
            with self.subTest(name=name):
                self.fixture.metadata['benchmark_environment'] = copy.deepcopy(baseline)
                del self.fixture.metadata['benchmark_environment'][name]
                self.sync_metadata()
                self.reject_before_preparation()


class AcceptanceContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.tolerances = {"tensor_atol": .001, "tensor_rtol": .001, "score_atol": .01, "coordinate_atol": .5}

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def pool_probe(self_test=False):
        return {"supported": True, "self_test": self_test, "peak_observed_unreturned_bytes": 128, "rented_bytes": 256,
                "rent_count": 9 if self_test else 3, "return_count": 9 if self_test else 3, "unreturned_rental_bytes": 0,
                "unmatched_return_count": 0, "unmatched_return_bytes": 0, "unreturned_by_type": [],
                "event_source": "System.Buffers.ArrayPoolEventSource", "contract": "separate untimed numeric pool batch",
                "numeric_types_observed": ["float", "int", "uint", "byte", "short", "char", "Vector256<float>", "Vector512<float>", "CtcRun"],
                "schemas": {"BufferRented": ["bufferId", "bufferSize", "poolId", "bucketId"],
                            "BufferReturned": ["bufferId", "bufferSize", "poolId"]},
                "pools": [{"pool_id": i, "name": name, "element_bytes": width} for i, (name, width) in enumerate(acceptance.POOL_TYPES.items())],
                "discard_boundary_self_test": {"supported": True, "expected_no_return_event": True,
                    "requested_elements": 8193, "actual_elements": 16384, "unreturned_capacity_bytes": 65536,
                    "measurements": {"rent_count": 1, "return_count": 0, "unreturned_rental_bytes": 65536}} if self_test else None}

    @staticmethod
    def capture(engine="rust", mode="latency", budget=1, wall=100):
        workers = budget if mode == "throughput" else 1
        cases = []
        for family in acceptance.FAMILIES:
            for image in acceptance.WORKLOADS:
                width = 1280 if image == "text-long" else 64
                line = {"detection": {"points": [[0., 0.], [100., 0.], [100., 20.], [0., 20.]], "score": .9},
                        "text": "中 OEM", "recognition_score": .8, "classification_score": .75, "classification_label": 0,
                        "applied_rotation_degrees": 0, "emitted_count": 1, "recognition_content_width": width,
                        "recognition_tensor_width": width, "recognition_time_steps": 80,
                        "ctc_spans": [{"text": "中 OEM", "score": .8, "start_column": 1, "end_column": 3}]}
                lines = [copy.deepcopy(line) for _ in range({"dense-multilingual": 32, "text-long": 3}.get(image, 1))]
                result = {"text": "\n".join(line["text"] for line in lines), "detected_count": len(lines),
                          "detector_resized_width": 320, "detector_resized_height": 640, "lines": lines}
                memory = {phase: {"activation_bytes": 1000, "workspace_bytes": 100} for phase in ("warmed", "after_samples", "repeated_shape")}
                for state in memory.values():
                    state.update({"kernel_scratch_bytes": 40 if engine == "rust" else None,
                                  "nonkernel_workspace_bytes": 60 if engine == "rust" else None,
                                  "kernel_scratch_contract": "retained subset" if engine == "rust" else "unavailable ArrayPool/stack accounting",
                                  "workspace_contract": "observed payloads plus Count-based collection-slot estimate",
                                  "workspace_measurement": "observed_payloads_plus_collection_count_estimate"})
                    if engine == "csharp":
                        state.update({"activation_usable_bytes": 744, "activation_guard_bytes": 256,
                                      "activation_contract": "native requested allocation including64 guard floats"})
                memory.update({"allocation_probe": {"allocated_bytes": 50}, "peak_process_working_set_bytes": 12345})
                if engine == "csharp":
                    memory["numeric_pool_scratch_probe"] = AcceptanceContracts.pool_probe()
                cases.append({"id": f"{family}/{image}", "result": result, "line_count": len(lines), "max_recognition_tensor_width": width,
                              "model_load_ns": 100, "compile_requests_ns": 100, "shared": {"normalized_weight_bytes": 1000, "packed_weight_bytes": 500},
                              "cold": None if mode == "validate" else {"wall_ns": 1000, "includes_binding_and_reshape": True, "instrumented": True,
                                                                        "profiles" if engine == "rust" else "profile": [{"binding_ns": 11, "reshape_ns": 10} for _ in range(workers)]},
                              "samples": [{"wall_ns": wall, "requests": workers} for _ in range(5)], "memory": memory,
                              "numeric_pool_scratch_qualification": AcceptanceContracts.pool_probe() if engine == "csharp" else None})
        capture = {"schema_version": 1, "complete": True, "engine": engine, "mode": mode, "budget": budget,
                "workers": workers, "capacity_per_request": 1 if mode == "throughput" else budget,
                "owned_outputs": True, "warmups": 2, "sample_count": 5, "iterations": 1, "cases": cases}
        if engine == "csharp":
            capture["runtime_identity"] = {"framework_description": ".NET 10.0.11", "environment_version": "10.0.11",
                "process_architecture": "X64", "os_description": "Microsoft Windows NT", "contract": "Inherited controls; unset is not disabled",
                "jit_environment": acceptance.jit_environment({})}
            capture["array_pool_probe_support"] = AcceptanceContracts.pool_probe(True) if mode == "validate" else None
            capture["scratch_only"] = False
            capture["scratch_observer_enabled"] = mode == "validate"
        return capture

    def pair(self, mode, original, rust):
        left = acceptance.validate_capture(self.capture("csharp", mode, wall=original), "csharp", mode, 1, 5, 1)
        right = acceptance.validate_capture(self.capture("rust", mode, wall=rust), "rust", mode, 1, 5, 1)
        return acceptance.compare_pair(left, right, self.tolerances, mode)

    def test_each_workload_has_its_own_latency_and_throughput_gate(self):
        self.assertTrue(self.pair("latency", 100, 110)["passed"])
        self.assertFalse(self.pair("latency", 100, 111)["passed"])
        self.assertTrue(self.pair("throughput", 900, 1000)["passed"])
        self.assertFalse(self.pair("throughput", 900, 1001)["passed"])

    def test_a_single_retained_memory_growth_fails_an_otherwise_faster_pair(self):
        left = acceptance.validate_capture(self.capture("csharp"), "csharp", "latency", 1, 5, 1)
        for phase in ("after_samples", "repeated_shape"):
            for field in acceptance.RUST_RETAINED_FIELDS:
                with self.subTest(phase=phase, field=field):
                    capture = self.capture(wall=50)
                    retained = next(case for case in capture["cases"] if case["id"] == "medium_rec/text-long")["memory"]
                    # Producer-supplied extra metadata cannot override the
                    # driver's computed requirement, engine or verdict.
                    retained.update({"required_for_acceptance": False, "bounded_reuse": True,
                                     "growth_observed": False, "engine": "csharp"})
                    memory = retained[phase]
                    memory[field] += 1
                    # Keep the measured partition valid. Kernel/nonkernel
                    # growth must fail even if total workspace is unchanged.
                    if field == "workspace_bytes":
                        memory["nonkernel_workspace_bytes"] += 1
                    elif field == "kernel_scratch_bytes":
                        memory["nonkernel_workspace_bytes"] -= 1
                    elif field == "nonkernel_workspace_bytes":
                        memory["kernel_scratch_bytes"] -= 1
                    right = acceptance.validate_capture(capture, "rust", "latency", 1, 5, 1)
                    report = acceptance.compare_pair(left, right, self.tolerances, "latency")
                    self.assertFalse(report["passed"])
                    failed = [row for row in report["cases"] if not row["passed"]]
                    self.assertEqual([row["id"] for row in failed], ["medium_rec/text-long"])
                    rust = next(item for item in failed[0]["memory"] if item["engine"] == "rust")
                    self.assertTrue(rust["required_for_acceptance"])
                    self.assertEqual(rust["deltas_from_warmed"][phase][field], 1)

    def test_reference_growth_remains_visible_without_becoming_a_rust_acceptance_requirement(self):
        capture = self.capture("csharp")
        memory = capture["cases"][0]["memory"]
        memory["after_samples"]["workspace_bytes"] += 7
        memory["repeated_shape"]["workspace_bytes"] += 9
        left = acceptance.validate_capture(capture, "csharp", "latency", 1, 5, 1)
        right = acceptance.validate_capture(self.capture(wall=50), "rust", "latency", 1, 5, 1)
        report = acceptance.compare_pair(left, right, self.tolerances, "latency")
        self.assertTrue(report["passed"])
        row = next(case for case in report["cases"] if case["id"] == capture["cases"][0]["id"])
        reference = next(item for item in row["memory"] if item["engine"] == "csharp")
        self.assertFalse(reference["required_for_acceptance"])
        self.assertFalse(reference["observed_stable"])
        self.assertFalse(reference["bounded_reuse"])
        self.assertTrue(reference["growth_observed"])
        self.assertFalse(reference["exact_total_bound_established"])
        self.assertEqual(reference["deltas_from_warmed"]["after_samples"], {"activation_bytes": 0, "workspace_bytes": 7})
        self.assertEqual(reference["deltas_from_warmed"]["repeated_shape"], {"activation_bytes": 0, "workspace_bytes": 9})
        self.assertEqual(reference["warmed"], memory["warmed"])
        self.assertEqual(reference["after_samples"], memory["after_samples"])
        self.assertEqual(reference["repeated_shape"], memory["repeated_shape"])
        policy = report["retained_memory_policy"]
        self.assertEqual(policy["version"], 2)
        self.assertEqual(policy["id"], "rust-bounded-reuse-reference-observation-v2")
        self.assertTrue(policy["rust"]["required_for_acceptance"])
        self.assertEqual(policy["rust"]["metrics"], list(acceptance.RUST_RETAINED_FIELDS))
        self.assertFalse(policy["csharp"]["required_for_acceptance"])
        # An observational reference still has to provide every phase and
        # numeric-pool diagnostic; missing evidence cannot become a pass.
        del capture["cases"][0]["memory"]["repeated_shape"]["workspace_bytes"]
        with self.assertRaisesRegex(RuntimeError, "retained-memory"):
            acceptance.validate_capture(capture, "csharp", "latency", 1, 5, 1)

    def test_missing_cases_false_partitions_and_unmeasured_samples_fail(self):
        for change in (lambda c: c["cases"].pop(), lambda c: c.update({"capacity_per_request": 4}),
                       lambda c: c["cases"][0].update({"samples": []}),
                       lambda c: c["cases"][0]["samples"][0].update({"wall_ns": 0})):
            capture = self.capture(mode="throughput", budget=4)
            change(capture)
            with self.assertRaises(RuntimeError):
                acceptance.validate_capture(capture, "rust", "throughput", 4, 5, 1)

    def test_kernel_scratch_accounting_is_partitioned_or_explicitly_unavailable(self):
        rust = self.capture()
        rust["cases"][0]["memory"]["warmed"]["kernel_scratch_bytes"] += 1
        with self.assertRaisesRegex(RuntimeError, "partition"):
            acceptance.validate_capture(rust, "rust", "latency", 1, 5, 1)
        original = self.capture("csharp")
        original["cases"][0]["memory"]["warmed"]["kernel_scratch_bytes"] = 0
        with self.assertRaisesRegex(RuntimeError, "explicit null"):
            acceptance.validate_capture(original, "csharp", "latency", 1, 5, 1)
        original = self.capture("csharp")
        original["cases"][0]["memory"]["warmed"]["activation_guard_bytes"] = 0
        with self.assertRaisesRegex(RuntimeError, "usable/guard"):
            acceptance.validate_capture(original, "csharp", "latency", 1, 5, 1)

    def test_numeric_pool_probe_requires_verified_schema_and_complete_balanced_rentals(self):
        for change in (lambda p: p.update({"supported": False}), lambda p: p.update({"return_count": 2}),
                       lambda p: p.update({"unreturned_rental_bytes": 1}), lambda p: p.update({"schemas": {}})):
            capture = self.capture("csharp", "validate")
            change(capture["array_pool_probe_support"])
            with self.assertRaises(RuntimeError):
                acceptance.validate_capture(capture, "csharp", "validate", 1, 5, 1)
        capture = self.capture("csharp")
        del capture["cases"][0]["memory"]["numeric_pool_scratch_probe"]
        with self.assertRaisesRegex(RuntimeError, "ArrayPool"):
            original = acceptance.validate_capture(capture, "csharp", "latency", 1, 5, 1)
            rewritten = acceptance.validate_capture(self.capture(), "rust", "latency", 1, 5, 1)
            acceptance.compare_pair(original, rewritten, self.tolerances, "latency")

    def test_valid_pool_discard_is_reported_without_claiming_a_live_leak(self):
        probe = self.pool_probe()
        probe.update({"rent_count": 4, "peak_observed_unreturned_bytes": 65536, "rented_bytes": 66000,
                      "unreturned_rental_bytes": 65536,
                      "unreturned_by_type": [{"type": "int", "count": 1, "bytes": 65536, "all_meet_discard_capacity": True}]})
        acceptance.validate_pool_probe(probe, False)
        for field in ("rented_bytes", "rent_count", "peak_observed_unreturned_bytes"):
            changed = self.pool_probe()
            changed[field] = 0
            with self.assertRaises(RuntimeError):
                acceptance.validate_pool_probe(changed, False)

    def test_missing_loading_shared_or_cold_phase_evidence_is_fatal(self):
        for field in ("model_load_ns", "compile_requests_ns", "shared", "cold"):
            capture = self.capture()
            del capture["cases"][0][field]
            with self.assertRaises(RuntimeError):
                acceptance.validate_capture(capture, "rust", "latency", 1, 5, 1)

    def test_pool_diagnostics_run_in_a_separate_process_with_exact_outputs(self):
        original = acceptance.validate_capture(self.capture("csharp"), "csharp", "latency", 1, 5, 1)
        diagnostic = self.capture("csharp")
        diagnostic.update({"scratch_only": True, "scratch_observer_enabled": True,
                           "array_pool_probe_support": self.pool_probe(True)})
        for case in diagnostic["cases"]:
            case["numeric_pool_scratch_probe"] = self.pool_probe()
        acceptance.attach_pool_probe(original, diagnostic, "latency", 1)
        diagnostic["cases"][0]["result"]["lines"][0]["recognition_score"] += .000001
        with self.assertRaisesRegex(RuntimeError, "changed public results"):
            acceptance.attach_pool_probe(original, diagnostic, "latency", 1)
        capture = self.capture("csharp")
        capture["scratch_observer_enabled"] = True
        with self.assertRaisesRegex(RuntimeError, "never enable"):
            acceptance.validate_capture(capture, "csharp", "latency", 1, 5, 1)

    def test_workload_qualification_uses_actual_lines_and_widths(self):
        capture = self.capture()
        capture["cases"][0]["line_count"] = 32
        with self.assertRaisesRegex(RuntimeError, "summary"):
            acceptance.validate_capture(capture, "rust", "latency", 1, 5, 1)
        capture = self.capture()
        long = next(case for case in capture["cases"] if case["id"] == "rec/text-long")
        long["max_recognition_tensor_width"] = 320
        for line in long["result"]["lines"]:
            line["recognition_tensor_width"] = 320
        with self.assertRaisesRegex(RuntimeError, "width of 1280"):
            acceptance.validate_capture(capture, "rust", "latency", 1, 5, 1)

    def test_dense_corpus_is_two_exact_vertical_copies_without_a_decoder(self):
        source = self.root / "source"
        source.mkdir()
        entries = []
        raw = (gate.ROOT / "tools/corpus-source/sample.bgr").read_bytes()
        (source / "sample.bgr").write_bytes(raw)
        entries.append({"id": "sample", "file": "sample.bgr", "width": 320, "height": 320,
                        "sha256": hashlib.sha256(raw).hexdigest()})
        gate.write_json(source / "corpus.json", {"schema_version": 1, "pixel_format": "bgr24", "images": entries})
        corpus = acceptance.prepare_corpus(source, self.root / "prepared")
        self.assertEqual([row["id"] for row in corpus["images"]], list(acceptance.WORKLOADS))
        self.assertEqual((self.root / "prepared/dense-multilingual.bgr").read_bytes(), raw * 2)
        self.assertEqual((corpus["images"][1]["width"], corpus["images"][1]["height"]), (320, 640))
        long = (self.root / "prepared/text-long.bgr").read_bytes()
        self.assertEqual((corpus["images"][2]["width"], corpus["images"][2]["height"]), (928, 128))
        self.assertEqual(hashlib.sha256(long).hexdigest(), "7b3731d14ac5f87a518f7c86c8c44862677c35945cf3dca1be7f2f23d0d2d02e")
        for top in (12, 48, 84):
            glyph_row = raw[(250 * 320 + 15) * 3:(250 * 320 + 240) * 3]
            self.assertEqual(long[(top * 928 + 14) * 3:(top * 928 + 914) * 3], glyph_row * 4)
        self.assertEqual(long[:928 * 12 * 3], bytes([255]) * (928 * 12 * 3))

    def test_three_round_limit_persists_across_outputs_and_rejects_the_fourth(self):
        ledger = self.root / "ledger.json"
        for number in range(3):
            acceptance.reserve_round(ledger, "same-task", ["sample", "dense", "long"], str(number), self.root / str(number))
        with self.assertRaisesRegex(RuntimeError, "three performance rounds"):
            acceptance.reserve_round(ledger, "same-task", ["sample"], "fourth", self.root / "fourth")
        self.assertEqual(len(gate.read_json(ledger)["tasks"]["same-task"]["sample"]), 3)

    def test_runtime_identity_does_not_require_revalidating_new_benchmark_only_code(self):
        before = {"files": {"src/lib.rs": "runtime", "Cargo.lock": "deps", "examples/cpu_parity.rs": "old"}}
        after = {"files": {**before["files"], "examples/cpu_parity.rs": "benchmark-only", "examples/support/cpu_acceptance.rs": "new"}}
        self.assertEqual(acceptance.runtime_sources(before), acceptance.runtime_sources(after))
        after["files"]["src/lib.rs"] = "changed"
        self.assertNotEqual(acceptance.runtime_sources(before), acceptance.runtime_sources(after))

    def test_full_reference_isa_and_concurrent_budget_are_explicit(self):
        env = acceptance.environment("throughput", 8)
        self.assertEqual(env["DOTNET_PROCESSOR_COUNT"], "1")
        self.assertEqual(env["SIMD_OCR_BACKEND"], "cpu")
        self.assertNotIn("DOTNET_EnableAVX512F", env)
        self.assertEqual(acceptance.environment("latency", 8)["DOTNET_PROCESSOR_COUNT"], "8")

    def test_runtime_jit_controls_are_recorded_without_changing_inheritance(self):
        inherited = {"DOTNET_TieredCompilation": "0", "COMPlus_TieredPGO": "1", "DOTNET_TC_CallCountingDelayMs": "123",
                     "COMPlus_ReadyToRun": "0", "DOTNET_JitDisasm": "SelectedMethod", "UNRELATED_SECRET": "omit"}
        before = inherited.copy()
        variables = acceptance.jit_environment(inherited)
        self.assertEqual(inherited, before)
        self.assertEqual(variables["DOTNET_TIEREDCOMPILATION"], "0")
        self.assertEqual(variables["COMPLUS_TIEREDPGO"], "1")
        self.assertEqual(variables["DOTNET_TC_CALLCOUNTINGDELAYMS"], "123")
        self.assertIsNone(variables["DOTNET_TC_QUICKJIT"])
        self.assertNotIn("UNRELATED_SECRET", variables)
        capture = self.capture("csharp")
        capture["runtime_identity"]["jit_environment"] = variables
        acceptance.validate_capture(capture, "csharp", "latency", 1, 5, 1)
        for key in ("runtime_identity",):
            invalid = copy.deepcopy(capture)
            invalid.pop(key)
            with self.assertRaisesRegex(RuntimeError, "runtime/JIT"):
                acceptance.validate_capture(invalid, "csharp", "latency", 1, 5, 1)
        for change in (lambda identity: identity.update({"process_architecture": "Arm64"}),
                       lambda identity: identity["jit_environment"].pop("DOTNET_TC_QUICKJIT"),
                       lambda identity: identity["jit_environment"].update({"DOTNET_TIEREDPGO": False})):
            invalid = copy.deepcopy(capture)
            change(invalid["runtime_identity"])
            with self.assertRaisesRegex(RuntimeError, "runtime/JIT"):
                acceptance.validate_capture(invalid, "csharp", "latency", 1, 5, 1)


if __name__ == "__main__":
    unittest.main()
