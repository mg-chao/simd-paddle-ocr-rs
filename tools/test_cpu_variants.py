"""Model-free contracts for full-model backend/thread validation."""
import hashlib
import copy
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from unittest import mock

import cpu_parity as gate
import cpu_variants as variants


class VariantContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_matrix_covers_intermediate_stages_portable_and_native_thread_counts(self):
        self.assertEqual(variants.VARIANTS, (
            ("simd-c1", "simd", "auto", 1), ("nhwc-c1", "nhwc", "auto", 1),
            ("portable4-c1", "auto", "portable4", 1), ("auto-c2", "auto", "auto", 2),
            ("auto-c4", "auto", "auto", 4), ("auto-c8", "auto", "auto", 8)))

    def test_forced_isa_matrix_is_explicit_and_rejects_duplicates_or_unknowns(self):
        self.assertEqual(variants.build_matrix(""), variants.VARIANTS)
        self.assertEqual(variants.build_matrix("avx,avx-fma,avx2-fma,avx512")[-4:], (
            ("avx-c1", "auto", "avx", 1), ("avx-fma-c1", "auto", "avx-fma", 1),
            ("avx2-fma-c1", "auto", "avx2-fma", 1), ("avx512-c1", "auto", "avx512", 1)))
        for invalid in ("avx,avx", "unknown", "avx,", "scalar"):
            with self.assertRaises(ValueError):
                variants.build_matrix(invalid)

    def test_native_parallel_tolerance_is_exact_without_changing_portable_budgets(self):
        locked = {"tensor_atol": .001, "tensor_rtol": .001,
                  "score_atol": .01, "coordinate_atol": .5}
        exact = variants.numerical_tolerances(locked, exact=True)
        portable = variants.numerical_tolerances(locked, exact=False)
        self.assertEqual(portable, locked)
        self.assertTrue(all(value == 0 for value in exact.values()))
        records = []
        # Adjacent finite f32 values: acceptable backend rounding, forbidden for
        # output-tile parallelism that preserves each element's arithmetic.
        for index, bits in enumerate((0x3f800000, 0x3f800001)):
            directory = self.root / str(index)
            directory.mkdir()
            data = struct.pack("<I", bits)
            (directory / "x.f32").write_bytes(data)
            records.append({"file": "x.f32", "shape": [1], "sha256": hashlib.sha256(data).hexdigest()})
        self.assertTrue(gate.compare_tensor(self.root / "0", records[0], self.root / "1", records[1], portable)["passed"])
        self.assertFalse(gate.compare_tensor(self.root / "0", records[0], self.root / "1", records[1], exact)["passed"])
        self.assertTrue(gate.compare_result({"score": .5}, {"score": .500001}, exact))

    def test_forced_fma_metadata_uses_serialized_enum_names_without_losing_the_cli_policy(self):
        baseline = self.root / "baseline"
        gate.write_json(baseline / "corpus/corpus.json", {})
        directory = self.root / "forced"
        for cli, serialized in (("avx-fma", "avx_fma"), ("avx2-fma", "avx2_fma")):
            gate.write_json(directory / "capture.json", {"schema_version": 1, "engine": "rust", "mode": "auto",
                "cases": [{"id": str(i)} for i in range(149)], "metadata": {"threads": 1,
                "compile_options": {"kernels": serialized, "layout": "Auto", "fusion": True},
                "requested_backend": cli, "correctness_only": True, "ocr_batch_size": 1,
                "os": "windows", "architecture": "x86_64", "profiling_enabled": True, "images_decoded_inside_timing": False,
                "corpus_sha256": gate.sha256(baseline / "corpus/corpus.json")}})
            with (mock.patch.object(gate, "run"), mock.patch.object(gate, "compare_bundles", return_value={"passed": True})):
                result = variants.capture_variant(["producer", "--backend", cli], directory, baseline, "auto", cli, 1,
                                                  {}, {}, "source", {"tolerances": {}})
            self.assertTrue(result["passed"])
            metadata = gate.read_json(directory / "capture.json")["metadata"]
            self.assertEqual(metadata["variant_validation"]["requested_backend"], cli)
            self.assertEqual(metadata["compile_options"]["kernels"], serialized)
            producer_block = copy.deepcopy(metadata["variant_validation"])
            proof = {"command": ["producer", "--backend", cli], "reported_status": "passed"}
            with (mock.patch.object(gate, "run") as run, mock.patch.object(gate, "compare_bundles", return_value={"passed": True})):
                variants.capture_variant(proof["command"], directory, baseline, "auto", cli, 1, {}, {}, "source", {"tolerances": {}}, proof)
                run.assert_not_called()
            revalidated = gate.read_json(directory / "capture.json")["metadata"]
            self.assertEqual(revalidated["variant_validation"], producer_block)
            self.assertTrue(revalidated["variant_revalidation"]["revalidation_only"])

    def test_reused_capture_raw_scope_and_original_validation_are_strictly_bound(self):
        command = ["verified-producer", "--backend", "avx-fma"]
        provenance, host = {"executable_sha256": "verified"}, {"host": "verified"}
        metadata = {"requested_backend": "avx-fma", "correctness_only": True, "ocr_batch_size": 1,
                    "os": "windows", "architecture": "x86_64", "profiling_enabled": True, "images_decoded_inside_timing": False}
        block = {"mode": "auto", "requested_backend": "avx-fma", "threads": 1, "rust_build": provenance,
                 "source_run_sha256": "source", "command": command, "validation_host": host,
                 "effective_threads": 1, "timings_are_diagnostic": True}
        def verify(value, required=False):
            variants.verify_produced_capture({"metadata": value}, "auto", "avx-fma", 1, provenance, host, "source", command, required)
        verify(metadata)
        with self.assertRaises(RuntimeError):
            verify(metadata, True)
        metadata["variant_validation"] = block
        verify(metadata, True)
        for field, value in (("requested_backend", "scalar"), ("correctness_only", False), ("ocr_batch_size", 4),
                             ("os", "linux"), ("architecture", "aarch64")):
            changed = copy.deepcopy(metadata)
            changed[field] = value
            with self.assertRaises(RuntimeError):
                verify(changed)
        for field, value in (("rust_build", {"executable_sha256": "foreign"}), ("validation_host", {"host": "foreign"}),
                             ("source_run_sha256", "foreign"), ("command", ["foreign"]), ("mode", "foreign"),
                             ("requested_backend", "foreign"), ("threads", 4), ("effective_threads", 4)):
            changed = copy.deepcopy(metadata)
            changed["variant_validation"][field] = value
            with self.assertRaises(RuntimeError):
                verify(changed)

    def test_retained_commands_cannot_change_paths_values_or_flags(self):
        directory = (self.root / "capture").resolve()
        command = ["exe", "--models-dir", "models", "--corpus", "corpus", "--output", str(directory),
                   "--mode", "auto", "--backend", "avx-fma", "--threads", "1", "--iterations", "1", "--correctness-only"]
        entry = {"command": command, "capture_directory": str(directory)}
        variants.verify_producing_command(entry, command, directory)
        for index in range(len(command)):
            changed = command.copy()
            changed[index] = "foreign"
            with self.assertRaises(RuntimeError):
                variants.verify_producing_command({**entry, "command": changed}, command, directory)
        with self.assertRaises(RuntimeError):
            variants.verify_producing_command({**entry, "command": command + ["--unknown"]}, command, directory)

    def test_reuse_keeps_original_producer_and_rejects_changed_host_or_binary(self):
        original = {"corpus_sha256": "corpus"}
        provenance = {"sources": "same", "executable_sha256": "same"}
        prior = {"schema_version": 1, "source_run_sha256": "source", "rust_build": provenance,
                 "validation_host": {"host": "same"}, "model_hashes": {}, "corpus_sha256": "corpus",
                 "reference_lock_sha256": gate.sha256(gate.LOCK), "comparison_helper_sha256": gate.sha256(Path(gate.__file__))}
        variants.verify_reuse_identity(prior, original, provenance, {"host": "same"}, "source", {"models": {}})
        for changed in ({**provenance, "executable_sha256": "other"}, {**provenance, "sources": "other"}):
            with self.assertRaisesRegex(RuntimeError, "different executable"):
                variants.verify_reuse_identity(prior, original, changed, {"host": "same"}, "source", {"models": {}})
        with self.assertRaisesRegex(RuntimeError, "different executable"):
            variants.verify_reuse_identity(prior, original, provenance, {"host": "other"}, "source", {"models": {}})

    def test_changed_binary_or_source_fails_before_running_inference(self):
        executable = self.root / "parity"
        executable.write_bytes(b"verified binary")
        sources = {"sha256": "verified sources", "files": {"src/runtime.rs": "abc"}}
        provenance = {"provenance": "built_by_driver", "executable_sha256": gate.sha256(executable), "sources": sources}
        with mock.patch.object(gate, "rust_sources", return_value=sources):
            variants.verify_current(executable, provenance)
            executable.write_bytes(b"replaced binary")
            with self.assertRaisesRegex(RuntimeError, "executable changed"):
                variants.verify_current(executable, provenance)
        executable.write_bytes(b"verified binary")
        with mock.patch.object(gate, "rust_sources", return_value={"sha256": "edited"}):
            with self.assertRaisesRegex(RuntimeError, "sources changed"):
                variants.verify_current(executable, provenance)
        with self.assertRaisesRegex(RuntimeError, "maintained driver"):
            variants.verify_current(executable, {"provenance": "preexisting_executable_unverified"})

    def test_incomplete_or_quick_baseline_is_rejected(self):
        gate.write_json(self.root / "run.json", {"schema_version": 1, "quick": True, "threads": 1})
        gate.write_json(self.root / "report.json", {"schema_version": 1, "passed": False})
        with self.assertRaisesRegex(RuntimeError, "successful"):
            variants.read_identity(self.root, {})
        gate.write_json(self.root / "report.json", {"schema_version": 1, "passed": True})
        with self.assertRaisesRegex(RuntimeError, "full 149-case"):
            variants.read_identity(self.root, {})

    def test_host_identity_cannot_change_between_variants(self):
        host = {"host_name": "host", "cpu_identity": "cpu", "operating_system": "os", "architecture": "x86_64"}
        with mock.patch.object(variants, "current_host", return_value=host):
            self.assertEqual(variants.verify_host(host), host)
            for field in host:
                changed = {**host, field: "different"}
                with self.assertRaisesRegex(RuntimeError, "host/CPU/OS/architecture"):
                    variants.verify_host(changed)

    def full_identity(self):
        lock = {"revision": "pinned", "models": {}}
        lock_path = self.root / "reference-lock.json"
        gate.write_json(lock_path, lock)
        images = [{"id": str(index), "public_case": index < 9, "e2e_case": index < 7} for index in range(11)]
        gate.write_json(self.root / "corpus" / "corpus.json", {"images": images})
        ids = [f"graph/{name}/{index}" for name in gate.GRAPH_NODE_COUNTS for index in range(3)]
        for image in images:
            id_ = image["id"]
            ids.extend(f"preprocess/{kind}/{id_}" for kind in ("det", "cls", "rec"))
            ids.append(f"crop/{id_}")
            if image["public_case"]:
                ids.append(f"classify/{id_}")
                for rec, det in (("rec", "det"), ("small_rec", "small_det"), ("medium_rec", "medium_det")):
                    ids.extend((f"recognize/{rec}/{id_}", f"detect/{det}/{id_}"))
                    if image["e2e_case"]:
                        ids.append(f"ocr/{rec}/{id_}")
        runtime = self.root / "binaries/csharp/Sdcb.SimdPaddleOCR.Tests.dll"
        runtime.parent.mkdir(parents=True)
        runtime.write_bytes(b"frozen reference exporter")
        metadata = {"schema_version": 1, "quick": False, "threads": 1, "backend": None,
                    "reference_lock_sha256": gate.sha256(lock_path), "reference_revision": "pinned",
                    "model_hashes": {}, "corpus_sha256": "corpus", "benchmark_environment": {"host_name": "test"},
                    "rust_build": {"provenance": "built_by_driver", "sources": {"sha256": "rust"}},
                    "exporter_build": {"provenance": "built_by_driver", "reference_revision": "pinned",
                                       "sources": {"sha256": "csharp", "files": {"historical.cs": "fingerprint"}},
                                       "executable_sha256": gate.sha256(runtime),
                                       "runtime_files": {runtime.name: gate.sha256(runtime)}}}
        gate.write_json(self.root / "run.json", metadata)
        gate.write_json(self.root / "report.json", {"schema_version": 1, "passed": True})
        for engine in ("csharp", "rust"):
            for mode in ("reference", "auto"):
                details = {"threads": 1, "corpus_sha256": "corpus"}
                if engine == "rust":
                    details.update({"rust_build": metadata["rust_build"], "benchmark_environment": metadata["benchmark_environment"],
                                    "compile_options": {"kernels": "auto" if mode == "auto" else "scalar",
                                                        "layout": "Auto" if mode == "auto" else "Nchw", "fusion": mode == "auto"}})
                else:
                    details.update({"exporter_build": metadata["exporter_build"], "hardware_intrinsics": "1" if mode == "auto" else "0",
                                    "nhwc": "1" if mode == "auto" else "0", "no_fuse": "0" if mode == "auto" else "1"})
                gate.write_json(self.root / f"{engine}-{mode}" / "capture.json", {
                    "schema_version": 1, "engine": engine, "mode": mode, "metadata": details,
                    "cases": [{"id": id_} for id_ in ids]})
        return lock, lock_path

    def test_replay_requires_frozen_historical_runtime_and_not_current_exporter_sources(self):
        lock, lock_path = self.full_identity()
        with (mock.patch.object(gate, "LOCK", lock_path),
              mock.patch.object(gate, "exporter_sources", side_effect=AssertionError("historical binary is used"))):
            variants.read_identity(self.root, lock)
            runtime = self.root / "binaries/csharp/Sdcb.SimdPaddleOCR.Tests.dll"
            runtime.write_bytes(b"changed reference exporter")
            with self.assertRaisesRegex(RuntimeError, "runtime file changed"):
                variants.read_identity(self.root, lock)

    def test_mixed_capture_identity_fails_even_with_matching_149_case_ids(self):
        lock, lock_path = self.full_identity()
        with mock.patch.object(gate, "LOCK", lock_path):
            variants.read_identity(self.root, lock)
            path = self.root / "rust-auto" / "capture.json"
            original = gate.read_json(path)
            for field, value in (("corpus_sha256", "stale"), ("rust_build", {"sources": "stale"}),
                                 ("benchmark_environment", {"host_name": "other"})):
                capture = gate.read_json(path)
                capture["metadata"][field] = value
                gate.write_json(path, capture)
                with self.assertRaisesRegex(RuntimeError, "Capture-level"):
                    variants.read_identity(self.root, lock)
                gate.write_json(path, original)

    def test_subprocess_failure_is_checkpointed_with_backend_budget_and_command(self):
        source, output = self.root / "source", self.root / "output"
        source.mkdir()
        gate.write_json(source / "run.json", {})
        lock_path = self.root / "reference-lock.json"
        gate.write_json(lock_path, {"models": {}, "tolerances": {}})
        identity = {"rust_build": {}, "exporter_build": {"sources": {}}, "corpus_sha256": "corpus",
                    "benchmark_environment": {}}
        with (mock.patch.object(gate, "LOCK", lock_path),
              mock.patch.object(variants, "read_identity", return_value=identity),
              mock.patch.object(variants, "verify_host", return_value={}),
              mock.patch.object(variants, "verify_current"),
              mock.patch.object(variants, "verify_inputs"),
              mock.patch.object(variants, "verify_original", return_value={"passed": True}),
              mock.patch.object(gate, "source_snapshot", return_value={"files": {}}),
              mock.patch.object(gate, "archive_sources"),
              mock.patch.object(gate, "exporter_sources", return_value={}),
              mock.patch.object(variants, "capture_variant", side_effect=subprocess.CalledProcessError(7, ["parity"]))):
            with self.assertRaises(subprocess.CalledProcessError):
                variants.main(["--bundle", str(source), "--models-dir", str(self.root), "--output", str(output)])
        report = gate.read_json(output / "report.json")
        self.assertFalse(report["complete"])
        self.assertFalse(report["passed"])
        failure = report["variants"][0]
        self.assertEqual((failure["mode"], failure["backend"], failure["threads"], failure["status"]), ("simd", "auto", 1, "failed"))
        self.assertIn("--backend", failure["command"])
        self.assertIn("7", failure["error"])
        self.assertIn("comparison_helper_sha256", report)


if __name__ == "__main__":
    unittest.main()
