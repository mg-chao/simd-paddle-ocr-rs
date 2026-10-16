"""Acceptance prerequisite contracts; synthetic files only, no model execution."""
import copy
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

import cpu_acceptance as acceptance
import cpu_parity as gate


class PreparationReached(Exception):
    pass


class AcceptancePrerequisiteContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.bundle = self.root / "baseline"
        self.output = self.root / "output"
        self.invocations = 0
        self.lock = {"revision": "pinned", "models": {}, "tolerances": {
            "tensor_atol": .001, "tensor_rtol": .001, "score_atol": .01, "coordinate_atol": .5}}
        self.lock_path = self.root / "reference-lock.json"
        gate.write_json(self.lock_path, self.lock)
        corpus = self.bundle / "corpus"
        corpus.mkdir(parents=True)
        (corpus / "pixel.bgr").write_bytes(b"\x00\x00\x00")
        images = [{"id": str(i), "width": 1, "height": 1, "file": "pixel.bgr",
                   "sha256": gate.sha256(corpus / "pixel.bgr"), "public_case": i < 9,
                   "e2e_case": i < 7} for i in range(11)]
        gate.write_json(corpus / "corpus.json", {"schema_version": 1, "pixel_format": "bgr24", "images": images})
        archived = self.bundle / "sources/rust"
        for name, value in {"src/lib.rs": "library", "examples/cpu_parity.rs": "old example"}.items():
            path = archived / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(value, encoding="utf-8")
        self.sources = gate.source_snapshot(archived, [archived / name for name in ("src/lib.rs", "examples/cpu_parity.rs")])
        gate.write_json(archived / "snapshot.json", self.sources)
        exporter_archive = self.bundle / "sources/exporter"
        exporter_archive.mkdir(parents=True)
        (exporter_archive / "Program.cs").write_text("pinned exporter", encoding="utf-8")
        exporter_sources = gate.source_snapshot(exporter_archive, [exporter_archive / "Program.cs"])
        gate.write_json(exporter_archive / "snapshot.json", exporter_sources)
        rust = self.bundle / "binaries/rust/cpu_parity.exe"
        native = self.bundle / "binaries/csharp/Sdcb.SimdPaddleOCR.Tests.dll"
        for path in (rust, native):
            path.parent.mkdir(parents=True)
            path.write_bytes(b"frozen producer")
        self.metadata = {"schema_version": 1, "quick": False, "batch_only": False, "threads": 1,
            "backend": "auto", "ocr_batch_size": 1, "reference_lock_sha256": gate.sha256(self.lock_path),
            "reference_revision": "pinned", "model_hashes": {}, "corpus_sha256": gate.sha256(corpus / "corpus.json"),
            "benchmark_environment": {"host_name": "historical producer"},
            "rust_build": {"provenance": "built_by_driver", "sources": self.sources, "executable_sha256": gate.sha256(rust)},
            "exporter_build": {"provenance": "built_by_driver", "reference_revision": "pinned", "sources": exporter_sources,
                "executable_sha256": gate.sha256(native), "runtime_files": {native.name: gate.sha256(native)}}}
        self.current_rustc = 'rustc 1.100.0-nightly\nhost: x86_64-pc-windows-msvc\nLLVM version: 23.1.1'
        self.current_build_flags = {'RUSTFLAGS': '', 'CARGO_ENCODED_RUSTFLAGS': '',
            'CARGO_BUILD_RUSTFLAGS': '', 'CARGO_BUILD_TARGET': '', 'manifest_sha256': 'manifest',
            'lock_sha256': 'lock', 'cargo_features': ['profiling'], 'cargo_configs': {}}
        self.metadata['benchmark_environment'].update({
            'rustc': self.current_rustc, 'build_flags': copy.deepcopy(self.current_build_flags)})
        gate.write_json(self.bundle / "run.json", self.metadata)
        gate.write_json(self.bundle / "report.json", {"schema_version": 1, "passed": True, "imports": {"passed": True}})
        ids = gate.required_case_ids(corpus, False)
        for engine in ("csharp", "rust"):
            for mode in ("reference", "auto"):
                directory = self.bundle / f"{engine}-{mode}"
                directory.mkdir()
                (directory / "value.f32").write_bytes(struct.pack("<f", 1.0))
                tensor = {"file": "value.f32", "shape": [1], "sha256": gate.sha256(directory / "value.f32")}
                details = {"threads": 1, "corpus_sha256": self.metadata["corpus_sha256"]}
                if engine == "rust":
                    details.update(rust_build=self.metadata["rust_build"], benchmark_environment=self.metadata["benchmark_environment"],
                        compile_options={"kernels": "auto" if mode == "auto" else "scalar",
                                         "layout": "Auto" if mode == "auto" else "Nchw", "fusion": mode == "auto"})
                else:
                    details.update(exporter_build=self.metadata["exporter_build"], hardware_intrinsics="1" if mode == "auto" else "0",
                                   nhwc="1" if mode == "auto" else "0", no_fuse="0" if mode == "auto" else "1")
                cases = [self.case(id_, tensor, engine, mode) for id_ in sorted(ids)]
                gate.write_json(directory / "capture.json", {"schema_version": 1, "engine": engine, "mode": mode,
                                                            "metadata": details, "cases": cases})
                if mode == "reference":
                    for name, count in gate.GRAPH_NODE_COUNTS.items():
                        gate.write_json(directory / "imports" / (name + ".json"), {"schema_version": 1,
                            "raw": {"inputs": ["x"], "outputs": ["y"], "nodes": ["raw"] * gate.RAW_GRAPH_NODE_COUNTS[name]},
                            "normalized": {"inputs": ["x"], "outputs": ["y"], "nodes": [{"axis": 1}] * count}})

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def case(id_, tensor, engine, mode):
        kind, *parts = id_.split("/")
        if kind in ("graph", "preprocess", "crop"):
            result = {"id": id_, "tensor": tensor}
            if kind == "graph" and parts[-1] == "0" and (engine, mode) != ("csharp", "auto"):
                result["nodes"] = [{"index": i, "name": "node", "op_type": "Identity", "tensor": tensor}
                                   for i in range(gate.GRAPH_NODE_COUNTS[parts[0]])]
            return result
        values = {
            "classify": {"label": 0, "score": .5, "resized_width": 1},
            "recognize": {"text": "", "score": 0., "emitted_count": 0, "resized_width": 1,
                          "tensor_width": 1, "time_steps": 1, "ctc_spans": []},
            "detect": {"boxes": [], "resized_width": 1, "resized_height": 1, "width_ratio": 1., "height_ratio": 1.},
            "ocr": {"text": "", "detected_count": 0, "detector_resized_width": 1, "detector_resized_height": 1, "lines": []}}
        return {"id": id_, "result": values[kind]}

    def invoke(self, sources=None):
        self.invocations += 1
        self.output = self.root / f"output-{self.invocations}"
        with (mock.patch.object(gate, "LOCK", self.lock_path),
              mock.patch.object(gate, "validate_inputs"),
              mock.patch.object(gate, "rust_sources", return_value=sources or self.sources),
              mock.patch.object(gate, "build_flags", side_effect=lambda: copy.deepcopy(self.current_build_flags)),
              mock.patch.object(gate, "require_tool", return_value="rustc"),
              mock.patch.object(acceptance, "read_round_policy", return_value={}),
              mock.patch.object(acceptance, "prepare_corpus", side_effect=PreparationReached) as prepare,
              mock.patch.object(gate, "run", side_effect=self.compiler_query) as run,
              mock.patch.object(acceptance, "reserve_round", side_effect=AssertionError("round must not reserve")) as reserve):
            try:
                acceptance.main(["--reference-dir", str(self.root), "--models-dir", str(self.root),
                    "--correctness-bundle", str(self.bundle), "--output", str(self.output), "--task-id", "same-task"])
            finally:
                for call in run.call_args_list:
                    self.assertEqual(call.args, (["rustc", "--version", "--verbose"],))
                    self.assertEqual(call.kwargs, {"capture": True})
                reserve.assert_not_called()
            return prepare

    def compiler_query(self, command, **kwargs):
        if command != ["rustc", "--version", "--verbose"] or kwargs != {"capture": True}:
            raise AssertionError("Only the compiler identity query may run; no build or inference")
        return self.current_rustc

    def reject(self):
        with self.assertRaises(RuntimeError):
            self.invoke()
        self.assertFalse(self.output.exists(), "Invalid prerequisite must fail before output creation")

    def test_corrupted_retained_tensor_is_rejected_before_preparation_or_build(self):
        (self.bundle / "rust-auto/value.f32").write_bytes(b"corrupt")
        self.reject()

    def test_self_consistent_tensor_change_cannot_reuse_an_old_pass_summary(self):
        path = self.bundle / "rust-auto/value.f32"
        path.write_bytes(struct.pack("<f", 2.0))
        capture_path = path.parent / "capture.json"
        capture = gate.read_json(capture_path)
        for case in capture["cases"]:
            if "tensor" in case:
                case["tensor"]["sha256"] = gate.sha256(path)
            for node in case.get("nodes", []):
                node["tensor"]["sha256"] = gate.sha256(path)
        gate.write_json(capture_path, capture)
        self.reject()

    def test_foreign_capture_producer_metadata_is_rejected_before_preparation_or_build(self):
        path = self.bundle / "rust-auto/capture.json"
        capture = gate.read_json(path)
        capture["metadata"]["rust_build"]["executable_sha256"] = "foreign"
        gate.write_json(path, capture)
        self.reject()

    def test_changed_frozen_producers_and_archived_sources_cannot_qualify(self):
        for name in ("binaries/rust/cpu_parity.exe", "binaries/csharp/Sdcb.SimdPaddleOCR.Tests.dll",
                     "sources/rust/src/lib.rs", "sources/exporter/Program.cs"):
            with self.subTest(name=name):
                path = self.bundle / name
                original = path.read_bytes()
                path.write_bytes(b"changed")
                self.reject()
                path.write_bytes(original)

    def test_changed_import_or_public_result_is_rejected_despite_matching_case_ids(self):
        for name in ("rust-reference/imports/cls.json", "rust-auto/capture.json"):
            with self.subTest(name=name):
                path = self.bundle / name
                original = path.read_bytes()
                value = gate.read_json(path)
                if "imports" in name:
                    value["normalized"]["nodes"][0]["axis"] = 2
                else:
                    next(case for case in value["cases"] if case["id"].startswith("classify/"))["result"]["label"] = 1
                gate.write_json(path, value)
                self.reject()
                path.write_bytes(original)

    def test_unchanged_library_allows_examples_only_drift_after_strict_revalidation(self):
        sources = copy.deepcopy(self.sources)
        sources["files"]["examples/cpu_parity.rs"] = "updated acceptance helper"
        sources["sha256"] = "new complete source identity"
        with self.assertRaises(PreparationReached):
            self.invoke(sources)
        self.assertTrue(self.output.is_dir())

    def test_library_drift_does_not_use_the_examples_only_exception(self):
        sources = copy.deepcopy(self.sources)
        sources["files"]["src/lib.rs"] = "changed library"
        with self.assertRaisesRegex(RuntimeError, "unchanged native Auto library"):
            self.invoke(sources)
        self.assertFalse(self.output.exists())

    def test_legacy_csharp_runtime_without_source_archive_is_recorded_honestly(self):
        archived = self.bundle / "sources/exporter"
        for path in archived.iterdir():
            path.unlink()
        archived.rmdir()
        with (mock.patch.object(gate, "LOCK", self.lock_path),
              mock.patch.object(gate, "build_flags", side_effect=lambda: copy.deepcopy(self.current_build_flags)),
              mock.patch.object(gate, "require_tool", return_value="rustc"),
              mock.patch.object(gate, "run", side_effect=self.compiler_query)):
            result = acceptance.validate_correctness_prerequisite(self.bundle, self.root, self.lock, self.sources)
        self.assertTrue(result["passed"])
        self.assertFalse(result["archived_exporter_sources_verified"])
        self.assertEqual(result["exporter_build"], self.metadata["exporter_build"])
        self.assertEqual(result["capture_sha256"]["rust-auto"], gate.sha256(self.bundle / "rust-auto/capture.json"))


if __name__ == "__main__":
    unittest.main()
