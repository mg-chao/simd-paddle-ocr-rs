"""Frozen build identity/copy contracts; synthetic payloads never execute inference."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
POWERSHELL = shutil.which("pwsh")
GIT = shutil.which("git")


def quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def identity(path, relative):
    raw = path.read_bytes()
    return {"path": relative, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def inventory_hash(entries):
    return hashlib.sha256("".join(f"{entry['path']}\0{entry['sha256']}\n" for entry in entries).encode()).hexdigest()


@unittest.skipUnless(os.name == "nt" and POWERSHELL and GIT, "Windows prepared benchmark launcher contract")
class PreparedBuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        helper = ROOT / "tools/benchmark-build.ps1"
        result = subprocess.run([POWERSHELL, "-NoProfile", "-Command", f". {quote(helper)}; @{{flags=(Get-OcrBuildFlags);toolchain=(Get-OcrBuildToolchain {quote(ROOT)})}}|ConvertTo-Json -Depth 8"], capture_output=True, timeout=30)
        if result.returncode:
            raise RuntimeError(result.stderr.decode(errors="replace"))
        cls.current_build = json.loads(result.stdout)

    def setUp(self):
        self.parent = (ROOT / "target/tool-tests").resolve()
        self.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="ocr-prepared-", dir=self.parent)
        self.root = Path(self.temporary.name).resolve()
        self.assertTrue(self.root.is_relative_to(self.parent))
        self.rust = self.root / "rust"
        self.native = self.root / "native"
        self.bundle = self.root / "reference"
        self.prepared = self.root / "prepared"
        for folder in (self.rust, self.native, self.bundle, self.prepared):
            folder.mkdir()
        for folder in (self.rust, self.native):
            subprocess.run([GIT, "init", "--quiet", str(folder)], check=True, capture_output=True)
        (self.rust / "source.rs").write_text("pub fn result() -> u32 { 42 }\n")
        (self.native / "source.cs").write_text("class NativeSource {}\n")
        sources = [identity(self.rust / "source.rs", "source.rs")]
        native_sources = [identity(self.native / "source.cs", "source.cs")]
        for prefix, origin, entries in (("source", self.rust, sources), ("reference-source", self.native, native_sources)):
            (self.prepared / prefix).mkdir()
            for entry in entries:
                shutil.copyfile(origin / entry["path"], self.prepared / prefix / entry["path"])
        models = []
        for role in ("det", "cls", "rec", "dictionary"):
            path = self.bundle / (role + ".bin")
            path.write_bytes((role + " frozen model").encode())
            models.append(identity(path, path.name) | {"id": role})
        self.reference = {"schema_version": 1, "contract": {"profile": "canonical", "capture_scope": "ocr", "model": "tiny", "options": {"box_threshold": 0.4}}, "models": models}
        (self.bundle / "manifest.json").write_text(json.dumps(self.reference))
        reference_binding = {"manifest_bytes": (self.bundle / "manifest.json").stat().st_size,
                             "manifest_sha256": hashlib.sha256((self.bundle / "manifest.json").read_bytes()).hexdigest(),
                             "model_family": "tiny", "contract": self.reference["contract"], "models": models}
        executables = []
        for relative in ("ocr-timing.exe", "ocr-heap.exe", "csharp/SimdPaddleOCR.ReferenceExport.dll", "csharp/Sdcb.SimdPaddleOCR.dll",
                         "csharp/SimdPaddleOCR.ReferenceExport.runtimeconfig.json", "csharp/SimdPaddleOCR.ReferenceExport.deps.json"):
            path = self.prepared / relative
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(("frozen synthetic " + relative).encode())
            executables.append(identity(path, relative))
        self.manifest = {"schema_version": 1, "kind": "ocr_prepared_benchmark_build", "status": "prepared", "profiling": False,
                         "timing_features": [], "heap_features": ["heap-profiler"],
                         "effective_build_flags": self.current_build["flags"], "toolchain": self.current_build["toolchain"],
                         "source_files": sources, "reference_source_files": native_sources, "source_sha256": inventory_hash(sources),
                         "reference_source_sha256": inventory_hash(native_sources), "references": [reference_binding], "executables": executables}
        self.manifest_path = self.prepared / "build.json"
        self.save()

    def tearDown(self):
        self.assertTrue(self.root.resolve().is_relative_to(self.parent))
        self.temporary.cleanup()

    def save(self):
        self.manifest_path.write_text(json.dumps(self.manifest))

    def verify(self, extra="", after=""):
        code = (f"$ErrorActionPreference='Stop'; . {quote(ROOT / 'tools/benchmark-build.ps1')}; "
                f"$prepared=Assert-OcrPreparedBuild -ManifestPath {quote(self.manifest_path)} -RustRoot {quote(self.rust)} "
                f"-ReferenceRoot {quote(self.native)} -ReferenceBundle {quote(self.bundle)} {extra}; {after}; Write-Output 'VERIFIED'")
        return subprocess.run([POWERSHELL, "-NoProfile", "-Command", code], capture_output=True, timeout=30)

    def assert_rejected(self, detail):
        result = self.verify()
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn(detail, result.stderr.decode(errors="replace"))

    def test_verified_bytes_reused_across_groups_without_building(self):
        destinations = [self.root / "group-one", self.root / "group-two"]
        calls = "; ".join(f"Copy-OcrPreparedExecutables $prepared {quote(path)}" for path in destinations)
        result = self.verify(after=calls)
        self.assertEqual(result.returncode, 0, result.stderr)
        for entry in self.manifest["executables"]:
            for folder in destinations:
                self.assertEqual(identity(folder / entry["path"], entry["path"]), entry)

    def test_changed_production_or_tool_source_rejected(self):
        (self.rust / "source.rs").write_text("pub fn result() -> u32 { 43 }\n")
        self.assert_rejected("Production/tool source identity differs")

    def test_changed_native_source_rejected(self):
        (self.native / "source.cs").write_text("class ChangedSource {}\n")
        self.assert_rejected("Native source identity differs")

    def test_changed_frozen_executable_rejected(self):
        (self.prepared / "ocr-timing.exe").write_bytes(b"tampered")
        self.assert_rejected("Corrupt frozen identity")

    def test_changed_effective_build_flag_rejected(self):
        self.manifest["effective_build_flags"] = dict(self.manifest["effective_build_flags"], RUSTFLAGS="-C target-cpu=native")
        self.save()
        self.assert_rejected("Effective build flag differs: RUSTFLAGS")

    def test_changed_model_or_options_binding_rejected(self):
        self.reference["contract"]["options"]["box_threshold"] = 0.45
        (self.bundle / "manifest.json").write_text(json.dumps(self.reference))
        self.assert_rejected("Frozen model/dictionary/options/reference binding differs")

    def test_changed_model_payload_rejected(self):
        (self.bundle / "rec.bin").write_bytes(b"wrong model")
        self.assert_rejected("Corrupt frozen identity: rec.bin")

    def test_wrong_feature_or_missing_native_role_rejected(self):
        self.manifest["profiling"] = True
        self.save()
        self.assert_rejected("contract/features differ")
        self.manifest["profiling"] = False
        self.manifest["executables"] = self.manifest["executables"][:-1]
        self.save()
        self.assert_rejected("Missing frozen executable role")

    def test_escaping_identity_path_rejected(self):
        self.manifest["executables"][0]["path"] = "../outside.exe"
        self.save()
        self.assert_rejected("Unsafe frozen identity path")

    def test_allocator_feature_cannot_enter_timing_executable(self):
        self.manifest["timing_features"] = ["heap-profiler"]
        self.save()
        self.assert_rejected("Frozen timing/heap feature identities differ")

    def cargo_artifact(self, messages):
        path = self.root / "cargo.stdout"
        path.write_text("\n".join(json.dumps(message) for message in messages))
        code = (f"$ErrorActionPreference='Stop'; . {quote(ROOT / 'tools/benchmark-build.ps1')}; "
                f"Get-OcrCargoExecutable @{{StdoutPath={quote(path)}}}")
        return subprocess.run([POWERSHELL, "-NoProfile", "-Command", code], capture_output=True, timeout=20)

    def test_cargo_artifact_selects_emitted_custom_target_executable(self):
        actual = str(self.root / "custom-cargo-target/release/ocr-benchmark.exe")
        result = self.cargo_artifact([
            {"reason": "compiler-artifact", "target": {"name": "library"}, "executable": None},
            {"reason": "compiler-artifact", "target": {"name": "another-binary"}, "executable": "stale-other.exe"},
            {"reason": "compiler-artifact", "target": {"name": "ocr-benchmark"}, "executable": actual},
            {"reason": "build-finished", "success": True},
        ])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.decode().strip(), actual)

    def test_missing_or_multiple_cargo_artifacts_rejected(self):
        record = {"reason": "compiler-artifact", "target": {"name": "ocr-benchmark"}, "executable": "benchmark.exe"}
        for records in ([], [record, record]):
            result = self.cargo_artifact(records)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("exactly one benchmark executable", result.stderr.decode(errors="replace"))


if __name__ == "__main__":
    unittest.main()
