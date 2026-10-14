"""Offline acquisition must accept both hash-pinned dictionary byte forms."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import tempfile
import unittest
import warnings
from unittest.mock import patch
import zipfile

SCRIPT = Path(__file__).resolve().parents[1] / "prepare-step12-fixtures.py"
SPEC = importlib.util.spec_from_file_location("step12_prepare", SCRIPT)
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)


def descriptor(data):
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


class OfflineDictionaryTests(unittest.TestCase):
    def acquire(self, supplied):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            models = root / "models"
            models.mkdir()
            local = models / "keys.txt"
            local.write_bytes(supplied)
            canonical, transport = b"first\r\nsecond\r\n", b"first\nsecond\n"
            model = descriptor(canonical) | {
                "source_name": "keys.txt",
                "download": descriptor(transport) | {"transform": "lf_to_crlf"},
            }
            args = SimpleNamespace(cache=root / "cache", models=models)
            with patch("urllib.request.urlopen", side_effect=AssertionError("offline path used network")):
                result = PREPARE.obtain(model, args, "https://unused.invalid/")
            self.assertEqual(result.read_bytes(), canonical)
            self.assertEqual(local.read_bytes(), supplied)

    def test_canonical_offline_bytes_are_preserved(self):
        self.acquire(b"first\r\nsecond\r\n")

    def test_pinned_lf_checkout_is_converted_without_mutating_source(self):
        self.acquire(b"first\nsecond\n")

    def test_unpinned_local_bytes_are_rejected(self):
        with self.assertRaises(ValueError):
            self.acquire(b"wrong\n")


class LocalGoldenTests(unittest.TestCase):
    def run_preparation(self, root, *arguments):
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--goldens", str(root / "goldens"),
             "--out", str(root / "out"), "--cache", str(root / "cache"), *arguments],
            capture_output=True, text=True,
        )

    def write_index(self, root, local_only):
        goldens = root / "goldens"
        goldens.mkdir()
        with zipfile.ZipFile(goldens / "fixture.zip", "w") as archive:
            archive.writestr("manifest.json", b'{"fixture": true}\n')
        fixture = descriptor((goldens / "fixture.zip").read_bytes()) | {
            "archive": "fixture.zip", "directory": "fixture", "models": [],
        }
        if local_only:
            fixture["local_only"] = True
        (goldens / "index.json").write_text(json.dumps({
            "schema_version": 1, "model_url_root": "https://unused.invalid/",
            "fixtures": [fixture],
        }), encoding="utf-8")
        return goldens / "fixture.zip"

    def test_default_preparation_succeeds_without_local_archives(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.write_index(root, local_only=True).unlink()
            result = self.run_preparation(root)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Skipped local-only golden", result.stdout)
            self.assertFalse((root / "out").exists())

    def test_explicit_local_preparation_restores_supplied_archive(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.write_index(root, local_only=True)
            result = self.run_preparation(root, "--include-local")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((root / "out/fixture/manifest.json").read_bytes(),
                             b'{"fixture": true}\n')

    def test_explicit_local_preparation_rejects_missing_or_changed_archive(self):
        for payload in (None, b"corrupted archive"):
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                archive = self.write_index(root, local_only=True)
                if payload is None:
                    archive.unlink()
                else:
                    archive.write_bytes(payload)
                result = self.run_preparation(root, "--include-local")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Golden archive hash/size differs", result.stderr)

    def test_required_archive_remains_mandatory(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.write_index(root, local_only=False).unlink()
            result = self.run_preparation(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Golden archive hash/size differs", result.stderr)


class HostedGoldenTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.goldens = self.root / "goldens"
        self.goldens.mkdir()
        self.args = SimpleNamespace(goldens=self.goldens, out=self.root / "out", cache=self.root / "cache",
                                    models=None, offline=False)

    def archive(self, contents=b"frozen pixels", extras=()):
        manifest = {"schema_version": 1, "models": [], "cases": [{"id": "page", "artifacts": [
            descriptor(b"frozen pixels") | {"path": "pixels.bin", "id": "input.pixels"}]}]}
        output = io.BytesIO()
        with warnings.catch_warnings(), zipfile.ZipFile(output, "w") as archive:
            warnings.simplefilter("ignore", UserWarning)
            archive.writestr("manifest.json", json.dumps(manifest).encode())
            archive.writestr("pixels.bin", contents)
            for name in extras:
                archive.writestr(name, b"extra")
        data = output.getvalue()
        fixture = descriptor(data) | {"archive": "fixture.zip", "directory": "fixture-v1", "models": [],
                                      "url": "https://example.invalid/fixtures-v1/fixture.zip"}
        return fixture, data

    def test_cold_hosted_acquisition_then_offline_cache_checks_every_payload(self):
        fixture, data = self.archive()
        with patch("urllib.request.urlopen", return_value=io.BytesIO(data)) as request:
            PREPARE.restore(fixture, self.args, "https://unused.invalid/")
            request.assert_called_once()
        self.args.offline = True
        with patch("urllib.request.urlopen", side_effect=AssertionError("offline used network")):
            PREPARE.restore(fixture, self.args, "https://unused.invalid/")
            destination = self.args.out / fixture["directory"]
            (destination / "pixels.bin").write_bytes(b"corrupted")
            (destination / "stale.bin").write_bytes(b"stale")
            PREPARE.restore(fixture, self.args, "https://unused.invalid/")
        self.assertEqual((destination / "pixels.bin").read_bytes(), b"frozen pixels")
        self.assertFalse((destination / "stale.bin").exists())
        self.assertEqual([path.name for path in self.args.out.iterdir()], [fixture["directory"]])

    def test_wrong_download_never_enters_verified_cache(self):
        fixture, _ = self.archive()
        with patch("urllib.request.urlopen", return_value=io.BytesIO(b"wrong payload")):
            with self.assertRaisesRegex(ValueError, "Acquired golden archive hash/size differs"):
                PREPARE.restore(fixture, self.args, "https://unused.invalid/")
        self.assertFalse((self.args.cache / "archives" / fixture["sha256"] / fixture["archive"]).exists())
        self.assertFalse(self.args.out.exists())

    def test_corrupt_cache_is_reacquired_and_bad_supplied_archive_cannot_hide(self):
        fixture, data = self.archive()
        cached = self.args.cache / "archives" / fixture["sha256"] / fixture["archive"]
        cached.parent.mkdir(parents=True)
        cached.write_bytes(b"corrupted cache")
        with patch("urllib.request.urlopen", return_value=io.BytesIO(data)) as request:
            self.assertEqual(PREPARE.obtain_archive(fixture, self.args), cached)
            request.assert_called_once()
        (self.goldens / fixture["archive"]).write_bytes(b"bad supplied archive")
        with patch("urllib.request.urlopen", side_effect=AssertionError("bad supplied source used network")):
            with self.assertRaisesRegex(ValueError, "Golden archive hash/size differs"):
                PREPARE.obtain_archive(fixture, self.args)

    def test_offline_archive_and_model_cache_misses_fail_without_network(self):
        fixture, _ = self.archive()
        self.args.offline = True
        model = descriptor(b"model") | {"source_name": "rec.onnx"}
        with patch("urllib.request.urlopen", side_effect=AssertionError("offline used network")):
            with self.assertRaisesRegex(ValueError, "Offline golden archive"):
                PREPARE.obtain_archive(fixture, self.args)
            with self.assertRaisesRegex(ValueError, "Offline model cache"):
                PREPARE.obtain(model, self.args, "https://unused.invalid/")

    def test_invalid_supplied_model_is_not_hidden_by_cache_or_restored_output(self):
        fixture, data = self.archive()
        self.args.models = self.root / "supplied-models"
        self.args.models.mkdir()
        local = self.args.models / "rec.onnx"
        local.write_bytes(b"model")
        model = descriptor(b"model") | {"source_name": "rec.onnx", "path": "models/rec.onnx"}
        fixture["models"] = [model]
        (self.goldens / fixture["archive"]).write_bytes(data)
        self.args.offline = True
        PREPARE.restore(fixture, self.args, "https://unused.invalid/")
        original = (self.args.out / fixture["directory"] / "manifest.json").read_bytes()
        local.write_bytes(b"invalid")
        with patch("urllib.request.urlopen", side_effect=AssertionError("invalid supplied model used network")):
            with self.assertRaisesRegex(ValueError, "Local model"):
                PREPARE.obtain(model, self.args, "https://unused.invalid/")
            with self.assertRaisesRegex(ValueError, "Local model"):
                PREPARE.restore(fixture, self.args, "https://unused.invalid/")
        self.assertEqual((self.args.out / fixture["directory"] / "manifest.json").read_bytes(), original)

    def test_payload_failure_preserves_previous_prepared_bundle(self):
        fixture, data = self.archive()
        (self.goldens / fixture["archive"]).write_bytes(data)
        PREPARE.restore(fixture, self.args, "https://unused.invalid/")
        previous = (self.args.out / fixture["directory"] / "manifest.json").read_bytes()
        bad_fixture, bad_data = self.archive(contents=b"wrong inner payload")
        (self.goldens / fixture["archive"]).write_bytes(bad_data)
        with self.assertRaisesRegex(ValueError, "Golden archive payload hash/size differs"):
            PREPARE.restore(bad_fixture, self.args, "https://unused.invalid/")
        self.assertEqual((self.args.out / fixture["directory"] / "manifest.json").read_bytes(), previous)
        self.assertEqual((self.args.out / fixture["directory"] / "pixels.bin").read_bytes(), b"frozen pixels")
        self.assertEqual([path.name for path in self.args.out.iterdir()], [fixture["directory"]])

    def test_zip_paths_and_duplicate_names_are_rejected_before_output(self):
        for name in ("../outside.bin", "/outside.bin", "C:/outside.bin", "nested\\outside.bin",
                     "pixels.bin", "PIXELS.BIN", "pixels.bin.", "nul.bin"):
            with self.subTest(name=name):
                fixture, data = self.archive(extras=(name,))
                (self.goldens / fixture["archive"]).write_bytes(data)
                with self.assertRaises(ValueError):
                    PREPARE.restore(fixture, self.args, "https://unused.invalid/")
                self.assertFalse(self.args.out.exists())

    def test_selections_do_not_acquire_unrequested_fixtures(self):
        fixture, data = self.archive()
        (self.goldens / fixture["archive"]).write_bytes(data)
        missing = fixture | {"archive": "missing.zip", "directory": "unselected-v1"}
        (self.goldens / "index.json").write_text(json.dumps({"schema_version": 1,
            "model_url_root": "https://unused.invalid/", "fixtures": [fixture, missing]}), encoding="utf-8")
        command = [sys.executable, str(SCRIPT), "--goldens", str(self.goldens), "--out", str(self.args.out),
                   "--cache", str(self.args.cache), "--offline", "--fixture", fixture["directory"]]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((self.args.out / missing["directory"]).exists())
        result = subprocess.run(command + ["--fixture", "unknown"], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unknown fixture selection", result.stderr)


if __name__ == "__main__":
    unittest.main()
