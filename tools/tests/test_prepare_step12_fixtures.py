"""Offline acquisition must accept both hash-pinned dictionary byte forms."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import tempfile
import unittest
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


if __name__ == "__main__":
    unittest.main()
