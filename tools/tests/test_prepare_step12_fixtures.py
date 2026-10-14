"""Offline acquisition must accept both hash-pinned dictionary byte forms."""
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
