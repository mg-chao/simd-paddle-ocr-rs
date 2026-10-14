"""Versioned packaging preserves oracle bytes/coverage and keeps models external."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

SCRIPT = Path(__file__).resolve().parents[1] / "pack-step12-fixtures.py"
SPEC = importlib.util.spec_from_file_location("fixture_pack", SCRIPT)
PACK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PACK)


class FixturePackagingTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.source = self.root / "source"
        self.source.mkdir()
        def saved(path, contents, **fields):
            target = self.source / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(contents)
            return {"path": path, "bytes": len(contents), "sha256": hashlib.sha256(contents).hexdigest()} | fields
        models = [saved("models/" + name, name.encode(), id=identifier) for identifier, name in
                  (("det", "det.onnx"), ("cls", "cls.onnx"), ("rec", "rec.onnx"))]
        models.append(saved("models/keys.txt", b"one\r\ntwo\r\n", id="dictionary"))
        self.manifest = {"contract": {"model": "tiny"}, "models": models, "cases": [
            {"id": case, "artifacts": [saved(case + "/pixels.bin", case.encode(), id="input.pixels")]
             } for case in ("upright", "rotated", "blank", "tiny")], "auxiliary_files": []}
        (self.source / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def test_complete_and_case_shard_archives_are_deterministic_and_model_free(self):
        first = PACK.package(self.source, self.root / "first.zip", lambda cases: cases, "det.onnx")
        second = PACK.package(self.source, self.root / "second.zip", lambda cases: cases, "det.onnx")
        self.assertEqual(first["sha256"], second["sha256"])
        with zipfile.ZipFile(self.root / "first.zip") as archive:
            self.assertFalse(any(name.startswith("models/") for name in archive.namelist()))
            manifest = json.loads(archive.read("manifest.json"))
            self.assertEqual(manifest["cases"], self.manifest["cases"])
            self.assertEqual(manifest["models"], self.manifest["models"])
            self.assertEqual(manifest["fixture_selection"]["source_manifest_sha256"],
                             PACK.file_digest(self.source / "manifest.json"))
        shard = PACK.package(self.source, self.root / "shard.zip",
                             lambda cases: [case for case in cases if case["id"] == "rotated"], "det.onnx")
        with zipfile.ZipFile(self.root / "shard.zip") as archive:
            self.assertEqual([case["id"] for case in json.loads(archive.read("manifest.json"))["cases"]], ["rotated"])
        self.assertLess(shard["bytes"], first["bytes"])
        self.assertEqual(first["models"][-1]["download"]["transform"], "lf_to_crlf")

    def test_small_medium_asset_mapping_and_source_integrity_are_explicit(self):
        for variant in ("small", "medium"):
            with self.subTest(variant=variant):
                records = [PACK.source_asset(asset, (self.source / asset["path"]).read_bytes(), variant + "_det.onnx")
                           for asset in self.manifest["models"]]
                self.assertEqual([record["source_name"] for record in records],
                                 [variant + "_det.onnx", "cls.onnx", variant + "_rec.onnx", "rec_keys.txt"])
        (self.source / "upright/pixels.bin").write_bytes(b"corrupted")
        with self.assertRaisesRegex(ValueError, "source artifact integrity differs"):
            PACK.package(self.source, self.root / "bad.zip", lambda cases: cases, "det.onnx")


if __name__ == "__main__":
    unittest.main()
