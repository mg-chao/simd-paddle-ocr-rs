"""Opt-in physical-macOS regression for incremental Metal catalog linking.

Run after sourcing env.sh:
    SIMD_OCR_TEST_METAL_BUILD=1 python3 -m unittest discover -s tools -p test_metal_build.py

Uses a temporary two-shader catalog and the real build module/toolchain. It does
not edit the production catalog or execute OCR/performance measurements.
"""
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
METAL = ROOT / "src/runtime/gpu/metal"


@unittest.skipUnless(platform.system() == "Darwin" and os.environ.get("SIMD_OCR_TEST_METAL_BUILD") == "1",
                     "opt-in macOS test requiring Xcode, Metal Toolchain and cached Cargo dependencies")
class MetalBuildTests(unittest.TestCase):
    def test_catalog_removal_readdition_and_missing_library(self):
        evidence = []
        report = os.environ.get("SIMD_OCR_METAL_BUILD_REPORT")
        if report:
            report = Path(report)
            report.mkdir(parents=True, exist_ok=False)
        with tempfile.TemporaryDirectory(prefix="ocr-metal-build-") as directory:
            root = Path(directory)
            src = root / "src"
            src.mkdir()
            module = src / "metal"
            module.mkdir()
            shutil.copy2(METAL / "build.rs", module / "build.rs")
            shutil.copytree(METAL / "shaders", module / "shaders")
            (src / "main.rs").write_text('#[path = "metal/build.rs"] mod metal;\nfn main() { metal::build(); }\n')
            (root / "Cargo.toml").write_text(
                '[package]\nname = "metal-build-probe"\nversion = "0.0.0"\nedition = "2024"\n'
                '[dependencies]\nserde_json = "1"\nminiz_oxide = "=0.8.9"\n[workspace]\n')
            catalog = (METAL / "catalog.rs").read_text()
            prefix = catalog[:catalog.index("pub const ENTRIES:")]
            suffix = catalog[catalog.index("pub fn source("):]

            def select(names):
                entries = []
                for name in names:
                    entries.append('Entry { id: "' + name + '", entry: "probe", '
                                   'body: "kernel void probe(device uint* x [[buffer(0)]], '
                                   'constant Params& p [[buffer(1)]]) { x[0] += 1u; }", '
                                   'defines: "", mask: 0, threads: 1, reads: &[false], '
                                   'neural: false, test: true },')
                (module / "catalog.rs").write_text(prefix + "pub const ENTRIES: &[Entry] = &[\n" +
                                                   "\n".join(entries) + "\n];\n" + suffix)

            # Record real tool invocations so a catalog-only change cannot pass
            # by recompiling every shader or silently retaining the old library.
            toolchain = root / "toolchain"
            binaries = toolchain / "usr/bin"
            binaries.mkdir(parents=True)
            log = root / "tools.jsonl"
            for name in ["metal", "metallib", "metal-tt"]:
                actual = subprocess.check_output(["xcrun", "--find", name], text=True).strip()
                wrapper = binaries / name
                wrapper.write_text('#!/usr/bin/env python3\nimport json, os, sys\n'
                                   f'with open({str(log)!r}, "a") as stream:\n'
                                   f'    stream.write(json.dumps([{name!r}, sys.argv[1:]]) + "\\n")\n'
                                   f'os.execv({actual!r}, [{actual!r}, *sys.argv[1:]])\n')
                wrapper.chmod(0o700)
            out = root / "out"
            out.mkdir()
            env = dict(os.environ, OUT_DIR=str(out), SIMD_OCR_METAL_TOOLCHAIN=str(toolchain))
            artifacts = ["ocr.metallib", "ocr.binary.metallib", "metal-objects.txt"]

            def build(label, names, links):
                log.write_text("")
                result = subprocess.run(["cargo", "run", "--offline", "--release", "--quiet",
                                         "--manifest-path", str(root / "Cargo.toml"),
                                         "--target-dir", str(root / "target")],
                                        cwd=ROOT, env=env, text=True, capture_output=True, timeout=180)
                calls = [json.loads(line) for line in log.read_text().splitlines()]
                if report:
                    (report / f"{label}.stdout.log").write_text(result.stdout)
                    (report / f"{label}.stderr.log").write_text(result.stderr)
                    (report / f"{label}.tools.json").write_text(json.dumps(calls, indent=2) + "\n")
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(sum(name == "metallib" for name, _ in calls), links)
                self.assertEqual(sum(name == "metal-tt" for name, _ in calls), links)
                if label != "initial":
                    self.assertEqual(sum(name == "metal" and "-c" in args for name, args in calls), 0)
                self.assertEqual((out / "metal-objects.txt").read_text().splitlines(), names)
                library = (out / "ocr.metallib").read_bytes()
                self.assertIn(b"ocr_KeepProbe_0", library)
                self.assertEqual(b"ocr_DropProbe_0" in library, "DropProbe" in names)
                hashes = {name: hashlib.sha256((out / name).read_bytes()).hexdigest() for name in artifacts}
                evidence.append(dict(step=label, inventory=names, calls=calls, sha256=hashes,
                                     portable_bytes=len(library)))
                return hashes

            select(["KeepProbe", "DropProbe"])
            build("initial", ["KeepProbe", "DropProbe"], 1)
            select(["KeepProbe"])
            removed = build("removed", ["KeepProbe"], 1)
            self.assertEqual(build("unchanged", ["KeepProbe"], 0), removed)
            select(["KeepProbe", "DropProbe"])
            build("readded", ["KeepProbe", "DropProbe"], 1)
            (out / "ocr.metallib").unlink()
            build("missing-library", ["KeepProbe", "DropProbe"], 1)
        if report:
            (report / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    unittest.main()
