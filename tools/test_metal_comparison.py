"""Acceptance-policy regressions without running OCR."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from metal_comparison import compare


class ComparisonPolicy(unittest.TestCase):
    def evaluate(self, *, tail=90., cer=.02, scope="all"):
        with tempfile.TemporaryDirectory() as directory:
            roots = [Path(directory) / name for name in ("csharp", "rust")]
            for root in roots:
                root.mkdir()
                manifest = dict(implementation=root.name, complete=True,
                                protocol=dict(scope=scope, presets=["tiny", "small", "medium"]),
                                runs=[dict(summary=dict(backend_verified=True)) for _ in range(162)])
                manifest.update({key: "same" for key in ("hardware", "machine", "platform", "csharp_revision", "inputs_sha256", "models_sha256")})
                (root / "manifest.json").write_text(json.dumps(manifest))

            def summary(manifest):
                rust = manifest["implementation"] == "rust"
                return [dict(preset=preset, metric=metric, n=3 if metric.startswith("sample") else 50,
                             mean_ms=90 if rust else 100, p50_ms=90 if rust else 100,
                             p95_ms=tail if rust else 100)
                        for preset in ("tiny", "small", "medium")
                        for metric in ("sample_cold", "sample_warm", "corpus_cold", "corpus_first_shape", "corpus_warm")]

            def accuracy(root, _):
                return dict(summary=[dict(preset=p, phase=phase, cer=cer if root.name == "rust" else .02)
                                     for p in ("tiny", "small", "medium") for phase in ("cold", "first_shape", "warm")])

            with patch("metal_comparison.summary_rows", side_effect=summary), patch("metal_comparison.report_accuracy", side_effect=accuracy):
                return compare(*roots, Path("unused"))

    def test_fast_medians_do_not_hide_tail_regression(self):
        result = self.evaluate(tail=101)
        self.assertFalse(result["latency_and_corpus_accuracy_passed"])
        failures = [g for g in result["gates"] if not g["passed"]]
        self.assertEqual(len(failures), 15)
        self.assertTrue(all(g["statistic"] == "p95_ms" for g in failures))

    def test_accuracy_limit_is_percentage_points(self):
        self.assertTrue(self.evaluate(cer=.02049)["latency_and_corpus_accuracy_passed"])
        self.assertFalse(self.evaluate(cer=.02051)["latency_and_corpus_accuracy_passed"])

    def test_sample_only_report_cannot_qualify(self):
        with self.assertRaisesRegex(ValueError, "all 50 corpus"):
            self.evaluate(scope="sample")


if __name__ == "__main__":
    unittest.main()
