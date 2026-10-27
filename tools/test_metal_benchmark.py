"""Protocol checks; no OCR workloads or performance measurements."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

from metal_benchmark import invoke


class ReadinessProtocol(unittest.TestCase):
    def call(self, source, timeout=2):
        with tempfile.TemporaryDirectory() as temporary:
            return invoke([sys.executable, "-u", "-c", source], os.environ.copy(), Path(temporary), "probe", timeout)

    def test_readiness_precedes_serialization(self):
        source = "\n".join([
            "import time, json", "print('RESULT_READY sample cold 1.25', flush=True)",
            "time.sleep(.15)",
            "print(json.dumps({'kind':'image','id':'sample','phase':'cold','ms':1.25,'output':{}}))",
            "print(json.dumps({'kind':'summary','calls':1,'backend_verified':True}))",
        ])
        result = self.call(source)
        self.assertGreater(result["elapsed_seconds"] * 1000 - result["launch_to_result_ms"], 100)

    def test_missing_marker_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "missing ready"):
            self.call("print('{\"kind\":\"summary\",\"calls\":0,\"backend_verified\":true}')")

    def test_mismatched_timing_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "mismatched"):
            self.call("print('RESULT_READY sample cold 1'); print('{\"kind\":\"image\",\"id\":\"sample\",\"phase\":\"cold\",\"ms\":2}'); print('{\"kind\":\"summary\",\"calls\":1,\"backend_verified\":true}')")

    def test_failed_child_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "exit 3"):
            self.call("raise SystemExit(3)")

    def test_timeout_terminates_child(self):
        with self.assertRaises(TimeoutError):
            self.call("import time; time.sleep(10)", timeout=.05)


if __name__ == "__main__":
    unittest.main()
