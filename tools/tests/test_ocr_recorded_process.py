"""Small synthetic process checks; these never load OCR models or collect latency."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
POWERSHELL = shutil.which("pwsh")


def quote(value):
    return "'" + str(value).replace("'", "''") + "'"


@unittest.skipUnless(os.name == "nt" and POWERSHELL, "Windows PowerShell runner contract")
class RecordedProcessTests(unittest.TestCase):
    def setUp(self):
        self.parent = (ROOT / "target/tool-tests").resolve()
        self.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="ocr-process-", dir=self.parent)
        self.directory = Path(self.temporary.name).resolve()
        self.assertTrue(self.directory.is_relative_to(self.parent))

    def tearDown(self):
        self.assertTrue(self.directory.resolve().is_relative_to(self.parent))
        self.temporary.cleanup()

    def invoke(self, code="print('ok')", stop=None, affinity=None, executable=None):
        extra = (f" -StopFile {quote(stop)}" if stop else "")
        extra += (f" -ProcessorAffinityMask {affinity}" if affinity is not None else "")
        script = (
            "$ErrorActionPreference='Stop'\n"
            f". {quote(ROOT / 'tools/benchmark-process.ps1')}\n"
            f"$run=Invoke-OcrRecordedProcess -RecordDirectory {quote(self.directory)} "
            f"-Identity synthetic -Executable {quote(executable or sys.executable)} "
            f"-Arguments @('-c',{quote(code)}) -WorkingDirectory {quote(ROOT)}{extra}\n"
            "$run.Record|ConvertTo-Json -Depth 12\n"
        )
        completed = subprocess.run([POWERSHELL, "-NoProfile", "-Command", script], capture_output=True, timeout=20)
        record = json.loads((self.directory / "synthetic.record.json").read_bytes())
        for field in ("raw_stdout", "raw_stderr"):
            identity = record[field]
            raw = (self.directory / identity["path"]).read_bytes()
            self.assertEqual(len(raw), identity["bytes"])
            self.assertEqual(hashlib.sha256(raw).hexdigest(), identity["sha256"])
        return completed, record

    def test_exact_pipe_bytes_and_complete_invocation(self):
        completed, record = self.invoke("import sys; sys.stdout.buffer.write(bytes.fromhex('ff000d0acebb')); sys.stderr.buffer.write(b'warning\\r\\n')")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual((self.directory / "synthetic.stdout").read_bytes(), bytes.fromhex("ff000d0acebb"))
        self.assertEqual((self.directory / "synthetic.stderr").read_bytes(), b"warning\r\n")
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["exit_code"], 0)
        self.assertGreater(record["process_id"], 0)
        self.assertTrue(record["started_at_utc"] and record["completed_at_utc"])

    def test_nonzero_exit_retains_partial_output(self):
        completed, record = self.invoke("import sys; print('partial'); sys.stderr.write('failed'); sys.exit(7)")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(record["status"], "exit_failed")
        self.assertEqual(record["exit_code"], 7)
        self.assertIn(b"partial", (self.directory / "synthetic.stdout").read_bytes())
        self.assertEqual((self.directory / "synthetic.stderr").read_bytes(), b"failed")

    def test_stop_before_start_has_durable_empty_payloads(self):
        stop = self.directory / "STOP"
        stop.write_text("stop")
        completed, record = self.invoke(stop=stop)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(record["status"], "interrupted_before_start")
        self.assertIsNone(record["process_id"])
        self.assertTrue(record["stop_requested"])
        self.assertEqual(record["raw_stdout"]["bytes"], 0)

    def test_stop_during_call_preserves_completed_process(self):
        stop = self.directory / "STOP"
        code = f"from pathlib import Path; Path({str(stop)!r}).write_text('stop'); print('completed')"
        completed, record = self.invoke(code, stop=stop)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["exit_code"], 0)
        self.assertTrue(record["stop_requested"] and record["stop_observed_at_utc"])

    def test_launch_failure_retains_error_and_raw_identities(self):
        completed, record = self.invoke(executable=self.directory / "missing-native-program.exe")
        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(record["status"], "launch_failed")
        self.assertTrue(record["error"])
        self.assertIsNone(record["process_id"])

    def test_affinity_failure_is_durable_and_child_is_stopped(self):
        completed, record = self.invoke("import time; time.sleep(10)", affinity=0)
        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(record["status"], "affinity_failed")
        self.assertFalse(record["affinity_applied"])
        self.assertTrue(record["error"])
        self.assertGreater(record["process_id"], 0)


if __name__ == "__main__":
    unittest.main()
