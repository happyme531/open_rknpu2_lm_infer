"""Reject missing test execution and invalid exact-comparison evidence."""
import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import json
import subprocess


spec = importlib.util.spec_from_file_location("regression", Path(__file__).with_name("regression.py"))
regression = importlib.util.module_from_spec(spec)
spec.loader.exec_module(regression)


class RegressionEvidenceTests(unittest.TestCase):
    def test_zero_tests_is_not_a_pass(self):
        with self.assertRaises(RuntimeError):
            regression.check_test_output("test result: ok. 0 passed; 0 failed; 1 filtered out")
        regression.check_test_output("test result: ok. 1 passed; 0 failed;")

    def test_exact_comparison_rejects_empty_different_and_nan_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            left, right = [Path(temp)/name for name in ["left.f32", "right.f32"]]
            for a, b in [(b"", b""), (struct.pack("f", 1), struct.pack("f", 2)),
                         (struct.pack("f", float("nan")), struct.pack("f", float("nan")))]:
                left.write_bytes(a)
                right.write_bytes(b)
                with self.assertRaises(RuntimeError):
                    regression.compare_files(left, right)
            left.write_bytes(struct.pack("ff", 1, -2))
            right.write_bytes(left.read_bytes())
            self.assertEqual(regression.compare_files(left, right), {"equal": True, "bytes": 8})

    def test_shared_nfs_output_keeps_worker_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            original = '{"passed":true,"cases":{"old":{"passed":true}}}'
            (output / "summary.json").write_text(original)
            config = {"boards": {"primary": {"ssh": "board", "output": str(output), "config": {}}}}
            result = subprocess.CompletedProcess([], 0, stdout=original)
            with patch.object(regression.subprocess, "run", return_value=result):
                self.assertEqual(regression.remote_boards(config, output, True), 0)
            self.assertEqual((output / "summary.json").read_text(), original)
            self.assertTrue(json.loads((output / "boards.json").read_text())["passed"])

    def test_resume_preserves_failed_attempt(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            binary = output / "binary"
            binary.write_bytes(b"test executable")
            (output / "summary.json").write_text(json.dumps({"cases": {"failed": {"passed": False}}}))
            (output / "failed.log").write_text("original allocation failure")
            suite = regression.Suite({"binaries": {"current": str(binary)}}, output, True)

            def run(*args, **kwargs):
                (output / "failed.clocks.json").write_text(json.dumps({"exit_code": 0, "restore_errors": []}))
                return subprocess.CompletedProcess([], 0)

            with patch.object(regression.subprocess, "run", side_effect=run):
                suite.run("failed", [str(binary)])
            previous = output / suite.report["cases"]["failed"]["previous_attempt"]
            self.assertEqual((previous / "failed.log").read_text(), "original allocation failure")
            self.assertTrue(suite.report["cases"]["failed"]["passed"])


if __name__ == "__main__":
    unittest.main()
