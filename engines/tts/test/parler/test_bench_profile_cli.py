"""Model-free argument checks for optional decoder profiling."""
import subprocess
import sys
import unittest

BENCH = sys.argv.pop(1)


class BenchProfileCli(unittest.TestCase):
    def run_bench(self, *args):
        return subprocess.run([BENCH, *args], capture_output=True, text=True, timeout=10)

    def test_help_describes_opt_in(self):
        result = self.run_bench("--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("--profile-coreml-decoder", result.stderr)
        self.assertIn("requires --coreml-decoder", result.stderr)

    def test_profile_requires_native_request_before_model_load(self):
        result = self.run_bench("--profile-coreml-decoder", "--model", "/missing.gguf", "--text", "a")
        self.assertEqual(result.returncode, 2)
        self.assertIn("--profile-coreml-decoder requires --coreml-decoder", result.stderr)
        self.assertNotIn("load failed", result.stderr)

    def test_native_request_accepts_profile_option(self):
        result = self.run_bench("--profile-coreml-decoder", "--coreml-decoder",
                                "--model", "/missing.gguf", "--text", "a")
        self.assertEqual(result.returncode, 1)
        self.assertIn("load failed", result.stderr)
        self.assertNotIn("Core ML decoder profile", result.stdout)


if __name__ == "__main__":
    unittest.main()
