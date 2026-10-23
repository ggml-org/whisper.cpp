import argparse
from pathlib import Path
import subprocess
import sys
import unittest


class ComputeUnitCliTests(unittest.TestCase):
    executable = None

    def run_cli(self, *arguments):
        return subprocess.run([str(self.executable), *arguments], capture_output=True, text=True, check=False)

    def assert_accepted_configuration(self, *arguments):
        result = self.run_cli(*arguments)
        self.assertEqual(result.returncode, 1)
        self.assertIn("usage: mm3-replay", result.stderr)
        self.assertIn("--ane-compute-units CPU_ONLY|CPU_AND_GPU|CPU_AND_NE|ALL", result.stderr)
        self.assertNotIn("ANE validation error:", result.stderr)

    def test_accepts_all_four_compute_unit_options(self):
        for units in ["CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE", "ALL"]:
            with self.subTest(compute_units=units):
                self.assert_accepted_configuration("--ane-compute-units", units)

    def test_accepts_existing_default_configuration(self):
        self.assert_accepted_configuration()

    def test_rejects_unknown_compute_units_before_loading_models(self):
        result = self.run_cli("--ane-compute-units", "INVALID")
        self.assertEqual(result.returncode, 1)
        self.assertIn("--ane-compute-units requires CPU_ONLY, CPU_AND_GPU, CPU_AND_NE or ALL", result.stderr)
        self.assertNotIn("[MM3-ANE] Loading", result.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, required=True)
    args = parser.parse_args()
    ComputeUnitCliTests.executable = args.executable.resolve(strict=True)
    unittest.main(argv=[sys.argv[0]])
