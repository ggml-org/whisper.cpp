#!/usr/bin/env python3
"""Offline security regressions for benchmark dependency locks."""
from pathlib import Path
import re
import unittest


HERE = Path(__file__).resolve().parent
REQUIREMENTS_FILES = ('requirements-minimax.txt', 'requirements-music-alignment.txt')
# GHSA-6q48-6c7w-9gw2: remove trust in the compromised emSign certificate chain.
MINIMUM_CERTIFI_VERSION = (2026, 10, 19)


class BenchmarkRequirementsTests(unittest.TestCase):
    def test_certifi_pins_exclude_compromised_ca_bundle(self):
        for filename in REQUIREMENTS_FILES:
            with self.subTest(requirements=filename):
                lock = (HERE / filename).read_text()
                versions = re.findall(r'^certifi==([0-9.]+)\s', lock, re.MULTILINE)
                self.assertEqual(len(versions), 1)
                version = tuple(int(part) for part in versions[0].split('.'))
                self.assertGreaterEqual(version, MINIMUM_CERTIFI_VERSION)


if __name__ == '__main__':
    unittest.main()
