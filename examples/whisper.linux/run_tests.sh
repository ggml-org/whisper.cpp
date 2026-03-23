#!/bin/bash
# Run all whisper.linux tests.
# Usage: ./run_tests.sh [-v] [--debug] [pytest-options...]

cd "$(dirname "$0")"
python3 -m pytest tests/ -v "$@"
