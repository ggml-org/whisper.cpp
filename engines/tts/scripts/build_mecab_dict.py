#!/usr/bin/env python3
"""Materialize the MeCab IPAdic dictionary for tts-cpp.

Copies the prebuilt IPAdic files distributed by the `ipadic` PyPI package
(https://pypi.org/project/ipadic/) into the requested output directory. The
tts-cpp MeCabTagger expects the directory to contain sys.dic, unk.dic,
matrix.bin, char.bin, dicrc and mecabrc.

The `ipadic` package must be installed first:
    pip install ipadic

Usage:
    python build_mecab_dict.py /tmp/mecab-ipadic
"""

import argparse
import shutil
import sys
from pathlib import Path


def import_ipadic():
    try:
        import ipadic  # noqa: F401
    except ImportError as err:
        raise SystemExit(
            "The 'ipadic' Python package is required.\n"
            "Install it with: pip install ipadic"
        ) from err
    return ipadic


def reset_output_directory(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_dictionary(source_dir, output_dir):
    for entry in source_dir.iterdir():
        if entry.is_file():
            shutil.copyfile(entry, output_dir / entry.name)


def print_summary(output_dir):
    print(f"Dictionary materialized at {output_dir}:")
    for entry in sorted(output_dir.iterdir()):
        if entry.is_file():
            print(f"  {entry.name:<16} {entry.stat().st_size:>12} bytes")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Copy the prebuilt IPAdic from the 'ipadic' pip package "
                    "into the requested directory.",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output directory for the dictionary files",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output.resolve()
    ipadic = import_ipadic()
    source_dir = Path(ipadic.DICDIR)
    reset_output_directory(output_dir)
    copy_dictionary(source_dir, output_dir)
    print_summary(output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
