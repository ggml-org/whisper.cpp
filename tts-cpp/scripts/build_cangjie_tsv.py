#!/usr/bin/env python3
"""Download and prepare the Cangjie5 TSV for tts-cpp.

Fetches the Cangjie5_TC.txt table from the Jackchows/Cangjie5 GitHub
repository, strips the optional annotation column (third field), and
writes a clean two-column TSV (character<TAB>code) that
CangjieTable::load() expects.

Usage:
    python build_cangjie_tsv.py /tmp/cangjie5.tsv
    python build_cangjie_tsv.py /tmp/cangjie5.tsv --tag v1.0.0
"""

import argparse
import sys
import urllib.request
from pathlib import Path

DEFAULT_TAG = "master"
RAW_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/Jackchows/Cangjie5/{tag}/Cangjie5_TC.txt"
)


def build_download_url(tag):
    return RAW_URL_TEMPLATE.format(tag=tag)


def download_raw_table(url):
    with urllib.request.urlopen(url) as resp:
        return resp.read().decode("utf-8")


def strip_annotation(line):
    fields = line.split("\t")
    if len(fields) < 2:
        return None
    return fields[0] + "\t" + fields[1]


def convert_to_clean_tsv(raw_text):
    lines = []
    for line in raw_text.splitlines():
        cleaned = strip_annotation(line)
        if cleaned is not None:
            lines.append(cleaned)
    return lines


def write_tsv(lines, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def print_summary(output_path, line_count):
    size = output_path.stat().st_size
    print(f"Cangjie5 TSV written to {output_path}")
    print(f"  entries: {line_count}")
    print(f"  size:    {size} bytes")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download the Cangjie5 table from Jackchows/Cangjie5 and "
                    "produce a clean two-column TSV for tts-cpp.",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output path for the TSV file",
    )
    parser.add_argument(
        "--tag",
        default=DEFAULT_TAG,
        help="Git tag or branch to fetch from (default: master)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = args.output.resolve()
    url = build_download_url(args.tag)
    print(f"Downloading from {url} ...")
    raw_text = download_raw_table(url)
    lines = convert_to_clean_tsv(raw_text)
    write_tsv(lines, output_path)
    print_summary(output_path, len(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
