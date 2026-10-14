"""Convert NVIDIA's Nemotron 3 Diarization NeMo checkpoint with NeMo-Speech.cpp."""

import argparse
import subprocess
import sys
from pathlib import Path

from download_nemotron_diarization import (
    DEFAULT_MODEL_DIRECTORY,
    NEMO_FILENAME,
    download_artifact,
)


CONVERTER_REPOSITORY = "https://github.com/NVIDIA/NeMo-Speech.cpp.git"
CONVERTER_REVISION = "97a15afa5caa9bce5baaa86c1184103877af4101"
CONVERTER_DIRECTORY = DEFAULT_MODEL_DIRECTORY / "sources" / "NeMo-Speech.cpp"
OUTPUT_FILENAME = "Nemotron-3-Diarization.converted.q8_0.gguf"
OUTPUT_TYPE = "q8_0"
OUTPUT_TYPES = ("f32", "f16", "bf16", "q8_0", "q6_k", "q5_k", "q4_k", "mxfp4", "nvfp4")


def ensure_converter(converter_directory, run=subprocess.run):
    converter = converter_directory / "convert_model.py"
    converter_directory.parent.mkdir(parents=True, exist_ok=True)
    if converter_directory.exists() and not (converter_directory / ".git").exists():
        raise RuntimeError(f"Converter directory is not a Git checkout: {converter_directory}")
    if converter.is_file() and converter_revision(converter_directory, run) == CONVERTER_REVISION:
        verify_converter_file(converter_directory, run)
        return converter
    if not (converter_directory / ".git").exists():
        run(["git", "clone", "--filter=blob:none", "--no-checkout",
             CONVERTER_REPOSITORY, str(converter_directory)], check=True)
    run(["git", "-C", str(converter_directory), "fetch", "origin",
         CONVERTER_REVISION], check=True)
    run(["git", "-C", str(converter_directory), "checkout",
         "--detach", CONVERTER_REVISION], check=True)
    if converter_revision(converter_directory, run) != CONVERTER_REVISION:
        raise RuntimeError("Converter checkout does not match the pinned revision")
    if not converter.is_file():
        raise FileNotFoundError(converter)
    verify_converter_file(converter_directory, run)
    return converter


def converter_revision(converter_directory, run):
    result = run(["git", "-C", str(converter_directory), "rev-parse", "HEAD"],
                 check=True, capture_output=True, text=True)
    return result.stdout.strip()


def verify_converter_file(converter_directory, run):
    run(["git", "-C", str(converter_directory), "ls-files", "--error-unmatch",
         "--", "convert_model.py"], check=True, capture_output=True, text=True)
    result = run(["git", "-C", str(converter_directory), "status", "--porcelain",
                  "--", "convert_model.py"], check=True, capture_output=True, text=True)
    if result.stdout.strip():
        raise RuntimeError("Converter script differs from the pinned revision")


def convert_model(source, output, converter, run=subprocess.run, outtype=OUTPUT_TYPE):
    if outtype not in OUTPUT_TYPES:
        raise ValueError(f"Unsupported output type: {outtype}")
    output.parent.mkdir(parents=True, exist_ok=True)
    run([sys.executable, str(converter), str(source.resolve()),
         "--outfile", str(output.resolve()), "--outtype", outtype],
        check=True, cwd=converter.parent)


def output_filename(outtype):
    return f"Nemotron-3-Diarization.converted.{outtype}.gguf"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIRECTORY)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--outtype", choices=OUTPUT_TYPES, default=OUTPUT_TYPE)
    parser.add_argument("--converter-dir", type=Path, default=CONVERTER_DIRECTORY)
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    if args.source is None:
        from huggingface_hub import hf_hub_download

        source = download_artifact(NEMO_FILENAME, args.model_dir, hf_hub_download)
    else:
        source = args.source
    output = args.output or args.model_dir / output_filename(args.outtype)
    converter = ensure_converter(args.converter_dir)
    if args.prepare_only:
        print(source)
        print(converter.parent / "requirements.txt")
        return
    convert_model(source, output, converter, outtype=args.outtype)
    print(output)


if __name__ == "__main__":
    main()
