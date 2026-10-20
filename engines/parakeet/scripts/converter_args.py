import argparse
from pathlib import Path
from typing import Sequence


QUANT_CHOICES = ["f32", "f16", "bf16", "q8_0", "q5_0", "q4_0"]
HEAD_CHOICES = ["auto", "ctc", "rnnt", "tdt", "eou", "sortformer"]
DEFAULT_MODEL_STEM = "parakeet-ctc-0.6b"


def parse_args(
    description: str | None = None,
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=Path(f"models/{DEFAULT_MODEL_STEM}.nemo"),
        help="Path to .nemo archive (tarball). Downloads from HF if missing.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output GGUF path. Defaults to "
            f"models/{DEFAULT_MODEL_STEM}.<quant>.gguf."
        ),
    )
    parser.add_argument(
        "--quant",
        choices=QUANT_CHOICES,
        default="q8_0",
        help=(
            "Weight dtype for 2D projection matrices. Biases / norms / BN "
            "stay at f32. q8_0 default (~2x smaller than f16, bit-equal "
            "transcripts on clean speech across CTC/TDT/EOU/Sortformer); "
            "pass --quant f16 for the bit-equal floating-point baseline, or "
            "--quant bf16 for bfloat16 projection weights."
        ),
    )
    parser.add_argument(
        "--hf-repo",
        default="nvidia/parakeet-ctc-0.6b",
        help="HF model id to download from if --ckpt is missing.",
    )
    parser.add_argument(
        "--description",
        default=None,
        help=(
            "Value for general.description. Defaults to the model family's "
            "own text, e.g. 'NVIDIA Parakeet-TDT FastConformer ASR "
            "(CC-BY-4.0)'. Published derivatives such as TranscribePsy set "
            "their own so the GGUF can be reproduced byte for byte."
        ),
    )
    parser.add_argument(
        "--long-form-window-sec",
        type=float,
        default=None,
        help=(
            "Long-form window the model recommends, in seconds, written as "
            "parakeet.long_form.window_frames; the engine takes it when the "
            "caller leaves the window automatic (TranscribePsy: 60)."
        ),
    )
    parser.add_argument(
        "--long-form-overlap",
        type=float,
        default=None,
        help=(
            "Fraction of the recommended window shared with its neighbours, "
            "written as parakeet.long_form.context_frames (TranscribePsy: 0.25)."
        ),
    )
    parser.add_argument(
        "--head",
        choices=HEAD_CHOICES,
        default="auto",
        help=(
            "Override the auto-detected head. Use 'rnnt' to export the "
            "Transducer branch of a hybrid RNNT+CTC checkpoint instead of "
            "its default CTC branch."
        ),
    )
    args = parser.parse_args(argv)
    if args.out is None:
        args.out = Path(f"models/{DEFAULT_MODEL_STEM}.{args.quant}.gguf")
    return args
