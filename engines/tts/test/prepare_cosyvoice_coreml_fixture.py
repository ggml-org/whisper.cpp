#!/usr/bin/env python3
"""Write a tiny CosyVoice LM GGUF for native generation-routing tests."""

import argparse
from pathlib import Path

import gguf
import numpy as np

from test_export_cosyvoice_coreml import DIM, HEAD_DIM, qwen_weights, tensor


def write_fixture(directory):
    directory.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(directory / "cosyvoice3-llm-f32.gguf"), "cosyvoice3")
    metadata = {"depth": 1, "hidden": DIM, "n_head": 2, "n_kv": 1,
                "head_dim": HEAD_DIM, "inter": DIM * 2,
                "sos": 9, "task_id": 10, "speech_token_size": 9}
    for name, value in metadata.items():
        writer.add_uint32("cosyvoice3.llm." + name, value)
    weights = qwen_weights()
    weights["lm/embed_tokens/weight"] = tensor(12, DIM)
    weights["lm/speech_embedding/weight"] = tensor(11, DIM)
    for name, weight in weights.items():
        writer.add_tensor(name, np.ascontiguousarray(weight))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    write_fixture(parser.parse_args().out)
