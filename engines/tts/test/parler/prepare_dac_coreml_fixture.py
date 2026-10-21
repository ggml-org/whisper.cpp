#!/usr/bin/env python3
"""Write a tiny, complete Parler GGUF for the Core ML DAC sidecar tests.

The hparams and tensor set mirror test_fit_params.cpp's tiny_parler (1 T5
layer, 2 decoder layers, a 2-block DAC with hop 4), so the engine loader
accepts it, but the DAC weights are random (fan-in scaled, nonzero biases,
Snake alphas in the checkpoints' range) so a sidecar decode is a meaningful
comparison against ggml. Export the sidecar from it with
scripts/parler/export-dac-coreml.py, e.g. --window 128 --fold-rows 64 so the
tiny decoder still exercises the folded path.

  python test/parler/prepare_dac_coreml_fixture.py --out parler-coreml-fixture
"""
import argparse
from pathlib import Path

import gguf
import numpy as np

T5 = dict(n_layer=1, d_model=32, d_ff=48, n_head=2, d_kv=8, rel_buckets=8, rel_max_dist=16, vocab=64)
DEC = dict(n_layer=2, d_model=32, n_head=4, d_ff=48, n_codebooks=3, vocab=44, max_position=96)
DAC = dict(sample_rate=44100, n_q=3, codebook_size=24, latent=8, decoder_dim=16, hop=4, cb_dim=4)
RATES = [2, 2]
STEM = "parler-tiny-f32"


def write_fixture(directory, seed=0, random_decoder=False):
    rng = np.random.default_rng(seed)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{STEM}.gguf"
    w = gguf.GGUFWriter(str(path), "parler")
    w.add_string("parler.arch", "parler")
    for k, v in T5.items():
        w.add_uint32("parler.t5." + ("vocab_size" if k == "vocab" else k), v)
    w.add_float32("parler.t5.rms_eps", 1e-6)
    for k in ("n_layer", "d_model", "n_head", "d_ff", "n_codebooks", "max_position"):
        w.add_uint32("parler.dec." + k, DEC[k])
    w.add_uint32("parler.dec.vocab_size", DEC["vocab"])
    w.add_float32("parler.dec.ln_eps", 1e-5)
    for k, v in (("bos_token_id", 41), ("eos_token_id", 40), ("pad_token_id", 40),
                 ("decoder_start_token_id", 41)):
        w.add_uint32("parler.dec." + k, v)
    w.add_bool("parler.enc_to_dec", False)
    w.add_uint32("parler.gen.max_length", 24)
    w.add_uint32("parler.gen.min_new_tokens", 2)
    w.add_bool("parler.gen.do_sample", True)
    w.add_float32("parler.gen.temperature", 1.0)
    w.add_uint32("parler.gen.top_k", 10)
    for k in ("sample_rate", "codebook_size", "hop"):
        w.add_uint32("parler.dac." + k, DAC[k])
    w.add_uint32("parler.dac.n_quantizers", DAC["n_q"])
    w.add_uint32("parler.dac.latent_dim", DAC["latent"])
    w.add_uint32("parler.dac.decoder_dim", DAC["decoder_dim"])
    w.add_array("parler.dac.rates", list(RATES))
    w.add_array("tokenizer.ggml.tokens", ["<unk>", "</s>", "▁a", "▁b", "a", "b", "c", "d"])
    w.add_array("tokenizer.ggml.scores", [0.0, 0.0, -1.0, -1.0, -2.0, -2.0, -2.0, -2.0])
    w.add_uint32("tokenizer.ggml.unknown_token_id", 0)
    w.add_uint32("tokenizer.ggml.eos_token_id", 1)

    def add(name, *ne, values=None):
        # ggml ne order in, numpy (reversed) order to gguf-py
        shape = tuple(reversed(ne))
        data = values if values is not None else np.full(shape, 0.01, dtype=np.float32)
        if random_decoder and name.startswith("dec.") and values is None:
            if "norm.weight" in name:
                data = np.ones(shape, dtype=np.float32)
            elif "norm.bias" in name:
                data = rng.standard_normal(shape) * 0.01
            else:
                data = rng.standard_normal(shape) / np.sqrt(ne[0])
        w.add_tensor(name, np.ascontiguousarray(data.reshape(shape), dtype=np.float32))

    def kernel(taps, inp, out):
        return rng.standard_normal((out, inp, taps)) / np.sqrt(inp * taps)

    def bias(n):
        return rng.standard_normal(n) * 0.1

    def alpha(c):
        return rng.uniform(0.15, 2.3, size=c)

    inner = T5["n_head"] * T5["d_kv"]
    add("t5.embed_tokens.weight", T5["d_model"], T5["vocab"])
    add("t5.output_norm.weight", T5["d_model"])
    for i in range(T5["n_layer"]):
        pre = f"t5.blk.{i}."
        if i == 0:
            add(pre + "attn_rel_b.weight", T5["n_head"], T5["rel_buckets"])
        add(pre + "attn_norm.weight", T5["d_model"])
        for n in ("attn_q", "attn_k", "attn_v"):
            add(pre + n + ".weight", T5["d_model"], inner)
        add(pre + "attn_o.weight", inner, T5["d_model"])
        add(pre + "ffn_norm.weight", T5["d_model"])
        add(pre + "ffn_gate.weight", T5["d_model"], T5["d_ff"])
        add(pre + "ffn_up.weight", T5["d_model"], T5["d_ff"])
        add(pre + "ffn_down.weight", T5["d_ff"], T5["d_model"])

    d = DEC["d_model"]
    add("dec.embed_prompts.weight", d, 8)
    add("dec.embed_positions.weight", d, DEC["max_position"])
    for k in range(DEC["n_codebooks"]):
        add(f"dec.embed_tokens.{k}.weight", d, DEC["vocab"])
        add(f"dec.lm_heads.{k}.weight", d, DEC["vocab"])
    add("dec.output_norm.weight", d)
    add("dec.output_norm.bias", d)
    for i in range(DEC["n_layer"]):
        pre = f"dec.blk.{i}."
        for n in ("attn_norm", "cross_norm", "ffn_norm"):
            add(pre + n + ".weight", d)
            add(pre + n + ".bias", d)
        for n in ("attn_q", "attn_k", "attn_v", "attn_o", "cross_q", "cross_k", "cross_v", "cross_o"):
            add(pre + n + ".weight", d, d)
        add(pre + "ffn_up.weight", d, DEC["d_ff"])
        add(pre + "ffn_down.weight", DEC["d_ff"], d)

    for k in range(DAC["n_q"]):
        pre = f"dac.quant.{k}."
        add(pre + "codebook.weight", DAC["cb_dim"], DAC["codebook_size"],
            values=rng.standard_normal((DAC["codebook_size"], DAC["cb_dim"])))
        add(pre + "out_proj.weight", 1, DAC["cb_dim"], DAC["latent"], values=kernel(1, DAC["cb_dim"], DAC["latent"]))
        add(pre + "out_proj.bias", DAC["latent"], values=bias(DAC["latent"]))
    c = DAC["decoder_dim"]
    add("dac.dec.conv_in.weight", 7, DAC["latent"], c, values=kernel(7, DAC["latent"], c))
    add("dac.dec.conv_in.bias", c, values=bias(c))
    for i, s in enumerate(RATES):
        pre = f"dac.dec.blk.{i}."
        out = c // 2
        add(pre + "snake.alpha", 1, c, 1, values=alpha(c))
        # ConvTranspose1d [in, out, 2s] in numpy order
        add(pre + "convt.weight", 2 * s, out, c,
            values=rng.standard_normal((c, out, 2 * s)) / np.sqrt(c * 2))
        add(pre + "convt.bias", out, values=bias(out))
        for j in range(3):
            rp = f"{pre}res.{j}."
            add(rp + "snake1.alpha", 1, out, 1, values=alpha(out))
            add(rp + "conv1.weight", 7, out, out, values=kernel(7, out, out))
            add(rp + "conv1.bias", out, values=bias(out))
            add(rp + "snake2.alpha", 1, out, 1, values=alpha(out))
            add(rp + "conv2.weight", 1, out, out, values=kernel(1, out, out))
            add(rp + "conv2.bias", out, values=bias(out))
        c = out
    add("dac.dec.snake_out.alpha", 1, c, 1, values=alpha(c))
    add("dac.dec.conv_out.weight", 7, c, 1, values=kernel(7, c, 1))
    add("dac.dec.conv_out.bias", 1, values=bias(1))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--random-decoder", action="store_true",
                        help="use nondegenerate decoder weights for state/replay parity checks")
    args = parser.parse_args()
    print(write_fixture(args.out, random_decoder=args.random_decoder))
