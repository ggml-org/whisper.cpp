#!/usr/bin/env python3
"""Convert a Parler-TTS checkpoint (mini-v1 / large-v1 / indic) to a tts-cpp GGUF.

Contains: T5 text encoder, Parler decoder LM, DAC decoder (decode path only),
T5 unigram tokenizer (tokens + scores). DAC/quantizer weight-norm (weight_g /
weight_v) is folded into plain weights at convert time. The DAC encoder and the
quantizer in_proj tensors are intentionally dropped (encode path unused).

Indic-class checkpoints (e.g. ai4bharat/indic-parler-tts) differ in three
storage details, all handled here: the repo tokenizer is a SentencePiece-BPE
PROMPT tokenizer (descriptions keep the text encoder's T5 unigram tokenizer,
fetched from its repo), the nine LM heads are stored fused as one matrix, and
the DAC uses transformers-DacModel tensor names with weight-norm pre-folded.

Two-sided completeness: every source tensor must be either consumed or on the
explicit ignore list, and every expected destination must be produced —
otherwise the converter aborts.

Usage:
  python3 convert-to-gguf.py --model-id parler-tts/parler-tts-mini-v1 \
      --dtype f32 --out tts-cpp/models/parler-mini-v1-f32.gguf
"""

import argparse
import json
import os
import sys

import numpy as np


def log(msg):
    print(f"[convert-parler] {msg}", flush=True)


def fold_weight_norm(g, v):
    """weight_norm dim=0: w[i] = v[i] * g[i] / ||v[i]||_2  (norm over all dims but 0)."""
    v = np.asarray(v, dtype=np.float32)
    g = np.asarray(g, dtype=np.float32).reshape(v.shape[0], *([1] * (v.ndim - 1)))
    norm = np.sqrt(np.sum(v.astype(np.float64) ** 2, axis=tuple(range(1, v.ndim)), keepdims=True))
    return (v * (g / norm.astype(np.float32))).astype(np.float32)


def parse_unigram(tj):
    """tokenizer.json (fast Unigram) -> (tokens, scores, charsmap, unk_id)."""
    assert tj["model"]["type"] == "Unigram", tj["model"]["type"]
    norm = tj.get("normalizer") or {}
    norms = norm.get("normalizers", [norm]) if norm else []
    charsmap = b""
    for nn in norms:
        if nn.get("type") == "Precompiled":
            import base64
            charsmap = base64.b64decode(nn["precompiled_charsmap"])
    vocab = tj["model"]["vocab"]          # [[piece, score], ...]
    unk_id = tj["model"]["unk_id"]
    tokens = [p for p, _ in vocab]
    scores = [float(s) for _, s in vocab]
    for at in tj.get("added_tokens", []):
        if at["id"] >= len(tokens):
            tokens.extend([""] * (at["id"] + 1 - len(tokens)))
            scores.extend([-1e9] * (at["id"] + 1 - len(scores)))
        tokens[at["id"]] = at["content"]
    return tokens, scores, charsmap, unk_id


def parse_bpe(tj):
    """tokenizer.json (fast BPE, llama-style SentencePiece flavour) ->
    dict(tokens, merges, unk_id, bos_id, add_bos). Asserts pin the exact
    configuration the C++ prompt tokenizer implements."""
    m = tj["model"]
    assert m.get("byte_fallback") is True, "BPE without byte_fallback unsupported"
    assert not m.get("continuing_subword_prefix"), "GPT2-style BPE unsupported"
    assert not m.get("ignore_merges"), "ignore_merges unsupported"
    assert tj.get("normalizer") is None, tj.get("normalizer")
    pre = tj.get("pre_tokenizer") or {}
    assert (pre.get("type") == "Metaspace" and pre.get("split") is False
            and pre.get("prepend_scheme") == "first"), pre

    vocab = m["vocab"]                    # {piece: id}
    tokens = [None] * len(vocab)
    for piece, idx in vocab.items():
        assert 0 <= idx < len(tokens) and tokens[idx] is None, \
            f"non-contiguous BPE vocab id {idx}"
        tokens[idx] = piece
    for at in tj.get("added_tokens", []):
        assert at["id"] < len(tokens), f"added token beyond vocab: {at}"
        tokens[at["id"]] = at["content"]

    merges = []
    for mg in m["merges"]:                # "l r" strings or [l, r] pairs
        pair = mg.split(" ") if isinstance(mg, str) else list(mg)
        assert len(pair) == 2 and all(" " not in p for p in pair), mg
        merges.append((pair[0], pair[1]))

    post = tj.get("post_processor") or {}
    assert post.get("type") == "TemplateProcessing", post.get("type")
    single = post["single"]
    seq_at = [i for i, e in enumerate(single) if "Sequence" in e]
    assert seq_at == [len(single) - 1], f"unsupported template: {single}"
    specials = [e["SpecialToken"]["id"] for e in single[:-1]]
    assert len(specials) <= 1, f"unsupported template: {single}"
    add_bos = bool(specials)
    bos_id = post["special_tokens"][specials[0]]["ids"][0] if add_bos else 0
    return dict(tokens=tokens, merges=merges, unk_id=vocab[m["unk_token"]],
                bos_id=bos_id, add_bos=add_bos)


def find_ggml_lib():
    """Locate a built ggml-base shared lib (for k-quant encoding)."""
    import glob
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    for pat in ("build*/ggml/src/libqvac-speech-ggml-base.*",
                "build*/ggml/src/libggml-base.*"):
        hits = [h for h in sorted(glob.glob(os.path.join(root, pat)))
                if h.endswith((".dylib", ".so"))
                and "san" not in os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(h))))]
        if hits:
            return hits[0]
    return None


def make_ggml_quantizer(lib_path):
    """ctypes binding to ggml_quantize_chunk for types gguf-py cannot encode
    (Q4_K/Q6_K are dequantize-only in Python) — upstream gguf-py tests use the
    same route, and it is the exact encoder inference runs against."""
    import ctypes
    import gguf
    lib = ctypes.CDLL(lib_path)
    fn = lib.ggml_quantize_chunk
    fn.restype = ctypes.c_size_t
    fn.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
                   ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]

    def quantize(arr, qtype, imatrix=None):
        blk, typesz = gguf.GGML_QUANT_SIZES[qtype]
        nrows = int(np.prod(arr.shape[:-1]))
        n_per_row = arr.shape[-1]
        assert n_per_row % blk == 0, (arr.shape, qtype)
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        dst = np.zeros(nrows * (n_per_row // blk) * typesz, dtype=np.uint8)
        im_ptr = None
        if imatrix is not None:
            # ggml quant_weights: one float per column, shared by all rows
            im = np.ascontiguousarray(imatrix, dtype=np.float32)
            assert im.size == n_per_row, (im.size, n_per_row)
            im_ptr = im.ctypes.data_as(ctypes.c_void_p)
        n = fn(int(qtype.value), arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
               dst.ctypes.data_as(ctypes.c_void_p), 0, nrows, n_per_row, im_ptr)
        assert n == dst.nbytes, f"ggml_quantize_chunk wrote {n} != {dst.nbytes}"
        return dst.reshape(*arr.shape[:-1], (n_per_row // blk) * typesz)

    return quantize


class Converter:
    def __init__(self, get, src_names):
        self.get = get                      # name -> np.ndarray
        self.src_names = set(src_names)
        self.consumed = set()
        self.out = {}                       # dest name -> np.ndarray (f32)
        self.src_of_dest = {}               # dest name -> HF source name

    def take(self, src):
        if src not in self.src_names:
            raise KeyError(f"source tensor missing: {src}")
        self.consumed.add(src)
        return np.asarray(self.get(src), dtype=np.float32)

    def has(self, src):
        return src in self.src_names

    def emit(self, dest, arr):
        assert dest not in self.out, f"duplicate dest {dest}"
        self.out[dest] = np.ascontiguousarray(arr, dtype=np.float32)

    def move(self, dest, src):
        self.emit(dest, self.take(src))
        self.src_of_dest[dest] = src

    def move_folded(self, dest, src_prefix):
        g = self.take(src_prefix + ".weight_g")
        v = self.take(src_prefix + ".weight_v")
        self.emit(dest + ".weight", fold_weight_norm(g, v))
        self.move(dest + ".bias", src_prefix + ".bias")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="parler-tts/parler-tts-mini-v1",
                    help="HF repo id or a local snapshot directory")
    ap.add_argument("--reference-repo", default=None,
                    help="provenance string when --model-id is a local directory")
    ap.add_argument("--dtype", choices=["f32", "f16", "q8_0", "q6_k"], default="f32")
    ap.add_argument("--ggml-lib", default=None,
                    help="ggml-base shared lib for ggml-side encoding (k-quants / --imatrix; "
                         "auto-detected from build dirs)")
    ap.add_argument("--imatrix", default=None,
                    help="npz of per-column importance vectors keyed by HF weight name "
                         "(scripts/parler/compute-imatrix.py); used by k-quants and q4_0/q5_0")
    ap.add_argument("--recipe", default=None,
                    help="override quant tiers, e.g. bulk=Q5_K,tables=F16,heads=Q8_0,t5=Q8_0 "
                         "(requires a quant --dtype; unlisted tiers keep the dtype's defaults)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    import gguf

    snap = args.model_id if os.path.isdir(args.model_id) else snapshot_download(args.model_id)
    log(f"snapshot: {snap}")

    with open(os.path.join(snap, "config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(snap, "generation_config.json")) as f:
        gen_cfg = json.load(f)
    with open(os.path.join(snap, "tokenizer.json")) as f:
        tok_json = json.load(f)

    t5 = cfg["text_encoder"]
    dec = cfg["decoder"]
    assert cfg["model_type"] == "parler_tts"
    assert dec.get("rope_embeddings", False) is False, "v1 models use sinusoidal positions"
    assert cfg.get("prompt_cross_attention", False) is False, "v1 models prepend the prompt"

    # ---- gather source tensors (single or sharded safetensors) ----
    files = [f for f in os.listdir(snap) if f.endswith(".safetensors")]
    handles = [safe_open(os.path.join(snap, f), framework="np") for f in sorted(files)]
    src_of = {}
    for h in handles:
        for n in h.keys():
            src_of[n] = h
    log(f"{len(src_of)} source tensors across {len(files)} file(s)")

    cv = Converter(lambda n: src_of[n].get_tensor(n), src_of.keys())

    # ---------------- T5 encoder ----------------
    n_t5 = t5["num_layers"]
    cv.move("t5.embed_tokens.weight", "text_encoder.shared.weight")
    for i in range(n_t5):
        b = f"text_encoder.encoder.block.{i}.layer"
        d = f"t5.blk.{i}"
        cv.move(f"{d}.attn_norm.weight", f"{b}.0.layer_norm.weight")
        for s, t in (("q", "attn_q"), ("k", "attn_k"), ("v", "attn_v"), ("o", "attn_o")):
            cv.move(f"{d}.{t}.weight", f"{b}.0.SelfAttention.{s}.weight")
        cv.move(f"{d}.ffn_norm.weight", f"{b}.1.layer_norm.weight")
        cv.move(f"{d}.ffn_gate.weight", f"{b}.1.DenseReluDense.wi_0.weight")
        cv.move(f"{d}.ffn_up.weight", f"{b}.1.DenseReluDense.wi_1.weight")
        cv.move(f"{d}.ffn_down.weight", f"{b}.1.DenseReluDense.wo.weight")
    cv.move("t5.blk.0.attn_rel_b.weight",
            "text_encoder.encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight")
    cv.move("t5.output_norm.weight", "text_encoder.encoder.final_layer_norm.weight")

    # ---------------- glue ----------------
    # enc_to_dec_proj exists (and is applied) only when t5.d_model != dec.hidden_size
    enc_to_dec = cv.has("enc_to_dec_proj.weight")
    expect_proj = t5["d_model"] != dec["hidden_size"]
    assert enc_to_dec == expect_proj, \
        f"enc_to_dec_proj presence ({enc_to_dec}) contradicts dims (t5 {t5['d_model']} vs dec {dec['hidden_size']})"
    if enc_to_dec:
        cv.move("enc_to_dec.weight", "enc_to_dec_proj.weight")
        cv.move("enc_to_dec.bias", "enc_to_dec_proj.bias")
    cv.move("dec.embed_prompts.weight", "embed_prompts.weight")

    # ---------------- decoder LM ----------------
    n_dec = dec["num_hidden_layers"]
    n_cb = dec["num_codebooks"]
    cv.move("dec.embed_positions.weight", "decoder.model.decoder.embed_positions.weights")
    for k in range(n_cb):
        cv.move(f"dec.embed_tokens.{k}.weight", f"decoder.model.decoder.embed_tokens.{k}.weight")
    if cv.has("decoder.lm_heads.weight"):
        # use_fused_lm_heads checkpoints stack the heads codebook-major:
        # logits.view(..., num_codebooks, vocab) => rows k*V:(k+1)*V = head k
        fused = cv.take("decoder.lm_heads.weight")
        v = dec["vocab_size"]
        assert fused.shape[0] == n_cb * v, (fused.shape, n_cb, v)
        for k in range(n_cb):
            cv.emit(f"dec.lm_heads.{k}.weight", fused[k * v:(k + 1) * v])
    else:
        for k in range(n_cb):
            cv.move(f"dec.lm_heads.{k}.weight", f"decoder.lm_heads.{k}.weight")
    for i in range(n_dec):
        b = f"decoder.model.decoder.layers.{i}"
        d = f"dec.blk.{i}"
        cv.move(f"{d}.attn_norm.weight", f"{b}.self_attn_layer_norm.weight")
        cv.move(f"{d}.attn_norm.bias", f"{b}.self_attn_layer_norm.bias")
        for s, t in (("q_proj", "attn_q"), ("k_proj", "attn_k"), ("v_proj", "attn_v"), ("out_proj", "attn_o")):
            cv.move(f"{d}.{t}.weight", f"{b}.self_attn.{s}.weight")
        cv.move(f"{d}.cross_norm.weight", f"{b}.encoder_attn_layer_norm.weight")
        cv.move(f"{d}.cross_norm.bias", f"{b}.encoder_attn_layer_norm.bias")
        for s, t in (("q_proj", "cross_q"), ("k_proj", "cross_k"), ("v_proj", "cross_v"), ("out_proj", "cross_o")):
            cv.move(f"{d}.{t}.weight", f"{b}.encoder_attn.{s}.weight")
        cv.move(f"{d}.ffn_norm.weight", f"{b}.final_layer_norm.weight")
        cv.move(f"{d}.ffn_norm.bias", f"{b}.final_layer_norm.bias")
        cv.move(f"{d}.ffn_up.weight", f"{b}.fc1.weight")
        cv.move(f"{d}.ffn_down.weight", f"{b}.fc2.weight")
    cv.move("dec.output_norm.weight", "decoder.model.decoder.layer_norm.weight")
    cv.move("dec.output_norm.bias", "decoder.model.decoder.layer_norm.bias")

    # ---------------- DAC (decode path) ----------------
    ae = cfg["audio_encoder"]
    n_q = ae.get("num_codebooks", ae.get("n_codebooks"))
    if cv.has("audio_encoder.decoder.block.0.res_unit1.conv1.weight"):
        # transformers-DacModel naming; weight-norm is already folded upstream
        for k in range(n_q):
            q = f"audio_encoder.quantizer.quantizers.{k}"
            cv.move(f"dac.quant.{k}.codebook.weight", f"{q}.codebook.weight")
            cv.move(f"dac.quant.{k}.out_proj.weight", f"{q}.out_proj.weight")
            cv.move(f"dac.quant.{k}.out_proj.bias", f"{q}.out_proj.bias")

        dacm = "audio_encoder.decoder"
        cv.move("dac.dec.conv_in.weight", f"{dacm}.conv1.weight")
        cv.move("dac.dec.conv_in.bias", f"{dacm}.conv1.bias")
        rates = []
        n_blocks = 0
        while cv.has(f"{dacm}.block.{n_blocks}.conv_t1.weight"):
            n_blocks += 1
        for i in range(n_blocks):
            blk = f"{dacm}.block.{i}"
            d = f"dac.dec.blk.{i}"
            cv.move(f"{d}.snake.alpha", f"{blk}.snake1.alpha")
            cv.move(f"{d}.convt.weight", f"{blk}.conv_t1.weight")
            cv.move(f"{d}.convt.bias", f"{blk}.conv_t1.bias")
            k = cv.out[f"{d}.convt.weight"].shape[2]
            assert k % 2 == 0, f"convT kernel {k} not 2*stride"
            rates.append(k // 2)
            for j in range(3):
                ru = f"{blk}.res_unit{1 + j}"
                cv.move(f"{d}.res.{j}.snake1.alpha", f"{ru}.snake1.alpha")
                cv.move(f"{d}.res.{j}.conv1.weight", f"{ru}.conv1.weight")
                cv.move(f"{d}.res.{j}.conv1.bias", f"{ru}.conv1.bias")
                cv.move(f"{d}.res.{j}.snake2.alpha", f"{ru}.snake2.alpha")
                cv.move(f"{d}.res.{j}.conv2.weight", f"{ru}.conv2.weight")
                cv.move(f"{d}.res.{j}.conv2.bias", f"{ru}.conv2.bias")
        cv.move("dac.dec.snake_out.alpha", f"{dacm}.snake1.alpha")
        cv.move("dac.dec.conv_out.weight", f"{dacm}.conv2.weight")
        cv.move("dac.dec.conv_out.bias", f"{dacm}.conv2.bias")
    else:
        for k in range(n_q):
            q = f"audio_encoder.model.quantizer.quantizers.{k}"
            cv.move(f"dac.quant.{k}.codebook.weight", f"{q}.codebook.weight")
            cv.move_folded(f"dac.quant.{k}.out_proj", f"{q}.out_proj")

        dacm = "audio_encoder.model.decoder.model"
        cv.move_folded("dac.dec.conv_in", f"{dacm}.0")
        rates = []
        n_blocks = 0
        while cv.has(f"{dacm}.{1 + n_blocks}.block.1.weight_v"):
            n_blocks += 1
        for i in range(n_blocks):
            blk = f"{dacm}.{1 + i}"
            d = f"dac.dec.blk.{i}"
            cv.move(f"{d}.snake.alpha", f"{blk}.block.0.alpha")
            cv.move_folded(f"{d}.convt", f"{blk}.block.1")
            k = cv.out[f"{d}.convt.weight"].shape[2]
            assert k % 2 == 0, f"convT kernel {k} not 2*stride"
            rates.append(k // 2)
            for j in range(3):
                ru = f"{blk}.block.{2 + j}.block"
                cv.move(f"{d}.res.{j}.snake1.alpha", f"{ru}.0.alpha")
                cv.move_folded(f"{d}.res.{j}.conv1", f"{ru}.1")
                cv.move(f"{d}.res.{j}.snake2.alpha", f"{ru}.2.alpha")
                cv.move_folded(f"{d}.res.{j}.conv2", f"{ru}.3")
        cv.move(f"dac.dec.snake_out.alpha", f"{dacm}.{1 + n_blocks}.alpha")
        cv.move_folded("dac.dec.conv_out", f"{dacm}.{2 + n_blocks}")

    conv_in_w = cv.out["dac.dec.conv_in.weight"]
    dac_latent, dac_decoder_dim = conv_in_w.shape[1], conv_in_w.shape[0]
    hop = int(np.prod(rates))
    log(f"DAC: latent={dac_latent} decoder_dim={dac_decoder_dim} rates={rates} hop={hop}")
    assert cv.out["dac.dec.conv_out.weight"].shape[0] == 1, "conv_out must emit mono"

    # ---------------- completeness check ----------------
    IGNORE_PREFIXES = (
        "audio_encoder.model.encoder.",          # encode path unused
        "audio_encoder.encoder.",                # same, transformers-DacModel naming
    )
    IGNORE_SUBSTR = (
        ".in_proj.",                             # quantizer encode-side proj
    )
    leftovers = [n for n in cv.src_names - cv.consumed
                 if not n.startswith(IGNORE_PREFIXES) and not any(s in n for s in IGNORE_SUBSTR)]
    if leftovers:
        for n in sorted(leftovers)[:40]:
            log(f"UNCONSUMED: {n}")
        raise SystemExit(f"{len(leftovers)} unconsumed source tensors — refusing to write GGUF")
    log(f"consumed {len(cv.consumed)}/{len(cv.src_names)} source tensors "
        f"({len(cv.src_names) - len(cv.consumed)} intentionally ignored), {len(cv.out)} dest tensors")

    # ---------------- tokenizer(s) ----------------
    # mini/large: the repo tokenizer is a T5 unigram shared by prompt and
    # description. Indic-class: the repo tokenizer is a SentencePiece-BPE
    # PROMPT tokenizer; descriptions use the text encoder's own T5 tokenizer.
    prompt_bpe = None
    if tok_json["model"]["type"] == "BPE":
        prompt_bpe = parse_bpe(tok_json)
        assert len(prompt_bpe["tokens"]) == cfg["vocab_size"], \
            (len(prompt_bpe["tokens"]), cfg["vocab_size"])
        from huggingface_hub import hf_hub_download
        desc_repo = t5["_name_or_path"]
        with open(hf_hub_download(desc_repo, "tokenizer.json")) as f:
            desc_tok_json = json.load(f)
        log(f"prompt tokenizer: BPE, {len(prompt_bpe['tokens'])} pieces, "
            f"{len(prompt_bpe['merges'])} merges; description tokenizer from {desc_repo}")
    else:
        desc_tok_json = tok_json
    tokens, scores, charsmap, unk_id = parse_unigram(desc_tok_json)
    log(f"tokenizer: {len(tokens)} pieces, unk_id={unk_id}")

    # ---------------- write GGUF ----------------
    w = gguf.GGUFWriter(args.out, "parler")
    w.add_string("parler.arch", "parler-tts")
    variant = args.model_id.rstrip("/").rsplit("/", 1)[-1]
    w.add_string("parler.variant", variant)
    w.add_string("parler.reference_repo", args.reference_repo or args.model_id)
    ftype = args.dtype + (f":{args.recipe}" if args.recipe else "") \
                       + (":imatrix" if args.imatrix else "")
    w.add_string("parler.ftype", ftype)

    w.add_uint32("parler.t5.n_layer", t5["num_layers"])
    w.add_uint32("parler.t5.d_model", t5["d_model"])
    w.add_uint32("parler.t5.d_ff", t5["d_ff"])
    w.add_uint32("parler.t5.n_head", t5["num_heads"])
    w.add_uint32("parler.t5.d_kv", t5["d_kv"])
    w.add_uint32("parler.t5.rel_buckets", t5["relative_attention_num_buckets"])
    w.add_uint32("parler.t5.rel_max_dist", t5["relative_attention_max_distance"])
    w.add_float32("parler.t5.rms_eps", t5["layer_norm_epsilon"])
    w.add_uint32("parler.t5.vocab_size", t5["vocab_size"])

    w.add_uint32("parler.dec.n_layer", dec["num_hidden_layers"])
    w.add_uint32("parler.dec.d_model", dec["hidden_size"])
    w.add_uint32("parler.dec.n_head", dec["num_attention_heads"])
    w.add_uint32("parler.dec.d_ff", dec["ffn_dim"])
    w.add_uint32("parler.dec.n_codebooks", dec["num_codebooks"])
    w.add_uint32("parler.dec.vocab_size", dec["vocab_size"])
    w.add_float32("parler.dec.ln_eps", 1e-5)   # nn.LayerNorm default, verified vs live model
    w.add_uint32("parler.dec.bos_token_id", gen_cfg["bos_token_id"])
    w.add_uint32("parler.dec.eos_token_id", gen_cfg["eos_token_id"])
    w.add_uint32("parler.dec.pad_token_id", gen_cfg["pad_token_id"])
    w.add_uint32("parler.dec.decoder_start_token_id", gen_cfg["decoder_start_token_id"])
    w.add_uint32("parler.dec.max_position", dec["max_position_embeddings"])
    w.add_bool("parler.enc_to_dec", enc_to_dec)

    w.add_uint32("parler.gen.max_length", gen_cfg["max_length"])
    # 0 = disabled. NOT defaulted: modeling_parler_tts.py has no code-side
    # min_new_tokens fallback; only mini-v1's generation_config.json sets it.
    w.add_uint32("parler.gen.min_new_tokens", gen_cfg.get("min_new_tokens", 0))
    w.add_bool("parler.gen.do_sample", gen_cfg.get("do_sample", False))
    w.add_float32("parler.gen.temperature", gen_cfg.get("temperature", 1.0))
    w.add_uint32("parler.gen.top_k", gen_cfg.get("top_k", 50))

    w.add_uint32("parler.dac.sample_rate", cfg["audio_encoder"]["sampling_rate"])
    w.add_uint32("parler.dac.n_quantizers", n_q)
    w.add_uint32("parler.dac.codebook_size", cfg["audio_encoder"]["codebook_size"])
    w.add_uint32("parler.dac.latent_dim", dac_latent)
    w.add_uint32("parler.dac.decoder_dim", dac_decoder_dim)
    w.add_uint32("parler.dac.hop", hop)
    w.add_array("parler.dac.rates", rates)

    w.add_string("tokenizer.ggml.model", "t5")
    w.add_token_list(tokens)
    w.add_token_scores(scores)
    if charsmap:
        w.add_precompiled_charsmap(charsmap)
        log(f"tokenizer: precompiled charsmap {len(charsmap)} bytes")
    w.add_uint32("tokenizer.ggml.unknown_token_id", unk_id)
    w.add_uint32("tokenizer.ggml.eos_token_id", 1)
    w.add_uint32("tokenizer.ggml.padding_token_id", 0)
    w.add_bool("parler.tokenizer.add_eos", True)  # T5 appends </s>; verified in reference dump

    if prompt_bpe:
        w.add_string("parler.prompt_tokenizer.model", "bpe")
        w.add_array("parler.prompt_tokenizer.tokens", prompt_bpe["tokens"])
        w.add_array("parler.prompt_tokenizer.merges",
                    [f"{l} {r}" for l, r in prompt_bpe["merges"]])
        w.add_uint32("parler.prompt_tokenizer.unknown_token_id", prompt_bpe["unk_id"])
        w.add_uint32("parler.prompt_tokenizer.bos_token_id", prompt_bpe["bos_id"])
        w.add_bool("parler.prompt_tokenizer.add_bos", prompt_bpe["add_bos"])

    # dtype policy. Numerically delicate tensors always stay f32: norms,
    # biases, alphas, rel-pos bias, positional table, all DAC tensors.
    def keep_f32(name):
        return (name.endswith(".bias") or ".alpha" in name or "norm" in name
                or name == "t5.blk.0.attn_rel_b.weight"
                or name == "dec.embed_positions.weight"
                or name.startswith("dac."))

    # Under f16 the ENTIRE T5 encoder additionally stays f32: flan-T5
    # activations overflow the f16 range, and ggml's f16 mul_mat converts
    # activation rows to f16 for the dot product -> encoder output NaN.
    # Quantized dots don't share that trap (activations are re-quantized
    # per 32/256-block with independent scales), so every recipe takes T5
    # weights to q8_0 — never F16. Decoder-side F16 tiers are safe (the
    # all-f16 GGUF passes strict parity). The 9 LM heads never go BELOW
    # q8_0: they feed the logits directly, and 6-bit heads derailed the
    # sampled trajectory on large-v1 (EOS never fires). Tier choices below
    # follow a 200-step teacher-forced argmax grid search (mini): F16 heads
    # are the dominant lever at 8/6-bit bulk (+3.4pt for +10 MB at q8_0);
    # embedding tables matter least.
    # Sub-q6 tiers are deliberately not shipped (quality floor); explore
    # them via --recipe overrides on a q6_k/q8_0 base if ever needed.
    Q = gguf.GGMLQuantizationType
    RECIPES = {  # dtype -> (bulk dec matmuls, embed tables, lm heads, t5 matmuls)
        "q8_0": (Q.Q8_0, Q.F16,  Q.F16, Q.Q8_0),
        "q6_k": (Q.Q6_K, Q.Q6_K, Q.F16, Q.Q8_0),
    }

    TIER_NAMES = ("bulk", "tables", "heads", "t5")
    tiers = dict(zip(TIER_NAMES, RECIPES.get(args.dtype, ())))
    if args.recipe:
        if not tiers:
            raise SystemExit("--recipe requires a quant --dtype (q8_0/q4_k_m/q4_0)")
        for kv in args.recipe.split(","):
            k, _, v = kv.partition("=")
            k = k.strip()
            if k not in tiers:
                raise SystemExit(f"--recipe: unknown tier '{k}' (use {'/'.join(TIER_NAMES)})")
            tiers[k] = Q[v.strip().upper()]
        log("recipe override: " + ", ".join(f"{k}={t.name}" for k, t in tiers.items()))

    def is_embed_table(name):
        return (name.startswith("dec.embed_tokens.")
                or name in ("dec.embed_prompts.weight", "t5.embed_tokens.weight"))

    def target_type(name, arr):
        if args.dtype == "f32" or keep_f32(name):
            return None
        if args.dtype == "f16":
            return None if name.startswith("t5.") else Q.F16
        qt = (tiers["tables"] if is_embed_table(name)
              else tiers["heads"] if name.startswith("dec.lm_heads.")
              else tiers["t5"] if name.startswith(("t5.", "enc_to_dec.")) else tiers["bulk"])
        if qt == Q.F16:
            return qt
        if arr.ndim != 2 or arr.shape[-1] % gguf.GGML_QUANT_SIZES[qt][0] != 0:
            log(f"keep f32 (shape {arr.shape} not blockable for {qt.name}): {name}")
            return None
        return qt

    imatrix = None
    if args.imatrix:
        imatrix = dict(np.load(args.imatrix).items())
        log(f"imatrix: {len(imatrix)} importance vectors from {args.imatrix}")

    # gguf-py's Python encoder covers Q8_0/Q4_0 (and cannot take an imatrix);
    # every other type — and any imatrix-weighted encode — goes through ggml.
    from gguf import quants as gq
    GQ_TYPES = (Q.Q8_0, Q.Q4_0)
    need_lib = imatrix is not None or any(
        t not in GQ_TYPES + (Q.F16,) for t in tiers.values())
    kquantize = None
    if need_lib:
        lib = args.ggml_lib or find_ggml_lib()
        if not lib:
            raise SystemExit("this recipe needs a built ggml-base shared lib for "
                             "encoding; build tts-cpp first or pass --ggml-lib")
        log(f"ggml encoder: {lib}")
        kquantize = make_ggml_quantizer(lib)

    counts = {}
    roundtrip = []  # (rel_rmse, name, qtype name) self-check per quantized tensor
    im_hits, im_misses = 0, []
    parity_checked = False
    for name in sorted(cv.out):
        arr = cv.out[name]
        qt = target_type(name, arr)
        if qt is None:
            w.add_tensor(name, arr)
            counts["f32"] = counts.get("f32", 0) + 1
        elif qt == Q.F16:
            w.add_tensor(name, arr.astype(np.float16))
            counts["f16"] = counts.get("f16", 0) + 1
        else:
            im = None
            if imatrix is not None:
                im = imatrix.get(cv.src_of_dest.get(name, ""))
                if im is None:
                    im_misses.append(name)
                else:
                    im_hits += 1
            if im is not None or qt not in GQ_TYPES:
                qdata = kquantize(arr, qt, im)
                if not parity_checked and qt == Q.Q8_0:
                    # q8_0 ignores quant_weights, so ggml and gguf-py (both
                    # reference RTN) must agree byte-for-byte — informational
                    same = qdata.tobytes() == gq.quantize(arr, qt).tobytes()
                    log(f"q8_0 byte parity ggml vs gguf-py: "
                        f"{'OK' if same else 'DIFFERS'} ({name})")
                    parity_checked = True
            else:
                qdata = gq.quantize(arr, qt)
            dq = gq.dequantize(qdata, qt).astype(np.float64)
            denom = float(np.sqrt(np.mean(arr.astype(np.float64) ** 2))) or 1.0
            rel = float(np.sqrt(np.mean((arr.astype(np.float64) - dq) ** 2))) / denom
            roundtrip.append((rel, name, qt.name))
            w.add_tensor(name, qdata, raw_dtype=qt)
            counts[qt.name] = counts.get(qt.name, 0) + 1
    if imatrix is not None:
        log(f"imatrix applied to {im_hits} tensors; missing for {len(im_misses)}")
        for name in im_misses:
            log(f"  no imatrix (quantized unweighted): {name}")
    log(f"tensors: {len(cv.out)} total, types: "
        + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if roundtrip:
        roundtrip.sort(reverse=True)
        log("worst dequantize-roundtrip rel-RMSE:")
        for rel, name, qn in roundtrip[:8]:
            log(f"  {rel:.4e}  {qn:5s}  {name}")
        # sanity net against catastrophic misquantization, NOT a quality bar
        assert roundtrip[0][0] < 0.25, "suspiciously large quantization error"

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    log(f"wrote {args.out} ({os.path.getsize(args.out) / 1e9:.2f} GB)")


if __name__ == "__main__":
    sys.exit(main())
