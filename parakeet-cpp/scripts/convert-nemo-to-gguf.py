#!/usr/bin/env python3
"""Convert an NVIDIA NeMo .nemo archive to a single GGUF for the
parakeet.cpp Engine.

Auto-detects the model flavour from ``cfg['target']``:

  - ``EncDecCTCModelBPE``                -> CTC head      (parakeet-ctc-0.6b, -1.1b)
  - ``EncDecRNNTBPEModel`` (with TDT durations)
                                         -> TDT (RNN-T + duration head)
                                            (parakeet-tdt-0.6b-v3, -1.1b)
  - ``EncDecRNNTBPEModel`` (no TDT durations, chunked-limited streaming
                            encoder, conv_norm_type=layer_norm,
                            ``<EOU>`` token in vocab)
                                         -> EOU (FastConformer-RNN-T 120M,
                                            cache-aware streaming, end-of-
                                            utterance token detection;
                                            parakeet_realtime_eou_120m-v1)
  - ``EncDecDiarLabelModel``             -> Sortformer    (diar_sortformer_4spk-v1,
                                            diar_streaming_sortformer_4spk-v2)

The FastConformer encoder topology is shared across all four flavours; only
the decoder / head tensors + metadata differ. EOU additionally swaps the
conv module's BatchNorm for a LayerNorm and carries cache-aware streaming
hyperparameters (att_context_size, subsampling-output cache lookback, and the
EOU reference chunk size) in metadata.

Warning: the script's ``--hf-repo`` default is ``nvidia/parakeet-ctc-0.6b``.
When ``--ckpt`` names a non-CTC file that is not present locally, pass ``--hf-repo``
explicitly or the downloader may fetch the default CTC checkpoint instead.

Output GGUF layout (see src/parakeet_ctc.h / src/parakeet_tdt.h /
src/parakeet_sortformer.h for the consumer structs):

  Metadata:
    general.architecture  = "parakeet-ctc"  (kept for GGUF compat)
    general.name          = "<derived from cfg>"
    parakeet.model.type   = "ctc", "tdt", "eou", or "sortformer"
    parakeet.encoder.*    (hyperparameters, incl. use_bias, xscaling,
                           conv_norm_type, att_context_size,
                           causal_downsampling, conv_context_size)
    parakeet.preproc.*    (mel/stft hyperparameters)
    parakeet.ctc.*        (vocab_size, blank_id)                    [CTC only]
    parakeet.tdt.*        (predictor + joint hyperparameters
                           + durations)                              [TDT only]
    parakeet.eou.*        (vocab_size, blank_id, eou_id, eob_id,
                           pred_hidden, pred_rnn_layers, joint_hidden,
                           encoder_chunk_mel_frames,
                           cache_lookback_frames, cache_time_steps,
                           max_symbols_per_step)                     [EOU only]
    parakeet.sortformer.* (num_spks, fc/tf dims, tf layer count, ...)[Sortformer only]
    tokenizer.ggml.model  = "sentencepiece"                          [CTC, TDT, EOU]
    tokenizer.ggml.sentencepiece_model = <raw tokenizer.model bytes> [CTC, TDT, EOU]

  Tensors:
    preproc.mel_filterbank            (n_mels, 257)   f32
    preproc.window                    (400,)          f32
    encoder.subsampling.{conv0,conv{1,2}_{dw,pw},out}.{weight,bias?}
    encoder.blk.{i}.* (17-42 blocks; biases omitted when use_bias=False;
                       conv module emits {bn.scale,bn.shift} for BatchNorm
                       checkpoints OR {norm.weight,norm.bias} for LayerNorm
                       checkpoints, gated by parakeet.encoder.conv_norm_type)
    ctc.decoder.{weight,bias}                                       [CTC only]
    tdt.predict.embed.weight                                         [TDT only]
    tdt.predict.lstm.{l}.{w_ih,w_hh,b_ih,b_hh}                       [TDT only]
    tdt.joint.{enc,pred}.{weight,bias}                               [TDT only]
    tdt.joint.out.{weight,bias}                                      [TDT only]
    eou.predict.embed.weight                                         [EOU only]
    eou.predict.lstm.0.{w_ih,w_hh,b_ih,b_hh}                         [EOU only]
    eou.joint.{enc,pred}.{weight,bias}                               [EOU only]
    eou.joint.out.{weight,bias}                                      [EOU only]
    sortformer.encoder_proj.{weight,bias}                            [Sortformer only]
    sortformer.transformer.blk.{i}.* (18 blocks)                     [Sortformer only]
    sortformer.head.{weight,bias}                                    [Sortformer only]
"""

import argparse
import io
import os
import sys
import tarfile
from pathlib import Path

import gguf
import numpy as np
import torch
import yaml


ARCH = "parakeet-ctc"
QUANT_CHOICES = ["f32", "f16", "q8_0", "q5_0", "q4_0"]

QUANT_MAP = {
    "q8_0": gguf.GGMLQuantizationType.Q8_0,
    "q5_0": gguf.GGMLQuantizationType.Q5_0,
    "q4_0": gguf.GGMLQuantizationType.Q4_0,
}

FILE_TYPE_MAP = {
    "f32":  gguf.LlamaFileType.ALL_F32,
    "f16":  gguf.LlamaFileType.MOSTLY_F16,
    "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
    "q5_0": gguf.LlamaFileType.MOSTLY_Q5_0,
    "q4_0": gguf.LlamaFileType.MOSTLY_Q4_0,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, default=Path("models/parakeet-ctc-0.6b.nemo"),
                   help="Path to .nemo archive (tarball). Downloads from HF if missing.")
    p.add_argument("--out", type=Path, default=Path("models/parakeet-ctc-0.6b.gguf"),
                   help="Output GGUF path.")
    p.add_argument("--quant", choices=QUANT_CHOICES, default="q8_0",
                   help="Weight dtype for 2D projection matrices. Biases / norms / BN "
                        "stay at f32. q8_0 default (~2x smaller than f16, bit-equal "
                        "transcripts on clean speech across CTC/TDT/EOU/Sortformer); "
                        "pass --quant f16 for the bit-equal floating-point baseline.")
    p.add_argument("--hf-repo", default="nvidia/parakeet-ctc-0.6b",
                   help="HF model id to download from if --ckpt is missing.")
    return p.parse_args()


def ensure_ckpt(path: Path, hf_repo: str) -> Path:
    if path.exists():
        return path
    print(f"[convert] {path} missing, downloading {hf_repo} from Hugging Face...", file=sys.stderr)
    from huggingface_hub import hf_hub_download
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    cache = path.parent / "hf-cache"
    cache.mkdir(parents=True, exist_ok=True)
    src = hf_hub_download(repo_id=hf_repo, filename=path.name, cache_dir=str(cache))
    path.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(src, path)
    return path


def _get_member(t: tarfile.TarFile, name: str) -> tarfile.TarInfo:
    # NeMo ships some checkpoints with `./` prefix on members, others without.
    for candidate in ("./" + name, name):
        try:
            return t.getmember(candidate)
        except KeyError:
            continue
    raise KeyError(name)


def load_nemo(ckpt: Path):
    with tarfile.open(ckpt, "r") as t:
        cfg_m = _get_member(t, "model_config.yaml")
        cfg   = yaml.safe_load(t.extractfile(cfg_m).read().decode())

        tok_bytes = b""
        tok_cfg   = cfg.get("tokenizer")
        if tok_cfg and tok_cfg.get("model_path"):
            tok_fname = Path(tok_cfg["model_path"].split("nemo:", 1)[1]).name
            for m in t.getmembers():
                if m.name.endswith("/" + tok_fname) or m.name.endswith(tok_fname):
                    tok_bytes = t.extractfile(m).read()
                    break
            else:
                raise RuntimeError(f"tokenizer.model ({tok_fname}) not found in {ckpt}")

        w_m = _get_member(t, "model_weights.ckpt")
        buf = io.BytesIO(t.extractfile(w_m).read())

    sd = torch.load(buf, map_location="cpu", weights_only=True)
    return cfg, sd, tok_bytes


def detect_model_type(cfg: dict) -> str:
    target = cfg.get("target", "")
    if "Sortformer" in target or "sortformer_modules" in cfg:
        return "sortformer"
    is_rnnt = "RNNT" in target or \
              "tdt" in cfg.get("loss", {}).get("loss_name", "").lower()
    if is_rnnt:
        durations = cfg.get("model_defaults", {}).get("tdt_durations")
        if durations:
            return "tdt"
        labels = cfg.get("labels") or cfg.get("decoder", {}).get("vocabulary") or []
        has_eou = any(str(lbl) == "<EOU>" for lbl in labels)
        if has_eou:
            return "eou"
        return "tdt"
    return "ctc"


def as_np(t: torch.Tensor, dtype=None) -> np.ndarray:
    a = t.detach().cpu().numpy()
    if dtype is not None:
        a = a.astype(dtype, copy=False)
    return np.ascontiguousarray(a)


def fuse_bn(weight, bias, running_mean, running_var, eps=1e-5):
    scale = weight / np.sqrt(running_var + eps)
    shift = bias - running_mean * scale
    return scale.astype(np.float32), shift.astype(np.float32)


def write_gguf(out: Path, cfg: dict, sd: dict, tok_bytes: bytes, quant: str):
    model_type = detect_model_type(cfg)

    enc = cfg["encoder"]
    pre = cfg["preprocessor"]
    dec = cfg.get("decoder", {})

    d_model       = int(enc["d_model"])
    n_layers      = int(enc["n_layers"])
    n_heads       = int(enc["n_heads"])
    head_dim      = d_model // n_heads
    ff_dim        = d_model * int(enc["ff_expansion_factor"])
    conv_kernel   = int(enc["conv_kernel_size"])
    sub_factor    = int(enc["subsampling_factor"])
    sub_channels  = int(enc["subsampling_conv_channels"])
    xscaling      = bool(enc.get("xscaling", True))
    untie_biases  = bool(enc.get("untie_biases", True))
    pos_max_len   = int(enc.get("pos_emb_max_len", 5000))
    use_bias      = bool(enc.get("use_bias", True))

    feat_in       = int(enc["feat_in"])
    sub_freq_bins = feat_in
    for _ in range(int(np.log2(sub_factor))):
        sub_freq_bins = (sub_freq_bins + 2 * 1 - 3) // 2 + 1

    sample_rate   = int(pre["sample_rate"])
    n_fft         = int(pre["n_fft"])
    n_mels        = int(pre["features"])
    win_length    = int(round(float(pre["window_size"]) * sample_rate))
    hop_length    = int(round(float(pre["window_stride"]) * sample_rate))

    writer = gguf.GGUFWriter(str(out), arch=ARCH)

    model_name = {
        "ctc":         f"parakeet-ctc-{d_model}-{n_layers}l",
        "tdt":         f"parakeet-tdt-{d_model}-{n_layers}l",
        "eou":         f"parakeet-eou-{d_model}-{n_layers}l",
        "sortformer":  f"sortformer-{d_model}-{n_layers}l",
    }[model_type]
    writer.add_name(model_name)
    writer.add_description(f"NVIDIA Parakeet-{model_type.upper()} FastConformer ASR (CC-BY-4.0)")
    writer.add_file_type(FILE_TYPE_MAP[quant])

    writer.add_string("parakeet.model.type", model_type)

    conv_norm_type   = str(enc.get("conv_norm_type", "batch_norm"))
    conv_context_str = str(enc.get("conv_context_size", "default"))
    causal_downsample = bool(enc.get("causal_downsampling", False))
    att_style        = str(enc.get("att_context_style", "regular"))
    att_ctx_raw      = enc.get("att_context_size", [-1, -1])
    if isinstance(att_ctx_raw, (list, tuple)) and len(att_ctx_raw) >= 2:
        att_ctx_left, att_ctx_right = int(att_ctx_raw[0]), int(att_ctx_raw[1])
    else:
        att_ctx_left, att_ctx_right = -1, -1

    writer.add_uint32("parakeet.encoder.d_model",                     d_model)
    writer.add_uint32("parakeet.encoder.n_layers",                    n_layers)
    writer.add_uint32("parakeet.encoder.n_heads",                     n_heads)
    writer.add_uint32("parakeet.encoder.head_dim",                    head_dim)
    writer.add_uint32("parakeet.encoder.ff_dim",                      ff_dim)
    writer.add_uint32("parakeet.encoder.conv_kernel",                 conv_kernel)
    writer.add_uint32("parakeet.encoder.subsampling_factor",          sub_factor)
    writer.add_uint32("parakeet.encoder.subsampling_conv_channels",   sub_channels)
    writer.add_uint32("parakeet.encoder.subsampling_freq_bins",       sub_freq_bins)
    writer.add_bool  ("parakeet.encoder.xscaling",                    xscaling)
    writer.add_bool  ("parakeet.encoder.untie_biases",                untie_biases)
    writer.add_bool  ("parakeet.encoder.use_bias",                    use_bias)
    writer.add_uint32("parakeet.encoder.pos_emb_max_len",             pos_max_len)
    writer.add_string("parakeet.encoder.conv_norm_type",              conv_norm_type)
    writer.add_string("parakeet.encoder.conv_context_size",           conv_context_str)
    writer.add_bool  ("parakeet.encoder.causal_downsampling",         causal_downsample)
    writer.add_string("parakeet.encoder.att_context_style",           att_style)
    writer.add_int32 ("parakeet.encoder.att_context_size_left",       att_ctx_left)
    writer.add_int32 ("parakeet.encoder.att_context_size_right",      att_ctx_right)

    normalize_str = str(pre.get("normalize", "per_feature"))

    writer.add_uint32 ("parakeet.preproc.sample_rate",               sample_rate)
    writer.add_uint32 ("parakeet.preproc.n_fft",                     n_fft)
    writer.add_uint32 ("parakeet.preproc.win_length",                win_length)
    writer.add_uint32 ("parakeet.preproc.hop_length",                hop_length)
    writer.add_uint32 ("parakeet.preproc.n_mels",                    n_mels)
    writer.add_float32("parakeet.preproc.preemph",                   0.97)
    writer.add_float32("parakeet.preproc.log_zero_guard_value",      float(2 ** -24))
    writer.add_string ("parakeet.preproc.normalize",                 normalize_str)

    if model_type == "ctc":
        vocab_size = int(dec["num_classes"]) + 1
        blank_id   = vocab_size - 1
        writer.add_uint32("parakeet.ctc.vocab_size", vocab_size)
        writer.add_uint32("parakeet.ctc.blank_id",   blank_id)
    elif model_type == "eou":
        pred_hidden       = int(dec["prednet"]["pred_hidden"])
        pred_rnn_layers   = int(dec["prednet"]["pred_rnn_layers"])
        joint_hidden      = int(cfg["joint"]["jointnet"]["joint_hidden"])
        pred_vocab_size   = int(dec["vocab_size"])
        joint_num_classes = int(cfg["joint"]["num_classes"])
        blank_id          = joint_num_classes
        labels = cfg.get("labels") or cfg.get("decoder", {}).get("vocabulary") or []
        eou_id = next((i for i, lbl in enumerate(labels) if str(lbl) == "<EOU>"), -1)
        eob_id = next((i for i, lbl in enumerate(labels) if str(lbl) == "<EOB>"), -1)
        if eou_id < 0:
            print(f"[convert] warn: <EOU> not found in vocabulary; consumer will fall back to id 1024",
                  file=sys.stderr)

        encoder_chunk_mel_frames = 25
        cache_lookback_frames    = att_ctx_left if att_ctx_left > 0 else 70
        cache_time_steps         = max(0, conv_kernel - 1)
        max_symbols_per_step     = 5

        writer.add_uint32("parakeet.eou.vocab_size",                pred_vocab_size)
        writer.add_uint32("parakeet.eou.blank_id",                  blank_id)
        writer.add_int32 ("parakeet.eou.eou_id",                    eou_id)
        writer.add_int32 ("parakeet.eou.eob_id",                    eob_id)
        writer.add_uint32("parakeet.eou.pred_hidden",               pred_hidden)
        writer.add_uint32("parakeet.eou.pred_rnn_layers",           pred_rnn_layers)
        writer.add_uint32("parakeet.eou.joint_hidden",              joint_hidden)
        writer.add_uint32("parakeet.eou.encoder_chunk_mel_frames",  encoder_chunk_mel_frames)
        writer.add_uint32("parakeet.eou.cache_lookback_frames",     cache_lookback_frames)
        writer.add_uint32("parakeet.eou.cache_time_steps",          cache_time_steps)
        writer.add_uint32("parakeet.eou.max_symbols_per_step",      max_symbols_per_step)
    elif model_type == "sortformer":
        sf  = cfg["sortformer_modules"]
        tfe = cfg["transformer_encoder"]
        writer.add_uint32("parakeet.sortformer.num_spks",        int(sf["num_spks"]))
        writer.add_uint32("parakeet.sortformer.fc_d_model",      int(sf["fc_d_model"]))
        writer.add_uint32("parakeet.sortformer.tf_d_model",      int(sf["tf_d_model"]))
        writer.add_uint32("parakeet.sortformer.tf_n_layers",     int(tfe["num_layers"]))
        writer.add_uint32("parakeet.sortformer.tf_inner_size",   int(tfe["inner_size"]))
        writer.add_uint32("parakeet.sortformer.tf_n_heads",      int(tfe["num_attention_heads"]))
        writer.add_bool  ("parakeet.sortformer.tf_pre_ln",       bool(tfe.get("pre_ln", False)))
        writer.add_string("parakeet.sortformer.tf_hidden_act",   str(tfe.get("hidden_act", "relu")))
    else:
        pred_hidden      = int(dec["prednet"]["pred_hidden"])
        pred_rnn_layers  = int(dec["prednet"]["pred_rnn_layers"])
        joint_hidden     = int(cfg["joint"]["jointnet"]["joint_hidden"])
        pred_vocab_size  = int(dec["vocab_size"])                   # label vocab (no blank)
        joint_num_classes = int(cfg["joint"]["num_classes"])        # label vocab + blank
        durations        = list(cfg["model_defaults"]["tdt_durations"])
        num_durations    = int(cfg["model_defaults"]["num_tdt_durations"])
        assert num_durations == len(durations), \
            f"num_tdt_durations {num_durations} != len(durations) {len(durations)}"
        blank_id         = joint_num_classes                         # blank_as_pad at vocab_size

        writer.add_uint32("parakeet.tdt.vocab_size",       pred_vocab_size)
        writer.add_uint32("parakeet.tdt.blank_id",         blank_id)
        writer.add_uint32("parakeet.tdt.pred_hidden",      pred_hidden)
        writer.add_uint32("parakeet.tdt.pred_rnn_layers",  pred_rnn_layers)
        writer.add_uint32("parakeet.tdt.joint_hidden",     joint_hidden)
        writer.add_uint32("parakeet.tdt.num_durations",    num_durations)
        writer.add_array ("parakeet.tdt.durations",        durations)

    if tok_bytes:
        writer.add_string("tokenizer.ggml.model", "sentencepiece")
        writer.add_array ("tokenizer.ggml.sentencepiece_model",
                          list(tok_bytes))

    try:
        if not tok_bytes:
            raise RuntimeError("no tokenizer in checkpoint (e.g. Sortformer)")
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.load_from_serialized_proto(tok_bytes)
        n_pieces = sp.get_piece_size()
        pieces    = [sp.id_to_piece(i) for i in range(n_pieces)]
        scores    = [float(sp.get_score(i)) for i in range(n_pieces)]
        piece_tp  = []
        for i in range(n_pieces):
            if   sp.is_unknown(i):  piece_tp.append(2)
            elif sp.is_control(i):  piece_tp.append(3)
            elif sp.is_unused(i):   piece_tp.append(5)
            elif sp.is_byte(i):     piece_tp.append(6)
            else:                   piece_tp.append(1)
        writer.add_array("tokenizer.ggml.tokens",      pieces)
        writer.add_array("tokenizer.ggml.scores",      scores)
        writer.add_array("tokenizer.ggml.token_type",  piece_tp)
        writer.add_uint32("tokenizer.ggml.unk_token_id",
                          sp.unk_id() if sp.unk_id() >= 0 else 0)
        writer.add_uint32("tokenizer.ggml.bos_token_id",
                          sp.bos_id() if sp.bos_id() >= 0 else 0)
        writer.add_uint32("tokenizer.ggml.eos_token_id",
                          sp.eos_id() if sp.eos_id() >= 0 else 0)
        writer.add_uint32("tokenizer.ggml.pad_token_id",
                          sp.pad_id() if sp.pad_id() >= 0 else 0)
    except Exception as e:
        print(f"[convert] warn: could not emit tokenizer pieces: {e}", file=sys.stderr)

    fb = as_np(sd["preprocessor.featurizer.fb"][0], np.float32)
    writer.add_tensor("preproc.mel_filterbank", fb)

    window = as_np(sd["preprocessor.featurizer.window"], np.float32)
    writer.add_tensor("preproc.window", window)

    if quant == "f32":
        fallback_dtype = np.float32
    else:
        fallback_dtype = np.float16

    qtype = QUANT_MAP.get(quant)

    def add_f32(name: str, t: torch.Tensor):
        writer.add_tensor(name, as_np(t, np.float32))

    def add_2d(name: str, t: torch.Tensor):
        arr = as_np(t, np.float32)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr.squeeze(-1)
        if qtype is None or arr.shape[-1] % 32 != 0:
            writer.add_tensor(name, arr.astype(fallback_dtype, copy=False))
            return
        packed = gguf.quants.quantize(arr, qtype)
        writer.add_tensor(name, packed, raw_dtype=qtype)

    def try_bias(name: str, key: str):
        if key in sd:
            add_f32(name, sd[key])

    add_2d ("encoder.subsampling.conv0.weight",  sd["encoder.pre_encode.conv.0.weight"])
    try_bias("encoder.subsampling.conv0.bias",    "encoder.pre_encode.conv.0.bias")
    add_2d ("encoder.subsampling.conv1_dw.weight", sd["encoder.pre_encode.conv.2.weight"])
    try_bias("encoder.subsampling.conv1_dw.bias",   "encoder.pre_encode.conv.2.bias")
    add_2d ("encoder.subsampling.conv1_pw.weight", sd["encoder.pre_encode.conv.3.weight"])
    try_bias("encoder.subsampling.conv1_pw.bias",   "encoder.pre_encode.conv.3.bias")
    add_2d ("encoder.subsampling.conv2_dw.weight", sd["encoder.pre_encode.conv.5.weight"])
    try_bias("encoder.subsampling.conv2_dw.bias",   "encoder.pre_encode.conv.5.bias")
    add_2d ("encoder.subsampling.conv2_pw.weight", sd["encoder.pre_encode.conv.6.weight"])
    try_bias("encoder.subsampling.conv2_pw.bias",   "encoder.pre_encode.conv.6.bias")
    add_2d ("encoder.subsampling.out.weight",      sd["encoder.pre_encode.out.weight"])
    try_bias("encoder.subsampling.out.bias",        "encoder.pre_encode.out.bias")

    for i in range(n_layers):
        k = f"encoder.layers.{i}"
        p = f"encoder.blk.{i}"

        add_f32(f"{p}.norm_ff1.weight",   sd[f"{k}.norm_feed_forward1.weight"])
        add_f32(f"{p}.norm_ff1.bias",     sd[f"{k}.norm_feed_forward1.bias"])
        add_2d (f"{p}.ff1.linear1.weight", sd[f"{k}.feed_forward1.linear1.weight"])
        try_bias(f"{p}.ff1.linear1.bias",  f"{k}.feed_forward1.linear1.bias")
        add_2d (f"{p}.ff1.linear2.weight", sd[f"{k}.feed_forward1.linear2.weight"])
        try_bias(f"{p}.ff1.linear2.bias",  f"{k}.feed_forward1.linear2.bias")

        add_f32(f"{p}.norm_attn.weight",  sd[f"{k}.norm_self_att.weight"])
        add_f32(f"{p}.norm_attn.bias",    sd[f"{k}.norm_self_att.bias"])
        q_w = sd[f"{k}.self_attn.linear_q.weight"]
        k_w = sd[f"{k}.self_attn.linear_k.weight"]
        v_w = sd[f"{k}.self_attn.linear_v.weight"]
        add_2d (f"{p}.attn.q.weight",     q_w)
        try_bias(f"{p}.attn.q.bias",      f"{k}.self_attn.linear_q.bias")
        add_2d (f"{p}.attn.k.weight",     k_w)
        try_bias(f"{p}.attn.k.bias",      f"{k}.self_attn.linear_k.bias")
        add_2d (f"{p}.attn.v.weight",     v_w)
        try_bias(f"{p}.attn.v.bias",      f"{k}.self_attn.linear_v.bias")

        add_2d (f"{p}.attn.qkv.weight",   torch.cat([q_w, k_w, v_w], dim=0))
        if use_bias:
            q_b = sd[f"{k}.self_attn.linear_q.bias"]
            k_b = sd[f"{k}.self_attn.linear_k.bias"]
            v_b = sd[f"{k}.self_attn.linear_v.bias"]
            add_f32(f"{p}.attn.qkv.bias",     torch.cat([q_b, k_b, v_b], dim=0))
        add_2d (f"{p}.attn.out.weight",   sd[f"{k}.self_attn.linear_out.weight"])
        try_bias(f"{p}.attn.out.bias",     f"{k}.self_attn.linear_out.bias")
        add_2d (f"{p}.attn.pos.weight",   sd[f"{k}.self_attn.linear_pos.weight"])
        add_f32(f"{p}.attn.pos_bias_u",   sd[f"{k}.self_attn.pos_bias_u"])
        add_f32(f"{p}.attn.pos_bias_v",   sd[f"{k}.self_attn.pos_bias_v"])

        add_f32(f"{p}.norm_conv.weight",  sd[f"{k}.norm_conv.weight"])
        add_f32(f"{p}.norm_conv.bias",    sd[f"{k}.norm_conv.bias"])
        add_2d (f"{p}.conv.pw1.weight",   sd[f"{k}.conv.pointwise_conv1.weight"])
        try_bias(f"{p}.conv.pw1.bias",    f"{k}.conv.pointwise_conv1.bias")
        add_2d (f"{p}.conv.dw.weight",    sd[f"{k}.conv.depthwise_conv.weight"])
        try_bias(f"{p}.conv.dw.bias",     f"{k}.conv.depthwise_conv.bias")

        if conv_norm_type == "layer_norm":
            ln_w = as_np(sd[f"{k}.conv.batch_norm.weight"], np.float32)
            ln_b = as_np(sd[f"{k}.conv.batch_norm.bias"],   np.float32)
            writer.add_tensor(f"{p}.conv.norm.weight", ln_w)
            writer.add_tensor(f"{p}.conv.norm.bias",   ln_b)
        else:
            bn_w    = as_np(sd[f"{k}.conv.batch_norm.weight"],        np.float32)
            bn_b    = as_np(sd[f"{k}.conv.batch_norm.bias"],          np.float32)
            bn_mean = as_np(sd[f"{k}.conv.batch_norm.running_mean"],  np.float32)
            bn_var  = as_np(sd[f"{k}.conv.batch_norm.running_var"],   np.float32)
            bn_scale, bn_shift = fuse_bn(bn_w, bn_b, bn_mean, bn_var, eps=1e-5)
            writer.add_tensor(f"{p}.conv.bn.scale", bn_scale)
            writer.add_tensor(f"{p}.conv.bn.shift", bn_shift)

        add_2d (f"{p}.conv.pw2.weight",   sd[f"{k}.conv.pointwise_conv2.weight"])
        try_bias(f"{p}.conv.pw2.bias",    f"{k}.conv.pointwise_conv2.bias")

        add_f32(f"{p}.norm_ff2.weight",   sd[f"{k}.norm_feed_forward2.weight"])
        add_f32(f"{p}.norm_ff2.bias",     sd[f"{k}.norm_feed_forward2.bias"])
        add_2d (f"{p}.ff2.linear1.weight", sd[f"{k}.feed_forward2.linear1.weight"])
        try_bias(f"{p}.ff2.linear1.bias",  f"{k}.feed_forward2.linear1.bias")
        add_2d (f"{p}.ff2.linear2.weight", sd[f"{k}.feed_forward2.linear2.weight"])
        try_bias(f"{p}.ff2.linear2.bias",  f"{k}.feed_forward2.linear2.bias")

        add_f32(f"{p}.norm_out.weight",   sd[f"{k}.norm_out.weight"])
        add_f32(f"{p}.norm_out.bias",     sd[f"{k}.norm_out.bias"])

    if model_type == "ctc":
        dec_w = sd["decoder.decoder_layers.0.weight"].squeeze(-1)
        dec_b = sd["decoder.decoder_layers.0.bias"]
        add_2d ("ctc.decoder.weight", dec_w)
        add_f32("ctc.decoder.bias",   dec_b)
    elif model_type == "eou":
        add_2d ("eou.predict.embed.weight", sd["decoder.prediction.embed.weight"])

        eou_pred_layers = int(cfg["decoder"]["prednet"]["pred_rnn_layers"])
        for l in range(eou_pred_layers):
            add_2d (f"eou.predict.lstm.{l}.w_ih",
                    sd[f"decoder.prediction.dec_rnn.lstm.weight_ih_l{l}"])
            add_2d (f"eou.predict.lstm.{l}.w_hh",
                    sd[f"decoder.prediction.dec_rnn.lstm.weight_hh_l{l}"])
            add_f32(f"eou.predict.lstm.{l}.b_ih",
                    sd[f"decoder.prediction.dec_rnn.lstm.bias_ih_l{l}"])
            add_f32(f"eou.predict.lstm.{l}.b_hh",
                    sd[f"decoder.prediction.dec_rnn.lstm.bias_hh_l{l}"])

        add_2d ("eou.joint.enc.weight",  sd["joint.enc.weight"])
        add_f32("eou.joint.enc.bias",    sd["joint.enc.bias"])
        add_2d ("eou.joint.pred.weight", sd["joint.pred.weight"])
        add_f32("eou.joint.pred.bias",   sd["joint.pred.bias"])
        add_2d ("eou.joint.out.weight",  sd["joint.joint_net.2.weight"])
        add_f32("eou.joint.out.bias",    sd["joint.joint_net.2.bias"])
    elif model_type == "sortformer":
        add_2d ("sortformer.encoder_proj.weight", sd["sortformer_modules.encoder_proj.weight"])
        add_f32("sortformer.encoder_proj.bias",   sd["sortformer_modules.encoder_proj.bias"])

        tf_n_layers = int(cfg["transformer_encoder"]["num_layers"])
        for i in range(tf_n_layers):
            k = f"transformer_encoder.layers.{i}"
            p = f"sortformer.transformer.blk.{i}"

            add_2d (f"{p}.attn.q.weight",   sd[f"{k}.first_sub_layer.query_net.weight"])
            add_f32(f"{p}.attn.q.bias",     sd[f"{k}.first_sub_layer.query_net.bias"])
            add_2d (f"{p}.attn.k.weight",   sd[f"{k}.first_sub_layer.key_net.weight"])
            add_f32(f"{p}.attn.k.bias",     sd[f"{k}.first_sub_layer.key_net.bias"])
            add_2d (f"{p}.attn.v.weight",   sd[f"{k}.first_sub_layer.value_net.weight"])
            add_f32(f"{p}.attn.v.bias",     sd[f"{k}.first_sub_layer.value_net.bias"])
            add_2d (f"{p}.attn.out.weight", sd[f"{k}.first_sub_layer.out_projection.weight"])
            add_f32(f"{p}.attn.out.bias",   sd[f"{k}.first_sub_layer.out_projection.bias"])

            add_f32(f"{p}.ln1.weight",      sd[f"{k}.layer_norm_1.weight"])
            add_f32(f"{p}.ln1.bias",        sd[f"{k}.layer_norm_1.bias"])

            add_2d (f"{p}.ffn.in.weight",   sd[f"{k}.second_sub_layer.dense_in.weight"])
            add_f32(f"{p}.ffn.in.bias",     sd[f"{k}.second_sub_layer.dense_in.bias"])
            add_2d (f"{p}.ffn.out.weight",  sd[f"{k}.second_sub_layer.dense_out.weight"])
            add_f32(f"{p}.ffn.out.bias",    sd[f"{k}.second_sub_layer.dense_out.bias"])

            add_f32(f"{p}.ln2.weight",      sd[f"{k}.layer_norm_2.weight"])
            add_f32(f"{p}.ln2.bias",        sd[f"{k}.layer_norm_2.bias"])

        add_2d ("sortformer.head.first_hidden_to_hidden.weight",
                sd["sortformer_modules.first_hidden_to_hidden.weight"])
        add_f32("sortformer.head.first_hidden_to_hidden.bias",
                sd["sortformer_modules.first_hidden_to_hidden.bias"])
        add_2d ("sortformer.head.single_hidden_to_spks.weight",
                sd["sortformer_modules.single_hidden_to_spks.weight"])
        add_f32("sortformer.head.single_hidden_to_spks.bias",
                sd["sortformer_modules.single_hidden_to_spks.bias"])
    else:
        add_2d ("tdt.predict.embed.weight", sd["decoder.prediction.embed.weight"])

        pred_rnn_layers = int(cfg["decoder"]["prednet"]["pred_rnn_layers"])
        for l in range(pred_rnn_layers):
            add_2d (f"tdt.predict.lstm.{l}.w_ih",
                    sd[f"decoder.prediction.dec_rnn.lstm.weight_ih_l{l}"])
            add_2d (f"tdt.predict.lstm.{l}.w_hh",
                    sd[f"decoder.prediction.dec_rnn.lstm.weight_hh_l{l}"])
            add_f32(f"tdt.predict.lstm.{l}.b_ih",
                    sd[f"decoder.prediction.dec_rnn.lstm.bias_ih_l{l}"])
            add_f32(f"tdt.predict.lstm.{l}.b_hh",
                    sd[f"decoder.prediction.dec_rnn.lstm.bias_hh_l{l}"])

        add_2d ("tdt.joint.enc.weight",  sd["joint.enc.weight"])
        add_f32("tdt.joint.enc.bias",    sd["joint.enc.bias"])
        add_2d ("tdt.joint.pred.weight", sd["joint.pred.weight"])
        add_f32("tdt.joint.pred.bias",   sd["joint.pred.bias"])
        add_2d ("tdt.joint.out.weight",  sd["joint.joint_net.2.weight"])
        add_f32("tdt.joint.out.bias",    sd["joint.joint_net.2.bias"])

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = out.stat().st_size / (1024 * 1024)
    if model_type == "ctc":
        vocab_note = f"ctc_vocab={int(cfg['decoder']['num_classes'])+1}"
    elif model_type == "sortformer":
        vocab_note = (f"num_spks={cfg['sortformer_modules']['num_spks']} "
                      f"tf_layers={cfg['transformer_encoder']['num_layers']} "
                      f"tf_d_model={cfg['transformer_encoder']['hidden_size']}")
    elif model_type == "eou":
        labels = cfg.get("labels") or cfg.get("decoder", {}).get("vocabulary") or []
        eou_pos = next((i for i, lbl in enumerate(labels) if str(lbl) == "<EOU>"), -1)
        vocab_note = (f"eou_vocab={int(cfg['decoder']['vocab_size'])} "
                      f"blank_id={int(cfg['joint']['num_classes'])} eou_id={eou_pos} "
                      f"att_ctx=[{att_ctx_left},{att_ctx_right}] "
                      f"conv_norm={conv_norm_type}")
    else:
        vocab_note = f"tdt_vocab={int(cfg['decoder']['vocab_size'])} durations={cfg['model_defaults']['tdt_durations']}"
    print(f"[convert] wrote {out} ({size_mb:.1f} MiB, type={model_type}, quant={quant}, {vocab_note}, layers={n_layers}, use_bias={use_bias})", file=sys.stderr)


def main():
    args = parse_args()
    ckpt = ensure_ckpt(args.ckpt, args.hf_repo)
    cfg, sd, tok_bytes = load_nemo(ckpt)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_gguf(args.out, cfg, sd, tok_bytes, args.quant)


if __name__ == "__main__":
    main()
