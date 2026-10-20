#!/usr/bin/env python3
"""Convert an NVIDIA NeMo .nemo archive to a single GGUF for the
parakeet.cpp Engine.

Auto-detects the model flavour from ``cfg['target']``:

  - ``EncDecCTCModelBPE``                -> CTC head      (parakeet-ctc-0.6b, -1.1b)
  - ``EncDecHybridRNNTCTCBPEModel`` (no TDT durations)
                                         -> CTC head by default
                                            (``--head rnnt`` exports the
                                            Transducer branch instead)
  - ``EncDecHybridRNNTCTCBPEModel`` (with TDT durations)
                                         -> TDT           (hybrid TDT+CTC ckpts
                                            such as parakeet-tdt_ctc-110m)
  - ``EncDecRNNTBPEModel`` (with TDT durations)
                                         -> TDT (RNN-T + duration head)
                                            (parakeet-tdt-0.6b-v3, -1.1b)
  - ``EncDecRNNTBPEModel`` (without TDT durations or EOU tokens)
                                         -> RNN-T
                                            (parakeet-unified-en-0.6b)
  - ``EncDecRNNTBPEModel`` (no TDT durations, chunked-limited streaming
                            encoder, conv_norm_type=layer_norm,
                            ``<EOU>`` token in vocab)
                                         -> EOU (FastConformer-RNN-T 120M,
                                            cache-aware streaming, end-of-
                                            utterance token detection;
                                            parakeet_realtime_eou_120m-v1)
  - ``EncDecRNNTBPEModelWithPrompt``      -> Nemotron (cache-aware
                                            FastConformer-RNN-T with a
                                            128-wide locale prompt;
                                            nemotron-3.5-asr-streaming-0.6b)
  - ``EncDecDiarLabelModel``             -> Sortformer    (diar_sortformer_4spk-v1,
                                            diar_streaming_sortformer_4spk-v2)

The FastConformer encoder topology is shared across all flavours; only
the decoder / head tensors + metadata differ. EOU additionally swaps the
conv module's BatchNorm for a LayerNorm and carries cache-aware streaming
hyperparameters (att_context_size, subsampling-output cache lookback, and the
chunk size used by the binding's reference EOU pipeline) in metadata.

Footgun: the script's ``--hf-repo`` default is ``nvidia/parakeet-ctc-0.6b``,
so when ``--ckpt`` points at a non-CTC path that does not exist locally
**you must pass ``--hf-repo`` explicitly** -- otherwise the script will
download the CTC checkpoint instead of the one named in ``--ckpt``.

Output GGUF layout (see src/parakeet_ctc.h / src/parakeet_tdt.h /
src/parakeet_sortformer.h for the consumer structs):

  Metadata:
    general.architecture  = "parakeet-ctc"  (kept for GGUF compat)
    general.name          = "<derived from cfg>"
    parakeet.model.type   = "ctc", "rnnt", "tdt", "eou", "nemotron",
                            or "sortformer"
    parakeet.encoder.*    (hyperparameters, incl. use_bias, xscaling,
                           conv_norm_type, att_context_size,
                           causal_downsampling, conv_context_size)
    parakeet.preproc.*    (mel/stft hyperparameters)
    parakeet.ctc.*        (vocab_size, blank_id)                    [CTC only]
    parakeet.tdt.*        (predictor + joint hyperparameters
                           + durations)                              [TDT only]
    parakeet.rnnt.*       (predictor + joint hyperparameters
                           + max symbols per step)                    [RNN-T only]
    parakeet.eou.*        (vocab_size, blank_id, eou_id, eob_id,
                           pred_hidden, pred_rnn_layers, joint_hidden,
                           encoder_chunk_mel_frames,
                           cache_lookback_frames, cache_time_steps,
                           max_symbols_per_step)                     [EOU only]
    parakeet.nemotron.*   (RNNT, prompt, locale, and cache-aware
                           streaming metadata)                       [Nemotron only]
    parakeet.unified.*    (cache-aware streaming contexts: left context
                           and convolution cache frames, allowed chunk
                           and right-context frames)                 [Unified RNN-T only]
    parakeet.sortformer.* (num_spks, fc/tf dims, tf layer count, ...)[Sortformer only]
    tokenizer.ggml.model  = "sentencepiece"                          [CTC, RNN-T, TDT, EOU]
    tokenizer.ggml.sentencepiece_model = <raw tokenizer.model bytes> [CTC, RNN-T, TDT, EOU]

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
    rnnt.predict.embed.weight                                        [RNN-T only]
    rnnt.predict.lstm.{l}.{w_ih,w_hh,b_ih,b_hh}                      [RNN-T only]
    rnnt.joint.{enc,pred}.{weight,bias}                              [RNN-T only]
    rnnt.joint.out.{weight,bias}                                     [RNN-T only]
    eou.predict.embed.weight                                         [EOU only]
    eou.predict.lstm.0.{w_ih,w_hh,b_ih,b_hh}                         [EOU only]
    eou.joint.{enc,pred}.{weight,bias}                               [EOU only]
    eou.joint.out.{weight,bias}                                      [EOU only]
    nemotron.predict.*                                                [Nemotron only]
    nemotron.joint.*                                                  [Nemotron only]
    nemotron.prompt.proj.{0,2}.{weight,bias}                          [Nemotron only]
    sortformer.encoder_proj.{weight,bias}                            [Sortformer only]
    sortformer.transformer.blk.{i}.* (18 blocks)                     [Sortformer only]
    sortformer.head.{weight,bias}                                    [Sortformer only]
"""

import io
import math
import os
import sys
import tarfile
from pathlib import Path

import gguf
import numpy as np
import torch
import yaml
from converter_args import parse_args


ARCH = "parakeet-ctc"

NEMOTRON_TARGET = (
    "nemo.collections.asr.models.rnnt_bpe_models_prompt."
    "EncDecRNNTBPEModelWithPrompt"
)
NEMOTRON_VARIANT = "nemotron-3.5-asr-streaming-0.6b-v1"
NEMOTRON_ALLOWED_RIGHT_CONTEXTS = [0, 1, 3, 6, 13]
NEMOTRON_ALLOWED_CHUNK_MS = [80, 160, 320, 560, 1120]
NEMOTRON_LEFT_CONTEXT = 56
NEMOTRON_DEFAULT_ATT_CONTEXT_RIGHT = 3
NEMOTRON_NUM_PROMPTS = 128
NEMOTRON_PROMPT_INPUT = 1152
NEMOTRON_PROMPT_HIDDEN = 2048
UNIFIED_ATT_CONTEXT_STYLE = "chunked_limited_with_rc"
EOU_REPO = "nvidia/parakeet_realtime_eou_120m-v1"
EOU_LICENSE = "NVIDIA Open Model License"

QUANT_MAP = {
    "bf16": gguf.GGMLQuantizationType.BF16,
    "q8_0": gguf.GGMLQuantizationType.Q8_0,
    "q5_0": gguf.GGMLQuantizationType.Q5_0,
    "q4_0": gguf.GGMLQuantizationType.Q4_0,
}

FILE_TYPE_MAP = {
    "f32":  gguf.LlamaFileType.ALL_F32,
    "f16":  gguf.LlamaFileType.MOSTLY_F16,
    "bf16": gguf.LlamaFileType.MOSTLY_BF16,
    "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
    "q5_0": gguf.LlamaFileType.MOSTLY_Q5_0,
    "q4_0": gguf.LlamaFileType.MOSTLY_Q4_0,
}


def ensure_ckpt(path: Path, hf_repo: str) -> Path:
    if path.exists():
        return path
    print(f"[convert] {path} missing, downloading {hf_repo} from Hugging Face...", file=sys.stderr)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit(
            f"[convert] huggingface_hub is not installed but {path} is missing.\n"
            "  Either:\n"
            "    1. Run 'npm run setup-models' / 'scripts/download-models.sh' "
            "(the documented path; uses curl, no Python deps), or\n"
            "    2. Install it manually: 'python -m pip install huggingface_hub'.\n"
            "  huggingface_hub is intentionally not in scripts/requirements.txt "
            "because download-models.sh is the supported entry point."
        )
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
        # Single-file SentencePiece (Parakeet CTC/TDT/EOU). Multilingual /
        # aggregate tokenizers have no top-level model_path; pieces are
        # resolved later from per-lang models or cfg labels.
        if tok_cfg and tok_cfg.get("model_path"):
            tok_fname = Path(tok_cfg["model_path"].split("nemo:", 1)[1]).name
            for m in t.getmembers():
                if m.name.endswith("/" + tok_fname) or m.name.endswith(tok_fname):
                    tok_bytes = t.extractfile(m).read()
                    break
            else:
                raise RuntimeError(f"tokenizer.model ({tok_fname}) not found in {ckpt}")

        multilingual_tok = {}
        if tok_cfg and str(tok_cfg.get("type", "")).lower() in ("multilingual", "agg", "aggregate"):
            langs = tok_cfg.get("langs") or {}
            for lang, lcfg in langs.items():
                if not isinstance(lcfg, dict) or not lcfg.get("model_path"):
                    continue
                tok_fname = Path(lcfg["model_path"].split("nemo:", 1)[1]).name
                for m in t.getmembers():
                    if m.name.endswith("/" + tok_fname) or m.name.endswith(tok_fname):
                        multilingual_tok[lang] = t.extractfile(m).read()
                        break
                else:
                    raise RuntimeError(
                        f"multilingual tokenizer.model for lang={lang} "
                        f"({tok_fname}) not found in {ckpt}"
                    )

        w_m = _get_member(t, "model_weights.ckpt")
        buf = io.BytesIO(t.extractfile(w_m).read())

    sd = torch.load(buf, map_location="cpu", weights_only=True)
    return cfg, sd, tok_bytes, multilingual_tok


def vocab_piece_list(cfg: dict):
    """Return the CTC/RNNT label vocabulary when no single SP model exists."""
    labels = cfg.get("labels")
    if labels:
        return [str(p) for p in labels]
    ctc = ctc_decoder_config(cfg)
    vocabulary = ctc.get("vocabulary")
    if vocabulary:
        return [str(p) for p in vocabulary]
    return None


def emit_ctc_language_ranges(writer, cfg: dict, multilingual_tok: dict):
    """Emit per-language CTC vocab slices for aggregate multilingual tokenizers.

    IndicConformer-style checkpoints concatenate 22 language SentencePiece
    models into one CTC head. NeMo / HF decode applies a language mask so
    greedy CTC only sees that language's token range (+ blank). Without these
    keys the engine keeps full-vocab greedy (Parakeet monolingual CTC).
    """
    tok_cfg = cfg.get("tokenizer") or {}
    langs = tok_cfg.get("langs")
    if not isinstance(langs, dict) or not langs:
        return

    import sentencepiece as spm

    lang_ids = []
    starts = []
    ends = []
    offset = 0
    for lang, _lcfg in langs.items():
        data = multilingual_tok.get(lang) if multilingual_tok else None
        if not data:
            raise RuntimeError(
                f"CTC language range for lang={lang}: tokenizer.model missing "
                f"from checkpoint (refusing 256-token fallback; offsets would "
                f"diverge from emitted tokenizer pieces)"
            )
        sp = spm.SentencePieceProcessor()
        sp.load_from_serialized_proto(data)
        n_pieces = sp.get_piece_size()
        if n_pieces <= 0:
            raise RuntimeError(
                f"CTC language range for lang={lang}: tokenizer has 0 pieces"
            )
        lang_ids.append(str(lang))
        starts.append(int(offset))
        ends.append(int(offset + n_pieces))
        offset += n_pieces

    writer.add_array("parakeet.ctc.lang_ids", lang_ids)
    writer.add_array("parakeet.ctc.lang_token_start", starts)
    writer.add_array("parakeet.ctc.lang_token_end", ends)
    print(f"[convert] ctc language ranges: {len(lang_ids)} langs, "
          f"covered_tokens={offset}", file=sys.stderr)


def emit_tokenizer_metadata(writer, tok_bytes: bytes, multilingual_tok: dict, cfg: dict):
    """Write tokenizer.ggml.* keys the engine uses for BPE detokenize.

    Prefer a single SentencePiece proto when present. For IndicConformer-style
    multilingual aggregate tokenizers, concatenate per-language SP models in
    ``tokenizer.langs`` order (matches ``cfg['labels']``). Fall back to the
    label list alone when SP protos are unavailable.
    """
    pieces = None
    scores = None
    piece_tp = None
    unk_id = 0
    bos_id = -1
    eos_id = -1
    pad_id = -1

    if tok_bytes:
        writer.add_string("tokenizer.ggml.model", "sentencepiece")
        writer.add_array("tokenizer.ggml.sentencepiece_model", list(tok_bytes))
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.load_from_serialized_proto(tok_bytes)
        n_pieces = sp.get_piece_size()
        pieces = [sp.id_to_piece(i) for i in range(n_pieces)]
        scores = [float(sp.get_score(i)) for i in range(n_pieces)]
        piece_tp = []
        for i in range(n_pieces):
            if   sp.is_unknown(i):  piece_tp.append(2)
            elif sp.is_control(i):  piece_tp.append(3)
            elif sp.is_unused(i):   piece_tp.append(5)
            elif sp.is_byte(i):     piece_tp.append(6)
            else:                   piece_tp.append(1)
        unk_id = sp.unk_id() if sp.unk_id() >= 0 else 0
        bos_id = sp.bos_id() if sp.bos_id() >= 0 else -1
        eos_id = sp.eos_id() if sp.eos_id() >= 0 else -1
        pad_id = sp.pad_id() if sp.pad_id() >= 0 else -1
    elif multilingual_tok:
        import sentencepiece as spm
        pieces = []
        scores = []
        piece_tp = []
        tok_cfg = cfg.get("tokenizer") or {}
        for lang in (tok_cfg.get("langs") or {}):
            data = multilingual_tok.get(lang)
            if data is None:
                raise RuntimeError(
                    f"multilingual tokenizer pieces for lang={lang} missing; "
                    f"cannot emit tokenizer.ggml.* aligned with lang ranges"
                )
            sp = spm.SentencePieceProcessor()
            sp.load_from_serialized_proto(data)
            for i in range(sp.get_piece_size()):
                pieces.append(sp.id_to_piece(i))
                scores.append(float(sp.get_score(i)))
                if   sp.is_unknown(i):  piece_tp.append(2)
                elif sp.is_control(i):  piece_tp.append(3)
                elif sp.is_unused(i):   piece_tp.append(5)
                elif sp.is_byte(i):     piece_tp.append(6)
                else:                   piece_tp.append(1)
        writer.add_string("tokenizer.ggml.model", "sentencepiece")
        unk_id = 0
    else:
        pieces = vocab_piece_list(cfg)
        if pieces:
            writer.add_string("tokenizer.ggml.model", "sentencepiece")
            scores = [0.0] * len(pieces)
            piece_tp = [2 if p == "<unk>" else 1 for p in pieces]
            unk_id = next((i for i, p in enumerate(pieces) if p == "<unk>"), 0)

    if not pieces:
        raise RuntimeError("no tokenizer in checkpoint (e.g. Sortformer)")

    writer.add_array("tokenizer.ggml.tokens",      pieces)
    writer.add_array("tokenizer.ggml.scores",      scores)
    writer.add_array("tokenizer.ggml.token_type",  piece_tp)
    writer.add_uint32("tokenizer.ggml.unk_token_id", unk_id if unk_id >= 0 else 0)
    if bos_id >= 0:
        writer.add_uint32("tokenizer.ggml.bos_token_id", bos_id)
    if eos_id >= 0:
        writer.add_uint32("tokenizer.ggml.eos_token_id", eos_id)
    if pad_id >= 0:
        writer.add_uint32("tokenizer.ggml.pad_token_id", pad_id)
    return len(pieces)


def nested_config_value(cfg: dict, path):
    value = cfg
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def tensor_shape(sd: dict, name: str):
    tensor = sd.get(name)
    if tensor is None or not hasattr(tensor, "shape"):
        return None
    return tuple(int(dimension) for dimension in tensor.shape)


def validate_config_value(errors, cfg: dict, path, expected):
    actual = nested_config_value(cfg, path)
    if actual != expected:
        errors.append(
            f"{'.'.join(path)}={actual!r}, expected {expected!r}"
        )


def validate_tensor_shape(errors, sd: dict, name: str, expected):
    actual = tensor_shape(sd, name)
    if actual != expected:
        errors.append(f"{name} shape={actual}, expected {expected}")


def validate_nemotron_prompt_dictionary(errors, cfg: dict):
    prompt_dictionary = nested_config_value(
        cfg,
        ("model_defaults", "prompt_dictionary"),
    )
    if not isinstance(prompt_dictionary, dict) or not prompt_dictionary:
        errors.append("model_defaults.prompt_dictionary is missing or empty")
        return

    invalid_ids = [
        f"{alias}={prompt_id!r}"
        for alias, prompt_id in prompt_dictionary.items()
        if not isinstance(prompt_id, int)
        or prompt_id < 0
        or prompt_id >= NEMOTRON_NUM_PROMPTS
    ]
    if invalid_ids:
        errors.append(
            "prompt ids must be integers in [0, 127]: "
            + ", ".join(invalid_ids)
        )
    if prompt_dictionary.get("auto") != 101:
        errors.append(
            "model_defaults.prompt_dictionary.auto must resolve to 101"
        )


def validate_nemotron_attention_contexts(errors, cfg: dict):
    contexts = nested_config_value(cfg, ("encoder", "att_context_size"))
    if not isinstance(contexts, list):
        errors.append("encoder.att_context_size must be a list of [left, right] pairs")
        return

    expected_checkpoint_contexts = {
        (NEMOTRON_LEFT_CONTEXT, 0),
        (NEMOTRON_LEFT_CONTEXT, 3),
        (NEMOTRON_LEFT_CONTEXT, 6),
        (NEMOTRON_LEFT_CONTEXT, 13),
    }
    try:
        actual_contexts = {tuple(int(value) for value in pair) for pair in contexts}
    except (TypeError, ValueError):
        errors.append("encoder.att_context_size contains an invalid context pair")
        return
    missing = sorted(expected_checkpoint_contexts - actual_contexts)
    if missing:
        errors.append(
            f"encoder.att_context_size is missing required contexts: {missing}"
        )


def validate_nemotron_contract(cfg: dict, sd: dict):
    errors = []
    config_expectations = (
        (("target",), NEMOTRON_TARGET),
        (("encoder", "n_layers"), 24),
        (("encoder", "d_model"), 1024),
        (("encoder", "n_heads"), 8),
        (("encoder", "subsampling_factor"), 8),
        (("encoder", "conv_kernel_size"), 9),
        (("encoder", "att_context_style"), "chunked_limited"),
        (("encoder", "conv_context_size"), "causal"),
        (("encoder", "causal_downsampling"), True),
        (("encoder", "conv_norm_type"), "layer_norm"),
        (("num_prompts",), NEMOTRON_NUM_PROMPTS),
        (("decoder", "vocab_size"), 13087),
        (("decoder", "prednet", "pred_hidden"), 640),
        (("decoder", "prednet", "pred_rnn_layers"), 2),
        (("joint", "num_classes"), 13087),
        (("joint", "jointnet", "encoder_hidden"), 1024),
        (("joint", "jointnet", "pred_hidden"), 640),
        (("joint", "jointnet", "joint_hidden"), 640),
        (("model_defaults", "initialize_prompt_feature"), True),
        (("model_defaults", "num_prompts"), NEMOTRON_NUM_PROMPTS),
    )
    for path, expected in config_expectations:
        validate_config_value(errors, cfg, path, expected)

    validate_nemotron_prompt_dictionary(errors, cfg)
    validate_nemotron_attention_contexts(errors, cfg)

    tensor_expectations = (
        ("prompt_kernel.0.weight", (NEMOTRON_PROMPT_HIDDEN, NEMOTRON_PROMPT_INPUT)),
        ("prompt_kernel.0.bias", (NEMOTRON_PROMPT_HIDDEN,)),
        ("prompt_kernel.2.weight", (1024, NEMOTRON_PROMPT_HIDDEN)),
        ("prompt_kernel.2.bias", (1024,)),
        ("decoder.prediction.embed.weight", (13088, 640)),
        ("decoder.prediction.dec_rnn.lstm.weight_ih_l0", (2560, 640)),
        ("decoder.prediction.dec_rnn.lstm.weight_hh_l0", (2560, 640)),
        ("decoder.prediction.dec_rnn.lstm.bias_ih_l0", (2560,)),
        ("decoder.prediction.dec_rnn.lstm.bias_hh_l0", (2560,)),
        ("decoder.prediction.dec_rnn.lstm.weight_ih_l1", (2560, 640)),
        ("decoder.prediction.dec_rnn.lstm.weight_hh_l1", (2560, 640)),
        ("decoder.prediction.dec_rnn.lstm.bias_ih_l1", (2560,)),
        ("decoder.prediction.dec_rnn.lstm.bias_hh_l1", (2560,)),
        ("joint.enc.weight", (640, 1024)),
        ("joint.enc.bias", (640,)),
        ("joint.pred.weight", (640, 640)),
        ("joint.pred.bias", (640,)),
        ("joint.joint_net.2.weight", (13088, 640)),
        ("joint.joint_net.2.bias", (13088,)),
    )
    for name, expected in tensor_expectations:
        validate_tensor_shape(errors, sd, name, expected)

    if errors:
        details = "\n  - ".join(errors)
        raise ValueError(
            "Nemotron checkpoint does not match the supported "
            f"{NEMOTRON_VARIANT} contract:\n  - {details}"
        )


def resolve_attention_context(enc: dict, default_right=None):
    raw_context = enc.get("att_context_size", [-1, -1])
    if (
        isinstance(raw_context, (list, tuple))
        and raw_context
        and isinstance(raw_context[0], (list, tuple))
    ):
        if default_right is not None:
            for pair in raw_context:
                if (
                    isinstance(pair, (list, tuple))
                    and len(pair) >= 2
                    and int(pair[1]) == default_right
                ):
                    return int(pair[0]), int(pair[1])
        raw_context = raw_context[0]
    if isinstance(raw_context, (list, tuple)) and len(raw_context) >= 2:
        return int(raw_context[0]), int(raw_context[1])
    return -1, -1


def eou_required_tensor_names(cfg: dict) -> list[str]:
    names = [
        "preprocessor.featurizer.fb",
        "preprocessor.featurizer.window",
        "encoder.pre_encode.conv.0.weight",
        "encoder.pre_encode.conv.2.weight",
        "encoder.pre_encode.conv.3.weight",
        "encoder.pre_encode.conv.5.weight",
        "encoder.pre_encode.conv.6.weight",
        "encoder.pre_encode.out.weight",
        "decoder.prediction.embed.weight",
        "joint.enc.weight", "joint.enc.bias",
        "joint.pred.weight", "joint.pred.bias",
        "joint.joint_net.2.weight", "joint.joint_net.2.bias",
    ]
    for layer in range(int(cfg.get("encoder", {}).get("n_layers", 0))):
        prefix = f"encoder.layers.{layer}"
        names.extend([
            f"{prefix}.norm_feed_forward1.weight",
            f"{prefix}.norm_feed_forward1.bias",
            f"{prefix}.feed_forward1.linear1.weight",
            f"{prefix}.feed_forward1.linear2.weight",
            f"{prefix}.norm_self_att.weight",
            f"{prefix}.norm_self_att.bias",
            f"{prefix}.self_attn.linear_q.weight",
            f"{prefix}.self_attn.linear_k.weight",
            f"{prefix}.self_attn.linear_v.weight",
            f"{prefix}.self_attn.linear_out.weight",
            f"{prefix}.self_attn.linear_pos.weight",
            f"{prefix}.self_attn.pos_bias_u",
            f"{prefix}.self_attn.pos_bias_v",
            f"{prefix}.norm_conv.weight",
            f"{prefix}.norm_conv.bias",
            f"{prefix}.conv.pointwise_conv1.weight",
            f"{prefix}.conv.depthwise_conv.weight",
            f"{prefix}.conv.batch_norm.weight",
            f"{prefix}.conv.batch_norm.bias",
            f"{prefix}.conv.pointwise_conv2.weight",
            f"{prefix}.norm_feed_forward2.weight",
            f"{prefix}.norm_feed_forward2.bias",
            f"{prefix}.feed_forward2.linear1.weight",
            f"{prefix}.feed_forward2.linear2.weight",
            f"{prefix}.norm_out.weight",
            f"{prefix}.norm_out.bias",
        ])
    for layer in range(int(cfg.get("decoder", {}).get("prednet", {}).get(
            "pred_rnn_layers", 0))):
        prefix = f"decoder.prediction.dec_rnn.lstm"
        names.extend([
            f"{prefix}.weight_ih_l{layer}",
            f"{prefix}.weight_hh_l{layer}",
            f"{prefix}.bias_ih_l{layer}",
            f"{prefix}.bias_hh_l{layer}",
        ])
    return names


def validate_eou_contract(cfg: dict, sd=None):
    errors = []
    expected = (
        (("encoder", "d_model"), 512),
        (("encoder", "n_layers"), 17),
        (("encoder", "n_heads"), 8),
        (("encoder", "subsampling_factor"), 8),
        (("encoder", "causal_downsampling"), True),
        (("encoder", "conv_context_size"), "causal"),
        (("encoder", "conv_norm_type"), "layer_norm"),
        (("encoder", "att_context_style"), "chunked_limited"),
        (("decoder", "vocab_size"), 1026),
        (("decoder", "prednet", "pred_hidden"), 640),
        (("decoder", "prednet", "pred_rnn_layers"), 1),
        (("joint", "num_classes"), 1026),
        (("joint", "jointnet", "joint_hidden"), 640),
    )
    for path, value in expected:
        validate_config_value(errors, cfg, path, value)
    if resolve_attention_context(cfg.get("encoder", {})) != (70, 1):
        errors.append("encoder.att_context_size must resolve to [70, 1]")
    labels = cfg.get("labels") or cfg.get("decoder", {}).get("vocabulary") or []
    label_set = {str(label) for label in labels}
    for token in ("<EOU>", "<EOB>"):
        if token not in label_set:
            errors.append(f"vocabulary is missing {token}")
    if labels:
        positions = {str(label): index for index, label in enumerate(labels)}
        if positions.get("<EOU>") != 1024:
            errors.append("<EOU> must have token id 1024")
        if positions.get("<EOB>") != 1025:
            errors.append("<EOB> must have token id 1025")
    if sd is not None:
        missing = [name for name in eou_required_tensor_names(cfg) if name not in sd]
        if missing:
            preview = ", ".join(missing[:8])
            suffix = f" (+{len(missing) - 8} more)" if len(missing) > 8 else ""
            errors.append(f"missing required tensors: {preview}{suffix}")
    if errors:
        raise ValueError(
            "incompatible EOU checkpoint:\n  - " + "\n  - ".join(errors))


def write_eou_provenance(writer, description=None):
    writer.add_description(
        description or
        "NVIDIA Parakeet Realtime EOU 120M "
        f"({EOU_LICENSE})"
    )
    writer.add_string("general.license", EOU_LICENSE)
    writer.add_string(
        "general.source.url",
        f"https://huggingface.co/{EOU_REPO}",
    )


def nemotron_prompt_entries(cfg: dict):
    prompt_dictionary = cfg["model_defaults"]["prompt_dictionary"]
    aliases = [str(alias) for alias in prompt_dictionary]
    prompt_ids = [int(prompt_dictionary[alias]) for alias in prompt_dictionary]
    return aliases, prompt_ids


def detect_model_type(cfg: dict, head: str = "auto") -> str:
    if head != "auto":
        return head
    target = str(cfg.get("target", ""))
    if "sortformer" in target.lower() or "sortformer_modules" in cfg:
        return "sortformer"
    if target == NEMOTRON_TARGET:
        return "nemotron"
    # Hybrid RNNT+CTC targets contain "RNNT", so resolve them before the
    # generic RNNT branch. Vanilla hybrid (IndicConformer) exports CTC for
    # v1; hybrid+TDT keeps the TDT export path.
    if "HybridRNNTCTC" in target:
        durations = cfg.get("model_defaults", {}).get("tdt_durations")
        if durations:
            return "tdt"
        return "ctc"
    is_rnnt = "rnnt" in target.lower() or \
              "tdt" in cfg.get("loss", {}).get("loss_name", "").lower()
    if is_rnnt:
        durations = cfg.get("model_defaults", {}).get("tdt_durations")
        if durations:
            return "tdt"
        labels = cfg.get("labels") or cfg.get("decoder", {}).get("vocabulary") or []
        has_eou = any(str(lbl) == "<EOU>" for lbl in labels)
        if has_eou:
            return "eou"
        return "rnnt"
    return "ctc"


def subsampling_freq_bins(feat_in: int, sub_factor: int,
                          causal_downsampling: bool) -> int:
    """Return the frequency width after the stride-2 subsampling stack."""
    if sub_factor <= 0 or sub_factor & (sub_factor - 1):
        raise ValueError("subsampling_factor must be a positive power of two")
    bins = int(feat_in)
    for _ in range(int(math.log2(sub_factor))):
        # NeMo CausalConv2D pads (2, 1), producing floor(L / 2) + 1.
        bins = bins // 2 + 1 if causal_downsampling else (bins + 2 - 3) // 2 + 1
    return bins


def rnnt_max_symbols(cfg: dict) -> int:
    greedy = (cfg.get("decoding") or {}).get("greedy") or {}
    configured = greedy.get("max_symbols", greedy.get("max_symbols_per_step"))
    if not configured:
        return 10
    return int(configured)


def validate_rnnt_contract(cfg: dict, sd: dict):
    decoder = cfg.get("decoder") or {}
    if "prednet" not in decoder or "joint" not in cfg:
        raise ValueError(
            "rnnt head requires a Transducer checkpoint "
            "(decoder.prednet / joint config missing)"
        )

    vocab_size = int(decoder["vocab_size"])
    joint_classes = int(cfg["joint"]["num_classes"])
    if joint_classes != vocab_size:
        raise ValueError(
            f"rnnt: joint.num_classes ({joint_classes}) != "
            f"decoder.vocab_size ({vocab_size})"
        )

    output_shape = tensor_shape(sd, "joint.joint_net.2.weight")
    if not output_shape or output_shape[0] != vocab_size + 1:
        raise ValueError(
            "rnnt: joint output shape="
            f"{output_shape}, expected ({vocab_size + 1}, joint_hidden); "
            "duration logits present?"
        )


def write_transducer_metadata(writer, cfg: dict, model_type: str):
    dec = cfg["decoder"]
    prefix = f"parakeet.{model_type}"
    pred_hidden = int(dec["prednet"]["pred_hidden"])
    pred_rnn_layers = int(dec["prednet"]["pred_rnn_layers"])
    joint_hidden = int(cfg["joint"]["jointnet"]["joint_hidden"])
    pred_vocab_size = int(dec["vocab_size"])
    joint_num_classes = int(cfg["joint"]["num_classes"])

    writer.add_uint32(f"{prefix}.vocab_size", pred_vocab_size)
    writer.add_uint32(f"{prefix}.blank_id", joint_num_classes)
    writer.add_uint32(f"{prefix}.pred_hidden", pred_hidden)
    writer.add_uint32(f"{prefix}.pred_rnn_layers", pred_rnn_layers)
    writer.add_uint32(f"{prefix}.joint_hidden", joint_hidden)

    if model_type in ("rnnt", "nemotron"):
        max_symbols = rnnt_max_symbols(cfg)
        writer.add_uint32(f"{prefix}.max_symbols_per_step", max_symbols)
        return

    durations = tdt_durations(cfg)
    writer.add_uint32(f"{prefix}.num_durations", len(durations))
    writer.add_array(f"{prefix}.durations", durations)


def tdt_durations(cfg: dict) -> list:
    """The TDT duration set, from whichever config location holds it.

    NeMo's released checkpoints keep it under model_defaults; some training
    configs (TranscribePsy among them) keep it under loss.tdt_kwargs or
    decoding instead, and a config with neither still implies 0..N-1 from
    the joint's extra duration outputs.
    """
    model_defaults = cfg.get("model_defaults") or {}
    if model_defaults.get("tdt_durations") is not None:
        durations = list(model_defaults["tdt_durations"])
        num_durations = int(model_defaults.get("num_tdt_durations", len(durations)))
        if num_durations != len(durations):
            raise ValueError(
                f"num_tdt_durations {num_durations} != len(durations) {len(durations)}"
            )
        return durations
    durations = (
        ((cfg.get("loss") or {}).get("tdt_kwargs") or {}).get("durations")
        or (cfg.get("decoding") or {}).get("durations")
    )
    if durations is not None:
        return list(durations)
    num_extra_outputs = int((cfg.get("joint") or {}).get("num_extra_outputs", 0))
    if num_extra_outputs <= 0:
        raise KeyError(
            "could not resolve TDT durations from config "
            "(model_defaults.tdt_durations / loss.tdt_kwargs.durations / "
            "decoding.durations all missing and joint.num_extra_outputs is 0)")
    return list(range(num_extra_outputs))


def unified_streaming_contexts(enc: dict):
    if str(enc.get("att_context_style", "regular")) != UNIFIED_ATT_CONTEXT_STYLE:
        return None
    raw = enc.get("att_chunk_context_size")
    if not raw or len(raw) != 3:
        return None
    left, chunks, rights = (sorted(int(value) for value in group) for group in raw)
    if len(left) != 1:
        raise ValueError(
            "encoder.att_chunk_context_size left context must hold exactly one value"
        )
    return left[0], chunks, rights


def write_unified_streaming_metadata(writer, cfg: dict) -> bool:
    contexts = unified_streaming_contexts(cfg["encoder"])
    if contexts is None:
        return False
    left, chunks, rights = contexts
    conv_kernel = int(cfg["encoder"]["conv_kernel_size"])
    writer.add_uint32("parakeet.unified.left_context_frames", left)
    writer.add_uint32("parakeet.unified.cache_time_steps", (conv_kernel - 1) // 2)
    writer.add_array("parakeet.unified.allowed_chunk_frames", chunks)
    writer.add_array("parakeet.unified.allowed_right_context_frames", rights)
    return True


def write_nemotron_metadata(writer, cfg: dict):
    write_transducer_metadata(writer, cfg, "nemotron")
    aliases, prompt_ids = nemotron_prompt_entries(cfg)

    writer.add_uint32(
        "parakeet.nemotron.num_prompts",
        NEMOTRON_NUM_PROMPTS,
    )
    writer.add_uint32(
        "parakeet.nemotron.prompt_width",
        NEMOTRON_NUM_PROMPTS,
    )
    writer.add_uint32(
        "parakeet.nemotron.prompt_input_width",
        NEMOTRON_PROMPT_INPUT,
    )
    writer.add_uint32(
        "parakeet.nemotron.left_context_frames",
        NEMOTRON_LEFT_CONTEXT,
    )
    writer.add_uint32(
        "parakeet.nemotron.cache_time_steps",
        int(cfg["encoder"]["conv_kernel_size"]) - 1,
    )
    writer.add_array(
        "parakeet.nemotron.allowed_right_context_frames",
        NEMOTRON_ALLOWED_RIGHT_CONTEXTS,
    )
    writer.add_array(
        "parakeet.nemotron.allowed_chunk_ms",
        NEMOTRON_ALLOWED_CHUNK_MS,
    )
    writer.add_array("parakeet.nemotron.locale_aliases", aliases)
    writer.add_array("parakeet.nemotron.locale_prompt_ids", prompt_ids)
    writer.add_string("parakeet.nemotron.default_locale", "auto")


def write_transducer_tensors(cfg: dict, sd: dict, model_type: str, add_2d, add_f32):
    prefix = model_type
    add_2d(f"{prefix}.predict.embed.weight", sd["decoder.prediction.embed.weight"])

    pred_rnn_layers = int(cfg["decoder"]["prednet"]["pred_rnn_layers"])
    for layer in range(pred_rnn_layers):
        source = "decoder.prediction.dec_rnn.lstm"
        target = f"{prefix}.predict.lstm.{layer}"
        add_2d(f"{target}.w_ih", sd[f"{source}.weight_ih_l{layer}"])
        add_2d(f"{target}.w_hh", sd[f"{source}.weight_hh_l{layer}"])
        add_f32(f"{target}.b_ih", sd[f"{source}.bias_ih_l{layer}"])
        add_f32(f"{target}.b_hh", sd[f"{source}.bias_hh_l{layer}"])

    add_2d(f"{prefix}.joint.enc.weight", sd["joint.enc.weight"])
    add_f32(f"{prefix}.joint.enc.bias", sd["joint.enc.bias"])
    add_2d(f"{prefix}.joint.pred.weight", sd["joint.pred.weight"])
    add_f32(f"{prefix}.joint.pred.bias", sd["joint.pred.bias"])
    add_2d(f"{prefix}.joint.out.weight", sd["joint.joint_net.2.weight"])
    add_f32(f"{prefix}.joint.out.bias", sd["joint.joint_net.2.bias"])


def write_nemotron_prompt_tensors(sd: dict, add_2d, add_f32):
    add_2d(
        "nemotron.prompt.proj.0.weight",
        sd["prompt_kernel.0.weight"],
    )
    add_f32(
        "nemotron.prompt.proj.0.bias",
        sd["prompt_kernel.0.bias"],
    )
    add_2d(
        "nemotron.prompt.proj.2.weight",
        sd["prompt_kernel.2.weight"],
    )
    add_f32(
        "nemotron.prompt.proj.2.bias",
        sd["prompt_kernel.2.bias"],
    )


def ctc_decoder_config(cfg: dict) -> dict:
    """Return the ConvASRDecoder config used for CTC vocab metadata.

    Pure CTC checkpoints store ``num_classes`` under ``decoder``. Hybrid
    RNNT+CTC checkpoints keep the RNNT predictor under ``decoder`` and the
    CTC head config under ``aux_ctc.decoder``.
    """
    aux = cfg.get("aux_ctc")
    if isinstance(aux, dict):
        decoder = aux.get("decoder")
        if isinstance(decoder, dict) and "num_classes" in decoder:
            return decoder
    return cfg.get("decoder") or {}


def resolve_ctc_head_tensors(sd: dict):
    """Locate the CTC 1x1 conv head weight/bias in a state dict.

    Hybrid checkpoints register the head as ``self.ctc_decoder``; pure CTC
    checkpoints use ``self.decoder``. Prefer the hybrid keys, then fall back.
    """
    hybrid_w = "ctc_decoder.decoder_layers.0.weight"
    hybrid_b = "ctc_decoder.decoder_layers.0.bias"
    plain_w = "decoder.decoder_layers.0.weight"
    plain_b = "decoder.decoder_layers.0.bias"
    if hybrid_w in sd and hybrid_b in sd:
        return sd[hybrid_w].squeeze(-1), sd[hybrid_b]
    if plain_w in sd and plain_b in sd:
        return sd[plain_w].squeeze(-1), sd[plain_b]
    raise KeyError(
        "CTC head not found: expected ctc_decoder.decoder_layers.0.{weight,bias} "
        "or decoder.decoder_layers.0.{weight,bias}"
    )


def as_np(t: torch.Tensor, dtype=None) -> np.ndarray:
    a = t.detach().cpu().numpy()
    if dtype is not None:
        a = a.astype(dtype, copy=False)
    return np.ascontiguousarray(a)


def fuse_bn(weight, bias, running_mean, running_var, eps=1e-5):
    scale = weight / np.sqrt(running_var + eps)
    shift = bias - running_mean * scale
    return scale.astype(np.float32), shift.astype(np.float32)


def detect_sortformer_variant(ckpt: Path) -> str:
    """
    Map a NeMo Sortformer .nemo filename to a stable variant tag the C++
    loader can match against. The tag is the only thing that distinguishes
    cache-aware v2.1 from architecturally-identical v1 / v2 at GGUF time
    (encoder shape alone is ambiguous against future variants).
    """
    stem = ckpt.stem
    if "streaming_sortformer" in stem and "-v2.1" in stem:
        return "sortformer-streaming-v2.1-aosc"
    if "streaming_sortformer" in stem and "-v2" in stem:
        return "sortformer-streaming-v2"
    if "diar_sortformer" in stem and "-v1" in stem:
        return "sortformer-v1"
    return ""


def long_form_hint_frames(hop_length: int, sample_rate: int, subsampling_factor: int,
                          window_sec: float, overlap: float | None) -> tuple:
    """Encoder frames of a recommended long-form window and of the context
    on each side for a share `overlap` of the window (half of it each side)."""
    frame_sec = hop_length * subsampling_factor / sample_rate
    window_frames = int(round(window_sec / frame_sec))
    context_frames = 0 if not overlap else int(round(window_frames * overlap / 2))
    return window_frames, context_frames


def write_long_form_hint(writer, hop_length: int, sample_rate: int, cfg: dict,
                         window_sec: float | None, overlap: float | None):
    """parakeet.long_form.* keys for a model that recommends its own window."""
    if window_sec is None:
        return
    subsampling_factor = int(cfg.get("encoder", {}).get("subsampling_factor", 8))
    window_frames, context_frames = long_form_hint_frames(
        hop_length, sample_rate, subsampling_factor, window_sec, overlap)
    writer.add_uint32("parakeet.long_form.window_frames", window_frames)
    writer.add_uint32("parakeet.long_form.context_frames", context_frames)


def write_provenance(writer, model_type: str, description=None):
    """general.description plus the per-family license keys.

    An explicit `description` (the converter's --description) replaces the
    family text only; the license and source keys stay.
    """
    if model_type == "nemotron":
        writer.add_description(
            description or
            "NVIDIA Nemotron 3.5 ASR Streaming 0.6B "
            "(OpenMDW-1.1)"
        )
        writer.add_string("general.license", "OpenMDW-1.1")
        writer.add_string(
            "general.source.url",
            "https://huggingface.co/nvidia/"
            "nemotron-3.5-asr-streaming-0.6b",
        )
        writer.add_string("parakeet.model_variant", NEMOTRON_VARIANT)
    elif model_type == "eou":
        write_eou_provenance(writer, description)
    else:
        writer.add_description(
            description or
            f"NVIDIA Parakeet-{model_type.upper()} "
            "FastConformer ASR (CC-BY-4.0)"
        )


def write_gguf(out: Path, ckpt: Path, cfg: dict, sd: dict, tok_bytes: bytes,
               multilingual_tok: dict, quant: str, head: str = "auto",
               description=None,
               long_form_window_sec=None, long_form_overlap=None):
    model_type = detect_model_type(cfg, head)
    if model_type == "rnnt":
        validate_rnnt_contract(cfg, sd)
    if model_type == "eou":
        validate_eou_contract(cfg, sd)
    if model_type == "nemotron":
        validate_nemotron_contract(cfg, sd)

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
    causal_downsample = bool(enc.get("causal_downsampling", False))
    sub_freq_bins = subsampling_freq_bins(
        feat_in, sub_factor, causal_downsample)

    sample_rate   = int(pre["sample_rate"])
    n_fft         = int(pre["n_fft"])
    n_mels        = int(pre["features"])
    win_length    = int(round(float(pre["window_size"]) * sample_rate))
    hop_length    = int(round(float(pre["window_stride"]) * sample_rate))

    writer = gguf.GGUFWriter(str(out), arch=ARCH)

    model_name = {
        "ctc":         f"parakeet-ctc-{d_model}-{n_layers}l",
        "rnnt":        f"parakeet-rnnt-{d_model}-{n_layers}l",
        "tdt":         f"parakeet-tdt-{d_model}-{n_layers}l",
        "eou":         f"parakeet-eou-{d_model}-{n_layers}l",
        "nemotron":    f"nemotron-3.5-asr-streaming-{d_model}-{n_layers}l",
        "sortformer":  f"sortformer-{d_model}-{n_layers}l",
    }[model_type]
    writer.add_name(model_name)
    write_provenance(writer, model_type, description)
    write_long_form_hint(writer, hop_length, sample_rate, cfg, long_form_window_sec, long_form_overlap)
    writer.add_file_type(FILE_TYPE_MAP[quant])

    writer.add_string("parakeet.model.type", model_type)

    conv_norm_type   = str(enc.get("conv_norm_type", "batch_norm"))
    conv_context_str = str(enc.get("conv_context_size", "default"))
    conv_context_style = str(enc.get("conv_context_style", "regular"))
    att_style        = str(enc.get("att_context_style", "regular"))
    att_ctx_left, att_ctx_right = resolve_attention_context(
        enc,
        default_right=(
            NEMOTRON_DEFAULT_ATT_CONTEXT_RIGHT
            if model_type == "nemotron"
            else None
        ),
    )

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
    writer.add_string("parakeet.encoder.conv_context_style",          conv_context_style)
    writer.add_bool  ("parakeet.encoder.causal_downsampling",         causal_downsample)
    writer.add_string("parakeet.encoder.att_context_style",           att_style)
    writer.add_int32 ("parakeet.encoder.att_context_size_left",       att_ctx_left)
    writer.add_int32 ("parakeet.encoder.att_context_size_right",      att_ctx_right)
    if model_type in ("rnnt", "eou", "nemotron"):
        unified_streaming = model_type == "rnnt" and write_unified_streaming_metadata(writer, cfg)
        writer.add_bool(
            "parakeet.encoder.streaming.enabled",
            model_type in ("eou", "nemotron") or unified_streaming,
        )

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
        ctc_dec = ctc_decoder_config(cfg)
        vocab_size = int(ctc_dec["num_classes"]) + 1
        blank_id   = vocab_size - 1
        writer.add_uint32("parakeet.ctc.vocab_size", vocab_size)
        writer.add_uint32("parakeet.ctc.blank_id",   blank_id)
        emit_ctc_language_ranges(writer, cfg, multilingual_tok)
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
        # Variant tag (preferred over shape-based detection on the C++ side).
        # Empty string = unknown checkpoint; loader falls back to encoder
        # shape so older GGUFs continue to load.
        variant = detect_sortformer_variant(ckpt)
        if variant:
            writer.add_string("parakeet.model_variant", variant)
    elif model_type == "nemotron":
        write_nemotron_metadata(writer, cfg)
    else:
        write_transducer_metadata(writer, cfg, model_type)

    try:
        n_tok = emit_tokenizer_metadata(writer, tok_bytes, multilingual_tok, cfg)
        print(f"[convert] tokenizer pieces={n_tok}", file=sys.stderr)
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
        block_quant = quant in {"q8_0", "q5_0", "q4_0"}
        if qtype is None or (block_quant and arr.shape[-1] % 32 != 0):
            writer.add_tensor(name, arr.astype(fallback_dtype, copy=False))
            return
        packed = gguf.quants.quantize(arr, qtype)
        writer.add_tensor(name, packed, raw_dtype=qtype)

    def add_conv(name: str, t: torch.Tensor):
        # ggml-metal supports BF16 matrix multiplication on recent Apple GPUs,
        # but its convolution IM2COL path does not accept BF16 kernels, and the
        # subsampler's pointwise lowering casts activations to its kernel type.
        # A MOSTLY_BF16 file therefore keeps all subsampling plus depthwise
        # kernels in F16; encoder/decoder projection matrices remain BF16.
        if quant != "bf16":
            add_2d(name, t)
            return
        arr = as_np(t, np.float32)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr.squeeze(-1)
        writer.add_tensor(name, arr.astype(np.float16, copy=False))

    def try_bias(name: str, key: str):
        if key in sd:
            add_f32(name, sd[key])

    add_conv("encoder.subsampling.conv0.weight",  sd["encoder.pre_encode.conv.0.weight"])
    try_bias("encoder.subsampling.conv0.bias",    "encoder.pre_encode.conv.0.bias")
    add_conv("encoder.subsampling.conv1_dw.weight", sd["encoder.pre_encode.conv.2.weight"])
    try_bias("encoder.subsampling.conv1_dw.bias",   "encoder.pre_encode.conv.2.bias")
    add_conv("encoder.subsampling.conv1_pw.weight", sd["encoder.pre_encode.conv.3.weight"])
    try_bias("encoder.subsampling.conv1_pw.bias",   "encoder.pre_encode.conv.3.bias")
    add_conv("encoder.subsampling.conv2_dw.weight", sd["encoder.pre_encode.conv.5.weight"])
    try_bias("encoder.subsampling.conv2_dw.bias",   "encoder.pre_encode.conv.5.bias")
    add_conv("encoder.subsampling.conv2_pw.weight", sd["encoder.pre_encode.conv.6.weight"])
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
        add_conv(f"{p}.conv.dw.weight",    sd[f"{k}.conv.depthwise_conv.weight"])
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
        dec_w, dec_b = resolve_ctc_head_tensors(sd)
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
    elif model_type == "nemotron":
        write_transducer_tensors(
            cfg,
            sd,
            model_type,
            add_2d,
            add_f32,
        )
        write_nemotron_prompt_tensors(sd, add_2d, add_f32)
    else:
        write_transducer_tensors(cfg, sd, model_type, add_2d, add_f32)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = out.stat().st_size / (1024 * 1024)
    if model_type == "ctc":
        vocab_note = f"ctc_vocab={int(ctc_decoder_config(cfg)['num_classes'])+1}"
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
    elif model_type == "rnnt":
        vocab_note = (
            f"rnnt_vocab={int(cfg['decoder']['vocab_size'])} "
            f"max_symbols={rnnt_max_symbols(cfg)}"
        )
    elif model_type == "nemotron":
        vocab_note = (
            f"nemotron_vocab={int(cfg['decoder']['vocab_size'])} "
            f"prompts={NEMOTRON_NUM_PROMPTS} "
            f"contexts={NEMOTRON_ALLOWED_RIGHT_CONTEXTS}"
        )
    else:
        vocab_note = f"tdt_vocab={int(cfg['decoder']['vocab_size'])} durations={tdt_durations(cfg)}"
    print(f"[convert] wrote {out} ({size_mb:.1f} MiB, type={model_type}, quant={quant}, {vocab_note}, layers={n_layers}, use_bias={use_bias})", file=sys.stderr)


def main():
    args = parse_args(__doc__)
    ckpt = ensure_ckpt(args.ckpt, args.hf_repo)
    cfg, sd, tok_bytes, multilingual_tok = load_nemo(ckpt)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_gguf(
        args.out, ckpt, cfg, sd, tok_bytes, multilingual_tok, args.quant,
        args.head, args.description,
        args.long_form_window_sec, args.long_form_overlap,
    )


if __name__ == "__main__":
    main()
