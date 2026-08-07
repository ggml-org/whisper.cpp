"""Shared helpers for the Audio8 TTS conversion and reference scripts.

Holds the pieces every Audio8 script needs: reading the checkpoint config,
instantiating the reference codec through its own remote code, loading the
generation model, the tensor-name mapping the GGUF converters use, and the
storage-type policy they share.

Scripts in this directory import it by adding their own directory to sys.path,
so they keep working when invoked by absolute path from anywhere.
"""
import json
import os
import re
from collections import Counter

import numpy as np

HEAD_DIM = 64
GGML_MAX_NAME = 64
LM_ARCH = "audio8-lm"
CODEC_ARCH = "audio8-codec"
LM_KV = "audio8.lm"
CODEC_KV = "audio8.codec"

CODEC_RENAMES = (
    ("encoder.block.", "enc/"),
    ("decoder.model.", "dec/"),
    ("quantizer.semantic_quantizer.quantizers.", "q/sem/"),
    ("quantizer.quantizer.quantizers.", "q/res/"),
    ("quantizer.downsample.", "q/down/"),
    ("quantizer.upsample.", "q/up/"),
    ("quantizer.pre_module.", "q/pre/"),
    ("quantizer.post_module.", "q/post/"),
    ("attention_layer_scale.gamma", "attn_scale"),
    ("ffn_layer_scale.gamma", "ffn_scale"),
    ("attention_norm.weight", "attn_norm"),
    ("ffn_norm.weight", "ffn_norm"),
    ("attention.wo.weight", "wo"),
    ("feed_forward.w1.weight", "w1"),
    ("feed_forward.w2.weight", "w2"),
    ("feed_forward.w3.weight", "w3"),
    ("layers.", "blk/"),
)

FUSED_ATTENTION_SUFFIX = "attention/wqkv/weight"

CODEC_PARTS = ("encoder", "decoder")
ENCODER_PREFIXES = ("enc/", "q/down/", "q/pre/")
DECODER_PREFIXES = ("dec/", "q/up/", "q/post/")
# Both halves carry the codebooks, so each file stands alone.
SHARED_SUFFIXES = ("/codebook/weight",)
ANALYSIS_PROJECTION = "/in_proj/"
# The encoder subtracts each quantizer's reconstruction from the residual it
# passes on, so it needs the projection back out of the codebook as well.
RECONSTRUCTION_PROJECTION = "/out_proj/"

# The engine runs the codec channels-inner, so a convolution is a stack of
# per-tap [in, out] matrices applied to shifted views of the signal rather than
# ggml's [tap, in, out] im2col kernel. Storing the taps outermost is what makes
# each of those matrices a contiguous slice.
DEPTHWISE_MARKER = "/dwconv/"
TRANSPOSED_CONV = re.compile(r"^(?:q/up/\d+/0|dec/\d+/block/1)/conv/weight$")
SNAKE_ALPHA_SUFFIX = "/alpha"

ROPE_PRECISIONS = ("bf16", "f32")

QUANT_BLOCK = 32
QUANT_TYPES = ("q8_0", "q4_0")
STORAGE_TYPES = ("f32", "f16") + QUANT_TYPES

# What a tensor is used for, which is what decides how far its storage may be
# degraded. EXACT tensors settle argmax and argmin ties or carry RoPE phases,
# where one rounding step moves the whole trajectory, and they are small enough
# that keeping them is free. BLOCK tensors are the body matmuls a block format
# is designed for. DENSE tensors are bulk weights that must stay addressable
# element-wise -- embedding tables read by ggml_get_rows, convolution kernels --
# so they follow the build down to f16 and no further.
EXACT = "exact"
BLOCK = "block"
DENSE = "dense"


def storage_encoding(dtype, role):
    if dtype == "f32" or role == EXACT:
        return "f32"
    if role == BLOCK and dtype in QUANT_TYPES:
        return dtype
    return "f16"


def encode_tensor(array, encoding):
    import gguf

    if encoding in QUANT_TYPES:
        quant_type = getattr(gguf.GGMLQuantizationType, encoding.upper())
        return gguf.quants.quantize(array, quant_type), quant_type
    if encoding == "f16":
        return array.astype(np.float16), None
    return array, None


def check_tensor_name(name):
    if len(name) >= GGML_MAX_NAME:
        raise ValueError(f"tensor name too long for ggml: {name}")


def write_tensors(writer, named, dtype, role_of):
    """Add every tensor at the storage its role allows, tallying the encodings."""
    tally = Counter()
    for name, array in sorted(named.items()):
        check_tensor_name(name)
        contiguous = np.ascontiguousarray(array)
        encoding = storage_encoding(dtype, role_of(name, contiguous))
        payload, raw_dtype = encode_tensor(contiguous, encoding)
        writer.add_tensor(name, payload, raw_dtype=raw_dtype)
        tally[encoding] += 1
    return tally


def format_tally(tally):
    return ", ".join(f"{count} {name}" for name, count in sorted(tally.items()))


def flush(writer):
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def load_config(model_dir):
    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as handle:
        return json.load(handle)


def precompute_rope(length, head_dim, theta, precision="bf16"):
    """Mirror the checkpoint's own RoPE table construction.

    The phases are accumulated in float32 and the table is rounded to bfloat16,
    exactly as _precompute_rope and the codec's _rope do. Both details are
    load-bearing: a float64 phase or a float32 table differs by up to one
    bfloat16 step, which is enough to change greedy codebook choices.
    """
    import torch

    frequencies = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2).float()[: head_dim // 2] / head_dim)
    )
    phases = torch.outer(torch.arange(length), frequencies)
    values = torch.polar(torch.ones_like(phases), phases)
    table = torch.stack((values.real, values.imag), dim=-1)
    if precision == "bf16":
        table = table.to(torch.bfloat16)
    return table.to(torch.float32).numpy()


def rope_planes(prefix, length, head_dim, theta, precision="bf16"):
    """The same table as two [length, head_dim / 2] planes.

    Split because the engine multiplies by cosine and sine separately; keeping
    them interleaved would cost a strided gather in every attention block.
    """
    table = precompute_rope(length, head_dim, theta, precision)
    return {
        f"{prefix}_cos": np.ascontiguousarray(table[..., 0]),
        f"{prefix}_sin": np.ascontiguousarray(table[..., 1]),
    }


def interleave_to_split_order(head_dim):
    """Row order taking a head from the checkpoint's interleaved RoPE layout,
    which rotates the adjacent pair (2j, 2j+1), to the split-half layout that
    rotates (j, head_dim / 2 + j)."""
    order = np.empty(head_dim, dtype=np.int64)
    order[: head_dim // 2] = np.arange(0, head_dim, 2)
    order[head_dim // 2:] = np.arange(1, head_dim, 2)
    return order


def permute_rope_layout(rows, head_dim):
    """Reorder every head's rows of a q or k projection into the split-half
    layout. Attention is unchanged by this: the same permutation lands on both
    q and k, and their dot product does not depend on the order of the terms.
    Doing it here means the engine can use one rotation for every model in the
    tree, and lets a float32 table be applied by ggml_rope_ext directly."""
    order = interleave_to_split_order(head_dim)
    heads = rows.reshape(-1, head_dim, *rows.shape[1:])
    return np.ascontiguousarray(heads[:, order].reshape(rows.shape))


def to_numpy(tensor):
    return tensor.detach().float().cpu().numpy()


def rename_codec_key(key):
    for source, target in CODEC_RENAMES:
        key = key.replace(source, target)
    return key.replace(".", "/")


def is_conv_kernel(name, array):
    return array.ndim == 3 and not name.endswith(SNAKE_ALPHA_SUFFIX)


def tap_major_kernel(name, array):
    """Reorder one convolution kernel from torch's layout to ggml [in, out, tap].

    Torch spells a Conv1d kernel [out, in, tap] and a ConvTranspose1d one
    [in, out, tap], and a depthwise kernel has no input axis at all.
    """
    if DEPTHWISE_MARKER in name:
        return np.ascontiguousarray(array[:, 0, :].T)
    if TRANSPOSED_CONV.match(name):
        return np.ascontiguousarray(array.transpose(2, 1, 0))
    return np.ascontiguousarray(array.transpose(2, 0, 1))


def to_engine_layout(name, array):
    """Both the converter and the verifier read the codec through this, so the
    checkpoint is only ever reshaped in one place."""
    if name.endswith(SNAKE_ALPHA_SUFFIX):
        return array.reshape(-1)
    if is_conv_kernel(name, array):
        return tap_major_kernel(name, array)
    return array


def belongs_to_encoder(name):
    if name.startswith(ENCODER_PREFIXES):
        return True
    return ANALYSIS_PROJECTION in name or RECONSTRUCTION_PROJECTION in name


def belongs_to_decoder(name):
    if name.startswith(DECODER_PREFIXES):
        return True
    return RECONSTRUCTION_PROJECTION in name


def is_shared(name):
    return name.endswith(SHARED_SUFFIXES)


def select_part(named, part):
    """Which of the checkpoint's tensors a half carries. The converter splits
    with this and the verifier rebuilds each file's expectation with it, so
    neither can drift from the other's idea of where a tensor belongs."""
    keep = belongs_to_encoder if part == "encoder" else belongs_to_decoder
    return {
        name: array for name, array in named.items() if keep(name) or is_shared(name)
    }


def split_qkv(fused, n_head, n_kv, head_dim=HEAD_DIM):
    query = n_head * head_dim
    key = query + n_kv * head_dim
    return fused[:query], fused[query:key], fused[key:]


def expand_fused(prefix, fused, n_head, n_kv, head_dim=HEAD_DIM, suffix=""):
    query, key, value = split_qkv(fused, n_head, n_kv, head_dim)
    return {
        f"{prefix}/wq{suffix}": permute_rope_layout(query, head_dim),
        f"{prefix}/wk{suffix}": permute_rope_layout(key, head_dim),
        f"{prefix}/wv{suffix}": value,
    }


def codec_head_counts(rows, dim, head_dim=HEAD_DIM):
    n_head = dim // head_dim
    n_kv = (rows // head_dim - n_head) // 2
    return n_head, n_kv


def strip_generator_prefix(state):
    if not any("generator." in key for key in state):
        return state
    return {
        key.replace("generator.", ""): value
        for key, value in state.items()
        if "generator." in key
    }


def drop_buffers(state):
    return {
        key: value
        for key, value in state.items()
        if not key.endswith(("freqs_cis", "causal_mask"))
    }


def raw_codec_state(model_dir, codec_file="codec.pth"):
    import torch

    state = torch.load(
        os.path.join(model_dir, codec_file), map_location="cpu", weights_only=True
    )
    if "state_dict" in state:
        state = state["state_dict"]
    return drop_buffers(strip_generator_prefix(state))


def is_weight_norm_hook(hook):
    return hasattr(hook, "compute_weight") and hasattr(hook, "name")


def refresh_legacy_weight_norm(module):
    """torch.nn.utils.weight_norm only recomputes `weight` inside its forward
    pre-hook, so after load_state_dict the attribute still holds the value from
    module construction."""
    for child in module.modules():
        for hook in child._forward_pre_hooks.values():
            if is_weight_norm_hook(hook):
                setattr(child, hook.name, hook.compute_weight(child))
    return module


def load_codec_module(model_dir, codec_file="codec.pth"):
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    codec_class = get_class_from_dynamic_module(
        "modeling_arktts_codec.ArkttsCodec", model_dir
    )
    codec = codec_class(config)
    codec.load_state_dict(raw_codec_state(model_dir, codec_file), strict=True)
    return refresh_legacy_weight_norm(codec.eval())


def load_generation_model(model_dir):
    import torch
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_dir, trust_remote_code=True, dtype=torch.float32
    ).eval()
    return processor, model


def save_arrays(out_dir, arrays):
    os.makedirs(out_dir, exist_ok=True)
    for name, array in arrays.items():
        np.save(os.path.join(out_dir, name), array)
        print(f"  {name}: {array.shape} {array.dtype}")


def write_meta(out_dir, meta, name="meta.json"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)
    print(f"  {name}: {len(meta)} keys")
