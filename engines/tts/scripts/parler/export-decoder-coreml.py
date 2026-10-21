#!/usr/bin/env python3
"""Export one Parler-TTS decoder step from the model GGUF to a stateful Core ML
package for the engine's Apple Neural Engine sidecar.

The 24-layer decoder (self-attention, cross-attention over the description,
GELU FFN, final LayerNorm, the nine codebook heads) is rebuilt in PyTorch from
the same GGUF tensors the ggml engine loads, any quantization (dequantized to
f32 here; the package carries its own fp16 / int8 / palettized weights).

One call advances one position:

  inputs   hidden      [1, d, 1, 1]      the summed code embeddings of the frame
                                         (or the prompt-token embedding) plus
                                         the position row, host side
           self_mask   [1, 1, 1, ctx]    0 for cached rows < n_past, MASK_OFF
                                         elsewhere
           cross_mask  [1, 1, 1, cross]  0 for real description tokens
  states   self_k_GG, self_v_GG   [G, H, ctx, HD]    read only
           cross_k_GG, cross_v_GG [G, H, cross, HD]  read only
  outputs  logits      [n_q, vocab]
           new_k, new_v [L, H, 1, HD]   this position's self-attention K / V

G layers share a state (--layers-per-state, default all L: four states);
layer l is index l % G of state <kind>_<l // G>. With per-layer states the M2
Neural Engine compiler rejects the full 24-layer model (MLComputePlan error
-14 from 13 layers up; CPU and GPU load it), while four stacked states let the
full decoder execute with ANE placement.

The graph never writes a state. Every self-attention reads the cached rows as
one score vector, scores the current position against its own fresh K
separately, softmaxes the concatenation and adds the current V's share to the
weighted cache sum; the host then copies new_k / new_v into row n_past of the
self states (one HD-wide row per layer and head, through the MLState's
multiarray view) before the next call. The model therefore costs no
whole-cache rewrite: the previous spike's one-hot blend wrote ~100 MB of state
per step at ctx 1024. Cross K / V are projected once per description on the
host and written into their states once per utterance. Rows the mask switches
off must still hold finite values (zero the states at session start): the
masked weight is an exact 0, and 0 * NaN is NaN.

Neural Engine spellings: activations are [1, C, 1, 1] and every projection is
a 1x1 conv2d (Apple's ane_transformers layout); LayerNorm normalises the
channel axis; the 1 / sqrt(HD) score scale is folded into the query weights
(an exact power of two for HD = 64), so new_k is the plain K projection; the
nine heads are one conv to n_q * vocab channels. Experimental --k-layout cols
stores K as [G, H, HD, rows], dropping the score matmul's per-step transpose.
The native runtime does not support this layout; use rows for engine sidecars.

Export next to the GGUF (the engine looks for <stem>-decoder.mlmodelc):

  python scripts/parler/export-decoder-coreml.py \\
      --gguf models/parler-mini-v1-f32.gguf --ctx 3072 --cross-context 128 \\
      --compile-dir models

Weights default to int8 per channel: one M2 case0 teacher-forced run measured
98.06% argmax agreement with the fp32 rebuild (401 frames x 9 codebooks).
This does not establish equivalence to ggml q8_0; the native KL/distribution
quality gate is still pending.

Check the rebuild against dump-reference.py's HF fixtures first (teacher
forced, fp32, no export):

  python scripts/parler/export-decoder-coreml.py \\
      --gguf models/parler-mini-v1-f32.gguf --parity-dir artifacts/parler-ref --no-export
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ARCH = "parler"
QUANT_TAGS = ("f32", "f16", "bf16", "q8_0", "q5_0", "q4_0")
SIDECAR_SUFFIX = "-decoder"
CONTRACT_VERSION = 2
# additive mask for switched-off rows; fp16-safe, exp(MASK_OFF - max) == 0
MASK_OFF = -3.0e4
DEFAULT_CTX = 1024
DEFAULT_CROSS = 128
# the generation cap (parler.gen.max_length) plus the prompt is the longest
# context an utterance can need; the engine must not ask for more than ctx
WEIGHT_MODES = ("fp16", "int8", "int8-b32", "pal8", "pal6", "pal4", "int4")
PARITY_LOGIT_ATOL = 2e-3
STATE_KINDS = ("self_k", "self_v", "cross_k", "cross_v")


def read_tensor(tensor):
    from gguf.quants import dequantize

    shape = [int(d) for d in reversed(tensor.shape)]
    if tensor.tensor_type.name in ("F32", "F16"):
        return np.asarray(tensor.data).astype(np.float32).reshape(shape)
    return dequantize(np.asarray(tensor.data), tensor.tensor_type).astype(np.float32).reshape(shape)


def read_gguf(path):
    """Decoder tensors (torch layouts: Linear [out, in]) and hyper-parameters."""
    from gguf import GGUFReader

    reader = GGUFReader(str(path))
    if reader.fields["general.architecture"].contents() != ARCH:
        raise ValueError(f"{path} is not a {ARCH} GGUF")

    def field(name):
        return reader.fields[f"{ARCH}.{name}"].contents()

    hp = {
        "n_layer": int(field("dec.n_layer")),
        "d_model": int(field("dec.d_model")),
        "n_head": int(field("dec.n_head")),
        "d_ff": int(field("dec.d_ff")),
        "n_q": int(field("dec.n_codebooks")),
        "vocab": int(field("dec.vocab_size")),
        "ln_eps": float(field("dec.ln_eps")),
        "max_position": int(field("dec.max_position")),
        "max_length": int(field("gen.max_length")),
        "bos": int(field("dec.bos_token_id")),
        "variant": str(field("variant")) if f"{ARCH}.variant" in reader.fields else Path(path).stem,
    }
    tensors = {t.name: read_tensor(t) for t in reader.tensors if t.name.startswith("dec.")}
    return tensors, hp


def strip_quant_tag(stem):
    for sep in ("-", "."):
        head, _, tag = stem.rpartition(sep)
        if head and tag.lower() in QUANT_TAGS:
            return head
    return stem


def sidecar_stem(gguf_path):
    return strip_quant_tag(Path(gguf_path).stem) + SIDECAR_SUFFIX


def state_name(kind, group):
    return f"{kind}_{group:02d}"


def constant(array):
    return nn.Parameter(torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32)),
                        requires_grad=False)


class Proj(nn.Module):
    """A bias-free projection on [1, C, 1, 1] as a 1x1 conv2d (Linear [out, in])."""

    def __init__(self, weight, scale=1.0):
        super().__init__()
        self.weight = constant(np.asarray(weight, np.float32)[:, :, None, None] * scale)

    def forward(self, x):
        return F.conv2d(x, self.weight)


class ChannelNorm(nn.Module):
    """Channel LayerNorm via a stable last-axis Core ML operation."""

    def __init__(self, weight, bias, eps):
        super().__init__()
        self.channels = int(np.asarray(weight).size)
        self.weight = constant(np.asarray(weight).reshape(-1))
        self.bias = constant(np.asarray(bias).reshape(-1))
        self.eps = eps

    def forward(self, x):
        # Batch/token dimensions are fixed at one, so these reshapes move no
        # data. Channel-axis arithmetic is mis-fused on macOS 26; LayerNorm on
        # the last axis also avoids overflowing an explicit fp16 variance.
        y = F.layer_norm(x.reshape(1, 1, 1, self.channels), (self.channels,),
                         self.weight, self.bias, self.eps)
        return y.reshape(1, self.channels, 1, 1)


class Layer(nn.Module):
    def __init__(self, t, hp, layer):
        super().__init__()
        p = f"dec.blk.{layer}."
        eps, hd = hp["ln_eps"], hp["d_model"] // hp["n_head"]
        self.self_norm = ChannelNorm(t[p + "attn_norm.weight"], t[p + "attn_norm.bias"], eps)
        self.q = Proj(t[p + "attn_q.weight"], hd ** -0.5)
        self.k = Proj(t[p + "attn_k.weight"])
        self.v = Proj(t[p + "attn_v.weight"])
        self.o = Proj(t[p + "attn_o.weight"])
        self.cross_norm = ChannelNorm(t[p + "cross_norm.weight"], t[p + "cross_norm.bias"], eps)
        self.cross_q = Proj(t[p + "cross_q.weight"], hd ** -0.5)
        self.cross_o = Proj(t[p + "cross_o.weight"])
        self.ffn_norm = ChannelNorm(t[p + "ffn_norm.weight"], t[p + "ffn_norm.bias"], eps)
        self.up = Proj(t[p + "ffn_up.weight"])
        self.down = Proj(t[p + "ffn_down.weight"])


def heads_split(x, h, hd):
    """[1, H * HD, 1, 1] -> [1, H, 1, HD]"""
    return x.reshape(1, h, hd, 1).permute(0, 1, 3, 2)


def heads_merge(x, d):
    """[1, H, 1, HD] -> [1, H * HD, 1, 1]"""
    return x.permute(0, 1, 3, 2).reshape(1, d, 1, 1)


class DecoderStep(nn.Module):
    """One decoder position over read-only per-layer K / V states."""

    def __init__(self, t, hp, ctx, cross, state_dtype=torch.float16, layers_per_state=1, k_layout="rows"):
        super().__init__()
        self.k_cols = k_layout == "cols"
        if hp["n_layer"] % layers_per_state:
            raise ValueError(f"--layers-per-state {layers_per_state} does not divide {hp['n_layer']} layers")
        self.per_state = layers_per_state
        self.L, self.d, self.h = hp["n_layer"], hp["d_model"], hp["n_head"]
        self.hd, self.n_q, self.vocab = self.d // self.h, hp["n_q"], hp["vocab"]
        self.ctx, self.cross = ctx, cross
        self.layers = nn.ModuleList(Layer(t, hp, l) for l in range(self.L))
        self.out_norm = ChannelNorm(t["dec.output_norm.weight"], t["dec.output_norm.bias"], hp["ln_eps"])
        self.heads = Proj(np.concatenate([t[f"dec.lm_heads.{k}.weight"] for k in range(self.n_q)], 0))
        for g in range(self.L // layers_per_state):
            for kind in STATE_KINDS:
                rows = ctx if kind.startswith("self") else cross
                inner = (self.hd, rows) if self.k_cols and kind.endswith("_k") else (rows, self.hd)
                self.register_buffer(state_name(kind, g),
                                     torch.zeros(layers_per_state, self.h, *inner, dtype=state_dtype))

    def state_slot(self, kind, layer):
        """The state tensor holding `layer` and the layer's index in it."""
        return getattr(self, state_name(kind, layer // self.per_state)), layer % self.per_state

    def keys(self, kind, layer):
        """K as [1, H, HD, rows]: stored that way (cols) or transposed from rows."""
        k = self.state(kind, layer)
        return k if self.k_cols else k.transpose(-1, -2)

    def state(self, kind, layer):
        buf, i = self.state_slot(kind, layer)
        return (buf if self.per_state == 1 else buf[i:i + 1]).float()

    def forward(self, hidden, self_mask, cross_mask):
        x = hidden
        new_k, new_v = [], []
        for l, m in enumerate(self.layers):
            y = m.self_norm(x)
            q = heads_split(m.q(y), self.h, self.hd)
            k = heads_split(m.k(y), self.h, self.hd)
            v = heads_split(m.v(y), self.h, self.hd)
            new_k.append(k)
            new_v.append(v)
            # cached rows < n_past as one score vector, the current position
            # as one more column: no state write in the graph
            past = torch.matmul(q, self.keys("self_k", l)) + self_mask                     # [1,H,1,ctx]
            cur = (q * k).sum(-1, keepdim=True)                                             # [1,H,1,1]
            w = torch.softmax(torch.cat([past, cur], -1), dim=-1)
            a = torch.matmul(w[..., :self.ctx], self.state("self_v", l)) + w[..., self.ctx:] * v
            x = x + m.o(heads_merge(a, self.d))

            q = heads_split(m.cross_q(m.cross_norm(x)), self.h, self.hd)
            s = torch.matmul(q, self.keys("cross_k", l)) + cross_mask
            a = torch.matmul(torch.softmax(s, dim=-1), self.state("cross_v", l))
            x = x + m.cross_o(heads_merge(a, self.d))

            x = x + m.down(F.gelu(m.up(m.ffn_norm(x))))
        logits = self.heads(self.out_norm(x)).reshape(self.n_q, self.vocab)
        return logits, torch.cat(new_k, 0), torch.cat(new_v, 0)


# ---------------------------------------------------------------- host side


def cross_kv(t, hp, cross_states, cross):
    """Per-layer cross K / V [1, H, cross, HD] of the description and its mask."""
    n = cross_states.shape[0]
    if n > cross:
        raise ValueError(f"description has {n} tokens; the package takes at most --cross-context {cross}")
    h, hd = hp["n_head"], hp["d_model"] // hp["n_head"]
    ks, vs = [], []
    for l in range(hp["n_layer"]):
        k = np.zeros((1, h, cross, hd), np.float32)
        v = np.zeros((1, h, cross, hd), np.float32)
        k[0, :, :n] = (cross_states @ t[f"dec.blk.{l}.cross_k.weight"].T).reshape(n, h, hd).transpose(1, 0, 2)
        v[0, :, :n] = (cross_states @ t[f"dec.blk.{l}.cross_v.weight"].T).reshape(n, h, hd).transpose(1, 0, 2)
        ks.append(k)
        vs.append(v)
    mask = np.full((1, 1, 1, cross), MASK_OFF, np.float32)
    mask[..., :n] = 0
    return ks, vs, mask


def self_mask(n_past, ctx):
    mask = np.full((1, 1, 1, ctx), MASK_OFF, np.float32)
    mask[..., :n_past] = 0
    return mask


class TorchSession:
    """Drives DecoderStep like the engine drives the package: zeroed states,
    one row written per call."""

    def __init__(self, model):
        self.model = model
        for buf in model.buffers():
            buf.zero_()

    def set_cross(self, ks, vs):
        for l, (k, v) in enumerate(zip(ks, vs)):
            for kind, a in (("cross_k", k), ("cross_v", v)):
                buf, i = self.model.state_slot(kind, l)
                a = a[0].transpose(0, 2, 1) if self.model.k_cols and kind == "cross_k" else a[0]
                buf[i].copy_(torch.from_numpy(np.ascontiguousarray(a)))

    def step(self, hidden, n_past, cross_mask):
        m = self.model
        with torch.no_grad():
            logits, nk, nv = m(torch.from_numpy(hidden.reshape(1, -1, 1, 1).astype(np.float32)),
                               torch.from_numpy(self_mask(n_past, m.ctx)), torch.from_numpy(cross_mask))
            for l in range(m.L):
                for kind, a in (("self_k", nk), ("self_v", nv)):
                    buf, i = m.state_slot(kind, l)
                    if m.k_cols and kind == "self_k":
                        buf[i, :, :, n_past] = a[l, :, 0]
                    else:
                        buf[i, :, n_past] = a[l, :, 0]
        return logits.numpy()


def teacher_inputs(t, hp, ref, case, n_frames):
    """Hidden rows the engine feeds for a dump-reference.py case: the prompt
    tokens, then the BOS frame and the greedy delayed trace (codebook k is BOS
    for the first k + 1 frames)."""
    prompt_ids = np.load(Path(ref) / f"{case}_prompt_ids.npy")[0]
    trace = np.load(Path(ref) / f"{case}_greedy_delayed.npy")
    pos = t["dec.embed_positions.weight"]
    emb = [t[f"dec.embed_tokens.{k}.weight"] for k in range(hp["n_q"])]
    prompt = t["dec.embed_prompts.weight"]
    rows = [prompt[i] + pos[j] for j, i in enumerate(prompt_ids)]
    bos = hp["bos"]

    def fed(k, f):
        return bos if f <= k else trace[k, f]
    rows += [sum(emb[k][fed(k, f)] for k in range(hp["n_q"])) + pos[len(prompt_ids) + f]
             for f in range(n_frames)]
    return rows, len(prompt_ids), trace


def run_parity(model, t, hp, ref, cases=("case0", "case1")):
    """Teacher-forced rebuild vs the HF raw logits of the greedy run (build the
    model with fp32 states for the tolerance check; fp16 states, the package's,
    add ~5e-3 of cache rounding)."""
    cases = [case for case in cases if (Path(ref) / f"{case}_step_logits.npy").exists()]
    if not cases:
        raise ValueError(f"[parity] no matching case step-logit files in {ref}")
    worst = 0.0
    for case in cases:
        want = np.load(Path(ref) / f"{case}_step_logits.npy")
        rows, n_prompt, _ = teacher_inputs(t, hp, ref, case, want.shape[0])
        if len(rows) > model.ctx:
            raise ValueError(f"{case} needs ctx {len(rows)} > {model.ctx}")
        ks, vs, cmask = cross_kv(t, hp, np.load(Path(ref) / f"{case}_cross_states.npy")[0], model.cross)
        session = TorchSession(model)
        session.set_cross(ks, vs)
        got = [session.step(r, n, cmask) for n, r in enumerate(rows)][n_prompt:]
        err = max(float(np.abs(g - w).max()) for g, w in zip(got, want))
        agree = sum(int((g.argmax(-1) == w.argmax(-1)).sum()) for g, w in zip(got, want))
        print(f"[parity] {case}: {len(want)} steps, max |logit err| {err:.2e}, "
              f"argmax {agree}/{want.shape[0] * want.shape[1]}")
        worst = max(worst, err)
    if worst > PARITY_LOGIT_ATOL:
        raise SystemExit(f"[parity] rebuild differs from HF by {worst:.2e} > {PARITY_LOGIT_ATOL}")


# ---------------------------------------------------------------- Core ML


def io_spec(model):
    inputs = {"hidden": (1, model.d, 1, 1), "self_mask": (1, 1, 1, model.ctx),
              "cross_mask": (1, 1, 1, model.cross)}
    states = {state_name(kind, g): tuple(getattr(model, state_name(kind, g)).shape)
              for g in range(model.L // model.per_state) for kind in STATE_KINDS}
    outputs = {"logits": (model.n_q, model.vocab), "new_k": (model.L, model.h, 1, model.hd),
               "new_v": (model.L, model.h, 1, model.hd)}
    return inputs, states, outputs


def contract(model, hp, weights, gguf):
    inputs, states, outputs = io_spec(model)
    return {
        "contract": CONTRACT_VERSION,
        "kind": "parler-decoder-step",
        "normalization": "last-axis",
        "gguf": Path(gguf).name if gguf else None,
        "variant": hp.get("variant"),
        "n_layer": model.L, "d_model": model.d, "n_head": model.h, "head_dim": model.hd,
        "n_q": model.n_q, "vocab": model.vocab, "ctx": model.ctx, "cross_context": model.cross,
        "weights": weights, "mask_off": MASK_OFF,
        "inputs": {k: list(v) for k, v in inputs.items()},
        "states": {"self_k": "self_k_%02d", "self_v": "self_v_%02d", "cross_k": "cross_k_%02d",
                   "cross_v": "cross_v_%02d", "layers_per_state": model.per_state,
                   "k_layout": "cols" if model.k_cols else "rows",
                   "layout": "V: [layers_per_state, H, rows, HD]; K: same (k_layout rows) or "
                             "[layers_per_state, H, HD, rows] (cols); fp16; layer l is index "
                             "l % layers_per_state of state <kind>_<l // layers_per_state>",
                   "self_rows": model.ctx, "cross_rows": model.cross,
                   "shapes": {k: list(v) for k, v in states.items()}},
        "outputs": {k: list(v) for k, v in outputs.items()},
        "host_step": "write new_k[l, h, 0, :] / new_v[l, h, 0, :] to self_k_gg / self_v_gg "
                     "[l % layers_per_state, h, n_past, :] (gg = l // layers_per_state) after each call; "
                     "zero every state at session start",
    }


def convert_coreml(model, weights, heads_fp16, meta, ffn_down_fp16=False):
    if ffn_down_fp16 and weights == "fp16":
        raise ValueError("--ffn-down-fp16 requires compressed --weights (not fp16)")
    import coremltools as ct

    inputs, states, outputs = io_spec(model)
    with torch.no_grad():
        traced = torch.jit.trace(model, tuple(torch.zeros(s) for s in inputs.values()))
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name=n, shape=s, dtype=np.float16) for n, s in inputs.items()],
        states=[ct.StateType(wrapped_type=ct.TensorType(shape=s, dtype=np.float16), name=n)
                for n, s in states.items()],
        outputs=[ct.TensorType(name=n, dtype=np.float16) for n in outputs],
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        convert_to="mlprogram",
    )
    check_states_read_only(mlmodel, states)
    down = ffn_down_projections(mlmodel, model) if ffn_down_fp16 else {}
    if ffn_down_fp16:
        meta = dict(meta, weights=meta["weights"] + "+fp16-ffn-down",
                    experimental_precision={
                        "ffn_down_fp16": True, "heads_fp16": bool(heads_fp16),
                        "ffn_down_projections": [
                            {"tensor": tensor, "operation": op.name, "dtype": "fp16",
                             "shape": list(op.weight.shape)} for tensor, op in down.items()],
                    })
    mlmodel = compress(mlmodel, weights, heads_fp16, model.n_q * model.vocab, down)
    mlmodel.user_defined_metadata["parler.decoder.contract"] = json.dumps(meta, sort_keys=True)
    mlmodel.short_description = (f"Parler-TTS decoder step ({meta.get('variant')}), ctx {model.ctx}, "
                                 f"cross {model.cross}, weights {meta['weights'] if ffn_down_fp16 else weights}")
    return mlmodel


def mil_ops(mlmodel):
    return list(mlmodel._mil_program.functions["main"].operations)


def check_states_read_only(mlmodel, states):
    """Every state must be read (a dropped read would mean the program ignores
    the host's cache) and none written."""
    read = set()
    for op in mlmodel._mil_program.functions["main"].find_ops(op_type="read_state"):
        read.add(op.inputs["input"].name)
    written = [op.name for op in mlmodel._mil_program.functions["main"].find_ops(op_type="coreml_update_state")]
    missing = sorted(set(states) - read)
    if missing or written:
        raise RuntimeError(f"state contract broken: unread {missing[:4]}... written {written[:4]}")


def head_projections(mlmodel, heads_rows):
    """Find the fused codebook conv by weight shape, independent of MIL names."""
    heads = [op for op in mlmodel._mil_program.functions["main"].find_ops(op_type="conv")
             if len(op.weight.shape) == 4 and op.weight.shape[0] == heads_rows
             and tuple(op.weight.shape[2:]) == (1, 1)]
    if not heads:
        raise RuntimeError("--heads-fp16: no matching codebook head projection ops")
    return heads


def head_weight_constants(var, option="--heads-fp16"):
    """Follow weight producers through casts, failing closed on compressed weights."""
    op = var.op
    if op is None or op.op_type.startswith("constexpr_"):
        raise RuntimeError(f"{option}: weight is constexpr quantized or has no constant producer")
    if op.op_type == "const":
        return {op.name}
    if op.op_type == "cast":
        return head_weight_constants(op.x, option)
    raise RuntimeError(f"{option}: unsupported weight producer {op.op_type}")


def fp16_projection_constants(op, option):
    names = head_weight_constants(op.weight, option)
    var = op.weight
    while True:
        if var.dtype.__name__ != "fp16":
            raise RuntimeError(f"{option}: {op.name} weight/constant is not fp16")
        if var.op.op_type == "const":
            return names
        var = var.op.x  # head_weight_constants already checked this cast chain


def ffn_down_projections(mlmodel, model):
    """Map source layers to down convs using module semantics, order and values.

    MIL constant/conv names change during conversion. TorchScript module
    scopes retain `down`; checking the fp16 source tensors in layer order
    prevents a same-shaped unrelated projection from silently matching.
    """
    from coremltools.converters.mil.mil.scope import ScopeSource

    convs = mlmodel._mil_program.functions["main"].find_ops(op_type="conv")
    down = [op for op in convs
            if op.scopes.get(ScopeSource.TORCHSCRIPT_MODULE_NAME, [])[:1] == ["down"]]
    if len(down) != model.L or len({op.name for op in down}) != model.L:
        raise RuntimeError(f"--ffn-down-fp16: expected exactly {model.L} unambiguous down projections, "
                           f"found {len(down)}")
    selected = {}
    for layer, op in enumerate(down):
        tensor = f"dec.blk.{layer}.ffn_down.weight"
        source = model.layers[layer].down.weight.detach().cpu().numpy().astype(np.float16)
        shape = (model.d, model.layers[layer].up.weight.shape[0], 1, 1)
        scopes = op.scopes.get(ScopeSource.TORCHSCRIPT_MODULE_TYPE, [])
        if scopes[:1] != ["Proj"] or tuple(op.weight.shape) != shape or source.shape != shape:
            raise RuntimeError(f"--ffn-down-fp16: {tensor} has unexpected module type or shape")
        fp16_projection_constants(op, "--ffn-down-fp16")
        if not np.array_equal(op.weight.val, source):
            raise RuntimeError(f"--ffn-down-fp16: {tensor} does not match its source layer")
        selected[tensor] = op
    return selected


def compress(mlmodel, weights, heads_fp16, heads_rows, ffn_down=None):
    ffn_down = ffn_down or {}
    head_names, op_names = set(), {}
    if heads_fp16:
        for head in head_projections(mlmodel, heads_rows):
            head_names.add(head.name)
            op_names.update({name: None for name in head_weight_constants(head.weight)})
    for down in ffn_down.values():
        op_names.update({name: None for name in fp16_projection_constants(down, "--ffn-down-fp16")})
    if ffn_down:
        if weights == "fp16":
            raise ValueError("--ffn-down-fp16 requires compressed --weights (not fp16)")
        projections = mlmodel._mil_program.functions["main"].find_ops(op_type="conv")
        convs = {op.name: op for op in projections}
        if len(convs) != len(projections):
            raise RuntimeError("--ffn-down-fp16: ambiguous projection operation names")
        selected_names = {op.name for op in ffn_down.values()}
        retained = {name: (tuple(convs[name].weight.shape), convs[name].weight.val.copy())
                    for name in selected_names | head_names}
        all_head_names = {op.name for op in head_projections(mlmodel, heads_rows)}
    if weights == "fp16":
        return mlmodel
    import coremltools.optimize.coreml as cto

    if weights == "int8-b32":
        # Match Q8_0's local weight-block size without claiming bitwise
        # equivalence: Core ML quantizes fp16 constants and rounds differently.
        # Placement and native quality must be measured for this separate tier.
        op = cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8",
                                         granularity="per_block", block_size=32)
    elif weights == "int8":
        op = cto.OpLinearQuantizerConfig(mode="linear_symmetric", granularity="per_channel")
    elif weights == "int4":
        op = cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int4",
                                         granularity="per_block", block_size=32)
    else:
        # uniform LUTs: seconds, where k-means on 350M weights takes hours
        op = cto.OpPalettizerConfig(mode="uniform", nbits=int(weights[3:]),
                                    granularity="per_grouped_channel", group_size=16)
    if ffn_down:
        # Use the compressor's own cutoff. Below-cutoff weights (including
        # norms) stay unchanged. Heads retain their existing mode-specific
        # behavior unless --heads-fp16 explicitly excludes them.
        quantized_names = {name for name, projection in convs.items()
                           if name not in (selected_names | all_head_names)
                           and np.prod(projection.weight.shape) > op.weight_threshold}
    config = cto.OptimizationConfig(global_config=op, op_name_configs=op_names)
    if weights in ("int8", "int8-b32", "int4"):
        compressed = cto.linear_quantize_weights(mlmodel, config)
    else:
        compressed = cto.palettize_weights(mlmodel, config)
    if heads_fp16:
        heads = head_projections(compressed, heads_rows)
        if {head.name for head in heads} != head_names:
            raise RuntimeError("--heads-fp16: head projection ops changed during compression")
        for head in heads:
            head_weight_constants(head.weight)
    if ffn_down:
        projections = compressed._mil_program.functions["main"].find_ops(op_type="conv")
        after = {op.name: op for op in projections}
        if len(projections) != len(convs) or set(after) != set(convs):
            raise RuntimeError("--ffn-down-fp16: projection operations changed during compression")
        for name in sorted(selected_names | head_names):
            shape, value = retained[name]
            if tuple(after[name].weight.shape) != shape:
                raise RuntimeError(f"--ffn-down-fp16: {name} weight shape changed during compression")
            fp16_projection_constants(after[name], "--ffn-down-fp16")
            if not np.array_equal(after[name].weight.val, value):
                raise RuntimeError(f"--ffn-down-fp16: {name} fp16 weight changed during compression")
        for name in sorted(quantized_names):
            producer = after[name].weight.op
            while producer is not None and producer.op_type == "cast":
                producer = producer.x.op
            if producer is None or not producer.op_type.startswith("constexpr_"):
                raise RuntimeError(f"--ffn-down-fp16: unselected projection {name} was not compressed")
    return compressed


def compile_model(package_path, compile_dir):
    Path(compile_dir).mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["xcrun", "coremlcompiler", "compile", str(package_path), str(compile_dir)],
                       capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"coremlcompiler failed (exit {exc.returncode}):\n{exc.stderr}") from exc
    return Path(compile_dir) / (Path(package_path).stem + ".mlmodelc")


def report_placement(compiled, units="ALL"):
    import coremltools as ct

    plan = ct.models.compute_plan.MLComputePlan.load_from_path(
        str(compiled), compute_units=getattr(ct.ComputeUnit, units))
    counts, cost, off_ane = {}, {}, {}
    for op in plan.model_structure.program.functions["main"].block.operations:
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        if usage is None:
            continue
        device = type(usage.preferred_compute_device).__name__
        counts[device] = counts.get(device, 0) + 1
        if "NeuralEngine" not in device:
            off_ane[op.operator_name] = off_ane.get(op.operator_name, 0) + 1
        estimate = plan.get_estimated_cost_for_mlprogram_operation(op)
        if estimate is not None:
            cost[device] = cost.get(device, 0.0) + estimate.weight
    total = sum(cost.values()) or 1.0
    share = {d: f"{100.0 * w / total:.1f}%" for d, w in cost.items()}
    print(f"[placement] {units}: ops {counts} estimated cost {share} off-ANE ops {off_ane}")
    return counts, share, off_ane


class CoremlSession:
    """The package driven like TorchSession. coremltools only writes whole
    states, so this rewrites each written layer state per call: correct, but
    not representative of the engine's per-row write cost."""

    def __init__(self, mlmodel, ctx, n_layer, per_state=1, k_cols=False):
        self.m, self.ctx, self.L, self.per_state, self.k_cols = mlmodel, ctx, n_layer, per_state, k_cols
        self.state = mlmodel.make_state()
        spec = {s.name: tuple(s.type.stateType.arrayType.shape) for s in mlmodel.get_spec().description.state}
        self.host = {n: np.zeros(s, np.float32) for n, s in spec.items()}  # write_state takes fp32
        for n, a in self.host.items():
            self.state.write_state(n, a)

    def set_cross(self, ks, vs):
        for l, (k, v) in enumerate(zip(ks, vs)):
            for kind, a in (("cross_k", k), ("cross_v", v)):
                name = state_name(kind, l // self.per_state)
                self.host[name][l % self.per_state] = (a[0].transpose(0, 2, 1)
                                                       if self.k_cols and kind == "cross_k" else a[0])
                self.state.write_state(name, self.host[name])

    def step(self, hidden, n_past, cross_mask):
        out = self.m.predict({"hidden": hidden.reshape(1, -1, 1, 1).astype(np.float16),
                              "self_mask": self_mask(n_past, self.ctx).astype(np.float16),
                              "cross_mask": cross_mask.astype(np.float16)}, state=self.state)
        nk, nv = np.asarray(out["new_k"]), np.asarray(out["new_v"])
        for l in range(self.L):
            for kind, a in (("self_k", nk), ("self_v", nv)):
                buf = self.host[state_name(kind, l // self.per_state)][l % self.per_state]
                if self.k_cols and kind == "self_k":
                    buf[:, :, n_past] = a[l, :, 0]
                else:
                    buf[:, n_past] = a[l, :, 0]
        for name in self.host:
            if name.startswith("self"):
                self.state.write_state(name, self.host[name])
        return np.asarray(out["logits"], np.float32)


def teacher_agreement(make_session, t, hp, ref, case, steps, ctx, cross, want_argmax=None):
    """Argmax agreement over `steps` frames once every codebook is live (frame
    >= n_q), against the greedy trace (or another run's argmaxes)."""
    rows, n_prompt, trace = teacher_inputs(t, hp, ref, case, min(steps + hp["n_q"] + 1, trace_len(ref, case) - 1))
    rows = rows[:ctx]
    ks, vs, cmask = cross_kv(t, hp, np.load(Path(ref) / f"{case}_cross_states.npy")[0], cross)
    session = make_session()
    session.set_cross(ks, vs)
    argmaxes, agree, total = [], 0, 0
    for n, r in enumerate(rows):
        logits = session.step(r, n, cmask)
        f = n - n_prompt
        if f >= hp["n_q"] and f + 1 < trace.shape[1]:
            am = logits.argmax(-1)
            want = want_argmax[len(argmaxes)] if want_argmax is not None else trace[:, f + 1]
            argmaxes.append(am)
            agree += int((am == want).sum())
            total += am.size
    return agree, total, np.array(argmaxes)


def trace_len(ref, case):
    return np.load(Path(ref) / f"{case}_greedy_delayed.npy").shape[1]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Optional post-export QKV fusion of an existing tiny/mini int8+fp16-heads rows "
               "package: run optimize-decoder-coreml.py --help. It preserves original head math "
               "and appends raw codes/scales (measured compiled disk size +20.2%); validate "
               "parity and speed for your compute units. Export defaults are unchanged.")
    p.add_argument("--gguf", required=True, help="a Parler-TTS GGUF (any quantization)")
    p.add_argument("--ctx", type=int, default=DEFAULT_CTX,
                   help="self-attention cache rows: prompt tokens + generated frames (default %(default)s)")
    p.add_argument("--cross-context", type=int, default=DEFAULT_CROSS,
                   help="description tokens the cross states hold (default %(default)s)")
    p.add_argument("--weights", default="int8", choices=WEIGHT_MODES,
                   help="weight storage: fp16, int8 per channel, experimental int8-b32 blocks, "
                        "uniform palettes (pal*), int4 blocks; block modes need placement/quality validation")
    p.add_argument("--heads-fp16", action="store_true", help="keep the nine codebook heads fp16")
    p.add_argument("--ffn-down-fp16", action="store_true",
                   help="experimental: keep every FFN down projection fp16 in a compressed graph; "
                        "composes with --heads-fp16; placement, speed and quality need validation")
    p.add_argument("--layers", type=int, help="export only the first N layers (benchmarking)")
    p.add_argument("--k-layout", default="rows", choices=["rows", "cols"],
                   help="K states as [.., rows, HD]; cols is experimental and unsupported by the native runtime")
    p.add_argument("--layers-per-state", type=int, default=0,
                   help="decoder layers packed into each K / V state, [N, H, rows, HD]; 0 (default) stacks "
                        "all layers into four states -- the M2 Neural Engine rejects the full model with "
                        "per-layer states (plan error -14 from 13 layers up)")
    p.add_argument("--out", help=".mlpackage path (default: <stem>-decoder.mlpackage next to the GGUF, where "
                                 "the engine looks for <stem>-decoder.mlmodelc; ctx lives in the metadata)")
    p.add_argument("--compile-dir", help="also compile to <dir>/<stem>.mlmodelc")
    p.add_argument("--parity-dir", help="dump-reference.py output to check the rebuild against")
    p.add_argument("--quality-steps", type=int, default=0,
                   help="after export, teacher-force this many frames of case0 through the package "
                        "(CPU_AND_NE) and the fp32 rebuild and report argmax agreement")
    p.add_argument("--no-export", action="store_true", help="only run the parity check")
    args = p.parse_args(argv)
    if args.ffn_down_fp16 and args.weights == "fp16":
        p.error("--ffn-down-fp16 requires compressed --weights (not fp16)")
    if args.ffn_down_fp16 and args.no_export:
        p.error("--ffn-down-fp16 requires export; remove --no-export")
    if args.layers is not None and args.layers <= 0:
        p.error("--layers must be positive")
    if args.layers is not None and args.parity_dir:
        p.error("--layers cannot be combined with --parity-dir (parity requires the full decoder)")
    if args.layers is not None and not args.out:
        p.error("--layers exports a truncated benchmark model; name it with --out")
    if args.quality_steps < 0:
        p.error("--quality-steps must be nonnegative")
    if args.quality_steps and not args.parity_dir:
        p.error("--quality-steps requires --parity-dir")
    if args.no_export and not args.parity_dir:
        p.error("--no-export requires --parity-dir")
    if args.no_export and args.quality_steps:
        p.error("--quality-steps requires export; remove --no-export")
    return args


def main():
    args = parse_args()
    t, hp = read_gguf(args.gguf)
    if args.layers:
        if args.layers > hp["n_layer"]:
            raise SystemExit(f"--layers {args.layers} exceeds the decoder's {hp['n_layer']} layers")
        hp["n_layer"] = args.layers
    if args.layers_per_state == 0:
        args.layers_per_state = hp["n_layer"]
    need = hp["max_length"]
    if not 0 < args.ctx <= hp["max_position"]:
        raise SystemExit(f"--ctx {args.ctx} outside 1..{hp['max_position']} (the position table)")
    print(f"[load] decoder from {args.gguf}: {hp['n_layer']} layers d {hp['d_model']} heads {hp['n_head']}; "
          f"ctx {args.ctx} (an uncapped utterance needs prompt + {need}), cross {args.cross_context}")
    if args.parity_dir:
        run_parity(DecoderStep(t, hp, max(args.ctx, 256), args.cross_context, torch.float32,
                               args.layers_per_state, args.k_layout).eval(),
                   t, hp, args.parity_dir)
    model = DecoderStep(t, hp, args.ctx, args.cross_context, torch.float16, args.layers_per_state,
                        args.k_layout).eval()
    if args.no_export:
        return
    out = Path(args.out) if args.out else Path(args.gguf).with_name(
        f"{sidecar_stem(args.gguf)}.mlpackage")
    meta = contract(model, hp, args.weights + ("+fp16-heads" if args.heads_fp16 else ""), args.gguf)
    start = time.time()
    mlmodel = convert_coreml(model, args.weights, args.heads_fp16, meta, args.ffn_down_fp16)
    mlmodel.save(str(out))
    print(f"[export] saved {out} in {time.time() - start:.1f}s")
    if args.compile_dir:
        start = time.time()
        compiled = compile_model(out, args.compile_dir)
        print(f"[export] compiled {compiled} in {time.time() - start:.1f}s")
        report_placement(compiled)
    if args.quality_steps and args.parity_dir:
        import coremltools as ct

        ref_model = DecoderStep(t, hp, args.ctx, args.cross_context, torch.float32).eval()
        ag, tot, twin = teacher_agreement(lambda: TorchSession(ref_model), t, hp, args.parity_dir, "case0",
                                          args.quality_steps, args.ctx, args.cross_context)
        print(f"[quality] fp32 rebuild vs greedy trace: {ag}/{tot} = {100 * ag / tot:.2f}%")
        start = time.time()
        ml = ct.models.MLModel(str(out), compute_units=ct.ComputeUnit.CPU_AND_NE)
        print(f"[quality] CPU_AND_NE load {time.time() - start:.1f}s")
        ag, tot, _ = teacher_agreement(lambda: CoremlSession(ml, args.ctx, hp["n_layer"], args.layers_per_state,
                                                                 args.k_layout == "cols"), t, hp,
                                       args.parity_dir, "case0", args.quality_steps, args.ctx,
                                       args.cross_context, want_argmax=twin)
        print(f"[quality] {args.weights} package vs fp32 rebuild: {ag}/{tot} = {100 * ag / tot:.2f}%")


if __name__ == "__main__":
    main()
