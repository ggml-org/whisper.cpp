#!/usr/bin/env python3
"""Export the offline FastConformer encoder from a parakeet GGUF to a Core ML
package laid out for the Apple Neural Engine.

The exported graph:

  - uses a channel-first [1, C, 1, T] layout, with every linear projection
    as a 1x1 conv;
  - has a fixed window length (--window-mel-frames); the engine's long-form
    plan tiles longer audio across overlapping windows, and the engine's
    independent-segment path cuts audio into windows of at most this length;
  - takes runtime masks: `mel_mask` zeroes padded mel frames and, through
    its even-index slices, the padded frames at every subsampling
    resolution (the valid length after a stride-2 conv is exactly the
    number of even indices below the previous one), `pad_mask` zeroes
    invalid encoder frames and `key_bias` masks invalid attention keys, so
    a partially filled window uses the same graph and padding never reaches
    a valid frame through a convolution kernel;
  - stores palettized weights (6-bit by default; --nbits 0 for fp16).

The output approximates the ggml encoder but is not bit-identical (fp16
throughout, palettized weights, different accumulation order).

I/O contract (see src/coreml/parakeet-encoder.h):

    mel          fp16 [1, n_mels, 1, window_mel_frames]
    mel_mask     fp16 [1, 1, 1, window_mel_frames]  1.0 valid, 0.0 pad
    pad_mask     fp16 [1, 1, 1, enc_frames]     1.0 valid, 0.0 pad
    key_bias     fp16 [1, enc_frames, 1, 1]     0.0 valid, -1e4 pad
    encoder_out  fp16 [1, d_model, 1, enc_frames]

The engine fills `mel_mask` when the export declares it; exports without it
(the published TranscribePsy model) still load and mask after subsampling only.

Example:

  python engines/parakeet/scripts/export-encoder-coreml-ane.py \
      --gguf engines/parakeet/models/transcribepsy-250m-eu-sub.gguf \
      --out  engines/parakeet/models/transcribepsy-250m-eu-sub-encoder.mlpackage \
      --compile-dir engines/parakeet/models

Needs the pinned scripts/requirements-coreml.txt environment and macOS for
--compile-dir. Unlike export-encoder-coreml.py, this layout is specific to
the Neural Engine: fully ANE-resident under CPU_AND_NE on the published
TranscribePsy export, where the generic single-input export keeps a few
operations on the GPU.
"""

import argparse
import importlib.util
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import coremltools as ct

# 15 s at hop 160, matching the subsampling stack's 8x decimation to 188 frames.
DEFAULT_WINDOW_MEL_FRAMES = 1500

# Additive mask for invalid attention keys. fp16 -inf would poison the softmax
# denominator for an all-pad row; -1e4 saturates the softmax just as hard while
# staying representable.
KEY_BIAS_PAD = -1.0e4

# Palettization group size along the output-channel axis, in channels per LUT.
PALETTIZE_GROUP_SIZE = 16


def load_reference_encoder(scripts_dir):
    path = scripts_dir / "ref-encoder-from-gguf.py"
    spec = importlib.util.spec_from_file_location("parakeet_ref_encoder", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def conv_out_len(length, k=3, s=2, p=1):
    return (length + 2 * p - k) // s + 1


def encoder_frames_for_mel(n_mel_frames, causal):
    def nxt(length):
        return (length // 2 + 1) if causal else conv_out_len(length)

    return nxt(nxt(nxt(n_mel_frames)))


def channel_norm(x, gamma, beta, eps):
    # LayerNorm over the channel axis of [1, C, 1, T], written as the explicit
    # mean/centre/rsqrt sequence coremltools' layernorm fusion pass recognises,
    # which lowers to `layer_norm(axes=[1])` plus a `batch_norm` carrying the
    # affine.
    mean = x.mean(dim=1, keepdim=True)
    centered = x - mean
    var = (centered * centered).mean(dim=1, keepdim=True)
    return centered * torch.rsqrt(var + eps) * gamma + beta


class AneEncoder(torch.nn.Module):
    """FastConformer encoder over a fixed window, in [1, C, 1, T] throughout."""

    def __init__(self, ref, weights, meta, window_mel_frames, eps=1e-5):
        super().__init__()
        self.ref = ref
        self.d_model = int(meta["parakeet.encoder.d_model"])
        self.n_layers = int(meta["parakeet.encoder.n_layers"])
        self.n_heads = int(meta["parakeet.encoder.n_heads"])
        self.head_dim = self.d_model // self.n_heads
        self.eps = float(eps)
        self.conv_layer_norm = (
            str(meta.get("parakeet.encoder.conv_norm_type", "batch_norm")) == "layer_norm"
        )
        self.window_mel_frames = int(window_mel_frames)
        self.n_mels = int(weights["preproc.mel_filterbank"].shape[0])
        causal = bool(meta.get("parakeet.encoder.causal_downsampling", False))
        self.enc_frames = encoder_frames_for_mel(self.window_mel_frames, causal)
        self.attn_scale = 1.0 / math.sqrt(self.head_dim)

        self._register_subsampling(weights, meta)
        for i in range(self.n_layers):
            self._register_block(weights, i)

    # -- weight staging ----------------------------------------------------
    # Everything is staged as a buffer in its final runtime shape so the traced
    # graph contains no reshape/transpose of a weight, only of activations.

    def _buf(self, name, tensor):
        self.register_buffer(name, tensor.contiguous().float())

    def _conv1x1(self, name, weight, bias=None, scale=1.0):
        # F.linear weight is [out, in]; a 1x1 conv wants [out, in, 1, 1].
        out_f, in_f = weight.shape
        self._buf(f"{name}_w", (weight * scale).reshape(out_f, in_f, 1, 1))
        if bias is not None:
            self._buf(f"{name}_b", bias * scale)
        else:
            self.register_buffer(f"{name}_b", None)

    def _norm(self, name, weights, prefix):
        # Channel-axis affine operands are [1, C, 1, 1] so they broadcast
        # against [1, C, 1, T] without a reshape at runtime.
        self._buf(f"{name}_g", weights[f"{prefix}.weight"].view(1, -1, 1, 1))
        self._buf(f"{name}_b", weights[f"{prefix}.bias"].view(1, -1, 1, 1))

    def _register_subsampling(self, weights, meta):
        for key in ("conv0", "conv1_dw", "conv1_pw", "conv2_dw", "conv2_pw"):
            self._buf(f"sub_{key}_w", weights[f"encoder.subsampling.{key}.weight"])
            self._buf(f"sub_{key}_b", weights[f"encoder.subsampling.{key}.bias"])
        # xscaling folds into the pre-encode output projection: it is a scalar
        # multiply on this projection's output and nothing reads the unscaled
        # value.
        xscale = (
            math.sqrt(self.d_model)
            if bool(meta.get("parakeet.encoder.xscaling", True))
            else 1.0
        )
        self._conv1x1(
            "sub_out",
            weights["encoder.subsampling.out.weight"],
            weights["encoder.subsampling.out.bias"],
            scale=xscale,
        )

    def _register_block(self, weights, i):
        p = f"encoder.blk.{i}"

        for tag, src in (("ff1", "norm_ff1"), ("attn", "norm_attn"),
                         ("conv", "norm_conv"), ("ff2", "norm_ff2"), ("out", "norm_out")):
            self._norm(f"b{i}_n_{tag}", weights, f"{p}.{src}")

        # The Macaron 0.5 folds into linear2 rather than becoming a runtime mul.
        for tag in ("ff1", "ff2"):
            self._conv1x1(f"b{i}_{tag}_l1", weights[f"{p}.{tag}.linear1.weight"],
                          weights.get(f"{p}.{tag}.linear1.bias"))
            self._conv1x1(f"b{i}_{tag}_l2", weights[f"{p}.{tag}.linear2.weight"],
                          weights.get(f"{p}.{tag}.linear2.bias"), scale=0.5)

        for tag in ("q", "k", "v", "out"):
            self._conv1x1(f"b{i}_attn_{tag}", weights[f"{p}.attn.{tag}.weight"],
                          weights.get(f"{p}.attn.{tag}.bias"))

        # pos_bias_u/v are [n_heads, head_dim]; u is consumed per-head against a
        # [H, hd, 1, T] view, v is consumed against the packed [1, C, 1, T] q.
        self._buf(f"b{i}_bias_u", weights[f"{p}.attn.pos_bias_u"].reshape(self.n_heads, self.head_dim, 1, 1))
        self._buf(f"b{i}_bias_v", weights[f"{p}.attn.pos_bias_v"].reshape(1, self.d_model, 1, 1))
        self._buf(f"b{i}_pos_proj", self._pos_proj(weights, f"{p}.attn.pos.weight"))

        self._conv1x1(f"b{i}_conv_pw1", weights[f"{p}.conv.pw1.weight"],
                      weights.get(f"{p}.conv.pw1.bias"))
        self._conv1x1(f"b{i}_conv_pw2", weights[f"{p}.conv.pw2.weight"],
                      weights.get(f"{p}.conv.pw2.bias"))
        dw = weights[f"{p}.conv.dw.weight"]           # [C, 1, k]
        self._buf(f"b{i}_conv_dw_w", dw.unsqueeze(2))  # [C, 1, 1, k]
        self._buf(f"b{i}_conv_dw_b", weights[f"{p}.conv.dw.bias"])
        if self.conv_layer_norm:
            self._norm(f"b{i}_conv_norm", weights, f"{p}.conv.norm")
        else:
            self._buf(f"b{i}_conv_bn_s", weights[f"{p}.conv.bn.scale"].view(1, -1, 1, 1))
            self._buf(f"b{i}_conv_bn_t", weights[f"{p}.conv.bn.shift"].view(1, -1, 1, 1))

    def _pos_proj(self, weights, key):
        # The relative-position table depends only on the (fixed) window length,
        # so its projection is a compile-time constant and the per-layer
        # projection matmul disappears from the graph entirely.
        t = self.enc_frames
        pe = self.ref.sinusoidal_rel_pe(t, self.d_model)        # [1, 2t-1, d_model]
        proj = F.linear(pe[0], weights[key])                    # [2t-1, d_model]
        return proj.reshape(2 * t - 1, self.n_heads, self.head_dim).permute(1, 0, 2)

    # -- forward -----------------------------------------------------------

    def _conv(self, x, name, **kw):
        bias = getattr(self, f"{name}_b", None)
        return F.conv2d(x, getattr(self, f"{name}_w"), bias, **kw)

    def _norm_apply(self, x, name):
        return channel_norm(x, getattr(self, f"{name}_g"), getattr(self, f"{name}_b"), self.eps)

    @staticmethod
    def halve_mask(mask):
        # A stride-2, kernel-3, pad-1 conv keeps floor((v - 1) / 2) + 1 valid
        # frames out of v, which is exactly the count of even indices below v,
        # so the next resolution's mask is this one's even entries.
        return mask[:, :, 0::2]

    def subsample(self, mel, mel_mask):
        # Runs in (time, freq) orientation, the orientation the GGUF subsampling
        # kernels are stored for; the result is permuted into channel-first
        # once, before the conformer blocks. Padded frames are zeroed at every
        # resolution, as the reference encoder does, so they cannot leak into
        # valid boundary frames through the conv kernels.
        mask = mel_mask.reshape(1, 1, -1, 1)                       # [1, 1, T_mel, 1]
        x = mel.squeeze(2).transpose(1, 2).unsqueeze(1) * mask     # [1, 1, T_mel, n_mels]
        x = F.conv2d(x, self.sub_conv0_w, self.sub_conv0_b, stride=2, padding=1)
        mask = self.halve_mask(mask)
        x = F.relu(x * mask)
        x = F.conv2d(x, self.sub_conv1_dw_w, self.sub_conv1_dw_b,
                     stride=2, padding=1, groups=x.size(1))
        mask = self.halve_mask(mask)
        x = F.conv2d(x * mask, self.sub_conv1_pw_w, self.sub_conv1_pw_b)
        x = F.relu(x * mask)
        x = F.conv2d(x, self.sub_conv2_dw_w, self.sub_conv2_dw_b,
                     stride=2, padding=1, groups=x.size(1))
        mask = self.halve_mask(mask)
        x = F.conv2d(x * mask, self.sub_conv2_pw_w, self.sub_conv2_pw_b)
        x = F.relu(x * mask)
        # [1, C, T_enc, F] -> [1, C*F, 1, T_enc], channel index c*F + f, which is
        # the input ordering encoder.subsampling.out.weight expects.
        b, c, t, f = x.shape
        x = x.permute(0, 1, 3, 2).reshape(b, c * f, 1, t)
        return self._conv(x, "sub_out")

    def attention(self, x, key_bias, i):
        h, hd, t = self.n_heads, self.head_dim, self.enc_frames
        q = self._conv(x, f"b{i}_attn_q")
        k = self._conv(x, f"b{i}_attn_k")
        v = self._conv(x, f"b{i}_attn_v")

        # matrix_bd: q biased by pos_bias_v against the baked positional
        # projection, then the Transformer-XL rel-shift as pad/reshape/slice.
        qb = (q + getattr(self, f"b{i}_bias_v")).reshape(h, hd, t)
        bd = torch.matmul(getattr(self, f"b{i}_pos_proj"), qb)     # [h, 2t-1, t]
        bd = bd.transpose(1, 2)                                    # [h, t, 2t-1]
        bd = F.pad(bd, (1, 0))                                     # [h, t, 2t]
        bd = bd.reshape(h, 2 * t, t)[:, 1:, :].reshape(h, t, 2 * t - 1)
        bd = bd[:, :, :t]                                          # [h, q, k]

        # matrix_ac in (head, key, 1, query) so the softmax reduces axis 1 and
        # every operand stays 4D and channel-first.
        qh = q.reshape(h, hd, 1, t) + getattr(self, f"b{i}_bias_u")
        kh = k.reshape(h, hd, 1, t).permute(0, 3, 2, 1)            # [h, k, 1, hd]
        ac = torch.einsum("bkhc,bchq->bkhq", kh, qh)               # [h, k, 1, q]

        scores = (ac + bd.transpose(1, 2).unsqueeze(2)) * self.attn_scale + key_bias
        attn = torch.softmax(scores, dim=1)                        # over keys

        ctx = torch.matmul(
            attn.permute(0, 2, 3, 1).reshape(h, t, t),             # [h, q, k]
            v.reshape(h, hd, 1, t).permute(0, 2, 3, 1).reshape(h, t, hd),
        )                                                          # [h, q, hd]
        ctx = ctx.reshape(h, 1, t, hd).permute(0, 3, 1, 2).reshape(1, self.d_model, 1, t)
        return self._conv(ctx, f"b{i}_attn_out")

    def conv_module(self, x, pad_mask, i):
        y = self._conv(x, f"b{i}_conv_pw1")
        a, b = y.chunk(2, dim=1)
        y = a * torch.sigmoid(b)
        # Zero the pad region before the depthwise conv so padding cannot leak
        # into valid frames through its receptive field.
        y = y * pad_mask
        k = getattr(self, f"b{i}_conv_dw_w").size(-1)
        y = F.conv2d(y, getattr(self, f"b{i}_conv_dw_w"), getattr(self, f"b{i}_conv_dw_b"),
                     padding=(0, (k - 1) // 2), groups=y.size(1))
        if self.conv_layer_norm:
            y = self._norm_apply(y, f"b{i}_conv_norm")
        else:
            y = y * getattr(self, f"b{i}_conv_bn_s") + getattr(self, f"b{i}_conv_bn_t")
        y = F.silu(y)
        return self._conv(y, f"b{i}_conv_pw2")

    def block(self, x, pad_mask, key_bias, i):
        y = self._norm_apply(x, f"b{i}_n_ff1")
        y = self._conv(F.silu(self._conv(y, f"b{i}_ff1_l1")), f"b{i}_ff1_l2")
        x = x + y

        y = self._norm_apply(x, f"b{i}_n_attn")
        x = x + self.attention(y, key_bias, i)

        y = self._norm_apply(x, f"b{i}_n_conv")
        x = x + self.conv_module(y, pad_mask, i)

        y = self._norm_apply(x, f"b{i}_n_ff2")
        y = self._conv(F.silu(self._conv(y, f"b{i}_ff2_l1")), f"b{i}_ff2_l2")
        x = x + y

        return self._norm_apply(x, f"b{i}_n_out")

    def forward(self, mel, mel_mask, pad_mask, key_bias):
        x = self.subsample(mel, mel_mask) * pad_mask
        for i in range(self.n_layers):
            x = self.block(x, pad_mask, key_bias, i)
        return x


PARITY_REL_TOLERANCE = 1e-4


def window_inputs(model, mel_2d, valid_mel=None):
    """The graph's four inputs for a mel of at most one window: the mel padded
    with zeros to the window and the three masks for its valid part, which is
    the whole mel unless `valid_mel` says its trailing frames are padding."""
    n_frames = mel_2d.shape[1]
    if valid_mel is None:
        valid_mel = n_frames
    valid_enc = encoder_frames_for_mel(valid_mel, causal=False)
    mel = torch.zeros(1, model.n_mels, 1, model.window_mel_frames)
    mel[0, :, 0, :n_frames] = mel_2d
    mel_mask = torch.zeros(1, 1, 1, model.window_mel_frames)
    mel_mask[..., :valid_mel] = 1.0
    pad_mask = torch.zeros(1, 1, 1, model.enc_frames)
    pad_mask[..., :valid_enc] = 1.0
    key_bias = torch.full((1, model.enc_frames, 1, 1), KEY_BIAS_PAD)
    key_bias[:, :valid_enc] = 0.0
    return (mel, mel_mask, pad_mask, key_bias), valid_enc


def ane_encoder_frames(model, mel_2d, valid_mel=None):
    """Valid encoder frames of the ANE graph for a mel of at most one window, as [T, d_model]."""
    inputs, valid_enc = window_inputs(model, mel_2d, valid_mel)
    with torch.inference_mode():
        out = model(*inputs)
    return out[0, :, 0, :valid_enc].transpose(0, 1)


def relative_error(got, want):
    """Relative L2 distance of `got` from `want`; infinite when the shapes differ."""
    if got.shape != want.shape:
        return float("inf")
    diff = (got - want).double()
    return float(torch.sqrt((diff * diff).sum() / (want.double() ** 2).sum()))


def reference_frames(ref, weights, meta, mel_2d, valid_mel=None):
    """Reference encoder frames for the audio frames of `mel_2d` (all of them,
    or the first `valid_mel`), as [T, d_model]."""
    audio = mel_2d if valid_mel is None else mel_2d[:, :valid_mel]
    with torch.inference_mode():
        return ref.encoder_forward(audio, weights, meta)[0]


def compare_with_reference(model, ref, weights, meta, mel_2d, label, valid_mel=None):
    """True when the graph's valid frames match the reference on the audio
    frames of `mel_2d` (all of them, or the first `valid_mel`)."""
    want = reference_frames(ref, weights, meta, mel_2d, valid_mel)
    got = ane_encoder_frames(model, mel_2d, valid_mel)
    if got.shape != want.shape:
        print(f"[check] FAIL {label} shape: ane={tuple(got.shape)} ref={tuple(want.shape)}")
        return False
    rel = relative_error(got, want)
    print(f"[check] {label} rel={rel:.3e} max_abs={float((got - want).abs().max()):.3e} "
          f"shape={tuple(got.shape)}")
    if rel >= PARITY_REL_TOLERANCE:
        print(f"[check] FAIL: the ANE graph does not match the reference encoder on {label}.")
        return False
    return True


def check_parity(model, ref, weights, meta, seed=0):
    """Compare the ANE graph against the reference encoder on a full window
    and on a partially filled one, whose valid frames must match the
    reference run on the short input alone.

    Returns True if both relative errors are below PARITY_REL_TOLERANCE.
    """
    torch.manual_seed(seed)
    full = torch.randn(model.n_mels, model.window_mel_frames)
    partial = full[:, : model.window_mel_frames // 3 + 7]
    ok = compare_with_reference(model, ref, weights, meta, full, "full window")
    ok = compare_with_reference(model, ref, weights, meta, partial, "partial window") and ok
    print("[check] OK" if ok else "[check] FAIL")
    return ok


def palettize(mlmodel, nbits):
    from coremltools.optimize.coreml import (
        OpPalettizerConfig, OptimizationConfig, palettize_weights,
    )

    config = OptimizationConfig(
        global_config=OpPalettizerConfig(
            nbits=nbits,
            mode="kmeans",
            granularity="per_grouped_channel",
            group_size=PALETTIZE_GROUP_SIZE,
        )
    )
    return palettize_weights(mlmodel, config)


def _placement_counts(mlmodelc_path, compute_units):
    from collections import Counter
    from coremltools.models.compute_plan import MLComputePlan

    plan = MLComputePlan.load_from_path(str(mlmodelc_path), compute_units=compute_units)

    def device_name(device):
        if device is None:
            return "None"
        return type(device).__name__.replace("ML", "").replace("ComputeDevice", "")

    counts = Counter()

    def walk(block):
        for op in block.operations:
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            counts[device_name(usage.preferred_compute_device) if usage is not None else "None"] += 1
            nested = getattr(op, "blocks", None) or getattr(op, "block", None)
            if nested:
                for inner in (nested if isinstance(nested, (list, tuple)) else [nested]):
                    walk(inner)

    walk(plan.model_structure.program.functions["main"].block)
    return counts


def report_compute_placement(mlmodelc_path):
    # Reports op placement under CPU_AND_NE, which parakeet-encoder.mm pins, and
    # under ALL, Core ML's default preference, which may resolve to the GPU even
    # for a fully ANE-capable graph.
    #
    # Requires macOS 14.4+ / coremltools>=8.
    for label, units in (("CPU_AND_NE (what the wrapper pins)", ct.ComputeUnit.CPU_AND_NE),
                         ("ALL        (Core ML's own preference)", ct.ComputeUnit.ALL)):
        try:
            counts = _placement_counts(mlmodelc_path, units)
        except Exception as exc:
            print(f"[export] compute-plan check skipped for {label} ({type(exc).__name__}: {exc})")
            continue

        compute = sum(v for key, v in counts.items() if key != "None")
        ane = counts.get("NeuralEngine", 0)
        share = f"{ane}/{compute} = {100.0 * ane / compute:.1f}%" if compute else "n/a"
        print(f"[export] {label}: ANE {share}  "
              f"[{', '.join(f'{k}={v}' for k, v in counts.most_common())}]")

        if units is ct.ComputeUnit.CPU_AND_NE and compute and ane < compute:
            print("[export] WARNING: not fully ANE-resident under CPU_AND_NE. Every op that falls "
                  "back also forces a round trip of its inputs and outputs, so partial residency "
                  "costs far more than the op itself. A fully ANE-shaped encoder measures 100% "
                  "here.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="output .mlpackage path")
    ap.add_argument("--scripts", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--window-mel-frames", type=int, default=DEFAULT_WINDOW_MEL_FRAMES,
                    help=f"fixed mel length of one window (default {DEFAULT_WINDOW_MEL_FRAMES}, "
                         "= 15 s at hop 160)")
    ap.add_argument("--nbits", type=int, default=6, choices=[0, 2, 3, 4, 6, 8],
                    help="weight palettization bit width; 0 exports fp16 weights (default 6)")
    ap.add_argument("--compile-dir", type=Path, default=None,
                    help="if set, compile the .mlpackage to a .mlmodelc here and report op placement")
    ap.add_argument("--check", action="store_true",
                    help="verify the ANE graph against the reference encoder before converting, "
                         "and stop if it does not match")
    args = ap.parse_args()

    ref = load_reference_encoder(args.scripts)
    weights, meta = ref.load_gguf(args.gguf)

    model = AneEncoder(ref, weights, meta, args.window_mel_frames).eval()
    print(f"[export] d_model={model.d_model} layers={model.n_layers} heads={model.n_heads} "
          f"head_dim={model.head_dim} n_mels={model.n_mels} "
          f"conv_norm={'layer_norm' if model.conv_layer_norm else 'batch_norm'}")
    print(f"[export] window: {model.window_mel_frames} mel frames -> {model.enc_frames} encoder frames "
          f"({model.window_mel_frames * 160 / 16000.0:.2f} s at hop 160)")

    if args.check and not check_parity(model, ref, weights, meta):
        return 1

    mel = torch.zeros(1, model.n_mels, 1, model.window_mel_frames)
    mel_mask = torch.ones(1, 1, 1, model.window_mel_frames)
    pad_mask = torch.ones(1, 1, 1, model.enc_frames)
    key_bias = torch.zeros(1, model.enc_frames, 1, 1)
    example = (mel, mel_mask, pad_mask, key_bias)

    with torch.inference_mode():
        out = model(*example)
    print(f"[export] traced output: {tuple(out.shape)}")

    traced = torch.jit.trace(model, example, check_trace=False)
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="mel", shape=mel.shape, dtype=np.float16),
            ct.TensorType(name="mel_mask", shape=mel_mask.shape, dtype=np.float16),
            ct.TensorType(name="pad_mask", shape=pad_mask.shape, dtype=np.float16),
            ct.TensorType(name="key_bias", shape=key_bias.shape, dtype=np.float16),
        ],
        outputs=[ct.TensorType(name="encoder_out", dtype=np.float16)],
        compute_units=ct.ComputeUnit.ALL,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
    )

    if args.nbits:
        mlmodel = palettize(mlmodel, args.nbits)
        print(f"[export] palettized weights to {args.nbits}-bit "
              f"(per_grouped_channel, group_size={PALETTIZE_GROUP_SIZE})")

    mlmodel.save(str(args.out))
    print(f"[export] saved {args.out}")

    if args.compile_dir is not None:
        import subprocess
        subprocess.run(["xcrun", "coremlc", "compile", str(args.out), str(args.compile_dir)],
                       check=True)
        compiled = Path(args.compile_dir) / (args.out.stem + ".mlmodelc")
        print(f"[export] compiled .mlmodelc into {args.compile_dir}")
        report_compute_placement(compiled)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
