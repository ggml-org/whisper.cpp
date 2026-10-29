"""Chatterbox S3Gen Core ML graphs rebuilt from the native GGUF tensors.

Each module mirrors the ggml graph of the same stage in src/chatterbox_tts.cpp
(build_encoder_graph_nodes, build_cfm_graph_nodes[_b2], build_hift_graph_nodes)
at a fixed exported width, with a 0/1 validity input so one graph serves every
shorter input:

- Encoder: token embeddings [1, T, 512] + token mask [1, 1, T] -> mu [1, 2T, 80].
  Padded keys are masked out of every attention; the only right-looking
  convolution (pre-lookahead) sees zeroed padding, as ggml's zero padding.
- CFM estimator: x, mu, cond [B, 80, F], spks [B, 80], t_emb [B, 1024] and
  frame mask [1, 1, F] -> velocity [B, 80, F]. Convolutions are causal and
  attention masks padded keys, so valid frames match a call at their length.
  B is 1 (Turbo meanflow) or 2 (Multilingual cond + uncond CFG).
- HiFT decoder window: mel [1, 80, W], source STFT [1, 18, 120 W + 1] and
  frame mask [1, 1, W] -> conv_post [1, 18, 120 W + 1] (log magnitude and
  phase rows). The host walks utterances in windows with receptive-field
  context and keeps exp/sin/iSTFT in float32.
"""

import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

MEL = 80
EMBED = 512
TIME_EMBED = 1024
HEADS = 8
HEAD_DIM = 64
ENCODER_BLOCKS = 6
UP_ENCODER_BLOCKS = 4
CFM_MID_BLOCKS = 12
CFM_TFM_BLOCKS = 4
NORM_EPS = 1e-5
MASKED = -1e4

HIFT_UPSAMPLE = ((8, 16), (5, 11), (3, 7))  # (stride, kernel) of each transposed conv
HIFT_HOP = 120                             # mel frame -> STFT frames (8 * 5 * 3)
HIFT_SOURCE = ((15, 7), (3, 1), (1, 0))    # (stride, padding) of each source_downs conv
RESBLOCK_KERNELS = (3, 7, 11)
SOURCE_KERNELS = (7, 7, 11)
DILATIONS = (1, 3, 5)
STFT_BINS = 18
SNAKE_EPS = 1e-9
HALF_SAFE_INVERSE = 6.0e4
# sin^2(pi f) ~= sum_k SIN2_POLY[k] f^(2k + 2) on [-1/2, 1/2] (scripts/parler/export-dac-coreml.py)
SIN2_POLY = (9.8672466, -32.37042703, 41.426299, -23.29808439)


def read_tensor(tensor):
    from gguf.quants import dequantize

    shape = tuple(int(d) for d in reversed(tensor.shape))
    if tensor.tensor_type.name in ("F32", "F16"):
        return np.asarray(tensor.data, dtype=np.float32).reshape(shape)
    return dequantize(np.asarray(tensor.data), tensor.tensor_type).astype(np.float32).reshape(shape)


def read_gguf(path, prefixes):
    from gguf import GGUFReader

    reader = GGUFReader(str(path))
    arch = reader.fields["general.architecture"].contents()
    if arch != "chatterbox-s3gen":
        raise ValueError(f"{path} is a {arch} GGUF, not a Chatterbox S3Gen GGUF")
    tensors = {t.name: read_tensor(t) for t in reader.tensors if t.name.startswith(prefixes)}
    metadata = {name: field.contents() for name, field in reader.fields.items() if name.startswith("s3gen.")}
    return tensors, metadata


def constant(array):
    return nn.Parameter(torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32)), requires_grad=False)


class Weights(nn.Module):
    def __init__(self, tensors):
        super().__init__()
        self.weights = nn.ParameterDict({name.replace("/", "__"): constant(value)
                                        for name, value in tensors.items()})
        # Static Python widths: a traced tensor.shape would become a graph op.
        self.widths = {name: int(np.shape(value)[0]) for name, value in tensors.items()}
        self.ranks = {name: np.ndim(value) for name, value in tensors.items()}

    def weight(self, name):
        return self.weights[name.replace("/", "__")]

    def has(self, name):
        return name.replace("/", "__") in self.weights

    def linear(self, x, weight, bias=None):
        return F.linear(x, self.weight(weight), self.weight(bias) if bias else None)


def key_mask(valid):
    """[1, 1, T] 0/1 -> additive [1, 1, 1, T] attention bias."""
    return ((1.0 - valid) * MASKED).unsqueeze(1)


def relative_positions(length, width=EMBED):
    """compute_pos_emb (chatterbox_tts.cpp): rows for relative positions
    length - 1 down to -(length - 1)."""
    positions = np.arange(length - 1, -length, -1, dtype=np.float64)[:, None]
    div = np.exp(-(np.arange(0, width, 2, dtype=np.float64) * math.log(10000.0) / width))
    table = np.zeros((2 * length - 1, width), dtype=np.float64)
    table[:, 0::2] = np.sin(positions * div)
    table[:, 1::2] = np.cos(positions * div)
    return table.astype(np.float32)


def relative_shift(scores, length):
    """[1, H, T, 2T - 1] scores against relative_positions rows ->
    [1, H, T, T] with column j holding relative position i - j."""
    heads = HEADS
    padded = F.pad(scores, (1, 0))
    padded = padded.reshape(1, heads, 2 * length, length)[:, :, 1:]
    return padded.reshape(1, heads, length, 2 * length - 1)[..., :length]


class Encoder(Weights):
    """build_encoder_graph_nodes at a fixed token width."""

    def __init__(self, tensors, metadata, tokens):
        super().__init__({k: v for k, v in tensors.items() if k.startswith(("flow/encoder/", "flow/encoder_proj/"))})
        self.tokens = tokens
        self.ratio = int(metadata.get("s3gen.encoder.token_mel_ratio", 2))
        self.eps = float(metadata.get("s3gen.layer_norm_eps", 1e-12))
        # Relative-position projections depend only on the exported width.
        for prefix, count, length in (("block", ENCODER_BLOCKS, tokens),
                                      ("up_block", UP_ENCODER_BLOCKS, tokens * self.ratio)):
            table = torch.from_numpy(relative_positions(length))
            for index in range(count):
                name = f"flow/encoder/{prefix}{index}/attn/pos/w"
                with torch.no_grad():
                    projected = F.linear(table, self.weight(name))
                self.weights[name.replace("/", "__") + "__projected"] = nn.Parameter(projected, requires_grad=False)

    def embed(self, x, prefix):
        x = self.linear(x, prefix + "/linear/w", prefix + "/linear/b")
        x = F.layer_norm(x, (EMBED,), self.weight(prefix + "/norm/w"), self.weight(prefix + "/norm/b"), NORM_EPS)
        return x * math.sqrt(EMBED)

    def conformer(self, x, prefix, length, bias):
        def norm(value, name):
            return F.layer_norm(value, (EMBED,), self.weight(f"{prefix}/{name}/w"), self.weight(f"{prefix}/{name}/b"),
                                self.eps)

        def heads(value):
            return value.reshape(1, length, HEADS, HEAD_DIM).transpose(1, 2)

        hidden = norm(x, "norm_mha")
        q = heads(self.linear(hidden, prefix + "/attn/q/w", prefix + "/attn/q/b"))
        k = heads(self.linear(hidden, prefix + "/attn/k/w", prefix + "/attn/k/b"))
        v = heads(self.linear(hidden, prefix + "/attn/v/w", prefix + "/attn/v/b"))
        p = self.weight(prefix + "/attn/pos/w__projected").reshape(2 * length - 1, HEADS, HEAD_DIM).permute(1, 2, 0)
        u = self.weight(prefix + "/attn/pos_bias_u").reshape(1, HEADS, 1, HEAD_DIM)
        w = self.weight(prefix + "/attn/pos_bias_v").reshape(1, HEADS, 1, HEAD_DIM)
        content = (q + u) @ k.transpose(-1, -2)
        position = relative_shift((q + w) @ p.unsqueeze(0), length)
        scores = (content + position) / math.sqrt(HEAD_DIM) + bias
        attended = (scores.softmax(-1) @ v).transpose(1, 2).reshape(1, length, EMBED)
        x = x + self.linear(attended, prefix + "/attn/o/w", prefix + "/attn/o/b")
        hidden = norm(x, "norm_ff")
        hidden = F.silu(self.linear(hidden, prefix + "/ff/w1/w", prefix + "/ff/w1/b"))
        return x + self.linear(hidden, prefix + "/ff/w2/w", prefix + "/ff/w2/b")

    def forward(self, embedding, valid):
        tokens, frames = self.tokens, self.tokens * self.ratio
        x = self.embed(embedding, "flow/encoder/embed")
        residual = x
        hidden = (x * valid.transpose(1, 2)).transpose(1, 2)
        hidden = F.conv1d(F.pad(hidden, (0, 3)), self.weight("flow/encoder/pre_lookahead/conv1/w"),
                          self.weight("flow/encoder/pre_lookahead/conv1/b"))
        hidden = F.conv1d(F.pad(F.leaky_relu(hidden, 0.01), (2, 0)), self.weight("flow/encoder/pre_lookahead/conv2/w"),
                          self.weight("flow/encoder/pre_lookahead/conv2/b"))
        x = hidden.transpose(1, 2) + residual
        bias = key_mask(valid)
        for index in range(ENCODER_BLOCKS):
            x = self.conformer(x, f"flow/encoder/block{index}", tokens, bias)
        hidden = x.transpose(1, 2).repeat_interleave(self.ratio, dim=-1)
        hidden = F.conv1d(F.pad(hidden, (2 * self.ratio, 0)), self.weight("flow/encoder/up_layer/conv/w"),
                          self.weight("flow/encoder/up_layer/conv/b"))
        x = self.embed(hidden.transpose(1, 2), "flow/encoder/up_embed")
        bias = key_mask(valid.repeat_interleave(self.ratio, dim=-1))
        for index in range(UP_ENCODER_BLOCKS):
            x = self.conformer(x, f"flow/encoder/up_block{index}", frames, bias)
        x = F.layer_norm(x, (EMBED,), self.weight("flow/encoder/after_norm/w"), self.weight("flow/encoder/after_norm/b"),
                         NORM_EPS)
        return self.linear(x, "flow/encoder_proj/w", "flow/encoder_proj/b")


def channel_norm(x, weight, bias, eps=NORM_EPS):
    """LayerNorm over the channel axis of a [B, C, 1, T] activation."""
    centered = x - x.mean(1, keepdim=True)
    variance = (centered * centered).mean(1, keepdim=True)
    return centered * torch.rsqrt(variance + eps) * weight.reshape(1, -1, 1, 1) + bias.reshape(1, -1, 1, 1)


class Estimator(Weights):
    """build_cfm_graph_nodes / build_cfm_graph_nodes_b2 at a fixed frame width.

    Neural Engine layout: activations stay [B, C, 1, T], projections are 1x1
    convolutions, LayerNorm reduces the channel axis and attention runs one
    head at a time (softmax over the key axis), so the graph has no
    sequence-channel transposes."""

    def __init__(self, tensors, frames, batch):
        selected = {k: v for k, v in tensors.items()
                    if k.startswith("cfm/") and not k.startswith(("cfm/time_mlp/", "cfm/time_embed_mixer/"))}
        for name in [k for k in selected if k.endswith("/attn1/to_q/weight")]:
            prefix = name[:-len("to_q/weight")]
            # 1 / sqrt(head_dim) folded into the query rows.
            selected[prefix + "to_qkv/weight"] = np.concatenate(
                [selected.pop(prefix + "to_q/weight") / math.sqrt(HEAD_DIM)] +
                [selected.pop(prefix + f"to_{p}/weight") for p in "kv"], axis=0)
        super().__init__(selected)
        self.frames, self.batch = frames, batch

    def pointwise(self, x, weight, bias=None):
        w = self.weight(weight)
        w = w.unsqueeze(2) if self.ranks[weight] == 3 else w[:, :, None, None]
        return F.conv2d(x, w, self.weight(bias) if bias else None)

    def causal(self, x, name, taps=3):
        return F.conv2d(F.pad(x, (taps - 1, 0)), self.weight(name + "/weight").unsqueeze(2), self.weight(name + "/bias"))

    def block(self, x, prefix):
        x = self.causal(x, prefix + "/block/0")
        x = channel_norm(x, self.weight(prefix + "/block/2/weight"), self.weight(prefix + "/block/2/bias"))
        return x * torch.tanh(F.softplus(x))

    def resnet(self, x, time, prefix):
        hidden = self.block(x, prefix + "/block1")
        projected = self.linear(time * torch.tanh(F.softplus(time)), prefix + "/mlp/1/weight", prefix + "/mlp/1/bias")
        hidden = hidden + projected.reshape(self.batch, -1, 1, 1)
        hidden = self.block(hidden, prefix + "/block2")
        return hidden + self.pointwise(x, prefix + "/res_conv/weight", prefix + "/res_conv/bias")

    def transformer(self, x, prefix, bias):
        hidden = channel_norm(x, self.weight(prefix + "/norm1/weight"), self.weight(prefix + "/norm1/bias"))
        q, k, v = self.pointwise(hidden, prefix + "/attn1/to_qkv/weight").split(HEADS * HEAD_DIM, dim=1)
        # Both products in the Neural Engine's native einsum forms (Apple's
        # ml-ane-transformers spelling): every other spelling either runs 2x
        # slower or lowers to transposes and reshapes that double the program,
        # and a function's load time grows with the whole program's size.
        heads = []
        for qh, kh, vh in zip(q.split(HEAD_DIM, dim=1), k.transpose(1, 3).split(HEAD_DIM, dim=3),
                              v.split(HEAD_DIM, dim=1)):
            scores = torch.einsum("bchq,bkhc->bkhq", qh, kh) + bias
            heads.append(torch.einsum("bkhq,bchk->bchq", scores.softmax(dim=1), vh))
        x = x + self.pointwise(torch.cat(heads, dim=1), prefix + "/attn1/to_out/0/weight", prefix + "/attn1/to_out/0/bias")
        hidden = channel_norm(x, self.weight(prefix + "/norm3/weight"), self.weight(prefix + "/norm3/bias"))
        hidden = F.gelu(self.pointwise(hidden, prefix + "/ff/net/0/proj/weight", prefix + "/ff/net/0/proj/bias"))
        return x + self.pointwise(hidden, prefix + "/ff/net/2/weight", prefix + "/ff/net/2/bias")

    def stage(self, x, time, prefix, bias):
        x = self.resnet(x, time, prefix + "/0")
        for index in range(CFM_TFM_BLOCKS):
            x = self.transformer(x, f"{prefix}/1/{index}", bias)
        return x

    def forward(self, x, mu, cond, spks, t_emb, valid):
        bias = ((1.0 - valid) * MASKED).reshape(1, self.frames, 1, 1)
        speaker = spks.reshape(self.batch, MEL, 1, 1).expand(self.batch, MEL, 1, self.frames)
        inputs = [t.reshape(self.batch, MEL, 1, self.frames) for t in (x, mu)]
        hidden = torch.cat(inputs + [speaker, cond.reshape(self.batch, MEL, 1, self.frames)], dim=1)
        hidden = self.stage(hidden, t_emb, "cfm/down_blocks/0", bias)
        skip = hidden
        hidden = self.causal(hidden, "cfm/down_blocks/0/2")
        for index in range(CFM_MID_BLOCKS):
            hidden = self.stage(hidden, t_emb, f"cfm/mid_blocks/{index}", bias)
        hidden = self.stage(torch.cat((hidden, skip), dim=1), t_emb, "cfm/up_blocks/0", bias)
        hidden = self.causal(hidden, "cfm/up_blocks/0/2")
        hidden = self.block(hidden, "cfm/final_block")
        out = self.pointwise(hidden, "cfm/final_proj/weight", "cfm/final_proj/bias")
        return out.reshape(self.batch, MEL, self.frames)


class Snake(nn.Module):
    """x + sin^2(alpha x) / (alpha + 1e-9) without a sine (the Neural Engine
    has none): alpha x / pi range-reduced to [-1/2, 1/2], then an even
    polynomial with the inverse alpha folded into its coefficients."""

    def __init__(self, alpha, exact=False):
        super().__init__()
        alpha = alpha.reshape(1, -1, 1).astype(np.float64)
        inverse = np.clip(1.0 / (alpha + SNAKE_EPS), -HALF_SAFE_INVERSE, HALF_SAFE_INVERSE)
        self.exact = exact
        self.alpha = constant(alpha)
        self.inverse = constant(inverse)
        self.scale = constant(alpha / math.pi)
        self.coef = nn.ParameterList(constant(inverse * c) for c in SIN2_POLY)

    def forward(self, x):
        if self.exact:
            s = torch.sin(self.alpha * x)
            return x + self.inverse * (s * s)
        u = x * self.scale
        f = u - torch.round(u)
        f2 = f * f
        p = f2 * self.coef[-1]
        for k in range(len(self.coef) - 2, -1, -1):
            p = f2 * (self.coef[k] + p)
        return x + p


def phase_kernel(weight, stride, padding):
    """ConvTranspose1d [IC, OC, K] (stride s, padding p) -> a causal Conv1d
    [s * OC, IC, M] whose output phase r at step q is the transposed
    convolution at sample q * s + r - p (M = ceil(K / s) taps)."""
    in_ch, out_ch, taps = weight.shape
    width = -(-taps // stride)
    phase = np.zeros((stride * out_ch, in_ch, width), dtype=np.float32)
    for r in range(stride):
        for m in range(width):
            if r + m * stride < taps:
                phase[r * out_ch:(r + 1) * out_ch, :, width - 1 - m] = weight[:, :, r + m * stride].T
    return phase


class Upsample(nn.Module):
    """Transposed convolution in phase form: the native ConvTranspose
    miscomputes on the Neural Engine (measured on Parler/Audio8)."""

    def __init__(self, weight, bias, stride, padding, length):
        super().__init__()
        self.stride, self.padding, self.length = stride, padding, length
        self.out_ch = weight.shape[1]
        self.weight = constant(phase_kernel(weight, stride, padding))
        self.bias = constant(bias.reshape(1, -1, 1))
        self.width = self.weight.shape[-1]

    def forward(self, x):
        length = self.length
        z = F.conv1d(F.pad(x, (self.width - 1, 1)), self.weight)
        z = z.reshape(1, self.stride, self.out_ch, length + 1).permute(0, 2, 3, 1)
        z = z.reshape(1, self.out_ch, (length + 1) * self.stride)
        return z[..., self.padding:self.padding + length * self.stride] + self.bias


def stage_mask(valid, repeat, lead=0):
    """[1, 1, W] frame mask -> the mask of a stage `repeat` samples per frame,
    with `lead` always-valid samples in front (the HiFT reflection pad)."""
    mask = valid.repeat_interleave(repeat, dim=-1) if repeat > 1 else valid
    return F.pad(mask, (lead, 0), value=1.0) if lead else mask


class HiFT(Weights):
    """build_hift_graph_nodes up to conv_post, over one fixed window."""

    def __init__(self, tensors, window, exact_snake=False):
        selected = {k: v for k, v in tensors.items()
                    if k.startswith("hift/") and not k.startswith(("hift/f0_predictor/", "hift/m_source/"))}
        alphas = {k: selected.pop(k) for k in list(selected) if k.endswith("/alpha")}
        super().__init__(selected)
        self.window = window
        self.kernels = {k[:-len("/weight")]: tuple(int(d) for d in np.shape(v))
                        for k, v in selected.items() if k.endswith("/weight")}
        self.snakes = nn.ModuleDict({k.replace("/", "__"): Snake(v, exact_snake) for k, v in alphas.items()})
        self.ups = nn.ModuleList(
            Upsample(tensors[f"hift/ups/{i}/weight"], tensors[f"hift/ups/{i}/bias"], stride, (kernel - stride) // 2,
                     window * int(np.prod([s for s, _ in HIFT_UPSAMPLE[:i]])))
            for i, (stride, kernel) in enumerate(HIFT_UPSAMPLE))

    def conv(self, x, name, padding, dilation=1, stride=1, length=None):
        x = F.pad(x, (padding, padding))
        weight, bias = self.weight(name + "/weight"), self.weight(name + "/bias")
        out_ch, in_ch, taps = self.kernels[name]
        if dilation > 1:
            # Dilated convolution as an undilated one over [L / d, d] phases
            # (the Neural Engine runs dilation 5 at these widths on the CPU).
            rows = -(-(length + (taps - 1) * dilation) // dilation)
            x = F.pad(x, (0, rows * dilation - (length + (taps - 1) * dilation)))
            y = F.conv2d(x.reshape(1, in_ch, rows, dilation), weight.unsqueeze(-1), bias)
            return y.reshape(1, out_ch, -1)[..., :length]
        if stride > 1 and taps == 2 * stride:
            # A stride-s, 2s-tap convolution as a [2, s] kernel over [L / s, s]
            # blocks (the Neural Engine runs the strided form on the CPU).
            y = F.conv2d(x[..., :(length + 1) * stride].reshape(1, in_ch, length + 1, stride),
                         weight.reshape(out_ch, in_ch, 2, stride), bias)
            return y.reshape(1, out_ch, length)
        return F.conv1d(x, weight, bias, stride=stride)

    def snake(self, x, name):
        return self.snakes[name.replace("/", "__")](x)

    def resblock(self, x, prefix, kernel, mask, length):
        for index, dilation in enumerate(DILATIONS):
            hidden = self.snake(x, f"{prefix}/activations1/{index}/alpha")
            hidden = self.conv(hidden, f"{prefix}/convs1/{index}", (kernel * dilation - dilation) // 2, dilation,
                               length=length) * mask
            hidden = self.snake(hidden, f"{prefix}/activations2/{index}/alpha")
            hidden = self.conv(hidden, f"{prefix}/convs2/{index}", (kernel - 1) // 2) * mask
            x = x + hidden
        return x

    def forward(self, mel, source, valid):
        x = self.conv(mel * valid, "hift/conv_pre", 3) * valid
        repeat = 1
        for index, ((stride, _), (source_stride, source_pad)) in enumerate(zip(HIFT_UPSAMPLE, HIFT_SOURCE)):
            repeat *= stride
            last = index == len(HIFT_UPSAMPLE) - 1
            mask = stage_mask(valid, repeat, 1 if last else 0)
            x = self.ups[index](F.leaky_relu(x, 0.1))
            if last:
                x = torch.cat((x[..., 1:2], x), dim=-1)
            x = x * mask
            length = self.window * repeat + (1 if last else 0)
            s = self.conv(source, f"hift/source_downs/{index}", source_pad, stride=source_stride, length=length) * mask
            s = self.resblock(s, f"hift/source_resblocks/{index}", SOURCE_KERNELS[index], mask, length)
            x = x + s
            x = sum(self.resblock(x, f"hift/resblocks/{index * 3 + j}", RESBLOCK_KERNELS[j], mask, length)
                    for j in range(len(RESBLOCK_KERNELS))) / len(RESBLOCK_KERNELS)
        return self.conv(F.leaky_relu(x, 0.01), "hift/conv_post", 3)


def hift_receptive_frames():
    """Support radius, in mel frames, of one HiFT output frame (the residual
    stacks dominate the parallel source branch at every stage)."""
    radius = 3.0                                                # conv_pre, 7 taps
    rate = 1
    for stride, kernel in HIFT_UPSAMPLE:
        radius += math.ceil(kernel / stride) / rate             # transposed conv
        rate *= stride
        radius += max(sum((k - 1) * d // 2 + (k - 1) // 2 for d in DILATIONS) for k in RESBLOCK_KERNELS) / rate
    radius += (3 + 1) / rate                                    # conv_post, reflection pad
    return math.ceil(radius)
