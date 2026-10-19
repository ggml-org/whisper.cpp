"""CosyVoice3 Core ML graphs rebuilt from the native GGUF tensors."""

import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

MEL_CHANNELS = 80
TIME_CHANNELS = 256
CFG_BATCH = 2
F0_LAYERS = 5
UPSAMPLE_RATES = (8, 5, 3)
SOURCE_STRIDES = (15, 3, 1)
RESBLOCK_KERNELS = (3, 7, 11)
SOURCE_KERNELS = (7, 7, 11)
DILATIONS = (1, 3, 5)
FFT_SIZE = 16
FFT_HOP = 4
NORM_EPS = 1e-6


def read_tensor(tensor):
    from gguf.quants import dequantize

    shape = tuple(int(d) for d in reversed(tensor.shape))
    if tensor.tensor_type.name in ("F32", "F16"):
        return np.asarray(tensor.data, dtype=np.float32).reshape(shape)
    return dequantize(tensor.data, tensor.tensor_type).astype(np.float32).reshape(shape)


def read_gguf(path, prefix):
    from gguf import GGUFReader

    reader = GGUFReader(str(path))
    tensors = {t.name: read_tensor(t) for t in reader.tensors if t.name.startswith(prefix)}
    metadata = {name: field.contents() for name, field in reader.fields.items()
                if name.startswith("cosyvoice3.")}
    if not tensors:
        raise ValueError(f"No {prefix} tensors in {path}")
    return tensors, metadata


def parameter(array):
    return nn.Parameter(torch.from_numpy(np.array(array, dtype=np.float32, copy=True)),
                        requires_grad=False)


class Weights(nn.Module):
    def __init__(self, tensors):
        super().__init__()
        self.weights = nn.ParameterDict({name.replace("/", "__"): parameter(value)
                                        for name, value in tensors.items()})

    def weight(self, name):
        return self.weights[name.replace("/", "__")]

    def linear(self, x, name):
        bias = self.weights.get((name + "/bias").replace("/", "__"))
        return F.linear(x, self.weight(name + "/weight"), bias)

    def conv(self, x, name, stride=1, dilation=1, groups=1):
        return F.conv1d(x, self.weight(name + "/weight"), self.weight(name + "/bias"),
                        stride=stride, dilation=dilation, groups=groups)


def rms_norm(x, weight, eps):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * weight


def rotate_half(x, cosine, sine):
    first, second = x.chunk(2, dim=-1)
    return torch.cat((first * cosine - second * sine,
                      first * sine + second * cosine), dim=-1)


class Qwen(Weights):
    def __init__(self, tensors, metadata, context, persistent=True):
        selected = {k: v for k, v in tensors.items()
                    if k.startswith("lm/blk/") or k in ("lm/norm/weight", "lm/llm_decoder/weight")}
        super().__init__(selected)
        self.depth = int(metadata.get("cosyvoice3.llm.depth", 24))
        self.heads = int(metadata.get("cosyvoice3.llm.n_head", 14))
        self.kv_heads = int(metadata.get("cosyvoice3.llm.n_kv", 2))
        self.head_dim = int(metadata.get("cosyvoice3.llm.head_dim", 64))
        self.eps = float(metadata.get("cosyvoice3.llm.rms_eps", NORM_EPS))
        self.context = context
        if persistent:
            self.register_caches()

    def register_caches(self):
        shape = (1, self.kv_heads, self.context, self.head_dim)
        for layer in range(self.depth):
            self.register_buffer(f"key_{layer}", torch.zeros(shape, dtype=torch.float16))
            self.register_buffer(f"value_{layer}", torch.zeros(shape, dtype=torch.float16))

    def reset(self):
        for buffer in self.buffers():
            buffer.zero_()

    def project(self, x, prefix):
        fused = (prefix + "/qkv_proj/weight").replace("/", "__")
        if fused in self.weights:
            widths = (self.heads * self.head_dim, self.kv_heads * self.head_dim,
                      self.kv_heads * self.head_dim)
            return self.linear(x, prefix + "/qkv_proj").split(widths, dim=-1)
        return tuple(self.linear(x, prefix + "/" + name + "_proj") for name in ("q", "k", "v"))

    def layer_step(self, x, index, cosine, sine, write_mask, mask, key, value):
        prefix = f"lm/blk/{index}"
        hidden = rms_norm(x, self.weight(prefix + "/in_ln/weight"), self.eps)
        q, k, v = self.project(hidden, prefix)
        q = rotate_half(q.reshape(1, 1, self.heads, self.head_dim).transpose(1, 2), cosine, sine)
        k = rotate_half(k.reshape(1, 1, self.kv_heads, self.head_dim).transpose(1, 2), cosine, sine)
        v = v.reshape(1, 1, self.kv_heads, self.head_dim).transpose(1, 2)
        updated_key = key.float() * (1 - write_mask) + k * write_mask
        updated_value = value.float() * (1 - write_mask) + v * write_mask
        key = updated_key.to(torch.float16)
        value = updated_value.to(torch.float16)
        groups = self.heads // self.kv_heads
        keys = key.float().repeat_interleave(groups, dim=1)
        values = value.float().repeat_interleave(groups, dim=1)
        scores = q @ keys.transpose(-1, -2) / math.sqrt(self.head_dim) + mask
        output = (scores.softmax(-1) @ values).transpose(1, 2).reshape(1, 1, -1)
        x = x + self.linear(output, prefix + "/o_proj")
        hidden = rms_norm(x, self.weight(prefix + "/post_ln/weight"), self.eps)
        x = x + self.linear(F.silu(self.linear(hidden, prefix + "/gate")) *
                            self.linear(hidden, prefix + "/up"), prefix + "/down")
        return x, key, value

    def layer(self, x, index, cosine, sine, write_mask, mask):
        key, value = getattr(self, f"key_{index}"), getattr(self, f"value_{index}")
        x, next_key, next_value = self.layer_step(x, index, cosine, sine, write_mask, mask, key, value)
        key[:, :, :, :] = next_key
        value[:, :, :, :] = next_value
        return x

    def forward(self, x, cosine, sine, write_mask, mask):
        for index in range(self.depth):
            x = self.layer(x, index, cosine, sine, write_mask, mask)
        return self.linear(rms_norm(x, self.weight("lm/norm/weight"), self.eps),
                           "lm/llm_decoder")[:, 0]


class QwenIO(Qwen):
    def __init__(self, tensors, metadata, context):
        super().__init__(tensors, metadata, context, persistent=False)

    def forward(self, x, cosine, sine, write_mask, mask, keys, values):
        next_keys, next_values = [], []
        key_layers = keys.reshape(self.depth, self.kv_heads, self.context, self.head_dim).unbind(0)
        value_layers = values.reshape(self.depth, self.kv_heads, self.context, self.head_dim).unbind(0)
        for index in range(self.depth):
            x, key, value = self.layer_step(x, index, cosine, sine, write_mask, mask,
                                            key_layers[index].unsqueeze(0), value_layers[index].unsqueeze(0))
            next_keys.append(key.float())
            next_values.append(value.float())
        logits = self.linear(rms_norm(x, self.weight("lm/norm/weight"), self.eps), "lm/llm_decoder")[:, 0]
        return logits, torch.stack(next_keys), torch.stack(next_values)


def rotate_first_head(x, cosine, sine, head_dim):
    first = x[..., :head_dim].reshape(x.shape[0], x.shape[1], head_dim // 2, 2)
    even, odd = first[..., 0], first[..., 1]
    rotated = torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), -1)
    return torch.cat((rotated.flatten(-2), x[..., head_dim:]), dim=-1)


def modulate(x, scale, shift):
    return x * (1 + scale[:, None]) + shift[:, None]


class Flow(Weights):
    def __init__(self, tensors, metadata):
        super().__init__({k: v for k, v in tensors.items()
                          if k.startswith("flow/") and k not in ("flow/rand_noise", "flow/input_embedding/weight")
                          and not k.startswith(("flow/pre_lookahead", "flow/spk_embed"))})
        self.depth = int(metadata.get("cosyvoice3.flow.depth", 22))
        self.heads = int(metadata.get("cosyvoice3.flow.heads", 16))
        self.head_dim = int(metadata.get("cosyvoice3.flow.dim_head", 64))
        self.groups = int(metadata.get("cosyvoice3.flow.conv_groups", 16))

    def project(self, hidden, prefix):
        if (prefix + "/attn/to_qkv/weight").replace("/", "__") in self.weights:
            return self.linear(hidden, prefix + "/attn/to_qkv").chunk(3, dim=-1)
        return tuple(self.linear(hidden, prefix + "/attn/to_" + name) for name in ("q", "k", "v"))

    def position_embedding(self, x):
        hidden = x.transpose(1, 2)
        for layer in range(2):
            name = f"flow/input_embed/conv_pos_embed/conv{layer + 1}/0"
            width = self.weight(name + "/weight").shape[-1]
            hidden = F.mish(self.conv(F.pad(hidden, (width - 1, 0)), name, groups=self.groups))
        return x + hidden.transpose(1, 2)

    def block(self, x, time, index, cosine, sine, mask):
        prefix = f"flow/blk/{index}"
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = self.linear(
            F.silu(time), prefix + "/attn_norm/linear").chunk(6, dim=-1)
        hidden = modulate(F.layer_norm(x, (x.shape[-1],), eps=NORM_EPS), scale_a, shift_a)
        q, k, v = self.project(hidden, prefix)
        q = rotate_first_head(q, cosine, sine, self.head_dim)
        k = rotate_first_head(k, cosine, sine, self.head_dim)
        shape = (x.shape[0], x.shape[1], self.heads, self.head_dim)
        q, k, v = (z.reshape(shape).transpose(1, 2) for z in (q, k, v))
        scores = (q.float() / math.sqrt(self.head_dim)) @ k.float().transpose(-1, -2) + mask.float()
        output = (scores.softmax(-1) @ v.float()).to(x.dtype).transpose(1, 2).flatten(-2)
        x = x + gate_a[:, None] * self.linear(output, prefix + "/attn/to_out/0")
        hidden = modulate(F.layer_norm(x, (x.shape[-1],), eps=NORM_EPS), scale_f, shift_f)
        hidden = F.gelu(self.linear(hidden, prefix + "/ff/ff/0/0"), approximate="tanh")
        return x + gate_f[:, None] * self.linear(hidden, prefix + "/ff/ff/2")

    def forward(self, x, mu, cond, spks, time_sin, cosine, sine, mask):
        time = self.linear(F.silu(self.linear(time_sin, "flow/time_embed/time_mlp/0")),
                           "flow/time_embed/time_mlp/2")
        speaker = spks[:, None].expand(-1, x.shape[1], -1)
        hidden = self.position_embedding(self.linear(torch.cat((x, cond, mu, speaker), -1),
                                                     "flow/input_embed/proj"))
        for index in range(self.depth):
            hidden = self.block(hidden, time, index, cosine, sine, mask)
        scale, shift = self.linear(F.silu(time), "flow/norm_out/linear").chunk(2, -1)
        hidden = modulate(F.layer_norm(hidden, (hidden.shape[-1],), eps=NORM_EPS), scale, shift)
        return self.linear(hidden, "flow/proj_out")


class FlowFrontend(Weights):
    def forward(self, token_ids, token_mask, speaker):
        embedded = F.embedding(token_ids.to(torch.int32), self.weight("flow/input_embedding/weight"))
        embedded = embedded * token_mask
        residual = embedded.transpose(1, 2)
        hidden = F.leaky_relu(self.conv(F.pad(residual, (0, 3)),
                                        "flow/pre_lookahead_layer/conv1"), 0.01)
        hidden = self.conv(F.pad(hidden, (2, 0)), "flow/pre_lookahead_layer/conv2") + residual
        mu = hidden.repeat_interleave(2, dim=-1).transpose(1, 2)
        speaker = speaker * torch.rsqrt(speaker.square().sum(-1, keepdim=True) + speaker.shape[-1] * 1e-12)
        return mu, self.linear(speaker, "flow/spk_embed_affine_layer")


class F0(Weights):
    def forward(self, mel):
        hidden = mel
        for index in range(F0_LAYERS):
            name = f"hift/f0_predictor/condnet/{index * 2}"
            padding = self.weight(name + "/weight").shape[-1] - 1
            hidden = F.elu(self.conv(F.pad(hidden, (0, padding) if index == 0 else (padding, 0)), name))
        return self.linear(hidden.transpose(1, 2), "hift/f0_predictor/classifier").abs().flatten(1)


class HiFT(Weights):
    def __init__(self, tensors):
        selected = dict(tensors)
        for field in ("weight", "bias"):
            full = selected.pop("hift/conv_post/" + field)
            magnitude, phase = np.split(full, 2, axis=0)
            selected["hift/conv_post_magnitude/" + field] = magnitude
            selected["hift/conv_post_phase/" + field] = phase
        super().__init__(selected)

    def snake(self, x, name):
        alpha = self.weight(name).reshape(1, -1, 1)
        return x + (alpha * x).sin().square() / (alpha + 1e-9)

    def residual(self, x, prefix):
        for index, dilation in enumerate(DILATIONS):
            hidden = self.snake(x, f"{prefix}/activations1/{index}/alpha")
            name = f"{prefix}/convs1/{index}"
            width = self.weight(name + "/weight").shape[-1]
            hidden = self.conv(F.pad(hidden, (dilation * (width - 1), 0)), name, dilation=dilation)
            hidden = self.snake(hidden, f"{prefix}/activations2/{index}/alpha")
            name = f"{prefix}/convs2/{index}"
            hidden = self.conv(F.pad(hidden, (width - 1, 0)), name)
            x = x + hidden
        return x

    def upsample(self, x, source, index):
        name = f"hift/ups/{index}"
        width = self.weight(name + "/weight").shape[-1]
        hidden = F.interpolate(F.leaky_relu(x, 0.1), scale_factor=UPSAMPLE_RATES[index], mode="nearest")
        hidden = self.conv(F.pad(hidden, (width - 1, 0)), name)
        if index == len(UPSAMPLE_RATES) - 1:
            hidden = F.pad(hidden, (1, 0), mode="reflect")
        stride = SOURCE_STRIDES[index]
        source = self.conv(F.pad(source, (stride - 1, 0)), f"hift/source_downs/{index}", stride=stride)
        source = self.residual(source, f"hift/source_resblocks/{index}")[..., :hidden.shape[-1]]
        hidden = hidden + source
        return sum(self.residual(hidden, f"hift/resblocks/{index * 3 + kernel}")
                   for kernel in range(len(RESBLOCK_KERNELS))) / len(RESBLOCK_KERNELS)

    def forward(self, mel, source_stft):
        hidden = self.conv(F.pad(mel, (0, 4)), "hift/conv_pre")
        for index in range(len(UPSAMPLE_RATES)):
            hidden = self.upsample(hidden, source_stft, index)
        hidden = F.pad(F.leaky_relu(hidden, 0.01), (6, 0))
        magnitude = self.conv(hidden, "hift/conv_post_magnitude")
        phase = self.conv(hidden, "hift/conv_post_phase")
        magnitude = magnitude.clamp(-1e6, 1e2).exp()
        phase = phase.sin()
        return torch.cat((magnitude * phase.cos(), magnitude * phase.sin()), dim=1)
