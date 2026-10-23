from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


TILE = 2048


def stable_silu(x):
    negative = torch.minimum(x, torch.zeros_like(x))
    return x*torch.exp(negative)/(1+torch.exp(-torch.abs(x)))


def stable_rms_normalize(x, dimension, eps):
    maximum = x.abs().amax(dimension, keepdim=True)
    root_eps = torch.full_like(maximum, eps**0.5)
    scale = torch.maximum(maximum, root_eps)
    scaled = x/scale
    relative_eps = root_eps/scale
    return scaled*torch.rsqrt(scaled.square().mean(dimension, keepdim=True)+relative_eps.square())


def compact_lm_head(tensors, semantic_offset, semantic_vocabulary, eos):
    weight = tensors["output.weight"]
    if semantic_offset < 0 or semantic_vocabulary <= 0 or semantic_offset+semantic_vocabulary > weight.shape[0]:
        raise ValueError("Semantic output rows are outside the original vocabulary")
    if eos < 0 or eos >= weight.shape[0]:
        raise ValueError("EOS output row is outside the original vocabulary")
    return np.concatenate([weight[semantic_offset:semantic_offset+semantic_vocabulary], weight[[eos]]])


class Weights(torch.nn.Module):
    def __init__(self, tensors):
        super().__init__()
        self.names = {}
        for index, (name, value) in enumerate(tensors.items()):
            key = f"weight_{index}"
            self.register_buffer(key, torch.from_numpy(np.array(value, copy=True)).float())
            self.names[name] = key

    def w(self, name):
        return getattr(self, self.names[name])

    def linear(self, x, name, bias=False):
        weight = self.w(name + ".weight")
        offset = self.w(name + ".bias") if bias else None
        return torch.cat([F.conv2d(x, weight[first:first+TILE, :, None, None],
                                  offset[first:first+TILE] if offset is not None else None)
                          for first in range(0, weight.shape[0], TILE)], 1)

    def rms(self, x, name, eps):
        return stable_rms_normalize(x, 1, eps) * self.w(name)[None, :, None, None]

    def layer_norm(self, x, name, eps):
        centered = x - x.mean(1, keepdim=True)
        return centered * torch.rsqrt(centered.square().mean(1, keepdim=True) + eps) * self.w(
            name + ".weight")[None, :, None, None] + self.w(name + ".bias")[None, :, None, None]

    def swiglu(self, x):
        gate, up, down = [self.w(name + ".weight") for name in ("ffn_gate", "ffn_up", "ffn_down")]
        partials = []
        for first in range(0, gate.shape[0], TILE):
            g = F.conv2d(x, gate[first:first+TILE, :, None, None])
            u = F.conv2d(x, up[first:first+TILE, :, None, None])
            partials.append(F.conv2d(g / (1 + torch.exp(-g)) * u, down[:, first:first+TILE, None, None]))
        return sum(partials[1:], partials[0])


class DepthBlock(Weights):
    def __init__(self, tensors, config, capacity):
        super().__init__(tensors)
        self.eps = config["rms_eps"]
        self.dim = config["head_dim"]
        self.capacity = capacity
        self.register_buffer("key_cache", torch.zeros(1, config["embedding_length"], 2, capacity, dtype=torch.float16))
        self.register_buffer("value_cache", torch.zeros_like(self.key_cache))

    def forward(self, hidden, selector, mask):
        x = self.rms(hidden, "attn_norm.weight", self.eps)
        q = self.linear(x, "attn_q").transpose(2, 3)
        k = self.linear(x, "attn_k").transpose(2, 3)
        v = self.linear(x, "attn_v").transpose(2, 3)
        selector = selector.reshape(1, 1, 1, self.capacity)
        self.key_cache.mul_(1-selector)
        self.key_cache.add_(k*selector)
        self.value_cache.mul_(1-selector)
        self.value_cache.add_(v*selector)
        heads = []
        for a, b, c in zip(q.split(self.dim, 1), self.key_cache.split(self.dim, 1), self.value_cache.split(self.dim, 1)):
            scores = (a*b.float()).sum(1, keepdim=True).permute(0, 3, 2, 1) / self.dim**0.5
            probability = (scores+mask).softmax(1).permute(0, 3, 2, 1)
            heads.append((probability*c.float()).sum(3, keepdim=True))
        x = hidden + self.linear(torch.cat(heads, 1).transpose(2, 3), "attn_output")
        return x + self.swiglu(self.rms(x, "ffn_norm.weight", self.eps))


class LMBlock(Weights):
    def __init__(self, tensors, config, capacity, stable_activation=False):
        super().__init__(tensors)
        self.eps = config["attention.layer_norm_rms_epsilon"]
        self.dim = config["attention.key_length"]
        self.heads = config["attention.head_count"]
        self.kv_heads = config["attention.head_count_kv"]
        self.capacity = capacity
        self.stable_activation = stable_activation
        self.register_buffer("key_cache", torch.zeros(1, self.kv_heads*self.dim, 2, capacity, dtype=torch.float16))
        self.register_buffer("value_cache", torch.zeros_like(self.key_cache))

    def linear(self, x, name, bias=False):
        return super().linear(x*64.0, name, bias)/64.0

    def swiglu(self, x):
        gate, up, down = [self.w(name + ".weight") for name in ("ffn_gate", "ffn_up", "ffn_down")]
        partials = []
        for first in range(0, gate.shape[0], TILE):
            g = F.conv2d(x*64.0, gate[first:first+TILE, :, None, None])/64.0
            u = F.conv2d(x*64.0, up[first:first+TILE, :, None, None])/64.0
            activated = stable_silu(g) if self.stable_activation else g/(1+torch.exp(-g))
            partials.append(F.conv2d(activated*u, down[:, first:first+TILE, None, None]))
        return sum(partials[1:], partials[0])

    def qk_norm_rope(self, x, norm, cosine, sine):
        shape = x.shape
        x = x.reshape(1, shape[1]//self.dim, self.dim, 2)
        x = stable_rms_normalize(x, 2, self.eps) * self.w(norm)[None, None, :, None]
        low, high = x[:, :, :self.dim//2], x[:, :, self.dim//2:]
        return torch.cat([low*cosine-high*sine, high*cosine+low*sine], 2).reshape(shape)

    def forward(self, hidden, selector, mask, cosine, sine):
        x = self.rms(hidden, "attn_norm.weight", self.eps)
        q = self.qk_norm_rope(self.linear(x, "attn_q"), "attn_q_norm.weight", cosine, sine).transpose(2, 3)
        k = self.qk_norm_rope(self.linear(x, "attn_k"), "attn_k_norm.weight", cosine, sine).transpose(2, 3)
        v = self.linear(x, "attn_v").transpose(2, 3)
        selector = selector.reshape(1, 1, 1, self.capacity)
        self.key_cache.mul_(1-selector)
        self.key_cache.add_(k*selector)
        self.value_cache.mul_(1-selector)
        self.value_cache.add_(v*selector)
        keys, values = self.key_cache.split(self.dim, 1), self.value_cache.split(self.dim, 1)
        heads = []
        group = self.heads//self.kv_heads
        for index, query in enumerate(q.split(self.dim, 1)):
            scores = (query*keys[index//group].float()).sum(1, keepdim=True).permute(0, 3, 2, 1) / self.dim**0.5
            probability = (scores+mask).softmax(1).permute(0, 3, 2, 1)
            heads.append((probability*values[index//group].float()).sum(3, keepdim=True))
        x = hidden + self.linear(torch.cat(heads, 1).transpose(2, 3), "attn_output")
        return x + self.swiglu(self.rms(x, "ffn_norm.weight", self.eps))


class LMBlockPair(torch.nn.Module):
    def __init__(self, tensors, config, capacity, stable_activation=False):
        super().__init__()
        if len(tensors) != 2:
            raise ValueError("An LM block pair requires two consecutive blocks")
        self.blocks = torch.nn.ModuleList([LMBlock(block, config, capacity, stable_activation) for block in tensors])

    def forward(self, hidden, selector, mask, cosine, sine):
        for block in self.blocks:
            hidden = block(hidden, selector, mask, cosine, sine)
        return hidden


class DepthInput(Weights):
    def forward(self, hidden, positional):
        return self.linear(hidden, "proj") + positional


class Head(Weights):
    def __init__(self, tensors, eps):
        super().__init__(tensors)
        self.eps = eps

    def forward(self, hidden):
        hidden = self.rms(hidden, "norm", self.eps)
        return hidden, self.linear(hidden, "head")


class Feedback(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, embeddings):
        return embeddings.sum(2, keepdim=True)*self.scale


class Conditioner(Weights):
    def forward(self, hidden):
        mix = self.w("layer_logits").softmax(0)[None, None, :, None]*self.w("layer_scale").reshape(1, 1, 1, 1)
        mixed = (hidden*mix).sum(2, keepdim=True)
        return F.conv2d(mixed, self.w("proj.weight")[:, :, None, :], self.w("proj.bias"), padding=(0, 1))


class DiTBlock(Weights):
    def __init__(self, tensors, config, sequence):
        super().__init__(tensors)
        self.eps = config["layer_norm_eps"]
        self.head_dim = config["head_dim"]
        self.heads = config["head_count"]
        self.rope_dim = config["rope_dim"]
        self.inner = config["ff_inner"]
        frequencies = config["rope_theta"]**(-torch.arange(0, self.rope_dim, 2).float()/self.rope_dim)
        phase = frequencies[:, None]*torch.arange(sequence).float()[None]
        self.register_buffer("cosine", phase.cos()[None, None])
        self.register_buffer("sine", phase.sin()[None, None])

    def rope(self, x):
        shape = x.shape
        x = x.reshape(shape[0], self.heads, self.head_dim, shape[-1])
        half = self.rope_dim//2
        low, high = x[:, :, :half], x[:, :, half:self.rope_dim]
        return torch.cat([low*self.cosine-high*self.sine, high*self.cosine+low*self.sine,
                          x[:, :, self.rope_dim:]], 2).reshape(shape)

    def forward(self, hidden, mask):
        q, k, v = self.linear(self.layer_norm(hidden, "attn_norm", self.eps), "attn_qkv").chunk(3, 1)
        q, k = self.rope(q), self.rope(k)
        outputs = []
        for a, b, c in zip(q.split(self.head_dim, 1), k.split(self.head_dim, 1), v.split(self.head_dim, 1)):
            scores = torch.einsum("bchq,bchk->bkhq", a, b)/self.head_dim**0.5
            outputs.append(torch.einsum("bkhq,bchk->bchq", (scores+mask).softmax(1), c))
        x = hidden + self.linear(torch.cat(outputs, 1), "attn_output")
        normalized = self.layer_norm(x, "ffn_norm", self.eps)
        weight, bias = self.w("ffn_in.weight"), self.w("ffn_in.bias")
        partials = []
        for first in range(0, self.inner, TILE):
            end = min(first+TILE, self.inner)
            signal = F.conv2d(normalized, weight[first:end, :, None, None], bias[first:end])
            gate = F.conv2d(normalized, weight[self.inner+first:self.inner+end, :, None, None], bias[self.inner+first:self.inner+end])
            partials.append(F.conv2d(signal*gate/(1+torch.exp(-gate)), self.w("ffn_out.weight")[:, first:end, None, None]))
        return x + sum(partials[1:], partials[0]) + self.w("ffn_out.bias")[None, :, None, None]


class DiTInput(Weights):
    def forward(self, latent, condition, fourier, gates):
        time = self.linear(fourier, "time_embd.0", True)
        time = self.linear(time/(1+torch.exp(-time)), "time_embd.1", True)
        x = torch.cat([latent.expand(2, -1, -1, -1), torch.zeros_like(latent).expand(2, -1, -1, -1),
                       condition.expand(2, -1, -1, -1)*gates], 1)
        x = x + F.conv2d(x, self.w("preprocess_conv.weight")[:, :, None, :])
        return torch.cat([time.expand(2, -1, -1, -1), self.linear(x, "proj_in")], 3)


class DiTOutput(Weights):
    def forward(self, hidden):
        x = self.linear(hidden[:, :, :, 1:], "proj_out")
        return x + F.conv2d(x, self.w("postprocess_conv.weight")[:, :, None, :])


def phase_weight(w, stride):
    output = w.shape[1]
    result = torch.zeros(stride*output, w.shape[0], 3)
    for phase in range(stride):
        rows = slice(phase*output, (phase+1)*output)
        result[rows, :, 1] = w[:, :, phase+stride//2].T
        if phase < stride//2:
            result[rows, :, 0] = w[:, :, phase+stride+stride//2].T
        else:
            result[rows, :, 2] = w[:, :, phase-stride//2].T
    return result


class Vocoder(Weights):
    def __init__(self, tensors, config):
        super().__init__(tensors)
        self.rates = config["upsample_rates"]
        self.dilations = config["res_dilations"]
        self.eps = config["snake_eps"]
        for index, stride in enumerate(self.rates):
            self.register_buffer(f"phase_{index}", phase_weight(self.w(f"blk.{index}.convt.weight"), stride))

    def snake(self, x, name):
        alpha = self.w(name).reshape(1, -1, 1, 1)
        return x + torch.sin(alpha*x).square()/(alpha+self.eps)

    def conv(self, x, name, dilation=1):
        weight = self.w(name+".weight")
        return self.convolve(x, weight, self.w(name+".bias"), dilation)

    def convolve(self, x, weight, bias, dilation=1):
        padding = weight.shape[-1]//2*dilation
        if x.shape[-1] < 1024:
            return F.conv2d(x, weight[:, :, None, :], bias, padding=(0, padding), dilation=(1, dilation))
        batch, channels, _, sequence = x.shape
        width = 256
        tiled = x.reshape(batch, channels, sequence//width, width)
        if padding:
            left = F.pad(tiled[:, :, :, -padding:], (0, 0, 1, 0))[:, :, :-1]
            right = F.pad(tiled[:, :, :, :padding], (0, 0, 0, 1))[:, :, 1:]
            tiled = torch.cat([left, tiled, right], 3)
        output = F.conv2d(tiled, weight[:, :, None, :], bias, dilation=(1, dilation))
        return output.reshape(batch, weight.shape[0], 1, sequence)

    def forward(self, latent, valid):
        x = self.conv(self.conv(latent, "dec_in")*valid, "conv_in")*valid
        mask = valid
        for index, stride in enumerate(self.rates):
            x = self.snake(x, f"blk.{index}.snake.alpha")*mask
            weight = getattr(self, f"phase_{index}")
            x = self.convolve(x, weight, self.w(f"blk.{index}.convt.bias").repeat(stride))
            batch, channels, _, frames = x.shape
            x = x.reshape(batch, stride, channels//stride, frames).permute(0, 2, 3, 1).reshape(batch, channels//stride, 1, frames*stride)
            mask = mask.repeat_interleave(stride, dim=3)
            x = x*mask
            for residual, dilation in enumerate(self.dilations):
                prefix = f"blk.{index}.res.{residual}."
                value = self.conv(self.snake(x, prefix+"snake1.alpha")*mask, prefix+"conv1", dilation)*mask
                value = self.conv(self.snake(value, prefix+"snake2.alpha")*mask, prefix+"conv2")*mask
                x = (x+value)*mask
        return torch.tanh(self.conv(self.snake(x, "snake_out.alpha")*mask, "conv_out"))*mask
