import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ENCODER_TYPES = (0, 2, 1, 2, 1)
ENCODER_STRIDES = (2, 2, 1, 1, 1)
ENCODER_GROUPS = (1, 2, 2, 2, 2)
ENCODER_CHANNELS = (12, 24, 24, 32, 16)


def parameter(array):
    return nn.Parameter(torch.from_numpy(np.array(array, dtype=np.float32, copy=True)),
                        requires_grad=False)


def read_gguf(path, architecture):
    from gguf import GGUFReader

    reader = GGUFReader(str(path))
    if reader.fields['general.architecture'].contents() != architecture:
        raise ValueError(f'{path}: expected {architecture}')
    tensors = {}
    for tensor in reader.tensors:
        if tensor.tensor_type.name not in ('F32', 'F16'):
            raise ValueError(f'{path}: unsupported tensor type {tensor.tensor_type.name}')
        shape = tuple(int(d) for d in reversed(tensor.shape))
        tensors[tensor.name] = np.asarray(tensor.data, dtype=np.float32).reshape(shape)
    prefix = architecture.replace('-', '.') + '.'
    metadata = {name[len(prefix):]: field.contents() for name, field in reader.fields.items()
                if name.startswith(prefix)}
    return tensors, metadata


def linear(tensors, prefix):
    weight = tensors[prefix + '.weight']
    layer = nn.Linear(weight.shape[1], weight.shape[0], bias=prefix + '.bias' in tensors)
    layer.weight = parameter(weight)
    if layer.bias is not None:
        layer.bias = parameter(tensors[prefix + '.bias'])
    return layer


def norm(tensors, prefix, eps):
    layer = nn.LayerNorm(tuple(tensors[prefix + '.weight'].shape), eps=eps)
    layer.weight = parameter(tensors[prefix + '.weight'])
    layer.bias = parameter(tensors[prefix + '.bias'])
    return layer


def conv1d(tensors, prefix, groups=1):
    weight = tensors[prefix + '.weight']
    layer = nn.Conv1d(weight.shape[1] * groups, weight.shape[0], weight.shape[2],
                      padding=weight.shape[2] // 2, groups=groups)
    layer.weight = parameter(weight)
    layer.bias = parameter(tensors[prefix + '.bias'])
    return layer


class EnhancerBlock(nn.Module):
    def __init__(self, tensors, prefix, eps):
        super().__init__()
        channels = tensors[prefix + '.dwconv.weight'].shape[0]
        self.dwconv = conv1d(tensors, prefix + '.dwconv', channels)
        self.norm = norm(tensors, prefix + '.norm', eps)
        self.pw1 = linear(tensors, prefix + '.pwconv1')
        self.pw2 = linear(tensors, prefix + '.pwconv2')
        self.gamma = parameter(tensors[prefix + '.gamma'].reshape(1, 1, -1))

    def forward(self, x, valid):
        y = self.norm(self.dwconv(x).transpose(1, 2))
        y = self.pw2(F.gelu(self.pw1(y), approximate='none')) * self.gamma
        return (x + y.transpose(1, 2)) * valid


class Enhancer(nn.Module):
    def __init__(self, tensors, metadata):
        super().__init__()
        eps = float(metadata['layernorm_eps'])
        self.embed = conv1d(tensors, 'enhancer.embed')
        self.norm = norm(tensors, 'enhancer.norm', eps)
        self.blocks = nn.ModuleList(EnhancerBlock(tensors, f'enhancer.block.{i}', eps)
                                    for i in range(int(metadata['n_blocks'])))
        self.final_norm = norm(tensors, 'enhancer.final_norm', eps)
        self.head = linear(tensors, 'spec_head.out')
        self.clip_max = float(metadata['clip_max'])
        self.context = ((int(metadata['kernel']) - 1) // 2) * (len(self.blocks) + 1)

    def forward(self, mel, valid):
        x = self.norm(self.embed(mel).transpose(1, 2)).transpose(1, 2) * valid
        for block in self.blocks:
            x = block(x, valid)
        logits = self.head(self.final_norm(x.transpose(1, 2))).transpose(1, 2)
        magnitude, phase = torch.chunk(logits, 2, dim=1)
        magnitude = torch.exp(magnitude).clamp(max=self.clip_max)
        return magnitude * torch.cos(phase) * valid, magnitude * torch.sin(phase) * valid


def gru(tensors, prefix, bidirectional=False):
    weight = tensors[prefix + '.weight_ih_l0']
    layer = nn.GRU(weight.shape[1], weight.shape[0] // 3, batch_first=True,
                   bidirectional=bidirectional)
    for name, _ in list(layer.named_parameters()):
        setattr(layer, name, parameter(tensors[prefix + '.' + name]))
    return layer


class Attention(nn.Module):
    def __init__(self, tensors, prefix, ratio):
        super().__init__()
        self.temporal = gru(tensors, prefix + '.ta_gru')
        self.temporal_fc = linear(tensors, prefix + '.ta_fc')
        self.frequency = gru(tensors, prefix + '.fa.gru', True)
        self.frequency_fc = linear(tensors, prefix + '.fa.fc')
        self.ratio = ratio

    def forward(self, x):
        batch, channels, frames, bins = x.shape
        temporal, _ = self.temporal(x.square().mean(dim=3).transpose(1, 2))
        temporal = self.temporal_fc(temporal).sigmoid().transpose(1, 2).unsqueeze(3)
        frequency = F.pad(x.square().mean(dim=1), (0, (-bins) % self.ratio))
        frequency = frequency.reshape(batch * frames, -1, self.ratio)
        frequency, _ = self.frequency(frequency)
        frequency = self.frequency_fc(frequency).sigmoid()
        frequency = frequency.reshape(batch, frames, -1)[:, :, :bins].unsqueeze(1)
        return temporal * x * frequency


class AffinePReLU(nn.Module):
    def __init__(self, tensors, prefix):
        super().__init__()
        self.weight = parameter(tensors[prefix + '.affine_weight'][None, :, None, :])
        self.bias = parameter(tensors[prefix + '.affine_bias'][None, :, None, :])
        self.slope = parameter(tensors[prefix + '.slope_weight'].reshape(1, -1, 1, 1))

    def forward(self, x):
        return self.weight * x + self.bias + torch.where(x > 0, x, x * self.slope)


class Conv2D(nn.Module):
    def __init__(self, tensors, prefix, stride, groups, transpose=False):
        super().__init__()
        weight = tensors[prefix + '.weight']
        self.weight = parameter(weight)
        self.bias = parameter(tensors[prefix + '.bias']) if prefix + '.bias' in tensors else None
        self.stride = stride
        self.groups = groups
        self.transpose = transpose
        self.time_pad = weight.shape[2] - 1
        self.frequency_pad = weight.shape[3] // 2

    def forward(self, x):
        x = F.pad(x, (0, 0, self.time_pad, 0))
        if self.transpose:
            return F.conv_transpose2d(x, self.weight, self.bias, stride=(1, self.stride),
                                      padding=(self.time_pad, self.frequency_pad), groups=self.groups)
        return F.conv2d(x, self.weight, self.bias, stride=(1, self.stride),
                         padding=(0, self.frequency_pad), groups=self.groups)


def batchnorm(tensors, prefix, eps):
    scale = tensors[prefix + '.weight'] / np.sqrt(tensors[prefix + '.running_var'] + eps)
    bias = tensors[prefix + '.bias'] - tensors[prefix + '.running_mean'] * scale
    return scale, bias


class ConvStage(nn.Module):
    def __init__(self, tensors, prefix, stride, groups, transpose, eps, activation):
        super().__init__()
        self.conv = Conv2D(tensors, prefix + '.1',
                           stride, groups, transpose)
        scale, bias = batchnorm(tensors, prefix + '.2', eps)
        self.scale = parameter(scale.reshape(1, -1, 1, 1))
        self.bias = parameter(bias.reshape(1, -1, 1, 1))
        self.activation = AffinePReLU(tensors, prefix + '.3') if activation else nn.Identity()

    def forward(self, x):
        return self.activation(self.conv(x) * self.scale + self.bias)


class PointStage(nn.Module):
    def __init__(self, tensors, prefix, groups, eps, activation=True):
        super().__init__()
        self.conv = Conv2D(tensors, prefix + '.0', 1, groups)
        scale, bias = batchnorm(tensors, prefix + '.1', eps)
        self.scale = parameter(scale.reshape(1, -1, 1, 1))
        self.bias = parameter(bias.reshape(1, -1, 1, 1))
        self.activation = AffinePReLU(tensors, prefix + '.2') if activation else nn.Identity()

    def forward(self, x):
        return self.activation(self.conv(x) * self.scale + self.bias)


def shuffle(x):
    batch, channels, frames, bins = x.shape
    return x.reshape(batch, 2, channels // 2, frames, bins).transpose(1, 2).reshape_as(x)


class Block(nn.Module):
    def __init__(self, tensors, prefix, kind, stride, groups, channels, transpose,
                 last, metadata):
        super().__init__()
        eps = float(metadata['batchnorm_eps'])
        self.kind, self.groups, self.last = kind, groups, last
        self.point1 = PointStage(tensors, prefix + ('.pconv' if kind == 1 else '.pconv1'),
                                 groups, eps) if kind != 0 else nn.Identity()
        self.spatial = ConvStage(tensors, prefix + ('.ops' if kind == 0 else '.dconv'),
                                  stride, groups if kind == 0 else channels,
                                  transpose, eps, not last if kind != 2 else True)
        self.point2 = PointStage(tensors, prefix + '.pconv2', groups, eps, False) if kind == 2 else nn.Identity()
        attention_prefix = '.ops.4' if kind == 0 else ('.dconv.4' if kind == 1 else '.pconv2.2')
        self.attention = Attention(tensors, prefix + attention_prefix, int(metadata['freq_comp_ratio']))

    def forward(self, x):
        y = self.point1(x)
        if self.kind != 0 and self.groups == 2:
            y = shuffle(y)
        y = self.attention(self.point2(self.spatial(y)))
        if self.kind == 2 and y.shape == x.shape:
            y = y + x
        if self.groups == 2 and not self.last and self.kind != 1:
            y = shuffle(y)
        return y


class GroupRNN(nn.Module):
    def __init__(self, tensors, prefix, bidirectional):
        super().__init__()
        self.first = gru(tensors, prefix + '.rnn1', bidirectional)
        self.second = gru(tensors, prefix + '.rnn2', bidirectional)

    def forward(self, x):
        first, second = torch.chunk(x, 2, dim=-1)
        first, _ = self.first(first)
        second, _ = self.second(second)
        return torch.cat((first, second), dim=-1)


class DualPath(nn.Module):
    def __init__(self, tensors, prefix, eps):
        super().__init__()
        self.intra = GroupRNN(tensors, prefix + '.intra_rnn', True)
        self.inter = GroupRNN(tensors, prefix + '.inter_rnn', False)
        self.intra_fc = linear(tensors, prefix + '.intra_fc')
        self.inter_fc = linear(tensors, prefix + '.inter_fc')
        self.intra_norm = norm(tensors, prefix + '.intra_ln', eps)
        self.inter_norm = norm(tensors, prefix + '.inter_ln', eps)

    def forward(self, x):
        batch, channels, frames, bins = x.shape
        y = x.permute(0, 2, 3, 1)
        z = self.intra_fc(self.intra(y.reshape(batch * frames, bins, channels))).reshape_as(y)
        y = y + self.intra_norm(z)
        z = y.transpose(1, 2).reshape(batch * bins, frames, channels)
        z = self.inter_fc(self.inter(z)).reshape(batch, bins, frames, channels).transpose(1, 2)
        return (y + self.inter_norm(z)).permute(0, 3, 1, 2)


class LogMagnitude(nn.Module):
    def forward(self, real, imag):
        return torch.log10(torch.sqrt(real.square() + imag.square()).clamp(min=1e-12))


class Denoiser(nn.Module):
    def __init__(self, tensors, metadata):
        super().__init__()
        self.log_magnitude = LogMagnitude()
        self.erb = parameter(tensors['erb.erb_fc.weight'])
        self.ierb = parameter(tensors['erb.ierb_fc.weight'])
        self.low = int(metadata['erb_low'])
        self.encoder = nn.ModuleList(Block(tensors, f'encoder.en_convs.{i}', ENCODER_TYPES[i],
            ENCODER_STRIDES[i], ENCODER_GROUPS[i], ENCODER_CHANNELS[i], False, False, metadata)
            for i in range(len(ENCODER_TYPES)))
        self.decoder = nn.ModuleList(Block(tensors, f'decoder.de_convs.{j}', ENCODER_TYPES[i],
            ENCODER_STRIDES[i], ENCODER_GROUPS[i], ENCODER_CHANNELS[i - 1] if i else 1,
            True, i == 0, metadata) for j, i in enumerate(reversed(range(len(ENCODER_TYPES)))))
        self.dual = nn.Sequential(*(DualPath(tensors, f'dpgrnn.{i}', float(metadata['layernorm_eps']))
                                     for i in range(2)))

    def forward(self, real, imag):
        x = self.log_magnitude(real, imag)
        x = torch.cat((x[..., :self.low], F.linear(x[..., self.low:], self.erb)), dim=-1)
        skips = []
        for block in self.encoder:
            x = block(x)
            skips.append(x)
        x = self.dual(x)
        for block, skip in zip(self.decoder, reversed(skips)):
            x = block(x + skip)
        x = x.sigmoid()
        mask = torch.cat((x[..., :self.low], F.linear(x[..., self.low:], self.ierb)), dim=-1)
        return real * mask, imag * mask
