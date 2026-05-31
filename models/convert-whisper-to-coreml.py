import argparse
import copy
import numpy as np
import shutil
import torch
import torch.nn.functional as F
import coremltools as ct

from dataclasses import replace
from torch import Tensor
from torch import nn
from typing import Dict
from typing import Optional
from ane_transformers.reference.layer_norm import LayerNormANE as LayerNormANEBase
from coremltools.models.neural_network.quantization_utils import quantize_weights
from whisper.model import Whisper, AudioEncoder, TextDecoder, ResidualAttentionBlock, MultiHeadAttention, ModelDimensions
from whisper import load_model

# Disable PyTorch Scaled Dot-Product Attention (SDPA) to avoid compatibility issues.
# The Whisper implementation expects a specific behavior from
# torch.nn.functional.scaled_dot_product_attention that differs between PyTorch
# versions. Setting use_sdpa=False forces Whisper to use its manual attention
# implementation instead, which is more stable across different PyTorch versions
# (2.5.0 required by coremltools vs newer versions).
import whisper.model
whisper.model.MultiHeadAttention.use_sdpa = False

# Use for changing dim of input in encoder and decoder embeddings
def linear_to_conv2d_map(state_dict, prefix, local_metadata, strict,
                         missing_keys, unexpected_keys, error_msgs):
    """
    Unsqueeze twice to map nn.Linear weights to nn.Conv2d weights
    """
    for k in state_dict:
        is_attention = all(substr in k for substr in ['attn', '.weight'])
        is_mlp = any(k.endswith(s) for s in ['mlp.0.weight', 'mlp.2.weight'])

        if (is_attention or is_mlp) and len(state_dict[k].shape) == 2:
            state_dict[k] = state_dict[k][:, :, None, None]


def correct_for_bias_scale_order_inversion(state_dict, prefix, local_metadata,
                                           strict, missing_keys,
                                           unexpected_keys, error_msgs):
    state_dict[prefix + 'bias'] = state_dict[prefix + 'bias'] / state_dict[prefix + 'weight']
    return state_dict

class LayerNormANE(LayerNormANEBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._register_load_state_dict_pre_hook(
            correct_for_bias_scale_order_inversion)

class MultiHeadAttentionANE(MultiHeadAttention):
    def __init__(self, n_state: int, n_head: int):
        super().__init__(n_state, n_head)
        self.query =  nn.Conv2d(n_state, n_state, kernel_size=1)
        self.key = nn.Conv2d(n_state, n_state, kernel_size=1, bias=False)
        self.value = nn.Conv2d(n_state, n_state, kernel_size=1)
        self.out = nn.Conv2d(n_state, n_state, kernel_size=1)

    def forward(self,
                x: Tensor,
                xa: Optional[Tensor] = None,
                mask: Optional[Tensor] = None,
                kv_cache: Optional[dict] = None):

        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)

        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention_ane(q, k, v, mask)

        return self.out(wv), qk

    def qkv_attention_ane(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None):

        _, dim, _, seqlen = q.size()

        dim_per_head = dim // self.n_head

        scale = float(dim_per_head)**-0.5

        q = q * scale

        mh_q = q.split(dim_per_head, dim=1)
        mh_k = k.transpose(1,3).split(dim_per_head, dim=3)
        mh_v = v.split(dim_per_head, dim=1)

        mh_qk = [
            torch.einsum('bchq,bkhc->bkhq', [qi, ki])
            for qi, ki in zip(mh_q, mh_k)
        ]  # (batch_size, max_seq_length, 1, max_seq_length) * n_heads

        if mask is not None:
            for head_idx in range(self.n_head):
                mh_qk[head_idx] = mh_qk[head_idx] + mask[:, :seqlen, :, :seqlen]

        attn_weights = [aw.softmax(dim=1) for aw in mh_qk]  # (batch_size, max_seq_length, 1, max_seq_length) * n_heads
        attn = [torch.einsum('bkhq,bchk->bchq', wi, vi) for wi, vi in zip(attn_weights, mh_v)]  # (batch_size, dim_per_head, 1, max_seq_length) * n_heads
        attn = torch.cat(attn, dim=1)  # (batch_size, dim, 1, max_seq_length)

        return attn, torch.cat(mh_qk, dim=1).float().detach()


class ResidualAttentionBlockANE(ResidualAttentionBlock):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__(n_state, n_head, cross_attention)
        self.attn =  MultiHeadAttentionANE(n_state, n_head)
        self.attn_ln = LayerNormANE(n_state)
        self.cross_attn =  MultiHeadAttentionANE(n_state, n_head) if cross_attention else None
        self.cross_attn_ln =  LayerNormANE(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp =  nn.Sequential(
            nn.Conv2d(n_state, n_mlp, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(n_mlp, n_state, kernel_size=1)
        )
        self.mlp_ln = LayerNormANE(n_state)


class AudioEncoderANE(AudioEncoder):
    def __init__(self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int):
        super().__init__(n_mels, n_ctx, n_state, n_head, n_layer)

        self.blocks = nn.ModuleList(
            [ResidualAttentionBlockANE(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNormANE(n_state)

    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))

        assert x.shape[1:] == self.positional_embedding.shape[::-1], "incorrect audio shape"

        # Add positional embedding and add dummy dim for ANE
        x = (x + self.positional_embedding.transpose(0,1)).to(x.dtype).unsqueeze(2)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        x = x.squeeze(2).transpose(1, 2)

        return x

class TextDecoderANE(TextDecoder):

    def __init__(self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int):
        super().__init__(n_vocab, n_ctx, n_state, n_head, n_layer)

        self.blocks= nn.ModuleList(
            [ResidualAttentionBlockANE(n_state, n_head, cross_attention=True) for _ in range(n_layer)]
        )
        self.ln= LayerNormANE(n_state)

    def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_mels, n_audio_ctx)
            the encoded audio features to be attended on
        """
        offset = next(iter(kv_cache.values())).shape[3] if kv_cache else 0
        x = self.token_embedding(x) + self.positional_embedding[offset : offset + x.shape[-1]]
        x = x.to(xa.dtype)

        # Reformat for ANE
        mask = self.mask[None, None, :, :].permute(0,3,1,2)
        x = x.transpose(1,2).unsqueeze(2)

        for block in self.blocks:
            x = block(x, xa, mask=mask, kv_cache=kv_cache)

        x = self.ln(x)

        # Reformat back from ANE
        x = x.permute(0,2,3,1).squeeze(0)

        # ANE can only load tensors with dim size of at most 16,384 - whisper uses 51,864 (en) or 51,865 (multi-lang) tokens so we need to compute in chunks
        if self.token_embedding.weight.shape[0] >= 51865:
            # split in 11 chunks - 4715 each
            splits = self.token_embedding.weight.split(self.token_embedding.weight.shape[0]//11, dim=0)
            logits = torch.cat([torch.einsum('bid,jd->bij', x, split) for split in splits]).view(*x.shape[:2], -1)
        else:
            # split in 12 chunks - 4322 each
            assert(self.token_embedding.weight.shape[0] == 51864)
            splits = self.token_embedding.weight.split(self.token_embedding.weight.shape[0]//12, dim=0)
            logits = torch.cat([torch.einsum('bid,jd->bij', x, split) for split in splits]).view(*x.shape[:2], -1)

        return logits

class WhisperANE(Whisper):
    def __init__(self, dims: ModelDimensions):
        super().__init__(dims)

        self.encoder = AudioEncoderANE(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )
        self.decoder = TextDecoderANE(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
        )

        self._register_load_state_dict_pre_hook(linear_to_conv2d_map)

    def forward(self, mel: torch.Tensor, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.decoder(tokens, self.encoder(mel))

    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[3] > self.decoder.positional_embedding.shape[0]:
                cache[module] = output  # save as-is, for the first token or cross attention
            else:
                cache[module] = torch.cat([cache[module], output], dim=3).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiHeadAttentionANE):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks

def decoder_logits(decoder: TextDecoder, x: Tensor, chunked: bool = True):
    weight = decoder.token_embedding.weight.to(x.dtype)
    if not chunked:
        return (x @ torch.transpose(weight, 0, 1)).float()
    if weight.shape[0] >= 51865:
        splits = weight.split(weight.shape[0]//11, dim=0)
        return torch.cat([torch.einsum('bid,jd->bij', x, split) for split in splits], dim=2).float()
    if weight.shape[0] == 51864:
        splits = weight.split(weight.shape[0]//12, dim=0)
        return torch.cat([torch.einsum('bid,jd->bij', x, split) for split in splits], dim=2).float()
    return (x @ torch.transpose(weight, 0, 1)).float()

class StatefulTextDecoder(nn.Module):
    def __init__(self, decoder: TextDecoder, hparams: ModelDimensions, max_tokens: int):
        super().__init__()

        self.decoder = decoder
        self.max_tokens = max_tokens
        self.n_head = hparams.n_text_head

        for i in range(hparams.n_text_layer):
            self.register_buffer(f"k_cache_{i}", torch.zeros((1, max_tokens, hparams.n_text_state), dtype=torch.float16))
            self.register_buffer(f"v_cache_{i}", torch.zeros((1, max_tokens, hparams.n_text_state), dtype=torch.float16))
            self.register_buffer(f"cross_k_{i}", torch.zeros((1, hparams.n_audio_ctx, hparams.n_text_state), dtype=torch.float16))
            self.register_buffer(f"cross_v_{i}", torch.zeros((1, hparams.n_audio_ctx, hparams.n_text_state), dtype=torch.float16))

    def attention(self, q: Tensor, k_cache: Tensor, v_cache: Tensor, mask: Optional[Tensor], k_prescaled: bool):
        n_batch, n_ctx, n_state = q.shape
        n_head = self.n_head
        n_ctx_kv = k_cache.shape[1]
        n_state_head = n_state // n_head
        scale = float(n_state_head)**-0.25

        q = q.view(n_batch, n_ctx, n_head, n_state_head).permute(0, 2, 1, 3)
        k = k_cache.to(q.dtype).view(n_batch, n_ctx_kv, n_head, n_state_head).permute(0, 2, 1, 3)
        v = v_cache.to(q.dtype).view(n_batch, n_ctx_kv, n_head, n_state_head).permute(0, 2, 1, 3)

        if k_prescaled:
            qk = (q*scale) @ k.transpose(-1, -2)
        else:
            qk = (q*scale) @ (k*scale).transpose(-1, -2)

        if mask is not None:
            qk = qk + mask[:, None, :, :]

        w = F.softmax(qk.float(), dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)

    def forward(self, token_data: Tensor, pos_data: Tensor, step_mask: Tensor, self_mask: Tensor):
        q_len = token_data.shape[-1]
        end_step = step_mask.shape[-1]
        past = end_step - q_len
        pos = pos_data.to(torch.long)

        x = self.decoder.token_embedding(token_data) + self.decoder.positional_embedding.index_select(0, pos).unsqueeze(0)
        x = x.float()

        for i, block in enumerate(self.decoder.blocks):
            xn = block.attn_ln(x)
            k_new = block.attn.key(xn)
            v_new = block.attn.value(xn)

            k_cache = getattr(self, f"k_cache_{i}")
            v_cache = getattr(self, f"v_cache_{i}")
            k_cache[:, past:end_step, :] = k_new.to(torch.float16)
            v_cache[:, past:end_step, :] = v_new.to(torch.float16)

            x = x + block.attn.out(self.attention(block.attn.query(xn), k_cache, v_cache, self_mask, False))

            xn = block.cross_attn_ln(x)
            x = x + block.cross_attn.out(self.attention(
                block.cross_attn.query(xn),
                getattr(self, f"cross_k_{i}"),
                getattr(self, f"cross_v_{i}"),
                None,
                True,
            ))

            x = x + block.mlp(block.mlp_ln(x))

        x = self.decoder.ln(x)
        return (x @ torch.transpose(self.decoder.token_embedding.weight.to(x.dtype), 0, 1)).float()

class StatefulNoWriteShardDecoderANE(nn.Module):
    def __init__(
            self,
            decoder: TextDecoderANE,
            hparams: ModelDimensions,
            max_tokens: int,
            start_layer: int,
            output_logits: bool,
            cross_kv_input: bool = False):
        super().__init__()

        self.decoder = decoder
        self.max_tokens = max_tokens
        self.n_head = hparams.n_text_head
        self.start_layer = start_layer
        self.output_logits = output_logits
        self.cross_kv_input = cross_kv_input

        for i in range(hparams.n_text_layer):
            il = start_layer + i
            self.register_buffer(f"k_cache_{il}", torch.zeros((1, hparams.n_text_state, 1, max_tokens), dtype=torch.float16))
            self.register_buffer(f"v_cache_{il}", torch.zeros((1, hparams.n_text_state, 1, max_tokens), dtype=torch.float16))
            if not getattr(self, "cross_kv_input", False):
                self.register_buffer(f"cross_k_{il}", torch.zeros((1, hparams.n_text_state, 1, hparams.n_audio_ctx), dtype=torch.float16))
                self.register_buffer(f"cross_v_{il}", torch.zeros((1, hparams.n_text_state, 1, hparams.n_audio_ctx), dtype=torch.float16))

    def attention(self, q: Tensor, k_cache: Tensor, v_cache: Tensor, mask: Optional[Tensor]):
        _, dim, _, _ = q.size()
        dim_per_head = dim // self.n_head
        scale = float(dim_per_head)**-0.5

        q = q * scale

        mh_q = q.split(dim_per_head, dim=1)
        mh_k = k_cache.to(q.dtype).transpose(1, 3).split(dim_per_head, dim=3)
        mh_v = v_cache.to(q.dtype).split(dim_per_head, dim=1)

        mh_qk = [
            torch.einsum('bchq,bkhc->bkhq', [qi, ki])
            for qi, ki in zip(mh_q, mh_k)
        ]

        if mask is not None:
            for head_idx in range(self.n_head):
                mh_qk[head_idx] = mh_qk[head_idx] + mask

        attn_weights = [aw.softmax(dim=1) for aw in mh_qk]
        attn = [torch.einsum('bkhq,bchk->bchq', wi, vi) for wi, vi in zip(attn_weights, mh_v)]
        return torch.cat(attn, dim=1)

    def forward_x(self, x_data: Tensor, slot_mask: Tensor, self_mask: Tensor, *cross_inputs: Tensor):
        x = x_data.float()
        slot = slot_mask.to(x.dtype)
        outputs = []
        if getattr(self, "cross_kv_input", False) and len(cross_inputs) != len(self.decoder.blocks)*2:
            raise RuntimeError("wrong cross input count")

        for i, block in enumerate(self.decoder.blocks):
            il = self.start_layer + i
            xn = block.attn_ln(x)
            k_new = block.attn.key(xn)
            v_new = block.attn.value(xn)

            k_cache = getattr(self, f"k_cache_{il}")
            v_cache = getattr(self, f"v_cache_{il}")
            k_eff = k_cache.to(x.dtype)*(1.0 - slot) + k_new*slot
            v_eff = v_cache.to(x.dtype)*(1.0 - slot) + v_new*slot

            x = x + block.attn.out(self.attention(block.attn.query(xn), k_eff, v_eff, self_mask))

            xn = block.cross_attn_ln(x)
            if getattr(self, "cross_kv_input", False):
                cross_k = cross_inputs[2*i]
                cross_v = cross_inputs[2*i + 1]
            else:
                cross_k = getattr(self, f"cross_k_{il}")
                cross_v = getattr(self, f"cross_v_{il}")
            x = x + block.cross_attn.out(self.attention(
                block.cross_attn.query(xn),
                cross_k,
                cross_v,
                None,
            ))

            x = x + block.mlp(block.mlp_ln(x))
            outputs.extend([k_new.to(torch.float16), v_new.to(torch.float16)])

        if self.output_logits:
            x = self.decoder.ln(x)
            x = x.permute(0, 2, 3, 1).squeeze(2)
            primary = decoder_logits(self.decoder, x)
        else:
            primary = x.float()

        return tuple([primary] + outputs)

    def forward(self, x_data: Tensor, slot_mask: Tensor, self_mask: Tensor, *cross_inputs: Tensor):
        return self.forward_x(x_data, slot_mask, self_mask, *cross_inputs)

class StatefulNoWriteTokenShardDecoderANE(StatefulNoWriteShardDecoderANE):
    def forward(self, token_data: Tensor, pos_data: Tensor, slot_mask: Tensor, self_mask: Tensor, *cross_inputs: Tensor):
        pos = pos_data.to(torch.long)
        x = self.decoder.token_embedding(token_data) + self.decoder.positional_embedding.index_select(0, pos).unsqueeze(0)
        x = x.float().transpose(1, 2).unsqueeze(2)
        return self.forward_x(x, slot_mask, self_mask, *cross_inputs)

class StatefulSelfKVTextDecoder(nn.Module):
    def __init__(self, decoder: TextDecoder, hparams: ModelDimensions, max_tokens: int):
        super().__init__()

        self.decoder = decoder
        self.max_tokens = max_tokens
        self.n_head = hparams.n_text_head

        for i in range(hparams.n_text_layer):
            self.register_buffer(f"k_cache_{i}", torch.zeros((1, max_tokens, hparams.n_text_state), dtype=torch.float16))
            self.register_buffer(f"v_cache_{i}", torch.zeros((1, max_tokens, hparams.n_text_state), dtype=torch.float16))

    def attention(self, q: Tensor, k_cache: Tensor, v_cache: Tensor, mask: Optional[Tensor], k_prescaled: bool):
        n_batch, n_ctx, n_state = q.shape
        n_head = self.n_head
        n_ctx_kv = k_cache.shape[1]
        n_state_head = n_state // n_head
        scale = float(n_state_head)**-0.25

        q = q.view(n_batch, n_ctx, n_head, n_state_head).permute(0, 2, 1, 3)
        k = k_cache.to(q.dtype).view(n_batch, n_ctx_kv, n_head, n_state_head).permute(0, 2, 1, 3)
        v = v_cache.to(q.dtype).view(n_batch, n_ctx_kv, n_head, n_state_head).permute(0, 2, 1, 3)

        if k_prescaled:
            qk = (q*scale) @ k.transpose(-1, -2)
        else:
            qk = (q*scale) @ (k*scale).transpose(-1, -2)

        if mask is not None:
            qk = qk + mask[:, None, :, :]

        w = F.softmax(qk.float(), dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)

    def forward(self, token_data: Tensor, audio_data: Tensor, pos_data: Tensor, step_mask: Tensor, self_mask: Tensor):
        q_len = token_data.shape[-1]
        end_step = step_mask.shape[-1]
        past = end_step - q_len
        pos = pos_data.to(torch.long)

        x = self.decoder.token_embedding(token_data) + self.decoder.positional_embedding.index_select(0, pos).unsqueeze(0)
        x = x.float()
        audio_data = audio_data.float()

        for i, block in enumerate(self.decoder.blocks):
            xn = block.attn_ln(x)
            k_new = block.attn.key(xn)
            v_new = block.attn.value(xn)

            k_cache = getattr(self, f"k_cache_{i}")
            v_cache = getattr(self, f"v_cache_{i}")
            k_cache[:, past:end_step, :] = k_new.to(torch.float16)
            v_cache[:, past:end_step, :] = v_new.to(torch.float16)

            x = x + block.attn.out(self.attention(block.attn.query(xn), k_cache, v_cache, self_mask, False))

            xn = block.cross_attn_ln(x)
            x = x + block.cross_attn.out(self.attention(
                block.cross_attn.query(xn),
                block.cross_attn.key(audio_data),
                block.cross_attn.value(audio_data),
                None,
                False,
            ))

            x = x + block.mlp(block.mlp_ln(x))

        x = self.decoder.ln(x)
        return (x @ torch.transpose(self.decoder.token_embedding.weight.to(x.dtype), 0, 1)).float()

def convert_encoder(hparams, model, quantize=False):
    model.eval()

    input_shape = (1, hparams.n_mels, 3000)
    input_data = torch.randn(input_shape)
    traced_model = torch.jit.trace(model, input_data)

    model = ct.convert(
        traced_model,
        convert_to="mlprogram",
        inputs=[ct.TensorType(name="logmel_data", shape=input_shape)],
        outputs=[ct.TensorType(name="output")],
        compute_units=ct.ComputeUnit.ALL,
    )

    if quantize:
        model = quantize_weights(model, nbits=16)

    return model

def convert_decoder(hparams, model, quantize=False, max_tokens=0, optimize_ane=False):
    model.eval()

    trace_tokens = max(1, min(max_tokens, 4)) if max_tokens > 0 else 1
    tokens_shape = (1, trace_tokens)
    if optimize_ane:
        audio_shape = (1, hparams.n_audio_state, 1, hparams.n_audio_ctx)
    else:
        audio_shape = (1, hparams.n_audio_ctx, hparams.n_audio_state)

    audio_data = torch.randn(audio_shape)
    token_data = torch.randint(hparams.n_vocab, tokens_shape).long()

    traced_model = torch.jit.trace(model, (token_data, audio_data))

    if max_tokens > 0:
        token_shape = (1, ct.RangeDim(lower_bound=1, upper_bound=max_tokens, default=trace_tokens))
    else:
        token_shape = tokens_shape

    model = ct.convert(
        traced_model,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name="token_data", shape=token_shape, dtype=np.int32),
            ct.TensorType(name="audio_data", shape=audio_shape, dtype=np.float32)
        ],
        outputs=[ct.TensorType(name="logits", dtype=np.float32)],
        compute_units=ct.ComputeUnit.ALL,
    )

    if quantize:
        model = quantize_weights(model, nbits=16)

    return model

def convert_stateful_decoder(hparams, model, max_tokens, input_tokens=1):
    if max_tokens <= 0:
        raise ValueError("--decoder-max-tokens must be set for --decoder-stateful")
    if not hasattr(ct, "StateType") or not hasattr(ct.target, "macOS15"):
        raise RuntimeError("--decoder-stateful requires CoreMLTools with macOS15 stateful model support")
    if input_tokens <= 0 or input_tokens > max_tokens:
        raise ValueError("--decoder-stateful input token count must be in [1, decoder-max-tokens]")

    model = StatefulTextDecoder(model, hparams, max_tokens).eval()

    tokens_shape = (1, input_tokens)
    token_data = torch.randint(hparams.n_vocab, tokens_shape).long()
    pos_data = torch.arange(input_tokens, dtype=torch.long)
    step_mask = torch.zeros((1, 1, input_tokens), dtype=torch.float32)
    self_mask = torch.zeros((1, input_tokens, max_tokens), dtype=torch.float32)

    traced_model = torch.jit.trace(model, (token_data, pos_data, step_mask, self_mask))
    step = ct.RangeDim(lower_bound=1, upper_bound=max_tokens, default=min(max_tokens, 4))

    states = []
    for i in range(hparams.n_text_layer):
        states.extend([
            ct.StateType(wrapped_type=ct.TensorType(shape=(1, max_tokens, hparams.n_text_state), dtype=np.float16), name=f"k_cache_{i}"),
            ct.StateType(wrapped_type=ct.TensorType(shape=(1, max_tokens, hparams.n_text_state), dtype=np.float16), name=f"v_cache_{i}"),
            ct.StateType(wrapped_type=ct.TensorType(shape=(1, hparams.n_audio_ctx, hparams.n_text_state), dtype=np.float16), name=f"cross_k_{i}"),
            ct.StateType(wrapped_type=ct.TensorType(shape=(1, hparams.n_audio_ctx, hparams.n_text_state), dtype=np.float16), name=f"cross_v_{i}"),
        ])

    return ct.convert(
        traced_model,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name="token_data", shape=tokens_shape, dtype=np.int32),
            ct.TensorType(name="pos_data", shape=(input_tokens,), dtype=np.int32),
            ct.TensorType(name="step_mask", shape=(1, 1, step), dtype=np.float32),
            ct.TensorType(name="self_mask", shape=(1, input_tokens, max_tokens), dtype=np.float32),
        ],
        outputs=[ct.TensorType(name="logits", dtype=np.float32)],
        states=states,
        minimum_deployment_target=ct.target.macOS15,
        compute_units=ct.ComputeUnit.ALL,
    )

def convert_stateful_no_write_shard_decoder(hparams, model, max_tokens, start_layer, n_layers, output_logits, token_input=False, cross_kv_input=False):
    if max_tokens <= 0:
        raise ValueError("--decoder-max-tokens must be set for --decoder-stateful-no-write-shard")
    if not hasattr(ct, "StateType") or not hasattr(ct.target, "macOS15"):
        raise RuntimeError("--decoder-stateful-no-write-shard requires CoreMLTools with macOS15 stateful model support")
    if start_layer < 0 or n_layers <= 0 or start_layer + n_layers > hparams.n_text_layer:
        raise ValueError("--decoder-shard-start and --decoder-shard-layers must select a valid decoder layer range")
    if token_input and start_layer != 0:
        raise ValueError("--decoder-shard-token-input is only valid for the first decoder shard")

    decoder = copy.deepcopy(model)
    decoder.blocks = nn.ModuleList(list(decoder.blocks)[start_layer:start_layer + n_layers])
    shard_hparams = replace(hparams, n_text_layer=n_layers)
    shard_cls = StatefulNoWriteTokenShardDecoderANE if token_input else StatefulNoWriteShardDecoderANE
    shard = shard_cls(
        decoder,
        shard_hparams,
        max_tokens=max_tokens,
        start_layer=start_layer,
        output_logits=output_logits,
        cross_kv_input=cross_kv_input,
    ).eval()

    x_shape = (1, hparams.n_text_state, 1, 1)
    slot_mask_shape = (1, 1, 1, max_tokens)
    self_mask_shape = (1, max_tokens, 1, 1)
    slot_mask = torch.zeros(slot_mask_shape, dtype=torch.float32)
    self_mask = torch.zeros(self_mask_shape, dtype=torch.float32)
    cross_shape = (1, hparams.n_text_state, 1, hparams.n_audio_ctx)
    cross_args = tuple(torch.zeros(cross_shape, dtype=torch.float16) for _ in range(n_layers*2)) if cross_kv_input else tuple()
    if token_input:
        token_shape = (1, 1)
        token_data = torch.randint(hparams.n_vocab, token_shape).long()
        pos_data = torch.zeros((1,), dtype=torch.long)
        traced_model = torch.jit.trace(shard, (token_data, pos_data, slot_mask, self_mask, *cross_args))
    else:
        x_data = torch.randn(x_shape, dtype=torch.float32)
        traced_model = torch.jit.trace(shard, (x_data, slot_mask, self_mask, *cross_args))

    states = []
    for i in range(n_layers):
        il = start_layer + i
        states.extend([
            ct.StateType(wrapped_type=ct.TensorType(shape=(1, hparams.n_text_state, 1, max_tokens), dtype=np.float16), name=f"k_cache_{il}"),
            ct.StateType(wrapped_type=ct.TensorType(shape=(1, hparams.n_text_state, 1, max_tokens), dtype=np.float16), name=f"v_cache_{il}"),
        ])
        if not cross_kv_input:
            states.extend([
                ct.StateType(wrapped_type=ct.TensorType(shape=(1, hparams.n_text_state, 1, hparams.n_audio_ctx), dtype=np.float16), name=f"cross_k_{il}"),
                ct.StateType(wrapped_type=ct.TensorType(shape=(1, hparams.n_text_state, 1, hparams.n_audio_ctx), dtype=np.float16), name=f"cross_v_{il}"),
            ])

    outputs = [ct.TensorType(name="logits" if output_logits else "x_out", dtype=np.float32)]
    for i in range(n_layers):
        il = start_layer + i
        outputs.extend([
            ct.TensorType(name=f"k_out_{il}", dtype=np.float16),
            ct.TensorType(name=f"v_out_{il}", dtype=np.float16),
        ])

    inputs = ([
            ct.TensorType(name="token_data", shape=token_shape, dtype=np.int32),
            ct.TensorType(name="pos_data", shape=(1,), dtype=np.int32),
            ct.TensorType(name="slot_mask", shape=slot_mask_shape, dtype=np.float32),
            ct.TensorType(name="self_mask", shape=self_mask_shape, dtype=np.float32),
        ] if token_input else [
            ct.TensorType(name="x_data", shape=x_shape, dtype=np.float32),
            ct.TensorType(name="slot_mask", shape=slot_mask_shape, dtype=np.float32),
            ct.TensorType(name="self_mask", shape=self_mask_shape, dtype=np.float32),
        ])
    if cross_kv_input:
        for i in range(n_layers):
            il = start_layer + i
            inputs.extend([
                ct.TensorType(name=f"cross_k_{il}", shape=cross_shape, dtype=np.float16),
                ct.TensorType(name=f"cross_v_{il}", shape=cross_shape, dtype=np.float16),
            ])

    return ct.convert(
        traced_model,
        convert_to="mlprogram",
        inputs=inputs,
        outputs=outputs,
        states=states,
        minimum_deployment_target=ct.target.macOS15,
        compute_units=ct.ComputeUnit.ALL,
    )

def convert_stateful_multifunction_decoder(hparams, model, max_tokens, prefill_tokens, output_path):
    if not hasattr(ct.utils, "MultiFunctionDescriptor") or not hasattr(ct.utils, "save_multifunction"):
        raise RuntimeError("multifunction decoder export requires CoreMLTools multifunction support")
    if prefill_tokens <= 1:
        raise ValueError("--decoder-stateful-prefill-tokens must be greater than 1")

    decode_path = f"{output_path}.decode.tmp.mlpackage"
    prefill_path = f"{output_path}.prefill.tmp.mlpackage"

    try:
        decode = convert_stateful_decoder(hparams, model, max_tokens=max_tokens, input_tokens=1)
        decode.save(decode_path)

        prefill = convert_stateful_decoder(hparams, model, max_tokens=max_tokens, input_tokens=prefill_tokens)
        prefill.save(prefill_path)

        desc = ct.utils.MultiFunctionDescriptor()
        desc.add_function(decode_path, src_function_name="main", target_function_name="decode")
        desc.add_function(prefill_path, src_function_name="main", target_function_name="prefill")
        desc.default_function_name = "decode"
        ct.utils.save_multifunction(desc, output_path)
    finally:
        shutil.rmtree(decode_path, ignore_errors=True)
        shutil.rmtree(prefill_path, ignore_errors=True)

def convert_stateful_self_kv_decoder(hparams, model, max_tokens):
    if max_tokens <= 0:
        raise ValueError("--decoder-max-tokens must be set for --decoder-stateful-self-kv")
    if not hasattr(ct, "StateType") or not hasattr(ct.target, "macOS15"):
        raise RuntimeError("--decoder-stateful-self-kv requires CoreMLTools with macOS15 stateful model support")

    model = StatefulSelfKVTextDecoder(model, hparams, max_tokens).eval()

    tokens_shape = (1, 1)
    audio_shape = (1, hparams.n_audio_ctx, hparams.n_audio_state)
    token_data = torch.randint(hparams.n_vocab, tokens_shape).long()
    audio_data = torch.randn(audio_shape)
    pos_data = torch.zeros((1,), dtype=torch.long)
    step_mask = torch.zeros((1, 1, min(max_tokens, 4)), dtype=torch.float32)
    self_mask = torch.zeros((1, 1, max_tokens), dtype=torch.float32)

    traced_model = torch.jit.trace(model, (token_data, audio_data, pos_data, step_mask, self_mask))
    step = ct.RangeDim(lower_bound=1, upper_bound=max_tokens, default=min(max_tokens, 4))

    states = []
    for i in range(hparams.n_text_layer):
        states.extend([
            ct.StateType(wrapped_type=ct.TensorType(shape=(1, max_tokens, hparams.n_text_state), dtype=np.float16), name=f"k_cache_{i}"),
            ct.StateType(wrapped_type=ct.TensorType(shape=(1, max_tokens, hparams.n_text_state), dtype=np.float16), name=f"v_cache_{i}"),
        ])

    return ct.convert(
        traced_model,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name="token_data", shape=tokens_shape, dtype=np.int32),
            ct.TensorType(name="audio_data", shape=audio_shape, dtype=np.float32),
            ct.TensorType(name="pos_data", shape=(1,), dtype=np.int32),
            ct.TensorType(name="step_mask", shape=(1, 1, step), dtype=np.float32),
            ct.TensorType(name="self_mask", shape=(1, 1, max_tokens), dtype=np.float32),
        ],
        outputs=[ct.TensorType(name="logits", dtype=np.float32)],
        states=states,
        minimum_deployment_target=ct.target.macOS15,
        compute_units=ct.ComputeUnit.ALL,
    )


if __name__ == "__main__":
    def str2bool(value):
        if isinstance(value, bool):
            return value
        value = value.lower()
        if value in ("yes", "true", "t", "1"):
            return True
        if value in ("no", "false", "f", "0"):
            return False
        raise argparse.ArgumentTypeError("expected boolean value")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model to convert (e.g. tiny, tiny.en, base, base.en, small, small.en, medium, medium.en, large-v1, large-v2, large-v3, large-v3-turbo)", required=True)
    parser.add_argument("--encoder-only", type=str2bool, help="only convert encoder", default=False)
    parser.add_argument("--decoder-only", action="store_true", help="only convert decoder")
    parser.add_argument("--decoder-stateful", action="store_true", help="export experimental one-token decoder with Core ML self/cross KV state")
    parser.add_argument("--decoder-stateful-self-kv", action="store_true", help="export experimental one-token decoder with Core ML self KV state and audio input")
    parser.add_argument("--decoder-stateful-no-write-shard", action="store_true", help="export experimental ANE decoder shard that reads KV state and returns new KV tensors for host-side state writes")
    parser.add_argument("--decoder-shard-start", type=int, help="first decoder layer for --decoder-stateful-no-write-shard", default=0)
    parser.add_argument("--decoder-shard-layers", type=int, help="number of decoder layers for --decoder-stateful-no-write-shard", default=8)
    parser.add_argument("--decoder-shard-logits", action="store_true", help="make --decoder-stateful-no-write-shard return final logits instead of hidden state")
    parser.add_argument("--decoder-shard-token-input", action="store_true", help="make the first no-write shard consume token_data and pos_data instead of x_data")
    parser.add_argument("--decoder-shard-cross-kv-input", action="store_true", help="make no-write shards consume cross KV as inputs instead of MLState")
    parser.add_argument("--decoder-stateful-prefill-tokens", type=int, help="also export a fixed-token prompt prefill function in a multifunction decoder", default=0)
    parser.add_argument("--decoder-max-tokens", type=int, help="maximum token context for experimental decoder export; defaults to the model text context", default=0)
    parser.add_argument("--quantize",     type=str2bool, help="quantize weights to F16", default=False)
    parser.add_argument("--optimize-ane", type=str2bool, help="optimize for ANE execution (currently broken)", default=False)
    args = parser.parse_args()

    if args.model not in ["tiny", "tiny.en", "base", "base.en", "small", "small.en", "small.en-tdrz", "medium", "medium.en", "large-v1", "large-v2", "large-v3", "large-v3-turbo"]:
        raise ValueError("Invalid model name")

    if args.decoder_stateful and args.optimize_ane and not args.decoder_stateful_no_write_shard:
        raise ValueError("--decoder-stateful does not support --optimize-ane")
    if args.decoder_stateful_self_kv and args.optimize_ane:
        raise ValueError("--decoder-stateful-self-kv does not support --optimize-ane")
    if args.decoder_stateful_self_kv and not args.decoder_stateful:
        raise ValueError("--decoder-stateful-self-kv requires --decoder-stateful")
    if args.decoder_stateful_no_write_shard and not args.decoder_stateful:
        raise ValueError("--decoder-stateful-no-write-shard requires --decoder-stateful")
    if args.decoder_stateful_no_write_shard and args.decoder_stateful_self_kv:
        raise ValueError("--decoder-stateful-no-write-shard does not support --decoder-stateful-self-kv")
    if args.decoder_stateful_no_write_shard and not args.optimize_ane:
        raise ValueError("--decoder-stateful-no-write-shard requires --optimize-ane true")
    if args.decoder_shard_token_input and not args.decoder_stateful_no_write_shard:
        raise ValueError("--decoder-shard-token-input requires --decoder-stateful-no-write-shard")
    if args.decoder_shard_cross_kv_input and not args.decoder_stateful_no_write_shard:
        raise ValueError("--decoder-shard-cross-kv-input requires --decoder-stateful-no-write-shard")
    if args.decoder_shard_token_input and args.decoder_shard_start != 0:
        raise ValueError("--decoder-shard-token-input is only valid for --decoder-shard-start 0")
    if args.decoder_stateful_prefill_tokens > 0 and (not args.decoder_stateful or args.decoder_stateful_self_kv):
        raise ValueError("--decoder-stateful-prefill-tokens requires --decoder-stateful without --decoder-stateful-self-kv")
    if args.decoder_stateful_prefill_tokens > 0 and args.decoder_stateful_no_write_shard:
        raise ValueError("--decoder-stateful-prefill-tokens does not support --decoder-stateful-no-write-shard")

    whisper = load_model(args.model).cpu()
    hparams = whisper.dims
    print(hparams)

    if args.optimize_ane:
        whisperANE = WhisperANE(hparams).eval()
        whisperANE.load_state_dict(whisper.state_dict())

        encoder = whisperANE.encoder
        decoder = whisperANE.decoder
    else:
        encoder = whisper.encoder
        decoder = whisper.decoder

    if args.encoder_only and args.decoder_only:
        raise ValueError("--encoder-only and --decoder-only are mutually exclusive")

    if not args.decoder_only:
        # Convert encoder
        encoder = convert_encoder(hparams, encoder, quantize=args.quantize)
        encoder.save(f"models/coreml-encoder-{args.model}.mlpackage")

    if args.encoder_only is False:
        # Convert decoder
        decoder_max_tokens = args.decoder_max_tokens if args.decoder_max_tokens > 0 else hparams.n_text_ctx
        if args.decoder_stateful_prefill_tokens > 0:
            decoder_source = decoder
            decoder = None
            convert_stateful_multifunction_decoder(
                hparams,
                decoder_source,
                max_tokens=decoder_max_tokens,
                prefill_tokens=args.decoder_stateful_prefill_tokens,
                output_path=f"models/coreml-decoder-{args.model}-prefill{args.decoder_stateful_prefill_tokens}.mlpackage",
            )
        elif args.decoder_stateful_no_write_shard:
            decoder = convert_stateful_no_write_shard_decoder(
                hparams,
                decoder,
                max_tokens=decoder_max_tokens,
                start_layer=args.decoder_shard_start,
                n_layers=args.decoder_shard_layers,
                output_logits=args.decoder_shard_logits,
                token_input=args.decoder_shard_token_input,
                cross_kv_input=args.decoder_shard_cross_kv_input,
            )
        elif args.decoder_stateful_self_kv:
            decoder = convert_stateful_self_kv_decoder(hparams, decoder, max_tokens=decoder_max_tokens)
        elif args.decoder_stateful:
            decoder = convert_stateful_decoder(hparams, decoder, max_tokens=decoder_max_tokens)
        else:
            decoder = convert_decoder(hparams, decoder, quantize=args.quantize, max_tokens=decoder_max_tokens, optimize_ane=args.optimize_ane)
        if decoder is not None:
            suffix = "-self-kv" if args.decoder_stateful_self_kv else ""
            if args.decoder_stateful_no_write_shard:
                suffix = ("-cross-input" if args.decoder_shard_cross_kv_input else "") + f"-no-write-s{args.decoder_shard_start}-l{args.decoder_shard_layers}"
                if args.decoder_shard_token_input:
                    suffix += "-token"
                if args.decoder_shard_logits:
                    suffix += "-logits"
            decoder.save(f"models/coreml-decoder-{args.model}{suffix}.mlpackage")

    print("done converting")
