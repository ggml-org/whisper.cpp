import argparse
import copy
import numpy as np
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

COREML_DECODER_SHARD_LAYERS = 8

def coreml_decoder_shard_plan(hparams):
    if hparams.n_text_layer == 4:
        return [4]
    if hparams.n_text_layer == 6:
        return [6]
    if hparams.n_text_layer == 12:
        return [12]
    if hparams.n_text_layer == 24:
        return [12, 12]
    if hparams.n_text_layer == 32:
        return [12, 8, 12]

    plan = []
    remaining = hparams.n_text_layer
    while remaining > 0:
        n_layers = min(COREML_DECODER_SHARD_LAYERS, remaining)
        plan.append(n_layers)
        remaining -= n_layers
    return plan

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

def convert_stateful_no_write_shard_decoder(hparams, model, max_tokens, start_layer, n_layers, output_logits, token_input=False, cross_kv_input=False):
    if max_tokens <= 0:
        raise ValueError("Core ML decoder export requires a positive token context")
    if not hasattr(ct, "StateType") or not hasattr(ct.target, "macOS15"):
        raise RuntimeError("--decoder-npu requires CoreMLTools with macOS15 stateful model support")
    if start_layer < 0 or n_layers <= 0 or start_layer + n_layers > hparams.n_text_layer:
        raise ValueError("invalid Core ML decoder shard layer range")
    if token_input and start_layer != 0:
        raise ValueError("token input is only valid for the first decoder shard")

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

def save_coreml_decoder_shards(hparams, decoder, output_prefix):
    start_layer = 0
    for n_layers in coreml_decoder_shard_plan(hparams):
        token_input = start_layer == 0
        output_logits = start_layer + n_layers == hparams.n_text_layer
        shard = convert_stateful_no_write_shard_decoder(
            hparams,
            decoder,
            max_tokens=hparams.n_text_ctx,
            start_layer=start_layer,
            n_layers=n_layers,
            output_logits=output_logits,
            token_input=token_input,
            cross_kv_input=True,
        )

        suffix = f"-cross-input-no-write-s{start_layer}-l{n_layers}"
        if token_input:
            suffix += "-token"
        if output_logits:
            suffix += "-logits"
        shard.save(f"{output_prefix}{suffix}.mlpackage")
        start_layer += n_layers


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
    parser.add_argument("--decoder-npu", action="store_true", help="export Core ML decoder shards for Apple Neural Engine inference")
    parser.add_argument("--quantize",     type=str2bool, help="quantize weights to F16", default=False)
    parser.add_argument("--optimize-ane", type=str2bool, help="optimize encoder and decoder graphs for Apple Neural Engine execution", default=False)
    args = parser.parse_args()

    if args.model not in ["tiny", "tiny.en", "base", "base.en", "small", "small.en", "small.en-tdrz", "medium", "medium.en", "large-v1", "large-v2", "large-v3", "large-v3-turbo"]:
        raise ValueError("Invalid model name")

    if args.decoder_npu and not args.optimize_ane:
        raise ValueError("--decoder-npu requires --optimize-ane true")

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
        if args.decoder_npu:
            save_coreml_decoder_shards(
                hparams,
                decoder,
                output_prefix=f"models/coreml-decoder-{args.model}",
            )
        else:
            decoder = convert_decoder(hparams, decoder, quantize=args.quantize, max_tokens=hparams.n_text_ctx, optimize_ane=args.optimize_ane)
            decoder.save(f"models/coreml-decoder-{args.model}.mlpackage")

    print("done converting")
