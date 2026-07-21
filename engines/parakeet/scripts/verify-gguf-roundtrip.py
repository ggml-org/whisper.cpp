#!/usr/bin/env python3
"""Verify every GGUF tensor numerically matches the original NeMo state_dict.

For each GGUF tensor we:

  1. Recreate the expected NumPy array from the NeMo state_dict (applying
     the same layout transforms as scripts/convert-nemo-to-gguf.py:
     squeezing the CTC Conv1d kernel, fusing conv-module BatchNorm into
     scale+shift, or copying the LayerNorm gamma/beta when the encoder uses
     LayerNorm in the conv module).
  2. Read the GGUF tensor back through the gguf Python reader.
  3. Compare values.  For f32 tensors we require max-abs == 0 (bit-exact
     round trip).  For f16 tensors we assert the max-abs diff is within
     half-precision quantization, i.e. <= max(|w|) * 2**-10.

Model type is auto-detected from the GGUF's ``parakeet.model.type``
metadata key (``ctc`` / ``eou``); per-type tensor maps live in
``build_expected_*`` helpers.
"""

import argparse
import io
import sys
import tarfile
from pathlib import Path

import numpy as np
import torch
import gguf


def fuse_bn(weight, bias, running_mean, running_var, eps=1e-5):
    scale = weight / np.sqrt(running_var + eps)
    shift = bias - running_mean * scale
    return scale.astype(np.float32), shift.astype(np.float32)


def load_sd(nemo_path: Path):
    with tarfile.open(nemo_path, 'r') as t:
        for cand in ('./model_weights.ckpt', 'model_weights.ckpt'):
            try:
                buf = io.BytesIO(t.extractfile(t.getmember(cand)).read())
                break
            except KeyError:
                continue
        else:
            raise RuntimeError(f"model_weights.ckpt not found in {nemo_path}")
    return torch.load(buf, map_location='cpu', weights_only=True)


def _np32(t):
    return t.detach().cpu().numpy().astype(np.float32, copy=False)


def _build_encoder_expected(sd: dict, n_layers: int, conv_norm_type: str, use_bias: bool):
    out = {
        'preproc.mel_filterbank': _np32(sd['preprocessor.featurizer.fb'][0]),
        'preproc.window':         _np32(sd['preprocessor.featurizer.window']),

        'encoder.subsampling.conv0.weight':    _np32(sd['encoder.pre_encode.conv.0.weight']),
        'encoder.subsampling.conv1_dw.weight': _np32(sd['encoder.pre_encode.conv.2.weight']),
        'encoder.subsampling.conv1_pw.weight': _np32(sd['encoder.pre_encode.conv.3.weight']),
        'encoder.subsampling.conv2_dw.weight': _np32(sd['encoder.pre_encode.conv.5.weight']),
        'encoder.subsampling.conv2_pw.weight': _np32(sd['encoder.pre_encode.conv.6.weight']),
        'encoder.subsampling.out.weight':      _np32(sd['encoder.pre_encode.out.weight']),
    }
    for sub_key, sd_key in (
        ('encoder.subsampling.conv0.bias',    'encoder.pre_encode.conv.0.bias'),
        ('encoder.subsampling.conv1_dw.bias', 'encoder.pre_encode.conv.2.bias'),
        ('encoder.subsampling.conv1_pw.bias', 'encoder.pre_encode.conv.3.bias'),
        ('encoder.subsampling.conv2_dw.bias', 'encoder.pre_encode.conv.5.bias'),
        ('encoder.subsampling.conv2_pw.bias', 'encoder.pre_encode.conv.6.bias'),
        ('encoder.subsampling.out.bias',      'encoder.pre_encode.out.bias'),
    ):
        if sd_key in sd:
            out[sub_key] = _np32(sd[sd_key])

    for i in range(n_layers):
        k = f'encoder.layers.{i}'
        p = f'encoder.blk.{i}'

        out[f'{p}.norm_ff1.weight']  = _np32(sd[f'{k}.norm_feed_forward1.weight'])
        out[f'{p}.norm_ff1.bias']    = _np32(sd[f'{k}.norm_feed_forward1.bias'])
        out[f'{p}.ff1.linear1.weight'] = _np32(sd[f'{k}.feed_forward1.linear1.weight'])
        out[f'{p}.ff1.linear2.weight'] = _np32(sd[f'{k}.feed_forward1.linear2.weight'])

        out[f'{p}.norm_attn.weight'] = _np32(sd[f'{k}.norm_self_att.weight'])
        out[f'{p}.norm_attn.bias']   = _np32(sd[f'{k}.norm_self_att.bias'])
        q_w = _np32(sd[f'{k}.self_attn.linear_q.weight'])
        k_w = _np32(sd[f'{k}.self_attn.linear_k.weight'])
        v_w = _np32(sd[f'{k}.self_attn.linear_v.weight'])
        out[f'{p}.attn.q.weight']    = q_w
        out[f'{p}.attn.k.weight']    = k_w
        out[f'{p}.attn.v.weight']    = v_w
        out[f'{p}.attn.qkv.weight']  = np.concatenate([q_w, k_w, v_w], axis=0)
        out[f'{p}.attn.out.weight']  = _np32(sd[f'{k}.self_attn.linear_out.weight'])
        out[f'{p}.attn.pos.weight']  = _np32(sd[f'{k}.self_attn.linear_pos.weight'])
        out[f'{p}.attn.pos_bias_u']  = _np32(sd[f'{k}.self_attn.pos_bias_u'])
        out[f'{p}.attn.pos_bias_v']  = _np32(sd[f'{k}.self_attn.pos_bias_v'])

        if use_bias:
            q_b = _np32(sd[f'{k}.self_attn.linear_q.bias'])
            k_b = _np32(sd[f'{k}.self_attn.linear_k.bias'])
            v_b = _np32(sd[f'{k}.self_attn.linear_v.bias'])
            out[f'{p}.attn.q.bias']      = q_b
            out[f'{p}.attn.k.bias']      = k_b
            out[f'{p}.attn.v.bias']      = v_b
            out[f'{p}.attn.qkv.bias']    = np.concatenate([q_b, k_b, v_b], axis=0)
            out[f'{p}.attn.out.bias']    = _np32(sd[f'{k}.self_attn.linear_out.bias'])
            out[f'{p}.ff1.linear1.bias'] = _np32(sd[f'{k}.feed_forward1.linear1.bias'])
            out[f'{p}.ff1.linear2.bias'] = _np32(sd[f'{k}.feed_forward1.linear2.bias'])

        out[f'{p}.norm_conv.weight'] = _np32(sd[f'{k}.norm_conv.weight'])
        out[f'{p}.norm_conv.bias']   = _np32(sd[f'{k}.norm_conv.bias'])
        out[f'{p}.conv.pw1.weight']  = np.squeeze(_np32(sd[f'{k}.conv.pointwise_conv1.weight']), axis=-1)
        out[f'{p}.conv.dw.weight']   = _np32(sd[f'{k}.conv.depthwise_conv.weight'])
        if use_bias:
            out[f'{p}.conv.pw1.bias'] = _np32(sd[f'{k}.conv.pointwise_conv1.bias'])
            out[f'{p}.conv.dw.bias']  = _np32(sd[f'{k}.conv.depthwise_conv.bias'])

        if conv_norm_type == 'layer_norm':
            out[f'{p}.conv.norm.weight'] = _np32(sd[f'{k}.conv.batch_norm.weight'])
            out[f'{p}.conv.norm.bias']   = _np32(sd[f'{k}.conv.batch_norm.bias'])
        else:
            bn_scale, bn_shift = fuse_bn(
                _np32(sd[f'{k}.conv.batch_norm.weight']),
                _np32(sd[f'{k}.conv.batch_norm.bias']),
                _np32(sd[f'{k}.conv.batch_norm.running_mean']),
                _np32(sd[f'{k}.conv.batch_norm.running_var']),
                eps=1e-5,
            )
            out[f'{p}.conv.bn.scale'] = bn_scale
            out[f'{p}.conv.bn.shift'] = bn_shift

        out[f'{p}.conv.pw2.weight'] = np.squeeze(_np32(sd[f'{k}.conv.pointwise_conv2.weight']), axis=-1)
        if use_bias:
            out[f'{p}.conv.pw2.bias'] = _np32(sd[f'{k}.conv.pointwise_conv2.bias'])

        out[f'{p}.norm_ff2.weight']  = _np32(sd[f'{k}.norm_feed_forward2.weight'])
        out[f'{p}.norm_ff2.bias']    = _np32(sd[f'{k}.norm_feed_forward2.bias'])
        out[f'{p}.ff2.linear1.weight'] = _np32(sd[f'{k}.feed_forward2.linear1.weight'])
        out[f'{p}.ff2.linear2.weight'] = _np32(sd[f'{k}.feed_forward2.linear2.weight'])
        if use_bias:
            out[f'{p}.ff2.linear1.bias'] = _np32(sd[f'{k}.feed_forward2.linear1.bias'])
            out[f'{p}.ff2.linear2.bias'] = _np32(sd[f'{k}.feed_forward2.linear2.bias'])

        out[f'{p}.norm_out.weight'] = _np32(sd[f'{k}.norm_out.weight'])
        out[f'{p}.norm_out.bias']   = _np32(sd[f'{k}.norm_out.bias'])

    return out


def build_expected_ctc(sd: dict, n_layers: int):
    out = _build_encoder_expected(sd, n_layers,
                                  conv_norm_type='batch_norm', use_bias=True)
    out['ctc.decoder.weight'] = _np32(sd['decoder.decoder_layers.0.weight'].squeeze(-1))
    out['ctc.decoder.bias']   = _np32(sd['decoder.decoder_layers.0.bias'])
    return out


def build_expected_sortformer(sd: dict, n_layers: int, tf_n_layers: int):
    out = _build_encoder_expected(sd, n_layers,
                                  conv_norm_type='batch_norm', use_bias=True)
    out['sortformer.encoder_proj.weight'] = _np32(sd['sortformer_modules.encoder_proj.weight'])
    out['sortformer.encoder_proj.bias']   = _np32(sd['sortformer_modules.encoder_proj.bias'])

    for i in range(tf_n_layers):
        k = f'transformer_encoder.layers.{i}'
        p = f'sortformer.transformer.blk.{i}'

        out[f'{p}.attn.q.weight']   = _np32(sd[f'{k}.first_sub_layer.query_net.weight'])
        out[f'{p}.attn.q.bias']     = _np32(sd[f'{k}.first_sub_layer.query_net.bias'])
        out[f'{p}.attn.k.weight']   = _np32(sd[f'{k}.first_sub_layer.key_net.weight'])
        out[f'{p}.attn.k.bias']     = _np32(sd[f'{k}.first_sub_layer.key_net.bias'])
        out[f'{p}.attn.v.weight']   = _np32(sd[f'{k}.first_sub_layer.value_net.weight'])
        out[f'{p}.attn.v.bias']     = _np32(sd[f'{k}.first_sub_layer.value_net.bias'])
        out[f'{p}.attn.out.weight'] = _np32(sd[f'{k}.first_sub_layer.out_projection.weight'])
        out[f'{p}.attn.out.bias']   = _np32(sd[f'{k}.first_sub_layer.out_projection.bias'])

        out[f'{p}.ln1.weight']      = _np32(sd[f'{k}.layer_norm_1.weight'])
        out[f'{p}.ln1.bias']        = _np32(sd[f'{k}.layer_norm_1.bias'])

        out[f'{p}.ffn.in.weight']   = _np32(sd[f'{k}.second_sub_layer.dense_in.weight'])
        out[f'{p}.ffn.in.bias']     = _np32(sd[f'{k}.second_sub_layer.dense_in.bias'])
        out[f'{p}.ffn.out.weight']  = _np32(sd[f'{k}.second_sub_layer.dense_out.weight'])
        out[f'{p}.ffn.out.bias']    = _np32(sd[f'{k}.second_sub_layer.dense_out.bias'])

        out[f'{p}.ln2.weight']      = _np32(sd[f'{k}.layer_norm_2.weight'])
        out[f'{p}.ln2.bias']        = _np32(sd[f'{k}.layer_norm_2.bias'])

    out['sortformer.head.first_hidden_to_hidden.weight'] = _np32(sd['sortformer_modules.first_hidden_to_hidden.weight'])
    out['sortformer.head.first_hidden_to_hidden.bias']   = _np32(sd['sortformer_modules.first_hidden_to_hidden.bias'])
    out['sortformer.head.single_hidden_to_spks.weight']  = _np32(sd['sortformer_modules.single_hidden_to_spks.weight'])
    out['sortformer.head.single_hidden_to_spks.bias']    = _np32(sd['sortformer_modules.single_hidden_to_spks.bias'])
    return out


def build_expected_eou(sd: dict, n_layers: int):
    out = _build_encoder_expected(sd, n_layers,
                                  conv_norm_type='layer_norm', use_bias=False)
    out['eou.predict.embed.weight']    = _np32(sd['decoder.prediction.embed.weight'])
    out['eou.predict.lstm.0.w_ih']     = _np32(sd['decoder.prediction.dec_rnn.lstm.weight_ih_l0'])
    out['eou.predict.lstm.0.w_hh']     = _np32(sd['decoder.prediction.dec_rnn.lstm.weight_hh_l0'])
    out['eou.predict.lstm.0.b_ih']     = _np32(sd['decoder.prediction.dec_rnn.lstm.bias_ih_l0'])
    out['eou.predict.lstm.0.b_hh']     = _np32(sd['decoder.prediction.dec_rnn.lstm.bias_hh_l0'])
    out['eou.joint.enc.weight']        = _np32(sd['joint.enc.weight'])
    out['eou.joint.enc.bias']          = _np32(sd['joint.enc.bias'])
    out['eou.joint.pred.weight']       = _np32(sd['joint.pred.weight'])
    out['eou.joint.pred.bias']         = _np32(sd['joint.pred.bias'])
    out['eou.joint.out.weight']        = _np32(sd['joint.joint_net.2.weight'])
    out['eou.joint.out.bias']          = _np32(sd['joint.joint_net.2.bias'])
    return out


def ggml_to_numpy(t):
    data = np.asarray(t.data)
    if t.tensor_type in (gguf.GGMLQuantizationType.F32,):
        return data.view(np.float32).reshape(tuple(reversed(t.shape))).astype(np.float32)
    if t.tensor_type in (gguf.GGMLQuantizationType.F16,):
        return data.view(np.float16).reshape(tuple(reversed(t.shape))).astype(np.float32)
    if t.tensor_type in (gguf.GGMLQuantizationType.Q8_0,
                         gguf.GGMLQuantizationType.Q5_0,
                         gguf.GGMLQuantizationType.Q4_0):
        deq = gguf.quants.dequantize(data, t.tensor_type)
        return np.asarray(deq, dtype=np.float32).reshape(tuple(reversed(t.shape)))
    raise RuntimeError(f"unexpected tensor type {t.tensor_type}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", type=Path, default=Path("models/parakeet-ctc-0.6b.gguf"))
    ap.add_argument("--nemo", type=Path, default=Path("models/parakeet-ctc-0.6b.nemo"))
    args = ap.parse_args()

    print(f"[verify] reading {args.gguf}", file=sys.stderr)
    reader = gguf.GGUFReader(str(args.gguf))
    gg = {t.name: t for t in reader.tensors}
    print(f"[verify] loaded {len(gg)} tensors from GGUF", file=sys.stderr)

    arch = None
    n_layers = 0
    tf_n_layers = 0
    model_type = "ctc"
    for field in reader.fields.values():
        if field.name == "general.architecture":
            arch = bytes(field.parts[field.data[0]]).decode()
        elif field.name == "parakeet.sortformer.tf_n_layers":
            tf_n_layers = int(field.parts[field.data[0]][0])
        elif field.name == "parakeet.encoder.n_layers":
            n_layers = int(field.parts[field.data[0]][0])
        elif field.name == "parakeet.model.type":
            model_type = bytes(field.parts[field.data[0]]).decode()
    print(f"[verify] arch={arch} model_type={model_type} n_layers={n_layers}", file=sys.stderr)

    print(f"[verify] loading {args.nemo}", file=sys.stderr)
    sd = load_sd(args.nemo)
    print(f"[verify] building expected tensor map ({model_type})", file=sys.stderr)
    if model_type == "eou":
        expected = build_expected_eou(sd, n_layers)
    elif model_type == "ctc":
        expected = build_expected_ctc(sd, n_layers)
    elif model_type == "sortformer":
        if tf_n_layers <= 0:
            print(f"[verify] sortformer GGUF missing parakeet.sortformer.tf_n_layers metadata",
                  file=sys.stderr)
            return 3
        expected = build_expected_sortformer(sd, n_layers, tf_n_layers)
    else:
        print(f"[verify] no expected-tensor map implemented for type {model_type!r}",
              file=sys.stderr)
        return 3

    missing = set(expected.keys()) - set(gg.keys())
    extra   = set(gg.keys()) - set(expected.keys())
    if missing:
        print(f"[verify] MISSING in GGUF ({len(missing)}):")
        for k in sorted(missing)[:10]:
            print(f"  - {k}")
        return 1
    if extra:
        print(f"[verify] UNEXPECTED extras in GGUF (ignoring): {sorted(extra)[:5]}")

    rel_gate = {
        gguf.GGMLQuantizationType.F16:  2 ** -10,
        gguf.GGMLQuantizationType.Q8_0: 2 ** -7,
        gguf.GGMLQuantizationType.Q5_0: 2 ** -4,
        gguf.GGMLQuantizationType.Q4_0: 2 ** -3,
    }

    f32_fail = 0
    f32_count = 0
    quant_count = 0
    quant_fail = 0
    worst_rel = 0.0
    worst_name = ""
    for name, exp in expected.items():
        t = gg[name]
        got = ggml_to_numpy(t)
        if got.shape != exp.shape:
            squeezed_got = np.squeeze(got)
            squeezed_exp = np.squeeze(exp)
            if squeezed_got.shape == squeezed_exp.shape:
                got = squeezed_got
                exp_cmp = squeezed_exp
            else:
                print(f"[verify] SHAPE {name}: got {got.shape} expected {exp.shape}")
                f32_fail += 1
                continue
        else:
            exp_cmp = exp
        diff = got.astype(np.float32) - exp_cmp.astype(np.float32)
        max_abs = float(np.abs(diff).max()) if diff.size else 0.0
        denom = float(np.abs(exp_cmp).max()) if exp_cmp.size else 0.0
        rel = max_abs / denom if denom > 0 else max_abs
        if t.tensor_type == gguf.GGMLQuantizationType.F32:
            f32_count += 1
            if max_abs != 0.0:
                f32_fail += 1
                print(f"[verify] F32 NOT bit-exact: {name}  max_abs={max_abs:.3e}")
        else:
            quant_count += 1
            gate = rel_gate.get(t.tensor_type)
            if gate is not None and rel > gate:
                quant_fail += 1
                print(f"[verify] {t.tensor_type.name} OUT OF GATE: {name}  rel={rel:.3e} > {gate:.3e}")
            if rel > worst_rel:
                worst_rel = rel
                worst_name = name

    print(f"[verify] f32 tensors: {f32_count} bit-exact: {f32_count - f32_fail}/{f32_count}",
          file=sys.stderr)
    print(f"[verify] quant tensors: {quant_count} within gate: {quant_count - quant_fail}/{quant_count}  "
          f"worst rel: {worst_rel:.3e}  (worst: {worst_name})", file=sys.stderr)
    ok = (f32_fail == 0) and (quant_fail == 0)
    print(f"[verify] {'PASS' if ok else 'FAIL'}", file=sys.stderr)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
