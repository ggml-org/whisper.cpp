#!/usr/bin/env python3
# Generate a tiny Parakeet TDT ggml model for testing purposes.
#
# The model has valid structure but random weights and a minimal vocabulary.
# It can be loaded by parakeet.cpp but produces meaningless output.
#
# Usage: python generate-parakeet-test-model.py [output_path]
#
import struct
import sys
import numpy as np
from pathlib import Path

def write_tensor(fout, name, data):
    """Write a tensor in the ggml format expected by parakeet.cpp."""
    n_dims = len(data.shape)
    # Always write as F32 (ftype=0); I32 and F32 have the same byte width (4),
    # so the C++ size check passes and zero-valued int tensors are unaffected.
    data = data.astype(np.float32)
    ftype = 0  # GGML_TYPE_F32

    name_bytes = name.encode('utf-8')
    fout.write(struct.pack("iii", n_dims, len(name_bytes), ftype))
    for i in range(n_dims):
        fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
    fout.write(name_bytes)
    data.tofile(fout)

def generate(output_path):
    rng = np.random.default_rng(42)

    # Tiny but valid hparams. Constraints:
    #   n_mels must be divisible by subsampling_factor
    #   n_audio_state must be divisible by n_audio_head
    hparams = {
        'n_vocab':                10,
        'n_audio_ctx':            16,
        'n_audio_state':          8,
        'n_audio_head':           2,
        'n_audio_layer':          1,
        'n_mels':                 16,
        'ftype':                  0,   # F32
        'n_fft':                  64,
        'subsampling_factor':     8,
        'n_subsampling_channels': 4,
        'n_conv_kernel':          3,
        'n_pred_dim':             8,
        'n_pred_layers':          1,
        'n_tdt_durations':        2,
        'n_max_tokens':           5,
    }

    n_vocab   = hparams['n_vocab']
    n_state   = hparams['n_audio_state']
    n_head    = hparams['n_audio_head']
    n_layer   = hparams['n_audio_layer']
    n_mels    = hparams['n_mels']
    n_fft     = hparams['n_fft']
    n_sub_fac  = hparams['subsampling_factor']
    n_sub_ch   = hparams['n_subsampling_channels']
    n_conv_ker = hparams['n_conv_kernel']
    dec_dim    = hparams['n_pred_dim']
    n_pred_l  = hparams['n_pred_layers']
    n_tdt     = hparams['n_tdt_durations']

    n_pre_enc = (n_mels // n_sub_fac) * n_sub_ch   # flattened pre-encode output
    n_head_dim = n_state // n_head                  # per-head dimension
    n_pred_embed = n_vocab + 1                       # vocab + blank
    n_lstm_gates = 4 * dec_dim                       # 4 LSTM gates
    n_joint_out  = n_vocab + n_tdt + 1              # vocab + durations + blank

    n_freqs = n_fft // 2 + 1                        # mel filterbank frequency bins

    def f32(*shape):
        return rng.standard_normal(shape).astype(np.float32)

    with open(output_path, 'wb') as fout:
        # Magic
        fout.write(struct.pack("I", 0x67676d6c))

        # Hyperparameters (14 x int32)
        for key in ['n_vocab','n_audio_ctx','n_audio_state','n_audio_head','n_audio_layer',
                    'n_mels','ftype','n_fft','subsampling_factor','n_subsampling_channels',
                    'n_conv_kernel','n_pred_dim','n_pred_layers','n_tdt_durations','n_max_tokens']:
            fout.write(struct.pack("i", hparams[key]))

        # Mel filterbank  [n_mels, n_freqs]
        fout.write(struct.pack("i", n_mels))
        fout.write(struct.pack("i", n_freqs))
        f32(n_mels, n_freqs).tofile(fout)

        # Window function  [n_fft]
        fout.write(struct.pack("i", n_fft))
        f32(n_fft).tofile(fout)

        # TDT durations  [n_tdt]
        for d in range(n_tdt):
            fout.write(struct.pack("I", d))

        # Vocabulary
        tokens = ['<unk>', '<s>', '</s>'] + [chr(ord('a') + i) for i in range(n_vocab - 3)]
        assert len(tokens) == n_vocab
        fout.write(struct.pack("i", n_vocab))
        for tok in tokens:
            tok_bytes = tok.encode('utf-8')
            fout.write(struct.pack("i", len(tok_bytes)))
            fout.write(tok_bytes)

        # ── Encoder pre_encode ──────────────────────────────────────────────
        # enc_pre_out_w  [n_pre_enc, n_state]  (numpy: n_state × n_pre_enc)
        write_tensor(fout, "encoder.pre_encode.out.weight",    f32(n_state, n_pre_enc))
        write_tensor(fout, "encoder.pre_encode.out.bias",      f32(n_state))

        # conv.0  3×3, 1 in-channel → n_sub_ch  (numpy: n_sub_ch × 1 × 3 × 3)
        write_tensor(fout, "encoder.pre_encode.conv.0.weight", f32(n_sub_ch, 1, 3, 3))
        write_tensor(fout, "encoder.pre_encode.conv.0.bias",   f32(1, n_sub_ch, 1, 1))

        # conv.2  3×3, n_sub_ch → n_sub_ch
        write_tensor(fout, "encoder.pre_encode.conv.2.weight", f32(n_sub_ch, 1, 3, 3))
        write_tensor(fout, "encoder.pre_encode.conv.2.bias",   f32(1, n_sub_ch, 1, 1))

        # conv.3  1×1, n_sub_ch → n_sub_ch
        write_tensor(fout, "encoder.pre_encode.conv.3.weight", f32(n_sub_ch, n_sub_ch, 1, 1))
        write_tensor(fout, "encoder.pre_encode.conv.3.bias",   f32(1, n_sub_ch, 1, 1))

        # conv.5  3×3, n_sub_ch → n_sub_ch
        write_tensor(fout, "encoder.pre_encode.conv.5.weight", f32(n_sub_ch, 1, 3, 3))
        write_tensor(fout, "encoder.pre_encode.conv.5.bias",   f32(1, n_sub_ch, 1, 1))

        # conv.6  1×1, n_sub_ch → n_sub_ch
        write_tensor(fout, "encoder.pre_encode.conv.6.weight", f32(n_sub_ch, n_sub_ch, 1, 1))
        write_tensor(fout, "encoder.pre_encode.conv.6.bias",   f32(1, n_sub_ch, 1, 1))

        # ── Encoder layers ──────────────────────────────────────────────────
        for i in range(n_layer):
            p = f"encoder.layers.{i}"

            # Feed-forward 1
            write_tensor(fout, f"{p}.norm_feed_forward1.weight",     f32(n_state))
            write_tensor(fout, f"{p}.norm_feed_forward1.bias",       f32(n_state))
            write_tensor(fout, f"{p}.feed_forward1.linear1.weight",  f32(4*n_state, n_state))
            write_tensor(fout, f"{p}.feed_forward1.linear2.weight",  f32(n_state, 4*n_state))

            # Convolution module
            write_tensor(fout, f"{p}.norm_conv.weight",                         f32(n_state))
            write_tensor(fout, f"{p}.norm_conv.bias",                            f32(n_state))
            write_tensor(fout, f"{p}.conv.pointwise_conv1.weight",              f32(2*n_state, n_state))
            write_tensor(fout, f"{p}.conv.depthwise_conv.weight",               f32(n_state, n_conv_ker))
            write_tensor(fout, f"{p}.conv.batch_norm.weight",                   f32(n_state))
            write_tensor(fout, f"{p}.conv.batch_norm.bias",                     f32(n_state))
            write_tensor(fout, f"{p}.conv.batch_norm.running_mean",             f32(n_state))
            write_tensor(fout, f"{p}.conv.batch_norm.running_var",              f32(n_state))
            num_batches = np.zeros(1, dtype=np.int32)
            write_tensor(fout, f"{p}.conv.batch_norm.num_batches_tracked",      num_batches)
            write_tensor(fout, f"{p}.conv.pointwise_conv2.weight",              f32(n_state, n_state))

            # Self-attention
            write_tensor(fout, f"{p}.norm_self_att.weight",          f32(n_state))
            write_tensor(fout, f"{p}.norm_self_att.bias",            f32(n_state))
            # pos_bias: [head_dim, n_head] in ggml → numpy shape (n_head, n_head_dim)
            write_tensor(fout, f"{p}.self_attn.pos_bias_u",          f32(n_head, n_head_dim))
            write_tensor(fout, f"{p}.self_attn.pos_bias_v",          f32(n_head, n_head_dim))
            write_tensor(fout, f"{p}.self_attn.linear_q.weight",     f32(n_state, n_state))
            write_tensor(fout, f"{p}.self_attn.linear_k.weight",     f32(n_state, n_state))
            write_tensor(fout, f"{p}.self_attn.linear_v.weight",     f32(n_state, n_state))
            write_tensor(fout, f"{p}.self_attn.linear_out.weight",   f32(n_state, n_state))
            write_tensor(fout, f"{p}.self_attn.linear_pos.weight",   f32(n_state, n_state))

            # Feed-forward 2
            write_tensor(fout, f"{p}.norm_feed_forward2.weight",     f32(n_state))
            write_tensor(fout, f"{p}.norm_feed_forward2.bias",       f32(n_state))
            write_tensor(fout, f"{p}.feed_forward2.linear1.weight",  f32(4*n_state, n_state))
            write_tensor(fout, f"{p}.feed_forward2.linear2.weight",  f32(n_state, 4*n_state))

            # Output norm
            write_tensor(fout, f"{p}.norm_out.weight",               f32(n_state))
            write_tensor(fout, f"{p}.norm_out.bias",                 f32(n_state))

        # ── Prediction network ──────────────────────────────────────────────
        # embed_w: [dec_dim, n_pred_embed] in ggml → numpy (n_pred_embed, dec_dim)
        write_tensor(fout, "decoder.prediction.embed.weight", f32(n_pred_embed, dec_dim))

        for i in range(n_pred_l):
            base = f"decoder.prediction.dec_rnn.lstm"
            # ih_w: [dec_dim, n_lstm_gates] in ggml → numpy (n_lstm_gates, dec_dim)
            write_tensor(fout, f"{base}.weight_ih_l{i}", f32(n_lstm_gates, dec_dim))
            write_tensor(fout, f"{base}.bias_ih_l{i}",   f32(n_lstm_gates))
            write_tensor(fout, f"{base}.bias_hh_l{i}",   f32(n_lstm_gates))
            write_tensor(fout, f"{base}.weight_hh_l{i}", f32(n_lstm_gates, dec_dim))

        # ── Joint network ───────────────────────────────────────────────────
        # pred_w: [dec_dim, dec_dim] in ggml → numpy (dec_dim, dec_dim)
        write_tensor(fout, "joint.pred.weight",        f32(dec_dim, dec_dim))
        write_tensor(fout, "joint.pred.bias",          f32(dec_dim))
        # enc_w: [n_state, dec_dim] in ggml → numpy (dec_dim, n_state)
        write_tensor(fout, "joint.enc.weight",         f32(dec_dim, n_state))
        write_tensor(fout, "joint.enc.bias",           f32(dec_dim))
        # net_w: [dec_dim, n_joint_out] in ggml → numpy (n_joint_out, dec_dim)
        write_tensor(fout, "joint.joint_net.2.weight", f32(n_joint_out, dec_dim))
        write_tensor(fout, "joint.joint_net.2.bias",   f32(n_joint_out))

    size = Path(output_path).stat().st_size
    print(f"Generated {output_path} ({size / 1024:.1f} KB)")

if __name__ == '__main__':
    output = sys.argv[1] if len(sys.argv) > 1 else 'models/for-tests-ggml-parakeet-tdt.bin'
    generate(output)
