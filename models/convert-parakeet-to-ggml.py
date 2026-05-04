#!/usr/bin/env python3
# Convert Parakeet TDT model from NeMo format to ggml format
#
# Usage: python convert-parakeet-to-ggml.py --model parakeet-model.nemo --output-dir output-dir [--use-f32]
#
# The NeMo file is a tar archive containing:
#   - model_weights.ckpt (PyTorch checkpoint)
#   - model_config.yaml (model configuration)
#   - tokenizer files
#
# This script extracts the NeMo archive, loads the model weights and configuration,
# and saves them in ggml format compatible with whisper.cpp.
#

import torch
import argparse
import io
import os
import sys
import struct
import tarfile
import tempfile
import shutil
import yaml
import numpy as np
from pathlib import Path
from typing import Optional

def hz_to_mel(freq):
    return 2595.0 * np.log10(1.0 + freq / 700.0)

def mel_to_hz(mel):
    return 700.0 * (10.0**(mel / 2595.0) - 1.0)

def extract_nemo_archive(nemo_path, extract_dir):
    print(f"Extracting {nemo_path} to {extract_dir}")
    with tarfile.open(nemo_path, 'r') as tar:
        tar.extractall(path=extract_dir)
    print("Extraction complete")

def load_model_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_tokenizer(extract_dir, config):
    tokenizer_model_path = None
    tokenizer_vocab_path = None

    for file in os.listdir(extract_dir):
        if file.endswith('_tokenizer.model'):
            tokenizer_model_path = os.path.join(extract_dir, file)
        elif file.endswith('tokenizer.vocab'):
            tokenizer_vocab_path = os.path.join(extract_dir, file)

    if not tokenizer_model_path:
        raise FileNotFoundError("Tokenizer model file not found")

    if not tokenizer_vocab_path:
        raise FileNotFoundError("Tokenizer vocab file not found")

    tokens = {}
    with open(tokenizer_vocab_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            parts = line.strip().split('\t')
            if len(parts) >= 1:
                token = parts[0]
                tokens[token.encode('utf-8')] = idx

    print(f"Loaded {len(tokens)} tokens from {os.path.basename(tokenizer_vocab_path)}")

    if len(tokens) != 8192:
        print(f"WARNING: Expected 8192 tokens, got {len(tokens)}")

    return tokens

def write_tensor(fout, name, data, use_f16=True, force_f32=False):
    if 'pre_encode.conv' in name and 'bias' in name and len(data.shape) == 1:
        data = data.reshape(1, -1, 1, 1)
        print(f"  Reshaped conv bias {name} to {data.shape}")

    n_dims = len(data.shape)

    ftype = 1 if use_f16 and not force_f32 else 0
    if force_f32:
        data = data.astype(np.float32)
    elif use_f16:
        if n_dims < 2 or 'bias' in name or 'norm' in name:
            data = data.astype(np.float32)
            ftype = 0
        else:
            data = data.astype(np.float16)
    else:
        data = data.astype(np.float32)

    dims_reversed = [data.shape[n_dims - 1 - i] for i in range(n_dims)]
    print(f"Processing: {name} {list(data.shape)}, dtype: {data.dtype}, n_dims: {n_dims}, reversed: {dims_reversed}")
    name_bytes = name.encode('utf-8')
    fout.write(struct.pack("iii", n_dims, len(name_bytes), ftype))
    for i in range(n_dims):
        fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
    fout.write(name_bytes)

    data.tofile(fout)

def convert_parakeet_to_ggml(nemo_path, output_dir, use_f16=True, out_name=None):
    nemo_path = Path(nemo_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create temporary directory for extraction
    with tempfile.TemporaryDirectory() as temp_dir:
        extract_nemo_archive(nemo_path, temp_dir)

        config_path = os.path.join(temp_dir, 'model_config.yaml')
        config = load_model_config(config_path)

        print("Model configuration:")
        print(f"  Sample rate: {config['sample_rate']}")
        print(f"  Encoder layers: {config['encoder']['n_layers']}")
        print(f"  Encoder d_model: {config['encoder']['d_model']}")
        print(f"  Mel features: {config['preprocessor']['features']}")

        weights_path = os.path.join(temp_dir, 'model_weights.ckpt')
        print(f"\nLoading model weights from {weights_path}")
        checkpoint = torch.load(weights_path, map_location='cpu')

        # Extract state dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        print(f"Loaded {len(state_dict)} tensors")

        # Load tokenizer
        print("\nLoading tokenizer...")
        tokens = load_tokenizer(temp_dir, config)
        print(f"Loaded {len(tokens)} tokens")

        # Prepare hyperparameters for the Parakeet ggml format.
        hparams = {
            'n_audio_ctx': 5000,
            'n_audio_state': config['encoder']['d_model'],
            'n_audio_head': config['encoder']['n_heads'],
            'n_audio_layer': config['encoder']['n_layers'],
            'n_mels': config['preprocessor']['features'],
            'n_fft': config['preprocessor']['n_fft'],
            'subsampling_factor': config['encoder']['subsampling_factor'],
            'n_subsampling_channels': config['encoder']['subsampling_conv_channels'],

            'n_pred_dim': config['decoder']['prednet']['pred_hidden'],
            'n_pred_layers': config['decoder']['prednet']['pred_rnn_layers'],
            'n_vocab': config['decoder']['vocab_size'],
            'n_tdt_durations': config['model_defaults']['num_tdt_durations'],
            'n_max_tokens': config['decoding']['greedy']['max_symbols'],
        }

        print("\nGGML hyperparameters:")
        for key, value in hparams.items():
            print(f"  {key}: {value}")

        # Create output file
        if out_name:
            fname_out = output_dir / out_name
        else:
            fname_out = output_dir / ("ggml-model-f32.bin" if not use_f16 else "ggml-model.bin")
        print(f"\nWriting to {fname_out}")

        with open(fname_out, 'wb') as fout:
            # Write magic number
            fout.write(struct.pack("i", 0x67676d6c))  # 'ggml' in hex

            # Write hyperparameters
            fout.write(struct.pack("i", hparams['n_vocab']))
            fout.write(struct.pack("i", hparams['n_audio_ctx']))
            fout.write(struct.pack("i", hparams['n_audio_state']))
            fout.write(struct.pack("i", hparams['n_audio_head']))
            fout.write(struct.pack("i", hparams['n_audio_layer']))
            fout.write(struct.pack("i", hparams['n_mels']))
            fout.write(struct.pack("i", 1 if use_f16 else 0))
            fout.write(struct.pack("i", hparams['n_fft']))
            fout.write(struct.pack("i", hparams['subsampling_factor']))
            fout.write(struct.pack("i", hparams['n_subsampling_channels']))
            fout.write(struct.pack("i", hparams['n_pred_dim']))
            fout.write(struct.pack("i", hparams['n_pred_layers']))
            fout.write(struct.pack("i", hparams['n_tdt_durations']))
            fout.write(struct.pack("i", hparams['n_max_tokens']))

            # Extract mel filterbank from model
            fb_key = None
            for key in state_dict.keys():
                if 'featurizer.fb' in key or 'filterbank' in key.lower():
                    fb_key = key
                    break

            if not fb_key:
                print("\nERROR: Mel filterbank not found in model!")
                print("Expected tensor with 'featurizer.fb' or 'filterbank' in name")
                print("\nAvailable preprocessor tensors:")
                for key in sorted(state_dict.keys()):
                    if 'preprocessor' in key or 'featurizer' in key:
                        print(f"  {key}: {state_dict[key].shape}")
                raise ValueError("Mel filterbank tensor not found in model")

            print(f"\nUsing model's mel filterbank from: {fb_key}")
            mel_filters = state_dict[fb_key].squeeze().numpy().astype(np.float32)
            print(f"  Filterbank shape: {mel_filters.shape}")
            print(f"  Filterbank min/max values: {mel_filters.min():.6f} / {mel_filters.max():.6f}")
            print(f"  Filterbank non-zero elements: {np.count_nonzero(mel_filters)} / {mel_filters.size}")
            print(f"  First row sum: {mel_filters[0].sum():.6f}")

            if len(mel_filters.shape) != 2:
                raise ValueError(f"Expected 2D filterbank, got shape {mel_filters.shape}")

            n_mels, n_freqs = mel_filters.shape
            fout.write(struct.pack("i", n_mels))      # n_mel
            fout.write(struct.pack("i", n_freqs))     # n_fb (frequency bins)

            # Write mel filterbank
            for i in range(n_mels):
                for j in range(n_freqs):
                    fout.write(struct.pack("f", mel_filters[i, j]))

            # Extract window function from model
            window_key = None
            for key in state_dict.keys():
                if 'featurizer.window' in key or 'preproc' in key and 'window' in key:
                    window_key = key
                    break

            if not window_key:
                print("\nERROR: Window function not found in model!")
                print("Expected tensor with 'featurizer.window' in name")
                raise ValueError("Window function tensor not found in model")

            print(f"\nUsing model's window function from: {window_key}")
            window = state_dict[window_key].squeeze().numpy().astype(np.float32)
            print(f"  Window shape: {window.shape}")
            print(f"  Window min/max values: {window.min():.6f} / {window.max():.6f}")
            print(f"  Window non-zero elements: {np.count_nonzero(window)} / {window.size}")
            print(f"  Window sum: {window.sum():.6f}")

            if len(window.shape) != 1:
                raise ValueError(f"Expected 1D window, got shape {window.shape}")

            n_window = window.shape[0]
            fout.write(struct.pack("i", n_window))

            # Write window function
            for i in range(n_window):
                fout.write(struct.pack("f", window[i]))

            # Write TDT durations
            tdt_durations = config['model_defaults']['tdt_durations']
            if len(tdt_durations) != hparams['n_tdt_durations']:
                raise ValueError(f"TDT durations count mismatch: {len(tdt_durations)} vs {hparams['n_tdt_durations']}")

            for duration in tdt_durations:
                fout.write(struct.pack("I", duration))

            fout.write(struct.pack("i", len(tokens)))
            for token_bytes, idx in sorted(tokens.items(), key=lambda x: x[1]):
                fout.write(struct.pack("i", len(token_bytes)))
                fout.write(token_bytes)

            print("\nConverting model weights...")
            for name, tensor in state_dict.items():
                # Skip the filterbank and window - already written in preprocessing section
                if name == fb_key:
                    continue
                if name == window_key:
                    continue

                # Don't squeeze Conv2d weights - they need to preserve all 4 dimensions
                if 'conv' in name and 'weight' in name and len(tensor.shape) == 4:
                    data = tensor.numpy()
                else:
                    data = tensor.squeeze().numpy()

                write_tensor(fout, name, data, use_f16=use_f16)

        print(f"\nConversion complete!")
        print(f"Output file: {fname_out}")
        print(f"File size: {fname_out.stat().st_size / (1024**2):.2f} MB")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert Parakeet TDT model from NeMo format to ggml format'
    )
    parser.add_argument('--model', type=str, required=True,
                        help='Path to Parakeet .nemo model file')
    parser.add_argument('--out-dir', type=str, required=True,
                        help='Directory to write ggml model file')
    parser.add_argument('--use-f32', action='store_true', default=False,
                        help='Use f32 instead of f16 (default: f16)')
    parser.add_argument('--out-name', type=str, default=None,
                        help='Output file name (default: ggml-model.bin or ggml-model-f32.bin)')

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: {args.model} not found")
        sys.exit(1)

    use_f16 = not args.use_f32
    convert_parakeet_to_ggml(args.model, args.out_dir, use_f16, args.out_name)
