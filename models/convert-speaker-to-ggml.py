#!/usr/bin/env python3
"""
Convert SpeechBrain speaker embedding model (ECAPA-TDNN) to GGML binary format.

This script loads the pre-trained SpeakerRecognition model from SpeechBrain,
extracts weights, and converts them to GGML binary format for use with whisper.cpp.

GGML Format:
- Magic: 0x67676d6c (4 bytes, "ggml")
- Model type: string length (4 bytes) + UTF-8 string
- Version: major, minor, patch (3 x 4 bytes)
- Hyperparameters: embedding_dim, n_channels (2 x 4 bytes)
- Tensor count (4 bytes)
- Tensors: for each tensor:
  - n_dims (4 bytes)
  - name_len (4 bytes)
  - dims (n_dims x 4 bytes)
  - name (name_len bytes)
  - data (product(dims) x 4 bytes, float32)

Usage:
    python convert-speaker-to-ggml.py --output ggml-speaker-ecapa-tdnn.bin
    python convert-speaker-to-ggml.py --model speechbrain/spkrec-ecapa-voxceleb --output custom.bin
    python convert-speaker-to-ggml.py --test  # Minimal test run
"""

import os
import struct
import argparse
import sys
import tempfile
import torch
import numpy as np
from pathlib import Path

def load_speaker_model(model_name: str, tmp_dir: str = None):
    """
    Load SpeakerRecognition model from SpeechBrain.

    Args:
        model_name: HuggingFace model identifier (default: speechbrain/spkrec-ecapa-voxceleb)
        tmp_dir: temporary directory for model cache

    Returns:
        Model object and state_dict
    """
    try:
        # Monkey-patch for torchaudio >= 2.6 compatibility
        import torchaudio
        if not hasattr(torchaudio, 'list_audio_backends'):
            torchaudio.list_audio_backends = lambda: ['default']

        # SpeechBrain >= 1.0 moved pretrained to inference
        try:
            from speechbrain.inference.speaker import SpeakerRecognition
        except ImportError:
            from speechbrain.pretrained import SpeakerRecognition
    except ImportError as e:
        print(f"Error: Failed to import SpeechBrain. Install with: pip install -r requirements-convert.txt")
        print(f"Original error: {e}")
        sys.exit(1)

    if tmp_dir is None:
        tmp_dir = tempfile.mkdtemp(prefix='spkrec_')

    print(f"Loading SpeechBrain model: {model_name}")
    print(f"Cache directory: {tmp_dir}")

    try:
        model = SpeakerRecognition.from_hparams(
            source=model_name,
            savedir=tmp_dir,
            run_opts={'device': 'cpu'},
            freeze_params=True
        )
    except Exception as e:
        print(f"Failed to load model from {model_name}: {e}")
        sys.exit(1)

    print("Model loaded successfully")

    # Extract state_dict
    state_dict = model.state_dict()

    print(f"State dict contains {len(state_dict)} tensors")

    return model, state_dict

def convert_speaker_model(model_name: str, output_path: str, test_mode: bool = False, validate_mode: bool = False):
    """
    Convert SpeechBrain speaker model to GGML binary format.

    Args:
        model_name: HuggingFace model identifier
        output_path: output file path for GGML binary
        test_mode: if True, skip validation
        validate_mode: if True, run validation after conversion
    """
    print("\n" + "="*60)
    print("SpeechBrain → GGML Speaker Model Conversion")
    print("="*60)

    # Load model
    model, state_dict = load_speaker_model(model_name)

    # Write GGML binary
    print(f"\nWriting GGML binary to: {output_path}")

    output_dir = os.path.dirname(output_path) or '.'
    os.makedirs(output_dir, exist_ok=True)

    with open(output_path, 'wb') as fout:
        # Write magic number
        magic = 0x67676d6c  # "ggml"
        fout.write(struct.pack('i', magic))
        print(f"Magic number: 0x{magic:08x}")

        # Write model type
        model_type = b"spkrec-ecapa-tdnn"
        fout.write(struct.pack('i', len(model_type)))
        fout.write(model_type)
        print(f"Model type: {model_type.decode('utf-8')}")

        # Write version
        version_major = 1
        version_minor = 0
        version_patch = 0
        fout.write(struct.pack('i', version_major))
        fout.write(struct.pack('i', version_minor))
        fout.write(struct.pack('i', version_patch))
        print(f"Version: {version_major}.{version_minor}.{version_patch}")

        # Write hyperparameters
        embedding_dim = 192   # ECAPA-TDNN output dimension
        n_channels = 1024    # Internal channel width
        fout.write(struct.pack('i', embedding_dim))
        fout.write(struct.pack('i', n_channels))
        print(f"Embedding dimension: {embedding_dim}")
        print(f"Internal channels: {n_channels}")

        # Count all tensors with dim > 0 (skip scalars like num_batches_tracked)
        n_tensors = sum(1 for k, v in state_dict.items() if v.dim() > 0)
        fout.write(struct.pack('i', n_tensors))
        n_scalars = len(state_dict) - n_tensors
        print(f"Tensor count: {n_tensors} (skipping {n_scalars} scalars)")

    # Write all tensors (no BN fusion — SpeechBrain uses Conv→ReLU→BN order,
    # so BN cannot be fused into conv weights; C++ applies runtime BN after ReLU)
    print("\nWriting tensors (no fusion, runtime BN in C++):")

    tensor_count = 0
    total_bytes = 0

    for name, tensor in state_dict.items():
        if tensor.dim() == 0:
            continue  # skip scalars (e.g. num_batches_tracked)

        data = tensor.detach().cpu().numpy().astype(np.float32)

        with open(output_path, 'ab') as fout:
            n_dims = len(data.shape)

            # Write tensor header
            fout.write(struct.pack('i', n_dims))

            name_bytes = name.encode('utf-8')
            fout.write(struct.pack('i', len(name_bytes)))

            # Write dimensions in REVERSED order for ggml column-major compatibility.
            # NumPy row-major: last dim varies fastest in memory.
            # ggml column-major: ne[0] varies fastest in memory.
            # By reversing, ne[0] = last PyTorch dim, matching the memory layout.
            for dim in reversed(data.shape):
                fout.write(struct.pack('i', dim))

            # Write tensor name
            fout.write(name_bytes)

            # Write tensor data (row-major bytes, matching reversed ggml dims)
            tensor_bytes = data.tobytes()
            fout.write(tensor_bytes)

            total_bytes += len(tensor_bytes)
            tensor_count += 1

            shape_str = 'x'.join(str(d) for d in data.shape)
            ggml_dims = 'x'.join(str(d) for d in reversed(data.shape))
            print(f"  [{tensor_count}] {name}: {shape_str} → ggml [{ggml_dims}]")

    # Verify output
    if not os.path.exists(output_path):
        print(f"\nError: Output file not created: {output_path}")
        sys.exit(1)

    file_size = os.path.getsize(output_path)
    file_size_mb = file_size / 1024 / 1024

    print(f"\n" + "="*60)
    print("Conversion complete!")
    print("="*60)
    print(f"Output file: {output_path}")
    print(f"File size: {file_size_mb:.2f} MB ({file_size} bytes)")
    print(f"Tensors written: {tensor_count}")
    print(f"Tensor data size: {total_bytes / 1024 / 1024:.2f} MB")

    if validate_mode:
        print("\nRunning validation...")
        validate_conversion(output_path)

    if not test_mode:
        print("\nNext steps:")
        print(f"  1. Compile C++ test: cd build && cmake .. && make")
        print(f"  2. Run test: ./test-speaker-model-load ../{output_path}")
        print(f"  3. Validate numerically: python models/validate-speaker-model.py {output_path}")

def validate_conversion(ggml_path: str):
    """
    Quick validation: load GGML file and check magic number and header.
    """
    if not os.path.exists(ggml_path):
        print(f"Error: File not found: {ggml_path}")
        return False

    with open(ggml_path, 'rb') as fin:
        # Read magic
        magic_bytes = fin.read(4)
        magic = struct.unpack('i', magic_bytes)[0]

        if magic != 0x67676d6c:
            print(f"Invalid magic number: 0x{magic:08x} (expected 0x67676d6c)")
            return False

        print(f"Magic number valid: 0x{magic:08x}")

        str_len = struct.unpack('i', fin.read(4))[0]
        model_type = fin.read(str_len).decode('utf-8')
        print(f"Model type: {model_type}")

        major, minor, patch = struct.unpack('iii', fin.read(12))
        print(f"Version: {major}.{minor}.{patch}")

        embedding_dim = struct.unpack('i', fin.read(4))[0]
        n_channels = struct.unpack('i', fin.read(4))[0]
        print(f"Embedding dimension: {embedding_dim}")
        print(f"Internal channels: {n_channels}")

        n_tensors = struct.unpack('i', fin.read(4))[0]
        print(f"Tensor count: {n_tensors}")

    return True

def main():
    parser = argparse.ArgumentParser(
        description='Convert SpeechBrain speaker embedding model to GGML binary format'
    )
    parser.add_argument(
        '--model',
        default='speechbrain/spkrec-ecapa-voxceleb',
        help='HuggingFace model identifier (default: speechbrain/spkrec-ecapa-voxceleb)'
    )
    parser.add_argument(
        '--output',
        default='ggml-speaker-ecapa-tdnn.bin',
        help='Output file path (default: ggml-speaker-ecapa-tdnn.bin)'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode: minimal output, no verification'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Run validation after conversion'
    )

    args = parser.parse_args()

    try:
        convert_speaker_model(
            args.model,
            args.output,
            test_mode=args.test,
            validate_mode=args.validate
        )
    except KeyboardInterrupt:
        print("\n\nConversion cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
