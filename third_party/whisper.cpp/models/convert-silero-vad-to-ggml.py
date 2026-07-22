import os
import struct
import argparse
import torch
import numpy as np
from silero_vad import load_silero_vad, __version__ as silero_version

def convert_silero_vad(output_path, print_tensors=True):
    model = load_silero_vad()
    state_dict = model.state_dict()

    # Clean up state dict keys - filter out 8k model
    cleaned_dict = {}
    for key, value in state_dict.items():
        # Skip 8k model
        if "_8k" not in key:
            clean_key = key
            if not key.startswith("_model."):
                clean_key = "_model." + key
            cleaned_dict[clean_key] = value

    base, ext = os.path.splitext(output_path)
    output_file = f"{base}-v{silero_version}-ggml{ext}"
    print(f"Saving GGML Silero-VAD model to {output_file}")

    print("\nTensor info for debugging:")
    for key, tensor in cleaned_dict.items():
        print(f"  - {key}: {tensor.shape} ({tensor.dtype})")
    print()

    with open(output_file, "wb") as fout:
        # Write magic and version
        fout.write(struct.pack("i", 0x67676d6c))

        model_type = "silero-16k"
        str_len = len(model_type)
        fout.write(struct.pack("i", str_len))
        fout.write(model_type.encode('utf-8'))

        version_parts = silero_version.split('.')
        major, minor, patch = map(int, version_parts)
        print(f"Version: {major}.{minor}.{patch}")
        fout.write(struct.pack("i", major))
        fout.write(struct.pack("i", minor))
        fout.write(struct.pack("i", patch))

        # Write model architecture parameters
        window_size = 512
        fout.write(struct.pack("i", window_size))
        context_size = 64
        fout.write(struct.pack("i", context_size))

        n_encoder_layers = 4
        fout.write(struct.pack("i", n_encoder_layers))

        # Write encoder dimensions
        input_channels = 129
        encoder_in_channels = [input_channels, 128, 64, 64]
        encoder_out_channels = [128, 64, 64, 128]
        kernel_size = 3

        for i in range(n_encoder_layers):
            fout.write(struct.pack("i", encoder_in_channels[i]))
            fout.write(struct.pack("i", encoder_out_channels[i]))
            fout.write(struct.pack("i", kernel_size))

        # Write LSTM dimensions
        lstm_input_size = 128
        lstm_hidden_size = 128
        fout.write(struct.pack("i", lstm_input_size))
        fout.write(struct.pack("i", lstm_hidden_size))

        # Write final conv dimensions
        final_conv_in = 128
        final_conv_out = 1
        fout.write(struct.pack("i", final_conv_in))
        fout.write(struct.pack("i", final_conv_out))

        # Define tensor keys to write
        tensor_keys = []

        # Encoder weights
        for i in range(n_encoder_layers):
            weight_key = f"_model.encoder.{i}.reparam_conv.weight"
            bias_key = f"_model.encoder.{i}.reparam_conv.bias"
            if weight_key in cleaned_dict and bias_key in cleaned_dict:
                tensor_keys.append(weight_key)
                tensor_keys.append(bias_key)

        # LSTM weights
        lstm_keys = [
            "_model.decoder.rnn.weight_ih",
            "_model.decoder.rnn.weight_hh",
            "_model.decoder.rnn.bias_ih",
            "_model.decoder.rnn.bias_hh"
        ]
        tensor_keys.extend([k for k in lstm_keys if k in cleaned_dict])

        # Final conv weights
        final_keys = [
            "_model.decoder.decoder.2.weight",
            "_model.decoder.decoder.2.bias"
        ]
        tensor_keys.extend([k for k in final_keys if k in cleaned_dict])

        # STFT basis - add this last
        stft_tensor = "_model.stft.forward_basis_buffer"
        tensor_keys.append(stft_tensor)

        print(f"Writing {len(tensor_keys)} tensors:")
        for key in tensor_keys:
            if key in cleaned_dict:
                print(f"  - {key}: {cleaned_dict[key].shape}")
            else:
                print(f"  - {key}: MISSING")

        # Process each tensor
        for key in tensor_keys:
            if key not in cleaned_dict:
                print(f"Warning: Missing tensor {key}, skipping")
                continue

            tensor = cleaned_dict[key]

            # Special handling for STFT tensor
            if key == "_model.stft.forward_basis_buffer":
                # Get the original numpy array without squeezing
                data = tensor.detach().cpu().numpy()
                # Ensure it has the expected shape
                print(f"STFT tensor original shape: {data.shape}")
                n_dims = 3
                tensor_shape = [data.shape[2], data.shape[1], data.shape[0]]
                is_conv_weight = True
            else:
                # For other tensors, we can use standard processing
                data = tensor.detach().cpu().squeeze().numpy()
                tensor_shape = list(data.shape)

                # Ensure we have at most 4 dimensions for GGML
                n_dims = min(len(tensor_shape), 4)

                # Reverse dimensions for GGML
                tensor_shape = tensor_shape[:n_dims]
                tensor_shape.reverse()

                # Check if this is a convolution weight tensor
                is_conv_weight = "weight" in key and ("encoder" in key or "_model.decoder.decoder.2" in key)

            # Convert to float16 for convolution weights
            if is_conv_weight:
                data = data.astype(np.float16)
                ftype = 1  # float16
            else:
                ftype = 0  # float32

            # Debug printing of tensor info
            print(f"\nWriting tensor: {key}")
            print(f"  Original shape: {tensor.shape}")
            print(f"  Processed shape: {data.shape}")
            print(f"  GGML dimensions: {n_dims}")
            print(f"  GGML shape: {tensor_shape}")
            print(f"  Type: {'float16' if ftype == 1 else 'float32'}")

            # Convert tensor name to bytes
            name_bytes = key.encode('utf-8')
            name_length = len(name_bytes)

            # Write tensor header
            fout.write(struct.pack("i", n_dims))
            fout.write(struct.pack("i", name_length))
            fout.write(struct.pack("i", ftype))

            # Write tensor dimensions
            for i in range(n_dims):
                size = tensor_shape[i] if i < len(tensor_shape) else 1
                fout.write(struct.pack("i", size))
                print(f"  Writing dimension {i}: {size}")

            # Write tensor name
            fout.write(name_bytes)

            # Write tensor data
            data.tofile(fout)

            print(f"  Wrote {data.size * (2 if ftype==1 else 4)} bytes")

    print(f"\nDone! Model has been converted to GGML format: {output_file}")
    print(f"File size: {os.path.getsize(output_file)} bytes")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Silero-VAD PyTorch model to GGML format")
    parser.add_argument("--output", type=str, required=True, help="Path to output GGML model file")
    parser.add_argument("--print-tensors", action="store_true", help="Print tensor values", default=True)
    args = parser.parse_args()

    convert_silero_vad(args.output, args.print_tensors)
