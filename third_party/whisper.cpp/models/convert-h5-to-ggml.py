# Convert Hugging Face fine-tuned models to ggml format
#
# Usage:
#
#   git clone https://github.com/openai/whisper
#   git clone https://github.com/ggml-org/whisper.cpp
#   git clone https://huggingface.co/openai/whisper-medium
#
#   python3 ./whisper.cpp/models/convert-h5-to-ggml.py ./whisper-medium/ ./whisper .
#
# This script is similar to "convert-pt-to-ggml.py"
#
# For more info:
#
#   https://github.com/ggml-org/whisper.cpp/issues/157
#

import io
import os
import sys
import struct
import json
import code
import torch
import numpy as np
from pathlib import Path

from transformers import WhisperForConditionalGeneration

conv_map = {
        'self_attn.k_proj'              : 'attn.key',
        'self_attn.q_proj'              : 'attn.query',
        'self_attn.v_proj'              : 'attn.value',
        'self_attn.out_proj'            : 'attn.out',
        'self_attn_layer_norm'          : 'attn_ln',
        'encoder_attn.q_proj'           : 'cross_attn.query',
        'encoder_attn.v_proj'           : 'cross_attn.value',
        'encoder_attn.out_proj'         : 'cross_attn.out',
        'encoder_attn_layer_norm'       : 'cross_attn_ln',
        'fc1'                           : 'mlp.0',
        'fc2'                           : 'mlp.2',
        'final_layer_norm'              : 'mlp_ln',
        'encoder.layer_norm.bias'       : 'encoder.ln_post.bias',
        'encoder.layer_norm.weight'     : 'encoder.ln_post.weight',
        'encoder.embed_positions.weight': 'encoder.positional_embedding',
        'decoder.layer_norm.bias'       : 'decoder.ln.bias',
        'decoder.layer_norm.weight'     : 'decoder.ln.weight',
        'decoder.embed_positions.weight': 'decoder.positional_embedding',
        'decoder.embed_tokens.weight'   : 'decoder.token_embedding.weight',
        'proj_out.weight'               : 'decoder.proj.weight',
        }

# ref: https://github.com/openai/gpt-2/blob/master/src/encoder.py
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a significant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

if len(sys.argv) < 4:
    print("Usage: convert-h5-to-ggml.py dir_model path-to-whisper-repo dir-output [use-f32]\n")
    sys.exit(1)

dir_model   = Path(sys.argv[1])
dir_whisper = Path(sys.argv[2])
dir_out     = Path(sys.argv[3])

encoder = json.load((dir_model / "vocab.json").open("r", encoding="utf8"))
encoder_added = json.load((dir_model / "added_tokens.json").open( "r", encoding="utf8"))
hparams = json.load((dir_model / "config.json").open("r", encoding="utf8"))

# Add this block to handle missing 'max_length'
if "max_length" not in hparams or hparams["max_length"] is None:
    hparams["max_length"] = hparams.get("max_target_positions", 448)  # Default to 448 if missing
elif not isinstance(hparams["max_length"], int):
    try:
        hparams["max_length"] = int(hparams["max_length"])  # Convert if necessary
    except ValueError:
        print(f"Warning: Invalid max_length value '{hparams['max_length']}', using default 448.")
        hparams["max_length"] = 448
        
model = WhisperForConditionalGeneration.from_pretrained(dir_model)

#code.interact(local=locals())

n_mels = hparams["num_mel_bins"]
with np.load(os.path.join(dir_whisper, "whisper/assets", "mel_filters.npz")) as f:
    filters = torch.from_numpy(f[f"mel_{n_mels}"])

dir_tokenizer = dir_model

fname_out = dir_out / "ggml-model.bin"

tokens = json.load(open(dir_tokenizer / "vocab.json", "r", encoding="utf8"))

# use 16-bit or 32-bit floats
use_f16 = True
if len(sys.argv) > 4:
    use_f16 = False
    fname_out = dir_out / "ggml-model-f32.bin"

fout = open(fname_out, "wb")

fout.write(struct.pack("i", 0x67676d6c)) # magic: ggml in hex
fout.write(struct.pack("i", hparams["vocab_size"]))
fout.write(struct.pack("i", hparams["max_source_positions"]))
fout.write(struct.pack("i", hparams["d_model"]))
fout.write(struct.pack("i", hparams["encoder_attention_heads"]))
fout.write(struct.pack("i", hparams["encoder_layers"]))
fout.write(struct.pack("i", hparams["max_length"]))
fout.write(struct.pack("i", hparams["d_model"]))
fout.write(struct.pack("i", hparams["decoder_attention_heads"]))
fout.write(struct.pack("i", hparams["decoder_layers"]))
fout.write(struct.pack("i", hparams["num_mel_bins"]))
fout.write(struct.pack("i", use_f16))

fout.write(struct.pack("i", filters.shape[0]))
fout.write(struct.pack("i", filters.shape[1]))
for i in range(filters.shape[0]):
    for j in range(filters.shape[1]):
        fout.write(struct.pack("f", filters[i][j]))

byte_encoder = bytes_to_unicode()
byte_decoder = {v:k for k, v in byte_encoder.items()}

fout.write(struct.pack("i", len(tokens)))

tokens = sorted(tokens.items(), key=lambda x: x[1])
for key in tokens:
    text = bytearray([byte_decoder[c] for c in key[0]])
    fout.write(struct.pack("i", len(text)))
    fout.write(text)

list_vars = model.state_dict()
for name in list_vars.keys():
    # this seems to not be used
    # ref: https://github.com/huggingface/transformers/blob/9a5b84a0076a04fe9596da72e8668069d4f09ea0/src/transformers/models/whisper/modeling_whisper.py#L1099-L1106
    if name == "proj_out.weight":
        print('Skipping', name)
        continue

    src = name

    nn = name
    if name != "proj_out.weight":
        nn = nn.split(".")[1:]
    else:
        nn = nn.split(".")

    if nn[1] == "layers":
        nn[1] = "blocks"
        if ".".join(nn[3:-1]) == "encoder_attn.k_proj":
            mapped = "attn.key" if nn[0] == "encoder" else "cross_attn.key"
        else:
            mapped = conv_map[".".join(nn[3:-1])]
        name = ".".join(nn[:3] + [mapped] + nn[-1:])
    else:
        name = ".".join(nn)
        name = conv_map[name] if name in conv_map else name

    print(src, ' -> ', name)
    data = list_vars[src].squeeze().numpy()
    data = data.astype(np.float16)

    # reshape conv bias from [n] to [n, 1]
    if name in ["encoder.conv1.bias", "encoder.conv2.bias"]:
        data = data.reshape(data.shape[0], 1)
        print("  Reshaped variable: " , name , " to shape: ", data.shape)

    n_dims = len(data.shape)
    print(name, n_dims, data.shape)

    # looks like the whisper models are in f16 by default
    # so we need to convert the small tensors to f32 until we fully support f16 in ggml
    # ftype == 0 -> float32, ftype == 1 -> float16
    ftype = 1
    if use_f16:
        if n_dims < 2 or \
                name == "encoder.conv1.bias"   or \
                name == "encoder.conv2.bias"   or \
                name == "encoder.positional_embedding" or \
                name == "decoder.positional_embedding":
            print("  Converting to float32")
            data = data.astype(np.float32)
            ftype = 0
    else:
        data = data.astype(np.float32)
        ftype = 0

    # header
    str_ = name.encode('utf-8')
    fout.write(struct.pack("iii", n_dims, len(str_), ftype))
    for i in range(n_dims):
        fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
    fout.write(str_)

    # data
    data.tofile(fout)

fout.close()

print("Done. Output file: " , fname_out)
print("")
