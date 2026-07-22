import struct
import torch
import numpy as np
from collections import OrderedDict
from pathlib import Path
import sys

if len(sys.argv) < 3:
    print(
        "Usage: convert-ggml-to-pt.py model.bin dir-output\n")
    sys.exit(1)

fname_inp = Path(sys.argv[1])
dir_out = Path(sys.argv[2])
fname_out = dir_out / "torch-model.pt"



# Open the ggml file
with open(fname_inp, "rb") as f:
    # Read magic number and hyperparameters
    magic_number, n_vocab, n_audio_ctx, n_audio_state, n_audio_head, n_audio_layer, n_text_ctx, n_text_state, n_text_head, n_text_layer, n_mels, use_f16 = struct.unpack("12i", f.read(48))
    print(f"Magic number: {magic_number}")
    print(f"Vocab size: {n_vocab}")
    print(f"Audio context size: {n_audio_ctx}")
    print(f"Audio state size: {n_audio_state}")
    print(f"Audio head size: {n_audio_head}")
    print(f"Audio layer size: {n_audio_layer}")
    print(f"Text context size: {n_text_ctx}")
    print(f"Text head size: {n_text_head}")
    print(f"Mel size: {n_mels}")
    # Read mel filters
    # mel_filters = np.fromfile(f, dtype=np.float32, count=n_mels * 2).reshape(n_mels, 2)
    # print(f"Mel filters: {mel_filters}")
    filters_shape_0 = struct.unpack("i", f.read(4))[0]
    print(f"Filters shape 0: {filters_shape_0}")
    filters_shape_1 = struct.unpack("i", f.read(4))[0]
    print(f"Filters shape 1: {filters_shape_1}")

    # Read tokenizer tokens
    # bytes = f.read(4)
    # print(bytes)
    

    # for i in range(filters.shape[0]):
    # for j in range(filters.shape[1]):
    #     fout.write(struct.pack("f", filters[i][j]))
    mel_filters = np.zeros((filters_shape_0, filters_shape_1))

    for i in range(filters_shape_0):
        for j in range(filters_shape_1):
            mel_filters[i][j] = struct.unpack("f", f.read(4))[0]
            
    bytes_data = f.read(4) 
    num_tokens = struct.unpack("i", bytes_data)[0]
    tokens = {}


    for _ in range(num_tokens):
        token_len = struct.unpack("i", f.read(4))[0]
        token = f.read(token_len)
        tokens[token] = {}
    
    # Read model variables
    model_state_dict = OrderedDict()
    while True:
        try:
            n_dims, name_length, ftype = struct.unpack("iii", f.read(12))
        except struct.error:
            break  # End of file
        dims = [struct.unpack("i", f.read(4))[0] for _ in range(n_dims)]
        dims = dims[::-1]
        name = f.read(name_length).decode("utf-8")
        if ftype == 1:  # f16
            data = np.fromfile(f, dtype=np.float16, count=np.prod(dims)).reshape(dims)
        else:  # f32
            data = np.fromfile(f, dtype=np.float32, count=np.prod(dims)).reshape(dims)

            
        if name in  ["encoder.conv1.bias", "encoder.conv2.bias"]:
            
            data = data[:, 0]
        
            
        model_state_dict[name] = torch.from_numpy(data)
    
# Now you have the model's state_dict stored in model_state_dict
# You can load this state_dict into a model with the same architecture

# dims = ModelDimensions(**checkpoint["dims"])
# model = Whisper(dims)
from whisper import Whisper, ModelDimensions
dims = ModelDimensions(
    n_mels=n_mels,
    n_audio_ctx=n_audio_ctx,
    n_audio_state=n_audio_state,
    n_audio_head=n_audio_head,
    n_audio_layer=n_audio_layer,
    n_text_ctx=n_text_ctx,
    n_text_state=n_text_state,
    n_text_head=n_text_head,
    n_text_layer=n_text_layer,
    n_vocab=n_vocab,
)
model = Whisper(dims)  # Replace with your model's class
model.load_state_dict(model_state_dict)

# Save the model in PyTorch format
torch.save(model.state_dict(), fname_out)
