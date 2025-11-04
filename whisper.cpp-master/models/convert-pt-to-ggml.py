# Convert Whisper transformer model from PyTorch to ggml format
#
# Usage: python convert-pt-to-ggml.py ~/.cache/whisper/medium.pt ~/path/to/repo/whisper/ ./models/whisper-medium
#
# You need to clone the original repo in ~/path/to/repo/whisper/
#
#  git clone https://github.com/openai/whisper ~/path/to/repo/whisper/
#
# It is used to various assets needed by the algorithm:
#
#  - tokenizer
#  - mel filters
#
# Also, you need to have the original models in ~/.cache/whisper/
# See the original repo for more details.
#
# This script loads the specified model and whisper assets and saves them in ggml format.
# The output is a single binary file containing the following information:
#
#  - hparams
#  - mel filters
#  - tokenizer vocab
#  - model variables
#
# For each variable, write the following:
#
#  - Number of dimensions (int)
#  - Name length (int)
#  - Dimensions (int[n_dims])
#  - Name (char[name_length])
#  - Data (float[n_dims])
#

import io
import os
import sys
import struct
import json
import code
import torch
import numpy as np
import base64
from pathlib import Path
#from transformers import GPTJForCausalLM
#from transformers import GPT2TokenizerFast

# ref: https://github.com/openai/whisper/blob/8cf36f3508c9acd341a45eb2364239a3d81458b9/whisper/tokenizer.py#L10-L110
#LANGUAGES = {
#    "en": "english",
#    "zh": "chinese",
#    "de": "german",
#    "es": "spanish",
#    "ru": "russian",
#    "ko": "korean",
#    "fr": "french",
#    "ja": "japanese",
#    "pt": "portuguese",
#    "tr": "turkish",
#    "pl": "polish",
#    "ca": "catalan",
#    "nl": "dutch",
#    "ar": "arabic",
#    "sv": "swedish",
#    "it": "italian",
#    "id": "indonesian",
#    "hi": "hindi",
#    "fi": "finnish",
#    "vi": "vietnamese",
#    "iw": "hebrew",
#    "uk": "ukrainian",
#    "el": "greek",
#    "ms": "malay",
#    "cs": "czech",
#    "ro": "romanian",
#    "da": "danish",
#    "hu": "hungarian",
#    "ta": "tamil",
#    "no": "norwegian",
#    "th": "thai",
#    "ur": "urdu",
#    "hr": "croatian",
#    "bg": "bulgarian",
#    "lt": "lithuanian",
#    "la": "latin",
#    "mi": "maori",
#    "ml": "malayalam",
#    "cy": "welsh",
#    "sk": "slovak",
#    "te": "telugu",
#    "fa": "persian",
#    "lv": "latvian",
#    "bn": "bengali",
#    "sr": "serbian",
#    "az": "azerbaijani",
#    "sl": "slovenian",
#    "kn": "kannada",
#    "et": "estonian",
#    "mk": "macedonian",
#    "br": "breton",
#    "eu": "basque",
#    "is": "icelandic",
#    "hy": "armenian",
#    "ne": "nepali",
#    "mn": "mongolian",
#    "bs": "bosnian",
#    "kk": "kazakh",
#    "sq": "albanian",
#    "sw": "swahili",
#    "gl": "galician",
#    "mr": "marathi",
#    "pa": "punjabi",
#    "si": "sinhala",
#    "km": "khmer",
#    "sn": "shona",
#    "yo": "yoruba",
#    "so": "somali",
#    "af": "afrikaans",
#    "oc": "occitan",
#    "ka": "georgian",
#    "be": "belarusian",
#    "tg": "tajik",
#    "sd": "sindhi",
#    "gu": "gujarati",
#    "am": "amharic",
#    "yi": "yiddish",
#    "lo": "lao",
#    "uz": "uzbek",
#    "fo": "faroese",
#    "ht": "haitian creole",
#    "ps": "pashto",
#    "tk": "turkmen",
#    "nn": "nynorsk",
#    "mt": "maltese",
#    "sa": "sanskrit",
#    "lb": "luxembourgish",
#    "my": "myanmar",
#    "bo": "tibetan",
#    "tl": "tagalog",
#    "mg": "malagasy",
#    "as": "assamese",
#    "tt": "tatar",
#    "haw": "hawaiian",
#    "ln": "lingala",
#    "ha": "hausa",
#    "ba": "bashkir",
#    "jw": "javanese",
#    "su": "sundanese",
#}

## ref: https://github.com/openai/whisper/blob/8cf36f3508c9acd341a45eb2364239a3d81458b9/whisper/tokenizer.py#L273-L292
#def build_tokenizer(path_to_whisper_repo: str, name: str = "gpt2"):
#    os.environ["TOKENIZERS_PARALLELISM"] = "false"
#    path = os.path.join(path_to_whisper_repo, "whisper/assets", name)
#    tokenizer = GPT2TokenizerFast.from_pretrained(path)
#
#    specials = [
#        "<|startoftranscript|>",
#        *[f"<|{lang}|>" for lang in LANGUAGES.keys()],
#        "<|translate|>",
#        "<|transcribe|>",
#        "<|startoflm|>",
#        "<|startofprev|>",
#        "<|nocaptions|>",
#        "<|notimestamps|>",
#    ]
#
#    tokenizer.add_special_tokens(dict(additional_special_tokens=specials))
#    return tokenizer

# ref: https://github.com/openai/gpt-2/blob/master/src/encoder.py
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
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
    print("Usage: convert-pt-to-ggml.py model.pt path-to-whisper-repo dir-output [use-f32]\n")
    sys.exit(1)

fname_inp   = Path(sys.argv[1])
dir_whisper = Path(sys.argv[2])
dir_out     = Path(sys.argv[3])

# try to load PyTorch binary data
try:
    model_bytes = open(fname_inp, "rb").read()
    with io.BytesIO(model_bytes) as fp:
        checkpoint = torch.load(fp, map_location="cpu")
except Exception:
    print("Error: failed to load PyTorch model file:" , fname_inp)
    sys.exit(1)

hparams = checkpoint["dims"]
print("hparams:", hparams)

list_vars = checkpoint["model_state_dict"]

#print(list_vars['encoder.positional_embedding'])
#print(list_vars['encoder.conv1.weight'])
#print(list_vars['encoder.conv1.weight'].shape)

# load mel filters
n_mels = hparams["n_mels"]
with np.load(dir_whisper / "whisper" / "assets" / "mel_filters.npz") as f:
    filters = torch.from_numpy(f[f"mel_{n_mels}"])
    #print (filters)

#code.interact(local=locals())

# load tokenizer
# for backwards compatibility, also check for older hf_transformers format tokenizer files
# old format: dir_whisper/whisper/assets/[multilingual/gpt2]/vocab.json
# new format: dir_whisper/whisper/assets/[multilingual/gpt2].tiktoken
multilingual = hparams["n_vocab"] >= 51865
tokenizer = dir_whisper / "whisper" / "assets" / (multilingual and "multilingual.tiktoken" or "gpt2.tiktoken")
tokenizer_type = "tiktoken"
if not tokenizer.is_file():
    tokenizer = dir_whisper / "whisper" / "assets" / (multilingual and "multilingual" or "gpt2") / "vocab.json"
    tokenizer_type = "hf_transformers"
    if not tokenizer.is_file():
        print("Error: failed to find either tiktoken or hf_transformers tokenizer file:", tokenizer)
        sys.exit(1)

byte_encoder = bytes_to_unicode()
byte_decoder = {v:k for k, v in byte_encoder.items()}

if tokenizer_type == "tiktoken":
    with open(tokenizer, "rb") as f:
        contents = f.read()
        tokens = {base64.b64decode(token): int(rank) for token, rank in (line.split() for line in contents.splitlines() if line)}
elif tokenizer_type == "hf_transformers":
    with open(tokenizer, "r", encoding="utf8") as f:
        _tokens_raw = json.load(f)
        if '<|endoftext|>' in _tokens_raw:
            # ensures exact same model as tokenizer_type == tiktoken
            # details: https://github.com/ggerganov/whisper.cpp/pull/725
            del _tokens_raw['<|endoftext|>']
        tokens = {bytes([byte_decoder[c] for c in token]): int(idx) for token, idx in _tokens_raw.items()}

# output in the same directory as the model
fname_out = dir_out / "ggml-model.bin"

# use 16-bit or 32-bit floats
use_f16 = True
if len(sys.argv) > 4:
    use_f16 = False
    fname_out = dir_out / "ggml-model-f32.bin"

fout = fname_out.open("wb")

fout.write(struct.pack("i", 0x67676d6c)) # magic: ggml in hex
fout.write(struct.pack("i", hparams["n_vocab"]))
fout.write(struct.pack("i", hparams["n_audio_ctx"]))
fout.write(struct.pack("i", hparams["n_audio_state"]))
fout.write(struct.pack("i", hparams["n_audio_head"]))
fout.write(struct.pack("i", hparams["n_audio_layer"]))
fout.write(struct.pack("i", hparams["n_text_ctx"]))
fout.write(struct.pack("i", hparams["n_text_state"]))
fout.write(struct.pack("i", hparams["n_text_head"]))
fout.write(struct.pack("i", hparams["n_text_layer"]))
fout.write(struct.pack("i", hparams["n_mels"]))
fout.write(struct.pack("i", use_f16))

# write mel filters
fout.write(struct.pack("i", filters.shape[0]))
fout.write(struct.pack("i", filters.shape[1]))
for i in range(filters.shape[0]):
    for j in range(filters.shape[1]):
        fout.write(struct.pack("f", filters[i][j]))

# write tokenizer
fout.write(struct.pack("i", len(tokens)))

for key in tokens:
    fout.write(struct.pack("i", len(key)))
    fout.write(key)

for name in list_vars.keys():
    data = list_vars[name].squeeze().numpy()
    print("Processing variable: " , name ,  " with shape: ", data.shape)

    # reshape conv bias from [n] to [n, 1]
    if name in ["encoder.conv1.bias", "encoder.conv2.bias"]:
        data = data.reshape(data.shape[0], 1)
        print(f"  Reshaped variable: {name} to shape: ", data.shape)

    n_dims = len(data.shape)

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

    #if name.startswith("encoder"):
    #    if name.endswith("mlp.0.weight") or \
    #       name.endswith("mlp.2.weight"):
    #        print("  Transposing")
    #        data = data.transpose()

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
