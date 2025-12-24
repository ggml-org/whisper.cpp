import base64
import os
import shutil
import struct
import numpy as np

# ggml magic number
GGML_FILE_MAGIC = 0x67676d6c  # "ggml"


# Hyperparameter settings (configuration using tiny.en model)
class HyperParams:
    def __init__(self,
                 n_vocab=51865,
                 n_audio_ctx=1500,
                 n_audio_state=384,
                 n_audio_head=6,
                 n_audio_layer=4,
                 n_text_ctx=448,
                 n_text_state=384,
                 n_text_head=6,
                 n_text_layer=4,
                 n_mels=80):
        self.n_vocab = n_vocab
        self.n_audio_ctx = n_audio_ctx
        self.n_audio_state = n_audio_state
        self.n_audio_head = n_audio_head
        self.n_audio_layer = n_audio_layer
        self.n_text_ctx = n_text_ctx
        self.n_text_state = n_text_state
        self.n_text_head = n_text_head
        self.n_text_layer = n_text_layer
        self.n_mels = n_mels
        self.ftype = True   # True: fp16, False: fp32

def write_ggml_metadata(fout, hparams):
    # write magic number
    fout.write(struct.pack("i", GGML_FILE_MAGIC))
    
    # write hyperparameters
    fout.write(struct.pack("i", hparams.n_vocab))
    fout.write(struct.pack("i", hparams.n_audio_ctx))
    fout.write(struct.pack("i", hparams.n_audio_state))
    fout.write(struct.pack("i", hparams.n_audio_head))
    fout.write(struct.pack("i", hparams.n_audio_layer))
    fout.write(struct.pack("i", hparams.n_text_ctx))
    fout.write(struct.pack("i", hparams.n_text_state))
    fout.write(struct.pack("i", hparams.n_text_head))
    fout.write(struct.pack("i", hparams.n_text_layer))
    fout.write(struct.pack("i", hparams.n_mels))
    fout.write(struct.pack("i", hparams.ftype))

def write_mel_filters(fout, hparams, mel_filters_path):
    print("loading real Mel filter data...")
    # load the Mel filter from the npz file
    with np.load(mel_filters_path) as f:
        filters = f[f"mel_{hparams.n_mels}"]
    fout.write(struct.pack("i", filters.shape[0]))
    fout.write(struct.pack("i", filters.shape[1]))
    for i in range(filters.shape[0]):
        for j in range(filters.shape[1]):
            fout.write(struct.pack("f", filters[i][j]))

def write_tokenizer(fout, tokenizer_path):
    # read tokenizer file
    with open(tokenizer_path, "r") as f:
        tokens = {base64.b64decode(token): int(rank) for token, rank in (line.split() for line in f.readlines() if line)}
    # write size of tokenizer
    fout.write(struct.pack("i", len(tokens)))
    # write vocabulary
    for t in tokens:
        fout.write(struct.pack("i", len(t)))
        fout.write(t)

def generate_empty_model(filename, hparams):
    print(f"generate empty model file: {filename}")
    with open(filename, "wb") as f:
        write_ggml_metadata(f, hparams)
        write_mel_filters(f, hparams, "whisper/whisper/assets/mel_filters.npz")
        write_tokenizer(f, f"whisper/whisper/assets/{'gpt2' if hparams.n_vocab < 51865 else 'multilingual'}.tiktoken")
        # ignore the rest of the model

if __name__ == "__main__":
    os.system("git clone https://github.com/openai/whisper.git")
    
    # Base models
    generate_empty_model("for-tests-ggml-base.bin", HyperParams(
        n_vocab=51865, n_audio_state=512, n_audio_head=8, n_audio_layer=6,
        n_text_state=512, n_text_head=8, n_text_layer=6
    ))
    generate_empty_model("for-tests-ggml-base.en.bin", HyperParams(
        n_vocab=51864, n_audio_state=512, n_audio_head=8, n_audio_layer=6,
        n_text_state=512, n_text_head=8, n_text_layer=6
    ))
    
    # Small models
    generate_empty_model("for-tests-ggml-small.bin", HyperParams(
        n_vocab=51865, n_audio_state=768, n_audio_head=12, n_audio_layer=12,
        n_text_state=768, n_text_head=12, n_text_layer=12
    ))
    generate_empty_model("for-tests-ggml-small.en.bin", HyperParams(
        n_vocab=51864, n_audio_state=768, n_audio_head=12, n_audio_layer=12,
        n_text_state=768, n_text_head=12, n_text_layer=12
    ))
    
    # Medium models
    generate_empty_model("for-tests-ggml-medium.bin", HyperParams(
        n_vocab=51865, n_audio_state=1024, n_audio_head=16, n_audio_layer=24,
        n_text_state=1024, n_text_head=16, n_text_layer=24
    ))
    generate_empty_model("for-tests-ggml-medium.en.bin", HyperParams(
        n_vocab=51864, n_audio_state=1024, n_audio_head=16, n_audio_layer=24,
        n_text_state=1024, n_text_head=16, n_text_layer=24
    ))
    
    # Large models
    generate_empty_model("for-tests-ggml-large.bin", HyperParams(
        n_vocab=51865, n_audio_state=1280, n_audio_head=20, n_audio_layer=32,
        n_text_state=1280, n_text_head=20, n_text_layer=32
    ))
    # generate_empty_model("for-tests-ggml-large-v3.bin", HyperParams(    # add <|yue|>
    #     n_vocab=51866, n_audio_state=1280, n_audio_head=20, n_audio_layer=32,
    #     n_text_state=1280, n_text_head=20, n_text_layer=32
    # ))
        
    # Tiny models
    generate_empty_model("for-tests-ggml-tiny.bin", HyperParams(n_vocab=51865))
    generate_empty_model("for-tests-ggml-tiny.en.bin", HyperParams(n_vocab=51864))
    
    # Turbo model (based on large-v3 with optimizations)
    # generate_empty_model("for-tests-ggml-turbo.bin", HyperParams(    # add <|yue|>
    #     n_vocab=51866, n_audio_state=1280, n_audio_head=20, n_audio_layer=32,
    #     n_text_state=1280, n_text_head=20, n_text_layer=32
    # ))

    shutil.rmtree("whisper", ignore_errors=True)
