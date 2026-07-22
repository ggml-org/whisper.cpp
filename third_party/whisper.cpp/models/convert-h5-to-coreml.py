import argparse
import importlib.util

spec = importlib.util.spec_from_file_location('whisper_to_coreml', 'models/convert-whisper-to-coreml.py')
whisper_to_coreml = importlib.util.module_from_spec(spec)
spec.loader.exec_module(whisper_to_coreml)

from whisper import load_model

from copy import deepcopy
import torch
from transformers import WhisperForConditionalGeneration
from huggingface_hub import metadata_update

# https://github.com/bayartsogt-ya/whisper-multiple-hf-datasets/blob/main/src/multiple_datasets/hub_default_utils.py
WHISPER_MAPPING = {
    "layers": "blocks",
    "fc1": "mlp.0",
    "fc2": "mlp.2",
    "final_layer_norm": "mlp_ln",
    "layers": "blocks",
    ".self_attn.q_proj": ".attn.query",
    ".self_attn.k_proj": ".attn.key",
    ".self_attn.v_proj": ".attn.value",
    ".self_attn_layer_norm": ".attn_ln",
    ".self_attn.out_proj": ".attn.out",
    ".encoder_attn.q_proj": ".cross_attn.query",
    ".encoder_attn.k_proj": ".cross_attn.key",
    ".encoder_attn.v_proj": ".cross_attn.value",
    ".encoder_attn_layer_norm": ".cross_attn_ln",
    ".encoder_attn.out_proj": ".cross_attn.out",
    "decoder.layer_norm.": "decoder.ln.",
    "encoder.layer_norm.": "encoder.ln_post.",
    "embed_tokens": "token_embedding",
    "encoder.embed_positions.weight": "encoder.positional_embedding",
    "decoder.embed_positions.weight": "decoder.positional_embedding",
    "layer_norm": "ln_post",
}

# https://github.com/bayartsogt-ya/whisper-multiple-hf-datasets/blob/main/src/multiple_datasets/hub_default_utils.py
def rename_keys(s_dict):
    keys = list(s_dict.keys())
    for key in keys:
        new_key = key
        for k, v in WHISPER_MAPPING.items():
            if k in key:
                new_key = new_key.replace(k, v)

        print(f"{key} -> {new_key}")

        s_dict[new_key] = s_dict.pop(key)
    return s_dict

# https://github.com/bayartsogt-ya/whisper-multiple-hf-datasets/blob/main/src/multiple_datasets/hub_default_utils.py
def convert_hf_whisper(hf_model_name_or_path: str, whisper_state_path: str):
    transformer_model = WhisperForConditionalGeneration.from_pretrained(hf_model_name_or_path)
    config = transformer_model.config

    # first build dims
    dims = {
        'n_mels': config.num_mel_bins,
        'n_vocab': config.vocab_size,
        'n_audio_ctx': config.max_source_positions,
        'n_audio_state': config.d_model,
        'n_audio_head': config.encoder_attention_heads,
        'n_audio_layer': config.encoder_layers,
        'n_text_ctx': config.max_target_positions,
        'n_text_state': config.d_model,
        'n_text_head': config.decoder_attention_heads,
        'n_text_layer': config.decoder_layers
    }

    state_dict = deepcopy(transformer_model.model.state_dict())
    state_dict = rename_keys(state_dict)

    torch.save({"dims": dims, "model_state_dict": state_dict}, whisper_state_path)

# Ported from models/convert-whisper-to-coreml.py
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, help="name of model to convert (e.g. tiny, tiny.en, base, base.en, small, small.en, medium, medium.en, large-v1, large-v2, large-v3, large-v3-turbo)", required=True)
    parser.add_argument("--model-path", type=str, help="path to the model (e.g. if published on HuggingFace: Oblivion208/whisper-tiny-cantonese)", required=True)
    parser.add_argument("--encoder-only", type=bool, help="only convert encoder", default=False)
    parser.add_argument("--quantize",     type=bool, help="quantize weights to F16", default=False)
    parser.add_argument("--optimize-ane", type=bool, help="optimize for ANE execution (currently broken)", default=False)
    args = parser.parse_args()

    if args.model_name not in ["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "large-v1", "large-v2", "large-v3", "large-v3-turbo"]:
        raise ValueError("Invalid model name")

    pt_target_path = f"models/hf-{args.model_name}.pt"
    convert_hf_whisper(args.model_path, pt_target_path)

    whisper = load_model(pt_target_path).cpu()
    hparams = whisper.dims
    print(hparams)

    if args.optimize_ane:
        whisperANE = whisper_to_coreml.WhisperANE(hparams).eval()
        whisperANE.load_state_dict(whisper.state_dict())

        encoder = whisperANE.encoder
        decoder = whisperANE.decoder
    else:
        encoder = whisper.encoder
        decoder = whisper.decoder

    # Convert encoder
    encoder = whisper_to_coreml.convert_encoder(hparams, encoder, quantize=args.quantize)
    encoder.save(f"models/coreml-encoder-{args.model_name}.mlpackage")

    if args.encoder_only is False:
        # Convert decoder
        decoder = whisper_to_coreml.convert_decoder(hparams, decoder, quantize=args.quantize)
        decoder.save(f"models/coreml-decoder-{args.model_name}.mlpackage")

    print("done converting")
