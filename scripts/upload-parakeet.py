import argparse
import os
from huggingface_hub import HfApi, create_repo

USER_NAME = "ggml-org"
REPO_ID   = f"{USER_NAME}/parakeet-GGUF"

MODELS = {
    "f32": {
        "local_path":   "models/ggml-parakeet-tdt-0.6b-v3-f32.bin",
        "remote_name":  "ggml-parakeet-tdt-0.6b-v3-f32.bin",
        "description":  "Full precision (F32)",
    },
    "f16": {
        "local_path":   "models/ggml-parakeet-tdt-0.6b-v3-f16.bin",
        "remote_name":  "ggml-parakeet-tdt-0.6b-v3-f16.bin",
        "description":  "Half precision (F16)",
    },
    "q8_0": {
        "local_path":   "models/ggml-parakeet-tdt-0.6b-v3-q8_0.bin",
        "remote_name":  "ggml-parakeet-tdt-0.6b-v3-q8_0.bin",
        "description":  "8-bit quantized (Q8_0)",
    },
    "q4_0": {
        "local_path":   "models/ggml-parakeet-tdt-0.6b-v3-q4_0.bin",
        "remote_name":  "ggml-parakeet-tdt-0.6b-v3-q4_0.bin",
        "description":  "4-bit quantized (Q4_0)",
    },
    "q4_k": {
        "local_path":   "models/ggml-parakeet-tdt-0.6b-v3-q4_k.bin",
        "remote_name":  "ggml-parakeet-tdt-0.6b-v3-q4_k.bin",
        "description":  "4-bit K-quantized (Q4_k)",
    },
}

def build_model_card(uploaded_variants):
    lines = [
        f"---",
        f"license: mit",
        f"base_model: nvidia/parakeet-tdt-0.6b-v3",
        f"tags:",
        f"- gguf",
        f"- asr",
        f"---",
        f"",
        f"# Parakeet TDT 0.6B v3 (GGUF)",
        f"",
        f"GGUF conversions of [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) for use with [whisper.cpp](https://github.com/ggml-org/whisper.cpp).",
        f"",
        f"## Available files",
        f"",
    ]

    for key, m in MODELS.items():
        if key in uploaded_variants:
            lines.append(f"- `{m['remote_name']}` — {m['description']}")

    lines += [
        f"",
        f"## Usage",
        f"",
        f"Build parakeet-cli:",
        f"```console",
        f"git clone https://github.com/ggml-org/whisper.cpp.git",
        f"cd whisper.cpp",
        f"cmake -B build -S .",
        f"cmake --build build --target parakeet-cli -j $(nproc)",
        f"```",
        f"",
        f"Download a model (e.g. Q8_0):",
        f"```console",
        f"hf download {REPO_ID} {MODELS['q8_0']['remote_name']} --local-dir models",
        f"```",
        f"",
        f"Run:",
        f"```console",
        f"./build/bin/parakeet-cli -m models/{MODELS['q8_0']['remote_name']} -f samples/jfk.wav",
        f"```",
        f"",
    ]

    return "\n".join(lines)


def upload_variant(api, key):
    m = MODELS[key]
    local_path = m["local_path"]

    if not os.path.exists(local_path):
        print(f"  Skipping {key}: {local_path} not found")
        return False

    print(f"  Uploading {m['remote_name']} ({m['description']})...")
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=m["remote_name"],
        repo_id=REPO_ID,
        repo_type="model",
        commit_message=f"Upload {m['remote_name']}",
    )
    return True


def main():
    parser = argparse.ArgumentParser(description="Upload parakeet GGUF models to Hugging Face")
    parser.add_argument(
        "variants",
        nargs="*",
        default=None,
        metavar="{" + ",".join(MODELS.keys()) + "}",
        help="Model variants to upload (default: all)",
    )
    parser.add_argument(
        "--no-model-card",
        action="store_true",
        help="Skip updating the model card README",
    )
    args = parser.parse_args()

    api = HfApi()
    create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)

    variants = args.variants if args.variants else list(MODELS.keys())

    unknown = [v for v in variants if v not in MODELS]
    if unknown:
        parser.error(f"unknown variant(s): {', '.join(unknown)} (choose from {', '.join(MODELS.keys())})")

    uploaded = []
    for key in variants:
        if upload_variant(api, key):
            uploaded.append(key)

    if not uploaded:
        print("No models were uploaded.")
        return

    if not args.no_model_card:
        print("Updating model card...")
        existing = [k for k in MODELS if k in uploaded or
                    any(f.rfilename == MODELS[k]["remote_name"]
                        for f in api.list_repo_files(REPO_ID, repo_type="model")
                        if hasattr(f, "rfilename"))]
        card = build_model_card(existing if existing else uploaded)
        api.upload_file(
            path_or_fileobj=card.encode(),
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type="model",
            commit_message="Update README.md",
        )

    print(f"\nDone. Repository: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
