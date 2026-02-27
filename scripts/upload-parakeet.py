import os
from huggingface_hub import HfApi, create_repo

# TODO: change to ggml-org once merged.
USER_NAME = "danbev"
REPO_ID = f"{USER_NAME}/parakeet"
LOCAL_GGUF_PATH = "models/ggml-parakeet-tdt-0.6b-v3.bin"
REMOTE_GGUF_NAME = "parakeet-tdt-0.6b-v3.bin"

MODEL_CARD_CONTENT = f"""---
license: apache-2.0
base_model: {USER_NAME}/parakeet
tags:
- gguf
---

# Parakeet Model Card

## Description
This is an iterative release of the Parakeet model in whisper.cpp format.

## Usage
You can use this file with [parakeet-cli](https://github.com/danbev/whisper.cpp/tree/parakeet-support/examples/parakeet-cli).

Build parakeet-cli:
```console
$ git clone -b parakeet-support https://github.com/danbev/whisper.cpp.git
$ cd whisper.cpp
$ cmake -B build -S .
$ cmake --build build --target parakeet-cli -j 12
```

Download the model:
```console
$ hf download danbev/parakeet parakeet-tdt-0.6b-v3.bin --local-dir models
```

Run:
```console
$ ./build/bin/parakeet-cli -m models/parakeet-tdt-0.6b-v3.bin -f samples/jfk.wav
```

"""

api = HfApi()

def deploy_iteration():
    create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)

    print("Updating Model Card...")
    api.upload_file(
        path_or_fileobj=MODEL_CARD_CONTENT.encode(),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Update README.md"
    )

    print(f"Uploading {REMOTE_GGUF_NAME}...")
    api.upload_file(
        path_or_fileobj=LOCAL_GGUF_PATH,
        path_in_repo=REMOTE_GGUF_NAME,
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Upload new parakeet iteration"
    )

    print(f"\nDeployment successful!")
    print(f"URL: https://huggingface.co/{REPO_ID}")

if __name__ == "__main__":
    if os.path.exists(LOCAL_GGUF_PATH):
        deploy_iteration()
    else:
        print(f"Error: {LOCAL_GGUF_PATH} not found.")
