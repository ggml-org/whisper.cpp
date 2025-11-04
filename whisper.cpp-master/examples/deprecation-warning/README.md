# Migration notice for binary filenames

> [!IMPORTANT]
[2024 Dec 20] Binaries have been renamed w/ a `whisper-` prefix. `main` is now `whisper-cli`, `server` is `whisper-server`, etc (https://github.com/ggerganov/whisper.cpp/pull/2648)

This migration was important, but it is a breaking change that may not always be immediately obvious to users.

Please update all scripts and workflows to use the new binary names.

| Old Filename | New Filename |
| ---- | ---- |
| main | whisper-cli |
| bench | whisper-bench |
| stream | whisper-stream |
| command | whisper-command |
| server | whisper-server |
| talk-llama | whisper-talk-llama |
