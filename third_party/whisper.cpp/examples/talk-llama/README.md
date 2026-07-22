# whisper.cpp/examples/talk-llama

Talk with an LLaMA AI in your terminal

*Latest perf as of 2 Nov 2023 using Whisper Medium + LLaMA v2 13B Q8_0 on M2 Ultra:*

https://github.com/ggerganov/whisper.cpp/assets/1991296/d97a3788-bf2a-4756-9a43-60c6b391649e

*Previous demo running on CPUs*

[Demo Talk](https://user-images.githubusercontent.com/1991296/228024237-848f998c-c334-46a6-bef8-3271590da83b.mp4)

## Building

The `whisper-talk-llama` tool depends on SDL2 library to capture audio from the microphone. You can build it like this:

```bash
# Install SDL2
# On Debian based linux distributions:
sudo apt-get install libsdl2-dev

# On Fedora Linux:
sudo dnf install SDL2 SDL2-devel

# Install SDL2 on Mac OS
brew install sdl2

# Build the "whisper-talk-llama" executable
cmake -B build -S . -DWHISPER_SDL2=ON
cmake --build build --config Release

# Run it
./build/bin/whisper-talk-llama -mw ./models/ggml-small.en.bin -ml ../llama.cpp/models/llama-13b/ggml-model-q4_0.gguf -p "Georgi" -t 8
```

- The `-mw` argument specifies the Whisper model that you would like to use. Recommended `base` or `small` for real-time experience
- The `-ml` argument specifies the LLaMA model that you would like to use. Read the instructions in https://github.com/ggerganov/llama.cpp for information about how to obtain a `ggml` compatible LLaMA model

## Session

The `whisper-talk-llama` tool supports session management to enable more coherent and continuous conversations. By maintaining context from previous interactions, it can better understand and respond to user requests in a more natural way.

To enable session support, use the `--session FILE` command line option when running the program. The `whisper-talk-llama` model state will be saved to the specified file after each interaction. If the file does not exist, it will be created. If the file exists, the model state will be loaded from it, allowing you to resume a previous session.

This feature is especially helpful for maintaining context in long conversations or when interacting with the AI assistant across multiple sessions. It ensures that the assistant remembers the previous interactions and can provide more relevant and contextual responses.

Example usage:

```bash
./build/bin/whisper-talk-llama --session ./my-session-file -mw ./models/ggml-small.en.bin -ml ../llama.cpp/models/llama-13b/ggml-model-q4_0.gguf -p "Georgi" -t 8
```

## TTS

For best experience, this example needs a TTS tool to convert the generated text responses to voice.
You can use any TTS engine that you would like - simply edit the [speak](speak) script to your needs.
By default, it is configured to use MacOS's `say` or Windows SpeechSynthesizer, but you can use whatever you wish.

## Discussion

If you have any feedback, please let "us" know in the following discussion: https://github.com/ggerganov/whisper.cpp/discussions/672?converting=1
