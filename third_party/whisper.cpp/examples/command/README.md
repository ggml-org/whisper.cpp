# whisper.cpp/examples/command

This is a basic Voice Assistant example that accepts voice commands from the microphone.
More info is available in [issue #171](https://github.com/ggerganov/whisper.cpp/issues/171).

```bash
# Run with default arguments and small model
./whisper-command -m ./models/ggml-small.en.bin -t 8

# On Raspberry Pi, use tiny or base models + "-ac 768" for better performance
./whisper-command -m ./models/ggml-tiny.en.bin -ac 768 -t 3 -c 0
```

https://user-images.githubusercontent.com/1991296/204038393-2f846eae-c255-4099-a76d-5735c25c49da.mp4

Web version: [examples/command.wasm](/examples/command.wasm)

## Guided mode

"Guided mode" allows you to specify a list of commands (i.e. strings) and the transcription will be guided to classify your command into one from the list. This can be useful in situations where a device is listening only for a small subset of commands.

Initial tests show that this approach might be extremely efficient in terms of performance, since it integrates very well with the "partial Encoder" idea from #137.

```bash
# Run in guided mode, the list of allowed commands is in commands.txt
./whisper-command -m ./models/ggml-base.en.bin -cmd ./examples/command/commands.txt

# On Raspberry Pi, in guided mode you can use "-ac 128" for extra performance
./whisper-command -m ./models/ggml-tiny.en.bin -cmd ./examples/command/commands.txt -ac 128 -t 3 -c 0
```

https://user-images.githubusercontent.com/1991296/207435352-8fc4ed3f-bde5-4555-9b8b-aeeb76bee969.mp4


## Building

The `whisper-command` tool depends on SDL2 library to capture audio from the microphone. You can build it like this:

```bash
# Install SDL2
# On Debian based linux distributions:
sudo apt-get install libsdl2-dev

# On Fedora Linux:
sudo dnf install SDL2 SDL2-devel

# Install SDL2 on Mac OS
brew install sdl2

cmake -B build -DWHISPER_SDL2=ON
cmake --build build --config Release
```
