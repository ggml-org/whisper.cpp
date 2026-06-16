# whisper.cpp/examples/stream-resumable

Reference for the **resumable / asynchronous streaming** API: feed audio
incrementally and transcribe it next to recording, at full quality, without the
sliding-window re-decoding (and output divergence) of `examples/stream`.

Each 30s window is decoded exactly once; the seek position and rolling text
context are persisted in the `whisper_state` across calls, so already-emitted
segments are never revised. The output is consistent with a single batch run.

## API

```c
whisper_resumable_reset_with_state(ctx, state);

// as audio arrives (16 kHz mono f32):
whisper_append_audio_with_state(ctx, state, pcm, n);
int n_new = whisper_full_resumable_with_state(ctx, state, params, /*finalize=*/false);
// ... consume the n_new newly produced segments ...

// when the stream ends, flush the trailing (< 30s) window:
whisper_full_resumable_with_state(ctx, state, params, /*finalize=*/true);
```

`whisper_full_resumable_with_state` decodes every **complete** 30s window
currently available and returns the number of new segments. When less than 30s
of undecoded audio is available (and `finalize == false`) it returns 0 without
decoding a partial window ("need more audio").

## Threading

The model weights (`whisper_context`) are shared read-only; each concurrent
stream uses its own `whisper_state` (allocated with `whisper_init_state`). This
example runs one **producer** thread (audio source) and one **worker** thread
(inference) connected by a PCM queue, so transcription is decoupled from
capture. In your application, replace the file-reading producer with your
microphone / network source.

## Mel normalization

`whisper_full_params.mel_norm_mode` selects how the log-mel is normalized:

- `WHISPER_MEL_NORM_GLOBAL` (default): normalize against the maximum seen across
  all audio appended so far. This matches `whisper_full()` / batch behavior
  exactly only when the whole signal is appended before the first decode; when
  decoding incrementally, early windows use the running max rather than the
  whole-signal max (the difference is negligible for typical speech).
- `WHISPER_MEL_NORM_WINDOW`: normalize each window against a reference level
  with an envelope follower — instantaneous attack (so loud passages never
  over-drive) and an exponential release with a half-life in audio seconds
  (`mel_norm_half_life`), so a brief silence does not amplify background noise
  and a steady background source that stops is forgotten only gradually. Useful
  for live audio with varying levels.

## Usage

```bash
# build (no SDL required)
cmake -B build -DWHISPER_BUILD_EXAMPLES=ON
cmake --build build --target whisper-stream-resumable -j

# transcribe a 16 kHz mono WAV, simulating a live source in real time
./build/bin/whisper-stream-resumable \
    -m models/ggml-base.en.bin -f samples/jfk.wav --realtime

# live mel normalization with a 2s release half-life
./build/bin/whisper-stream-resumable \
    -m models/ggml-base.en.bin -f samples/jfk.wav --window-norm --half-life 2.0
```

Run with `--help` for all options.
