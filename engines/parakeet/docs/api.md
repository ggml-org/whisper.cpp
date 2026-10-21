# parakeet engine: C++ API

Part of the [parakeet engine documentation](../README.md).

## Public C++ API

Include `<parakeet/parakeet.h>` for the complete surface or individual headers.

| API | Purpose |
|---|---|
| `Engine::transcribe` | One-shot WAV transcription for CTC, RNN-T, TDT, EOU, or Nemotron |
| `Engine::transcribe_samples` | One-shot float PCM transcription |
| `Engine::compute_mel` / `compute_mel_wav` / `transcribe_mel` | The mel stage ahead of time as a `PrecomputedMel`, then everything from the encoder on; `compute_mel*` is the one member safe to call on a worker thread during a transcription, so a batch overlaps the next file's audio front-end with the current one's encoder |
| `Engine::transcribe_stream` / `transcribe_samples_stream` | Chunked callbacks; native cache-aware streaming for Unified RNN-T and Nemotron 3.5 ASR, one offline encode for other ASR families |
| `Engine::stream_start` | Live session; native cache-aware streaming for Unified RNN-T and Nemotron 3.5 ASR, rolling-window re-encoding for other ASR families |
| `StreamSession::feed_pcm_f32` / `feed_pcm_i16` | Push arbitrary-sized PCM blocks |
| `Engine::diarize` / `diarize_samples` | Offline Sortformer or Nemotron 3 Diarization |
| `Engine::diarize_start` | Live Sortformer or Nemotron 3 Diarization session |
| `transcribe_with_speakers` / `transcribe_samples_with_speakers` | CTC, RNN-T, TDT, or EOU transcription attributed with Sortformer or Nemotron 3 Diarization |
| `EngineOptions::prewarm` / `prewarm_audio_seconds` | Run a configurable encoder-only synthetic forward during construction; Nemotron 3 Diarization runs one offline pass and one chunk at the default live geometry |
| `Engine::encoder_on_coreml` / `encoder_backend` | Whether an Apple Core ML encoder or diarizer sidecar loaded (`encoder_backend()` returns the Core ML label, otherwise `backend_name()`); see [Core ML encoder sidecar](backends.md#core-ml-encoder-sidecar) and [diarizer sidecars](backends.md#core-ml-diarizer-sidecars) |
| `EngineResult::encoder_used_coreml` | Whether every encoder invocation of this transcription ran on the Core ML sidecar |
| `EngineResult::segment_plan_ms` / `assemble_ms` | Non-zero only on the Core ML independent-segment path (TDT decoder sidecar): pause-cut planning, and per-segment concatenation plus stitching; with `preprocess_ms`, `encoder_ms` and `decode_ms` they account for `total_ms` |
| `parakeet_log_set` | Install the host logging sink |
| `parakeet_cli_main` | Embed the same CLI entry point exported by the library |
| `moss::TranscribeEngine::transcribe` | MOSS transcription with speakers, timestamps and hotwords; separate API in `<parakeet/moss_transcribe.h>`, see [MOSS guide](moss-transcribe.md); `encoder_on_coreml()` and `TranscribeResult::encoder_backend` report the [Core ML encoder sidecar](moss-transcribe.md#core-ml-encoder-sidecar) |
| `fit_params` | Memory-fit preflight: project whether a GGUF + workload fits the device without loading weights (`<parakeet/fit.h>`) |
| `moss::fit_params` | MOSS-Transcribe metadata-only projection for an explicit audio duration and transcription request (`<parakeet/moss_transcribe_fit.h>`) |
| `parakeet_fit_cli_main` | Embed the `parakeet-fit-params` tool entry point |

Minimal one-shot transcription:

```cpp
#include <parakeet/parakeet.h>

#include <cstdio>

int main() {
    parakeet::EngineOptions options;
    options.model_gguf_path = "models/parakeet-ctc-0.6b.q8_0.gguf";

    parakeet::Engine engine(options);
    const auto result = engine.transcribe("audio.wav");
    std::puts(result.text.c_str());
}
```

Mode 3 uses rolling-context/sliding-window re-encoding for CTC, non-Unified
RNN-T, TDT and EOU. It re-runs the encoder with `left_context_ms` and
`right_lookahead_ms`, without encoder attention/convolution caches. Unified
RNN-T and Nemotron 3.5 ASR use native cache-aware encoders in callback and live
streaming. Unified selects trained chunk/right-context values and ignores
`left_context_ms`; Nemotron selects its operating point from `chunk_ms` and
ignores both sliding-window context knobs.

Always call `finalize()` to drain the partial tail. Destroying a
`StreamSession` or `SortformerStreamSession` cancels it and does not finalize
it. `cancel()` stops future work; `Engine::cancel()` may be called from another
thread, but inference calls on one Engine are otherwise not concurrent-safe.
Cancel, join the worker, and only then destroy an Engine; destruction does not
wait for in-flight calls.

`StreamingCallback` and `StreamEventCallback` run synchronously from the API
call processing the audio. `StreamingSegment::is_eou_boundary` and
`StreamEventType::EndOfTurn` mean the EOU model emitted `<EOU>`. They are not
VAD state changes. Sortformer emits `VadStateChanged` from speaker
probabilities. CTC/RNN-T/TDT/Nemotron can opt into RMS energy VAD with
`enable_energy_vad`; configure it with `energy_vad_threshold_db`,
`energy_vad_window_ms`, and `energy_vad_hangover_ms`.

Long-form one-shot transcription uses bounded encoder windows when
`long_form_window_frames` is exceeded and trims
`long_form_context_frames` at seams. Short inputs retain the single-pass path.
A windowed pass holds one encoder graph at a time and releases the audio and
the STFT scratch once the mel exists, so peak memory is bounded by the window.
`long_form_window_frames = 750` with `long_form_context_frames = 94` is the
60 s / 25 % setting validated for TranscribePsy on phones, and the engine
takes it by itself when both options are left at 0: a GGUF recommends its
window through `parakeet.long_form.window_frames` and
`parakeet.long_form.context_frames` (the converter writes them from
`--long-form-window-sec` and `--long-form-overlap`), and the published
TranscribePsy GGUF, which predates the keys, is recognised by its
description. Any other model keeps the automatic window: the fewest windows
of at most 300 s, balanced to equal length. With a Core ML TDT
decoder sidecar the engine instead cuts the recording at pauses into
independent segments and decodes them batched on the Neural Engine; see
[Neural Engine sidecars](backends.md#neural-engine-fixed-window-encoder-and-tdt-decoder-sidecars).

Transcribing a batch of recordings overlaps the CPU mel stage with the encoder:

```cpp
std::future<parakeet::PrecomputedMel> next = std::async(std::launch::async,
    [&] { return engine.compute_mel_wav(files[0]); });
for (size_t i = 0; i < files.size(); ++i) {
    parakeet::PrecomputedMel mel = next.get();
    if (i + 1 < files.size()) {
        next = std::async(std::launch::async,
            [&, i] { return engine.compute_mel_wav(files[i + 1]); });
    }
    const parakeet::EngineResult r = engine.transcribe_mel(std::move(mel));
}
```

`transcribe_mel(compute_mel(...))` returns exactly what `transcribe_samples`
returns on the same audio; `compute_mel*` is the only member that may run
concurrently with a `transcribe*` call on the same engine. `PrecomputedMel` is
a plain struct, so `transcribe_mel` validates it before the encoder reads it
and throws `std::runtime_error` when the buffer does not hold exactly
`n_frames * n_mels` values, when `n_frames` is zero, or when `n_mels` is not
the model's bin count.

## Sortformer AOSC

AOSC is primarily activated by the explicit GGUF metadata
`parakeet.model_variant=sortformer-streaming-v2.1-aosc`. Encoder shape
`n_layers=17` and `n_mels=128` is only a legacy fallback for GGUFs created
before that metadata key existed. Convert current v2.1 checkpoints again to
avoid relying on the heuristic.

`SortformerStreamingOptions` exposes `spkcache_enable`, `spkcache_len`,
`fifo_len`, `chunk_left_context_ms`, `chunk_right_context_ms`, and
`spkcache_update_period`. After `diarize_start()`,
`SortformerStreamSession::aosc_active()` reports whether the session actually
took the AOSC path.

Every non-cancelled `finalize()` drains any trailing partial chunk and then
emits exactly one final synthetic terminator (`speaker_id=-1`,
`is_final=true`, and `start_s == end_s`). Real speaker segments always remain
non-final. Repeated `finalize()` calls are idempotent, while cancellation
suppresses the terminator.

## Unified RNN-T streaming

Unified RNN-T uses standard greedy transducer decoding. Offline inference runs
the encoder in full-context mode, and with `PARAKEET_COREML=ON` eligible batch
windows route through the fixed-capacity Core ML sidecar. Mode 2 and
`StreamSession` use the native cache-aware encoder, which builds its own ggml
graph and never uses the fixed-shape sidecar: `StreamingOptions::chunk_ms`
selects the chunk and `right_lookahead_ms` the right context, both snapped down
to the nearest trained value (chunks 80, 160, 560, 1040 ms; right context 0, 80,
160, 240, 320, 560, 1040 ms; the default 1000 ms chunk runs as 560 ms); the
encoder keeps a 5.6 s attention cache plus a 4-frame convolution cache per layer
and only encodes `chunk + right context` new frames per step. The
`left_context_ms` knob is ignored. GGUFs converted before the
`parakeet.unified.*` metadata existed fall back to the published checkpoint
contexts.


## Nemotron 3.5 ASR streaming

Nemotron 3.5 ASR offline inference uses the GGUF's default 320 ms operating point
(`att_context_size=[56,3]`). `EngineOptions::language` accepts the locale aliases
stored in the GGUF, and an empty value resolves to `auto`. The selected locale is
broadcast as a 128-wide one-hot prompt, concatenated to every encoder frame, and
projected before RNN-T decoding. The cache-aware streaming path incrementally
converts arbitrary PCM bursts to mel frames, maintains bounded 56-frame
attention and 8-frame convolution caches, and matches NeMo at all five supported
operating points. Set `StreamingOptions::chunk_ms` to `80`, `160`, `320`, `560`,
or `1120` to select the corresponding trained right-context configuration.
Both callback streaming and live `StreamSession` input use the native caches;
the sliding-window `left_context_ms` and `right_lookahead_ms` knobs are ignored
for Nemotron.
