# Changelog

## [1.3.9] - Unreleased

Bundled whisper.cpp: `1.9.3`

- Fixed test fixture JFKReader's format string.
- Add #free method to `Whisper::Context`, `Whisper::VAD::Context` and `Whisper::Parakeet::Context`.
- Checked MemoryView format more strictly.

## [1.3.8] - 2026-07-28

Bundled whisper.cpp: `1.9.1`

- Added `Whisper::Context#full_n_vad_segments`,
  `#full_get_vad_segment_t0`, and `#full_get_vad_segment_t1` for reading the
  speech segments produced by VAD-enabled transcription.
- Changed standalone VAD processing to release Ruby's Global VM Lock.
- Fixed VAD parameter ownership and type handling, compatibility with older
  CMake option syntax, and model cache lookup behavior.
  ([#3931](https://github.com/ggml-org/whisper.cpp/pull/3931))

## [1.3.7] - 2026-06-17

Bundled whisper.cpp: `1.9.0`

- Added Ruby bindings for NVIDIA Parakeet, including models, contexts,
  parameters, segments, tokens, callbacks, and shared output helpers.
  ([#3885](https://github.com/ggml-org/whisper.cpp/pull/3885))
- Added Windows build support and support for additional Ruby MemoryView
  inputs, and changed transcription to release Ruby's Global VM Lock.
  ([#3775](https://github.com/ggml-org/whisper.cpp/pull/3775))
- Fixed dangling pointers, a memory leak, and a segmentation fault during
  parallel transcription.
  ([#3715](https://github.com/ggml-org/whisper.cpp/pull/3715))
- Raised the minimum supported Ruby version from 3.1 to 3.3.

## [1.3.6] - 2026-03-19

Bundled whisper.cpp: `1.8.4`

- Added the low-level `Whisper::VAD::Context#segments_from_samples` API and
  accepted `Pathname`-compatible model and audio paths.
  ([#3633](https://github.com/ggml-org/whisper.cpp/pull/3633))
- Added `Whisper::Context::Params` for configuring context initialization and
  fixed token memory ownership.
  ([#3647](https://github.com/ggml-org/whisper.cpp/pull/3647))
- Added a missing context null check.
  ([#3689](https://github.com/ggml-org/whisper.cpp/pull/3689))

## [1.3.5] - 2026-01-16

Bundled whisper.cpp: `1.8.3`

- Added standalone voice activity detection through
  `Whisper::VAD::Context`, `Whisper::VAD::Segments`, and
  `Whisper::VAD::Segment`.
  ([#3518](https://github.com/ggml-org/whisper.cpp/pull/3518))
- Updated the packaged Silero VAD model support to v6.2.0.
  ([#3524](https://github.com/ggml-org/whisper.cpp/pull/3524))
- Added `Whisper::Token` and corrected model URI handling.
  ([#3575](https://github.com/ggml-org/whisper.cpp/pull/3575))
- Fixed segmentation faults in token and segment access.
  ([#3591](https://github.com/ggml-org/whisper.cpp/pull/3591))

## [1.3.4] - 2025-10-08

Bundled whisper.cpp: `1.8.0`

- Added `Whisper::Params#max_len`.
  ([#3365](https://github.com/ggml-org/whisper.cpp/pull/3365))
- Fixed negative model-download padding values and made installation tests less
  sensitive to output formatting.
  ([#3389](https://github.com/ggml-org/whisper.cpp/pull/3389),
  [#3448](https://github.com/ggml-org/whisper.cpp/pull/3448))

## [1.3.3] - 2025-07-12

Bundled whisper.cpp: `1.7.6`

- Added Core ML build support.
  ([#3214](https://github.com/ggml-org/whisper.cpp/pull/3214))
- Added parallel transcription through `Whisper::Context#full_parallel`.
  ([#3222](https://github.com/ggml-org/whisper.cpp/pull/3222))
- Added higher-level segment iteration and text, SRT, VTT, CSV, and JSON output
  helpers.
  ([#3237](https://github.com/ggml-org/whisper.cpp/pull/3237))
- Added `Whisper::VERSION`.
  ([#3292](https://github.com/ggml-org/whisper.cpp/pull/3292))
- Improved install-time build option handling and Apple framework linkage.
  ([#3206](https://github.com/ggml-org/whisper.cpp/pull/3206),
  [#3270](https://github.com/ggml-org/whisper.cpp/pull/3270))

## [1.3.2] - 2025-05-28

Bundled whisper.cpp: `1.7.5`

- Added VAD-enabled transcription and `Whisper::VAD::Params`.
  ([#3197](https://github.com/ggml-org/whisper.cpp/pull/3197))
- Moved native extension builds to CMake and added install-time CMake build
  options.
  ([#3043](https://github.com/ggml-org/whisper.cpp/pull/3043),
  [#3056](https://github.com/ggml-org/whisper.cpp/pull/3056))
- Added encoder-begin callback APIs and refined model download caching.
  ([#3076](https://github.com/ggml-org/whisper.cpp/pull/3076),
  [#3109](https://github.com/ggml-org/whisper.cpp/pull/3109))
- Expanded context initialization and segment retrieval APIs.
  ([#2749](https://github.com/ggml-org/whisper.cpp/pull/2749))

## [1.3.1] - 2024-12-20

Bundled whisper.cpp: `1.7.3`

- Added Metal build support and new-segment callbacks.
  ([#2516](https://github.com/ggml-org/whisper.cpp/pull/2516),
  [#2506](https://github.com/ggml-org/whisper.cpp/pull/2506))
- Expanded context, model, parameter, and segment APIs, including low-level
  transcription methods.
  ([#2518](https://github.com/ggml-org/whisper.cpp/pull/2518),
  [#2551](https://github.com/ggml-org/whisper.cpp/pull/2551),
  [#2585](https://github.com/ggml-org/whisper.cpp/pull/2585))
- Added model download support and `no_speech_thold`.
  ([#2617](https://github.com/ggml-org/whisper.cpp/pull/2617),
  [#2641](https://github.com/ggml-org/whisper.cpp/pull/2641))

## [1.3.0] - 2024-05-16

Bundled whisper.cpp: `1.5.5`-era source snapshot

- Refreshed the packaged whisper.cpp and ggml sources for the newer backend
  architecture.
- Updated the native extension for upstream context initialization changes and
  fixed builds against the newer source layout.
- Expanded the Rake packaging and test tasks used to produce the gem.

## [1.2.0.2] - 2023-02-27

Bundled whisper.cpp: `1.2.0`-era source snapshot

- Added Rake build, clean, package, and test tasks.
- Changed long-running transcription to run without Ruby's Global VM Lock and
  added interruption handling.

## [1.2.0.1] - 2023-02-25

Bundled whisper.cpp: `1.2.0`-era source snapshot

- Fixed gem packaging so `LICENSE` and `README.md` were included at valid
  in-package paths.

## [1.2.0] - 2023-02-25

Bundled whisper.cpp: `1.2.0`-era source snapshot

- First published `whispercpp` Ruby gem, based on the initial Ruby binding.
  ([#500](https://github.com/ggml-org/whisper.cpp/pull/500))

[1.3.8]: https://rubygems.org/gems/whispercpp/versions/1.3.8
[1.3.7]: https://rubygems.org/gems/whispercpp/versions/1.3.7
[1.3.6]: https://rubygems.org/gems/whispercpp/versions/1.3.6
[1.3.5]: https://rubygems.org/gems/whispercpp/versions/1.3.5
[1.3.4]: https://rubygems.org/gems/whispercpp/versions/1.3.4
[1.3.3]: https://rubygems.org/gems/whispercpp/versions/1.3.3
[1.3.2]: https://rubygems.org/gems/whispercpp/versions/1.3.2
[1.3.1]: https://rubygems.org/gems/whispercpp/versions/1.3.1
[1.3.0]: https://rubygems.org/gems/whispercpp/versions/1.3.0
[1.2.0.2]: https://rubygems.org/gems/whispercpp/versions/1.2.0.2
[1.2.0.1]: https://rubygems.org/gems/whispercpp/versions/1.2.0.1
[1.2.0]: https://rubygems.org/gems/whispercpp/versions/1.2.0
