# Repository Guidelines

## Project Structure & Module Organization
Runtime and inference code lives under `src/` with headers in `include/`; tuned kernels belong to `ggml/`. CLIs and demos (for example `examples/main`, `examples/command`) stay in `examples/`. Tests and fixtures go in `tests/`, audio samples in `samples/`, language bindings in `bindings/`, and model helpers or notes in `models/`. Mirror the existing layout when adding files and update the closest README so new assets are discoverable.

## Build, Test, and Development Commands
From a clean checkout run `cmake -B build` followed by `cmake --build build -j --config Release` to generate `build/bin/whisper-cli`. Use `./build/bin/whisper-cli -f samples/jfk.wav` for a quick sanity check. Fetch the reference English model with `make base.en` (never commit the binary). Execute `ctest --test-dir build` for compiled suites and `tests/run-tests.sh base.en [threads]` for end-to-end coverage once `models/ggml-base.en.bin` is present.

## Coding Style & Naming Conventions
Use four-space indentation with braces on their own line. Group includes as standard headers first, then project headers, matching `src/whisper.cpp`. Prefer STL containers, keep helpers and internal functions in `snake_case`, and reserve `PascalCase` for user-facing structs, enums, and constants. Default to UTF-8 text and ASCII characters unless the file already mixes locales.

## Testing Guidelines
Register deterministic checks with CTest so they appear in `ctest` output. Store audio fixtures beneath `tests/` with expected transcripts named `*-ref.txt`. Run targeted audio regressions via `tests/run-tests.sh base.en <threads>` before sending a review, and expand the script when adding new scenarios.

## Commit & Pull Request Guidelines
Format commits as `<area> : short summary`, for example `whisper : tighten ja defaults`. Pull requests should describe motivation, note verification commands (build, `ctest`, audio runs), and link related issues or discussions. Call out new tooling, GPU toggles (`WHISPER_USE_*`), or download steps and keep evidence-focused logs.

## Security & Configuration Tips
Do not commit model binaries; rely on scripts in `models/` and document new assets in `models/README.md`. Gate optional GPU back ends behind the relevant `WHISPER_USE_*` CMake switches and confirm CPU fallbacks before landing changes.
