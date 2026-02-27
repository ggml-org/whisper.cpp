# Stable Timestamps - Build & Verify Notes

## Build

```bash
cd /Users/yqbqwlny/Documents/audio/stable-ts/whisper.cpp
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --target whisper -j8      # library only
cmake --build . --target whisper-cli -j8  # CLI
```

## Verify

Library and CLI both compile clean as of Phase 1 completion.

## Files Changed

- `include/whisper.h` — added `stable_timestamps` bool to `whisper_full_params`
- `src/whisper.cpp` — added `#include "whisper-stable.h"`, forward decl, param forcing, post-processing hook in `whisper_full()`
- `src/CMakeLists.txt` — added `whisper-stable.h` and `whisper-stable.cpp` to build

## Files Added

- `src/whisper-stable.h` — internal header
- `src/whisper-stable.cpp` — silence map builder + timestamp snapping

## Status

- Phase 1 (silence snapping): done, compiles clean
- CLI flag (`--stable-timestamps`): not yet wired up in `examples/cli/cli.cpp`
- Phases 2 & 3: not started
