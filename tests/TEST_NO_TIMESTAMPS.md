# Test: no_timestamps Flag Behavior

## Purpose

This test verifies that the `--no-timestamps` flag only affects output formatting and **does not change** the transcription quality or decoding process.

## Background

Previously, the `--no-timestamps` flag would:
1. Add a `<|notimestamps|>` token to the prompt
2. Suppress all timestamp tokens during decoding
3. Result in **different transcription text** compared to running without the flag

This was incorrect behavior because it degraded transcription quality.

## Fix

The fix ensures that:
1. ✅ Timestamp logic is **always** applied during decoding (for better quality)
2. ✅ The `--no-timestamps` flag **only** controls whether timestamps are shown in output
3. ✅ Transcription text is **identical** regardless of the flag

## Test Implementation

**File:** `tests/test-no-timestamps.cpp`

The test:
1. Loads a model and audio sample (JFK speech)
2. Runs transcription **with** timestamps enabled
3. Runs transcription **with** `no_timestamps` flag
4. Compares the normalized text from both runs
5. **Passes** if the texts are identical

## Running the Test

### Via CTest

```bash
# Run only this test
cd build
ctest -R test-no-timestamps -V

# Run with related tests
ctest -R "base.en|no-timestamps" --output-on-failure
```

### Direct Execution

```bash
# Build the test
cd build
make test-no-timestamps

# Run directly
./bin/test-no-timestamps
```

## Expected Output

```
Testing no_timestamps behavior
Model:  /path/to/models/ggml-base.en.bin
Sample: /path/to/samples/jfk.wav

Loaded audio: 11.00 seconds

Test 1: Transcribing with timestamps enabled...
Result:  And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.

Test 2: Transcribing with no_timestamps flag...
Result:  And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.

Comparison:
  With timestamps:    'and so my fellow americans, ask not what your country can do for you, ask what you can do for your country.'
  Without timestamps: 'and so my fellow americans, ask not what your country can do for you, ask what you can do for your country.'

✓ SUCCESS: Transcriptions are IDENTICAL
  The no_timestamps flag only affects output formatting,
  not the decoding process. Quality is preserved!
```

## Integration

The test is automatically included in the CTest suite with labels:
- `base` - uses base.en model
- `en` - English language test
- `unit` - unit test category

## Dependencies

- `whisper.h` - Core whisper API
- `common-whisper.h` - Audio loading utilities
- Model: `ggml-base.en.bin` (or any whisper model)
- Audio: `samples/jfk.wav` (or any test audio)

## Success Criteria

✅ Test passes if normalized transcription texts are identical  
❌ Test fails if texts differ, indicating a regression in the fix

