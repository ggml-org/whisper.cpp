# Voice-clone test-harness fixtures — v1 (QVAC-20979)

Golden fixtures for the voice-clone metrics self-tests
(`test/test_voiceclone_metrics.cpp`). Everything here is **small, committed, and
model-free** so the `unit` ctest tier reproduces known metric scores with no
downloads.

Regenerate with:

```bash
python3 scripts/dump-voiceclone-fixtures.py synthetic --out test/fixtures/voiceclone/v1
```

`synthetic` mode is pure-Python (no numpy required) and writes `.npy` v1.0
little-endian float32 arrays compatible with `src/npy.h`.

## Files

| File | Shape / type | Meaning |
|------|--------------|---------|
| `embed_a.npy` | `[8]` f32 | speaker embedding A (cosine input) |
| `embed_b.npy` | `[8]` f32 | speaker embedding B (cosine input) |
| `expected_cosine.npy` | `[1]` f32 | reference `cosine_similarity(A, B)` |
| `hidden_ref.npy` | `[5, 6]` f32 | WavLM-L3 hidden states, reference clip `[T, D]` |
| `hidden_target.npy` | `[4, 6]` f32 | WavLM-L3 hidden states, target clip `[T, D]` |
| `expected_style_loss.npy` | `[1]` f32 | reference `wavlm_style_loss(ref, target)` |
| `wer_ref.txt` | text | ground-truth transcript |
| `wer_hyp.txt` | text | hypothesis transcript |
| `expected_wer.npy` | `[1]` f32 | reference `word_error_rate(ref, hyp)` |

The `expected_*` values are computed by the **reference (Python) metric
implementations** in the dump script and re-checked against the **C++**
implementations in `src/voiceclone_metrics.h` on every test run, so this folder
is a cross-language regression lock, not just sample data. Inputs use values
exactly representable in float32 (integers and halves) so the only cross-language
difference is the float64 metric arithmetic, which the test tolerances cover.

## Versioning

`v1` is the schema/contents version. Bump to `v2/` (new folder) for any change to
shapes, semantics, or expected values so older harness builds keep resolving
their pinned fixtures.

## Not committed here (heavy / off-device)

The real speaker-similarity and gradient references the cloning tasks depend on
are produced off-device and live out-of-tree (point the build at them with
`-DTTS_CPP_TEST_REF_DIR`):

- CAMPPlus (on-device) + WavLM-base-plus-sv / ECAPA-TDNN / ResNet (off-device,
  SpeechBrain) target speaker embeddings,
- WavLM-Large layer-3 time-averaged `(mu, sigma)` statistics (the inverse-
  optimization objective),
- reference cloned voice vectors (`style_ttl` / `style_dp`),
- expected per-stage gradients (PyTorch autograd through the onnx2torch
  Supertonic pipeline) for `gradcheck`.

See the `reference` mode in `scripts/dump-voiceclone-fixtures.py` for the
contract; it is wired up by a later cloning task alongside the C++ analytic
gradients it validates.
