# audiogen engine: command-line tools

Part of the [audiogen engine documentation](../README.md).

## Command line tools

| Binary | Purpose |
|---|---|
| `music-cli` | end-to-end text-to-music |
| `mm3-replay` | MiniMax full text-to-music generation and controlled replay |
| `acestep-cli` | VAE decode and reconstruction roundtrip harness |
| `textenc-smoke`, `lm-smoke`, `lmgen-smoke`, `bpe-smoke`, `detok-smoke`, `cond-smoke`, `dit-smoke` | per-stage smoke harnesses |

### MiniMax Core ML conditioner

With an Apple `AUDIOGEN_COREML=ON` build and the
[exported conditioner](backends.md#minimax-core-ml-conditioner):

```bash
./build/audiogen/mm3-replay --models models/minimax --mode full \
  --device gpu --threads 8 --seed 7 --steps 8 --max-frames 150 \
  --coreml-cond models/minimax/conditioner.mlmodelc \
  --caption "Warm piano, gentle acoustic guitar and soft drums." \
  --lyrics "[Instrumental]" --out output/minimax-coreml
```

The output directory contains `audio.wav` and replay tensors. `--max-frames`
is the existing request length in 25 Hz AR frames; 150 requests about six
seconds. Omitting `--coreml-cond` keeps the native conditioner.

### music-cli

```sh
# all four GGUFs in one directory
./build/audiogen/music-cli --models models/acestep --out song.wav --dur 8 --seed 42

# explicit per-stage paths, GPU, custom prompt
./build/audiogen/music-cli --dit dit.gguf --lm lm.gguf --text emb.gguf --vae vae.gguf \
                  --caption "driving synth pop, bright analog leads" \
                  --lyrics "[Instrumental]" --bpm 128 --key "C major" --tsig 4/4 \
                  --lang en --steps 8 --shift 3.0 --gpu --out song.wav

# Simple Mode: one short query becomes a complete song (caption, lyrics and
# metadata composed by the LM; --dur 0 lets the LM pick the duration too)
./build/audiogen/music-cli --models models/acestep --simple \
                  --caption "a romantic modern salsa with male lead vocals for a wedding" \
                  --dur 0 --gpu --out salsa.wav

# condition generation with a 48 kHz PCM16 WAV timbre reference
./build/audiogen/music-cli --models models/acestep --ref-audio reference-48k.wav \
                  --caption "warm Latin pop with a male lead vocal" \
                  --lyrics "[Verse]\nA brand new lyric" --gpu --out referenced.wav

# repaint 10s..20s while preserving the rest of the source
./build/audiogen/music-cli --models models/acestep --src-audio source-48k.wav \
                  --caption "expressive analog synthesizer solo" \
                  --lyrics "[Instrumental]" --repaint-start 10 --repaint-end 20 \
                  --repaint-mode balanced --repaint-strength 0.5 \
                  --seed 22883 --gpu --out repainted.wav

# FlowEdit uses --caption/--lyrics as the target
./build/audiogen/music-cli --models models/acestep --src-audio source-48k.wav \
                  --flow-source-caption "bright late-1990s pop" \
                  --flow-source-lyrics "[Instrumental]" \
                  --caption "dark analog synthwave" --lyrics "[Instrumental]" \
                  --flow-n-min 0 --flow-n-max 1 --flow-n-avg 1 \
                  --seed 22883 --gpu --out flow-edited.wav

# ordered, repeated, composable operations
./build/audiogen/music-cli --models models/acestep --src-audio source-48k.wav \
                  --edit-plan edit-plan.json --seed 22883 --gpu --out edited.wav
```

| Flag | Default | Meaning |
|---|---|---|
| `--models DIR` | | directory holding the four stage GGUFs |
| `--dit`, `--lm`, `--text`, `--vae` | | explicit per-stage GGUF paths |
| `--caption TEXT` | built-in pop-rock prompt | text prompt |
| `--lyrics TEXT` | `[Instrumental]` | lyrics |
| `--dur SECONDS` | `8` | target length, drives the LM code count |
| `--seed N` | `42` | negative means random |
| `--bpm N`, `--key STR`, `--tsig STR`, `--lang CODE` | inferred | optional metadata hints |
| `--steps N`, `--shift F` | per variant | sampler overrides |
| `--no-dcw` | DCW enabled | disable the official Haar low/high correction applied after each DiT step |
| `--no-loudness` | normalization on | skip the percentile loudness normalization (99.999th percentile to 1.0, tail hard-clipped) applied to generation output; edits and lego stems are never normalized |
| `--lrc out.lrc` | off | write synchronized lyric timestamps (LRC) next to the WAV; requires lyrics |
| `--temp F`, `--topp F`, `--topk N`, `--cfg F` | `0.85`, `0.9`, off, `2.0` | LM sampling for the audio codes |
| `--no-phase1` | off | skip the LM metadata auto-fill pass |
| `--simple` | off | Simple Mode: expand `--caption` into a full request (lyrics regenerate unless `--lyrics "[Instrumental]"` is passed) |
| `--rewrite` | off | Query Rewriting: the LM formats `--caption` and `--lyrics` into a detailed request, preserving the lyric content (needs the 1.7B LM) |
| `--score` | off | teacher-forced LM quality score of the generated codes, printed with its per-condition breakdown |
| `--understand FILE` | | reverse pipeline: describe a 48 kHz PCM16 WAV (metadata + caption + recovered codes) instead of generating |
| `--req FILE` | | request JSON; pre-supplied `audio_codes` skip the LM stage |
| `--ref-audio FILE` | | 48 kHz PCM16 WAV used by ACE-Step's timbre-conditioning path |
| `--src-audio FILE` | | 48 kHz PCM16 source WAV; required for editing |
| `--repaint-start SEC`, `--repaint-end SEC` | `0`, source end | standalone repaint range |
| `--repaint-mode MODE` | `balanced` | `conservative`, `balanced`, or `aggressive` |
| `--repaint-strength F` | `0.5` | balanced-mode strength in `[0, 1]` |
| `--flow-source-caption TEXT` | | enables standalone FlowEdit and describes the current source |
| `--flow-source-lyrics TEXT` | `[Instrumental]` | current source lyrics |
| `--flow-n-min F`, `--flow-n-max F` | `0`, `1` | active FlowEdit diffusion window |
| `--flow-n-avg N` | `1` | forward-noise samples averaged per active step |
| `--edit-plan FILE` | | ordered JSON edit plan; cannot be combined with standalone edit flags |
| `--normalize` | off for edit plans | peak-normalize edited CLI output; avoid for partial repaint when outside-region PCM must remain exact |
| `--gpu`, `--threads N` | CPU, hardware concurrency | compute placement |
| `--backend NAME` | `auto` | `cpu`, `opencl`, `hexagon` (the HTP0 NPU) or an exact ggml device name; overrides `--gpu` and fails instead of falling back. Hexagon is only used when requested |
| `--lm-backend NAME` | `auto` | put the LM on this device instead of the placement policy's choice, e.g. `--backend hexagon --lm-backend opencl` |
| `--backends-dir DIR` | | directory containing staged dynamic ggml backend modules; required by Android and Linux arm64 dynamic-backend builds |
| `--dump-stages DIR` | | write one `.bin` per stage into an existing directory |
| `--out PATH` | `music_out.wav` | output WAV path |

`--req` accepts a flat JSON object. Request values override overlapping CLI
values after CLI parsing:

| Field | Accepted value |
|---|---|
| `caption`, `lyrics`, `keyscale`, `timesignature`, `vocal_language` | string |
| `bpm`, `inference_steps`, `seed` | number |
| `duration`, `shift`, `dcw_scaler`, `dcw_high_scaler` | number |
| `dcw_enabled` | boolean, or `0` / `1` |
| `audio_codes` | quoted comma-separated string, for example `"12,34,56"` |

`audio_codes` is currently not parsed as a JSON numeric array. A non-empty
value bypasses the LM and feeds the codes directly to the FSQ detokenizer.

`--edit-plan` accepts an object with an `operations` array. Operations execute
in array order and may repeat:

```json
{
  "operations": [
    {
      "type": "flow-edit",
      "source_caption": "bright late-1990s pop",
      "source_lyrics": "[Instrumental]",
      "target_caption": "dark analog synthwave",
      "target_lyrics": "[Instrumental]",
      "n_min": 0,
      "n_max": 1,
      "n_avg": 1
    },
    {
      "type": "repaint",
      "start": 10,
      "end": 20,
      "mode": "balanced",
      "strength": 0.5,
      "caption": "expressive analog synthesizer solo",
      "lyrics": "[Instrumental]"
    }
  ]
}
```

The sampler enables ACE-Step's single-level Haar DCW `double` mode by default.
At timestep `t`, the low band uses `t * 0.05` and the high band uses
`(1 - t) * 0.02`, matching the official Python defaults. The correction runs
on the host latent between DiT steps, so it is backend-independent and adds
negligible work compared with a transformer forward.

Reference WAV input is decoded at the `music-cli` boundary and must be PCM16
stereo or mono at 48 kHz. The engine receives normalized interleaved stereo PCM
and encodes it in overlapping VAE windows for bounded memory on long inputs.

### acestep-cli

```sh
./build/audiogen/acestep-cli --model vae.gguf --t-latent 32 --out out.wav
./build/audiogen/acestep-cli --model vae.gguf --roundtrip --in in.wav --seconds 2.56 --out out.wav
./build/audiogen/acestep-cli --model vae.gguf --latent-bin dump/08_dit_latent.bin --backend hexagon --dump pcm.bin
```

The first form is the default mode: it decodes a synthetic latent, which checks that real weights load and that the decode graph (`ggml_col2im_1d` + `ggml_snake`) runs on the selected backend. `--roundtrip` encodes a real WAV and prints the per-channel reconstruction correlation, the audible end-to-end VAE check. `--latent-bin` decodes a `music-cli --dump-stages` DiT latent instead and `--dump` writes the PCM in the same format, so two backends can be compared on an identical input; the decode time is printed. Every form takes `--gpu` or `--backend NAME`. `detok-smoke`, `cond-smoke` and `lm-smoke` take `--backend NAME` too.

### acestep-fit-params

```sh
./build/audiogen/acestep-fit-params --models-dir /path/to/acestep --duration 60 --n-gpu-layers 99
```

Memory-fit preflight: projects whether the four stage GGUFs plus a generation
workload fit the memory available right now, reading only GGUF metadata (no
weights load, nothing runs). It resolves the same backends and stage placement
`Engine::create` would, drives each stage loader in metadata-only mode, and
prices the real compute graphs at the workload's shapes through ggml's
size-only allocator APIs, so the projection tracks the runtime by construction.
The default projection follows the per-stage low-memory residency above (the
peak phase per memory pool); `--keep-stages 1` (or `ACESTEP_KEEP_STAGES`)
projects the everything-resident mode. Exit code follows the `@qvac/model-fit`
contract: 0 = fits, 1 = does not fit (a valid answer), 2 = error. `--json`
emits the projection for the SDK; the library entry point is
`tts_cpp::acestep::fit_params` (`include/audiogen-cpp/acestep/fit.h`), and
hosts can link the CLI as `acestep_fit_cli_main`. Workload knobs: `--duration`,
`--text-tokens`, `--lyric-tokens`, `--lm-cfg`, `--guidance`,
`--with-source-audio`, `--margin-mib`. Placement knobs: `--n-gpu-layers`, `--backend` and
`--lm-backend`, with the same meaning as in `music-cli`. Every device a stage
lands on is its own memory pool: a third device picked by `--lm-backend` is
listed under `extraDevices` in the JSON and must hold its stages plus the
margin, while devices that share host memory (unified memory, or the Hexagon
NPU, which reports none of its own) are charged against host RAM.
