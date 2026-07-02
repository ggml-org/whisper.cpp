# Audio Inputs

Place fixed benchmark inputs in this directory:

- `short.wav` (~30 seconds)
- `medium.wav` (~5 minutes)
- `long.wav` (~30 minutes)

All inputs must be:

- 16 kHz
- mono
- 16-bit PCM WAV

Example conversion:

```bash
ffmpeg -i input.ext -ar 16000 -ac 1 -c:a pcm_s16le output.wav
```

After files are in place, create/update the reproducibility lock:

```bash
../bench.sh --create-lock
```
