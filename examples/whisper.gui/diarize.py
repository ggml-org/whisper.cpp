#!/usr/bin/env python3
"""
Accurate speaker diarization for whisper.cpp, using sherpa-onnx (offline, neural).

Unlike the GUI's built-in MFCC diarization (lightweight but weak), this uses a
pyannote segmentation model + a neural speaker-embedding model, which separates
real-world voices far more reliably.

Typical workflow:
  1. Transcribe in the GUI (or whisper-cli) and export JSON.
  2. Run this on the same audio to label each segment by speaker:

       python3 diarize.py audio.wav --json audio.json --speakers 3

     -> writes audio.spk.txt  (Speaker N: text, per segment)
     and updates audio.json   (adds a "speaker" field to each segment)

  Without --json it just prints the speaker timeline (RTTM-style).

Models (download once with download-diarization-models.sh):
  --seg-model  pyannote segmentation .onnx
  --emb-model  speaker-embedding .onnx (e.g. nemo_en_titanet_small.onnx)

Requires: pip install sherpa-onnx numpy
"""

import argparse, json, sys, wave
import numpy as np
import sherpa_onnx


def read_wav(path):
    # accept any sample rate / channel count: downmix to mono and resample to 16 kHz
    with wave.open(path) as w:
        sr, ch, width, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    if width != 2:
        sys.exit(f"error: {path} is {width*8}-bit; only 16-bit PCM WAV is supported. "
                 f"Convert with: ffmpeg -i in -ar 16000 -ac 1 -c:a pcm_s16le out.wav")
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        if len(a) % ch:                            # truncated/corrupt frame data
            a = a[:len(a) - (len(a) % ch)]
        a = a.reshape(-1, ch).mean(axis=1)         # downmix to mono
    if len(a) == 0:
        sys.exit(f"error: {path} contains no audio samples")
    if sr != 16000:                                # linear resample to 16 kHz
        n_out = int(round(len(a) * 16000 / sr))
        a = np.interp(np.linspace(0, len(a), n_out, endpoint=False),
                      np.arange(len(a)), a).astype(np.float32)
    return a


def diarize(audio, seg_model, emb_model, num_speakers, threshold):
    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=seg_model)),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=emb_model),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=num_speakers if num_speakers > 0 else -1,
            threshold=threshold),
        min_duration_on=0.3, min_duration_off=0.5,
    )
    sd = sherpa_onnx.OfflineSpeakerDiarization(cfg)
    return sd.process(audio).sort_by_start_time()


def speaker_for(t0, t1, spk_segments):
    # assign the speaker whose diarized span overlaps [t0,t1] the most
    best, best_ov = -1, 0.0
    for s in spk_segments:
        ov = max(0.0, min(t1, s.end) - max(t0, s.start))
        if ov > best_ov:
            best_ov, best = ov, s.speaker
    return best


def main():
    ap = argparse.ArgumentParser(description="Neural speaker diarization for whisper.cpp")
    ap.add_argument("audio", help="16 kHz mono WAV")
    ap.add_argument("--json", help="whisper JSON export to label (from the GUI or whisper-cli -oj)")
    ap.add_argument("--speakers", type=int, default=0, help="known speaker count (0 = auto-detect)")
    ap.add_argument("--threshold", type=float, default=0.5, help="auto-mode clustering threshold")
    ap.add_argument("--seg-model", default="models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx")
    ap.add_argument("--emb-model", default="models/nemo_en_titanet_small.onnx")
    args = ap.parse_args()

    audio = read_wav(args.audio)
    print(f"diarizing {len(audio)/16000:.1f}s ...", file=sys.stderr)
    segs = diarize(audio, args.seg_model, args.emb_model, args.speakers, args.threshold)
    n_spk = len({s.speaker for s in segs})
    print(f"detected {n_spk} speaker(s), {len(segs)} segment(s)", file=sys.stderr)

    if not args.json:
        for s in segs:
            print(f"{s.start:8.2f} {s.end:8.2f}  Speaker {s.speaker + 1}")
        return

    with open(args.json) as f:
        data = json.load(f)
    items = data.get("transcription", data if isinstance(data, list) else [])
    txt_path = args.json.rsplit(".", 1)[0] + ".spk.txt"
    with open(txt_path, "w") as out:
        for it in items:
            # whisper JSON timestamps are in centiseconds
            t0, t1 = it["from"] / 100.0, it["to"] / 100.0
            spk = speaker_for(t0, t1, segs)
            it["speaker"] = spk + 1 if spk >= 0 else None
            text = (it.get("text") or "").strip()
            label = f"Speaker {spk + 1}: " if spk >= 0 else ""
            out.write(f"{label}{text}\n")
    with open(args.json, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"wrote {txt_path} and updated {args.json}", file=sys.stderr)


if __name__ == "__main__":
    main()
