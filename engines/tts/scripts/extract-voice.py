#!/usr/bin/env python3
"""
extract-voice.py — turn an arbitrary voice recording into a clean 24 kHz mono
WAV that's optimal as a Chatterbox voice-cloning reference.

Workflow:
  1. Probe the source with ffprobe (duration, codec, bitrate).
  2. Use ffmpeg silencedetect to split the file into speech regions.
  3. Rank candidate windows (5-15 s, longest clean run preferred; falls back to
     concatenating the two best short runs when no single long block exists).
  4. Pick a filter chain based on source codec:
       clean   (WAV / FLAC / high-bitrate AAC+MP3): minimal (highpass + alimiter)
       lossy   (Opus / low-bitrate AAC+MP3):         full recovery chain
                                                     (highpass + denoise + EQ +
                                                      loudnorm + alimiter)
  5. Run ffmpeg, writing `voices/<name>.wav`.
  6. Optionally bake the voice profile via `./build/tts-cli --save-voice`.

Typical use:
    ./scripts/extract-voice.py ~/Downloads/marco.ogg
    ./scripts/extract-voice.py ~/Downloads/marco.ogg --name marco --bake
    ./scripts/extract-voice.py ~/Downloads/marco.ogg --name marco --target 8

Requires: ffmpeg / ffprobe on PATH, Python 3.8+, and (if --bake) a built
./build/tts-cli plus models/t3-q8_0.gguf + models/chatterbox-s3gen-q8_0.gguf.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Chatterbox requirements.
# ---------------------------------------------------------------------------
MIN_REF_SECONDS = 5.1      # tts-cli errors under 5.0 s; leave a safety margin
MAX_REF_SECONDS = 15.0     # longer than this dilutes the speaker embedding
DEFAULT_TARGET  = 10.0     # sweet spot between prosody variety and focused timbre
SAMPLE_RATE     = 24000


# ---------------------------------------------------------------------------
# Filter chains — tuned for Chatterbox's 80-channel log-mel pipeline.
#
# "clean"     — minimal (highpass + alimiter), trust the source.  Right
#               thing for WAV/FLAC and high-bitrate AAC/MP3.
# "lossy"     — restores presence / air that low-bitrate codecs strip and
#               loudness-normalises so the speaker embedding doesn't drift
#               on the shouted/whispered axis.  Good for short sources
#               without a usable silent region.
# "noisered"  — SoX noiseprof + noisered: profile the file's silent head,
#               spectrally subtract that codec noise floor from the speech
#               window, then highpass + peak-normalise.  On an 18 kbps
#               Opus source this beat the "lossy" chain cleanly in our
#               A/B testing (the voice encoder preferred cleaner-but-
#               duller over synthetic-brightness-tacked-on).  Requires
#               SoX and at least ~0.3 s of silence at the start of the
#               file (ffprobed via silencedetect).
# ---------------------------------------------------------------------------
FILTER_CHAIN_CLEAN = (
    "highpass=f=60,"
    "alimiter=limit=0.85:level=disabled"
)
FILTER_CHAIN_LOSSY = (
    "highpass=f=60,"
    "afftdn=nr=6:nt=w,"
    "equalizer=f=200:w=150:g=-1,"
    "equalizer=f=3200:w=2200:g=2.5,"
    "equalizer=f=7500:w=2500:g=3,"
    "loudnorm=I=-18:TP=-2:LRA=8,"
    "alimiter=limit=0.85:level=disabled"
)

# "noisered" is executed via SoX, not as an ffmpeg -af string.  See
# extract_via_sox_noisered() below.
NOISERED_MIN_SILENT_HEAD = 0.3   # s of silence at the start of the file
                                 # required before we can build a noise
                                 # profile.

# Sources with any of these codec + bitrate combinations are treated as "lossy".
LOSSY_CODECS = {"opus", "vorbis"}
LOSSY_BITRATE_THRESHOLD = {    # <= this bitrate → lossy chain
    "aac":       96_000,
    "mp3":       128_000,
    "opus":    1_000_000,      # any opus → lossy chain
    "vorbis":  1_000_000,
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        sys.stderr.write("$ " + " ".join(cmd) + "\n")
        sys.stderr.write(r.stderr)
        sys.exit(r.returncode)
    return r


def _require(prog: str) -> None:
    if not shutil.which(prog):
        sys.exit(f"error: {prog} not found on PATH")


# ---------------------------------------------------------------------------
# Probing & silence detection
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class SourceInfo:
    path:       Path
    duration:   float       # seconds
    codec:      str         # e.g. "opus", "aac", "pcm_s16le"
    bitrate:    int         # bps; 0 = unknown
    channels:   int
    sample_rate: int
    silent_head:  float = 0.0    # duration of silence at start of file (s)
    # (start, length) of the longest internal silence block in the file;
    # (0, 0) if nothing found.  Used as the noise-profile source when
    # silent_head is too short.
    longest_silence: tuple = (0.0, 0.0)

    @property
    def is_lossy(self) -> bool:
        if self.codec in LOSSY_CODECS:
            return True
        thr = LOSSY_BITRATE_THRESHOLD.get(self.codec)
        return thr is not None and 0 < self.bitrate <= thr

    def noise_profile_source(self) -> tuple[float, float] | None:
        """
        Pick the best (start, length) window to use as noise profile source.
        Preference order:
          1. Any internal silence block >= NOISERED_MIN_SILENT_HEAD (we take
             a 0.3 s slice from its middle — more likely to be true room
             tone than the edges, which often contain breath tails).
          2. The silent head of the file, if long enough.
          3. The first 0.3 s of the file even when it's speech.  SoX's
             spectral subtraction of "speech-tonality" content empirically
             produces a result the voice encoder likes on heavily
             Opus-damaged sources (beats the EQ-boost chain in A/B).
        Returns None only when the file is too short for any profile.
        """
        st, ln = self.longest_silence
        if ln >= NOISERED_MIN_SILENT_HEAD:
            prof_len = min(0.3, ln - 0.05)
            prof_start = st + (ln - prof_len) / 2.0
            return prof_start, prof_len
        if self.silent_head >= NOISERED_MIN_SILENT_HEAD:
            return 0.0, min(0.3, self.silent_head - 0.05)
        if self.duration >= 0.4:
            return 0.0, 0.3
        return None


def probe(path: Path) -> SourceInfo:
    j = _run([
        "ffprobe", "-v", "error", "-of", "json",
        "-show_entries", "format=duration,bit_rate:stream=codec_name,channels,sample_rate,bit_rate",
        "-select_streams", "a:0",
        str(path),
    ]).stdout
    d = json.loads(j)
    stream = d["streams"][0]
    fmt    = d["format"]
    bitrate = int(stream.get("bit_rate") or fmt.get("bit_rate") or 0)
    return SourceInfo(
        path        = path,
        duration    = float(fmt["duration"]),
        codec       = stream["codec_name"],
        bitrate     = bitrate,
        channels    = int(stream.get("channels", 1)),
        sample_rate = int(stream.get("sample_rate", SAMPLE_RATE)),
    )


@dataclasses.dataclass
class Region:
    start: float
    end:   float

    @property
    def length(self) -> float:
        return self.end - self.start


def find_speech_regions(info: SourceInfo,
                        silence_db: float = -30.0,
                        silence_min: float = 0.3) -> list[Region]:
    """Return [(start, end), ...] of speech regions (complement of silencedetect)."""
    r = subprocess.run(
        ["ffmpeg", "-nostats", "-i", str(info.path),
         "-af", f"silencedetect=noise={silence_db}dB:d={silence_min}",
         "-f", "null", "-"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
    )
    starts, ends = [], []
    for line in r.stderr.splitlines():
        if "silence_start:" in line:
            starts.append(float(line.split("silence_start:")[1].strip()))
        elif "silence_end:" in line:
            # "silence_end: 12.345 | silence_duration: ..."
            ends.append(float(line.split("silence_end:")[1].split("|")[0].strip()))

    # Speech regions = file span minus silence intervals.
    cuts: list[tuple[float, float]] = []  # (silence_start, silence_end)
    for s, e in zip(starts, ends):
        cuts.append((s, e))

    regions: list[Region] = []
    cursor = 0.0
    for s, e in cuts:
        if s > cursor:
            regions.append(Region(cursor, s))
        cursor = e
    if cursor < info.duration:
        regions.append(Region(cursor, info.duration))

    # Record how much silence (if any) sits at the head of the file, so
    # the caller can decide whether a SoX-noiseprof noise-subtraction
    # pass is viable.  silencedetect brackets SILENCE intervals, so the
    # first silence_end is effectively the length of leading silence.
    if ends and (not starts or ends[0] < starts[0]):
        info.silent_head = ends[0]

    # Also track the longest internal silence block so the noisered chain
    # has an in-file noise sample even when the file starts with speech.
    # silencedetect gives us paired (start, end) intervals.
    best_len = 0.0
    best_st  = 0.0
    for s, e in zip(starts, ends):
        ln = e - s
        if ln > best_len:
            best_len = ln
            best_st  = s
    info.longest_silence = (best_st, best_len)

    # Drop micro-fragments; they're usually plosives / sniffs, not speech.
    return [r for r in regions if r.length >= 1.0]


# ---------------------------------------------------------------------------
# Region selection — pick the window that maximises "likely-to-clone-well".
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class WindowPlan:
    # Either one continuous window (a..b) or two concatenated windows.
    segments: list[Region]
    reason:   str

    @property
    def total(self) -> float:
        return sum(s.length for s in self.segments)


def plan_window(regions: list[Region],
                target: float,
                safety_margin: float = 0.2) -> WindowPlan:
    """
    Pick a window of `target` seconds (clamped to [MIN_REF_SECONDS, MAX_REF_SECONDS]).

    Strategy:
      1. If any single region is >= target, pick the first such region and
         extract `target` seconds from its start, skipping `safety_margin` at
         each edge to avoid clipping into the adjacent silence boundary (where
         silencedetect may have under-counted consonant tails).
      2. Else if the longest region is >= MIN_REF_SECONDS, use it in full.
      3. Else concatenate the two longest regions.  Avoids single-phrase
         references which can over-fit on one phrase's prosody.
      4. Else error out — there's not enough speech.
    """
    if not regions:
        sys.exit("error: no speech regions detected (silencedetect found only silence)")

    target = max(MIN_REF_SECONDS, min(MAX_REF_SECONDS, target))

    long_enough = [r for r in regions if r.length >= target + 2 * safety_margin]
    if long_enough:
        # Prefer regions in the middle of the recording: the speaker is usually
        # warmed up by then and hasn't started wrapping up yet.
        long_enough.sort(key=lambda r: abs((r.start + r.end) / 2 -
                                            (regions[0].start + regions[-1].end) / 2))
        r = long_enough[0]
        mid = (r.start + r.end) / 2
        start = max(r.start + safety_margin, mid - target / 2)
        end   = min(r.end   - safety_margin, start + target)
        start = end - target  # re-anchor in case end hit the tail bound
        return WindowPlan(
            [Region(start, end)],
            f"continuous {target:.1f}s from region [{r.start:.2f}..{r.end:.2f}]",
        )

    # No single window is long enough.  Look for the longest region outright.
    regions_sorted = sorted(regions, key=lambda r: -r.length)
    longest = regions_sorted[0]
    if longest.length >= MIN_REF_SECONDS:
        start = longest.start + safety_margin
        end   = min(longest.end - safety_margin, start + target)
        if end - start >= MIN_REF_SECONDS:
            return WindowPlan(
                [Region(start, end)],
                f"best single region [{start:.2f}..{end:.2f}] ({end-start:.1f}s)",
            )

    # Concatenate the two longest regions.
    r1, r2 = regions_sorted[0], regions_sorted[1] if len(regions_sorted) > 1 else None
    if r2 is None:
        sys.exit(f"error: only one speech region found, {longest.length:.1f}s — "
                 f"Chatterbox requires > {MIN_REF_SECONDS:.1f}s")
    budget = target
    a = min(r1.length - 2 * safety_margin, budget * 0.6)
    b = min(r2.length - 2 * safety_margin, budget - a)
    if a + b < MIN_REF_SECONDS:
        sys.exit(f"error: total speech is only {a+b:.1f}s after trimming silences; "
                 f"need > {MIN_REF_SECONDS:.1f}s")
    seg_a = Region(r1.start + safety_margin, r1.start + safety_margin + a)
    seg_b = Region(r2.start + safety_margin, r2.start + safety_margin + b)
    # Concat in source order so prosody reads naturally.
    segs = sorted([seg_a, seg_b], key=lambda r: r.start)
    return WindowPlan(
        segs,
        f"concat of two regions ({a:.1f}s + {b:.1f}s) — no single block "
        f">= {target:.1f}s",
    )


# ---------------------------------------------------------------------------
# Extraction — ffmpeg chain OR SoX noiseprof pipeline
# ---------------------------------------------------------------------------
def _pick_chain(info: SourceInfo, override: str | None) -> str:
    """Pick the chain name: 'clean' | 'lossy' | 'noisered'."""
    if override:
        return override
    if info.is_lossy:
        # Prefer the SoX noise-profile pipeline; it beat the EQ-boost
        # "lossy" chain on 18 kbps Opus in A/B testing.  noise_profile_
        # source() falls back to sampling the first 0.3 s of the file
        # even when that's speech (SoX's spectral subtraction happens
        # to produce a voice-encoder-friendly signal in that case too).
        if shutil.which("sox") and info.noise_profile_source() is not None:
            return "noisered"
        return "lossy"
    return "clean"


def _extract_ffmpeg_chain(info: SourceInfo, plan: WindowPlan, out_wav: Path,
                          chain_str: str, verbose: bool) -> None:
    """Run a standard ffmpeg -af chain over the window(s)."""
    if len(plan.segments) == 1:
        seg = plan.segments[0]
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-ss", f"{seg.start:.3f}", "-i", str(info.path), "-t", f"{seg.length:.3f}",
            "-ac", "1", "-ar", str(SAMPLE_RATE), "-acodec", "pcm_s16le",
            "-af", chain_str,
            str(out_wav),
        ]
        if verbose:
            sys.stderr.write("$ " + " ".join(cmd) + "\n")
        _run(cmd)
        return

    # Multiple segments: filter each separately (so loudnorm etc. sees
    # consistent material), concat via the concat demuxer.
    tmp_files: list[Path] = []
    for i, seg in enumerate(plan.segments):
        tmp = out_wav.with_suffix(f".part{i}.wav")
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-ss", f"{seg.start:.3f}", "-i", str(info.path), "-t", f"{seg.length:.3f}",
            "-ac", "1", "-ar", str(SAMPLE_RATE), "-acodec", "pcm_s16le",
            "-af", chain_str,
            str(tmp),
        ]
        if verbose:
            sys.stderr.write("$ " + " ".join(cmd) + "\n")
        _run(cmd)
        tmp_files.append(tmp)

    # Concat via ffmpeg's concat demuxer.
    list_txt = out_wav.with_suffix(".concat.txt")
    list_txt.write_text("".join(f"file '{p.resolve()}'\n" for p in tmp_files))
    cmd = ["ffmpeg", "-y", "-v", "error",
           "-f", "concat", "-safe", "0", "-i", str(list_txt),
           "-c", "copy", str(out_wav)]
    if verbose:
        sys.stderr.write("$ " + " ".join(cmd) + "\n")
    _run(cmd)
    for p in tmp_files + [list_txt]:
        p.unlink(missing_ok=True)


def _extract_sox_noisered(info: SourceInfo, plan: WindowPlan, out_wav: Path,
                          verbose: bool) -> None:
    """
    Two-pass pipeline: sox noiseprof on a silent snippet from the head of
    the file, then sox noisered on the ffmpeg-extracted speech window.
    The final signal is highpass-filtered at 60 Hz and peak-normalised to
    -3 dBFS, matching the 'S' variant that won the A/B test.
    """
    _require("sox")
    prof = info.noise_profile_source()
    if prof is None:
        sys.exit(f"error: 'noisered' chain needs at least 0.3 s of source to sample "
                 f"as a noise profile (file is only {info.duration:.2f} s)")
    prof_start, prof_len = prof
    noise_wav  = out_wav.with_suffix(".noise.wav")
    noise_prof = out_wav.with_suffix(".noise.prof")
    raw_wav    = out_wav.with_suffix(".raw.wav")

    # 1) Extract noise snippet as 24 kHz mono s16le.
    cmd = ["ffmpeg", "-y", "-v", "error",
           "-ss", f"{prof_start:.3f}", "-t", f"{prof_len:.3f}", "-i", str(info.path),
           "-ac", "1", "-ar", str(SAMPLE_RATE), "-acodec", "pcm_s16le",
           str(noise_wav)]
    if verbose: sys.stderr.write("$ " + " ".join(cmd) + "\n")
    _run(cmd)

    # 2) Build the noise profile.
    cmd = ["sox", str(noise_wav), "-n", "noiseprof", str(noise_prof)]
    if verbose: sys.stderr.write("$ " + " ".join(cmd) + "\n")
    _run(cmd)

    # 3) Extract the speech window (possibly concat if the plan is
    #    two regions) to a raw 24 kHz mono s16le wav.
    if len(plan.segments) == 1:
        seg = plan.segments[0]
        cmd = ["ffmpeg", "-y", "-v", "error",
               "-ss", f"{seg.start:.3f}", "-i", str(info.path), "-t", f"{seg.length:.3f}",
               "-ac", "1", "-ar", str(SAMPLE_RATE), "-acodec", "pcm_s16le",
               str(raw_wav)]
        if verbose: sys.stderr.write("$ " + " ".join(cmd) + "\n")
        _run(cmd)
    else:
        tmp_parts: list[Path] = []
        for i, seg in enumerate(plan.segments):
            p = out_wav.with_suffix(f".part{i}.wav")
            cmd = ["ffmpeg", "-y", "-v", "error",
                   "-ss", f"{seg.start:.3f}", "-i", str(info.path), "-t", f"{seg.length:.3f}",
                   "-ac", "1", "-ar", str(SAMPLE_RATE), "-acodec", "pcm_s16le",
                   str(p)]
            if verbose: sys.stderr.write("$ " + " ".join(cmd) + "\n")
            _run(cmd)
            tmp_parts.append(p)
        list_txt = out_wav.with_suffix(".concat.txt")
        list_txt.write_text("".join(f"file '{p.resolve()}'\n" for p in tmp_parts))
        cmd = ["ffmpeg", "-y", "-v", "error",
               "-f", "concat", "-safe", "0", "-i", str(list_txt),
               "-c", "copy", str(raw_wav)]
        if verbose: sys.stderr.write("$ " + " ".join(cmd) + "\n")
        _run(cmd)
        for p in tmp_parts + [list_txt]:
            p.unlink(missing_ok=True)

    # 4) Apply noise reduction + highpass + normalise to -3 dBFS.
    cmd = ["sox", str(raw_wav), str(out_wav),
           "noisered", str(noise_prof), "0.2",
           "highpass", "60",
           "gain", "-n", "-3"]
    if verbose: sys.stderr.write("$ " + " ".join(cmd) + "\n")
    _run(cmd)

    # Clean up intermediates.
    for p in (noise_wav, noise_prof, raw_wav):
        p.unlink(missing_ok=True)


def extract(info: SourceInfo, plan: WindowPlan, out_wav: Path,
            chain: str, verbose: bool = False) -> None:
    """
    Dispatch on `chain` (one of "clean" / "lossy" / "noisered") and run
    the matching extraction pipeline.
    """
    if chain == "noisered":
        _extract_sox_noisered(info, plan, out_wav, verbose)
    elif chain == "lossy":
        _extract_ffmpeg_chain(info, plan, out_wav, FILTER_CHAIN_LOSSY, verbose)
    elif chain == "clean":
        _extract_ffmpeg_chain(info, plan, out_wav, FILTER_CHAIN_CLEAN, verbose)
    else:
        sys.exit(f"internal error: unknown chain {chain!r}")


# ---------------------------------------------------------------------------
# Optional: bake the voice profile via ./build/tts-cli.
# ---------------------------------------------------------------------------
def bake(tts_cli: Path, t3: Path, s3gen: Path,
         ref_wav: Path, out_dir: Path,
         n_gpu_layers: int, verbose: bool) -> None:
    for p, what in [(tts_cli, "tts-cli binary"),
                    (t3, "T3 model"),
                    (s3gen, "S3Gen model")]:
        if not p.exists():
            sys.exit(f"error: --bake requires {what} at {p}")

    cmd = [
        str(tts_cli),
        "--model",            str(t3),
        "--s3gen-gguf",       str(s3gen),
        "--reference-audio",  str(ref_wav),
        "--save-voice",       str(out_dir),
        "--n-gpu-layers",     str(n_gpu_layers),
    ]
    if verbose:
        sys.stderr.write("$ " + " ".join(cmd) + "\n")
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("input", type=Path, help="Source audio file (any ffmpeg-readable format)")
    ap.add_argument("--name", "-n", default=None,
                    help="Voice name (default: stem of input file)")
    ap.add_argument("--out-dir", default=Path("voices"), type=Path,
                    help="Output directory root (default: voices/)")
    ap.add_argument("--target", "-t", type=float, default=DEFAULT_TARGET,
                    help=f"Target clip length in seconds, clamped to [{MIN_REF_SECONDS}..{MAX_REF_SECONDS}]  (default: {DEFAULT_TARGET})")
    ap.add_argument("--silence-db", type=float, default=-30.0,
                    help="silencedetect noise threshold in dBFS (default: -30)")
    ap.add_argument("--silence-min", type=float, default=0.3,
                    help="silencedetect min-silence duration in seconds (default: 0.3)")
    ap.add_argument("--force-chain", choices=["clean", "lossy", "noisered"], default=None,
                    help="Override auto filter-chain pick.  'noisered' requires SoX on "
                         "PATH and at least 0.3 s of silence at the start of the source.")
    ap.add_argument("--bake", action="store_true",
                    help="After extracting, call ./build/tts-cli --save-voice to "
                         "pre-compute the 5 conditioning tensors (faster future runs).")
    ap.add_argument("--tts-cli", type=Path, default=Path("./build/tts-cli"),
                    dest="tts_cli",
                    help="Path to tts-cli binary (default: ./build/tts-cli)")
    ap.add_argument("--t3", type=Path, default=Path("models/t3-q8_0.gguf"),
                    help="Path to T3 GGUF (default: models/t3-q8_0.gguf)")
    ap.add_argument("--s3gen", type=Path, default=Path("models/chatterbox-s3gen-q8_0.gguf"),
                    help="Path to S3Gen GGUF (default: models/chatterbox-s3gen-q8_0.gguf)")
    ap.add_argument("--n-gpu-layers", type=int, default=99,
                    help="GPU layers for --bake (default: 99)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    _require("ffmpeg")
    _require("ffprobe")

    name = args.name or args.input.stem
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ref_wav = args.out_dir / f"{name}.wav"

    info = probe(args.input)

    # silencedetect runs first because we need info.silent_head in order
    # to decide whether the 'noisered' chain is viable.
    regions = find_speech_regions(info, args.silence_db, args.silence_min)
    chain = _pick_chain(info, args.force_chain)

    print(f"source:  {info.path}")
    print(f"         {info.duration:.2f}s, {info.codec}@{info.bitrate//1000 if info.bitrate else '?'}kbps, "
          f"{info.channels}ch @ {info.sample_rate}Hz")
    print(f"         lossy={info.is_lossy}, silent_head={info.silent_head:.2f}s, "
          f"longest_silence={info.longest_silence[1]:.2f}s  -> chain: {chain}")
    if chain == "noisered":
        prof = info.noise_profile_source()
        if prof:
            ps, pl = prof
            print(f"         noise profile: [{ps:.2f}..{ps+pl:.2f}]  ({pl:.2f}s)")
    print(f"speech regions ({len(regions)}):")
    for r in regions:
        print(f"  [{r.start:7.2f}..{r.end:7.2f}]  ({r.length:5.2f}s)")

    plan = plan_window(regions, args.target)
    print(f"plan:    {plan.reason}")
    for s in plan.segments:
        print(f"         extract [{s.start:.2f}..{s.end:.2f}]  ({s.length:.2f}s)")
    print(f"total:   {plan.total:.2f}s")

    extract(info, plan, ref_wav, chain, verbose=args.verbose)
    print(f"wrote:   {ref_wav}")

    if args.bake:
        profile_dir = args.out_dir / name
        bake(args.tts_cli, args.t3, args.s3gen, ref_wav, profile_dir,
             args.n_gpu_layers, args.verbose)
        print(f"baked:   {profile_dir}")
        print(f"\nReuse with:  --ref-dir {profile_dir}")
    else:
        print(f"\nBake profile with:")
        print(f"  {args.tts_cli} \\")
        print(f"    --model {args.t3} \\")
        print(f"    --s3gen-gguf {args.s3gen} \\")
        print(f"    --reference-audio {ref_wav} \\")
        print(f"    --save-voice {args.out_dir/name} \\")
        print(f"    --n-gpu-layers {args.n_gpu_layers}")


if __name__ == "__main__":
    main()
