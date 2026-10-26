import argparse
from array import array
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
import wave


ASR_RATE = 16000
MAX_ATTEMPTS = 10
MAX_SEED = (1 << 32) - 1
MAX_TEXT_CHARS = 16384
DEFAULT_TOKEN_LIMIT = 1024
ASR_TOKEN_LIMIT = 1024
TTSD_TEMPERATURE = 1.1
TTSD_TOP_P = 0.9
TTSD_TOP_K = 50
TTSD_AUDIO_REPETITION_PENALTY = 1.1
DEFAULT_TIMEOUT = 300.0
DEFAULT_ERROR_LIMIT = 0.25
MIN_REFERENCE_WORDS = 8
FRAME_PATTERN = re.compile(r"\((\d+) frames, generate ")


def words(text):
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(char if char.isalnum() else " " for char in text).split()


def distance_row(previous, token, hypothesis, index):
    current = [index]
    for column, observed in enumerate(hypothesis, 1):
        current.append(min(current[-1] + 1, previous[column] + 1,
                           previous[column - 1] + (token != observed)))
    return current


def edit_distance(reference, hypothesis):
    row = list(range(len(hypothesis) + 1))
    for index, token in enumerate(reference, 1):
        row = distance_row(row, token, hypothesis, index)
    return row[-1]


def text_scores(expected, transcript):
    target, observed = words(expected), words(transcript)
    ending = words(re.split(r"\[S\d+\]", expected)[-1])
    if not observed:
        return {"word_error_rate": 1.0, "ending_error_rate": 1.0}
    if not ending:
        raise ValueError("Expected text must end with words")
    lengths = range(max(1, len(ending) - 3), len(ending) + 4)
    ending_errors = min(edit_distance(ending, observed[-size:]) for size in lengths)
    return {"word_error_rate": edit_distance(target, observed) / len(target),
            "ending_error_rate": ending_errors / len(ending)}


def phrase_error_rate(reference, transcript):
    row = [0] * (len(transcript) + 1)
    for index, token in enumerate(reference, 1):
        row = distance_row(row, token, transcript, index)
    return min(row) / len(reference)


def unexpected_references(args, transcript):
    matches = []
    expected, observed = words(args.target), words(transcript)
    for index, (_, reference) in enumerate(args.reference, 1):
        phrase = words(reference)
        if len(phrase) < MIN_REFERENCE_WORDS:
            continue
        if phrase_error_rate(phrase, expected) <= args.max_error_rate:
            continue
        if phrase_error_rate(phrase, observed) <= args.max_error_rate:
            matches.append(index)
    return matches


def synthesis_command(args, seed, output):
    prefix = " ".join(f"[S{index}] {text}" for index, (_, text) in enumerate(args.reference, 1))
    command = [args.moss_cli, "--backbone", args.backbone, "--encoder", args.encoder,
               "--decoder", args.decoder, "--text", prefix + " " + args.target,
               "--language", "English", "--out", str(output), "--seed", str(seed),
               "--threads", str(args.threads), "--max-new-tokens", str(args.max_new_tokens),
               "--text-temperature", str(TTSD_TEMPERATURE), "--text-top-p", str(TTSD_TOP_P),
               "--text-top-k", str(TTSD_TOP_K),
               "--audio-temperature", str(TTSD_TEMPERATURE), "--audio-top-p", str(TTSD_TOP_P),
               "--audio-top-k", str(TTSD_TOP_K),
               "--audio-repetition-penalty", str(TTSD_AUDIO_REPETITION_PENALTY)]
    for audio, _ in args.reference:
        command.extend(["--dialogue-ref", audio])
    return command + (["--gpu", "--backend", "CUDA0"] if args.gpu else [])


def asr_command(args, audio, output):
    command = [args.asr_cli, "--model", args.asr_model, "--audio", str(audio),
               "--out", str(output), "--threads", str(args.threads),
               "--max-new-tokens", str(ASR_TOKEN_LIMIT)]
    return command + (["--gpu", "--backend", "cuda"] if args.gpu else [])


def run_command(command, directory, name, timeout):
    (directory / (name + "-command.json")).write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
    with (directory / (name + ".log")).open("w", encoding="utf-8") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=timeout)


def read_audio(path):
    with wave.open(str(path), "rb") as source:
        if source.getnchannels() != 1 or source.getsampwidth() != 2:
            raise ValueError("Expected mono PCM16 synthesis output")
        frames, rate = source.getnframes(), source.getframerate()
        data = source.readframes(frames)
    if len(data) != frames * 2 or not frames or rate <= 0:
        raise ValueError("Empty or truncated synthesis output")
    samples = array("h", data)
    if sys.byteorder != "little":
        samples.byteswap()
    if not any(samples):
        raise ValueError("Silent synthesis output")
    return samples, rate


def resample(samples, rate):
    output = array("h")
    for index in range(round(len(samples) * ASR_RATE / rate)):
        position = index * rate / ASR_RATE
        left = min(int(position), len(samples) - 1)
        right = min(left + 1, len(samples) - 1)
        output.append(round(samples[left] + (samples[right] - samples[left]) * (position - left)))
    return output


def prepare_asr_audio(source, output):
    samples, rate = read_audio(source)
    converted = resample(samples, rate)
    if sys.byteorder != "little":
        converted.byteswap()
    with wave.open(str(output), "wb") as destination:
        destination.setparams((1, 2, ASR_RATE, 0, "NONE", "not compressed"))
        destination.writeframes(converted.tobytes())
    return len(samples) / rate


def evaluate_attempt(args, directory, seed):
    output = directory / "audio.wav"
    run_command(synthesis_command(args, seed, output), directory, "synthesis", args.timeout)
    matches = FRAME_PATTERN.findall((directory / "synthesis.log").read_text(encoding="utf-8"))
    if len(matches) != 1:
        raise ValueError("Cannot verify generated-frame count from moss-cli output")
    frames = int(matches[0])
    if frames >= args.max_new_tokens:
        return {"status": "rejected", "reason": "generation_limit", "generated_frames": frames}
    try:
        seconds = prepare_asr_audio(output, directory / "asr-input.wav")
    except (ValueError, wave.Error) as error:
        return {"status": "rejected", "reason": "invalid_audio", "error": str(error)}
    transcript_path = directory / "transcript.json"
    run_command(asr_command(args, directory / "asr-input.wav", transcript_path),
                directory, "asr", args.timeout)
    payload = json.loads(transcript_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
        raise ValueError("ASR output must contain a text string")
    generated_tokens = payload.get("generated_tokens")
    if type(generated_tokens) is not int or generated_tokens < 0:
        raise ValueError("ASR output must contain a nonnegative generated_tokens integer")
    if generated_tokens >= ASR_TOKEN_LIMIT:
        return {"status": "rejected", "reason": "asr_generation_limit",
                "generated_frames": frames, "asr_generated_tokens": generated_tokens}
    transcript = payload["text"]
    if not words(transcript):
        return {"status": "rejected", "reason": "empty_transcript", "generated_frames": frames}
    scores = text_scores(args.target, transcript)
    references = unexpected_references(args, transcript)
    accepted = max(scores.values()) <= args.max_error_rate and not references
    reason = "reference_repetition" if references else "text_match" if accepted else "text_mismatch"
    return {"status": "accepted" if accepted else "rejected",
            "reason": reason, "unexpected_references": references, **scores,
            "generated_frames": frames, "audio_seconds": seconds, "transcript": transcript}


def publish(source, destination):
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        shutil.copyfile(source, temporary_path)
        os.link(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def save_report(directory, report):
    (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def run_guard(args):
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError("Output already exists; choose a new path")
    output.parent.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=output.stem + "-validation-", dir=output.parent))
    report = {"status": "running", "output": str(output), "evidence": str(directory),
              "target_text": args.target, "max_error_rate": args.max_error_rate, "attempts": []}
    for index in range(args.attempts):
        attempt_dir = directory / f"attempt-{index + 1}"
        attempt_dir.mkdir()
        attempt = {"seed": args.seed + index, "directory": str(attempt_dir)}
        report["attempts"].append(attempt)
        try:
            attempt.update(evaluate_attempt(args, attempt_dir, attempt["seed"]))
            if attempt["status"] == "accepted":
                publish(attempt_dir / "audio.wav", output)
                report["status"] = "accepted"
                save_report(directory, report)
                return 0, report
        except (OSError, ValueError, subprocess.SubprocessError, wave.Error) as error:
            attempt.update(status="error", error=str(error))
            report["status"] = "error"
            save_report(directory, report)
            return 1, report
        save_report(directory, report)
    report["status"] = "rejected"
    save_report(directory, report)
    return 2, report


def validate(args):
    if not 1 <= args.attempts <= MAX_ATTEMPTS or not 0 <= args.seed <= MAX_SEED - args.attempts + 1:
        raise ValueError("Invalid attempt count or seed range")
    if args.threads < 1 or args.max_new_tokens < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("Threads, token limit, and timeout must be positive")
    if not math.isfinite(args.max_error_rate) or not 0 <= args.max_error_rate <= 1:
        raise ValueError("Error threshold must be between zero and one")
    args.target = Path(args.text_file).read_text(encoding="utf-8").strip()
    if len(args.target) > MAX_TEXT_CHARS or not words(args.target) or not words(re.split(r"\[S\d+\]", args.target)[-1]):
        raise ValueError("Target text is empty or too long")
    for path in (args.moss_cli, args.asr_cli, args.backbone, args.encoder, args.decoder, args.asr_model):
        if not Path(path).is_file():
            raise ValueError(f"File does not exist: {path}")
    for audio, transcript in args.reference:
        if not Path(audio).is_file() or not words(transcript):
            raise ValueError("Each reference needs an audio file and a nonempty transcript")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Validate English TTSD cloning and retry rejected generations.")
    for name in ("moss-cli", "asr-cli", "backbone", "encoder", "decoder", "asr-model", "text-file", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--reference", nargs=2, action="append", required=True, metavar=("WAV", "TRANSCRIPT"))
    parser.add_argument("--gpu", action="store_true", help="Use CUDA device 0 for synthesis and ASR")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_TOKEN_LIMIT)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--max-error-rate", type=float, default=DEFAULT_ERROR_LIMIT)
    return parser.parse_args(argv)


def main(argv=None):
    try:
        args = parse_args(argv)
        validate(args)
        status, report = run_guard(args)
        print(json.dumps(report, indent=2))
        return status
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
