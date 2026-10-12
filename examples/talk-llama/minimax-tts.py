import sys
import os
import json
import argparse
import subprocess
import tempfile
import urllib.request

MINIMAX_VOICES = [
    'English_Graceful_Lady',
    'English_Insightful_Speaker',
    'English_radiant_girl',
    'English_Persuasive_Man',
    'English_Lucky_Robot',
    'English_expressive_narrator',
]

parser = argparse.ArgumentParser(add_help=False,
    formatter_class=argparse.RawTextHelpFormatter,
    description='MiniMax TTS client for whisper.cpp talk-llama example')

modes = parser.add_argument_group("action")
modes.add_argument("inputfile", metavar="TEXTFILE",
    nargs='?', type=argparse.FileType(), default=sys.stdin,
    help="read the text file (default: stdin)")
modes.add_argument("-l", "--list", action="store_true",
    help="show the list of voices and exit")
modes.add_argument("-h", "--help", action="help",
    help="show this help and exit")

selopts = parser.add_argument_group("voice selection")
selmodes = selopts.add_mutually_exclusive_group()
selmodes.add_argument("-n", "--name",
    default="English_Graceful_Lady",
    help="voice ID to use (default: English_Graceful_Lady)")
selmodes.add_argument("-v", "--voice", type=int, metavar="NUMBER",
    help="voice by index number (see --list)")

outmodes = parser.add_argument_group("output")
outgroup = outmodes.add_mutually_exclusive_group()
outgroup.add_argument("-s", "--save", metavar="FILE",
    default="audio.mp3",
    help="save the TTS to a file (default: audio.mp3)")
outgroup.add_argument("-p", "--play", action="store_true",
    help="play the TTS with ffplay")

apiopts = parser.add_argument_group("API options")
apiopts.add_argument("-k", "--api-key", metavar="KEY",
    default=os.environ.get("MINIMAX_API_KEY", ""),
    help="MiniMax API key (default: $MINIMAX_API_KEY)")
apiopts.add_argument("-m", "--model",
    default="speech-2.8-hd",
    help="TTS model to use (default: speech-2.8-hd)")
apiopts.add_argument("-b", "--base-url",
    default=os.environ.get("MINIMAX_BASE_URL", "https://api.minimax.io"),
    help="MiniMax base URL (default: https://api.minimax.io)")

args = parser.parse_args()

if args.list:
    for i, v in enumerate(MINIMAX_VOICES):
        print(str(i) + ": " + v)
    sys.exit()

if not args.api_key:
    print("MiniMax API key is required. Set MINIMAX_API_KEY environment variable or use -k.")
    sys.exit(1)

if args.voice is not None:
    voice_id = MINIMAX_VOICES[args.voice % len(MINIMAX_VOICES)]
else:
    voice_id = args.name

text = args.inputfile.read()

url = args.base_url.rstrip("/") + "/v1/t2a_v2"
payload = json.dumps({
    "model": args.model,
    "text": text,
    "stream": True,
    "voice_setting": {
        "voice_id": voice_id,
        "speed": 1,
        "vol": 1,
        "pitch": 0,
    },
    "audio_setting": {
        "sample_rate": 32000,
        "bitrate": 128000,
        "format": "mp3",
        "channel": 1,
    },
}).encode("utf-8")

req = urllib.request.Request(url, data=payload, method="POST")
req.add_header("Content-Type", "application/json")
req.add_header("Authorization", "Bearer " + args.api_key)

audio_chunks = []
buffer = b""

with urllib.request.urlopen(req) as resp:
    for raw_line in resp:
        line = raw_line.decode("utf-8", errors="replace").rstrip("\n\r")
        if not line.startswith("data:"):
            continue
        json_str = line[5:].strip()
        if not json_str or json_str == "[DONE]":
            continue
        try:
            event = json.loads(json_str)
            audio_hex = event.get("data", {}).get("audio", "")
            if audio_hex:
                audio_chunks.append(bytes.fromhex(audio_hex))
        except (json.JSONDecodeError, ValueError):
            pass

audio = b"".join(audio_chunks)

if args.play:
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(audio)
        tmp_path = tmp.name
    try:
        subprocess.run(
            ["ffplay", "-autoexit", "-nodisp", "-loglevel", "quiet",
             "-hide_banner", "-i", tmp_path],
            check=False,
        )
    finally:
        os.unlink(tmp_path)
else:
    with open(args.save, "wb") as f:
        f.write(audio)
