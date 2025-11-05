import sys
import argparse
import textwrap

parser = argparse.ArgumentParser(add_help=False,
    formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-q", "--quick", action="store_true",
    help="skip checking the required library")

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
    default="Arnold",
    help="get a voice object by name (default: Arnold)")
selmodes.add_argument("-v", "--voice", type=int, metavar="NUMBER",
    help="get a voice object by number (see --list)")
selopts.add_argument("-f", "--filter", action="append", metavar="KEY=VAL",
    default=["use case=narration"],
    help=textwrap.dedent('''\
        filter voices by labels (default: "use case=narration")
        this option can be used multiple times
        filtering will be disabled if the first -f has no "=" (e.g. -f "any")
        '''))

outmodes = parser.add_argument_group("output")
outgroup = outmodes.add_mutually_exclusive_group()
outgroup.add_argument("-s", "--save", metavar="FILE",
    default="audio.mp3",
    help="save the TTS to a file (default: audio.mp3)")
outgroup.add_argument("-p", "--play", action="store_true",
    help="play the TTS with ffplay")

args = parser.parse_args()

if not args.quick:
    import importlib.util
    if importlib.util.find_spec("elevenlabs") is None:
        print("elevenlabs library is not installed, you can install it to your enviroment using 'pip install elevenlabs'")
        sys.exit()

from elevenlabs import voices, generate, play, save

if args.filter and "=" in args.filter[0]:
    voicelist = voices()
    for f in args.filter:
        label, value = f.split("=")
        voicelist = filter(lambda x: x.labels.get(label) == value, voicelist)
    voicelist = list(voicelist)
else:
    voicelist = list(voices())

if args.list:
    for i, v in enumerate(voicelist):
        print(str(i) + ": " + v.name + " " + str(v.labels))
    sys.exit()

if args.voice:
    voice = voicelist[args.voice % len(voicelist)]
else:
    voice = args.name
    # if -n should consult -f, use the following
    #voice = next(x for x in voicelist if x.name == args.name)

audio = generate(
    text=str(args.inputfile.read()),
    voice=voice
)
if args.play:
    play(audio)
else:
    save(audio, args.save) 
