import os
import sys
import glob
import jiwer
from normalizers import EnglishTextNormalizer

def decode_hypothesis(b):
    try:
        # Depending on platforms, Whisper can emit a left double quotation
        # mark (0x93), which is Microsoft's extension to ASCII. See #3185
        # for the background.
        return b.decode('windows-1252')
    except UnicodeDecodeError:
        return b.decode('utf-8', errors='ignore')

def get_reference():
    ref = {}
    for path in glob.glob("speech-datasets/earnings21/transcripts/nlp_references/*.nlp"):
        code = os.path.basename(path).replace(".nlp", "")
        buf = []
        with open(path) as fp:
            fp.readline()
            for line in fp:
                token = line.split("|", maxsplit=1)[0]
                buf.append(token)
            ref[code] = " ".join(buf)
    return ref

def get_hypothesis():
    hyp = {}
    for path in glob.glob("speech-datasets/earnings21/media/*.mp3.txt"):
        with open(path, 'rb') as fp:
            text = decode_hypothesis(fp.read()).strip()
        code = os.path.basename(path).replace(".mp3.txt", "")
        hyp[code] = text
    return hyp

def get_codes(metadata_csv):
    codes = []
    with open(metadata_csv) as fp:
        fp.readline()
        for line in fp:
            codes.append(line.split(",")[0])
    return sorted(codes)

def main():
    if len(sys.argv) < 2:
        print("Usage: %s METADATA_CSV" % sys.argv[0], file=sys.stderr)
        return 1

    metadata_csv = sys.argv[1]
    normalizer = EnglishTextNormalizer()

    ref_orig = get_reference()
    hyp_orig = get_hypothesis()

    ref_clean = []
    hyp_clean = []

    for code in get_codes(metadata_csv):
        ref_clean.append(normalizer(ref_orig[code]))
        hyp_clean.append(normalizer(hyp_orig[code]))

    wer = jiwer.wer(ref_clean, hyp_clean)
    print(f"WER: {wer * 100:.2f}%")

if __name__ == "__main__":
    main()
