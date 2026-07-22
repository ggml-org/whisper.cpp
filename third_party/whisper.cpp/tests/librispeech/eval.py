import os
import glob
import jiwer
from normalizers import EnglishTextNormalizer

def get_reference():
    ref = {}
    for path in glob.glob('LibriSpeech/*/*/*/*.trans.txt'):
        with open(path) as fp:
            for line in fp:
                code, text = line.strip().split(" ", maxsplit=1)
                ref [code] = text
    return ref

def get_hypothesis():
    hyp = {}
    for path in glob.glob('LibriSpeech/*/*/*/*.flac.txt'):
        with open(path) as fp:
            text = fp.read().strip()
        code = os.path.basename(path).replace('.flac.txt', '')
        hyp[code] = text
    return hyp

def get_codes():
    codes = []
    for path in glob.glob('LibriSpeech/*/*/*/*.flac'):
        codes.append(os.path.basename(path).replace('.flac', ''))
    return sorted(codes)

def main():
    normalizer = EnglishTextNormalizer()

    ref_orig = get_reference()
    hyp_orig = get_hypothesis()

    ref_clean = []
    hyp_clean = []

    for code in get_codes():
        ref_clean.append(normalizer(ref_orig[code]))
        hyp_clean.append(normalizer(hyp_orig[code]))

    wer = jiwer.wer(ref_clean, hyp_clean)
    print(f"WER: {wer * 100:.2f}%")

if __name__ == '__main__':
    main()
