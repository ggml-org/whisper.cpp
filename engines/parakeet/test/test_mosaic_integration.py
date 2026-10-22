#!/usr/bin/env python3
"""Verify exact Mosaic artifacts, then run real audio through the public Engine API."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import wave


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--binary', type=Path, required=True)
    ap.add_argument('--fixtures', type=Path, required=True)
    ap.add_argument('--clip', required=True)
    ap.add_argument('--variant', choices=['mosaic', 'compact'], default='mosaic')
    ap.add_argument('--mode', choices=['native', 'coreml'], required=True)
    ap.add_argument('--backend', choices=['cpu', 'MTL0', 'CUDA0', 'ROCm0'], required=True)
    args = ap.parse_args()
    engine = Path(__file__).resolve().parents[1]
    stem = 'parakeet-mosaic' + ('-compact' if args.variant == 'compact' else '')
    definition = json.loads((engine / f'configs/{stem}.default.json').read_text())
    clips = json.loads(Path(__file__).with_name('mosaic-fixtures.json').read_text())['clips']
    clip = next(c for c in clips if c['id'] == args.clip)
    bundle = args.fixtures / ('compact' if args.variant == 'compact' else 'bundle')
    model = bundle / f'{stem}.gguf'
    require(model.stat().st_size == definition['bytes'] and sha(model) == definition['sha256'],
            'GGUF differs from the frozen ' + stem)
    audio = args.fixtures / 'audio' / (clip['id'] + '.wav')
    receipt = json.loads((args.fixtures / 'audio/receipt.json').read_text())[clip['id']]
    require(receipt['source_flac_sha256'] == clip['source_flac_sha256'] and
            receipt['wav_sha256'] == sha(audio), 'audio provenance/hash mismatch')
    with wave.open(str(audio)) as wav:
        require((wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getnframes()) ==
                (16000, 1, 2, clip['samples']), 'audio shape differs from the complete source recording')
    if args.mode == 'coreml':
        complete = json.loads((bundle / 'complete.json').read_text())
        require(complete['returncode'] == 0 and complete['gguf_sha256'] == definition['sha256'] and
                complete['plan_sha256'] == sha(bundle / 'plan.json'), 'sidecar provenance mismatch')
        require(complete['inputs'] == {'mel': [128, 512], 'mel_valid_mask': [512]}, 'wrong sidecar IO')
        compiled = bundle / f'{stem}-encoder.mlmodelc'
        actual = {str(p.relative_to(compiled)): {'bytes': p.stat().st_size, 'sha256': sha(p)}
                  for p in sorted(compiled.rglob('*')) if p.is_file()}
        require(bool(actual) and actual == complete['compiled_inventory'], 'sidecar inventory mismatch')
    subprocess.run([str(args.binary.resolve()), str(model.resolve()), str(audio.resolve()),
                    clip['expected'], str(clip['samples']), args.mode, args.backend],
                   check=True, timeout=540)


if __name__ == '__main__':
    main()
