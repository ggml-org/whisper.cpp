#!/usr/bin/env python3
"""Provision public, hash-verified inputs for the opt-in Mosaic integration CI."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import urllib.request
import wave


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fixtures', type=Path, required=True)
    ap.add_argument('--coreml', action='store_true', help='Also prepare the optional Apple encoder sidecar')
    ap.add_argument('--compact-url', help='CI-accessible download URL for the exact supplied Compact GGUF')
    args = ap.parse_args()
    root = args.fixtures.resolve()
    engine = Path(__file__).resolve().parents[1]
    scripts = engine / 'scripts'
    root.mkdir(parents=True, exist_ok=True)
    archive = root / 'parakeet-tdt-0.6b-v3.nemo'
    expected = json.loads((engine / 'configs/parakeet-mosaic.recipe.json').read_text())['source_archive_sha256']
    url = 'https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/resolve/main/parakeet-tdt-0.6b-v3.nemo'
    print('Downloading original NVIDIA archive', flush=True)
    with urllib.request.urlopen(url, timeout=120) as source, archive.open('xb') as target:
        while block := source.read(8 << 20):
            target.write(block)
    if sha(archive) != expected:
        raise RuntimeError('source archive hash mismatch')
    print('Generating exact Mosaic fixtures', flush=True)
    subprocess.run([sys.executable, str(scripts / 'prepare-mosaic.py'), '--archive', str(archive),
                    '--output-dir', str(root / 'bundle')] + (['--coreml'] if args.coreml else []), check=True)

    if args.compact_url:
        compact = root / 'parakeet-mosaic-compact.gguf'
        definition = json.loads((engine / 'configs/parakeet-mosaic-compact.default.json').read_text())
        print('Downloading hash-pinned Mosaic Compact', flush=True)
        with urllib.request.urlopen(args.compact_url, timeout=120) as source, compact.open('xb') as target:
            while block := source.read(8 << 20):
                target.write(block)
        if compact.stat().st_size != definition['bytes'] or sha(compact) != definition['sha256']:
            raise RuntimeError('Compact artifact identity mismatch')
        subprocess.run([sys.executable, str(scripts / 'prepare-mosaic.py'), '--gguf', str(compact),
                        '--variant', 'compact', '--output-dir', str(root / 'compact')] +
                       (['--coreml'] if args.coreml else []), check=True)

    import soundfile as sf
    manifest = json.loads(Path(__file__).with_name('mosaic-fixtures.json').read_text())
    wanted = {c['member']: c for c in manifest['clips']}
    audio = root / 'audio'
    audio.mkdir()
    receipt = {}
    print('Fetching complete LibriSpeech fixtures', flush=True)
    with urllib.request.urlopen(manifest['source_url'], timeout=120) as stream:
        with tarfile.open(fileobj=stream, mode='r|gz') as tar:
            for member in tar:
                if member.name not in wanted:
                    continue
                clip = wanted.pop(member.name)
                payload = tar.extractfile(member).read()
                if hashlib.sha256(payload).hexdigest() != clip['source_flac_sha256']:
                    raise RuntimeError('source FLAC hash mismatch: ' + clip['id'])
                pcm, rate = sf.read(io.BytesIO(payload), dtype='int16')
                if rate != 16000 or pcm.ndim != 1 or len(pcm) != clip['samples']:
                    raise RuntimeError('unexpected source audio shape: ' + clip['id'])
                path = audio / (clip['id'] + '.wav')
                with wave.open(str(path), 'wb') as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(16000)
                    wav.writeframes(pcm.astype('<i2').tobytes())
                receipt[clip['id']] = {'source_flac_sha256': clip['source_flac_sha256'], 'wav_sha256': sha(path)}
                if not wanted:
                    break
    if wanted:
        raise RuntimeError('missing source audio members: ' + str(sorted(wanted)))
    (audio / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('Mosaic integration fixtures ready', flush=True)


if __name__ == '__main__':
    main()
