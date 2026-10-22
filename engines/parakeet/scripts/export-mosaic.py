#!/usr/bin/env python3
"""Export production Mosaic, preserving its encoder and quantizing eight decoder matrices."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def verify(path, identity):
    if path.stat().st_size != identity['bytes'] or sha(path) != identity['sha256']:
        raise ValueError(f'Artifact identity mismatch: {path}')


def export_decoder(source, output, recipe):
    import gguf
    import numpy as np
    source, output = Path(source), Path(output)
    receipt = output.with_suffix('.receipt.json')
    if output.exists() or receipt.exists():
        raise FileExistsError('Preserve existing output and receipt')
    verify(source, recipe['source'])
    if sha(Path(gguf.quants.__file__)) != recipe['quantizer_source_sha256']:
        raise ValueError('GGUF quantizer source differs from the qualified implementation')
    old = gguf.GGUFReader(str(source))
    tensors = {t.name: t for t in old.tensors}
    targets = set(recipe['targets'])
    if len(tensors) != recipe['source']['tensors'] or not targets <= tensors.keys():
        raise ValueError('Unexpected source tensor inventory')
    for name in targets:
        t = tensors[name]
        if t.tensor_type != gguf.GGMLQuantizationType.F16 or len(t.shape) != 2:
            raise ValueError(f'Unexpected decoder source type/shape: {name}')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb'):
        pass
    try:
        writer = gguf.GGUFWriter(str(output), old.get_field('general.architecture').contents())
        for name, field in old.fields.items():
            if name.startswith('GGUF.') or name == 'general.architecture':
                continue
            writer.add_key_value(name, field.contents(), field.types[0],
                                 field.types[-1] if len(field.types) > 1 else None)
        for tensor in old.tensors:
            if tensor.name in targets:
                payload = gguf.quants.quantize(np.asarray(tensor.data, dtype=np.float32),
                                              gguf.GGMLQuantizationType.Q8_0)
                writer.add_tensor(tensor.name, payload, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
            else:
                writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        verify(output, recipe['output'])
        result = {'name': 'parakeet-mosaic', 'version': '1',
                  'source_sha256': sha(source), 'bytes': output.stat().st_size,
                  'sha256': sha(output), 'decoder_matrices_q8': sorted(targets),
                  'unchanged_payloads': len(tensors) - len(targets),
                  'metadata_preserved': True, 'requires_affine_fused_kernel': False}
        with receipt.open('x') as stream:
            stream.write(json.dumps(result, indent=2) + '\n')
        return result
    except BaseException:
        output.unlink(missing_ok=True)
        raise


def main():
    scripts = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--archive', type=Path, help='Original NVIDIA .nemo; reconstruct the exact encoder source first')
    source.add_argument('--source-gguf', type=Path, help='Exact retained 503,670,112-byte encoder-source GGUF')
    parser.add_argument('--output', type=Path, default=Path('models/parakeet-mosaic.gguf'))
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.receipt.json').exists():
        parser.error('Preserve existing output and receipt')
    recipe = json.loads((scripts.parent / 'configs/parakeet-mosaic.decoder.recipe.json').read_text())
    if args.source_gguf:
        result = export_decoder(args.source_gguf, args.output, recipe)
    else:
        with tempfile.TemporaryDirectory(prefix='mosaic-export-') as tmp:
            retained = Path(tmp) / 'encoder-source.gguf'
            subprocess.run([sys.executable, str(scripts / 'export-mosaic-compact.py'),
                            '--archive', str(args.archive), '--output', str(retained)], check=True)
            result = export_decoder(retained, args.output, recipe)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
