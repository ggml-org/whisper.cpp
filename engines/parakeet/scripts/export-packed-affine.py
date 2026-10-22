#!/usr/bin/env python3
"""Export an explicit per-tensor affine recipe without re-quantizing its codes.

Recipe JSON: {"source_archive_sha256": "...", "tensors": {"source.weight": 4|8|16}}.
Unlisted tensors retain FP16. Depthwise kernels keep their decoded FP16 values,
as in the original packed-projection implementation. INT4/INT8 matrices use
unit-scale GGML carriers plus independent FP32 minimum/step tensors.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch
from packed_affine import encode, decode, writer_factory

MAPPING = {
    'feed_forward1.linear1.weight': 'ff1.linear1.weight',
    'feed_forward1.linear2.weight': 'ff1.linear2.weight',
    'feed_forward2.linear1.weight': 'ff2.linear1.weight',
    'feed_forward2.linear2.weight': 'ff2.linear2.weight',
    'self_attn.linear_q.weight': 'attn.q.weight',
    'self_attn.linear_k.weight': 'attn.k.weight',
    'self_attn.linear_v.weight': 'attn.v.weight',
    'self_attn.linear_out.weight': 'attn.out.weight',
    'self_attn.linear_pos.weight': 'attn.pos.weight',
    'conv.pointwise_conv1.weight': 'conv.pw1.weight',
    'conv.pointwise_conv2.weight': 'conv.pw2.weight',
    'conv.depthwise_conv.weight': None,
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def apply_recipe(state, precisions):
    unknown = set(precisions) - set(state)
    if unknown:
        raise ValueError('unknown recipe tensors: ' + str(sorted(unknown)))
    records, coverage = {}, []
    for name in state:
        value = state[name]
        value = value.half() if value.is_floating_point() else value
        bits = precisions.get(name, 16)
        if bits not in (4, 8, 16):
            raise ValueError('unsupported precision: ' + name)
        if bits != 16:
            parts = name.split('.', 3)
            if len(parts) != 4 or parts[:2] != ['encoder', 'layers'] or not parts[2].isdigit() or parts[3] not in MAPPING:
                raise ValueError('unsupported packed target: ' + name)
            record = encode(value, bits)
            decoded = decode(record).reshape(tuple(value.shape))
            value = torch.from_numpy(decoded.copy())
            target = MAPPING[parts[3]]
            if target is not None:
                target = f'encoder.blk.{int(parts[2])}.{target}'
                records[target] = record
            coverage.append({'source': name, 'bits': bits, 'gguf': target,
                             'storage': 'carrier' if target else 'decoded_fp16_depthwise'})
        state[name] = value
    return records, coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dense-control', type=Path)
    parser.add_argument('--expected-tensors', type=Path,
                        help='Optional exact source tensor SHA/shape/dtype receipts before any GGUF layout conversion')
    args = parser.parse_args()
    outputs = [args.output, args.output.with_suffix('.receipt.json')]
    if args.dense_control:
        outputs.append(args.dense_control)
    if any(p.exists() for p in outputs):
        raise ValueError('choose fresh output paths')
    recipe = json.loads(args.recipe.read_text())
    if sha(args.archive) != recipe['source_archive_sha256']:
        raise ValueError('source archive hash mismatch')
    script = Path(__file__).with_name('convert-nemo-to-gguf.py')
    spec = importlib.util.spec_from_file_location('affine_converter', script)
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    cfg, state, tok, multilingual = converter.load_nemo(args.archive)
    if converter.detect_model_type(cfg) != 'tdt':
        raise ValueError('this experimental export currently supports offline TDT only')
    records, coverage = apply_recipe(state, recipe['tensors'])
    receipts = {n: {'shape': list(t.shape), 'dtype': str(t.dtype),
        'sha256': hashlib.sha256(t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}
        for n, t in state.items()}
    if args.expected_tensors:
        expected = json.loads(args.expected_tensors.read_text())['tensors']
        if receipts != expected:
            raise ValueError('effective tensor receipts differ: ' + str([n for n in expected if receipts.get(n) != expected[n]][:10]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for output, dense in [(args.output, False), (args.dense_control, True)]:
        if output:
            converter.write_gguf(output, args.archive, cfg, state, tok, multilingual,
                                 'f16', 'tdt', writer_factory=writer_factory(records, dense=dense))
    receipt = {'recipe_sha256': sha(args.recipe), 'archive_sha256': sha(args.archive),
        'source_tensors': receipts, 'coverage': coverage,
        'carrier_count': len(records), 'export_sha256': sha(args.output),
        'export_bytes': args.output.stat().st_size,
        'dense_control_sha256': sha(args.dense_control) if args.dense_control else None,
        'dense_control_bytes': args.dense_control.stat().st_size if args.dense_control else None,
        'expected_source_receipts_verified': bool(args.expected_tensors),
        'weight_reconstruction': 'FP32 multiply, FP32 add, round to FP16; no second quantizer',
        'execution': 'transient per-projection reconstruction; no fused low-bit matmul claim'}
    args.output.with_suffix('.receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('EXPORTED', args.output, receipt['export_bytes'], 'carriers', len(records), flush=True)


if __name__ == '__main__':
    main()
