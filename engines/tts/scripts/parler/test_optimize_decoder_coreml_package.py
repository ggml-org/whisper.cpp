#!/usr/bin/env python3
"""Explicit integration test against an original tiny package and measured golden.

This test imports Core ML protobuf and runs the real optimizer CLI. It does not
run inference or convert a model. Nothing runs merely by importing this file.
All outputs/logs are retained in a NEW --out directory; --compile optionally
exercises xcrun on the optimized tiny copy. Paths are supplied by the caller,
so no experimental model path or identity is part of the production workflow.
"""
import argparse
import copy
import importlib.util
import json
import shutil
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiny-package', type=Path, required=True)
    parser.add_argument('--golden-package', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--compile', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parent
    helper = root/'parler_qkv'
    module_spec = importlib.util.spec_from_file_location('parler_qkv', helper/'__init__.py',
                                                       submodule_search_locations=[str(helper)])
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    from parler_qkv import package, spec
    from coremltools.proto import Model_pb2
    from google.protobuf.json_format import MessageToDict, ParseDict

    def read(path):
        model = Model_pb2.Model()
        model.ParseFromString(package.spec_path(path).read_bytes())
        return model

    def run(name, command, expected):
        done = subprocess.run(command, capture_output=True, text=True)
        (args.out/(name+'.stdout')).write_text(done.stdout, encoding='utf-8')
        (args.out/(name+'.stderr')).write_text(done.stderr, encoding='utf-8')
        (args.out/(name+'.command.json')).write_text(json.dumps(command, indent=2)+'\n', encoding='utf-8')
        if done.returncode != expected:
            raise AssertionError(f'{name}: expected exit {expected}, got {done.returncode}; see retained logs')
        return done

    original = read(args.tiny_package)
    doc = MessageToDict(original, preserving_proto_field_name=True)
    if spec.metadata(doc)['n_layer'] != 2:
        raise AssertionError('this integration test is tiny only')
    original_before = package.file_inventory(args.tiny_package)
    golden_before = package.file_inventory(args.golden_package)
    package.roundtrip_equal(original, ParseDict(copy.deepcopy(doc), Model_pb2.Model()), 'real protobuf preflight')

    # Real protobuf unknown fields survive parsing/serialization but are lost by
    # JSON conversion. The optimizer must reject that loss BEFORE graph surgery.
    def varint(value):
        result = bytearray()
        while value > 127:
            result.append((value & 127) | 128)
            value >>= 7
        result.append(value)
        return bytes(result)
    unknown = Model_pb2.Model()
    unknown.ParseFromString(original.SerializeToString(deterministic=True) + varint(19000 << 3) + b'\x01')
    dropped = ParseDict(MessageToDict(unknown, preserving_proto_field_name=True), Model_pb2.Model())
    try:
        package.roundtrip_equal(unknown, dropped, 'deliberate unknown field')
    except ValueError:
        pass
    else:
        raise AssertionError('unknown protobuf field loss was accepted')

    output = args.out/'optimized'/args.tiny_package.name
    command = [sys.executable, '-I', '-B', str(root/'optimize-decoder-coreml.py'),
               '--package', str(args.tiny_package), '--out', str(output)]
    if args.compile:
        command += ['--compile-dir', str(args.out/'compiled')]
    done = run('optimize', command, 0)
    proof = json.loads(done.stdout)
    actual = read(output)
    golden = read(args.golden_package)
    golden_doc = MessageToDict(golden, preserving_proto_field_name=True)
    ud = golden_doc['description']['metadata']['userDefined']
    if ud['parler.decoder.graph_variant'] != 'diagnostic-original-heads-self-qkv':
        raise AssertionError('golden must be the measured diagnostic transformation')
    ud['parler.decoder.graph_variant'] = spec.MARKER
    package.roundtrip_equal(actual, ParseDict(golden_doc, Model_pb2.Model()), 'measured golden, stable marker only')
    actual_files = package.file_inventory(output)
    golden_spec = str(package.spec_path(args.golden_package).relative_to(args.golden_package))
    if set(actual_files) != set(golden_before) or any(actual_files[p] != golden_before[p]
                                                    for p in golden_before if p != golden_spec):
        raise AssertionError('golden manifest/weight payload bytes differ')
    run('duplicate-output-refused', command, 1)
    run('in-place-refused', [sys.executable, '-I', '-B', str(root/'optimize-decoder-coreml.py'),
                           '--package', str(args.tiny_package), '--out', str(args.tiny_package)], 1)
    corrupt = args.out/'corrupt-source'/args.tiny_package.name
    shutil.copytree(args.tiny_package, corrupt)
    package.spec_path(corrupt).write_bytes(b'\x80')
    bad = run('corrupt-protobuf-refused', [sys.executable, '-I', '-B', str(root/'optimize-decoder-coreml.py'),
              '--package', str(corrupt), '--out', str(args.out/'refused-corrupt'/corrupt.name)], 1)
    if 'invalid serialized Core ML model protobuf' not in bad.stderr or 'Traceback' in bad.stderr:
        raise AssertionError('corrupt protobuf must produce a clean CLI error')
    if package.file_inventory(output) != actual_files:
        raise AssertionError('refused command changed validated output')
    if package.file_inventory(args.tiny_package) != original_before or package.file_inventory(args.golden_package) != golden_before:
        raise AssertionError('original or golden package changed')
    receipt = {'status':'PASS', 'scope':'serialized tiny utility integration; no inference parity claim',
               'golden_difference_only_stable_marker':True, 'all_manifest_and_weight_bytes_golden_exact':True,
               'unknown_field_preflight_rejected':True, 'corrupt_protobuf_clean_cli_error':True, 'duplicate_and_in_place_refused':True,
               'original_and_golden_unchanged':True, 'optimizer_proof':proof}
    (args.out/'RECEIPT.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
