"""Copy, verify and publish a serialized package; never reconvert a model."""
import copy
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from . import inventory, spec, storage, verify, weights


def file_inventory(root):
    """Hash all files, rejecting package links and special files."""
    result = {}
    spec.need(root.is_dir() and not root.is_symlink(), 'package must be a directory, not a link')
    for path in sorted(root.rglob('*')):
        spec.need(not path.is_symlink(), 'package symlinks are unsupported')
        if path.is_dir():
            continue
        spec.need(path.is_file(), 'package special files are unsupported')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b''):
                digest.update(chunk)
        result[str(path.relative_to(root))] = (path.stat().st_size, digest.hexdigest())
    return result


def spec_path(package):
    manifest = spec.strict_json((package / 'Manifest.json').read_text(encoding='utf-8'))
    spec.need(manifest['fileFormatVersion'] == '1.0.0', 'unsupported package manifest version')
    entries = manifest['itemInfoEntries']
    spec.need(len(entries) == 2, 'expected exactly model and weights manifest entries')
    item = entries[manifest['rootModelIdentifier']]
    spec.need(item['name'] == 'model.mlmodel' and item['author'] == 'com.apple.CoreML',
              'unsupported root model manifest entry')
    for entry in entries.values():
        relative = Path(entry['path'])
        spec.need(not relative.is_absolute() and '..' not in relative.parts,
                  'unsafe package manifest path')
    relative = Path('Data') / item['path']
    path = package / relative
    spec.need(path.is_file() and path.suffix == '.mlmodel', 'missing root model specification')
    weight_entries = [v for k, v in entries.items() if k != manifest['rootModelIdentifier']]
    weight = weight_entries[0]
    spec.need(weight['name'] == 'weights' and weight['author'] == 'com.apple.CoreML' and
              Path('Data') / weight['path'] == relative.parent / 'weights',
              'unsupported weights manifest entry')
    return path


def roundtrip_equal(original, restored, stage):
    spec.need(original.SerializeToString(deterministic=True) ==
              restored.SerializeToString(deterministic=True),
              stage + ': unknown/default/presence protobuf fields changed')


def new_paths(package, out, compile_dir=None):
    """Validate paths before importing protobuf or creating outputs."""
    spec.need(sys.platform == 'darwin' or
              (sys.platform.startswith('linux') and hasattr(ctypes.CDLL(None), 'renameat2')),
              'atomic no-replace publication is unsupported on this platform')
    spec.need(not package.is_symlink(), 'source package must not be a symlink')
    package = package.resolve(strict=True)
    out = out.absolute()
    spec.need(package.is_dir() and package.suffix == '.mlpackage', 'expected existing .mlpackage')
    spec.need(out.name == package.name, 'output must retain the original package basename; use a new parent directory')
    targets = [out] + ([compile_dir.absolute()] if compile_dir is not None else [])
    for target in targets:
        spec.need(not os.path.lexists(target), 'output already exists: ' + str(target))
        resolved = target.resolve()
        spec.need(not resolved.is_relative_to(package) and not package.is_relative_to(resolved),
                  'source and outputs must not overlap')
    if len(targets) == 2:
        a, b = [p.resolve() for p in targets]
        spec.need(not a.is_relative_to(b) and not b.is_relative_to(a), 'outputs must not overlap')
    return package, out, targets[1] if len(targets) == 2 else None


def publish_new(source, destination):
    """Atomic directory rename that cannot replace even an empty raced output.

    Core ML targets macOS. Linux supports the same operation for stdlib tests.
    Unsupported platforms fail closed instead of falling back to os.rename.
    """
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == 'darwin':
        fn = libc.renamex_np
        fn.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        args = (os.fsencode(source), os.fsencode(destination), 0x00000004)  # RENAME_EXCL
    elif sys.platform.startswith('linux') and hasattr(libc, 'renameat2'):
        fn = libc.renameat2
        fn.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        args = (-100, os.fsencode(source), -100, os.fsencode(destination), 1)  # RENAME_NOREPLACE
    else:
        raise ValueError('atomic no-replace publication is unsupported on this platform')
    fn.restype = ctypes.c_int
    if fn(*args) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def validate_compiled(model_dir, candidate):
    """Compilation is separate from parity; check the emitted interface/identity."""
    doc = spec.strict_json((model_dir / 'metadata.json').read_text(encoding='utf-8'))
    spec.need(type(doc) is list and len(doc) == 1, 'expected one compiled model metadata entry')
    meta = doc[0]
    ud = candidate['description']['metadata']['userDefined']
    spec.need(meta['userDefinedMetadata'] == ud, 'compiler changed optimization identity or contract metadata')
    contract = spec.strict_json(ud['parler.decoder.contract'])
    for key, expected in [('inputSchema', contract['inputs']), ('outputSchema', contract['outputs']),
                          ('stateSchema', contract['states']['shapes'])]:
        entries = meta[key]
        shapes = {v['name']: spec.strict_json(v['shape']) for v in entries}
        spec.need(len(entries) == len(shapes) and shapes == expected and
                  all(v['dataType'] == 'Float16' for v in entries), 'compiled interface changed: ' + key)


def compile_copy(stage, destination, candidate, before_publish):
    """Compile in private scratch; publish a normal directory respecting umask."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.parler-qkv-compile-', dir=destination.parent) as scratch:
        # TemporaryDirectory itself is 0700. Publish this normal child instead,
        # preserving both a caller's usual 0755 and a restrictive umask.
        output = Path(scratch) / 'out'
        output.mkdir()
        command = ['/usr/bin/xcrun', 'coremlcompiler', 'compile', str(stage), str(output)]
        subprocess.run(command, check=True, stdout=sys.stderr, stderr=sys.stderr)
        model_dir = output / (stage.stem + '.mlmodelc')
        spec.need((model_dir / 'metadata.json').is_file(), 'compiler did not produce expected model')
        validate_compiled(model_dir, candidate)
        before_publish()
        publish_new(output, destination)
        return str(destination / model_dir.name)


def optimize(package, out, compile_dir=None):
    package, out, compile_dir = new_paths(Path(package), Path(out),
                                         Path(compile_dir) if compile_dir is not None else None)
    before = file_inventory(package)
    source_spec = spec_path(package)
    spec_rel = str(source_spec.relative_to(package))
    weight_rel = str((source_spec.parent / 'weights/weight.bin').relative_to(package))
    spec.need(set(before) == {'Manifest.json', spec_rel, weight_rel}, 'unexpected package files')

    # Deliberately delayed: CLI --help, path validation and stdlib tests need no
    # frameworks. Importing CT protobuf may initialize CT internals; neither
    # ct.convert nor MLModel is invoked, and no Torch forward is constructed.
    from coremltools.proto import Model_pb2
    from google.protobuf.json_format import MessageToDict, ParseDict
    from google.protobuf.message import DecodeError

    original = Model_pb2.Model()
    try:
        original.ParseFromString(source_spec.read_bytes())
    except DecodeError as error:
        raise ValueError('invalid serialized Core ML model protobuf') from error
    doc = MessageToDict(original, preserving_proto_field_name=True)
    roundtrip_equal(original, ParseDict(copy.deepcopy(doc), Model_pb2.Model()), 'baseline JSON preflight')
    contract, left = inventory.inventory(doc)
    plan = spec.make_plan(doc)
    weights.physical_inventory(source_spec.parent, left)  # before any copy/append

    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.parler-qkv-', dir=out.parent) as scratch:
        stage = Path(scratch) / package.name
        shutil.copytree(package, stage, copy_function=shutil.copy2)
        dest = spec_path(stage)
        offsets, records = storage.append(source_spec.parent / 'weights/weight.bin',
                                           dest.parent / 'weights/weight.bin', plan)
        candidate, _ = spec.patch(doc, offsets)
        model = ParseDict(candidate, Model_pb2.Model(), ignore_unknown_fields=False)
        restored = spec.verify(doc, MessageToDict(model, preserving_proto_field_name=True), offsets)
        roundtrip_equal(original, ParseDict(restored, Model_pb2.Model()), 'inverse')
        dest.write_bytes(model.SerializeToString(deterministic=True))
        saved = Model_pb2.Model()
        saved.ParseFromString(dest.read_bytes())
        saved_doc = MessageToDict(saved, preserving_proto_field_name=True)
        roundtrip_equal(saved, ParseDict(copy.deepcopy(saved_doc), Model_pb2.Model()), 'saved JSON preflight')
        roundtrip_equal(original, ParseDict(spec.verify(doc, saved_doc, offsets), Model_pb2.Model()), 'saved inverse')
        _, right = inventory.inventory(saved_doc, fused=True)
        result = verify.verify(source_spec.parent, dest.parent, left, right, contract)
        after = file_inventory(stage)
        spec.need(set(before) == set(after) and all(before[p] == after[p] for p in before
                  if p not in (spec_rel, weight_rel)), 'manifest or unrelated package file changed')
        spec.need(file_inventory(package) == before, 'source package changed during optimization')
        result.update(marker=spec.MARKER, package=str(out), source=str(package),
                      source_files=before, output_files=after, storage=records)
        if compile_dir is not None:
            def unchanged():
                spec.need(file_inventory(stage) == after, 'compiler modified source package copy')
                spec.need(file_inventory(package) == before, 'source changed during compilation')
            # Each path is atomic/new-only. A publication race can leave the
            # verified compiled output; it cannot overwrite either output.
            result['compiled'] = compile_copy(stage, compile_dir, saved_doc, unchanged)
        publish_new(stage, out)
    return result
