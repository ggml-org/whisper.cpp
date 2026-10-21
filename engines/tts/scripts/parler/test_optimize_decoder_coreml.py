#!/usr/bin/env python3
"""Stdlib-only fabricated graph/storage and new-output safety tests.

Run: python -I -B test_optimize_decoder_coreml.py
No Core ML, protobuf, Torch, real model reads, compilation or inference.
"""
import ast
import copy
import importlib.util
import json
import math
import os
import stat
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parent
path = ROOT / 'parler_qkv'
spec = importlib.util.spec_from_file_location('parler_qkv', path / '__init__.py',
                                             submodule_search_locations=[str(path)])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
from parler_qkv import spec as core, inventory as si, storage as ba, verify as bc, package

def typ(shape, dtype='FLOAT16'):
    t = {'dataType': dtype}
    if shape:
        t.update(rank=str(len(shape)), dimensions=[{'constant': {'size': str(v)}} for v in shape])
    return {'tensorType': t}

def node(kind, name, shape, args=None, attrs=None, dtype='FLOAT16'):
    return {'type': kind, 'inputs': args or {}, 'attributes': {'name': core.name_attr(name), **(attrs or {})}, 'outputs': [{'name': name, 'type': typ(shape, dtype)}]}

def bind(name):
    return core.binding(name)

def fixture():
    c = {
        'contract': 2, 'kind': 'parler-decoder-step', 'normalization': 'last-axis',
        'weights': 'int8+fp16-heads', 'mask_off': -30000, 'n_layer': 2, 'd_model': 64,
        'n_head': 2, 'head_dim': 32, 'ctx': 16, 'cross_context': 8, 'n_q': 2, 'vocab': 5,
        'inputs': {'hidden': [1,64,1,1], 'self_mask': [1,1,1,16], 'cross_mask': [1,1,1,8]},
        'outputs': {'logits': [2,5], 'new_k': [2,2,1,32], 'new_v': [2,2,1,32]},
        'states': {'k_layout': 'rows', 'layers_per_state': 2, 'self_rows': 16, 'cross_rows': 8,
                   'shapes': {k: [2,2,16 if k.startswith('self') else 8,32] for k in core.KINDS}}}

    ops = []
    payloads = {}
    cursor = 64

    def blob(shape, dtype):
        nonlocal cursor
        size = math.prod(shape) * {'INT8': 1, 'FLOAT16': 2}[dtype]
        off = cursor
        cursor = (off + 64 + size + 63) // 64 * 64
        payloads[off] = (dtype, bytes(((i + len(payloads) * 17) % 251 for i in range(size))))
        return {'type': typ(shape, dtype), 'blobFileValue': {'fileName': '@model_path/weights/weight.bin', 'offset': str(off)}}

    def const(n, vals):
        ops.append(core.vector_const(n, vals))
        return n

    def params(prefix):
        out = {}
        for key, vals in [('dilations', [1, 1]), ('groups', [1]), ('pad', [0, 0, 0, 0]), ('strides', [1, 1])]:
            out[key] = bind(const(prefix + '_' + key, vals))
        name=prefix+'_pad_type'
        value={'type': typ([], 'STRING'), 'immediateValue': {'tensor': {'strings': {'values': ['valid']}}}}
        ops.append(node('const', name, [], attrs={'val': value}, dtype='STRING'))
        out['pad_type']=bind(name)
        return out

    def weight(name, shape):
        op = node('constexpr_blockwise_shift_scale', name, shape, {'data': {'arguments': [{'value': blob(shape, 'INT8')}]}, 'scale': {'arguments': [{'value': blob([shape[0], 1, 1, 1], 'FLOAT16')}]}})
        ops.append(op)
        return name

    def raw(name, shape):
        ops.append(node('const', name, shape, attrs={'val': blob(shape, 'FLOAT16')}))
        return name
    d = c['d_model']
    H = c['n_head']
    D = c['head_dim']
    ops.append(node('read_state', 'fixture_read', c['states']['shapes']['self_k_00'], {'input': bind('self_k_00')}))
    for l in range(c['n_layer']):
        for k in ('q', 'k', 'v', 'o', 'cross_q', 'cross_o', 'up', 'down'):
            shape = [128, d, 1, 1] if k == 'up' else [d, 128, 1, 1] if k == 'down' else [d, d, 1, 1]
            w = weight(f'layers_{l}_{k}_weight_to_fp16_quantized', shape)
            par = params(f'p{l}_{k}')
            name = f'conv{l}_{k}'
            ops.append(node('conv', name, [1, shape[0], 1, 1], {'x': bind('hidden'), 'weight': bind(w), **par}))
            if k in ('q', 'k', 'v'):
                sh = const(f'headshape{l}_{k}', [1, H, D, 1])
                rh = f'head{l}_{k}'
                ops.append(node('reshape', rh, [1, H, D, 1], {'x': bind(name), 'shape': bind(sh)}))
                pm = const(f'perm{l}_{k}', [1, 0, 3, 2])
                ops.append(node('transpose', f'trans{l}_{k}', [H, 1, 1, D], {'x': bind(rh), 'perm': bind(pm)}))
        ops.append(node('mul', f'curmul{l}', [H, 1, 1, D], {'x': bind(f'trans{l}_q'), 'y': bind(f'trans{l}_k')}))
        ax = const(f'curaxes{l}', [2])
        ops.append(node('reduce_sum', f'cur{l}', [H, 1, 1, D], {'x': bind(f'curmul{l}'), 'axes': bind(ax)}))
        for sec in ('self', 'cross', 'ffn'):
            g = raw(f'layers_{l}_{sec}_norm_weight_to_fp16', [d])
            b = raw(f'layers_{l}_{sec}_norm_bias_to_fp16', [d])
            ops.append(node('layer_norm', f'norm{l}_{sec}', [1, d, 1, 1], {'x': bind('hidden'), 'gamma': bind(g), 'beta': bind(b)}))
    g = raw('out_norm_weight_to_fp16', [d])
    b = raw('out_norm_bias_to_fp16', [d])
    ops.append(node('layer_norm', 'finalnorm', [1, d, 1, 1], {'x': bind('hidden'), 'gamma': bind(g), 'beta': bind(b)}))
    w = raw('heads_weight_to_fp16', [c['n_q'] * c['vocab'], d, 1, 1])
    par = params('headparams')
    ops.append(node('conv', 'headsconv', [1, c['n_q'] * c['vocab'], 1, 1], {'x': bind('finalnorm'), 'weight': bind(w), **par}))
    for out, shape in c['outputs'].items():
        sh = const(out + 'shape', shape)
        ops.append(node('reshape', out, shape, {'x': bind('headsconv'), 'shape': bind(sh)}))
    inputs = [{'name': n, 'type': typ(s)} for n, s in c['inputs'].items()]
    states = [{'name': n, 'type': {'stateType': {'wrappedType': typ(s)}}} for n, s in c['states']['shapes'].items()]
    desc = {'input': [{'name': n, 'type': {'multiArrayType': {'shape': list(map(str, s)), 'dataType': 'FLOAT16'}}} for n, s in c['inputs'].items()], 'state': [{'name': n, 'type': {'stateType': {'arrayType': {'shape': list(map(str, s)), 'dataType': 'FLOAT16'}}}} for n, s in c['states']['shapes'].items()], 'output': [{'name': n, 'type': {'multiArrayType': {'shape': list(map(str, s)), 'dataType': 'FLOAT16'}}} for n, s in c['outputs'].items()], 'metadata': {'userDefined': {'parler.decoder.contract': json.dumps(c)}}}
    return ({'description': desc, 'mlProgram': {'functions': {'main': {'inputs': inputs + states, 'opset': 'CoreML8', 'block_specializations': {'CoreML8': {'outputs': ['logits', 'new_k', 'new_v'], 'operations': ops}}}}}}, payloads)

def write_fake(root, payloads):
    p = root / 'weights/weight.bin'
    p.parent.mkdir(parents=True)
    data = bytearray(64)
    struct.pack_into('<II', data, 0, len(payloads), 2)
    for off, (dt, body) in payloads.items():
        data.extend(b'\x00' * (off - len(data)))
        h = bytearray(64)
        struct.pack_into('<IIQQ', h, 0, 3735928559, 4 if dt == 'INT8' else 1, len(body), off + 64)
        data.extend(h)
        data.extend(body)
        data.extend(b'\x00' * (-len(data) % 64))
    p.write_bytes(data)
    return p

class SpecTests(unittest.TestCase):

    def setUp(self):
        self.a, _ = fixture()
        self.plan = core.make_plan(self.a)
        self.offsets = {'APPEND_' + str(l) + '_' + k: 100000 + l * 1000 + i * 100 for l in range(2) for i, k in enumerate(('data', 'scale'))}
        self.b, _ = core.patch(self.a, self.offsets)

    def bad(self, mut):
        x = copy.deepcopy(self.b)
        mut(x)
        with self.assertRaises(ValueError):
            core.verify(self.a, x, self.offsets)

    def ops(self, d):
        return core.scan(d)[2]

    def test_positive_inverse(self):
        self.assertEqual(core.verify(self.a, self.b, self.offsets), self.a)

    def test_complete_inventory(self):
        self.assertEqual(len(si.inventory(self.a)[1]), 31)
        self.assertEqual(len(si.inventory(self.b, True)[1]), 27)

    def test_original_heads_identical(self):
        a = [o for o in self.ops(self.a) if o['type'] in ('transpose', 'reduce_sum', 'mul')]
        b = [o for o in self.ops(self.b) if o['type'] in ('transpose', 'reduce_sum', 'mul')]
        self.assertEqual(a, b)

    def test_cur_axis(self):
        self.bad(lambda d: self.ops(d)[next((i for i, o in enumerate(self.ops(d)) if core.named(o) == 'curaxes0'))]['attributes']['val']['immediateValue']['tensor']['ints'].update(values=[3]))

    def test_head_permutation(self):
        self.bad(lambda d: next((o for o in self.ops(d) if core.named(o) == 'perm0_q'))['attributes']['val']['immediateValue']['tensor']['ints'].update(values=[0, 1, 3, 2]))

    def test_bounds(self):
        self.bad(lambda d: next((o for o in self.ops(d) if core.named(o) == 'qkv_slice_0_1_begin'))['attributes']['val']['immediateValue']['tensor']['ints'].update(values=[0, 0, 0, 0]))

    def test_qkv_order(self):
        self.bad(lambda d: next((o for o in self.ops(d) if core.named(o) == 'conv0_k'))['inputs'].update(x=bind('hidden')))

    def test_metadata(self):
        self.bad(lambda d: d['description']['metadata']['userDefined'].update(other='x'))

    def test_description(self):
        self.bad(lambda d: d['description'].update(shortDescription='changed'))

    def test_extra_function(self):
        self.bad(lambda d: d['mlProgram']['functions'].update(extra=copy.deepcopy(d['mlProgram']['functions']['main'])))

    def test_extra_op(self):
        self.bad(lambda d: self.ops(d).append(core.vector_const('extra', [1])))

    def test_output_dangling(self):
        self.bad(lambda d: d['mlProgram']['functions']['main']['block_specializations']['CoreML8'].update(outputs=['absent', 'new_k', 'new_v']))

    def test_unknown_op(self):
        self.bad(lambda d: self.ops(d)[0].update(type='custom'))

    def test_wrong_dtype(self):
        self.bad(lambda d: next((o for o in self.ops(d) if core.named(o) == 'qkv_fused_activation_0'))['outputs'][0]['type']['tensorType'].update(dataType='FLOAT32'))

    def test_different_input_rejected_before_mutation(self):
        next((o for o in self.ops(self.a) if core.named(o) == 'conv0_k'))['inputs']['x'] = bind('self_mask')
        with self.assertRaises(ValueError):
            core.make_plan(self.a)

    def test_different_conv_parameter(self):
        next((o for o in self.ops(self.a) if core.named(o) == 'p0_k_strides'))['attributes']['val']['immediateValue']['tensor']['ints']['values'] = [2, 1]
        with self.assertRaises(ValueError):
            core.make_plan(self.a)

    def test_original_reshape_not_rewritten(self):
        next((o for o in self.ops(self.a) if core.named(o) == 'headshape0_q'))['attributes']['val']['immediateValue']['tensor']['ints']['values'] = [2, 1, 1, 32]
        with self.assertRaises(ValueError):
            core.make_plan(self.a)

    def test_ast(self):
        for p in list(ROOT.glob('*.py')) + list((ROOT/'parler_qkv').glob('*.py')):
            ast.parse(p.read_text(encoding='utf-8'), filename=str(p))

class ByteTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.left = self.root / 'left'
        self.right = self.root / 'right'
        self.a, payloads = fixture()
        self.source = write_fake(self.left, payloads)
        shutil.copytree(self.left, self.right)
        self.dest = self.right / 'weights/weight.bin'
        self.plan = core.make_plan(self.a)
        self.offsets, self.storage = ba.append(self.source, self.dest, self.plan)
        self.b, _ = core.patch(self.a, self.offsets)
        self.c, self.li = si.inventory(self.a)
        _, self.ri = si.inventory(self.b, True)

    def check(self):
        return bc.verify(self.left, self.right, self.li, self.ri, self.c)

    def flip(self, offset):
        with self.dest.open('r+b') as f:
            f.seek(offset)
            v = f.read(1)
            f.seek(offset)
            f.write(bytes([v[0] ^ 1]))

    def test_positive_complete_inactive(self):
        r = self.check()
        self.assertEqual((r['baseline_learned'], r['active_candidate_learned'], r['physical_candidate_records'], r['retained_inactive_records']), (31, 27, 51, 12))
        self.assertTrue(ba.prefix_equal(self.source, self.dest))

    def test_code_byte(self):
        self.flip(self.offsets['APPEND_0_data'] + 64)
        with self.assertRaises(ValueError):
            self.check()

    def test_scale_byte(self):
        self.flip(self.offsets['APPEND_0_scale'] + 64)
        with self.assertRaises(ValueError):
            self.check()

    def test_qkv_order(self):
        off = self.offsets['APPEND_0_data'] + 64
        n = 64 * 64
        with self.dest.open('r+b') as f:
            f.seek(off)
            b = f.read(3 * n)
            f.seek(off)
            f.write(b[n:2 * n] + b[:n] + b[2 * n:])
        with self.assertRaises(ValueError):
            self.check()

    def test_original_inactive_changed(self):
        self.flip(128)
        with self.assertRaises(ValueError):
            self.check()

    def test_extra_record(self):
        with self.dest.open('r+b') as f:
            f.write(struct.pack('<I', 52))
        with self.assertRaises(ValueError):
            self.check()

    def test_unaccounted_trailing(self):
        with self.dest.open('ab') as f:
            f.write(b'x' * 65)
        with self.assertRaises(ValueError):
            self.check()

    def test_missing_scale(self):
        self.ri['layers_0_qkv_weight_to_fp16_quantized']['blobs'].pop()
        with self.assertRaises((ValueError, IndexError)):
            self.check()

    def test_wrong_scale_offset(self):
        self.ri['layers_0_qkv_weight_to_fp16_quantized']['blobs'][1]['offset'] = self.offsets['APPEND_1_scale']
        with self.assertRaises(ValueError):
            self.check()

    def test_nonqkv_shifted(self):
        self.ri['layers_0_o_weight_to_fp16_quantized']['blobs'][0]['offset'] += 64
        with self.assertRaises(ValueError):
            self.check()

    def test_alias_original_forbidden(self):
        with self.assertRaises(ValueError):
            ba.append(self.source, self.source, self.plan)

class CompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.doc, _ = fixture()

    def reject_contract(self, change):
        ud = self.doc['description']['metadata']['userDefined']
        contract = json.loads(ud['parler.decoder.contract'])
        change(contract)
        ud['parler.decoder.contract'] = json.dumps(contract)
        with self.assertRaises(ValueError):
            core.make_plan(self.doc)

    def test_grouped_states(self):
        self.reject_contract(lambda c: c['states'].update(layers_per_state=1))

    def test_column_layout(self):
        self.reject_contract(lambda c: c['states'].update(k_layout='cols'))

    def test_weight_mode(self):
        self.reject_contract(lambda c: c.update(weights='int4+fp16-heads'))

    def test_unmeasured_geometry(self):
        self.reject_contract(lambda c: c.update(ctx=128))

    def test_bad_contract_interface(self):
        self.reject_contract(lambda c: c['inputs'].update(hidden=[1,65,1,1]))

    def mini_metadata(self):
        # Only schema dictionaries: no full-model tensors or payload allocation.
        doc = copy.deepcopy(self.doc)
        c = json.loads(doc['description']['metadata']['userDefined']['parler.decoder.contract'])
        c.update(n_layer=24, d_model=1024, n_head=16, head_dim=64,
                 ctx=512, cross_context=128, n_q=9, vocab=1088)
        c['inputs'] = {'hidden':[1,1024,1,1], 'self_mask':[1,1,1,512], 'cross_mask':[1,1,1,128]}
        c['outputs'] = {'logits':[9,1088], 'new_k':[24,16,1,64], 'new_v':[24,16,1,64]}
        c['states'].update(layers_per_state=24, self_rows=512, cross_rows=128,
                           shapes={k:[24,16,512 if k.startswith('self') else 128,64] for k in core.KINDS})
        for field, shapes in [('input',c['inputs']), ('output',c['outputs']), ('state',c['states']['shapes'])]:
            for feature in doc['description'][field]:
                array = feature['type']['stateType']['arrayType'] if field == 'state' else feature['type']['multiArrayType']
                array['shape'] = list(map(str, shapes[feature['name']]))
        doc['description']['metadata']['userDefined']['parler.decoder.contract'] = json.dumps(c)
        return doc

    def test_mini_metadata_accepted(self):
        self.assertEqual(core.metadata(self.mini_metadata())['d_model'], 1024)

    def test_mini_wrong_dimension_rejected(self):
        doc = self.mini_metadata()
        doc['description']['state'][0]['type']['stateType']['arrayType']['shape'][-1] = '32'
        with self.assertRaisesRegex(ValueError, 'description/contract'):
            core.metadata(doc)

    def test_duplicate_fusion(self):
        self.doc['description']['metadata']['userDefined']['parler.decoder.graph_variant'] = core.MARKER
        with self.assertRaises(ValueError):
            core.make_plan(self.doc)

    def test_duplicate_contract_key(self):
        self.doc['description']['metadata']['userDefined']['parler.decoder.contract'] = '{"contract":2,"contract":2}'
        with self.assertRaisesRegex(ValueError, 'duplicate JSON'):
            core.make_plan(self.doc)

    def test_unsupported_conv_groups(self):
        for op in core.scan(self.doc)[2]:
            if core.named(op).endswith('_groups'):
                op['attributes']['val']['immediateValue']['tensor']['ints']['values'] = [2]
        with self.assertRaisesRegex(ValueError, 'groups'):
            core.make_plan(self.doc)

    def test_unsupported_padding(self):
        for op in core.scan(self.doc)[2]:
            if core.named(op).endswith('_pad_type'):
                op['attributes']['val']['immediateValue']['tensor']['strings']['values'] = ['custom']
        with self.assertRaisesRegex(ValueError, 'padding'):
            core.make_plan(self.doc)

    def test_external_weight(self):
        value = core.all_blob_values(self.doc)[0]
        value['blobFileValue']['fileName'] = '/tmp/external.bin'
        with self.assertRaises(ValueError):
            si.inventory(self.doc)

    def test_unknown_quantizer_argument(self):
        op = next(o for o in core.scan(self.doc)[2] if o['type'] == 'constexpr_blockwise_shift_scale')
        op['inputs']['offset'] = copy.deepcopy(op['inputs']['scale'])
        with self.assertRaises(ValueError):
            core.make_plan(self.doc)


class OutputTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root/'source'/'decoder.mlpackage'
        self.source.mkdir(parents=True)
        self.out = self.root/'new'/'decoder.mlpackage'

    def test_in_place(self):
        with self.assertRaises(ValueError):
            package.new_paths(self.source, self.source)

    def test_existing_empty_output(self):
        self.out.mkdir(parents=True)
        with self.assertRaises(ValueError):
            package.new_paths(self.source, self.out)

    def test_basename_preserved(self):
        with self.assertRaisesRegex(ValueError, 'basename'):
            package.new_paths(self.source, self.out.with_name('renamed.mlpackage'))

    def test_output_overlap(self):
        with self.assertRaises(ValueError):
            package.new_paths(self.source, self.out, self.out/'compiled')

    def test_source_symlink(self):
        link = self.root/'link.mlpackage'
        link.symlink_to(self.source)
        with self.assertRaises(ValueError):
            package.new_paths(link, self.out)

    def test_package_file_symlink(self):
        (self.source/'weights.bin').symlink_to(self.root/'missing')
        with self.assertRaises(ValueError):
            package.file_inventory(self.source)

    def test_publish_atomic_new(self):
        staging = self.root/'staging'
        staging.mkdir()
        (staging/'complete').write_text('verified', encoding='utf-8')
        destination = self.root/'published'
        package.publish_new(staging, destination)
        self.assertFalse(staging.exists())
        self.assertEqual((destination/'complete').read_text(encoding='utf-8'), 'verified')

    def test_publish_raced_empty_directory(self):
        staging, destination = self.root/'staging', self.root/'raced'
        staging.mkdir()
        destination.mkdir()
        with self.assertRaises(OSError):
            package.publish_new(staging, destination)
        self.assertTrue(staging.is_dir())
        self.assertTrue(destination.is_dir())

    def test_publish_raced_symlink(self):
        staging, destination = self.root/'staging', self.root/'raced'
        staging.mkdir()
        destination.symlink_to(self.source)
        with self.assertRaises(OSError):
            package.publish_new(staging, destination)
        self.assertTrue(destination.is_symlink())

    def test_unsafe_manifest(self):
        manifest = {'fileFormatVersion':'1.0.0','rootModelIdentifier':'model',
                    'itemInfoEntries':{'model':{'name':'model.mlmodel','author':'com.apple.CoreML','path':'../../model.mlmodel'},
                                       'weights':{'name':'weights','author':'com.apple.CoreML','path':'weights'}}}
        (self.source/'Manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
        with self.assertRaises(ValueError):
            package.spec_path(self.source)

    def test_paths_fail_before_protobuf_import(self):
        with mock.patch.dict(sys.modules, {'coremltools':None}):
            with self.assertRaisesRegex(ValueError, 'already exists'):
                package.optimize(self.source, self.source)

    def test_unsupported_platform_fails_before_import(self):
        with mock.patch.object(package.sys, 'platform', 'win32'):
            with self.assertRaisesRegex(ValueError, 'atomic no-replace'):
                package.new_paths(self.source, self.out)

    def fake_compiler(self, argv, **kwargs):
        output = Path(argv[-1])
        self.assertEqual(output.name, 'out')
        model = output/(self.source.stem+'.mlmodelc')
        model.mkdir()
        doc, _ = fixture()
        c = core.metadata(doc)
        metadata = {'userDefinedMetadata':doc['description']['metadata']['userDefined']}
        for key, shapes in [('inputSchema',c['inputs']), ('outputSchema',c['outputs']),
                            ('stateSchema',c['states']['shapes'])]:
            metadata[key] = [{'name':n, 'shape':json.dumps(shape), 'dataType':'Float16'} for n,shape in shapes.items()]
        (model/'metadata.json').write_text(json.dumps([metadata]), encoding='utf-8')

    def test_compiled_directory_respects_umask(self):
        doc, _ = fixture()
        for mask, expected in [(0o022,0o755),(0o077,0o700)]:
            with self.subTest(umask=oct(mask)):
                destination = self.root/('compiled-'+str(mask))
                old_mask = os.umask(mask)
                try:
                    with mock.patch.object(package.subprocess, 'run', side_effect=self.fake_compiler):
                        package.compile_copy(self.source, destination, doc, lambda: None)
                finally:
                    os.umask(old_mask)
                self.assertEqual(stat.S_IMODE(destination.stat().st_mode), expected)

    def test_compile_failure_no_publication(self):
        doc, _ = fixture()
        destination = self.root/'compiled'
        with mock.patch.object(package.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, ['xcrun'])):
            with self.assertRaises(subprocess.CalledProcessError):
                package.compile_copy(self.source, destination, doc, lambda: None)
        self.assertFalse(destination.exists())

    def test_help_without_framework_imports(self):
        # -I excludes the current directory; --help still needs only stdlib.
        done = subprocess.run([sys.executable, '-I', '-B', str(ROOT/'optimize-decoder-coreml.py'), '--help'],
                              text=True, capture_output=True)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn('20.2%', done.stdout)
        self.assertIn('--compile-dir', done.stdout)


if __name__ == '__main__':
    unittest.main()
