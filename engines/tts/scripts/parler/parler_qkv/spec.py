"""Serialized original-head self-attention QKV fusion (no framework imports).

The rewrite is the measured postconversion transformation: concatenate Q/K/V
output channels, then slice into the ORIGINAL projection outputs. All original
head permutations, current-token reductions, cross attention and interface stay
unchanged. The inverse rejects any other change. Only the measured tiny and
mini contract-2 geometries are supported; support is deliberately fail closed.
"""
import copy
import json
MARKER = 'original-heads-self-qkv-v1'
KINDS = ('self_k_00', 'self_v_00', 'cross_k_00', 'cross_v_00')
ALLOWED_OPS = {'const', 'read_state', 'reshape', 'layer_norm', 'conv', 'constexpr_blockwise_shift_scale', 'matmul', 'gelu', 'concat', 'add', 'softmax', 'slice_by_index', 'transpose', 'reduce_sum', 'mul'}

def need(ok, why):
    if not ok:
        raise ValueError(why)

def at(doc, path):
    for key in path:
        doc = doc[key]
    return doc

def put(doc, path, value):
    at(doc, path[:-1])[path[-1]] = copy.deepcopy(value)

def tensor_type(value_type):
    need(set(value_type) == {'tensorType'}, 'expected tensor type only')
    t = value_type['tensorType']
    need(int(t.get('rank', 0)) == len(t.get('dimensions', [])), 'tensor rank mismatch')
    need(all((set(d) == {'constant'} and set(d['constant']) == {'size'} for d in t.get('dimensions', []))), 'symbolic/unknown tensor dimension')
    return t

def dims(vt):
    return [int(d['constant']['size']) for d in tensor_type(vt).get('dimensions', [])]

def named(op):
    need(len(op.get('outputs', [])) == 1, 'single output required for selected node')
    return op['outputs'][0]['name']

def ref(op, key):
    args = op['inputs'][key]['arguments']
    need(len(args) == 1 and set(args[0]) == {'name'}, 'selected binding must be one SSA name')
    return args[0]['name']

def const_value(op, kind):
    need(op['type'] == 'const' and set(op.get('attributes', {})) == {'name', 'val'}, 'selected constant schema')
    v = op['attributes']['val']
    need(set(v) == {'type', 'immediateValue'}, 'immediate selected constant required')
    data = v['immediateValue']['tensor']
    need(set(data) == {kind}, 'selected constant kind')
    need(set(data[kind]) == {'values'}, 'constant values schema')
    return data[kind]['values']

def scan(spec):
    need('mlProgram' in spec, 'MLProgram required')
    prog = spec['mlProgram']
    need(set(prog['functions']) == {'main'}, 'exactly one main function required')
    fn = prog['functions']['main']
    need(len(fn['block_specializations']) == 1, 'exactly one block specialization required')
    key = next(iter(fn['block_specializations']))
    need(fn['opset'] == key, 'opset/block mismatch')
    block = fn['block_specializations'][key]
    need(not block.get('inputs'), 'unexpected block inputs/state signature')
    need(block['outputs'] == ['logits', 'new_k', 'new_v'], 'exact block output order required')
    desc = spec['description']
    need(not desc.get('functions'), 'unexpected description functions')
    need({x['name'] for x in desc['state']} == set(KINDS) and len(desc['state']) == 4, 'description states')
    need({x['name'] for x in desc['input']} == {'hidden', 'self_mask', 'cross_mask'} and len(desc['input']) == 3, 'description inputs')
    need([x['name'] for x in desc['output']] == block['outputs'], 'description/block outputs mismatch')
    external = {x['name']: x for x in fn['inputs']}
    need(len(external) == len(fn['inputs']) and set(external) == set(KINDS) | {'hidden', 'self_mask', 'cross_mask'}, 'function complete input/state signature')
    desc_state = {x['name']: x for x in desc['state']}
    state_paths = []
    for i, x in enumerate(fn['inputs']):
        if x['name'] in KINDS:
            need(set(x['type']) == {'stateType'} and set(x['type']['stateType']) == {'wrappedType'}, 'state signature schema')
            vt = x['type']['stateType']['wrappedType']
            t = tensor_type(vt)
            declared = desc_state[x['name']]['type']['stateType']['arrayType']
            need(dims(vt) == [int(y) for y in declared['shape']] and declared['dataType'] == 'FLOAT16' and (t['dataType'] == 'FLOAT16'), 'function/description state dtype/geometry')
            state_paths.append(('mlProgram', 'functions', 'main', 'inputs', i, 'type', 'stateType'))
        else:
            t = tensor_type(x['type'])
            d = next((v for v in desc['input'] if v['name'] == x['name']))['type']['multiArrayType']
            need(t['dataType'] == 'FLOAT16' and d['dataType'] == 'FLOAT16' and (dims(x['type']) == [int(v) for v in d['shape']]), 'input signature geometry/dtype')
    for i, x in enumerate(desc['state']):
        state_paths.append(('description', 'state', i, 'type', 'stateType'))

    def state_locations(obj, path=()):
        found = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == 'stateType':
                    found.append(path + (k,))
                found.extend(state_locations(v, path + (k,)))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                found.extend(state_locations(v, path + (i,)))
        return found
    need(set(state_locations(spec)) == set(state_paths), 'unexpected state/type signature elsewhere')
    ops = block['operations']
    producers = {}
    uses = {}
    seen = set(external)
    for i, op in enumerate(ops):
        need(op['type'] in ALLOWED_OPS, 'unknown op ' + op['type'])
        need(not op.get('blocks'), 'nested operation block forbidden')
        for k, arg in op.get('inputs', {}).items():
            for j, b in enumerate(arg.get('arguments', [])):
                need(set(b) in ({'name'}, {'value'}), 'unknown binding')
                if 'name' in b:
                    need(b['name'] in seen, 'dangling or non-topological op input')
                    uses.setdefault(b['name'], []).append((i, k, j))
        for out in op.get('outputs', []):
            name = out['name']
            need(name not in seen, 'SSA duplicate')
            seen.add(name)
            producers[name] = i
    need(all((o in producers for o in block['outputs'])), 'dangling block output')
    for feature in desc['output']:
        vt = ops[producers[feature['name']]]['outputs'][0]['type']
        a = feature['type']['multiArrayType']
        need(dims(vt) == [int(v) for v in a['shape']] and tensor_type(vt)['dataType'] == a['dataType'] == 'FLOAT16', 'description/output producer signature')
    path = ('mlProgram', 'functions', 'main', 'block_specializations', key, 'operations')
    return (fn, block, ops, external, producers, uses, path)

def set_shape(vt, shape):
    out = copy.deepcopy(vt)
    t = tensor_type(out)
    need(len(t.get('dimensions', [])) == len(shape), 'rank change forbidden')
    for d, v in zip(t['dimensions'], shape):
        d['constant']['size'] = str(v)
    return out

def name_attr(name):
    return {'type': {'tensorType': {'dataType': 'STRING'}}, 'immediateValue': {'tensor': {'strings': {'values': [name]}}}}

def rename(op, name):
    op = copy.deepcopy(op)
    op['outputs'][0]['name'] = name
    op['attributes']['name'] = name_attr(name)
    return op

def binding(name):
    return {'arguments': [{'name': name}]}

def vector_const(name, values):
    vt = {'tensorType': {'dataType': 'INT32', 'rank': '1', 'dimensions': [{'constant': {'size': str(len(values))}}]}}
    return {'type': 'const', 'attributes': {'name': name_attr(name), 'val': {'type': copy.deepcopy(vt), 'immediateValue': {'tensor': {'ints': {'values': values}}}}}, 'outputs': [{'name': name, 'type': vt}]}

def all_blob_values(doc):
    out = []

    def walk(v):
        if isinstance(v, dict):
            if 'blobFileValue' in v:
                need(set(v) == {'type', 'blobFileValue'}, 'unsupported BLOB value')
                out.append(v)
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
    walk(doc)
    return out

def blob(op, key):
    need(op['type'] == 'constexpr_blockwise_shift_scale' and
         set(op['inputs']) == {'data', 'scale'} and set(op['attributes']) == {'name'} and
         len(op['outputs']) == 1, 'only perchannel shift-scale quantizer')
    a = op['inputs'][key]['arguments']
    need(len(a) == 1 and set(a[0]) == {'value'}, 'literal quantizer blob required')
    v = a[0]['value']
    need(set(v) == {'type', 'blobFileValue'} and set(v['blobFileValue']) == {'fileName', 'offset'}, 'blob schema')
    need(v['blobFileValue']['fileName'] == '@model_path/weights/weight.bin', 'single reviewed weightfile only')
    return v

def metadata(baseline):
    ud = baseline['description']['metadata']['userDefined']
    need('parler.decoder.graph_variant' not in ud, 'baseline must be original')
    c = strict_json(ud['parler.decoder.contract'])
    L, d, H, D, T, C, Q, V = (c[k] for k in ('n_layer', 'd_model', 'n_head', 'head_dim', 'ctx', 'cross_context', 'n_q', 'vocab'))
    need((L, d, H, D, T, C, Q, V) in ((2, 64, 2, 32, 16, 8, 2, 5), (24, 1024, 16, 64, 512, 128, 9, 1088)), 'exact reviewed tiny/full geometry')
    need(type(c['contract']) is int and c['contract'] == 2 and
         c['kind'] == 'parler-decoder-step' and c['normalization'] == 'last-axis' and
         c['weights'] == 'int8+fp16-heads' and c['mask_off'] == -30000,
         'original per-channel int8 contract-2 policy required')
    need(c['states']['k_layout'] == 'rows' and
         type(c['states']['layers_per_state']) is int and c['states']['layers_per_state'] == L and
         'v_layout' not in c['states'], 'all layers per state with original row layout required')
    need(not any((k.startswith('diagnostic_') for k in c)), 'original nondiagnostic contract required')
    need(all((type(c[k]) is int for k in ('n_layer', 'd_model', 'n_head', 'head_dim', 'ctx', 'cross_context', 'n_q', 'vocab'))), 'integer contract dimensions required')
    need(c['states']['self_rows'] == T and c['states']['cross_rows'] == C, 'contract state rows')
    states = {k: [L, H, T if k.startswith('self_') else C, D] for k in KINDS}
    inputs = {'hidden': [1, d, 1, 1], 'self_mask': [1, 1, 1, T], 'cross_mask': [1, 1, 1, C]}
    outputs = {'logits': [Q, V], 'new_k': [L, H, 1, D], 'new_v': [L, H, 1, D]}
    need(c['states']['shapes'] == states and c['inputs'] == inputs and (c['outputs'] == outputs), 'contract interface geometry')
    desc = baseline['description']
    for field, shapes in [('input', inputs), ('output', outputs), ('state', states)]:
        actual = {f['name']: [int(x) for x in (f['type']['stateType']['arrayType'] if field == 'state' else f['type']['multiArrayType'])['shape']] for f in desc[field]}
        need(actual == shapes, 'description/contract mismatch: ' + field)
    return c

def strict_json(text):

    def pairs(items):
        result = {}
        for k, v in items:
            need(k not in result, 'duplicate JSON key: ' + k)
            result[k] = v
        return result
    return json.loads(text, object_pairs_hook=pairs, parse_constant=lambda x: need(False, 'nonfinite JSON'))

def make_plan(baseline):
    fn, block, ops, external, prod, uses, path = scan(baseline)
    c = metadata(baseline)
    L, d, H, D = (c[k] for k in ('n_layer', 'd_model', 'n_head', 'head_dim'))
    remove = set()
    replacements = {}
    groups = []
    append = []
    for layer in range(L):
        weights = []
        convs = []
        for letter in ('q', 'k', 'v'):
            name = f'layers_{layer}_{letter}_weight_to_fp16_quantized'
            need(name in prod, 'missing exact learned role ' + name)
            wi = prod[name]
            w = ops[wi]
            need(dims(w['outputs'][0]['type']) == [d, d, 1, 1] and tensor_type(w['outputs'][0]['type'])['dataType'] == 'FLOAT16', 'projection weight geometry')
            data, scale = (blob(w, 'data'), blob(w, 'scale'))
            need(dims(data['type']) == [d, d, 1, 1] and tensor_type(data['type'])['dataType'] == 'INT8' and (dims(scale['type']) == [d, 1, 1, 1]) and (tensor_type(scale['type'])['dataType'] == 'FLOAT16'), 'perchannel code/scale geometry')
            us = uses.get(name, [])
            need(len(us) == 1 and us[0][1:] == ('weight', 0), 'learned projection must have sole conv consumer')
            ci = us[0][0]
            cv = ops[ci]
            need(cv['type'] == 'conv' and set(cv['inputs']) == {'x', 'weight', 'dilations', 'groups', 'pad', 'pad_type', 'strides'} and (set(cv['attributes']) == {'name'}), 'exact biasfree conv schema')
            need(dims(cv['outputs'][0]['type']) == [1, d, 1, 1] and
                 tensor_type(cv['outputs'][0]['type'])['dataType'] == 'FLOAT16', 'projection output geometry/dtype')
            consumers = uses.get(named(cv), [])
            need(len(consumers) == 1 and consumers[0][1:] == ('x', 0), 'projection original head input must be soleconsumer')
            head = ops[consumers[0][0]]
            need(head['type'] == 'reshape' and dims(head['outputs'][0]['type']) == [1, H, D, 1], 'original heads_split reshape path required')
            need(const_value(ops[prod[ref(head, 'shape')]], 'ints') == [1, H, D, 1], 'original heads shape constant')
            weights.append(wi)
            convs.append(ci)
        need(weights[0] < convs[0] < weights[1] < convs[1] < weights[2] < convs[2], 'reviewed original projection order')
        qconv = ops[convs[0]]
        x = ref(qconv, 'x')
        need(all((ref(ops[i], 'x') == x for i in convs)), 'Q/K/V must share exact normalized input')
        param_keys = ('dilations', 'groups', 'pad', 'pad_type', 'strides')
        qparams = {}
        for key in param_keys:
            qi = prod[ref(qconv, key)]
            qo = ops[qi]
            need(qo['type'] == 'const', 'conv parameter must be literal const')
            qparams[key] = qo['attributes']['val']
            for ci in convs[1:]:
                pi = prod[ref(ops[ci], key)]
                po = ops[pi]
                need(po['type'] == 'const' and po['attributes']['val'] == qparams[key], 'conv attributes must have identical typed values')
                if uses[named(po)] == [(ci, key, 0)]:
                    remove.add(pi)
        for key, values in [('dilations', [1, 1]), ('groups', [1]), ('pad', [0, 0, 0, 0]), ('strides', [1, 1])]:
            need(const_value(ops[prod[ref(qconv, key)]], 'ints') == values, 'unsupported conv ' + key)
        need(const_value(ops[prod[ref(qconv, 'pad_type')]], 'strings') == ['valid'], 'unsupported conv padding')
        qname = f'layers_{layer}_qkv_weight_to_fp16_quantized'
        convname = f'qkv_fused_activation_{layer}'
        need(qname not in prod and convname not in prod, 'reserved fusion name collision')
        # Copy Q's producer and conv verbatim except name/output-channel shape.
        # Learned Q already contains query scaling: never rescale or dequantize.
        fused = rename(ops[weights[0]], qname)
        fused['outputs'][0]['type'] = set_shape(fused['outputs'][0]['type'], [3 * d, d, 1, 1])
        for key, shape in [('data', [3 * d, d, 1, 1]), ('scale', [3 * d, 1, 1, 1])]:
            target = blob(fused, key)
            target['type'] = set_shape(target['type'], shape)
            # Filled only after raw payload records are appended to the copy.
            target['blobFileValue']['offset'] = 'APPEND_' + str(layer) + '_' + key
            append.append({'layer': layer, 'kind': key, 'sources': [copy.deepcopy(blob(ops[wi], key)) for wi in weights], 'target_shape': shape, 'target_dtype': 'INT8' if key == 'data' else 'FLOAT16'})
        replacements[weights[0]] = [fused]
        remove.update(weights[1:])
        cv = rename(qconv, convname)
        cv['inputs']['weight'] = binding(qname)
        cv['outputs'][0]['type'] = set_shape(cv['outputs'][0]['type'], [1, 3 * d, 1, 1])
        for part, ci in enumerate(convs):
            original = ops[ci]
            prefix = f'qkv_slice_{layer}_{part}'
            begin = vector_const(prefix + '_begin', [0, part * d, 0, 0])
            end = vector_const(prefix + '_end', [1, (part + 1) * d, 1, 1])
            need(not any((named(v) in prod for v in (begin, end))), 'slice constant name collision')
            sl = {'type': 'slice_by_index', 'inputs': {'x': binding(convname), 'begin': binding(named(begin)), 'end': binding(named(end))}, 'attributes': {'name': copy.deepcopy(original['attributes']['name'])}, 'outputs': copy.deepcopy(original['outputs'])}
            # Preserve each original projection SSA output and all head/cur ops.
            replacements[ci] = ([cv] if part == 0 else []) + [begin, end, sl]
        groups.append({'layer': layer, 'old_weights': weights, 'old_convs': convs, 'weight_names': [named(ops[i]) for i in weights], 'original_output_names': [named(ops[i]) for i in convs], 'fused_weight': qname, 'fused_activation': convname, 'shared_input': x})
    need(not set(replacements) & remove, 'rewrite/remove conflict')
    return {'operations_path': list(path), 'remove': sorted(remove), 'replace': replacements, 'groups': groups, 'append_payloads': append, 'geometry': {'L': L, 'd': d, 'H': H, 'D': D}, 'marker': MARKER}

def patch(baseline, offsets):
    p = make_plan(baseline)
    out = copy.deepcopy(baseline)
    old = at(out, p['operations_path'])
    new = []
    for i, op in enumerate(old):
        if i in p['remove']:
            continue
        new.extend(copy.deepcopy(p['replace'].get(i, [op])))
    for op in new:
        for v in all_blob_values(op):
            off = v['blobFileValue']['offset']
            if isinstance(off, str) and off.startswith('APPEND_'):
                need(off in offsets, 'missing append offset')
                v['blobFileValue']['offset'] = str(offsets[off])
    put(out, tuple(p['operations_path']), new)
    ud = out['description']['metadata']['userDefined']
    ud['parler.decoder.graph_variant'] = MARKER
    scan(out)
    verify(baseline, out, offsets)
    return (out, p)

def verify(baseline, candidate, offsets):
    """Exact inverse: consume approved rewrite, restore original ops.
    Every original nonfusion field/op/order/ref/shape, including allhead/cur math,
    must then equal baseline; protobufwrapper compares to ORIGINALparsedbytes.
    """
    p = make_plan(baseline)
    scan(candidate)
    restored = copy.deepcopy(candidate)
    candops = at(candidate, p['operations_path'])
    original = at(baseline, p['operations_path'])
    j = 0
    recovered = []
    for i, op in enumerate(original):
        if i in p['remove']:
            recovered.append(copy.deepcopy(op))
            continue
        want = copy.deepcopy(p['replace'].get(i, [op]))
        for node in want:
            for v in all_blob_values(node):
                key = v['blobFileValue']['offset']
                if isinstance(key, str) and key.startswith('APPEND_'):
                    v['blobFileValue']['offset'] = str(offsets[key])
        need(candops[j:j + len(want)] == want, 'unlisted fusion/head/cur/dataflow/constant change at originalop' + str(i))
        j += len(want)
        recovered.append(copy.deepcopy(op))
    need(j == len(candops), 'extra/dangling candidate nodes')
    put(restored, tuple(p['operations_path']), recovered)
    ud = restored['description']['metadata']['userDefined']
    need(ud.get('parler.decoder.graph_variant') == MARKER, 'exact marker')
    del ud['parler.decoder.graph_variant']
    need(restored == baseline, 'unlisted specification/interface/metadata change')
    return restored
