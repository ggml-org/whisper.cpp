"""Serialized learned inventory/consumer oracle; no model/framework imports."""
import json
from . import spec as core, weights as wi

def inventory(doc, fused=False):
    fn, b, ops, ext, prod, uses, _ = core.scan(doc)
    c = core.strict_json(doc['description']['metadata']['userDefined']['parler.decoder.contract'])
    out = {}

    def ref(v):
        core.need(set(v) == {'type', 'blobFileValue'}, 'serialized BLOB schema')
        t = core.tensor_type(v['type'])
        r = v['blobFileValue']
        dt = {'FLOAT16': 'fp16', 'INT8': 'int8'}
        core.need(t['dataType'] in dt and set(r) == {'fileName', 'offset'} and (r['fileName'] == '@model_path/weights/weight.bin'), 'dtype/path schema')
        return {'dtype': dt[t['dataType']], 'shape': tuple(core.dims(v['type'])), 'path': 'weights/weight.bin', 'offset': int(r['offset'])}
    for op in ops:
        if op['type'] == 'constexpr_blockwise_shift_scale':
            refs = [ref(core.blob(op, k)) for k in ('data', 'scale')]
            kind = 'int8'
        elif op['type'] == 'const' and 'blobFileValue' in op.get('attributes', {}).get('val', {}):
            refs = [ref(op['attributes']['val'])]
            kind = 'raw'
        else:
            continue
        name = core.named(op)
        core.need(name not in out, 'duplicate learned')
        out[name] = {'kind': kind, 'dtype': 'fp16', 'shape': tuple(core.dims(op['outputs'][0]['type'])), 'blobs': refs}
    L = c['n_layer']
    core.need(len(out) == (12 if fused else 14) * L + 3 and sum((len(v['blobs']) for v in out.values())) == (18 if fused else 22) * L + 3, 'complete active producer/blob coverage')
    core.need(sum((len(v['blobs']) for v in out.values())) == len(core.all_blob_values(doc)), 'unaccounted serialized blob')
    wi.expected_inventory(c, out, fused=fused)
    conv = [core.ref(o, 'weight') for o in ops if o['type'] == 'conv']
    gam = [core.ref(o, 'gamma') for o in ops if o['type'] == 'layer_norm']
    bet = [core.ref(o, 'beta') for o in ops if o['type'] == 'layer_norm']
    for actual, want in [(conv, {n for n, v in out.items() if len(v['shape']) == 4}), (gam, {n for n in out if '_norm_weight_' in n}), (bet, {n for n in out if '_norm_bias_' in n})]:
        core.need(len(actual) == len(want) and set(actual) == want, 'complete distinct learned consumer coverage')
    return (c, out)
