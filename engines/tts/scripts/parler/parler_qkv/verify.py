"""Independent all-learned byte mapping. No dequantization/requantization.
Serialized append case accounts known inactive ORIGINAL QKV records explicitly.
Unknown or unaccounted records are rejected. All active and retained inactive records are accounted for.
"""
import struct
from pathlib import Path
from . import weights as wi, storage as ba

def verify(left_root, right_root, left, right, c):
    L = c['n_layer']
    wi.expected_inventory(c, left)
    wi.expected_inventory(c, right, fused=True)
    need = wi.require
    need(len(left) == 14 * L + 3 and len(right) == 12 * L + 3, 'active learned counts')
    need(sum((len(v['blobs']) for v in left.values())) == 22 * L + 3 and sum((len(v['blobs']) for v in right.values())) == 18 * L + 3, 'active blob counts')
    ls = wi.physical_inventory(left_root, left)
    weight = Path(right_root) / 'weights/weight.bin'
    with weight.open('rb') as f:
        count, version = struct.unpack('<II', f.read(8))
    need(version == 2, 'MILstoragev2')
    old_count = 22 * L + 3
    active = 18 * L + 3
    appended = old_count + 2 * L
    if count == appended:
        need(ba.prefix_equal(Path(left_root) / 'weights/weight.bin', weight), 'append storage ORIGINALprefix differs beyond firstuint32')
        combined = {'active::' + n: v for n, v in right.items()}
        combined.update({'retained-baseline::' + n: v for n, v in left.items()})
        rs = wi.physical_inventory(right_root, combined)
        storage = 'append_with_explicit_known_inactive_QKV'
    else:
        raise wi.VerificationError('unknown physical count/layout; no orphan skipping')
    mapping = {}
    for name in right:
        if name.endswith('_qkv_weight_to_fp16_quantized'):
            mapping[name] = [name.replace('_qkv_', '_' + k + '_') for k in ('q', 'k', 'v')]
        else:
            mapping[name] = [name]
    covered = [v for ns in mapping.values() for v in ns]
    need(len(covered) == len(set(covered)) and set(covered) == set(left), 'complete ONE-to-ONE baseline logicalcoverage')
    for name, parts in mapping.items():
        v = right[name]
        sources = [left[n] for n in parts]
        need(all((s['kind'] == v['kind'] and s['dtype'] == v['dtype'] for s in sources)), 'learned kind/dtype')
        for i, r in enumerate(v['blobs']):
            target = rs[r['path'], r['offset']]
            source = [ls[s['blobs'][i]['path'], s['blobs'][i]['offset']] for s in sources]
            need(all((x[3] == target[3] for x in source)) and sum((x[2] for x in source)) == target[2], 'typed mapping length')
            need(wi.payloads_equal(source, [target]), 'ordered RAW codes/scales differ ' + name + ' ' + str(i))
            if len(parts) == 1 and storage.startswith('append'):
                need(r == sources[0]['blobs'][i], 'nonQKV physical reference moved')
    return {'status': 'PASS', 'baseline_learned': len(left), 'baseline_blob_references': old_count, 'active_candidate_learned': len(right), 'active_candidate_blob_references': active, 'physical_candidate_records': count, 'retained_inactive_records': count - active, 'physical_storage': storage, 'complete_original_logical_coverage': len(covered), 'Q_K_V_ordered_codes_and_perchannel_scales_exact': True, 'all_nonQKV_heads_norms_rawbytes_exact': True, 'no_requantization_or_precisionchange': True}
