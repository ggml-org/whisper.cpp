"""Complete learned tensor inventory and streaming MILstoragev2 byte checks."""
import math
import struct
from pathlib import Path
SIZES = {'int8': 1, 'fp16': 2}
BLOB_DTYPE_CODES = {'fp16': 1, 'int8': 4}
PROJECTIONS = ('q', 'k', 'v', 'o', 'cross_q', 'cross_o', 'up', 'down')

class VerificationError(ValueError):
    pass

def require(ok, message):
    if not ok:
        raise VerificationError(message)

def expected_inventory(c, inventory, fused=False):
    expected = {}
    d = c['d_model']
    n = c['n_layer']
    for l in range(n):
        prefix = f'layers_{l}_'
        upname = prefix + 'up_weight_to_fp16_quantized'
        require(upname in inventory, 'missing FFN ' + upname)
        up = inventory[upname]['shape']
        require(len(up) == 4 and up[1:] == (d, 1, 1), 'invalid FFN shape ' + upname)
        ff = up[0]
        require(ff == (128 if n == 2 else 4096), 'unsupported FFN geometry')
        projs = ('qkv', 'o', 'cross_q', 'cross_o', 'up', 'down') if fused else PROJECTIONS
        for p in projs:
            sh = (ff, d, 1, 1) if p == 'up' else (d, ff, 1, 1) if p == 'down' else (3 * d, d, 1, 1) if p == 'qkv' else (d, d, 1, 1)
            name = prefix + p + '_weight_to_fp16_quantized'
            expected[name] = ('int8', sh)
        for section in ['self', 'cross', 'ffn']:
            for part in ['weight', 'bias']:
                expected[prefix + section + '_norm_' + part + '_to_fp16'] = ('raw', (d,))
    for part in ['weight', 'bias']:
        expected['out_norm_' + part + '_to_fp16'] = ('raw', (d,))
    expected['heads_weight_to_fp16'] = ('raw', (c['n_q'] * c['vocab'], d, 1, 1))
    require(set(inventory) == set(expected), 'missing/renamed/unknown learned tensors: missing=' + str(sorted(set(expected) - set(inventory))) + ' extra=' + str(sorted(set(inventory) - set(expected))))
    for name, (kind, sh) in expected.items():
        item = inventory[name]
        require((item['kind'], item['dtype'], item['shape']) == (kind, 'fp16', sh), 'learned tensor kind/dtype/shape mismatch ' + name)
        if kind == 'int8':
            a, b = item['blobs']
            require(a['shape'] == sh and b['shape'] == (sh[0], 1, 1, 1), 'quant data/scale shape mismatch ' + name)
    return expected

def physical_inventory(root, inventory):
    byfile = {}
    for item in inventory.values():
        for ref in item['blobs']:
            path = ref['path']
            off = ref['offset']
            byfile.setdefault(path, {})
            if off in byfile[path]:
                require(byfile[path][off] == ref, 'inconsistent aliased blob declaration')
            byfile[path][off] = ref
    actual = {str(p.relative_to(root)) for p in (Path(root) / 'weights').rglob('*') if p.is_file()}
    require(actual == set(byfile), 'unaccounted/missing weights files')
    spans = {}
    for relative, refs in byfile.items():
        path = Path(root) / relative
        length = path.stat().st_size
        with path.open('rb') as f:
            header = f.read(64)
            require(len(header) == 64, 'truncated blob file header')
            count, version = struct.unpack('<II', header[:8])
            require(version == 2 and count == len(refs), 'unknown blob format or unreferenced physical blobs')
            cursor = 64
            for off, ref in sorted(refs.items()):
                require(off == cursor, 'unaccounted physical blob region at ' + str(cursor))
                f.seek(off)
                header = f.read(64)
                require(len(header) == 64, 'truncated blob metadata')
                magic, dt, size, data = struct.unpack('<IIQQ', header[:24])
                require(magic == 0xDEADBEEF and data == off + 64, 'unknown blob header/layout')
                require(ref['dtype'] in BLOB_DTYPE_CODES and dt == BLOB_DTYPE_CODES[ref['dtype']], 'blob storage dtype disagrees with MIL spec')
                require(size == math.prod(ref['shape']) * SIZES[ref['dtype']], 'blob byte count/type/shape mismatch')
                require(data + size <= length, 'truncated blob payload')
                spans[relative, off] = (path, data, size, dt)
                cursor = (data + size + 63) // 64 * 64
            require(length in [data + size, cursor], 'unaccounted trailing physical weights')
    return spans

def payloads_equal(left, right):
    """Compare one blob or an explicitly ordered fused concatenation, streaming."""

    def chunks(spans):
        for path, offset, length, dt in spans:
            with path.open('rb') as f:
                f.seek(offset)
                while length:
                    b = f.read(min(length, 1 << 20))
                    require(b, 'truncated payload')
                    length -= len(b)
                    yield b
    a = iter(chunks(left))
    b = iter(chunks(right))
    x = y = b''
    while True:
        if not x:
            x = next(a, b'')
        if not y:
            y = next(b, b'')
        if not x or not y:
            return x == y
        n = min(len(x), len(y))
        if x[:n] != y[:n]:
            return False
        x = x[n:]
        y = y[n:]
