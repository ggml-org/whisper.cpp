"""Append-only MILstoragev2. Original offsets and opaque record header bytes are preserved.
No dequantization, floating math or requantization. Existing offsets unchanged.
"""
import math
import struct
SIZES = {'INT8': 1, 'FLOAT16': 2}
CODES = {'INT8': 4, 'FLOAT16': 1}

def need(ok, why):
    if not ok:
        raise ValueError(why)

def span(path, value):
    t = value['type']['tensorType']
    dtype = t['dataType']
    shape = [int(d['constant']['size']) for d in t['dimensions']]
    ref = value['blobFileValue']
    off = int(ref['offset'])
    need(ref['fileName'] == '@model_path/weights/weight.bin' and dtype in SIZES, 'knownsinglefile dtype required')
    with path.open('rb') as f:
        f.seek(off)
        header = f.read(64)
    need(len(header) == 64, 'truncated source header')
    magic, code, size, data = struct.unpack('<IIQQ', header[:24])
    need(magic == 0xDEADBEEF and code == CODES[dtype] and (size == math.prod(shape) * SIZES[dtype]) and (data == off + 64) and (data + size <= path.stat().st_size), 'typed source record mismatch')
    return {'offset': off, 'data': data, 'size': size, 'dtype': dtype, 'header': header}

def append(source, destination, plan):
    need(not source.samefile(destination), 'copy must not aliasoriginal')
    with source.open('rb') as f:
        header = f.read(64)
    count, version = struct.unpack('<II', header[:8])
    need(version == 2, 'MILstoragev2 only')
    need(destination.stat().st_size == source.stat().st_size and prefix_equal(source, destination), 'destination must be untouched copy before append')
    offsets = {}
    records = []
    with destination.open('r+b') as out:
        out.seek(0, 2)
        for item in plan['append_payloads']:
            spans = [span(source, v) for v in item['sources']]
            dtype = item['target_dtype']
            need(all((s['dtype'] == dtype for s in spans)), 'fused dtype')
            size = sum((s['size'] for s in spans))
            need(size == math.prod(item['target_shape']) * SIZES[dtype], 'fused bytegeometry')
            padding = -out.tell() % 64
            out.write(b'\x00' * padding)
            off = out.tell()
            # Preserve the original record's opaque header bytes, replacing only
            # magic/dtype/length/data-offset. No other record is rewritten.
            h = bytearray(spans[0]['header'])
            struct.pack_into('<IIQQ', h, 0, 0xDEADBEEF, CODES[dtype], size, off + 64)
            out.write(h)
            for s in spans:
                with source.open('rb') as f:
                    f.seek(s['data'])
                    remain = s['size']
                    while remain:
                        b = f.read(min(remain, 1 << 20))
                        need(b, 'truncated source payload')
                        out.write(b)
                        remain -= len(b)
            out.write(b'\x00' * (-out.tell() % 64))
            key = 'APPEND_' + str(item['layer']) + '_' + item['kind']
            offsets[key] = off
            records.append({'key': key, 'offset': off, 'size': size, 'dtype': dtype, 'sources': [s['offset'] for s in spans]})
        out.seek(0)
        out.write(struct.pack('<I', count + len(records)))
    return (offsets, {'old_physical_records': count, 'new_physical_records': count + len(records), 'appended_records': records, 'original_prefix_bytes_except_first_count_uint32': source.stat().st_size - 4})

def prefix_equal(source, destination):
    with source.open('rb') as a, destination.open('rb') as b:
        a.seek(4)
        b.seek(4)
        remain = source.stat().st_size - 4
        while remain:
            n = min(remain, 1 << 20)
            x = a.read(n)
            y = b.read(n)
            if len(x) != n or x != y:
                return False
            remain -= n
    return True
