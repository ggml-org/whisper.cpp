"""Exact affine codes in unit-scale GGML carriers; no second weight quantizer.

The architecture tag prevents stock engines from treating carrier integers as
model weights. FP32 scales/minima are independent tensors. Non-matrix weights
remain effective FP16, matching the original packed projection runtime.
"""
import numpy as np
import torch
import gguf


def encode(weight, bits):
    if bits not in (4, 8) or weight.ndim < 2 or not weight.is_floating_point():
        raise ValueError('expected a floating matrix and 4/8 bits')
    flat = weight.detach().cpu().float().reshape(weight.shape[0], -1)
    if not torch.isfinite(flat).all():
        raise ValueError('nonfinite source')
    minimum = flat.amin(dim=1)
    step = (flat.amax(dim=1) - minimum) / (2**bits - 1)
    safe = torch.where(step == 0, 1, step)
    codes = torch.round((flat - minimum[:, None]) / safe[:, None]).clamp(0, 2**bits - 1).to(torch.uint8)
    return dict(bits=bits, codes=codes.numpy(), minimum=minimum.numpy(), step=step.numpy())


def decode(record):
    codes = record['codes'].astype(np.float32)
    scaled = codes * record['step'][:, None]
    return (scaled + record['minimum'][:, None]).astype(np.float16)


def carrier(record):
    codes = record['codes']
    if codes.ndim != 2 or codes.shape[1] % 32:
        raise ValueError('carrier matrices require input width divisible by 32')
    q = codes.reshape(codes.shape[0], -1, 32)
    if record['bits'] == 4:
        if np.any(q > 15):
            raise ValueError('INT4 code outside range')
        packed = q[..., :16] | (q[..., 16:] << 4)
        qtype = gguf.GGMLQuantizationType.Q4_0
    elif record['bits'] == 8:
        packed = (q.astype(np.int16) - 128).astype(np.int8).view(np.uint8)
        qtype = gguf.GGMLQuantizationType.Q8_0
    else:
        raise ValueError('unsupported bits')
    unit = np.ones((*q.shape[:2], 1), dtype=np.float16).view(np.uint8)
    payload = np.concatenate((unit, packed), axis=-1).reshape(codes.shape[0], -1)
    # Prove transport preserves every code, including all-zero/constant rows.
    restored = gguf.quants.dequantize(payload, qtype) + (8 if record['bits'] == 4 else 128)
    if not np.array_equal(restored, codes):
        raise ValueError('GGML carrier changed codes')
    return payload, qtype


def writer_factory(records, *, dense=False):
    class AffineWriter(gguf.GGUFWriter):
        def __init__(self, path, arch):
            super().__init__(path, arch if dense else 'parakeet-affine-v1')
            self.seen = set()
            if not dense:
                self.add_uint32('parakeet.affine.version', 1)
                self.add_array('parakeet.affine.names', sorted(records))

        def add_tensor(self, name, tensor, **kwargs):
            # Both packed and dense controls use separate Q/K/V, with no duplicate
            # fused copy that would hide or undo the chosen precision allocation.
            if '.attn.qkv.' in name:
                return
            if name in records:
                record = records[name]
                expected = decode(record)
                supplied = np.asarray(tensor).reshape(expected.shape).astype(np.float16)
                if not np.array_equal(supplied, expected):
                    raise ValueError('effective source mismatch: ' + name)
                self.seen.add(name)
                if not dense:
                    payload, qtype = carrier(record)
                    super().add_tensor(name, payload, raw_dtype=qtype)
                    for field in ('minimum', 'step'):
                        super().add_tensor(name + '.affine.' + field, record[field])
                    return
            super().add_tensor(name, tensor, **kwargs)

        def write_header_to_file(self, *args, **kwargs):
            if self.seen != set(records):
                raise ValueError('unwritten affine records: ' + str(set(records) - self.seen))
            return super().write_header_to_file(*args, **kwargs)
    return AffineWriter
