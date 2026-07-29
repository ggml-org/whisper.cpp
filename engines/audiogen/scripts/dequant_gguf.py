#!/usr/bin/env python3
"""Rewrite a GGUF with every quantized tensor dequantized to F32.

Diagnostic tool. Dequantizing Q8_0 -> F32 keeps the exact same weight *values*
(Q8_0 is already a rounding of the original weights), so running the F32 copy on
a backend that does exact f32 arithmetic yields the ground-truth trajectory for
those weights. Comparing a quantized run against it shows which backend's
quantized matmul path is actually further from truth, which "CPU vs GPU" cannot.

An optional comma-separated list of name substrings keeps only the matching
tensors, so a single stage can be extracted from a multi-stage GGUF without
paying for the rest in f32 (the ACE-Step DiT file holds the detokenizer too).

usage: dequant_gguf.py in.gguf out.gguf [keep_substr,keep_substr,...]
"""
import sys

import numpy as np
from gguf import GGUFReader, GGUFWriter, GGUFValueType
from gguf.quants import dequantize

# Fields the writer emits itself from the tensor/kv counts it tracks.
SKIP = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"}


def field_value(field):
    """Decode a GGUFReader field into a plain Python value."""
    if field.types[0] == GGUFValueType.ARRAY:
        elem = field.types[1]
        if elem == GGUFValueType.STRING:
            return [str(bytes(field.parts[i]), "utf-8") for i in field.data]
        return [field.parts[i].tolist()[0] for i in field.data]
    if field.types[0] == GGUFValueType.STRING:
        return str(bytes(field.parts[field.data[0]]), "utf-8")
    return field.parts[field.data[0]].tolist()[0]


def main():
    if len(sys.argv) not in (3, 4):
        print(__doc__)
        return 1
    src, dst = sys.argv[1], sys.argv[2]
    keep = [s for s in sys.argv[3].split(",") if s] if len(sys.argv) == 4 else []

    reader = GGUFReader(src)
    arch = field_value(reader.fields["general.architecture"])
    writer = GGUFWriter(dst, arch)

    for name, field in reader.fields.items():
        if name in SKIP or name == "general.architecture":
            continue
        value = field_value(field)
        vtype = field.types[0]
        if vtype == GGUFValueType.ARRAY:
            writer.add_array(name, value)
        else:
            writer.add_key_value(name, value, vtype)

    n_deq = 0
    n_kept = 0
    for tensor in reader.tensors:
        if keep and not any(k in tensor.name for k in keep):
            continue
        n_kept += 1
        # reader.shape is ggml ne order; numpy wants it reversed.
        shape = tuple(int(d) for d in reversed(tensor.shape))
        if tensor.tensor_type.name == "F32":
            data = np.asarray(tensor.data, dtype=np.float32).reshape(shape)
        else:
            data = dequantize(tensor.data, tensor.tensor_type).astype(np.float32).reshape(shape)
            n_deq += 1
        writer.add_tensor(tensor.name, data)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {dst}: {n_kept} tensors kept of {len(reader.tensors)}, {n_deq} dequantized to f32")
    return 0


if __name__ == "__main__":
    sys.exit(main())
