#!/usr/bin/env python3
"""Convert the LavaSR enhancer (Vocos bandwidth-extension) ONNX pair into a
single GGUF for the tts-cpp CPU/GGML enhancer.

The enhancer is two ONNX graphs:
  * enhancer_backbone.onnx   mel[B,80,T] -> hidden[B,T,512]
        embed Conv1d(80->512,k7,pad3) -> LayerNorm
        8x ConvNeXt block:
            dwconv Conv1d(512->512,k7,pad3,group=512)
            LayerNorm(eps=1e-6)
            pwconv1 Linear(512->1536) + erf-GELU
            pwconv2 Linear(1536->512)
            *gamma (layer scale) + residual
        final LayerNorm
  * enhancer_spec_head.onnx  hidden[B,T,512] -> real[B,1025,T], imag[B,1025,T]
        Linear(512->2050) -> transpose -> split(1025,1025)
        mag = clip(exp(split0), max=clip_max);  real = mag*cos(split1);  imag = mag*sin(split1)

Linear (MatMul) weights are stored ONNX-side as [in,out]; we transpose them to
[out,in] (PyTorch convention) so the C++ loader reads ggml ne=[in,out] and runs
ggml_mul_mat(W, x) directly.  Conv weights are stored as ONNX [out,in,k].

Usage:
  python convert-lavasr-enhancer-to-gguf.py \
      --backbone  enhancer_backbone.onnx \
      --spec-head enhancer_spec_head.onnx \
      --out       lavasr-enhancer.gguf \
      --ftype     f32            # or f16
"""
import argparse
import sys

import numpy as np
import onnx
from gguf import GGUFWriter
from onnx import numpy_helper

ARCH = "lavasr-enhancer"

# Mel / STFT params (must match src/lavasr/dsp + the @qvac/tts-onnx enhancer).
N_MELS = 80
DIM = 512
FFN_DIM = 1536
N_BLOCKS = 8
KERNEL = 7
N_FFT = 2048
HOP = 512
WIN = 2048
SPEC_BINS = N_FFT // 2 + 1  # 1025
MEL_REF_SR = 44100          # Slaney mel reference rate (Vocos training)
WORK_SR = 48000             # enhancer operates on 48 kHz audio
LN_EPS = 1e-6


def init_map(graph):
    return {t.name: numpy_helper.to_array(t) for t in graph.initializer}


def node_by_output(graph):
    out = {}
    for n in graph.node:
        for o in n.output:
            out[o] = n
    return out


def find_matmul_weight(graph, inits, by_out, bias_name):
    """Given a `*.bias` initializer name added right after a MatMul, return the
    MatMul's weight array (an initializer)."""
    for n in graph.node:
        if n.op_type == "Add" and bias_name in n.input:
            other = [i for i in n.input if i != bias_name][0]
            mm = by_out.get(other)
            if mm is None or mm.op_type != "MatMul":
                raise RuntimeError(f"expected MatMul feeding Add of {bias_name}")
            for i in mm.input:
                if i in inits:
                    return inits[i]
            raise RuntimeError(f"MatMul for {bias_name} has no initializer input")
    raise RuntimeError(f"no Add node consuming bias {bias_name}")


def store(writer, name, arr, ftype, allow_f16=True):
    arr = np.ascontiguousarray(arr)
    if ftype == "f16" and allow_f16 and arr.ndim >= 2 and arr.dtype == np.float32:
        arr = arr.astype(np.float16)
    elif arr.dtype != np.float32 and arr.dtype != np.float16:
        arr = arr.astype(np.float32)
    writer.add_tensor(name, arr)
    print(f"  {name:42s} {str(arr.dtype):8s} {list(arr.shape)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--spec-head", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ftype", choices=["f32", "f16"], default="f32")
    args = ap.parse_args()

    bb = onnx.load(args.backbone, load_external_data=True).graph
    sh = onnx.load(args.spec_head, load_external_data=True).graph
    bi = init_map(bb)
    si = init_map(sh)
    bb_by_out = node_by_output(bb)

    # The spec head clamps the log-magnitude via Clip(exp(x), None, max) before
    # the cos/sin polar reconstruction (see the graph: Exp -> Clip -> Mul). Read
    # the clamp upper bound straight from that Clip node's `max` input (Clip's
    # 3rd input in opset >= 11) rather than scanning all scalar constants, so a
    # re-export carrying another scalar can't silently change the clamp. Falls
    # back to 1000.0 (with a warning) only if the graph shape ever changes.
    clip_max = 1000.0
    clip_nodes = [n for n in sh.node if n.op_type == "Clip"]
    if (len(clip_nodes) == 1 and len(clip_nodes[0].input) >= 3
            and clip_nodes[0].input[2] in si):
        # si values are already numpy arrays (init_map -> numpy_helper.to_array).
        clip_max = float(si[clip_nodes[0].input[2]].reshape(-1)[0])
    else:
        print(f"WARNING: could not uniquely resolve the spec-head Clip max input "
              f"({len(clip_nodes)} Clip node(s)); using fallback clip_max={clip_max}")

    writer = GGUFWriter(args.out, ARCH)
    writer.add_uint32("lavasr.enhancer.dim", DIM)
    writer.add_uint32("lavasr.enhancer.ffn_dim", FFN_DIM)
    writer.add_uint32("lavasr.enhancer.n_blocks", N_BLOCKS)
    writer.add_uint32("lavasr.enhancer.n_mels", N_MELS)
    writer.add_uint32("lavasr.enhancer.kernel", KERNEL)
    writer.add_uint32("lavasr.enhancer.n_fft", N_FFT)
    writer.add_uint32("lavasr.enhancer.hop", HOP)
    writer.add_uint32("lavasr.enhancer.win", WIN)
    writer.add_uint32("lavasr.enhancer.spec_bins", SPEC_BINS)
    writer.add_uint32("lavasr.enhancer.mel_ref_sample_rate", MEL_REF_SR)
    writer.add_uint32("lavasr.enhancer.work_sample_rate", WORK_SR)
    writer.add_float32("lavasr.enhancer.clip_max", clip_max)
    writer.add_float32("lavasr.enhancer.layernorm_eps", LN_EPS)

    print("tensors:")
    # --- embed + first norm ---
    store(writer, "enhancer.embed.weight", bi["backbone.embed.weight"], args.ftype)
    store(writer, "enhancer.embed.bias", bi["backbone.embed.bias"], args.ftype, allow_f16=False)
    store(writer, "enhancer.norm.weight", bi["backbone.norm.weight"], args.ftype, allow_f16=False)
    store(writer, "enhancer.norm.bias", bi["backbone.norm.bias"], args.ftype, allow_f16=False)

    # --- 8 ConvNeXt blocks ---
    for i in range(N_BLOCKS):
        p = f"backbone.convnext.{i}"
        store(writer, f"enhancer.block.{i}.dwconv.weight", bi[f"{p}.dwconv.weight"], args.ftype)
        store(writer, f"enhancer.block.{i}.dwconv.bias", bi[f"{p}.dwconv.bias"], args.ftype, allow_f16=False)
        store(writer, f"enhancer.block.{i}.norm.weight", bi[f"{p}.norm.weight"], args.ftype, allow_f16=False)
        store(writer, f"enhancer.block.{i}.norm.bias", bi[f"{p}.norm.bias"], args.ftype, allow_f16=False)
        w1 = find_matmul_weight(bb, bi, bb_by_out, f"{p}.pwconv1.bias")  # [in=512, out=1536]
        w2 = find_matmul_weight(bb, bi, bb_by_out, f"{p}.pwconv2.bias")  # [in=1536, out=512]
        store(writer, f"enhancer.block.{i}.pwconv1.weight", w1.T, args.ftype)  # -> [out,in]
        store(writer, f"enhancer.block.{i}.pwconv1.bias", bi[f"{p}.pwconv1.bias"], args.ftype, allow_f16=False)
        store(writer, f"enhancer.block.{i}.pwconv2.weight", w2.T, args.ftype)  # -> [out,in]
        store(writer, f"enhancer.block.{i}.pwconv2.bias", bi[f"{p}.pwconv2.bias"], args.ftype, allow_f16=False)
        store(writer, f"enhancer.block.{i}.gamma", bi[f"{p}.gamma"], args.ftype, allow_f16=False)

    # --- final layer norm ---
    store(writer, "enhancer.final_norm.weight", bi["backbone.final_layer_norm.weight"], args.ftype, allow_f16=False)
    store(writer, "enhancer.final_norm.bias", bi["backbone.final_layer_norm.bias"], args.ftype, allow_f16=False)

    # --- spec head ---
    w_out = find_matmul_weight(sh, si, node_by_output(sh), "out.bias")  # [in=512, out=2050]
    store(writer, "spec_head.out.weight", w_out.T, args.ftype)  # -> [out=2050, in=512]
    store(writer, "spec_head.out.bias", si["out.bias"], args.ftype, allow_f16=False)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"\nWrote {args.out} (arch={ARCH}, ftype={args.ftype}, clip_max={clip_max})")


if __name__ == "__main__":
    sys.exit(main())
