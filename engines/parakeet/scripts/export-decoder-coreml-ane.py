#!/usr/bin/env python3
"""Export the TDT prediction network + joint from a parakeet GGUF to a Core ML
package laid out for Apple Neural Engine execution.

Each dispatch steps the prediction LSTM once and scores WIDTH consecutive
encoder frames for LANES independent utterance slots, carried on the batch
axis. The prediction network is memory-bandwidth bound, so lanes batch
independent windows into one dispatch that reads the weights once. The TDT
walk (choosing the next frame from the emitted token and duration) stays on
the host.

Layout follows export-encoder-coreml-ane.py: every activation is [N, C, 1, T],
every linear projection is a 1x1 conv, and all shapes are compile-time
constants.

The token embedding is not part of the graph, since a gather is not an ANE op:
the caller looks up the row for its last emitted token and passes it in.
The engine reads that table from the GGUF; --dump-embedding writes a copy as
fp16 [vocab+1, H_pred] for inspection.

Weights are palettized to --nbits bits (default 6) except the joint output
projection, which stays fp16 unless --nbits-out is set; the published model uses
--nbits-out 6. The graph geometry is also written to <out>.meta.json for
reference; the engine does not read it.

I/O contract:

    embed      fp16 [LANES, H_pred,     1, 1]
    h_in       fp16 [LANES, H_pred * L, 1, 1]
    c_in       fp16 [LANES, H_pred * L, 1, 1]
    enc_step   fp16 [LANES, D_enc,      1, WIDTH]   WIDTH consecutive encoder frames
    logits     fp16 [LANES, V_out,      1, WIDTH]   vocab+blank, then durations
    h_out      fp16 [LANES, H_pred * L, 1, 1]
    c_out      fp16 [LANES, H_pred * L, 1, 1]

h_in/c_in carry all L LSTM layers concatenated along the channel axis.
`logits` is the raw joint output: the host takes the argmax over channels
[0, V+1) for the token and over [V+1, V_out) for the duration, as the ggml
path does.

Example:

  python engines/parakeet/scripts/export-decoder-coreml-ane.py \
      --gguf engines/parakeet/models/transcribepsy-250m-eu-sub.gguf \
      --out  engines/parakeet/models/transcribepsy-250m-eu-sub-decoder.mlpackage \
      --compile-dir engines/parakeet/models --nbits-out 6

Needs the pinned scripts/requirements-coreml.txt environment and macOS for
--compile-dir. The engine loads the result from `<model>-decoder.mlmodelc`
next to the GGUF (src/parakeet_coreml_path.cpp) and drives it through
tdt_decode_lanes (src/parakeet_tdt.cpp).
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import coremltools as ct

# Utterance slots per dispatch, and encoder frames scored per dispatch. Both
# are compile-time constants: they are the shape of the graph, not arguments.
DEFAULT_LANES = 16
DEFAULT_WIDTH = 1


def _load_sibling(scripts_dir, filename, module_name):
    # These live next to us under names with dashes in them, so they cannot be
    # imported normally.
    path = scripts_dir / filename
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TdtGeometry:
    """The decoder's shapes, read off the checkpoint rather than assumed."""

    def __init__(self, weights, meta):
        self.n_layers = int(meta["parakeet.tdt.pred_rnn_layers"])
        self.h_pred = int(meta["parakeet.tdt.pred_hidden"])
        self.h_joint = int(meta["parakeet.tdt.joint_hidden"])
        self.vocab = int(meta["parakeet.tdt.vocab_size"])
        self.blank_id = int(meta["parakeet.tdt.blank_id"])
        self.n_durations = int(meta["parakeet.tdt.num_durations"])
        # D_enc is whatever the encoder actually emits, read off the joint.
        self.d_enc = int(weights["tdt.joint.enc.weight"].shape[1])
        self.v_out = int(weights["tdt.joint.out.weight"].shape[0])

        embed_rows = int(weights["tdt.predict.embed.weight"].shape[0])
        expect_v_out = self.vocab + 1 + self.n_durations
        if self.v_out != expect_v_out:
            raise RuntimeError(
                f"joint.out has {self.v_out} rows, expected vocab+1+durations "
                f"= {expect_v_out}")
        if embed_rows != self.vocab + 1:
            raise RuntimeError(
                f"embed has {embed_rows} rows, expected vocab+1 = {self.vocab + 1}")
        for layer in range(self.n_layers):
            if f"tdt.predict.lstm.{layer}.w_ih" not in weights:
                raise RuntimeError(
                    f"checkpoint declares {self.n_layers} LSTM layers but "
                    f"lstm.{layer} is missing")

    def __str__(self):
        return (f"pred_rnn_layers={self.n_layers} H_pred={self.h_pred} "
                f"H_joint={self.h_joint} D_enc={self.d_enc} "
                f"vocab={self.vocab} V_out={self.v_out} "
                f"blank_id={self.blank_id} n_durations={self.n_durations}")


class AneTdtDecoder(torch.nn.Module):
    """One dispatch: step the LSTM once, score WIDTH frames, in [N, C, 1, T]."""

    def __init__(self, weights, geom, lanes, width):
        super().__init__()
        self.geom = geom
        self.lanes = lanes
        self.width = width
        H = geom.h_pred

        for layer in range(geom.n_layers):
            p = f"tdt.predict.lstm.{layer}"
            self._conv1x1(f"ih{layer}", weights[f"{p}.w_ih"])
            self._conv1x1(f"hh{layer}", weights[f"{p}.w_hh"])
            # The two LSTM bias vectors are only ever added together, so fold
            # them here and carry one.
            self._buf(f"bias{layer}", weights[f"{p}.b_ih"] + weights[f"{p}.b_hh"])

        # The two joint input projections are summed before the relu, so their
        # biases fold into one. Carry it on the enc conv, which is the one with
        # the time axis; the pred conv then needs no bias at all.
        self._conv1x1("j_enc", weights["tdt.joint.enc.weight"])
        self._conv1x1("j_pred", weights["tdt.joint.pred.weight"])
        self._buf("j_enc_b", weights["tdt.joint.enc.bias"]
                  + weights["tdt.joint.pred.bias"])
        self._conv1x1("j_out", weights["tdt.joint.out.weight"],
                      weights["tdt.joint.out.bias"])

    def _buf(self, name, tensor):
        self.register_buffer(name, tensor.contiguous().float())

    def _conv1x1(self, name, weight, bias=None):
        # GGUF gives [out, in] once reversed, which is F.linear's layout; a 1x1
        # conv wants [out, in, 1, 1].
        out_f, in_f = weight.shape
        self._buf(f"{name}_w", weight.reshape(out_f, in_f, 1, 1))
        if bias is not None:
            self._buf(f"{name}_b", bias)

    def forward(self, embed, h_in, c_in, enc_step):
        H = self.geom.h_pred
        h_out, c_out = [], []

        x = embed
        for layer in range(self.geom.n_layers):
            lo, hi = layer * H, (layer + 1) * H
            h_l = h_in[:, lo:hi]
            c_l = c_in[:, lo:hi]
            gates = (F.conv2d(x, getattr(self, f"ih{layer}_w"),
                              getattr(self, f"bias{layer}"))
                     + F.conv2d(h_l, getattr(self, f"hh{layer}_w")))
            # Gate order is i, f, g, o -- PyTorch's, and the ggml path's.
            i_g = torch.sigmoid(gates[:, 0 * H:1 * H])
            f_g = torch.sigmoid(gates[:, 1 * H:2 * H])
            g_g = torch.tanh   (gates[:, 2 * H:3 * H])
            o_g = torch.sigmoid(gates[:, 3 * H:4 * H])
            c_new = f_g * c_l + i_g * g_g
            h_new = o_g * torch.tanh(c_new)
            h_out.append(h_new)
            c_out.append(c_new)
            x = h_new
        pred = x                                              # [N, H, 1, 1]

        # enc_h is the only term with a time axis; pred_h broadcasts across it.
        enc_h = F.conv2d(enc_step, self.j_enc_w, self.j_enc_b)  # [N, Hj, 1, W]
        pred_h = F.conv2d(pred, self.j_pred_w)                  # [N, Hj, 1, 1]
        hidden = torch.relu(enc_h + pred_h)
        logits = F.conv2d(hidden, self.j_out_w, self.j_out_b)   # [N, Vo, 1, W]

        return logits, torch.cat(h_out, dim=1), torch.cat(c_out, dim=1)


def reference_step(weights, geom, embed, h, c, enc_frames):
    """Scalar numpy transcription of host_lstm_step + host_joint_step.

    Deliberately literal -- one layer at a time, one frame at a time -- to
    catch a transposed or misrouted weight.
    """
    def np_(name):
        return weights[name].numpy().astype(np.float64)

    H = geom.h_pred
    h = h.astype(np.float64).copy()
    c = c.astype(np.float64).copy()
    x = embed.astype(np.float64)

    for layer in range(geom.n_layers):
        p = f"tdt.predict.lstm.{layer}"
        lo, hi = layer * H, (layer + 1) * H
        g = (np_(f"{p}.w_ih") @ x + np_(f"{p}.b_ih") + np_(f"{p}.b_hh")
             + np_(f"{p}.w_hh") @ h[lo:hi])
        sig = lambda v: 1.0 / (1.0 + np.exp(-v))
        i_g, f_g = sig(g[0:H]), sig(g[H:2 * H])
        g_g, o_g = np.tanh(g[2 * H:3 * H]), sig(g[3 * H:4 * H])
        c[lo:hi] = f_g * c[lo:hi] + i_g * g_g
        h[lo:hi] = o_g * np.tanh(c[lo:hi])
        x = h[lo:hi]
    pred = x

    logits = []
    for frame in enc_frames:
        hidden = (np_("tdt.joint.enc.weight") @ frame.astype(np.float64)
                  + np_("tdt.joint.enc.bias")
                  + np_("tdt.joint.pred.weight") @ pred
                  + np_("tdt.joint.pred.bias"))
        hidden = np.maximum(hidden, 0.0)
        logits.append(np_("tdt.joint.out.weight") @ hidden
                      + np_("tdt.joint.out.bias"))
    return np.stack(logits, axis=-1), h, c


def check_parity(model, weights, geom, seed=0):
    rng = np.random.default_rng(seed)
    L, W = model.lanes, model.width
    HL = geom.h_pred * geom.n_layers

    embed = rng.standard_normal((L, geom.h_pred), dtype=np.float32) * 0.1
    h = rng.standard_normal((L, HL), dtype=np.float32) * 0.1
    c = rng.standard_normal((L, HL), dtype=np.float32) * 0.1
    enc = rng.standard_normal((L, geom.d_enc, W), dtype=np.float32)

    with torch.inference_mode():
        logits, h_out, c_out = model(
            torch.from_numpy(embed).view(L, geom.h_pred, 1, 1),
            torch.from_numpy(h).view(L, HL, 1, 1),
            torch.from_numpy(c).view(L, HL, 1, 1),
            torch.from_numpy(enc).view(L, geom.d_enc, 1, W))
    logits = logits.view(L, geom.v_out, W).numpy()
    h_out = h_out.view(L, HL).numpy()
    c_out = c_out.view(L, HL).numpy()

    worst = {}
    for lane in range(L):
        ref_logits, ref_h, ref_c = reference_step(
            weights, geom, embed[lane], h[lane], c[lane],
            [enc[lane, :, t] for t in range(W)])
        for tag, got, want in (("logits", logits[lane], ref_logits),
                               ("h_out", h_out[lane], ref_h),
                               ("c_out", c_out[lane], ref_c)):
            scale = max(float(np.abs(want).max()), 1e-6)
            rel = float(np.abs(got - want).max()) / scale
            worst[tag] = max(worst.get(tag, 0.0), rel)

    print("[check] max relative error vs scalar reference: "
          + "  ".join(f"{k}={v:.2e}" for k, v in sorted(worst.items())))
    # fp32 conv against an fp64 scalar reference; anything above this is a
    # wiring error, not accumulation order.
    if max(worst.values()) > 1e-4:
        print("[check] FAIL: the ANE graph does not match the scalar reference.")
        return False
    print("[check] OK")
    return True


def find_joint_out_ops(mlmodel, v_out):
    """Op names of the convs that produce V_out channels, to exempt them."""
    names = []
    for func in mlmodel._mil_program.functions.values():
        for op in func.operations:
            if op.op_type != "conv":
                continue
            weight = op.inputs.get("weight")
            val = getattr(weight, "val", None)
            if val is not None and val.shape[0] == v_out:
                names.append(op.name)
    return names


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="output .mlpackage path")
    ap.add_argument("--scripts", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--lanes", type=int, default=DEFAULT_LANES,
                    help=f"independent utterance slots per dispatch (default {DEFAULT_LANES})")
    ap.add_argument("--width", type=int, default=DEFAULT_WIDTH,
                    help=f"encoder frames scored per dispatch (default {DEFAULT_WIDTH})")
    ap.add_argument("--nbits", type=int, default=6, choices=[0, 2, 3, 4, 6, 8],
                    help="weight palettization bit width; 0 exports fp16 (default 6)")
    ap.add_argument("--nbits-out", type=int, default=0, choices=[0, 2, 3, 4, 6, 8],
                    help="separate bit width for the joint output projection; "
                         "0 keeps it fp16 (default 0; the published model uses 6)")
    ap.add_argument("--dump-embedding", type=Path, default=None,
                    help="write the token embedding table as fp16 [vocab+1, H_pred] "
                         "for inspection; the engine reads it from the GGUF")
    ap.add_argument("--compile-dir", type=Path, default=None,
                    help="if set, compile the .mlpackage to a .mlmodelc here and report op placement")
    ap.add_argument("--check", action="store_true",
                    help="verify the graph against a scalar reference before converting, "
                         "and stop if it does not match")
    args = ap.parse_args()

    ane = _load_sibling(args.scripts, "export-encoder-coreml-ane.py", "parakeet_ane_encoder")
    ref = _load_sibling(args.scripts, "ref-encoder-from-gguf.py", "parakeet_ref_encoder")
    weights, meta = ref.load_gguf(args.gguf)

    geom = TdtGeometry(weights, meta)
    model = AneTdtDecoder(weights, geom, args.lanes, args.width).eval()
    print(f"[export] {geom}")
    print(f"[export] lanes={args.lanes} width={args.width} "
          f"(one dispatch scores {args.lanes * args.width} frame-slots)")

    if args.check and not check_parity(model, weights, geom):
        return 1

    L, W = args.lanes, args.width
    HL = geom.h_pred * geom.n_layers
    example = (torch.zeros(L, geom.h_pred, 1, 1),
               torch.zeros(L, HL, 1, 1),
               torch.zeros(L, HL, 1, 1),
               torch.zeros(L, geom.d_enc, 1, W))

    with torch.inference_mode():
        out = model(*example)
    print(f"[export] traced outputs: logits={tuple(out[0].shape)} "
          f"h_out={tuple(out[1].shape)} c_out={tuple(out[2].shape)}")

    traced = torch.jit.trace(model, example, check_trace=False)
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="embed",    shape=example[0].shape, dtype=np.float16),
            ct.TensorType(name="h_in",     shape=example[1].shape, dtype=np.float16),
            ct.TensorType(name="c_in",     shape=example[2].shape, dtype=np.float16),
            ct.TensorType(name="enc_step", shape=example[3].shape, dtype=np.float16),
        ],
        outputs=[
            ct.TensorType(name="logits", dtype=np.float16),
            ct.TensorType(name="h_out",  dtype=np.float16),
            ct.TensorType(name="c_out",  dtype=np.float16),
        ],
        compute_units=ct.ComputeUnit.ALL,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
    )

    if args.nbits:
        from coremltools.optimize.coreml import (
            OpPalettizerConfig, OptimizationConfig, palettize_weights,
        )
        out_ops = find_joint_out_ops(mlmodel, geom.v_out)
        op_configs = {
            name: (None if args.nbits_out == 0 else OpPalettizerConfig(
                nbits=args.nbits_out, mode="kmeans",
                granularity="per_grouped_channel",
                group_size=ane.PALETTIZE_GROUP_SIZE))
            for name in out_ops
        }
        config = OptimizationConfig(
            global_config=OpPalettizerConfig(
                nbits=args.nbits, mode="kmeans",
                granularity="per_grouped_channel",
                group_size=ane.PALETTIZE_GROUP_SIZE),
            op_name_configs=op_configs,
        )
        mlmodel = palettize_weights(mlmodel, config)
        kept = "fp16" if args.nbits_out == 0 else f"{args.nbits_out}-bit"
        print(f"[export] palettized weights to {args.nbits}-bit "
              f"(per_grouped_channel, group_size={ane.PALETTIZE_GROUP_SIZE}); "
              f"joint output projection ({', '.join(out_ops)}) kept {kept}")

    mlmodel.save(str(args.out))
    print(f"[export] saved {args.out}")

    # Graph geometry for reference only; the engine does not read this file.
    sidecar = {
        "lanes": args.lanes,
        "width": args.width,
        "h_pred": geom.h_pred,
        "h_joint": geom.h_joint,
        "d_enc": geom.d_enc,
        "pred_rnn_layers": geom.n_layers,
        "vocab_size": geom.vocab,
        "blank_id": geom.blank_id,
        "v_out": geom.v_out,
        "n_durations": geom.n_durations,
        "token_logits": [0, geom.vocab + 1],
        "duration_logits": [geom.vocab + 1, geom.v_out],
    }
    meta_path = args.out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(sidecar, indent=2) + "\n")
    print(f"[export] wrote {meta_path}")

    if args.dump_embedding is not None:
        table = weights["tdt.predict.embed.weight"].numpy().astype(np.float16)
        args.dump_embedding.write_bytes(np.ascontiguousarray(table).tobytes())
        print(f"[export] wrote embedding {table.shape} fp16 to {args.dump_embedding}")

    if args.compile_dir is not None:
        import subprocess
        subprocess.run(["xcrun", "coremlc", "compile", str(args.out), str(args.compile_dir)],
                       check=True)
        compiled = Path(args.compile_dir) / (args.out.stem + ".mlmodelc")
        print(f"[export] compiled .mlmodelc into {args.compile_dir}")
        ane.report_compute_placement(compiled)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
