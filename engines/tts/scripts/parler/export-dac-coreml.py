#!/usr/bin/env python3
"""Export Parler-TTS's DAC decoder from the model GGUF to a Core ML package for
the engine's Apple Neural Engine sidecar.

The decoder -- input conv, four Snake / transposed-conv / residual-unit
stages, Snake, output conv, tanh -- is rebuilt in PyTorch from the same GGUF
tensors the ggml engine loads (dac.cpp), so it needs neither the checkpoint
nor parler_tts. The residual vector quantizer stays on the host: the sidecar
takes the summed latent, [1, latent_dim, window] frames, plus a 0/1 frame mask
[1, 1, window], and returns [1, 1, window * hop] samples.

Every convolution is non-causal ("same" zero padding). The engine walks an
utterance in fixed windows of the exported width with parler_dac_rf_frames of
real context on each side, like the ggml decoder, so one fixed-shape graph
covers every length. Frames past the end of the utterance are marked invalid:
the activation is zeroed there after every op that can make it nonzero, which
is exactly the zero padding a decode of the shorter sequence sees, so a
partial last window matches a whole-sequence decode.

Neural Engine spellings (each measured on Audio8 / MOSS on an M2):
- Snake's sin^2 is a range-reduced even polynomial with 1 / alpha folded into
  its coefficients: the Neural Engine has no sin, x ** 2 lowers to a pow it
  miscomputes, and on the high-rate stages the Snakes, not the convolutions,
  dominate its time (each tensor-by-tensor op is a full pass over memory).
- Every transposed convolution is its exact phase form, a two-tap convolution
  followed by a depth-to-space shuffle; the native ConvTranspose kernel
  miscomputes at stride 4.
- Sequences longer than FOLD_ROW samples are folded into rows of FOLD_ROW,
  (1, C, L) -> (1, C, L / FOLD_ROW, FOLD_ROW), because the Neural Engine drops
  1-D convolutions over more than 16384 samples. Each convolution takes its
  halo from the neighbouring rows (zeros past either end), so the fold is
  exact.

Export and compile next to the GGUF (the engine finds `parler-mini-v1-dac.mlmodelc`
beside `parler-mini-v1-<quant>.gguf`):

  python scripts/parler/export-dac-coreml.py \\
      --gguf models/parler-mini-v1-f32.gguf --compile-dir models

Check the rebuild against dump-reference.py's fixtures first:

  python scripts/parler/export-dac-coreml.py \\
      --gguf models/parler-mini-v1-f32.gguf --parity-dir artifacts/parler-ref --no-export
"""
import argparse
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ARCH = "parler"
QUANT_TAGS = ("f32", "f16", "bf16", "q8_0", "q5_0", "q4_0")
SIDECAR_SUFFIX = "-dac"
RESIDUAL_DILATIONS = (1, 3, 9)
EDGE_KERNEL = 7
SNAKE_EPS = 1e-9
DEFAULT_WINDOW = 128
PARITY_COSINE = 0.9999
COSINE_EPS = 1e-12
HALF_SAFE_INVERSE = 6.0e4
# sin^2(pi f) ~= sum_k SIN2_POLY[k] f^(2k + 2) on [-1/2, 1/2] (least squares,
# max error 6.2e-5; 1 / alpha <= 8 in the Parler checkpoints)
SIN2_POLY = (9.8672466, -32.37042703, 41.426299, -23.29808439)
FOLD_ROW = 8192
# Longest 1-D convolution row the Neural Engine keeps (M2, measured on Audio8)
ANE_ROW_LIMIT = 16384


def constant(array):
    return nn.Parameter(torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32)),
                        requires_grad=False)


def read_tensor(tensor):
    from gguf.quants import dequantize

    shape = [int(d) for d in reversed(tensor.shape)]
    if tensor.tensor_type.name in ("F32", "F16"):
        return np.asarray(tensor.data).astype(np.float32).reshape(shape)
    return dequantize(np.asarray(tensor.data), tensor.tensor_type).astype(np.float32).reshape(shape)


def read_gguf(path):
    from gguf import GGUFReader

    reader = GGUFReader(str(path))
    if reader.fields["general.architecture"].contents() != ARCH:
        raise ValueError(f"{path} is not a {ARCH} GGUF")

    def field(name):
        return reader.fields[f"{ARCH}.dac.{name}"].contents()

    hp = {
        "hop": int(field("hop")),
        "latent_dim": int(field("latent_dim")),
        "decoder_dim": int(field("decoder_dim")),
        "rates": [int(r) for r in field("rates")],
        "n_q": int(field("n_quantizers")),
        "codebook_size": int(field("codebook_size")),
        "sample_rate": int(field("sample_rate")),
    }
    if int(np.prod(hp["rates"])) != hp["hop"]:
        raise ValueError(f"dac rates {hp['rates']} do not multiply to the hop {hp['hop']}")
    tensors = {t.name: read_tensor(t) for t in reader.tensors if t.name.startswith("dac.")}
    return tensors, hp


def receptive_field_frames(t, hp):
    """parler_dac_rf_frames (dac.cpp): the support radius in latent frames."""
    r = (t["dac.dec.conv_in.weight"].shape[-1] - 1) // 2
    for block, s in enumerate(hp["rates"]):
        r = r * s + (3 * s + 1) // 2
        for j, d in enumerate(RESIDUAL_DILATIONS):
            r += (t[f"dac.dec.blk.{block}.res.{j}.conv1.weight"].shape[-1] - 1) // 2 * d
    r += (t["dac.dec.conv_out.weight"].shape[-1] - 1) // 2
    return -(-r // hp["hop"])


def rvq_latent(t, hp, codes):
    """The quantizer's summed projection, [n_q, frames] codes -> [latent_dim, frames]."""
    latent = np.zeros((hp["latent_dim"], codes.shape[1]), dtype=np.float32)
    for k in range(hp["n_q"]):
        vectors = t[f"dac.quant.{k}.codebook.weight"][codes[k]]          # [frames, cb_dim]
        proj = t[f"dac.quant.{k}.out_proj.weight"].reshape(hp["latent_dim"], -1)
        latent += proj @ vectors.T + t[f"dac.quant.{k}.out_proj.bias"].reshape(-1, 1)
    return latent


def folds(length):
    """Whether a sequence of `length` samples is folded into rows of FOLD_ROW:
    longer than a row and evenly divisible. Shapes are static Python ints so
    the traced graph carries only constant reshapes."""
    return bool(FOLD_ROW) and length > FOLD_ROW and length % FOLD_ROW == 0


def fold(x, length):
    """(1, C, length) -> (1, C, length / FOLD_ROW, FOLD_ROW) when folds(length)."""
    if x.dim() == 3 and folds(length):
        return x.reshape(1, -1, length // FOLD_ROW, FOLD_ROW)
    return x


def unfold(x, length):
    if x.dim() == 4:
        return x.reshape(1, -1, length)
    return x


def halo(x, size):
    """Zero "same" padding of `size` samples on each side of the sequence. A
    folded (1, C, R, W) tensor takes each row's halo from the end of the
    previous row and the start of the next one (zeros past either end)."""
    if size == 0:
        return x
    if x.dim() == 3:
        return F.pad(x, (size, size))
    # shift only the edge columns between rows, so the whole activation is
    # copied once (the final concat), not once per shifted operand
    zero = torch.zeros_like(x[:, :, :1, :size])
    prev = torch.cat([zero, x[:, :, :-1, -size:]], dim=2)
    nxt = torch.cat([x[:, :, 1:, :size], zero], dim=2)
    return torch.cat([prev, x, nxt], dim=3)


def per_channel(param, x):
    return param if x.dim() == 3 else param.unsqueeze(-1)


class Snake(nn.Module):
    """x + sin^2(alpha x) / alpha.

    "poly" (the default) spells it with no sine and as few tensor-by-tensor
    ops as possible, the only ones that cost anything on the Neural Engine
    (per-channel constant multiplies and adds fuse): f = alpha x / pi reduced
    to [-1/2, 1/2] by round, then sin^2(pi f) as an even polynomial in f with
    1 / alpha folded into its per-channel coefficients. That is round, one
    subtraction, f * f, three multiplies and the residual add."""
    mode = "poly"

    def __init__(self, alpha):
        super().__init__()
        alpha = alpha.reshape(1, -1, 1)
        inv = np.clip(1.0 / (alpha.astype(np.float64) + SNAKE_EPS), -HALF_SAFE_INVERSE, HALF_SAFE_INVERSE)
        self.alpha = constant(alpha)
        self.inv = constant(inv)
        self.scale = constant(alpha / math.pi)
        self.coef = nn.ParameterList(constant(inv * c) for c in SIN2_POLY)

    def forward(self, x):
        if Snake.mode == "sin":
            s = torch.sin(per_channel(self.alpha, x) * x)
            return x + per_channel(self.inv, x) * (s * s)
        u = x * per_channel(self.scale, x)
        f = u - torch.round(u)
        f2 = f * f
        p = f2 * per_channel(self.coef[-1], x)
        for k in range(len(self.coef) - 2, -1, -1):
            p = f2 * (per_channel(self.coef[k], x) + p)
        return x + p


class Conv(nn.Module):
    """A "same"-padded Conv1d over a flat or folded sequence."""

    def __init__(self, t, name, dilation=1):
        super().__init__()
        self.weight = constant(t[name + ".weight"])                     # [OC, IC, K]
        self.bias = constant(t[name + ".bias"].reshape(-1))
        self.dilation = dilation
        self.pad = (self.weight.shape[-1] - 1) * dilation // 2

    def forward(self, x):
        if x.dim() == 4:
            return F.conv2d(halo(x, self.pad), self.weight.unsqueeze(2), self.bias,
                            dilation=(1, self.dilation))
        return F.conv1d(halo(x, self.pad), self.weight, self.bias, dilation=self.dilation)


def phase_kernel(weight, stride):
    """ConvTranspose1d [IC, OC, 2 * stride] -> the two-tap Conv1d
    [stride * OC, IC, 2] whose depth-to-space output, cropped by stride / 2,
    is the transposed convolution trimmed by stride / 2 on each side."""
    in_ch, out_ch, taps = weight.shape
    if taps != 2 * stride:
        raise ValueError(f"transposed kernel of {taps} taps is not twice the stride {stride}")
    phase = np.zeros((stride * out_ch, in_ch, 2), dtype=np.float32)
    for r in range(stride):
        phase[r * out_ch:(r + 1) * out_ch, :, 0] = weight[:, :, r + stride].T
        phase[r * out_ch:(r + 1) * out_ch, :, 1] = weight[:, :, r].T
    return phase


class Upsample(nn.Module):
    def __init__(self, t, name, stride, length):
        super().__init__()
        bias = t[name + ".bias"].reshape(-1)
        self.out_ch = bias.shape[0]
        self.weight = constant(phase_kernel(t[name + ".weight"], stride))
        self.bias = constant(bias.reshape(1, -1, 1))
        self.stride = stride
        self.length = length
        self.crop = (stride + 1) // 2

    def forward(self, x):
        s, oc = self.stride, self.out_ch
        if x.dim() == 4:
            # Each row of FOLD_ROW samples yields FOLD_ROW + 1 phase columns
            # from its one-sample halo, (FOLD_ROW + 1) * s samples, of which
            # [crop, crop + FOLD_ROW * s) are the row's own; that row is then
            # split into s rows of FOLD_ROW.
            rows, width = self.length // FOLD_ROW, FOLD_ROW
            z = F.conv2d(halo(x, 1), self.weight.unsqueeze(2))
            z = z.reshape(1, s, oc, rows, width + 1).permute(0, 2, 3, 4, 1)
            z = z.reshape(1, oc, rows, (width + 1) * s)[:, :, :, self.crop:self.crop + width * s]
            return z.reshape(1, oc, rows * s, width) + self.bias.unsqueeze(-1)
        z = F.conv1d(halo(x, 1), self.weight)
        z = z.reshape(1, s, oc, self.length + 1).permute(0, 2, 3, 1).reshape(1, oc, (self.length + 1) * s)
        return fold(z[:, :, self.crop:self.crop + self.length * s] + self.bias, self.length * s)


class ResidualUnit(nn.Module):
    def __init__(self, t, name, dilation):
        super().__init__()
        self.snake1 = Snake(t[name + ".snake1.alpha"])
        self.conv1 = Conv(t, name + ".conv1", dilation)
        self.snake2 = Snake(t[name + ".snake2.alpha"])
        self.conv2 = Conv(t, name + ".conv2")

    def forward(self, x, mask):
        return (x + self.conv2(self.snake2(self.conv1(self.snake1(x))))) * mask


class DecoderBlock(nn.Module):
    def __init__(self, t, block, stride, length):
        super().__init__()
        name = f"dac.dec.blk.{block}"
        self.snake = Snake(t[name + ".snake.alpha"])
        self.up = Upsample(t, name + ".convt", stride, length)
        self.units = nn.ModuleList(ResidualUnit(t, f"{name}.res.{j}", d)
                                   for j, d in enumerate(RESIDUAL_DILATIONS))

    def forward(self, x, mask):
        x = self.up(self.snake(x)) * mask
        for unit in self.units:
            x = unit(x, mask)
        return x


def rate_mask(valid, frames, repeat):
    """valid [1, 1, frames] -> [1, 1, frames * repeat], folded like the activation."""
    if repeat == 1:
        return valid
    expanded = valid.reshape(1, 1, frames, 1).expand(1, 1, frames, repeat)
    return fold(expanded.reshape(1, 1, frames * repeat), frames * repeat)


class DacDecoder(nn.Module):
    """latent [1, latent_dim, W] + valid [1, 1, W] -> pcm [1, 1, W * hop].

    The activation is zeroed at invalid frames after the input conv, every
    transposed conv and every residual sum, so each "same" convolution sees
    the zero padding of a sequence that ends at the last valid frame (Snake
    and the 1-tap conv2 map a zero column to itself or keep it local)."""

    def __init__(self, t, hp, window):
        super().__init__()
        widest = (EDGE_KERNEL - 1) // 2 * max(RESIDUAL_DILATIONS)
        if FOLD_ROW and FOLD_ROW < widest:
            raise ValueError(f"fold rows of {FOLD_ROW} samples cannot hold a {widest}-sample convolution halo")
        self.rates = hp["rates"]
        self.window = window
        for rate_product in np.cumprod([1] + list(self.rates)):
            length = window * int(rate_product)
            if FOLD_ROW and length > ANE_ROW_LIMIT and length % FOLD_ROW:
                raise ValueError(f"a {window}-frame window leaves a {length}-sample stage that neither fits a "
                                 f"{ANE_ROW_LIMIT}-sample row nor folds into rows of {FOLD_ROW}; use a window "
                                 f"whose stage lengths divide by {FOLD_ROW}, e.g. a multiple of 128 frames")
        self.samples = window * hp["hop"]
        lengths = [window * int(np.prod(self.rates[:b])) for b in range(len(self.rates))]
        self.conv_in = Conv(t, "dac.dec.conv_in")
        self.blocks = nn.ModuleList(DecoderBlock(t, b, s, lengths[b]) for b, s in enumerate(self.rates))
        self.snake_out = Snake(t["dac.dec.snake_out.alpha"])
        self.conv_out = Conv(t, "dac.dec.conv_out")

    def forward(self, latent, valid):
        x = self.conv_in(latent * valid) * valid
        repeat = 1
        for block, stride in zip(self.blocks, self.rates):
            repeat *= stride
            x = block(x, rate_mask(valid, self.window, repeat))
        return unfold(torch.tanh(self.conv_out(self.snake_out(x))), self.samples)


def strip_quant_tag(stem):
    for sep in ("-", "."):
        head, _, tag = stem.rpartition(sep)
        if head and tag.lower() in QUANT_TAGS:
            return head
    return stem


def sidecar_stem(gguf_path):
    return strip_quant_tag(Path(gguf_path).stem) + SIDECAR_SUFFIX


def windows(n_frames, window, rf):
    """The engine's walk: (w0, keep0, keep1) per call, each window `window`
    frames from w0 with rf frames of real context around [keep0, keep1)."""
    keep = window - 2 * rf
    if keep <= 0:
        raise ValueError(f"window {window} leaves no frames after {rf} context frames per side")
    out = []
    for a in range(0, n_frames, keep):
        b = min(a + keep, n_frames)
        out.append((max(0, a - rf), a, b))
    return out


def decode_windowed(run, latent, window, rf, hop):
    """Decodes [latent_dim, frames] through `run(latent_win, valid_win)` in
    fixed windows and stitches the kept spans, as the engine does."""
    dim, frames = latent.shape
    pcm = []
    for w0, a, b in windows(frames, window, rf):
        w1 = min(frames, w0 + window)
        lat = np.zeros((1, dim, window), dtype=np.float32)
        lat[0, :, :w1 - w0] = latent[:, w0:w1]
        valid = np.zeros((1, 1, window), dtype=np.float32)
        valid[0, 0, :w1 - w0] = 1.0
        out = run(lat, valid).reshape(-1)
        pcm.append(out[(a - w0) * hop:(b - w0) * hop])
    return np.concatenate(pcm)


def compare(tag, got, want):
    n = min(got.size, want.size)
    a = got.reshape(-1)[:n].astype(np.float64)
    b = want.reshape(-1)[:n].astype(np.float64)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + COSINE_EPS))
    print(f"[parity] {tag}: n={n} cosine={cos:.8f} max_abs_err={float(np.abs(a - b).max()):.3e}")
    return cos


def torch_runner(model):
    def run(latent, valid):
        with torch.no_grad():
            return model(torch.from_numpy(latent), torch.from_numpy(valid)).numpy()
    return run


def run_parity(model, t, hp, ref, window, rf):
    ref = Path(ref)
    worst = 1.0
    codes = np.load(ref / "dacrand_codes.npy")[0]
    worst = min(worst, compare("rvq latent (dacrand)", rvq_latent(t, hp, codes), np.load(ref / "dacrand_latent.npy")[0]))
    run = torch_runner(model)
    for case, wav in (("dacrand", "dacrand_wav"), ("case0", "case0_wav_greedy"), ("case1", "case1_wav_greedy")):
        name = "dacrand_latent" if case == "dacrand" else f"{case}_dac_latent"
        if not (ref / f"{name}.npy").exists():
            continue
        latent = np.load(ref / f"{name}.npy")[0]
        pcm = decode_windowed(run, latent, window, rf, hp["hop"])
        worst = min(worst, compare(f"{case} windowed W={window}", pcm, np.load(ref / f"{wav}.npy")))
    if worst < PARITY_COSINE:
        raise SystemExit("[parity] the PyTorch rebuild does not match the reference pipeline")


def convert_coreml(model, hp, window, precision):
    import coremltools as ct

    shapes = {"latent": (1, hp["latent_dim"], window), "valid": (1, 1, window)}
    with torch.no_grad():
        traced = torch.jit.trace(model, tuple(torch.zeros(s) for s in shapes.values()))
    half = precision == "float16"
    io_dtype = np.float16 if half else np.float32
    return ct.convert(
        traced,
        inputs=[ct.TensorType(name=n, shape=s, dtype=io_dtype) for n, s in shapes.items()],
        outputs=[ct.TensorType(name="pcm", dtype=io_dtype)],
        minimum_deployment_target=ct.target.macOS14,
        compute_precision=ct.precision.FLOAT16 if half else ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.ALL,
        convert_to="mlprogram",
    )


def compile_model(package_path, compile_dir):
    Path(compile_dir).mkdir(parents=True, exist_ok=True)
    subprocess.run(["xcrun", "coremlcompiler", "compile", str(package_path), str(compile_dir)],
                   capture_output=True, text=True, check=True)
    return Path(compile_dir) / (Path(package_path).stem + ".mlmodelc")


def report_placement(compiled):
    import coremltools as ct

    plan = ct.models.compute_plan.MLComputePlan.load_from_path(str(compiled), compute_units=ct.ComputeUnit.ALL)
    counts, cost = {}, {}
    for op in plan.model_structure.program.functions["main"].block.operations:
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        if usage is None:
            continue
        device = type(usage.preferred_compute_device).__name__
        counts[device] = counts.get(device, 0) + 1
        estimate = plan.get_estimated_cost_for_mlprogram_operation(op)
        if estimate is not None:
            cost[device] = cost.get(device, 0.0) + estimate.weight
    total = sum(cost.values()) or 1.0
    share = {d: f"{100.0 * w / total:.1f}%" for d, w in cost.items()}
    print(f"[placement] ops {counts} estimated cost {share}")


def coreml_parity(package, t, hp, ref, window, rf, units):
    import coremltools as ct

    start = time.time()
    mlmodel = ct.models.MLModel(str(package), compute_units=getattr(ct.ComputeUnit, units))
    print(f"[coreml] {units} loaded in {time.time() - start:.1f}s")
    dtype = np.float16 if mlmodel.get_spec().description.input[0].type.multiArrayType.dataType == \
        ct.proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT16 else np.float32

    def run(latent, valid):
        out = mlmodel.predict({"latent": latent.astype(dtype), "valid": valid.astype(dtype)})
        return np.asarray(out["pcm"], dtype=np.float32)

    latent = np.load(Path(ref) / "case0_dac_latent.npy")[0]
    lat = np.zeros((1, hp["latent_dim"], window), dtype=np.float32)
    lat[0, :, :min(window, latent.shape[1])] = latent[:, :window]
    valid = np.ones((1, 1, window), dtype=np.float32)
    for _ in range(2):
        run(lat, valid)
    times = []
    for _ in range(5):
        start = time.perf_counter()
        run(lat, valid)
        times.append(time.perf_counter() - start)
    print(f"[coreml] {units} window {window}: {1e3 * float(np.median(times)):.1f} ms/call")
    pcm = decode_windowed(run, latent, window, rf, hp["hop"])
    compare(f"coreml {units} case0", pcm, np.load(Path(ref) / "case0_wav_greedy.npy"))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gguf", required=True, help="a Parler-TTS GGUF (any quantization; the DAC is F32)")
    p.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                   help="latent frames per sidecar call, context included (default %(default)s)")
    p.add_argument("--fold-rows", type=int, default=FOLD_ROW,
                   help="fold sequences longer than this many samples into rows of that width so the "
                        "Neural Engine keeps them (16384 is its limit on an M2); 0 leaves them flat")
    p.add_argument("--snake", default="poly", choices=["poly", "sin"],
                   help="sin^2 spelling: a range-reduced polynomial (Neural Engine) or torch.sin")
    p.add_argument("--precision", default="float16", choices=["float16", "float32"],
                   help="compute and I/O precision of the exported program (default %(default)s)")
    p.add_argument("--out", help=".mlpackage path (default: <sidecar stem>.mlpackage next to the GGUF)")
    p.add_argument("--compile-dir", help="also compile to <dir>/<stem>.mlmodelc, where the engine looks")
    p.add_argument("--parity-dir", help="dump-reference.py output to check the rebuild against")
    p.add_argument("--coreml-parity", nargs="*", default=[],
                   choices=["ALL", "CPU_AND_NE", "CPU_AND_GPU", "CPU_ONLY"],
                   help="after export, decode case0 through the package on these compute units")
    p.add_argument("--no-export", action="store_true", help="only run the parity check")
    return p.parse_args()


def main():
    global FOLD_ROW
    args = parse_args()
    Snake.mode = args.snake
    FOLD_ROW = args.fold_rows
    tensors, hp = read_gguf(args.gguf)
    model = DacDecoder(tensors, hp, args.window).eval()
    rf = receptive_field_frames(tensors, hp)
    print(f"[load] DAC from {args.gguf}: rates {hp['rates']} hop {hp['hop']}, "
          f"{rf} context frames per side, window {args.window}")
    if args.parity_dir:
        run_parity(model, tensors, hp, args.parity_dir, args.window, rf)
    if args.no_export:
        return
    out = Path(args.out) if args.out else Path(args.gguf).with_name(sidecar_stem(args.gguf) + ".mlpackage")
    start = time.time()
    mlmodel = convert_coreml(model, hp, args.window, args.precision)
    mlmodel.save(str(out))
    print(f"[export] saved {out} in {time.time() - start:.1f}s")
    if args.compile_dir:
        compiled = compile_model(out, args.compile_dir)
        print(f"[export] compiled {compiled}")
        report_placement(compiled)
    for units in args.coreml_parity:
        coreml_parity(out, tensors, hp, args.parity_dir, args.window, rf, units)


if __name__ == "__main__":
    main()
