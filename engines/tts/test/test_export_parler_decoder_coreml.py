#!/usr/bin/env python3
"""Model-free unit tests for scripts/parler/export-decoder-coreml.py.

A tiny Parler decoder is built from random tensors in the GGUF's names and
torch layouts and checked against an independent plain-PyTorch decoder written
from the definitions (pre-norm layers, causal self-attention over the whole
sequence, cross-attention, GELU FFN, final LayerNorm, per-codebook heads). The
exported step, driven like the engine drives the package -- read-only cache
states, the current position scored separately, one row written by the host
after each call -- must reproduce the full-sequence decode at every position,
for per-layer and stacked state groupings. With coremltools, the converted
program must read every state and write none. Requires torch; skips without it.
"""
import importlib.util
import contextlib
import io
import json
import pathlib
import platform
import subprocess
import tempfile
import types
import unittest
from unittest import mock

try:
    import numpy as np
    import torch
    import torch.nn.functional as F
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

try:
    import coremltools  # noqa: F401
    HAVE_COREMLTOOLS = True
except ImportError:
    HAVE_COREMLTOOLS = False

SCRIPT = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "parler" / "export-decoder-coreml.py"

LAYERS = 2
D = 32
HEADS = 4
FF = 48
N_Q = 3
VOCAB = 44
CTX = 16
CROSS = 8
EPS = 1e-5


def load_exporter():
    spec = importlib.util.spec_from_file_location("export_parler_decoder_coreml", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hparams():
    return {"n_layer": LAYERS, "d_model": D, "n_head": HEADS, "d_ff": FF, "n_q": N_Q, "vocab": VOCAB,
            "ln_eps": EPS, "max_position": 64, "max_length": 24, "bos": VOCAB - 3 + 1, "variant": "tiny"}


def tiny_tensors(rng):
    def w(out, inp):
        return (rng.standard_normal((out, inp)) / np.sqrt(inp)).astype(np.float32)

    def norm(name, t):
        t[name + ".weight"] = (1 + 0.1 * rng.standard_normal(D)).astype(np.float32)
        t[name + ".bias"] = (0.1 * rng.standard_normal(D)).astype(np.float32)

    t = {}
    for l in range(LAYERS):
        p = f"dec.blk.{l}."
        for n in ("attn_q", "attn_k", "attn_v", "attn_o", "cross_q", "cross_k", "cross_v", "cross_o"):
            t[p + n + ".weight"] = w(D, D)
        t[p + "ffn_up.weight"] = w(FF, D)
        t[p + "ffn_down.weight"] = w(D, FF)
        for n in ("attn_norm", "cross_norm", "ffn_norm"):
            norm(p + n, t)
    norm("dec.output_norm", t)
    for k in range(N_Q):
        t[f"dec.lm_heads.{k}.weight"] = w(VOCAB, D)
    return t


def reference_decode(t, x, cross_states):
    """Full-sequence causal decode, [S, D] inputs -> [S, N_Q, VOCAB] logits."""
    x = torch.from_numpy(x)
    c = torch.from_numpy(cross_states)
    s, hd = x.shape[0], D // HEADS
    g = {k: torch.from_numpy(v) for k, v in t.items()}

    def ln(y, name):
        return F.layer_norm(y, (D,), g[name + ".weight"], g[name + ".bias"], EPS)

    def attn(q, k, v, causal):
        q, k, v = (z.reshape(z.shape[0], HEADS, hd).transpose(0, 1) for z in (q, k, v))
        scores = q @ k.transpose(-1, -2) / hd ** 0.5
        if causal:
            scores = scores.masked_fill(torch.ones(s, s, dtype=torch.bool).triu(1), float("-inf"))
        return (torch.softmax(scores, -1) @ v).transpose(0, 1).reshape(-1, D)

    for l in range(LAYERS):
        p = f"dec.blk.{l}."
        y = ln(x, p + "attn_norm")
        x = x + attn(y @ g[p + "attn_q.weight"].T, y @ g[p + "attn_k.weight"].T,
                     y @ g[p + "attn_v.weight"].T, True) @ g[p + "attn_o.weight"].T
        y = ln(x, p + "cross_norm")
        x = x + attn(y @ g[p + "cross_q.weight"].T, c @ g[p + "cross_k.weight"].T,
                     c @ g[p + "cross_v.weight"].T, False) @ g[p + "cross_o.weight"].T
        y = ln(x, p + "ffn_norm")
        x = x + F.gelu(y @ g[p + "ffn_up.weight"].T) @ g[p + "ffn_down.weight"].T
    x = ln(x, "dec.output_norm")
    return torch.stack([x @ g[f"dec.lm_heads.{k}.weight"].T for k in range(N_Q)], 1).numpy()


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class DecoderStepTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ex = load_exporter()
        rng = np.random.default_rng(7)
        cls.t = tiny_tensors(rng)
        cls.x = rng.standard_normal((CTX - 2, D)).astype(np.float32)
        cls.cross_states = rng.standard_normal((CROSS - 3, D)).astype(np.float32)
        cls.want = reference_decode(cls.t, cls.x, cls.cross_states)

    def stepped(self, layers_per_state, k_layout="rows"):
        ex = self.ex
        model = ex.DecoderStep(self.t, hparams(), CTX, CROSS, torch.float32, layers_per_state, k_layout).eval()
        ks, vs, cmask = ex.cross_kv(self.t, hparams(), self.cross_states, CROSS)
        session = ex.TorchSession(model)
        session.set_cross(ks, vs)
        return np.stack([session.step(row, n, cmask) for n, row in enumerate(self.x)])

    def test_step_matches_full_sequence_per_layer_states(self):
        np.testing.assert_allclose(self.stepped(1), self.want, atol=2e-5, rtol=0)

    def test_step_matches_full_sequence_stacked_states(self):
        np.testing.assert_allclose(self.stepped(LAYERS), self.want, atol=2e-5, rtol=0)

    def test_step_matches_full_sequence_transposed_keys(self):
        np.testing.assert_allclose(self.stepped(LAYERS, "cols"), self.want, atol=2e-5, rtol=0)

    def test_new_kv_are_unscaled_projections(self):
        ex = self.ex
        model = ex.DecoderStep(self.t, hparams(), CTX, CROSS, torch.float32).eval()
        x = torch.from_numpy(self.x[0].reshape(1, D, 1, 1))
        with torch.no_grad():
            _, nk, nv = model(x, torch.from_numpy(ex.self_mask(0, CTX)),
                              torch.from_numpy(ex.self_mask(CROSS, CROSS)))
        y = F.layer_norm(torch.from_numpy(self.x[0]), (D,), torch.from_numpy(self.t["dec.blk.0.attn_norm.weight"]),
                         torch.from_numpy(self.t["dec.blk.0.attn_norm.bias"]), EPS).numpy()
        hd = D // HEADS
        np.testing.assert_allclose(nk[0, :, 0].numpy(), (self.t["dec.blk.0.attn_k.weight"] @ y).reshape(HEADS, hd),
                                   atol=1e-5)
        np.testing.assert_allclose(nv[0, :, 0].numpy(), (self.t["dec.blk.0.attn_v.weight"] @ y).reshape(HEADS, hd),
                                   atol=1e-5)
        self.assertEqual(tuple(nk.shape), (LAYERS, HEADS, 1, hd))

    def test_grouping_must_divide_layers(self):
        with self.assertRaises(ValueError):
            self.ex.DecoderStep(self.t, hparams(), CTX, CROSS, torch.float32, 3)

    @unittest.skipUnless(HAVE_COREMLTOOLS and platform.system() == "Darwin" and
                         int(platform.mac_ver()[0].split(".")[0] or 0) >= 15,
                         "Core ML execution requires macOS 15+")
    def test_channel_norm_coreml_cpu_is_stable_for_random_constant_and_wide_inputs(self):
        import coremltools as ct

        rng = np.random.default_rng(813)
        weight = (1 + rng.standard_normal(D) * 0.1).astype(np.float32)
        bias = (rng.standard_normal(D) * 0.1).astype(np.float32)
        norm = self.ex.ChannelNorm(weight, bias, EPS).eval()
        traced = torch.jit.trace(norm, torch.zeros(1, D, 1, 1))
        ml = ct.convert(traced, inputs=[ct.TensorType(name="hidden", shape=(1, D, 1, 1), dtype=np.float16)],
                        outputs=[ct.TensorType(name="normalized", dtype=np.float16)],
                        minimum_deployment_target=ct.target.macOS15, compute_precision=ct.precision.FLOAT16,
                        compute_units=ct.ComputeUnit.CPU_ONLY, convert_to="mlprogram")
        ops = ml._mil_program.functions["main"].find_ops(op_type="layer_norm")
        self.assertEqual(len(ops), 1)
        self.assertEqual(list(ops[0].axes.val), [-1])
        inputs = [(f"random-{i}", rng.standard_normal(D).astype(np.float16)) for i in range(3)]
        inputs += [("constant-zero", np.zeros(D, np.float16)),
                   ("constant-nonzero", np.full(D, 3.5, np.float16)),
                   ("wide", np.tile(np.array([-1024, 1024], np.float16), D // 2))]
        for name, value in inputs:
            with self.subTest(input=name):
                want = F.layer_norm(torch.from_numpy(value.astype(np.float32)), (D,),
                                    torch.from_numpy(weight), torch.from_numpy(bias), EPS).numpy()
                got = np.asarray(ml.predict({"hidden": value.reshape(1, D, 1, 1)})["normalized"],
                                 dtype=np.float32).reshape(D)
                self.assertTrue(np.isfinite(got).all())
                np.testing.assert_allclose(got, want, atol=0.02, rtol=0)
                print(f"Core ML CPU normalization input={name} peak={float(np.max(np.abs(got - want))):.6f}")

    def test_cross_kv_masks_and_refuses_overflow(self):
        ks, vs, mask = self.ex.cross_kv(self.t, hparams(), self.cross_states, CROSS)
        n = self.cross_states.shape[0]
        self.assertEqual(ks[0].shape, (1, HEADS, CROSS, D // HEADS))
        self.assertTrue((mask[..., :n] == 0).all() and (mask[..., n:] == self.ex.MASK_OFF).all())
        self.assertTrue((ks[1][:, :, n:] == 0).all() and (vs[1][:, :, n:] == 0).all())
        with self.assertRaises(ValueError):
            self.ex.cross_kv(self.t, hparams(), np.zeros((CROSS + 1, D), np.float32), CROSS)

    def test_sidecar_name_drops_quant_tag(self):
        self.assertEqual(self.ex.sidecar_stem("m/parler-mini-v1-q8_0.gguf"), "parler-mini-v1-decoder")
        self.assertEqual(self.ex.sidecar_stem("parler-indic-f32.gguf"), "parler-indic-decoder")
        self.assertEqual(self.ex.sidecar_stem("parler-mini-v1.Q8_0.gguf"), "parler-mini-v1-decoder")
        self.assertEqual(self.ex.sidecar_stem("parler-mini-v1-F16.gguf"), "parler-mini-v1-decoder")

    def test_parity_requires_at_least_one_matching_case(self):
        with tempfile.TemporaryDirectory() as ref:
            # An unrelated file must not make a missing reference set pass.
            np.save(pathlib.Path(ref) / "other_step_logits.npy", np.zeros((1, N_Q, VOCAB)))
            with self.assertRaisesRegex(ValueError, "no matching case"):
                self.ex.run_parity(None, {}, {}, ref)

    def test_parity_accepts_one_available_case_and_checks_tolerance(self):
        logits = np.zeros((1, N_Q, VOCAB), np.float32)
        model = types.SimpleNamespace(ctx=1, cross=1)
        with tempfile.TemporaryDirectory() as ref:
            np.save(pathlib.Path(ref) / "case1_step_logits.npy", logits)
            np.save(pathlib.Path(ref) / "case1_cross_states.npy", np.zeros((1, 1, D), np.float32))
            with mock.patch.object(self.ex, "teacher_inputs", return_value=([np.zeros(D)], 0, None)), \
                    mock.patch.object(self.ex, "cross_kv", return_value=([], [], None)), \
                    mock.patch.object(self.ex, "TorchSession") as session, contextlib.redirect_stdout(io.StringIO()):
                session.return_value.step.return_value = logits[0]
                self.ex.run_parity(model, {}, {}, ref)
                session.return_value.step.return_value = logits[0] + 1
                with self.assertRaisesRegex(SystemExit, "rebuild differs"):
                    self.ex.run_parity(model, {}, {}, ref)

    def test_cli_rejects_invalid_combinations_before_reading_weights(self):
        invalid = ((["--layers", "1", "--out", "tiny.mlpackage", "--parity-dir", "ref"], "parity requires"),
                   (["--quality-steps", "1"], "requires --parity-dir"),
                   (["--no-export"], "requires --parity-dir"),
                   (["--layers", "0"], "must be positive"),
                   (["--layers", "-1"], "must be positive"),
                   (["--layers", "1"], "name it with --out"),
                   (["--quality-steps", "-1"], "must be nonnegative"),
                   (["--no-export", "--parity-dir", "ref", "--quality-steps", "1"], "requires export"),
                   (["--ffn-down-fp16", "--weights", "fp16"], "requires compressed --weights"),
                   (["--ffn-down-fp16", "--no-export", "--parity-dir", "ref"], "requires export"))
        for argv, message in invalid:
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()) as stderr:
                with self.assertRaises(SystemExit) as error:
                    self.ex.parse_args(["--gguf", "unused.gguf"] + argv)
                self.assertEqual(error.exception.code, 2)
                self.assertIn(message, stderr.getvalue())
        args = self.ex.parse_args(["--gguf", "unused.gguf", "--parity-dir", "ref", "--no-export"])
        self.assertTrue(args.no_export)

    def test_ffn_down_cli_composes_with_heads_for_every_compression_mode(self):
        for weights in self.ex.WEIGHT_MODES:
            if weights == "fp16":
                continue
            for heads in ([], ["--heads-fp16"]):
                args = self.ex.parse_args(["--gguf", "unused.gguf", "--weights", weights,
                                           "--ffn-down-fp16"] + heads)
                self.assertTrue(args.ffn_down_fp16)
                self.assertEqual(args.heads_fp16, bool(heads))
        self.assertFalse(self.ex.parse_args(["--gguf", "unused.gguf"]).ffn_down_fp16)
        with self.assertRaisesRegex(ValueError, "requires compressed --weights"):
            self.ex.convert_coreml(None, "fp16", False, {}, ffn_down_fp16=True)

    def test_layers_exceeding_model_fails_before_building_decoder(self):
        args = self.ex.parse_args(["--gguf", "unused.gguf", "--layers", "3", "--out", "tiny.mlpackage"])
        with mock.patch.object(self.ex, "parse_args", return_value=args), \
                mock.patch.object(self.ex, "read_gguf", return_value=({}, hparams())), \
                mock.patch.object(self.ex, "DecoderStep") as decoder:
            with self.assertRaisesRegex(SystemExit, "exceeds the decoder's 2 layers"):
                self.ex.main()
            decoder.assert_not_called()

    def test_main_passes_composed_precision_options_to_conversion(self):
        args = self.ex.parse_args(["--gguf", "unused.gguf", "--ctx", str(CTX),
                                   "--out", "tiny.mlpackage", "--heads-fp16", "--ffn-down-fp16"])
        with mock.patch.object(self.ex, "parse_args", return_value=args), \
                mock.patch.object(self.ex, "read_gguf", return_value=(self.t, hparams())), \
                mock.patch.object(self.ex, "convert_coreml") as convert, \
                contextlib.redirect_stdout(io.StringIO()):
            self.ex.main()
        self.assertEqual(convert.call_args.args[1:3], ("int8", True))
        self.assertTrue(convert.call_args.args[4])
        self.assertEqual(convert.call_args.args[3]["weights"], "int8+fp16-heads")
        convert.return_value.save.assert_called_once_with("tiny.mlpackage")

    def test_compile_failure_includes_compiler_stderr(self):
        failure = subprocess.CalledProcessError(1, ["xcrun"], stderr="specific compiler diagnostic")
        with tempfile.TemporaryDirectory() as out, mock.patch.object(self.ex.subprocess, "run", side_effect=failure):
            with self.assertRaisesRegex(RuntimeError, "specific compiler diagnostic") as error:
                self.ex.compile_model("tiny.mlpackage", out)
            self.assertIs(error.exception.__cause__, failure)

    def test_head_weight_guard_rejects_constexpr_and_unknown_producers(self):
        for op_type in ("constexpr_affine_dequantize", "constexpr_blockwise_shift_scale",
                        "constexpr_lut_to_dense", "add"):
            weight = types.SimpleNamespace(op=types.SimpleNamespace(op_type=op_type))
            with self.subTest(op_type=op_type), self.assertRaisesRegex(RuntimeError, "--heads-fp16"):
                self.ex.head_weight_constants(weight)
        constant = types.SimpleNamespace(op=types.SimpleNamespace(op_type="const", name="head_weight"))
        cast = types.SimpleNamespace(op=types.SimpleNamespace(op_type="cast", x=constant))
        self.assertEqual(self.ex.head_weight_constants(cast), {"head_weight"})

    def test_heads_fp16_fails_when_no_projection_matches(self):
        fn = mock.Mock()
        fn.find_ops.return_value = []
        ml = types.SimpleNamespace(_mil_program=types.SimpleNamespace(functions={"main": fn}))
        with self.assertRaisesRegex(RuntimeError, "no matching.*head projection"):
            self.ex.compress(ml, "fp16", True, N_Q * VOCAB)

    @unittest.skipUnless(HAVE_COREMLTOOLS, "coremltools not installed")
    def test_compression_excludes_heads_and_checks_result(self):
        import coremltools.optimize.coreml as cto

        ex = self.ex
        const = types.SimpleNamespace(op=types.SimpleNamespace(op_type="const", name="fused_head_weight"),
                                      shape=(N_Q * VOCAB, D, 1, 1))
        head = types.SimpleNamespace(name="fused_heads", weight=const)
        fn = mock.Mock()
        fn.find_ops.return_value = [head]
        ml = types.SimpleNamespace(_mil_program=types.SimpleNamespace(functions={"main": fn}))
        for mode, compressor in (("int8", "linear_quantize_weights"), ("int8-b32", "linear_quantize_weights"),
                                 ("int4", "linear_quantize_weights"),
                                 ("pal6", "palettize_weights")):
            with self.subTest(mode=mode), mock.patch.object(cto, compressor, return_value=ml) as compress:
                self.assertIs(ex.compress(ml, mode, True, N_Q * VOCAB), ml)
                config = compress.call_args.args[1]
                self.assertEqual(config.op_name_configs, {"fused_head_weight": None})
        bad_weight = types.SimpleNamespace(shape=const.shape,
                                          op=types.SimpleNamespace(op_type="constexpr_blockwise_shift_scale"))
        bad_fn = mock.Mock()
        bad_fn.find_ops.return_value = [types.SimpleNamespace(name=head.name, weight=bad_weight)]
        bad_ml = types.SimpleNamespace(_mil_program=types.SimpleNamespace(functions={"main": bad_fn}))
        with mock.patch.object(cto, "linear_quantize_weights", return_value=bad_ml):
            with self.assertRaisesRegex(RuntimeError, "constexpr quantized"):
                ex.compress(ml, "int8", True, N_Q * VOCAB)

    @unittest.skipUnless(HAVE_COREMLTOOLS, "coremltools not installed")
    def test_int8_heads_stay_constant_while_other_weights_are_quantized(self):
        ex = self.ex
        # Cross the compressor's default 2048-element threshold for non-head
        # weights as well, so successful exclusion cannot pass vacuously.
        t, hp = dict(self.t), hparams()
        hp["d_ff"] = FF * 2
        for layer in range(LAYERS):
            for name, axis in (("ffn_up", 0), ("ffn_down", 1)):
                key = f"dec.blk.{layer}.{name}.weight"
                t[key] = np.concatenate([t[key], t[key]], axis=axis)
        model = ex.DecoderStep(t, hp, CTX, CROSS, torch.float16, LAYERS).eval()
        ml = ex.convert_coreml(model, "int8", True, ex.contract(model, hp, "int8+fp16-heads", None))
        for head in ex.head_projections(ml, N_Q * VOCAB):
            self.assertTrue(ex.head_weight_constants(head.weight))
            self.assertEqual(head.weight.dtype.__name__, "fp16")
        ops = ml._mil_program.functions["main"].operations
        self.assertTrue(any(op.op_type.startswith("constexpr_") for op in ops))

    @unittest.skipUnless(HAVE_COREMLTOOLS, "coremltools not installed")
    def test_int8_block32_encodes_independent_input_block_scales(self):
        ex = self.ex
        # A 96-wide FFN down projection must carry three scales per output
        # row. One scale per row would silently reproduce the old tier.
        t, hp = dict(self.t), hparams()
        hp["d_ff"] = FF * 2
        for layer in range(LAYERS):
            for name, axis in (("ffn_up", 0), ("ffn_down", 1)):
                key = f"dec.blk.{layer}.{name}.weight"
                t[key] = np.concatenate([t[key], t[key]], axis=axis)
        model = ex.DecoderStep(t, hp, CTX, CROSS, torch.float16, LAYERS).eval()
        ml = ex.convert_coreml(model, "int8-b32", False, ex.contract(model, hp, "int8-b32", None))
        block_ops = ml._mil_program.functions["main"].find_ops(op_type="constexpr_blockwise_shift_scale")
        split_rows = [op for op in block_ops if len(op.data.shape) == 4 and op.data.shape[1] == FF * 2]
        self.assertEqual(len(split_rows), LAYERS)
        for op in split_rows:
            self.assertEqual(tuple(op.scale.shape), (D, FF * 2 // 32, 1, 1))
            self.assertEqual(op.data.val.dtype, np.dtype(np.int8))
        meta = json.loads(ml.user_defined_metadata["parler.decoder.contract"])
        self.assertEqual(meta["weights"], "int8-b32")

    @unittest.skipUnless(HAVE_COREMLTOOLS, "coremltools not installed")
    def test_converted_program_reads_every_state_and_writes_none(self):
        ex = self.ex
        for per_state, k_layout in ((1, "rows"), (LAYERS, "rows"), (LAYERS, "cols")):
            model = ex.DecoderStep(self.t, hparams(), CTX, CROSS, torch.float16, per_state, k_layout).eval()
            meta = ex.contract(model, hparams(), "fp16", None)
            ml = ex.convert_coreml(model, "fp16", False, meta)   # raises if a state is dropped or written
            fn = ml._mil_program.functions["main"]
            self.assertEqual(len(fn.find_ops(op_type="read_state")), 4 * LAYERS // per_state)
            norms = fn.find_ops(op_type="layer_norm")
            self.assertEqual(len(norms), 3 * LAYERS + 1)
            for norm in norms:
                self.assertEqual(list(norm.axes.val), [-1])
                self.assertEqual(tuple(norm.x.shape), (1, 1, 1, D))
            got = json.loads(ml.user_defined_metadata["parler.decoder.contract"])
            self.assertEqual(got["contract"], 2)
            self.assertEqual(got["normalization"], "last-axis")
            self.assertEqual(got["states"]["layers_per_state"], per_state)
            k_inner = [CTX, D // HEADS] if k_layout == "rows" else [D // HEADS, CTX]
            self.assertEqual(got["states"]["k_layout"], k_layout)
            self.assertEqual(got["states"]["shapes"]["self_k_00"], [per_state, HEADS] + k_inner)
            self.assertEqual(got["states"]["shapes"]["self_v_00"], [per_state, HEADS, CTX, D // HEADS])
            self.assertEqual(got["outputs"]["new_k"], [LAYERS, HEADS, 1, D // HEADS])


@unittest.skipUnless(HAVE_TORCH and HAVE_COREMLTOOLS, "torch/coremltools not installed")
class FfnDownCompressionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ex = load_exporter()
        # Every projection exceeds the compressor's 2048-element cutoff;
        # below-threshold attention weights must not make this test vacuous.
        tensors = tiny_tensors(np.random.default_rng(29))
        for name, value in tensors.items():
            repeats = (1, 2) if "lm_heads" in name else ((2, 2) if value.ndim == 2 else (2,))
            tensors[name] = np.tile(value, repeats)
        cls.hp = dict(hparams(), d_model=D * 2, d_ff=FF * 2)
        cls.model = cls.ex.DecoderStep(tensors, cls.hp, CTX, CROSS, torch.float16, LAYERS).eval()
        cls.ml = cls.ex.convert_coreml(cls.model, "fp16", False,
                                     cls.ex.contract(cls.model, cls.hp, "fp16", None))

    def convs(self, ml=None):
        return (ml or self.ml)._mil_program.functions["main"].find_ops(op_type="conv")

    def fake_model(self, convs):
        fn = mock.Mock()
        fn.find_ops.return_value = convs
        return types.SimpleNamespace(_mil_program=types.SimpleNamespace(functions={"main": fn}))

    def test_all_compression_modes_retain_only_requested_projections(self):
        ex = self.ex
        selected = ex.ffn_down_projections(self.ml, self.model)
        self.assertEqual(list(selected), [f"dec.blk.{layer}.ffn_down.weight" for layer in range(LAYERS)])
        self.assertEqual(len(self.convs()), LAYERS * 8 + 1)
        for mode in ex.WEIGHT_MODES:
            if mode == "fp16":
                continue
            for heads_fp16 in (False, True):
                with self.subTest(mode=mode, heads_fp16=heads_fp16):
                    result = ex.compress(self.ml, mode, heads_fp16, N_Q * VOCAB, selected)
                    retained = {op.name for op in selected.values()}
                    # The tiny head has 132 channels: palettes already leave
                    # it uncompressed because 132 is not divisible by 16.
                    head_retained = heads_fp16 or mode.startswith("pal")
                    if head_retained:
                        retained.update(op.name for op in ex.head_projections(self.ml, N_Q * VOCAB))
                    compressed_count = 0
                    for op in self.convs(result):
                        self.assertEqual(op.weight.dtype.__name__, "fp16")
                        if op.name in retained:
                            ex.fp16_projection_constants(op, "test")
                            original = next(c for c in self.convs() if c.name == op.name)
                            np.testing.assert_array_equal(op.weight.val, original.weight.val)
                        else:
                            self.assertTrue(op.weight.op.op_type.startswith("constexpr_"), op.name)
                            compressed_count += 1
                    self.assertEqual(compressed_count, LAYERS * 7 + (not head_retained))

    def test_default_int8_still_compresses_down_weights(self):
        result = self.ex.compress(self.ml, "int8", True, N_Q * VOCAB)
        for name in (op.name for op in self.ex.ffn_down_projections(self.ml, self.model).values()):
            op = next(c for c in self.convs(result) if c.name == name)
            self.assertTrue(op.weight.op.op_type.startswith("constexpr_"))
        meta = json.loads(self.ml.user_defined_metadata["parler.decoder.contract"])
        self.assertNotIn("experimental_precision", meta)

    def test_selection_and_exclusions_survive_renamed_weight_producers(self):
        # Unrelated producer names exercise source matching and exclusion by
        # actual producers, rather than guessing layers_N_down_weight names.
        changed = self.ex.convert_coreml(self.model, "fp16", False,
                                        self.ex.contract(self.model, self.hp, "fp16", None))
        selected = self.ex.ffn_down_projections(changed, self.model)
        for index, op in enumerate(selected.values()):
            producer = op.weight.op
            producer.name = f"transformed_constant_{index}"
            op.weight.name = producer.name
        # Compression reloads the serialized spec. Re-serialize the renamed
        # program so its spec and live MIL represent the same transformation.
        import coremltools as ct
        changed = ct.convert(changed._mil_program, source="milinternal", convert_to="mlprogram",
                             minimum_deployment_target=ct.target.macOS15, skip_model_load=True)
        selected = self.ex.ffn_down_projections(changed, self.model)
        result = self.ex.compress(changed, "int8", True, N_Q * VOCAB, selected)
        for op in self.convs(result):
            if op.name in {down.name for down in selected.values()}:
                self.assertTrue(self.ex.fp16_projection_constants(op, "test"))

    def test_selection_rejects_missing_ambiguous_wrong_shape_and_wrong_layer(self):
        ex = self.ex
        convs = self.convs()
        down = list(ex.ffn_down_projections(self.ml, self.model).values())
        other = [op for op in convs if op not in down]
        duplicate = types.SimpleNamespace(name="duplicate_down", weight=down[0].weight, scopes=down[0].scopes)
        wrong_shape = types.SimpleNamespace(name=down[0].name, scopes=down[0].scopes,
                                            weight=types.SimpleNamespace(shape=(D, FF, 1, 1)))
        no_semantics = types.SimpleNamespace(name=down[0].name, scopes={}, weight=down[0].weight)
        for projections, message in ((other + down[:1], "exactly 2"),
                                     (convs + [duplicate], "unambiguous"),
                                     (other + [down[0], down[0]], "unambiguous"),
                                     (other + [no_semantics, down[1]], "exactly 2"),
                                     (other + [wrong_shape, down[1]], "shape"),
                                     (other + down[::-1], "source layer")):
            with self.subTest(message=message), self.assertRaisesRegex(RuntimeError, message):
                ex.ffn_down_projections(self.fake_model(projections), self.model)

    def test_fp16_guard_checks_cast_source_constant_dtype(self):
        from coremltools.converters.mil.mil import types as mil_types

        source = types.SimpleNamespace(dtype=mil_types.fp16,
                                       op=types.SimpleNamespace(op_type="const", name="renamed_constant"))
        cast = types.SimpleNamespace(dtype=mil_types.fp16, op=types.SimpleNamespace(op_type="cast", x=source))
        projection = types.SimpleNamespace(name="down_conv", weight=cast)
        self.assertEqual(self.ex.fp16_projection_constants(projection, "--ffn-down-fp16"),
                         {"renamed_constant"})
        source.dtype = mil_types.fp32
        with self.assertRaisesRegex(RuntimeError, "--ffn-down-fp16.*not fp16"):
            self.ex.fp16_projection_constants(projection, "--ffn-down-fp16")

    def test_post_compression_guards_reject_changed_precision_values_and_selection(self):
        import coremltools.optimize.coreml as cto

        ex = self.ex
        selected = ex.ffn_down_projections(self.ml, self.model)
        quantized = ex.compress(self.ml, "int8", False, N_Q * VOCAB)
        # Compressing the retained down weights or skipping all compression
        # must both fail, even if the compressor returns a valid MLModel.
        for result, message in ((quantized, "constexpr quantized"), (self.ml, "was not compressed")):
            with mock.patch.object(cto, "linear_quantize_weights", return_value=result), \
                    self.assertRaisesRegex(RuntimeError, message):
                ex.compress(self.ml, "int8", False, N_Q * VOCAB, selected)
        duplicate = self.fake_model(self.convs(quantized) + [self.convs(quantized)[0]])
        with mock.patch.object(cto, "linear_quantize_weights", return_value=duplicate), \
                self.assertRaisesRegex(RuntimeError, "operations changed"):
            ex.compress(self.ml, "int8", False, N_Q * VOCAB, selected)
        first = next(iter(selected.values()))
        for change, message in (({"name": "renamed_projection"}, "operations changed"),
                                ({"shape": (1, 1, 1, 1)}, "shape changed"),
                                ({"dtype": types.SimpleNamespace(__name__="fp32")}, "not fp16"),
                                ({"val": first.weight.val + np.float16(1)}, "weight changed")):
            weight = types.SimpleNamespace(shape=first.weight.shape, dtype=first.weight.dtype,
                                           val=first.weight.val, op=first.weight.op)
            for key, value in change.items():
                if key != "name":
                    setattr(weight, key, value)
            altered = types.SimpleNamespace(name=change.get("name", first.name), weight=weight)
            convs = [altered if op.name == first.name else op for op in self.convs(quantized)]
            # Both retained down projections start as their original constants.
            replacements = {op.name: op for op in selected.values() if op.name != first.name}
            convs = [replacements.get(op.name, op) for op in convs]
            with self.subTest(change=list(change)), \
                    mock.patch.object(cto, "linear_quantize_weights", return_value=self.fake_model(convs)), \
                    self.assertRaisesRegex(RuntimeError, message):
                ex.compress(self.ml, "int8", False, N_Q * VOCAB, selected)

    def test_converted_metadata_records_exact_composed_selection(self):
        ex = self.ex
        meta = ex.contract(self.model, self.hp, "int8+fp16-heads", "tiny-f32.gguf")
        result = ex.convert_coreml(self.model, "int8", True, meta, ffn_down_fp16=True)
        got = json.loads(result.user_defined_metadata["parler.decoder.contract"])
        self.assertEqual(got["weights"], "int8+fp16-heads+fp16-ffn-down")
        precision = got.pop("experimental_precision")
        self.assertEqual(precision["ffn_down_fp16"], True)
        self.assertEqual(precision["heads_fp16"], True)
        down = precision["ffn_down_projections"]
        self.assertEqual([p["tensor"] for p in down],
                         [f"dec.blk.{layer}.ffn_down.weight" for layer in range(LAYERS)])
        for layer, entry in enumerate(down):
            op = next(c for c in self.convs(result) if c.name == entry["operation"])
            self.assertEqual(entry["dtype"], "fp16")
            self.assertEqual(entry["shape"], list(op.weight.shape))
            np.testing.assert_array_equal(op.weight.val,
                self.model.layers[layer].down.weight.detach().numpy().astype(np.float16))
        got["weights"] = meta["weights"]
        self.assertEqual(got, meta)  # normalization, contract, states and I/O unchanged
        self.assertNotIn("experimental_precision", meta)  # caller metadata not mutated

    def test_converted_metadata_without_head_exclusion(self):
        ex = self.ex
        result = ex.convert_coreml(self.model, "int8", False,
            ex.contract(self.model, self.hp, "int8", None), ffn_down_fp16=True)
        got = json.loads(result.user_defined_metadata["parler.decoder.contract"])
        self.assertEqual(got["weights"], "int8+fp16-ffn-down")
        self.assertFalse(got["experimental_precision"]["heads_fp16"])
        self.assertEqual(len(got["experimental_precision"]["ffn_down_projections"]), LAYERS)
        for head in ex.head_projections(result, N_Q * VOCAB):
            self.assertTrue(head.weight.op.op_type.startswith("constexpr_"))


if __name__ == "__main__":
    unittest.main()
