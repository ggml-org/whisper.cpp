#!/usr/bin/env python3
"""ctypes ownership/type tests and opt-in tiny native bridge integration.

PARLER_DECODER_VALIDATION_BRIDGE=/build/parler-decoder-validation.so
PARLER_DECODER_VALIDATION_TINY_GGUF=/tmp/tiny/parler-tiny-f32.gguf
The opt-in tests only use that synthetic fixture; never point at full models.
"""
import ctypes
import importlib.util
import os
from pathlib import Path
import shutil
import tempfile
import types
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parent.parent / "scripts/parler/decoder-native-adapter.py"
spec = importlib.util.spec_from_file_location("parler_native_validation_adapter", SCRIPT)
N = importlib.util.module_from_spec(spec)
spec.loader.exec_module(N)


class NativeTypesTest(unittest.TestCase):
    def test_integer_conversion_preserves_contiguous_owner(self):
        source = np.arange(12, dtype=np.int64)[::2]
        value = N.ids(source, "tokens")
        self.assertEqual(value.dtype, np.int32)
        self.assertTrue(value.flags.c_contiguous)
        source[:] = 99
        np.testing.assert_array_equal(value, np.arange(12)[::2])

    def test_bad_token_types_shapes_ranges_rejected_before_ctypes(self):
        for value in ([], [1.2], [[1], [2]], [2**40], [-1], np.array([2**63], dtype=np.uint64)):
            with self.subTest(value=value), self.assertRaises(ValueError):
                N.ids(value, "tokens")

    def test_abi_version_and_every_signature_explicit(self):
        class Function:
            def __init__(self, result=2): self.result = result
            def __call__(self): return self.result
        class Library: pass
        library = Library()
        names = ("abi_version", "create", "destroy", "dimensions", "prepare", "prefill", "step", "synchronize", "backend")
        for name in names:
            setattr(library, "parler_validation_" + name, Function())
        N.configure_abi(library)
        for name in names:
            fn = getattr(library, "parler_validation_" + name)
            self.assertIsInstance(fn.argtypes, list)
            self.assertTrue(hasattr(fn, "restype"))
        self.assertIs(library.parler_validation_create.restype, ctypes.c_void_p)
        self.assertEqual(library.parler_validation_prepare.argtypes[5], ctypes.c_int)
        library.parler_validation_abi_version.result = 1
        with self.assertRaisesRegex(ValueError, "ABI"):
            N.configure_abi(library)

    def test_prepare_passes_exact_trace_bound_including_bos(self):
        calls = []
        runner = N.NativeRunner.__new__(N.NativeRunner)
        runner.handle = 1
        runner.nq = 3
        runner.error = ctypes.create_string_buffer(256)
        def prepare(*args):
            calls.append(args)
            return 1
        runner.lib = types.SimpleNamespace(parler_validation_prepare=prepare,
                                            parler_validation_destroy=lambda handle: None)
        try:
            frames = np.ones((200, 3), dtype=np.int64)
            frames[0] = 41
            runner.prepare_case({"id": "bounded"}, {"description_ids": [[2, 1]],
                                "prompt_ids": [[2, 3]], "decoder_inputs": frames})
            self.assertEqual(calls[0][5], 200)
            self.assertEqual(runner.max_frames, 200)
            self.assertIsNone(runner.next_index)
        finally:
            runner.close()


@unittest.skipUnless(os.environ.get("PARLER_DECODER_VALIDATION_BRIDGE") and os.environ.get("PARLER_DECODER_VALIDATION_TINY_GGUF"),
                     "set bridge + synthetic tiny GGUF env vars for native integration")
class NativeTinyTest(unittest.TestCase):
    def config(self):
        gguf = Path(os.environ["PARLER_DECODER_VALIDATION_TINY_GGUF"])
        self.assertLess(gguf.stat().st_size, 2_000_000, "native unit tests may only use a tiny synthetic GGUF")
        return dict(bridge_library=os.environ["PARLER_DECODER_VALIDATION_BRIDGE"], gguf=str(gguf),
                    threads=1, n_gpu_layers=0, backend="cpu")

    def test_native_ggml_repeated_cases_reset_and_invalid_frame_does_not_advance(self):
        runner = N.NativeRunner({}, self.config(), coreml=False)
        try:
            frames = np.array([[41, 41, 41], [1, 41, 41], [2, 3, 41], [3, 4, 5]], dtype=np.int64)
            arrays = {"description_ids": np.array([[2, 1]]), "prompt_ids": np.array([[2, 3]]), "decoder_inputs": frames}
            outputs = []
            for case_id in ("a", "b", "a"):
                case = {"id": case_id}
                runner.prepare_case(case, arrays)
                runner.start_case(case, arrays)
                result = []
                for i, frame in enumerate(frames):
                    if i == 1:
                        with self.assertRaises(RuntimeError):
                            runner.step(i, np.array([10000, 1, 1]))
                    result.append(runner.step(i, frame).copy())
                    runner.synchronize()
                runner.finish_case()
                outputs.append(np.stack(result))
            np.testing.assert_array_equal(outputs[0], outputs[1])
            np.testing.assert_array_equal(outputs[0], outputs[2])
            self.assertTrue(np.isfinite(outputs[0]).all())
            self.assertEqual(runner.metadata()["case_backends"]["a"], "CPU/ggml")
        finally:
            runner.close()
            runner.close()
        self.assertIsNone(runner.handle)

    def test_failed_prepare_invalidates_previous_case(self):
        runner = N.NativeRunner({}, self.config(), coreml=False)
        try:
            arrays = {"description_ids": [[2, 1]], "prompt_ids": [[2]], "decoder_inputs": np.ones((3, 3), dtype=np.int32)}
            runner.prepare_case({"id": "a"}, arrays)
            runner.start_case({"id": "a"}, arrays)
            runner.step(0, np.array([41] * 3))
            arrays["description_ids"] = [[99999]]
            with self.assertRaises(RuntimeError):
                runner.prepare_case({"id": "b"}, arrays)
            with self.assertRaises(RuntimeError):
                runner.step(1, np.array([1] * 3))
        finally:
            runner.close()

    def test_absent_coreml_sidecar_fails_instead_of_measuring_ggml(self):
        config = self.config()
        with tempfile.TemporaryDirectory() as directory:
            gguf = Path(directory) / "no-sidecar-f32.gguf"
            shutil.copyfile(config["gguf"], gguf)
            config["gguf"] = str(gguf)
            with self.assertRaisesRegex(RuntimeError, "sidecar"):
                N.NativeRunner({}, config, coreml=True)

    def test_staged_symlink_uses_sidecars_beside_supplied_path(self):
        config = self.config()
        with tempfile.TemporaryDirectory() as directory:
            gguf = Path(directory) / "stage-only-f32.gguf"
            gguf.symlink_to(Path(config["gguf"]).resolve())
            config["gguf"] = str(gguf)
            # Resolving the GGUF before native open would discover the source
            # directory's model, changing the candidate being measured.
            try:
                runner = N.NativeRunner({}, config, coreml=True)
            except RuntimeError as error:
                reason = str(error)
                if "this build has no Core ML decoder support" in reason or "Core ML is disabled" in reason:
                    self.skipTest(reason)
                self.assertIn("stage-only-decoder", reason)
            else:
                runner.close()
                self.fail("staged model unexpectedly found a sidecar in another directory")


if __name__ == "__main__":
    unittest.main()
