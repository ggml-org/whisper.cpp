#!/usr/bin/env python3
"""Shared ctypes runner for decoder-ggml-adapter/decoder-coreml-adapter.

Pass --adapter-config FILE to validate-decoder-coreml.py run. Config example:
{"bridge_library":"/build/parler-decoder-validation.so",
 "gguf":"/models/parler-mini-v1-q8_0.gguf", "threads":4,
 "n_gpu_layers":99, "backend":"metal"}

The Core ML wrapper always requires a sidecar and uses STRICT runtime mode.
Select Core ML units through the parent's PARLER_COREML_COMPUTE_UNITS setting;
this module never mutates process environment variables. The ggml wrapper
never attaches a decoder sidecar, even if one is beside the model.

prepare_case runs native description/T5/cross-KV outside decoder timing.
Consequently quality includes native T5 differences from the HF cross_states
stored in fixtures; this is explicitly recorded, not an exact-HF-cross ablation.
The decoder timing includes ggml prefill, native KV download/layout/seed for
Core ML, host token embedding/device reads, input dtype copies, prediction,
partial state updates, native output copy, synchronization and Python copy.
"""
import ctypes as C
import hashlib
import os
from pathlib import Path

import numpy as np

I32P = C.POINTER(C.c_int32)
F32P = C.POINTER(C.c_float)
CHARP = C.POINTER(C.c_char)


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ids(value, label):
    a = np.asarray(value)
    if a.ndim == 2 and a.shape[0] == 1:
        a = a[0]
    if a.ndim != 1 or not a.size or not np.issubdtype(a.dtype, np.integer):
        raise ValueError(f"{label} must be a nonempty integer vector or [1,length]")
    limits = np.iinfo(np.int32)
    if np.min(a) < 0 or np.max(a) > limits.max:
        raise ValueError(f"{label} exceeds nonnegative int32 token range")
    return np.ascontiguousarray(a, dtype=np.int32)


def configure_abi(lib):
    signatures = {
        "abi_version": ([], C.c_int),
        "create": ([C.c_char_p, C.c_int, C.c_int, C.c_int, C.c_char_p, CHARP, C.c_size_t], C.c_void_p),
        "destroy": ([C.c_void_p], None),
        "dimensions": ([C.c_void_p, C.POINTER(C.c_int), C.POINTER(C.c_int), C.POINTER(C.c_int), CHARP, C.c_size_t], C.c_int),
        "prepare": ([C.c_void_p, I32P, C.c_size_t, I32P, C.c_size_t, C.c_int, CHARP, C.c_size_t], C.c_int),
        "prefill": ([C.c_void_p, I32P, C.c_size_t, F32P, C.c_size_t, CHARP, C.c_size_t], C.c_int),
        "step": ([C.c_void_p, I32P, C.c_size_t, F32P, C.c_size_t, CHARP, C.c_size_t], C.c_int),
        "synchronize": ([C.c_void_p, CHARP, C.c_size_t], C.c_int),
        "backend": ([C.c_void_p, CHARP, C.c_size_t, CHARP, C.c_size_t], C.c_int),
    }
    for name, (arguments, result) in signatures.items():
        fn = getattr(lib, "parler_validation_" + name)
        fn.argtypes, fn.restype = arguments, result
    if lib.parler_validation_abi_version() != 2:
        raise ValueError("unsupported decoder validation bridge ABI (need version 2)")


class NativeRunner:
    def __init__(self, manifest, config, coreml):
        if not isinstance(config, dict):
            raise ValueError("native adapters require --adapter-config JSON")
        unknown = set(config) - {"bridge_library", "gguf", "n_gpu_layers", "threads", "backend"}
        if unknown:
            raise ValueError(f"unknown adapter config keys: {sorted(unknown)}")
        self.handle = None
        self.lib = None
        self.coreml = bool(coreml)
        self.library = Path(config["bridge_library"]).expanduser().resolve(strict=True)
        # Sidecars are discovered beside the supplied GGUF path. Preserve a
        # staged symlink's parent directory, just like the native CLI does.
        self.gguf = Path(os.path.abspath(Path(config["gguf"]).expanduser()))
        if not self.gguf.is_file():
            raise FileNotFoundError(self.gguf)
        self.threads = config.get("threads", 4)
        self.n_gpu_layers = config.get("n_gpu_layers", 0)
        self.backend = config.get("backend", "cpu" if self.n_gpu_layers == 0 else "metal")
        if type(self.threads) is not int or not 1 <= self.threads <= 256:
            raise ValueError("threads must be an integer in 1..256")
        if type(self.n_gpu_layers) is not int or not 0 <= self.n_gpu_layers <= 2**31 - 2:
            raise ValueError("n_gpu_layers must be a nonnegative int32")
        if not isinstance(self.backend, str):
            raise ValueError("backend must be a string")
        self.error = C.create_string_buffer(4096)
        self.lib = C.CDLL(str(self.library))  # kept alive until handle destruction
        configure_abi(self.lib)
        self.handle = self.lib.parler_validation_create(os.fsencode(self.gguf), self.n_gpu_layers,
            int(self.coreml), self.threads, self.backend.encode(), self.error, len(self.error))
        if not self.handle:
            raise RuntimeError(self.error.value.decode(errors="replace") or "native decoder creation failed")
        try:
            nq, vocab, gpu = C.c_int(), C.c_int(), C.c_int()
            self.check(self.lib.parler_validation_dimensions(self.handle, C.byref(nq), C.byref(vocab), C.byref(gpu),
                       self.error, len(self.error)))
            self.nq, self.vocab, self.on_gpu = nq.value, vocab.value, bool(gpu.value)
            if self.nq < 1 or self.vocab < 1:
                raise ValueError("native model returned invalid decoder dimensions")
            if self.n_gpu_layers > 0 and not self.on_gpu:
                raise ValueError("requested GPU baseline actually loaded CPU; measurement rejected")
            self.output = np.empty((self.nq, self.vocab), dtype=np.float32)
            self.output_ptr = self.output.ctypes.data_as(F32P)
            self.next_index = None
            self.prepared_id = None
            self.labels = {}
            self._identity = {"gguf_sha256": file_sha256(self.gguf), "bridge_sha256": file_sha256(self.library),
                              "common_adapter_sha256": file_sha256(__file__)}
        except BaseException:
            self.close()
            raise

    def check(self, status):
        if status != 1:
            raise RuntimeError(self.error.value.decode(errors="replace") or "native decoder call failed")

    def prepare_case(self, case, arrays):
        # Preserve numpy owners for all ctypes pointers for the whole case,
        # even though the C++ prepare call copies rather than retaining them.
        if "description_ids" not in arrays:
            raise ValueError("native adapter needs description_ids; regenerate legacy-manifest with current CLI")
        self.description = ids(arrays["description_ids"], "description_ids")
        self.prompt = ids(arrays["prompt_ids"], "prompt_ids")
        frames = np.asarray(arrays["decoder_inputs"])
        if frames.ndim != 2 or frames.shape[1] != self.nq or not 1 <= frames.shape[0] <= 2**31 - 2:
            raise ValueError("fixture needs a nonempty int32-bounded trace with the GGUF codebook count")
        self.max_frames = int(frames.shape[0])  # includes the BOS input at index zero
        self.check(self.lib.parler_validation_prepare(self.handle,
            self.description.ctypes.data_as(I32P), self.description.size,
            self.prompt.ctypes.data_as(I32P), self.prompt.size, self.max_frames, self.error, len(self.error)))
        self.prepared_id = case["id"]
        self.next_index = None

    def start_case(self, case, arrays):
        if self.prepared_id != case["id"]:
            raise RuntimeError("prepare_case must precede every timed start_case")
        self.next_index = 0

    def step(self, index, frame):
        if index != self.next_index:
            raise ValueError("decoder step index out of order")
        if index >= self.max_frames:
            raise ValueError("decoder step exceeds the prepared trace bound")
        # Conversion is intentionally inside the harness's step timer.
        self.frame = ids(frame, "decoder frame")
        if self.frame.size != self.nq:
            raise ValueError("wrong codebook count")
        fn = self.lib.parler_validation_prefill if index == 0 else self.lib.parler_validation_step
        self.check(fn(self.handle, self.frame.ctypes.data_as(I32P), self.frame.size,
                      self.output_ptr, self.output.size, self.error, len(self.error)))
        self.next_index += 1
        return self.output  # harness copies before the next call overwrites it

    def synchronize(self):
        self.check(self.lib.parler_validation_synchronize(self.handle, self.error, len(self.error)))

    def finish_case(self):
        label = C.create_string_buffer(256)
        self.check(self.lib.parler_validation_backend(self.handle, label, len(label), self.error, len(self.error)))
        backend = label.value.decode()
        if self.coreml and "/coreml-" not in backend:
            raise RuntimeError(f"Core ML candidate did not execute sidecar steps: {backend}")
        self.labels[self.prepared_id] = backend
        self.prepared_id = None
        self.next_index = None

    def metadata(self):
        return {**self._identity, "gguf": str(self.gguf), "bridge_library": str(self.library),
                "coreml_strict": self.coreml, "native_backend_request": self.backend, "on_gpu": self.on_gpu,
                "threads": self.threads, "n_gpu_layers": self.n_gpu_layers,
                "bridge_abi": 2,
                "codebooks": self.nq, "vocab": self.vocab, "case_backends": self.labels,
                "cross_source": "native_T5_from_fixture_description_ids (may differ from HF cross_states)",
                "prepare_scope": "native_T5_and_cross_KV_projection_excluded_from_decoder_timing",
                "coreml_units_env": os.environ.get("PARLER_COREML_COMPUTE_UNITS", "all")}

    def close(self):
        if self.handle:
            self.lib.parler_validation_destroy(self.handle)
            self.handle = None

    def __del__(self):
        self.close()
