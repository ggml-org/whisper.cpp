#!/usr/bin/env python3
"""Build a tiny read-only-state model for test_decoder_coreml.cpp (macOS 15+).

Run with the Core ML Python environment; destination must be outside the repo.
No Parler weights, network, or large model conversion is needed.
"""
import argparse
from pathlib import Path
import shutil

import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)

    def outputs(hidden, position, cross, k, v, layers=2):
        k = mb.cast(x=k, dtype="fp32")
        v = mb.cast(x=v, dtype="fp32")
        total_k = mb.reduce_sum(x=k, axes=[0, 1, 2, 3], keep_dims=False)
        total_v = mb.reduce_sum(x=v, axes=[0, 1, 2, 3], keep_dims=False)
        hidden_sum = mb.reduce_sum(x=mb.cast(x=hidden, dtype="fp32"), axes=[0, 1, 2, 3], keep_dims=False)
        position_f = mb.reduce_sum(x=mb.cast(x=position, dtype="fp32"), axes=[0], keep_dims=False)
        cross_sum = mb.reduce_sum(x=cross, axes=[0, 1], keep_dims=False)
        history = mb.add(x=total_k, y=mb.mul(x=total_v, y=np.float32(3)))
        current = mb.add(x=hidden_sum, y=mb.add(x=position_f, y=cross_sum))
        logits = mb.add(x=mb.add(x=history, y=current), y=np.arange(6, dtype=np.float32).reshape(2, 3), name="logits")
        base = np.arange(layers * 6, dtype=np.float32).reshape(layers, 2, 1, 3)
        new_k = mb.cast(x=mb.add(x=base, y=hidden_sum), dtype="fp16", name="new_k")
        new_v = mb.cast(x=mb.sub(x=base, y=mb.mul(x=hidden_sum, y=np.float32(2))), dtype="fp16", name="new_v")
        return logits, new_k, new_v

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(1, 4, 1, 1), dtype=types.fp16),
            mb.TensorSpec(shape=(1,), dtype=types.int32),
            mb.TensorSpec(shape=(1, 2), dtype=types.fp32),
            mb.StateTensorSpec(shape=(2, 2, 8, 3), dtype=types.fp16),
            mb.StateTensorSpec(shape=(2, 2, 8, 3), dtype=types.fp16),
        ],
        opset_version=ct.target.macOS15,
    )
    def program(hidden, position, cross, cache_k, cache_v):
        # Deliberately no coreml_update_state: this is the critical contract.
        # Every state element affects the output, including nonzero CTX rows.
        return outputs(hidden, position, cross, mb.read_state(input=cache_k), mb.read_state(input=cache_v))

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(1, 4, 1, 1), dtype=types.fp16),
            mb.TensorSpec(shape=(1,), dtype=types.int32),
            mb.TensorSpec(shape=(1, 2), dtype=types.fp32),
            *[mb.StateTensorSpec(shape=(1, 2, 8, 3), dtype=types.fp16) for _ in range(4)],
        ],
        opset_version=ct.target.macOS15,
    )
    def layered(hidden, position, cross, self_k_00, self_k_01, self_v_00, self_v_01):
        k = mb.concat(values=[mb.read_state(input=self_k_00), mb.read_state(input=self_k_01)], axis=0)
        v = mb.concat(values=[mb.read_state(input=self_v_00), mb.read_state(input=self_v_01)], axis=0)
        return outputs(hidden, position, cross, k, v)

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(1, 4, 1, 1), dtype=types.fp16),
            mb.TensorSpec(shape=(1,), dtype=types.int32),
            mb.TensorSpec(shape=(1, 2), dtype=types.fp32),
            *[mb.StateTensorSpec(shape=(2, 2, 8, 3), dtype=types.fp16) for _ in range(4)],
        ],
        opset_version=ct.target.macOS15,
    )
    def grouped(hidden, position, cross, group_k_00, group_k_01, group_v_00, group_v_01):
        k = mb.concat(values=[mb.read_state(input=group_k_00), mb.read_state(input=group_k_01)], axis=0)
        v = mb.concat(values=[mb.read_state(input=group_v_00), mb.read_state(input=group_v_01)], axis=0)
        return outputs(hidden, position, cross, k, v, layers=4)

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(1, 4, 1, 1), dtype=types.fp16),
            mb.TensorSpec(shape=(1,), dtype=types.int32),
            mb.TensorSpec(shape=(1, 2), dtype=types.fp32),
            mb.StateTensorSpec(shape=(24, 2, 8, 3), dtype=types.fp16),
            mb.StateTensorSpec(shape=(24, 2, 8, 3), dtype=types.fp16),
        ],
        opset_version=ct.target.macOS15,
    )
    def fullgroup(hidden, position, cross, self_k_00, self_v_00):
        return outputs(hidden, position, cross,
                       mb.read_state(input=self_k_00), mb.read_state(input=self_v_00), layers=24)

    @mb.program(
        input_specs=[mb.TensorSpec(shape=(4,), dtype=types.fp16),
                     mb.TensorSpec(shape=(1,), dtype=types.int32),
                     mb.StateTensorSpec(shape=(1,), dtype=types.fp16)],
        opset_version=ct.target.macOS15,
    )
    def failing(hidden, position, cache):
        # The MIL gather contract requires an error for invalid indices when
        # validate_indices=True; this tests NSError handling without a mock.
        value = mb.gather(x=mb.cast(x=hidden, dtype="fp32"), indices=position,
                          axis=0, validate_indices=True)
        return mb.add(x=value, y=mb.cast(x=mb.read_state(input=cache), dtype="fp32"), name="logits")

    for name, prog, states in [
        ("readonly_cache", program, {"cache_k", "cache_v"}),
        ("readonly_layers", layered, {"self_k_00", "self_k_01", "self_v_00", "self_v_01"}),
        ("readonly_grouped_g2", grouped, {"group_k_00", "group_k_01", "group_v_00", "group_v_01"}),
        ("readonly_fullgroup_l24", fullgroup, {"self_k_00", "self_v_00"}),
        ("inference_error", failing, {"cache"}),
    ]:
        model = ct.convert(prog, minimum_deployment_target=ct.target.macOS15,
                           compute_precision=ct.precision.FLOAT32, compute_units=ct.ComputeUnit.CPU_ONLY)
        spec = model.get_spec()
        assert {s.name for s in spec.description.state} == states
        ops = [op.type for f in spec.mlProgram.functions.values() for block in f.block_specializations.values() for op in block.operations]
        assert "read_state" in ops and not {"write_state", "coreml_update_state"}.intersection(ops), ops
        package = args.destination / (name + ".mlpackage")
        model.user_defined_metadata["parler.test.metadata"] = (
            "readonly-fullgroup-l24-g24-v1" if name == "readonly_fullgroup_l24" else "readonly-state-fixture")
        model.save(str(package))
        destination = args.destination / (name + ".mlmodelc")
        # compile_model uses shutil.move: an existing directory would nest the
        # new model inside it instead of replacing it. Only remove this known
        # generated fixture; never the caller's destination directory.
        if destination.exists():
            shutil.rmtree(destination)
        compiled = ct.models.utils.compile_model(str(package), destination_path=str(destination))
        print(compiled)


if __name__ == "__main__":
    main()
