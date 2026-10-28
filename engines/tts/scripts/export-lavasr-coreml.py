import argparse
import subprocess
from pathlib import Path

import numpy as np
import torch

from lavasr_coreml_models import Denoiser, Enhancer, read_gguf


STAGES = {'enhancer': Enhancer, 'denoiser': Denoiser}


def use_float16(operation):
    return not any(scope.endswith('LogMagnitude')
                   for scopes in operation.scopes.values() for scope in scopes)


def compute_precision(precision):
    import coremltools as ct

    if precision == 'float16':
        return ct.transform.FP16ComputePrecision(op_selector=use_float16)
    if precision == 'float32':
        return ct.precision.FLOAT32
    raise ValueError(f'Unsupported precision: {precision}')


def export_inputs(stage, metadata, frames):
    if stage == 'enhancer':
        return [('mel', (1, int(metadata['n_mels']), frames)), ('valid', (1, 1, frames))]
    shape = (1, 1, int(metadata['chunk_frames']), int(metadata['spec_bins']))
    return [('real', shape), ('imag', shape)]


def export_model(stage, gguf, output, frames, precision):
    import coremltools as ct

    tensors, metadata = read_gguf(gguf, 'lavasr-' + stage)
    model = STAGES[stage](tensors, metadata).eval()
    if stage == 'enhancer' and (frames is None or frames <= 2 * model.context):
        raise ValueError(f'--frames must exceed the enhancer context ({2 * model.context} frames)')
    inputs = export_inputs(stage, metadata, frames)
    samples = tuple(torch.ones(shape, dtype=torch.float32) for _, shape in inputs)
    with torch.inference_mode():
        traced = torch.jit.trace(model, samples)
    converted = ct.convert(traced, convert_to='mlprogram',
        inputs=[ct.TensorType(name=name, shape=shape, dtype=np.float32) for name, shape in inputs],
        outputs=[ct.TensorType(name='real_out', dtype=np.float32),
                 ct.TensorType(name='imag_out', dtype=np.float32)],
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=compute_precision(precision),
        compute_units=ct.ComputeUnit.CPU_ONLY, skip_model_load=True)
    converted.user_defined_metadata['lavasr.stage'] = stage
    converted.user_defined_metadata['lavasr.source'] = Path(gguf).name
    converted.user_defined_metadata['lavasr.precision'] = precision
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    converted.save(str(output))
    return output


def main():
    parser = argparse.ArgumentParser(description='Export a LavaSR GGUF neural core to Core ML.')
    parser.add_argument('--stage', choices=STAGES, required=True)
    parser.add_argument('--gguf', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--frames', type=int)
    parser.add_argument('--precision', choices=('float32', 'float16'), required=True)
    parser.add_argument('--compile-dir')
    args = parser.parse_args()
    path = export_model(args.stage, args.gguf, args.output, args.frames, args.precision)
    if args.compile_dir:
        Path(args.compile_dir).mkdir(parents=True, exist_ok=True)
        subprocess.run(['xcrun', 'coremlcompiler', 'compile', str(path), args.compile_dir], check=True)


if __name__ == '__main__':
    main()
