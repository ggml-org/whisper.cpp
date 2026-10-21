#!/usr/bin/env python3
"""Opt in to serialized original-head self-attention QKV fusion.

Optimize an EXISTING per-channel int8 + fp16 heads, rows contract-2 package;
no conversion, retracing, requantization or default export changes. Supported
geometries: tiny L2/D64/H2/HD32/ctx16/cross8/Q2/vocab5 and mini
L24/D1024/H16/HD64/ctx512/cross128/Q9/vocab1088, with all layers per state.

Example:
  python optimize-decoder-coreml.py --package old/decoder.mlpackage \
    --out optimized/decoder.mlpackage --compile-dir optimized/compiled

Output package and optional compile directory must be NEW. Keep the original
package basename to preserve the compiler's generated class name. Each output
is published atomically; a publication failure may leave one validated output.
The append-only format retains inactive Q/K/V payloads. In the measured mini
package the compiled disk footprint increased about 20.2% (373.9 -> 449.5 MB).
Performance and output parity depend on model/hardware/compute units: validate
your deployment. The measured tiny CPU_ONLY case was not bitwise equivalent.
This command checks serialized graph/learned bytes, not inference equivalence.
Requires coremltools protobuf; compilation additionally requires macOS/Xcode.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--package', type=Path, required=True, help='original, unoptimized .mlpackage')
    parser.add_argument('--out', type=Path, required=True, help='new path with the SAME .mlpackage basename')
    parser.add_argument('--compile-dir', type=Path, help='optional NEW directory for the compiled .mlmodelc')
    args = parser.parse_args(argv)
    # Explicit sibling package import also supports python -I -B without adding
    # cwd or user-provided directories to sys.path.
    root = Path(__file__).resolve().parent / 'parler_qkv'
    module_spec = importlib.util.spec_from_file_location('parler_qkv', root / '__init__.py',
                                                       submodule_search_locations=[str(root)])
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    from parler_qkv.package import optimize
    try:
        print(json.dumps(optimize(args.package, args.out, args.compile_dir), indent=2))
    except (ValueError, KeyError, TypeError, IndexError, OSError, ImportError, subprocess.CalledProcessError) as error:
        print('QKV optimization failed: ' + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
