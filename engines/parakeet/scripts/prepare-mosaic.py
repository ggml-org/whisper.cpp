#!/usr/bin/env python3
"""Prepare Mosaic GGUF, optionally with a matching masked Core ML sidecar."""
import argparse
import hashlib
import importlib.util
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import traceback


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def inventory(folder):
    return {str(p.relative_to(folder)):{'bytes':p.stat().st_size,'sha256':sha(p)}
            for p in sorted(folder.rglob('*')) if p.is_file()}


def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    source=ap.add_mutually_exclusive_group(required=True)
    source.add_argument('--archive',type=Path,help='Original NVIDIA .nemo; rebuild exact Mosaic GGUF')
    source.add_argument('--gguf',type=Path,help='Reuse an existing exact Mosaic GGUF, verified by full SHA256')
    ap.add_argument('--output-dir',type=Path,default=Path('models/mosaic'))
    ap.add_argument('--coreml',action='store_true',help='Also generate the optional Apple Core ML encoder sidecar')
    ap.add_argument('--variant',choices=('mosaic','compact'),default='mosaic',
                    help='Select the exact shared Mosaic or Mosaic Compact artifact')
    args=ap.parse_args()
    if args.output_dir.exists():ap.error('Output directory exists; use a fresh directory to preserve prior artifacts')
    scripts=Path(__file__).resolve().parent
    stem='parakeet-mosaic' + ('-compact' if args.variant=='compact' else '')
    definition=json.loads((scripts.parent/f'configs/{stem}.default.json').read_text())
    if args.variant=='compact' and args.archive:
        ap.error('Compact requires --gguf with the exact supplied artifact; archive reconstruction is not supported')
    input_path=args.archive or args.gguf
    if not input_path.is_file():ap.error(f'Missing source: {input_path}')
    if args.gguf and (input_path.stat().st_size!=definition['bytes'] or sha(args.gguf)!=definition['sha256']):
        ap.error(f'GGUF is not the frozen {stem} artifact')
    # Native GGUF preparation works on Linux without Core ML dependencies.
    if args.coreml:
        if sys.platform != 'darwin':
            ap.error('--coreml requires macOS; omit it for native GGUF deployment')
        import torch
        import numpy as np
        import coremltools as ct
        if torch.__version__.split('+')[0]!='2.7.0' or ct.__version__!='9.0':
            ap.error('Use the validated sidecar exporter environment: torch==2.7.0, coremltools==9.0')
    out=args.output_dir;out.mkdir(parents=True,exist_ok=False)
    try:
        source_files=list(scripts.glob('*.py'))+list((scripts.parent/'configs').glob('parakeet-mosaic*json'))
        plan={'name':stem,'version':'1','source':str(input_path.resolve()),'source_sha256':sha(input_path),
              'source_files':{str(p.relative_to(scripts.parent)):sha(p) for p in sorted(source_files)},
              'environment':sorted(f'{d.metadata["Name"]}=={d.version}' for d in importlib.metadata.distributions() if d.metadata.get('Name')),
              'graph':'Optional masked512 FLOAT16 graph' if args.coreml else 'native GGML only'}
        save(out/'plan.json',plan)
        model=out/f'{stem}.gguf'
        if args.archive:
            with (out/'gguf-export.log').open('w') as log:
                subprocess.run([sys.executable,str(scripts/'export-mosaic.py'),'--archive',str(args.archive),'--output',str(model)],stdout=log,stderr=subprocess.STDOUT,check=True)
        else:shutil.copy2(args.gguf,model)
        assert sha(model)==definition['sha256'] and model.stat().st_size==definition['bytes']
        if not args.coreml:
            save(out/'complete.json',{'returncode':0,'plan_sha256':sha(out/'plan.json'),
                'gguf_sha256':sha(model),'shipping_bytes':model.stat().st_size,
                'sidecar_generated':False,'scope':'Native GGUF preparation; inference validation is separate'})
            print(f'Ready: {model}',flush=True)
            return
        spec=importlib.util.spec_from_file_location('mosaic_coreml_exporter',scripts/'export-encoder-coreml.py')
        exporter=importlib.util.module_from_spec(spec);spec.loader.exec_module(exporter)
        ref=exporter.load_reference_encoder(scripts);weights,meta=ref.load_gguf(model)
        assert exporter.validate_export_contract(meta,flexible=False,bypass_pre_encode=False)=='tdt'
        torch.set_num_threads(4);torch.manual_seed(17);np.random.seed(17)
        original_convert=ct.convert
        def unloaded(*a,**kw):return original_convert(*a,skip_model_load=True,**kw)
        ct.convert=unloaded
        try:
            mlmodel=exporter.convert_fixed(ref,weights,meta,torch.zeros(128,512),128,512,
                    int(meta['parakeet.encoder.d_model']),ct.precision.FLOAT16,np.float16,ct.target.macOS13)
        finally:ct.convert=original_convert
        contract={x.name:list(x.type.multiArrayType.shape) for x in mlmodel.get_spec().description.input}
        assert contract=={'mel':[128,512],'mel_valid_mask':[512]}
        package=out/f'{stem}-encoder.mlpackage';mlmodel.save(str(package))
        compiled=Path(ct.models.utils.compile_model(str(package)))
        final=out/f'{stem}-encoder.mlmodelc';shutil.move(str(compiled),final)
        assert sha(package/'Data/com.apple.CoreML/weights/weight.bin')==sha(final/'weights/weight.bin')
        for name,h in plan['source_files'].items():assert sha(scripts.parent/name)==h
        save(out/'complete.json',{'returncode':0,'plan_sha256':sha(out/'plan.json'),'gguf_sha256':sha(model),
            'inputs':contract,'source_inventory':inventory(package),'compiled_inventory':inventory(final),
            'shipping_bytes':model.stat().st_size+sum(p.stat().st_size for p in final.rglob('*') if p.is_file()),
            'scope':'Artifact generation, not transcript/numerical validation or proof of ANE placement'})
        print(f'Ready: {model} + {final}\nSource mlpackage is optional for deployment.',flush=True)
    except BaseException as error:
        save(out/'failure.json',{'error':str(error),'traceback':traceback.format_exc()});raise


if __name__=='__main__':main()
