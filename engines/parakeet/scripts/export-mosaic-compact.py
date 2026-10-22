#!/usr/bin/env python3
"""Reconstruct the retained Mosaic encoder-source artifact from the original archive."""
import argparse
import collections
import hashlib
import importlib.util
import importlib.metadata
import json
from pathlib import Path
import sys


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def main():
    import gguf
    import numpy as np
    import torch
    scripts=Path(__file__).resolve().parent;configs=scripts.parent/'configs'
    ap=argparse.ArgumentParser(prog="export-mosaic.py",description=__doc__)
    ap.add_argument('--archive',type=Path,required=True)
    ap.add_argument('--output',type=Path,default=Path('models/parakeet-mosaic.encoder-source.gguf'))
    args=ap.parse_args()
    assert not args.output.exists(),'Preserve existing artifact'
    recipe_path=configs/'parakeet-mosaic-compact.recipe.json'
    historical_path=configs/'parakeet-mosaic.recipe.json';golden_path=configs/'parakeet-mosaic.tensors.json'
    recipe=json.loads(recipe_path.read_text());historical=json.loads(historical_path.read_text())
    assert sha(args.archive)==recipe['source_archive_sha256']
    assert sha(historical_path)==recipe['historical_recipe_sha256']
    assert sha(golden_path)==recipe['historical_effective_tensor_receipts_sha256']
    assert importlib.metadata.version('gguf')=='0.19.0','Exact artifact reproduction requires gguf==0.19.0'
    torch.set_num_threads(4);sys.path.insert(0,str(scripts))
    converter=load('compact_converter',scripts/'convert-nemo-to-gguf.py')
    exporter=load('compact_historical',scripts/'export-packed-affine.py')
    cfg,state,tok,multi=converter.load_nemo(args.archive)
    state={n:t.half() if t.is_floating_point() else t for n,t in state.items()}
    effective=dict(state);_,coverage=exporter.apply_recipe(effective,historical['tensors'])
    receipts={n:{'shape':list(t.shape),'dtype':str(t.dtype),'sha256':hashlib.sha256(t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()} for n,t in effective.items()}
    assert len(receipts)==725 and receipts==json.loads(golden_path.read_text())['tensors']
    # Preserve the historical depthwise effective values; target matrices start
    # from original FP16 values, not previously affine-rounded matrix values.
    for item in coverage:
        if item['gguf'] is None:state[item['source']]=effective[item['source']]
    del effective,receipts
    targets=recipe['target_bits'];assert len(targets)==264
    assert collections.Counter(targets.values())=={4:122,8:142}
    seen=set()
    class Writer(gguf.GGUFWriter):
        def __init__(self,path,arch):
            super().__init__(path,arch)
            self.add_string('mosaic.quantizer','ggml_block32');self.add_string('mosaic.recipe','group32_attention8')
        def add_name(self,name):super().add_name('parakeet-mosaic-group32_attention8')
        def add_file_type(self,kind):super().add_file_type(gguf.LlamaFileType.MOSTLY_Q4_0)
        def add_tensor(self,name,tensor,**kwargs):
            if '.attn.qkv.' in name:return
            if name in targets:
                arr=np.asarray(tensor,dtype=np.float32)
                assert arr.ndim==2 and arr.shape[-1]%32==0
                typ=gguf.GGMLQuantizationType.Q4_0 if targets[name]==4 else gguf.GGMLQuantizationType.Q8_0
                seen.add(name)
                return super().add_tensor(name,gguf.quants.quantize(arr,typ),raw_dtype=typ)
            return super().add_tensor(name,tensor,**kwargs)
        def write_header_to_file(self,*a,**kw):
            assert seen==set(targets)
            return super().write_header_to_file(*a,**kw)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    converter.write_gguf(args.output,args.archive,cfg,state,tok,multi,'f16','tdt',writer_factory=Writer)
    assert args.output.stat().st_size==recipe['artifact']['bytes'] and sha(args.output)==recipe['artifact']['sha256'],'Artifact differs from frozen recipe; preserve for diagnosis'
    receipt={'source_archive_sha256':sha(args.archive),'recipe_sha256':sha(recipe_path),'source_tensors_verified':725,
        'matrix_counts':dict(collections.Counter(targets.values())),'bytes':args.output.stat().st_size,'sha256':sha(args.output),
        'gguf_version':'0.19.0','torch_version':torch.__version__,'requires_affine_fused_kernel':False}
    args.output.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
