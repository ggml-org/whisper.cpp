#!/usr/bin/env python3
"""Frozen real-GGUF affine cases; synthetic activations, no ASR timing claims."""
import argparse
import hashlib
import json
from pathlib import Path
import gguf
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--packed',type=Path,required=True);ap.add_argument('--dense',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    assert not args.output.exists();args.output.mkdir(parents=True)
    packed={t.name:t for t in gguf.GGUFReader(args.packed).tensors};dense={t.name:t for t in gguf.GGUFReader(args.dense).tensors}
    names=[f'encoder.blk.{layer}.ff1.linear{part}.weight' for layer in [0,6] for part in [1,2]]
    rng=np.random.default_rng(17);cases=[]
    for index,name in enumerate(names):
        t=packed[name];bits=4 if t.tensor_type==gguf.GGMLQuantizationType.Q4_0 else 8
        assert t.tensor_type in [gguf.GGMLQuantizationType.Q4_0,gguf.GGMLQuantizationType.Q8_0]
        w=np.asarray(dense[name].data).astype(np.float16);n,k=w.shape
        step=np.asarray(packed[name+'.affine.step'].data,dtype=np.float32).reshape(n)
        minimum=np.asarray(packed[name+'.affine.minimum'].data,dtype=np.float32).reshape(n)
        raw=np.asarray(t.data).view(np.uint8).reshape(-1,18 if bits==4 else 34)
        assert np.all(raw[:,:2]==np.array([0,60],dtype=np.uint8))
        restored=((gguf.quants.dequantize(t.data,t.tensor_type).reshape(n,k)+(8 if bits==4 else 128))*step[:,None]+minimum[:,None]).astype(np.float16)
        assert np.array_equal(restored.view(np.uint16),w.view(np.uint16))
        folder=args.output/f'weight_{index}';folder.mkdir()
        for f,data in [('codes',t.data),('step',step),('minimum',minimum),('dense',w)]:folder.joinpath(f+'.bin').write_bytes(np.ascontiguousarray(data).tobytes())
        for m in [1,8,32,60,128]:
            case=folder/f'm_{m}';case.mkdir()
            x=rng.standard_normal((m,k)).astype(np.float16).astype(np.float32)
            x.tofile(case/'x.bin');(x@w.astype(np.float32).T).tofile(case/'cpu.bin')
            cases.append({'name':name,'M':m,'N':n,'K':k,'bits':bits,'folder':str(folder.resolve()),'path':str(case.resolve())})
    # Non-multiple output dimensions, constant metadata and extrema exercise tails.
    for bits in [4,8]:
        n,k=70,96;codes=rng.integers(0,2**bits,(n,k),dtype=np.uint8);codes[0]=0;codes[1]=2**bits-1
        step=rng.uniform(0.0001,0.1,n).astype(np.float32);step[0]=0
        minimum=rng.uniform(-1,0,n).astype(np.float32)
        q=codes.reshape(n,k//32,32)
        payload=q[:,:,:16]|(q[:,:,16:]<<4) if bits==4 else (q.astype(np.int16)-128).astype(np.int8).view(np.uint8)
        raw=np.concatenate([np.ones((n,k//32,1),np.float16).view(np.uint8),payload],axis=-1)
        w=(codes.astype(np.float32)*step[:,None]+minimum[:,None]).astype(np.float16)
        folder=args.output/f'edge_{bits}';folder.mkdir()
        for f,data in [('codes',raw),('step',step),('minimum',minimum),('dense',w)]:np.ascontiguousarray(data).tofile(folder/(f+'.bin'))
        for m in [1,7,33,65]:
            case=folder/f'm_{m}';case.mkdir();x=rng.standard_normal((m,k)).astype(np.float16).astype(np.float32)
            x.tofile(case/'x.bin');(x@w.astype(np.float32).T).tofile(case/'cpu.bin')
            cases.append({'name':'synthetic_tail_constant','M':m,'N':n,'K':k,'bits':bits,'folder':str(folder.resolve()),'path':str(case.resolve())})
    hashes={str(p.relative_to(args.output)):sha(p) for p in args.output.rglob('*.bin')}
    (args.output/'cases.json').write_text(json.dumps({'cases':cases,'payload_hashes':hashes,'packed_sha256':sha(args.packed),'dense_sha256':sha(args.dense),'seed':17,'scope':'Real unchanged FFN weights; synthetic FP16-representable activations stored F32. Edge/constant cases separate. Resident primitive timings are not transcription times.'},indent=2)+'\n')
    print('Exported',len(cases),'cases',flush=True)

if __name__=='__main__':main()
