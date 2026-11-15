#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""motif 分块 + 前 4 音轨迹/局部线宽上下文。"""
import os,glob,json,argparse,time
import numpy as np
from build_motif_index import line_motifs,L,X_SCALE

def flush(buf,out,idx):
    Q=np.stack([m[0] for m in buf]);X=np.stack([m[1] for m in buf]);R=np.stack([m[2] for m in buf]);T=np.stack([m[3] for m in buf])
    C=np.array([m[8] for m in buf],np.float32)
    P4=np.stack([m[9] for m in buf]).astype(np.float32)
    SD=np.array([m[10] for m in buf],np.float32); MN=np.array([m[11] for m in buf],np.float32); MX=np.array([m[12] for m in buf],np.float32)
    meta={"fid":np.array([m[5] for m in buf]),"line":np.array([m[6] for m in buf]),"start":np.array([m[7] for m in buf]),"t0":np.array([m[4][0] for m in buf])}
    fp=os.path.join(out,f"part_{idx:03d}.npz");np.savez_compressed(fp,Q=Q,X=X,R=R,T=T,ctx=C,prev4=P4,ctx_std=SD,ctx_min=MN,ctx_max=MX,**meta)
    print('saved',fp,len(buf),flush=True);return idx+1

def main():
    p=argparse.ArgumentParser();p.add_argument('--roots',required=True);p.add_argument('--out',required=True);p.add_argument('--part-size',type=int,default=150000)
    a=p.parse_args();roots=[x.strip() for x in a.roots.split(',') if x.strip()]
    fids=set()
    for r in roots:fids|={os.path.basename(x)[:-5] for x in glob.glob(os.path.join(r,'parsed','*.json'))}
    fids=sorted(fids);os.makedirs(a.out,exist_ok=True);buf=[];idx=0;t0=time.time()
    for fi,fid in enumerate(fids):
        fp=None
        for r in roots:
            q=os.path.join(r,'parsed',f'{fid}.json')
            if os.path.exists(q):fp=q;break
        if not fp:continue
        try:notes=json.load(open(fp)).get('notes',[])
        except Exception:continue
        if not notes:continue
        for lid in sorted({int(n.get('line_id',0)) for n in notes}):
            ns=[n for n in notes if int(n.get('line_id',0))==lid and int(n.get('type',1)) in (1,2,3,4)];ns.sort(key=lambda n:n['t'])
            if len(ns)<L:continue
            x=np.clip(np.array([q['x'] for q in ns],dtype=np.float32)/X_SCALE,-1.2,1.2)
            for m in line_motifs(notes,lid,fid):
                s=m[7];h=x[max(0,s-16):s];c=float(np.median(h)) if len(h) else float(x[s])
                p4=np.full(4,c,np.float32)
                for k in range(4):
                    if s-4+k>=0:p4[k]=x[s-4+k]
                sd=float(np.std(h)) if len(h) else 0.;mn=float(np.min(h)) if len(h) else c;mx=float(np.max(h)) if len(h) else c
                buf.append(m+(c,p4,sd,mn,mx))
                if len(buf)>=a.part_size:idx=flush(buf,a.out,idx);buf=[]
        if (fi+1)%100==0:print(f'{fi+1}/{len(fids)} charts {time.time()-t0:.0f}s',flush=True)
    if buf:idx=flush(buf,a.out,idx)
    json.dump({'parts':idx},open(os.path.join(a.out,'meta.json'),'w'));print('done',idx,flush=True)
if __name__=='__main__':main()
