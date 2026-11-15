#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全量 Phira motif 分块索引：每 15 万 motif 存一个 npz，避免一次性吃满内存。
用法：python3 build_motif_parts.py --roots <corpus,...> --out <dir>
"""
import os, glob, json, argparse, time
import numpy as np
from build_motif_index import line_motifs, L, STRIDE

def flush(buf, out, idx, totals):
    if not buf: return idx, totals
    Q=np.stack([m[0] for m in buf]); X=np.stack([m[1] for m in buf]); R=np.stack([m[2] for m in buf]); T=np.stack([m[3] for m in buf])
    meta={"fid":np.array([m[5] for m in buf]),"line":np.array([m[6] for m in buf]),
          "start":np.array([m[7] for m in buf]),"t0":np.array([m[4][0] for m in buf])}
    fp=os.path.join(out,f"part_{idx:03d}.npz")
    np.savez_compressed(fp,Q=Q,X=X,R=R,T=T,**meta)
    totals["motifs"]+=len(buf); totals["jack"]+=float(np.sum(np.abs(np.diff(X,axis=1))<1e-4))
    totals["hold"]+=int(np.sum(T==2)); totals["drag"]+=int(np.sum(T==3)); totals["flick"]+=int(np.sum(T==4))
    print(f"saved {fp} {len(buf)} motifs",flush=True)
    return idx+1, totals

def main():
    p=argparse.ArgumentParser(); p.add_argument("--roots",required=True); p.add_argument("--out",required=True); p.add_argument("--part-size",type=int,default=150000); p.add_argument("--max-charts",type=int,default=0)
    a=p.parse_args(); roots=[r.strip() for r in a.roots.split(",") if r.strip()]
    fids=set()
    for r in roots: fids|={os.path.basename(x)[:-5] for x in glob.glob(os.path.join(r,"parsed","*.json"))}
    fids=sorted(fids)
    if a.max_charts>0: fids=fids[:a.max_charts]
    os.makedirs(a.out,exist_ok=True)
    buf=[]; idx=0; totals={"charts":0,"motifs":0,"jack":0.0,"hold":0,"drag":0,"flick":0}; t0=time.time()
    for fi,fid in enumerate(fids):
        fp=None
        for r in roots:
            q=os.path.join(r,"parsed",f"{fid}.json")
            if os.path.exists(q): fp=q; break
        if not fp: continue
        try: notes=json.load(open(fp)).get("notes",[])
        except Exception: continue
        if not notes: continue
        totals["charts"]+=1
        for lid in sorted({int(n.get("line_id",0)) for n in notes}):
            buf+=line_motifs(notes,lid,fid)
        if len(buf)>=a.part_size:
            idx,totals=flush(buf,a.out,idx,totals); buf=[]
        if (fi+1)%100==0: print(f"{fi+1}/{len(fids)} charts, motifs {totals['motifs']+len(buf)}, {time.time()-t0:.0f}s",flush=True)
    idx,totals=flush(buf,a.out,idx,totals)
    m=totals["motifs"]
    stats={"charts":totals["charts"],"motifs":m,"L":L,"stride":STRIDE,"q_dim":96,"parts":idx,
           "jack_rate":totals["jack"]/(11*max(m,1)),"hold_rate":totals["hold"]/(12*max(m,1)),
           "drag_rate":totals["drag"]/(12*max(m,1)),"flick_rate":totals["flick"]/(12*max(m,1))}
    json.dump(stats,open(os.path.join(a.out,"stats.json"),"w"),ensure_ascii=False,indent=1)
    print("saved",a.out,stats,flush=True)

if __name__=="__main__": main()
