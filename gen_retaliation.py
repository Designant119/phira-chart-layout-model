#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 motif MLP v2 给 Retaliation 生成一版 x 排法，输出可渲染目录。"""
import os,json,argparse
import numpy as np, torch
from build_motif_index import line_motifs,L,X_SCALE
from train_motif_mlp2 import Net

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--chartdir',required=True); ap.add_argument('--parsed',required=True); ap.add_argument('--ckpt',required=True); ap.add_argument('--out',default=None); ap.add_argument('--scale',type=float,default=2.2)
    a=ap.parse_args(); d=torch.load(a.ckpt,map_location='cpu'); net=Net(d['d']); net.load_state_dict(d['model']); net.eval()
    P=json.load(open(a.parsed)); notes=P['notes']
    chart=json.load(open(os.path.join(a.chartdir,'59045584.json')))
    lines=chart['judgeLineList']
    gen={}  # lid -> x array
    for lid in sorted(set(int(n['line_id']) for n in notes)):
        ns=[n for n in notes if int(n['line_id'])==lid]; ns.sort(key=lambda n:n['t']); n=len(ns)
        seed=float(np.mean([n['x'] for n in ns]))/X_SCALE
        qm={m[7]:m[0] for m in line_motifs(notes,lid,'x')}
        xpred=np.full(n,np.nan,np.float32); cnt=np.zeros(n,np.float32)
        for s in range(0,max(0,n-L+1),4):
            q=qm.get(s)
            if q is None: break
            hist=xpred[:s]; hist=hist[np.isfinite(hist)]
            ctx=float(np.median(hist[-16:])) if len(hist) else seed
            p4=np.full(4,ctx,np.float32)
            if s>0:
                for k in range(4):
                    j=s-4+k
                    if 0<=j<n and np.isfinite(xpred[j]): p4[k]=xpred[j]
            sd=float(np.std(hist[-16:])) if len(hist) else 0.; mn=float(np.min(hist[-16:])) if len(hist) else ctx; mx=float(np.max(hist[-16:])) if len(hist) else ctx
            F=np.concatenate([q.astype(np.float32),np.array([ctx],np.float32),p4-ctx,np.array([sd],np.float32),np.array([mn-ctx,mx-ctx],np.float32)])
            with torch.no_grad(): y=net(torch.from_numpy(F)[None,:]).numpy()[0]
            pred=np.clip(ctx+a.scale*y,-1.2,1.2)
            for k in range(L):
                j=s+k
                if j<n and not np.isfinite(xpred[j]):
                    xpred[j]=pred[k]; cnt[j]+=1
        xpred[~np.isfinite(xpred)]=0.0
        gen[lid]=xpred
        print('line',lid,'notes',n,'x range',round(float(xpred.min()),3),round(float(xpred.max()),3),flush=True)
    # 写回 chart json：按 line + 真音符排序对应
    for lid,xg in gen.items():
        if lid>=len(lines): continue
        raw=[n for n in lines[lid].get('notes',[]) if n.get('isFake',0)!=1 and int(n.get('type',1)) in (1,2,3,4)]
        raw.sort(key=lambda n:(float(n['startTime'][0])+float(n['startTime'][1])/float(n['startTime'][2])))
        if len(raw)!=len(xg):
            print('WARN line',lid,'raw',len(raw),'gen',len(xg)); continue
        for n,x in zip(raw,xg):
            n['positionX']=float(x)*X_SCALE
    out=os.path.join(a.chartdir,'retaliation_gen.json'); json.dump(chart,open(out,'w'),ensure_ascii=False)
    info=open(os.path.join(a.chartdir,'info.yml')).read().replace('chart: 59045584.json','chart: retaliation_gen.json')
    open(os.path.join(a.chartdir,'info_gen.yml'),'w').write(info)
    print('saved',out)
if __name__=='__main__':main()
