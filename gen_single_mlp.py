#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""单判定线：用 motif MLP 生成整条线的 x，再按线宽归一化。"""
import os,json,argparse
import numpy as np,torch
from build_motif_index import line_motifs,L,X_SCALE
from train_motif_mlp2 import Net
ap=argparse.ArgumentParser();ap.add_argument('--chart',required=True);ap.add_argument('--out',required=True);ap.add_argument('--ckpt',required=True);ap.add_argument('--width',type=float,default=.46);ap.add_argument('--alpha',type=float,default=1.0);a=ap.parse_args()
d=torch.load(a.ckpt,map_location='cpu');net=Net(d['d']);net.load_state_dict(d['model']);net.eval();ch=json.load(open(a.chart));ln=ch['judgeLineList'][0];raw=ln['notes'];raw.sort(key=lambda n:(float(n['startTime'][0])+float(n['startTime'][1])/float(n['startTime'][2])));syn=[]
for n in raw:
 t=float(n['startTime'][0])+float(n['startTime'][1])/float(n['startTime'][2]);e=float(n['endTime'][0])+float(n['endTime'][1])/float(n['endTime'][2]);syn.append({'t':t,'x':0.0,'type':int(n.get('type',1)),'dur':max(0,e-t),'line_id':0})
n=len(syn);qm={m[7]:m[0] for m in line_motifs(syn,0,'x')};xs=np.full(n,np.nan,np.float32);seed=0.0
for s in range(0,max(0,n-L+1),4):
 q=qm.get(s)
 if q is None:break
 hist=xs[:s];hist=hist[np.isfinite(hist)];ctx=float(np.median(hist[-16:])) if len(hist) else seed
 p4=np.full(4,ctx,np.float32)
 for k in range(4):
  j=s-4+k
  if 0<=j<n and np.isfinite(xs[j]):p4[k]=xs[j]
 sd=float(np.std(hist[-16:])) if len(hist) else a.width;mn=float(np.min(hist[-16:])) if len(hist) else ctx;mx=float(np.max(hist[-16:])) if len(hist) else ctx
 F=np.concatenate([q.astype(np.float32),np.array([ctx],np.float32),p4-ctx,np.array([sd],np.float32),np.array([mn-ctx,mx-ctx],np.float32)])
 with torch.no_grad():y=net(torch.from_numpy(F)[None,:]).numpy()[0]
 z=(y-y.mean())/(y.std()+1e-6);pred=np.clip(ctx+a.alpha*a.width*z,-1.2,1.2)
 for k in range(L):
  j=s+k
  if j<n and not np.isfinite(xs[j]):xs[j]=pred[k]
xs[~np.isfinite(xs)]=seed
for n_,x in zip(raw,xs):n_['positionX']=float(x)*X_SCALE
json.dump(ch,open(a.out,'w'),ensure_ascii=False);print('notes',n,'gen std',round(float(np.std(xs)),3),'target',a.width)
