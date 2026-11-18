#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把每张谱的主判定线抽成单线序列，做成长度 128 的窗口，训练单线模型。"""
import os,glob,json,argparse,numpy as np
from build_motif_index import X_SCALE
W=128;STRIDE=64
ap=argparse.ArgumentParser();ap.add_argument('--roots',required=True);ap.add_argument('--split',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
roots=[r.strip() for r in a.roots.split(',') if r.strip()];split=json.load(open(a.split));val=set(split['val']);test=set(split['test'])
seen=set();rows=[]
for r in roots:
 for fp in sorted(glob.glob(os.path.join(r,'parsed','*.json'))):
  fid=os.path.basename(fp)[:-5]
  if fid in seen:continue
  seen.add(fid)
  try:d=json.load(open(fp))
  except Exception:continue
  notes=d.get('notes',[])
  if not notes:continue
  by={}
  for n in notes:by.setdefault(int(n.get('line_id',0)),[]).append(n)
  lid=max(by,key=lambda k:len(by[k]));ns=sorted(by[lid],key=lambda n:float(n['t']))
  if len(ns)<8:continue
  x=np.array([float(n['x']) for n in ns],np.float32)/X_SCALE;t=np.array([float(n['t']) for n in ns],np.float32);ty=np.array([int(n.get('type',1)) for n in ns],np.int32);dur=np.array([float(n.get('dur',0)) for n in ns],np.float32)
  rows.append((fid,lid,x,t,ty,dur))
print('charts',len(rows),'notes',sum(len(r[2]) for r in rows),'val',sum(r[0] in val for r in rows),'test',sum(r[0] in test for r in rows),flush=True)
F=[];Y=[];M=[];FID=[];ST=[]
for fid,lid,x,t,ty,dur in rows:
 n=len(x);chord=np.zeros(n,np.float32);dens=np.zeros(n,np.float32)
 for i in range(n):
  chord[i]=np.sum(np.abs(t-t[i])<1e-4)
  dens[i]=np.sum(np.abs(t-t[i])<4.0)
 for s in range(0,max(1,n-W+1),STRIDE):
  e=min(n,s+W);m=np.zeros(W,np.float32);m[:e-s]=1
  f=np.zeros((W,11),np.float32);y=np.zeros(W,np.float32);y[:e-s]=x[s:e]
  prev=0.0;prev2=0.0
  for k in range(W):
   i=s+k
   if i>=e:break
   dt=0.0 if k==0 else max(0.0,t[i]-t[i-1])
   f[k]=[np.log1p(dt),*(ty[i]==np.arange(1,5)),np.log1p(max(0.0,dur[i])),np.log1p(chord[i]),np.log1p(dens[i]),prev,prev2,prev-prev2]
   prev2=prev;prev=x[i]
  F.append(f);Y.append(y);M.append(m);FID.append(fid);ST.append(s)
  if e>=n:break
F=np.stack(F);Y=np.stack(Y);M=np.stack(M);FID=np.array(FID);ST=np.array(ST)
np.savez_compressed(a.out,F=F,Y=Y,M=M,fid=FID,start=ST);print('windows',len(F),'dim',F.shape[-1],flush=True)
