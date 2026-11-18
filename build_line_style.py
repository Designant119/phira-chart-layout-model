#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""每条判定线的 x 位置风格：均值/标准差（生成时作为整条线的宽度条件）。"""
import os,glob,json,argparse,numpy as np
from build_motif_index import X_SCALE
ap=argparse.ArgumentParser();ap.add_argument('--roots',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
roots=[r.strip() for r in a.roots.split(',') if r.strip()]
out={}
for r in roots:
 for fp in glob.glob(os.path.join(r,'parsed','*.json')):
  fid=os.path.basename(fp)[:-5]
  try:notes=json.load(open(fp)).get('notes',[])
  except Exception:continue
  for lid in sorted({int(n.get('line_id',0)) for n in notes}):
   xs=np.array([float(n['x']) for n in notes if int(n.get('line_id',0))==lid],np.float32)/X_SCALE
   if len(xs):out.setdefault(fid,{})[str(lid)]=[float(xs.mean()),float(xs.std())]
json.dump(out,open(a.out,'w'));print('lines',sum(len(v) for v in out.values()),'charts',len(out))
