#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""每张谱音符最多的判定线，作为单线训练主目标。"""
import os,glob,json,sys
root=sys.argv[1];out=sys.argv[2];m={}
for fp in glob.glob(os.path.join(root,'parsed','*.json')):
 fid=os.path.basename(fp)[:-5]
 try:notes=json.load(open(fp)).get('notes',[])
 except Exception:continue
 cnt={}
 for n in notes:cnt[int(n.get('line_id',0))]=cnt.get(int(n.get('line_id',0)),0)+1
 if cnt:m[fid]=max(cnt,key=cnt.get)
json.dump(m,open(out,'w'));print(len(m))
