#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pec 老格式谱面补角度：读取 cr 旋转事件，计算每个 note 的 theta/theta_end。
输出与 build_angle_parsed.py 相同结构，并只处理 angle_stats 里 NOCHART 的 fid。
"""
import os,glob,json,zipfile,argparse,sys
sys.path.insert(0,'.')
from easing_lib import ease

X_SCALE=675.0

def beat_to_sec(beat,segs):
    total=0.0
    for i,(sb,bpm) in enumerate(segs):
        if i+1<len(segs):
            nb=segs[i+1][0]
            if beat<nb:return total+(beat-sb)*60.0/bpm
            total+=(nb-sb)*60.0/bpm
        else:return total+(beat-sb)*60.0/bpm
    return total

def angle_at(events,t):
    """events: [(sb,eb,value,easing)]，从上一事件目标值插值到当前目标值。"""
    if not events:
        return 0.0
    evs=sorted(events,key=lambda x:x[0])
    prev=0.0
    for sb,eb,val,et in evs:
        if t>=eb:
            prev=val
            continue
        if t>=sb:
            u=(t-sb)/(eb-sb) if eb>sb else 1.0
            return prev+(val-prev)*ease(u,et)
        return prev
    return prev

def parse_pec_angle(raw):
    segs=[];raw_notes=[];events={}
    for line in raw.splitlines():
        s=line.strip()
        if not s:continue
        p=s.split();tag=p[0]
        try:
            if tag=='bp':
                segs.append([float(p[1]),float(p[2])])
            elif tag=='n1':
                raw_notes.append((int(p[1]),float(p[2]),float(p[3]),1,int(p[5]),None))
            elif tag=='n2':
                raw_notes.append((int(p[1]),float(p[2]),float(p[4]),2,int(p[6]),float(p[3])))
            elif tag=='n3':
                raw_notes.append((int(p[1]),float(p[2]),float(p[3]),3,int(p[5]),None))
            elif tag=='n4':
                raw_notes.append((int(p[1]),float(p[2]),float(p[3]),4,int(p[5]),None))
            elif tag=='cr':
                # cr <line> <startBeat> <endBeat> <valueDeg> <easing>
                lid=int(p[1]);sb=float(p[2]);eb=float(p[3]);v=float(p[4]);et=int(p[5]) if len(p)>5 else 1
                events.setdefault(lid,[]).append((sb,eb,v,et))
        except (ValueError,IndexError):
            continue
    segs.sort(key=lambda x:x[0])
    if not segs:segs=[[0.0,120.0]]
    notes=[]
    for lid,tb,x,typ,fake,endb in raw_notes:
        if fake:continue
        th=angle_at(events.get(lid,[]),tb)
        th=((th+180)%360)-180
        dur=0.0;the=th
        if endb is not None:
            dur=max(0.0,beat_to_sec(endb,segs)-beat_to_sec(tb,segs))
            the=angle_at(events.get(lid,[]),endb);the=((the+180)%360)-180
        notes.append({'t':round(beat_to_sec(tb,segs),6),'x':x,'type':typ,'dur':round(dur,6),'line_id':lid,'theta':round(th,3),'theta_end':round(the,3)})
    notes.sort(key=lambda n:n['t'])
    return notes

def pick_pec(zf):
    best='';bestn=-1
    for nm in zf.namelist():
        base=nm.rsplit('/',1)[-1].lower()
        if not base.endswith('.json') or base in ('info.json','extra.json'):continue
        try:raw=zf.read(nm).decode('utf-8','replace')
        except Exception:continue
        n=sum(1 for l in raw.splitlines() if l.strip().startswith(('n1','n2','n3','n4')))
        if n>bestn:bestn=n;best=raw
    return best if bestn>0 else None

def one(zp,outdir,fdfilter):
    fid=os.path.basename(zp).split('_',1)[0]
    if not fid.isdigit() or (fdfilter and fid not in fdfilter):return None
    try:
        with zipfile.ZipFile(zp) as zf:raw=pick_pec(zf)
    except Exception as e:return (fid,'ERR',repr(e)[:80],0,0)
    if not raw:return (fid,'NOCHART','',0,0)
    notes=parse_pec_angle(raw)
    if not notes:return (fid,'NOCHART','',0,0)
    good=[n for n in notes if abs(n['theta'])<=10 and abs(n['theta_end'])<=10]
    json.dump({'fid':fid,'notes':notes,'source':'pec'},open(os.path.join(outdir,'angle_parsed',fid+'.json'),'w'),ensure_ascii=False)
    json.dump({'fid':fid,'notes':good,'source':'pec'},open(os.path.join(outdir,'angle_filtered',fid+'.json'),'w'),ensure_ascii=False)
    return (fid,'OK','',len(notes),len(good))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--zips',required=True);ap.add_argument('--parsed',required=True);ap.add_argument('--out',required=True);ap.add_argument('--stats',required=True);ap.add_argument('--workers',type=int,default=4);a=ap.parse_args()
    fdfilter=set(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(a.parsed,'*.json')))
    old=json.load(open(a.stats));nochart={x[0] for x in old if x[1]=='NOCHART'}
    zps=[p for p in sorted(glob.glob(os.path.join(a.zips,'*','*.zip')))+sorted(glob.glob(os.path.join(a.zips,'*.zip'))) if os.path.basename(p).split('_',1)[0] in nochart]
    print('pec candidates',len(zps),flush=True)
    os.makedirs(os.path.join(a.out,'angle_parsed'),exist_ok=True);os.makedirs(os.path.join(a.out,'angle_filtered'),exist_ok=True)
    from concurrent.futures import ProcessPoolExecutor
    stats=[];n=0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(one,zps,[a.out]*len(zps),[fdfilter]*len(zps)):
            if not r:continue
            stats.append(r);n+=1
            if n%100==0:print('done',n,'last',r[0],r[1],r[3],r[4],flush=True)
    json.dump(stats,open(os.path.join(a.out,'angle_pec_stats.json'),'w'),ensure_ascii=False)
    print('ok',sum(s[1]=='OK' for s in stats),'nochart',sum(s[1]=='NOCHART' for s in stats),'err',sum(s[1]=='ERR' for s in stats),flush=True)
    print('notes',sum(s[3] for s in stats),'valid',sum(s[4] for s in stats),flush=True)

if __name__=='__main__':main()
