#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从本地完整 zip 重解析谱面，并计算每个 note 所在判定线的旋转角度。
输出 angle_parsed（保留全部 note+theta）和 angle_filtered（只留 |theta|<=10 的 note）。"""
import os,glob,json,zipfile,argparse,sys
from collections import defaultdict
sys.path.insert(0,'.')
from easing_lib import line_state_at
from phira_fetch import rpe_time_to_beat
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

def parse_angle(chart):
    bpm_raw=chart.get('BPMList',[])
    segs=sorted([(rpe_time_to_beat(b['startTime']),float(b['bpm'])) for b in bpm_raw],key=lambda x:x[0])
    notes=[]
    for li,jl in enumerate(chart.get('judgeLineList',[])):
        cache={}
        for n in jl.get('notes',[]):
            if n.get('isFake',0)==1:continue
            typ=int(n.get('type',1));tb=rpe_time_to_beat(n['startTime']);t=beat_to_sec(tb,segs)
            th=line_state_at(jl,chart,tb,cache)[2];th=((th+180)%360)-180
            dur=0.0;the=th
            if typ==2 and isinstance(n.get('endTime'),list):
                eb=rpe_time_to_beat(n['endTime']);dur=max(0.0,beat_to_sec(eb,segs)-t);the=line_state_at(jl,chart,eb,cache)[2];the=((the+180)%360)-180
            notes.append({'t':round(t,6),'x':float(n.get('positionX',0)),'type':typ,'dur':round(dur,6),'line_id':li,'theta':round(float(th),3),'theta_end':round(float(the),3)})
    notes.sort(key=lambda n:n['t']);return notes

def pick_chart(zf):
    best=None;bestn=-1
    for nm in zf.namelist():
        base=nm.rsplit('/',1)[-1].lower()
        if not base.endswith('.json') or base in ('info.json','extra.json'):continue
        try:obj=json.loads(zf.read(nm))
        except Exception:continue
        if isinstance(obj,dict) and obj.get('judgeLineList'):
            n=sum(len(jl.get('notes',[])) for jl in obj['judgeLineList'])
            if n>bestn:best=obj;bestn=n
    return best

def one(zp,outdir,fdfilter):
    fid=os.path.basename(zp).split('_',1)[0]
    if not fid.isdigit() or (fdfilter and fid not in fdfilter):return None
    try:
        with zipfile.ZipFile(zp) as zf:ch=pick_chart(zf)
    except Exception as e:return (fid,'ERR',repr(e)[:80],0,0)
    if not ch:return (fid,'NOCHART','',0,0)
    notes=parse_angle(ch)
    allp=os.path.join(outdir,'angle_parsed',fid+'.json');filp=os.path.join(outdir,'angle_filtered',fid+'.json')
    good=[n for n in notes if abs(n['theta'])<=10 and abs(n['theta_end'])<=10]
    json.dump({'fid':fid,'notes':notes},open(allp,'w'),ensure_ascii=False)
    json.dump({'fid':fid,'notes':good},open(filp,'w'),ensure_ascii=False)
    return (fid,'OK','',len(notes),len(good))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--zips',required=True);ap.add_argument('--parsed',required=True);ap.add_argument('--out',required=True);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--limit',type=int,default=0);a=ap.parse_args()
    fdfilter=set(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(a.parsed,'*.json')))
    zps=sorted(glob.glob(os.path.join(a.zips,'*','*.zip')))+sorted(glob.glob(os.path.join(a.zips,'*.zip')))
    if a.limit:zps=zps[:a.limit]
    os.makedirs(os.path.join(a.out,'angle_parsed'),exist_ok=True);os.makedirs(os.path.join(a.out,'angle_filtered'),exist_ok=True)
    print('zips',len(zps),'fds',len(fdfilter),flush=True)
    from concurrent.futures import ProcessPoolExecutor
    stats=[];n=0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(one,zps,[a.out]*len(zps),[fdfilter]*len(zps)):
            if not r:continue
            stats.append(r);n+=1
            if n%100==0:print('done',n,'last',r[0],r[1],r[3],r[4],flush=True)
    json.dump(stats,open(os.path.join(a.out,'angle_stats.json'),'w'),ensure_ascii=False)
    print('ok',sum(s[1]=='OK' for s in stats),'nochart',sum(s[1]=='NOCHART' for s in stats),'err',sum(s[1]=='ERR' for s in stats))
    print('notes',sum(s[3] for s in stats),'valid',sum(s[4] for s in stats))
if __name__=='__main__':main()
