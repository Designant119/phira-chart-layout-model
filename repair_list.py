import json, subprocess, time, os
from concurrent.futures import ThreadPoolExecutor
from phira_corpus import API, api
first=api(API+'?page=1'); total=int(first.get('count',0)); pages=(total+29)//30
print('total',total,'pages',pages,flush=True)
def get(p):
    return p, api(f'{API}?page={p}').get('results',[])
res={}; short=[]
with ThreadPoolExecutor(max_workers=24) as ex:
    for p,rs in ex.map(get, range(1,pages+1)):
        for c in rs: res[c['id']]=c
        if len(rs)<30: short.append((p,rs))
print('got',len(res),'short',[(p,len(r)) for p,r in short],flush=True)
for attempt in range(6):
    need=[p for p,r in short if len(r)<30 and (p<pages or len(r)<19)]
    if not need: break
    print('retry',attempt,need,flush=True)
    with ThreadPoolExecutor(max_workers=12) as ex:
        for p,rs in ex.map(get, need):
            for c in rs: res[c['id']]=c
            if len(rs)>=30 or (p==pages and len(rs)>=19): short=[(q,r) for q,r in short if q!=p]
            else: short=[(q,r) if q!=p else (p,rs) for q,r in short]
print('unique',len(res),'short',[(p,len(r)) for p,r in short],flush=True)
sel=sorted(res.values(),key=lambda c:(float(c.get('rating',0) or 0),float(c.get('ratingCount',0) or 0)),reverse=True)
json.dump({'selected':sel,'candidates':len(sel)},open('/home/user/Doubao/chats/38453392661469954/_run_real_traj/phira_all/index_full.json','w'),ensure_ascii=False,indent=1)
print('saved',len(sel),flush=True)
