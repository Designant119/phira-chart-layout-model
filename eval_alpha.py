import json,glob,numpy as np
from eval_motif_full import build_targets, gather_candidates, score_one
root='/home/user/Doubao/chats/38453392661469954/_run_real_traj'
parts=sorted(glob.glob(root+'/motif_parts/part_*.npz'))
split=json.load(open(root+'/train_data_traj_fixed/split.json'))
rng=np.random.default_rng(0)
val=build_targets(root,'val',split['val'],rng,100); test=build_targets(root,'test',split['test'],rng,140)
W=(1.0,0,8,0,0.3)
def ev(rows,alpha,topn=1):
 es=[]; abses=[]; wins=0; bases=[]
 for c in rows:
  t=c['target']; sc=score_one(c,W)
  if topn==1: j=int(np.argmax(sc)); p=c['pred'][j]
  else:
   js=np.argsort(-sc)[:topn]; p=np.mean([c['pred'][j] for j in js],axis=0)
  p=t['ctx_med']+alpha*(p-t['ctx_med'])
  es.append(np.mean(np.abs(p-t['x']))); bases.append(np.mean(np.abs(np.full(12,t['ctx_med'])-t['x']))); wins+=es[-1]<bases[-1]
 return round(float(np.mean(es))*675,1),round(float(np.mean(bases))*675,1),round(wins/len(es),3)
for name,tg in [('val',val),('test',test)]:
 rows=gather_candidates(parts,tg,ctxw=2)
 print(name,'baseline',ev(rows,0),flush=True)
 for a in [0.05,0.1,0.15,0.2,0.3,0.5,0.7,1.0]:
  print(name,'alpha',a,'top1',ev(rows,a,1),'top3',ev(rows,a,3),flush=True)
