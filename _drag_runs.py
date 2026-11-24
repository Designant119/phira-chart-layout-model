import json, glob, numpy as np

PARS = '/home/user/Doubao/chats/38453392661469954/_run_real_traj/phira_all/parsed'


def runs_of_drag(notes):
    notes = sorted(notes, key=lambda n: n['t'])
    from collections import defaultdict
    g = defaultdict(list)
    for n in notes:
        g[round(n['t'], 3)].append(n)
    order = sorted(g)
    runs, cur = [], []
    for t in order:
        if any(n['type'] == 3 for n in g[t]):
            cur.append(t)
        else:
            if cur:
                runs.append(cur); cur = []
    if cur:
        runs.append(cur)
    out = []
    for r in runs:
        pts = [(t, g[t][0]['x'] if g[t] else 0) for t in r]
        out.append(pts)
    return out


best = []
for f in sorted(glob.glob(PARS + '/*.json')):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    rr = runs_of_drag(d['notes'])
    mx = max((len(r) for r in rr), default=0)
    if mx >= 8:
        best.append((mx, f.split('/')[-1][:-5], rr))

best.sort(reverse=True)
print('长drag段谱面数 >=8:', len(best))
for mx, fid, rr in best[:6]:
    longest = max(rr, key=len)
    xs = [p[1] for p in longest]
    ts = [p[0] for p in longest]
    iv = np.diff(ts)
    print()
    print('fid', fid, '最长段', len(longest), '间隔中位', round(float(np.median(iv)), 3))
    print('  x序列:', [int(x) for x in xs[:25]])
