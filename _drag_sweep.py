import json, glob, os, sys
import numpy as np
from collections import defaultdict

PARS = '/home/user/Doubao/chats/38453392661469954/_run_real_traj/phira_all/parsed'


def analyze(f):
    d = json.load(open(f))
    notes = sorted(d['notes'], key=lambda n: n['t'])
    g = defaultdict(list)
    for n in notes:
        g[round(n['t'], 3)].append(n)
    order = sorted(g)

    # 连续 drag 段（时间上相邻全部是 drag 拍）
    runs, cur = [], []
    for t in order:
        grp = g[t]
        ds = [n for n in grp if n['type'] == 3]
        if ds and len(ds) == len(grp):   # 该拍全是 drag
            if cur and abs(t - cur[-1][0]) > 1e-6:
                cur.append((t, ds[0]['x'], ds[0]['line_id']))
            elif not cur:
                cur.append((t, ds[0]['x'], ds[0]['line_id']))
        else:
            if cur:
                runs.append(cur); cur = []
    if cur:
        runs.append(cur)

    res = {'n_runs_ge4': 0, 'n_runs_ge8': 0, 'sweep_mono': [], 'sweep_cv': [],
           'sweep_flip': [], 'sweep_single_line': [], 'sweep_len': []}
    for r in runs:
        if len(r) < 4:
            continue
        res['n_runs_ge4'] += 1
        if len(r) >= 8:
            res['n_runs_ge8'] += 1
        ps = [p for p in r if p[1] != 0]
        if len(ps) < 4:
            continue
        xs = np.array([p[1] for p in ps], dtype=float)
        ts = np.array([p[0] for p in ps], dtype=float)
        dx = np.diff(xs)
        nz = dx[np.abs(dx) > 1e-6]
        if len(nz) < 3:
            continue
        # 步长一致性：|dx| 的 CV（越小越规整）
        adx = np.abs(nz)
        res['sweep_cv'].append(float(adx.std() / adx.mean()) if adx.mean() > 0 else 9.9)
        # 同向率（三角波往复会翻转，单调斜线不翻转）
        s = np.sign(nz)
        res['sweep_mono'].append(float((s[:-1] * s[1:] > 0).mean()))
        # 方向翻转次数比例
        res['sweep_flip'].append(float((s[:-1] * s[1:] < 0).mean()))
        lids = set(p[2] for p in ps)
        res['sweep_single_line'].append(1.0 if len(lids) == 1 else 0.0)
        res['sweep_len'].append(len(ps))
    return res


agg = defaultdict(list)
tot = {'n_runs_ge4': 0, 'n_runs_ge8': 0}
songs_ge8 = 0
songs_sweep = 0
done = 0
for f in sorted(glob.glob(PARS + '/*.json')):
    try:
        r = analyze(f)
    except Exception:
        continue
    done += 1
    tot['n_runs_ge4'] += r['n_runs_ge4']
    tot['n_runs_ge8'] += r['n_runs_ge8']
    if r['n_runs_ge8'] > 0:
        songs_ge8 += 1
    if len(r['sweep_cv']) > 0:
        songs_sweep += 1
    for k in ['sweep_mono', 'sweep_cv', 'sweep_flip', 'sweep_single_line', 'sweep_len']:
        agg[k].extend(r[k])

print('谱面', done)
print('长drag段(len>=4) 总数', tot['n_runs_ge4'], '其中>=8 的', tot['n_runs_ge8'])
print('含>=8长段的谱面数', songs_ge8, '(%.1f%%)' % (100 * songs_ge8 / done))
print()
for k in ['sweep_cv', 'sweep_mono', 'sweep_flip', 'sweep_single_line', 'sweep_len']:
    a = np.array(agg[k], dtype=float)
    if len(a) == 0:
        continue
    print('%-18s 中位 %.3f  25%% %.3f  75%% %.3f' % (k, np.median(a), np.percentile(a, 25), np.percentile(a, 75)))
print()
cv = np.array(agg['sweep_cv'], dtype=float)
print('步长规整(CV<0.15)的段占比 %.1f%%' % (100 * (cv < 0.15).mean()))
mono = np.array(agg['sweep_mono'], dtype=float)
print('单调同向(>0.8)的段占比 %.1f%%' % (100 * (mono > 0.8).mean()))
print('往复扫(同向<0.3, 即频繁掉头)的段占比 %.1f%%' % (100 * (mono < 0.3).mean()))
