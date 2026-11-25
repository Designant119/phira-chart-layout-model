"""序列级最小风险微调（MRT）。

原 train_ar.py 的目标是逐 token 交叉熵，最优解是"每步取条件众数"，
连起来的轨迹不一定像真谱：真谱里"连着几颗站同一侧"是成段出现的，
逐 token 独立采样会把这种段落概率压成 0.22^k，模型于是主动避开站住。

这里改成对整段轨迹打分：
  1. 用当前模型对每段采样 G 条轨迹（温度 1.0）
  2. 每条轨迹算结构距离 R（步长中位、同列率、左右交替、中间带占比 四项归一化绝对偏差）
  3. 用 softmax(-R/tau) 当目标分布，最小化 KL(target || model)
这样"那段看起来像不像真谱"直接进梯度，而不是靠逐音似然间接带出来。
"""
import os
import glob
import random
import argparse

import numpy as np
import torch
import torch.nn as nn

import train_ar as T

# 真谱基准（1500 首口径）
REF = {
    'step': 253.0,      # 相邻 |dx| 中位
    'same': 0.220,      # |dx| < 135 占比
    'alt': 0.694,       # 左右交替率
    'mid': 0.094,       # |x| < 80 占比
}
# 真谱 9 桶频率（-675..675），用于约束整体形状，防止模型卡在一侧刷同列率
REF_H = np.array([0.002, 0.103, 0.208, 0.140, 0.093, 0.142, 0.215, 0.097, 0.000])
W = {'step': 1.2, 'same': 2.6, 'alt': 0.8, 'mid': 0.6, 'hist': 1.1}


def risk_of(xs):
    """xs: 一段轨迹的 x（坐标），返回结构距离（越小越像真谱）"""
    xs = np.asarray(xs, dtype=np.float64)
    if len(xs) < 4:
        return 9.9
    d = np.abs(np.diff(xs))
    step = float(np.median(d))
    same = float((d < 135).mean())
    sgn = np.sign(np.diff(xs))
    sgn = sgn[sgn != 0]
    alt = float((np.diff(sgn) != 0).mean()) if len(sgn) > 1 else 0.0
    mid = float((np.abs(xs) < 80).mean())
    r = 0.0
    r += W['step'] * abs(step - REF['step']) / REF['step']
    r += W['same'] * abs(same - REF['same']) / REF['same']
    r += W['alt'] * abs(alt - REF['alt']) / REF['alt']
    r += W['mid'] * abs(mid - REF['mid']) / REF['mid']
    h, _ = np.histogram(xs, bins=9, range=(-675, 675))
    ph = h / max(1, h.sum())
    r += W['hist'] * float(np.abs(ph - REF_H).sum())
    return float(r)


def sample_traj(model, f, g, temp=1.0, rng=None):
    """f:(T,DIM) g:(T,) 真值格；返回 (轨迹 x 列表, 该轨迹的 log 概率)"""
    ft = torch.from_numpy(f[None])
    prev = T.NB
    st = None
    xs, lps = [], 0.0
    with torch.no_grad():
        for i in range(len(f)):
            lg, st = model(ft[:, i:i + 1], torch.tensor([[prev]]), st)
            lp = torch.log_softmax(lg[0, 0] / temp, -1).numpy()
            p = np.exp(lp)
            b = int(rng.choice(T.NB, p=p / p.sum()))
            lps += float(lp[b])
            prev = b
            xs.append(T.g2x(b))
    return xs, lps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pars', default='/home/user/Doubao/chats/38453392661469954/_run_real_traj/abs_full/parsed')
    ap.add_argument('--limit', type=int, default=1500)
    ap.add_argument('--ckpt', default='arnet_j.pt')
    ap.add_argument('--out', default='arnet_mrt3.pt')
    ap.add_argument('--steps', type=int, default=900)
    ap.add_argument('--G', type=int, default=4)
    ap.add_argument('--tau', type=float, default=0.45)
    ap.add_argument('--seg', type=int, default=16)
    a = ap.parse_args()

    files = sorted(glob.glob(a.pars + '/*.json'))
    random.seed(0); random.shuffle(files)
    files = files[:a.limit] if a.limit > 0 else files
    songs = T.load_data(files)
    print('曲目', len(songs), flush=True)

    model = T.ARNet()
    model.load_state_dict(torch.load(a.ckpt, map_location='cpu'))
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    rng = np.random.default_rng(0)

    n = len(songs)
    tridx = list(range(0, int(n * 0.95)))
    for step in range(a.steps):
        i = tridx[rng.integers(0, len(tridx))]
        f, y, _ = songs[i]
        L = min(a.seg, len(f))
        st0 = int(rng.integers(0, len(f) - L + 1)) if len(f) > L else 0
        fs = f[st0:st0 + L]

        # 采样 G 条 + 记录每条的对数概率
        trajs, lps = [], []
        model.eval()
        for g_ in range(a.G):
            xs, lp = sample_traj(model, fs, None, 1.0, rng)
            trajs.append(xs); lps.append(lp)
        model.train()
        R = np.array([risk_of(t) for t in trajs])
        wgt = np.exp(-(R - R.min()) / a.tau)
        wgt = wgt / wgt.sum()

        # 只用一次前向（teacher forcing 并行）算所有轨迹的 NLL，再按 wgt 加权
        keep = [k for k in range(a.G) if wgt[k] >= 0.02]
        if not keep:
            continue
        B = len(keep)
        FB = np.repeat(fs[None], B, 0)
        PB = np.full((B, L), T.NB, np.int64)
        YB = np.zeros((B, L), np.int64)
        for r, k in enumerate(keep):
            gb = [T.x2g(v) for v in trajs[k]]
            YB[r] = gb
            PB[r, 1:] = gb[:-1]
        lg, _ = model(torch.from_numpy(FB), torch.from_numpy(PB))
        ce = nn.functional.cross_entropy(lg.reshape(-1, T.NB),
                                         torch.from_numpy(YB).reshape(-1), reduction='none')
        ce = ce.reshape(B, L).mean(1)
        loss = (torch.from_numpy(wgt[keep] / wgt[keep].sum()).float() * ce).sum()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0:
            print('  step %d risk %.3f/%.3f 同列 %.2f 步长 %.0f  loss %.4f'
                  % (step, R.min(), R.mean(),
                     np.mean([(np.abs(np.diff(t)) < 135).mean() for t in trajs]),
                     np.mean([np.median(np.abs(np.diff(t))) for t in trajs]),
                     loss), flush=True)
        if step and step % 400 == 0:
            torch.save(model.state_dict(), a.out)
            print('  存 %s' % a.out, flush=True)
    torch.save(model.state_dict(), a.out)
    print('完成 %s' % a.out, flush=True)


if __name__ == '__main__':
    main()
