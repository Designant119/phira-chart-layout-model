#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""整曲序列模型（双向 LSTM）：一首歌一条完整序列喂进去，全曲双向上下文逐音出位置分类。
不切窗口；按长度分桶组 batch，动态 padding。CPU 上 LSTM 是 O(L)，比 self-attention 快。
数据：song_data_clean.npz（扁平 F/Y + offs/lengths/split）。"""
import os, json, time, argparse
import numpy as np, torch, torch.nn as nn

NB = 81


def to_bin(y):
    return np.clip(np.round((y + 1.0) / 2.0 * (NB - 1)).astype(np.int64), 0, NB - 1)


class SongNet(nn.Module):
    def __init__(self, nf, hidden=96, layers=2, nb=NB):
        super().__init__()
        self.nb = nb
        self.inp = nn.Linear(nf, hidden)
        self.lstm = nn.LSTM(hidden, hidden, layers, batch_first=True,
                            bidirectional=True, dropout=.1)
        self.out = nn.Linear(2 * hidden, nb)

    def forward(self, f, lens):
        h = self.inp(f)
        h = nn.utils.rnn.pack_padded_sequence(h, lens, batch_first=True, enforce_sorted=False)
        h, _ = self.lstm(h)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True)
        return self.out(h)


def bucket_batches(lengths, split_idx, max_tokens=2048, max_batch=8, seed=0):
    idx = list(split_idx)
    idx.sort(key=lambda i: lengths[i])
    batches, cur, cur_len = [], [], 0
    for i in idx:
        L = lengths[i]
        if cur and (cur_len + L > max_tokens or len(cur) >= max_batch):
            batches.append(cur); cur, cur_len = [], 0
        cur.append(i); cur_len += L
    if cur:
        batches.append(cur)
    rng = np.random.default_rng(seed)
    rng.shuffle(batches)
    return batches


def load_batch(data, batch_idx, maxlen):
    offs = data['offs']; F = data['F']; Y = data['Y']
    Lmax = min(maxlen, max(offs[i + 1] - offs[i] for i in batch_idx))
    B = len(batch_idx)
    f = np.zeros((B, Lmax, F.shape[1]), np.float32)
    y = np.zeros((B, Lmax), np.int64)
    mask = np.zeros((B, Lmax), bool)
    lens = np.zeros(B, np.int64)
    for b, i in enumerate(batch_idx):
        s, e = offs[i], offs[i + 1]
        n = min(e - s, Lmax)
        f[b, :n] = F[s:s + n]
        y[b, :n] = to_bin(Y[s:s + n])
        mask[b, :n] = True
        lens[b] = n
    return (torch.from_numpy(f), torch.from_numpy(y), torch.from_numpy(mask),
            torch.from_numpy(lens))


def eval_net(net, data, idx, maxlen, device, bs=4):
    net.eval(); gok = tot = 0; mae = []
    with torch.no_grad():
        for k in range(0, len(idx), bs):
            bi = idx[k:k + bs]
            f, y, m, lens = load_batch(data, bi, maxlen)
            lg = net(f.to(device), lens.to(device))
            pred = lg.argmax(-1).cpu().numpy()
            yv = y.numpy(); mv = m.numpy()
            gok += (pred[mv] == yv[mv]).sum(); tot += mv.sum()
            px = pred / (NB - 1) * 2 - 1
            yt = yv[mv] / (NB - 1) * 2 - 1
            mae.append(np.abs(px[mv] - yt).mean() * 675)
    return gok / tot, float(np.mean(mae))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=24)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--maxlen', type=int, default=1536)
    ap.add_argument('--max-tokens', type=int, default=2048)
    ap.add_argument('--max-batch', type=int, default=8)
    ap.add_argument('--hidden', type=int, default=96)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--max-train', type=int, default=0)
    a = ap.parse_args()

    device = 'cpu'
    torch.set_num_threads(2)
    data = np.load(a.data, allow_pickle=True)
    lengths = data['lengths']; split = data['split']
    tr = np.where(split == 0)[0]; va = np.where(split == 1)[0]; te = np.where(split == 2)[0]
    if a.max_train:
        rng0 = np.random.default_rng(0)
        tr = rng0.choice(tr, min(a.max_train, len(tr)), replace=False)
    print('train %d val %d test %d 中位长度 %d' % (len(tr), len(va), len(te),
          int(np.median(lengths))), flush=True)

    net = SongNet(nf=data['F'].shape[1], hidden=a.hidden, layers=a.layers)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-5)
    ce = nn.CrossEntropyLoss(reduction='none')
    best = 1e9
    rpath = a.out[:-3] + '_last.pt'

    for ep in range(a.epochs):
        t0 = time.time(); net.train()
        batches = bucket_batches(lengths, tr, a.max_tokens, a.max_batch, seed=ep)
        tot_loss = tot_n = 0; t_ep = time.time()
        for bi_i, bi in enumerate(batches):
            f, y, m, lens = load_batch(data, bi, a.maxlen)
            f, y, m, lens = f.to(device), y.to(device), m.to(device), lens.to(device)
            opt.zero_grad()
            lg = net(f, lens)
            loss = (ce(lg.reshape(-1, NB), y.reshape(-1)).reshape(y.shape) * m).sum() / m.sum()
            loss.backward(); opt.step()
            tot_loss += float(loss.detach()) * int(m.sum()); tot_n += int(m.sum())
            if bi_i % 50 == 0:
                print('  batch %d/%d  %.0fs  loss %.3f' % (bi_i, len(batches),
                      time.time() - t_ep, float(loss)), flush=True)
        vacc, vmae = eval_net(net, data, va, a.maxlen, device)
        tacc, tmae = eval_net(net, data, te, a.maxlen, device)
        torch.save({'model': net.state_dict(), 'epoch': ep + 1, 'val': vmae}, rpath)
        tag = ''
        if vmae < best:
            best = vmae
            torch.save({'model': net.state_dict(), 'nf': data['F'].shape[1],
                        'hidden': a.hidden, 'layers': a.layers}, a.out)
            tag = ' *saved'
        print('ep%d ce %.4f | val acc %.1f%% MAE %.1fpx | test acc %.1f%% MAE %.1fpx | %ds%s' % (
            ep + 1, tot_loss / tot_n, 100 * vacc, vmae, 100 * tacc, tmae,
            time.time() - t0, tag), flush=True)

    base = float(np.mean(np.abs(data['Y'])) * 675)
    print('baseline(全猜0) %.1fpx' % base, flush=True)


if __name__ == '__main__':
    main()
