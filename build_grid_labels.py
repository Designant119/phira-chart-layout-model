#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从并行窗口数据提取窗级网格标签：间距 gap、列数 ncol、中心 center、跨度 span。
同时做窗级特征聚合，存成网格参数模型用的 npz。"""
import os, json, argparse
import numpy as np

KEEP = [0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13]  # 纯节奏，去掉 prev/prev2 与组内 x 衍生特征
NB = 81
GAP_BINS = [3, 4, 5, 6, 7]
NCOL_BINS = [8, 9, 10, 11, 12, 13, 14]


def bin_idx(v, bins):
    v = float(v)
    return int(np.argmin([abs(v - b) for b in bins]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--split', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    d = np.load(a.data, allow_pickle=True)
    F = np.ascontiguousarray(d['F'][:, :, KEEP], dtype=np.float32)
    Y = d['Y'].astype(np.float32)
    M = d['M'].astype(np.float32)
    G = d['G'].astype(np.int64)
    fid = d['fid'].astype(str)
    N = len(F)

    gap_l, ncol_l, center_l, span_l = [], [], [], []
    Fwin = np.zeros((N, F.shape[-1] + 2), np.float32)

    for i in range(N):
        m = M[i] > 0.5
        y = Y[i][m]
        f = F[i][m]
        if len(y) < 4:
            gap_l.append(4); ncol_l.append(11); center_l.append(0.0); span_l.append(0)
            Fwin[i, :F.shape[-1]] = f.mean(0) if len(f) else 0
            continue
        cell = np.clip(np.round((y + 1.0) / 2.0 * (NB - 1)), 0, NB - 1).astype(int)
        cs = np.unique(cell)
        if len(cs) >= 2:
            dif = np.diff(cs)
            gap = float(np.median(dif))
        else:
            gap = 4.0
        ncol = len(cs)
        center = float(y.mean())
        span = int(cs[-1] - cs[0])
        gap_l.append(gap)
        ncol_l.append(ncol)
        center_l.append(center)
        span_l.append(span)
        # 窗级聚合特征
        Fwin[i, :F.shape[-1]] = f.mean(0)
        Fwin[i, F.shape[-1] + 0] = len(y) / 128.0          # 有效音占比
        # 多押占比：同组大于1的音
        gm = G[i][m]
        gcnt = np.bincount(gm)
        multi = (gcnt[gm] > 1).mean()
        Fwin[i, F.shape[-1] + 1] = multi

    gap_l = np.array(gap_l); ncol_l = np.array(ncol_l)
    center_l = np.array(center_l); span_l = np.array(span_l)
    gap_bin = np.array([bin_idx(v, GAP_BINS) for v in gap_l])
    ncol_bin = np.array([bin_idx(v, NCOL_BINS) for v in ncol_l])

    print('窗 %d' % N)
    print('gap 分布(原始格) 中位 %.1f 分位 %s' % (np.median(gap_l), np.percentile(gap_l, [25, 75]).round(1)))
    print('ncol 分布 中位 %.0f 分位 %s' % (np.median(ncol_l), np.percentile(ncol_l, [25, 75]).round(1)))
    print('center 均值 %.3f 标准差 %.3f' % (center_l.mean(), center_l.std()))
    print('gap 分箱', {GAP_BINS[k]: int((gap_bin == k).sum()) for k in range(len(GAP_BINS))})
    print('ncol 分箱', {NCOL_BINS[k]: int((ncol_bin == k).sum()) for k in range(len(NCOL_BINS))})

    split = json.load(open(a.split))
    val, test = set(split['val']), set(split['test'])
    va = np.isin(fid, list(val)); te = np.isin(fid, list(test))
    print('train %d val %d test %d' % ((~(va | te)).sum(), va.sum(), te.sum()))

    np.savez_compressed(a.out, F=Fwin, gap=gap_bin, ncol=ncol_bin, center=center_l,
                        gap_raw=gap_l, ncol_raw=ncol_l, span=span_l, fid=fid,
                        gap_bins=np.array(GAP_BINS), ncol_bins=np.array(NCOL_BINS))
    print('saved', a.out, 'feat', Fwin.shape[-1])


if __name__ == '__main__':
    main()
