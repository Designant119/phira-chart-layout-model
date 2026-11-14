#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""音频 -> 每帧 15 维特征（12 色度 + 对数能量 + 谱通量 + 谱质心），缓存 npz。
输出 data/audio_feat/<fid>.npz: times(F,) / feats(F,15) float16。
"""
import os, subprocess, glob, json, sys, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

DATA = "/home/user/Doubao/chats/38453392661469954/_run_real_traj/data"
AUD = os.path.join(DATA, "audio")
OUT = os.path.join(DATA, "audio_feat")
os.makedirs(OUT, exist_ok=True)
SR, WIN, HOP = 22050, 2048, 512


def decode(path):
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"]
    out = subprocess.run(cmd, capture_output=True, timeout=300)
    if out.returncode != 0 or len(out.stdout) < SR * 4:
        return None
    return np.frombuffer(out.stdout, dtype=np.float32)


def stft_feats(y):
    n = 1 + max(0, (len(y) - WIN) // HOP)
    win = np.hanning(WIN).astype(np.float32)
    freqs = np.fft.rfftfreq(WIN, 1.0 / SR).astype(np.float32)
    # 色度折叠：只取 40~2000Hz，A440
    band = (freqs >= 40) & (freqs <= 2000)
    f_band = freqs[band]
    pc = np.round(12 * np.log2(np.maximum(f_band, 1e-6) / 440.0) + 69).astype(np.int32) % 12
    feats = np.zeros((n, 15), dtype=np.float32)
    prev = None
    for i0 in range(0, n, 2048):
        i1 = min(n, i0 + 2048)
        idx = np.arange(i0, i1)
        frames = np.stack([y[k * HOP:k * HOP + WIN] for k in idx])
        spec = np.abs(np.fft.rfft(frames * win, axis=1)) + 1e-8  # (B,1025)
        chroma = np.zeros((len(idx), 12), dtype=np.float32)
        sb = spec[:, band]
        for c in range(12):
            chroma[:, c] = sb[:, pc == c].sum(axis=1)
        chroma /= (chroma.sum(axis=1, keepdims=True) + 1e-8)
        energy = np.log1p(np.sqrt((frames ** 2).mean(axis=1)) * 100.0)
        centroid = (spec * freqs).sum(axis=1) / spec.sum(axis=1) / (SR / 2)
        d = np.diff(spec, axis=0, prepend=spec[:1])
        flux = np.maximum(d, 0).sum(axis=1) / spec.sum(axis=1)
        feats[i0:i1, :12] = chroma
        feats[i0:i1, 12] = energy
        feats[i0:i1, 13] = flux
        feats[i0:i1, 14] = centroid
    # 能量/通量按歌曲中位与分位数归一，抵消录音音量差
    feats[:, 12] = (feats[:, 12] - np.median(feats[:, 12])) / (np.std(feats[:, 12]) + 1e-6)
    q = np.percentile(feats[:, 13], 99) + 1e-6
    feats[:, 13] = np.clip(feats[:, 13] / q, 0, 3)
    return np.arange(n, dtype=np.float32) * HOP / SR, feats.astype(np.float16)


def one(fid):
    out = os.path.join(OUT, fid + ".npz")
    if os.path.exists(out):
        return "skip"
    ap = glob.glob(os.path.join(AUD, fid + ".*"))
    if not ap:
        return "noaudio"
    y = decode(ap[0])
    if y is None:
        return "decodefail"
    times, feats = stft_feats(y)
    np.savez(out, times=times, feats=feats)
    return "ok"


if __name__ == "__main__":
    fids = sorted(os.path.basename(p).rsplit(".", 1)[0] for p in glob.glob(os.path.join(AUD, "*")))
    print("fids", len(fids), flush=True)
    res = {}
    with ProcessPoolExecutor(max_workers=2) as ex:
        for i, (fid, st) in enumerate(zip(fids, ex.map(one, fids))):
            res[fid] = st
            if (i + 1) % 10 == 0:
                print(i + 1, st, flush=True)
    print({k: list(res.values()).count(k) for k in set(res.values())}, flush=True)
