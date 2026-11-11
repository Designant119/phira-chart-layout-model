#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v4 采样推理评估：逐音符从 softmax 采样 dx 累积生成 x。
对比：期望累积 vs 采样累积；dx 分布 JS；自相关。"""
import os
import numpy as np
import torch
from train_layout_v4 import GRUCls, BIN_CENTERS, NB, to_bins

BASE = "/home/user/Doubao/chats/38445055719679234/phira_charts_dl"
X_SCALE = 675.0


def gen_autoreg(model, Xw, mode="sample", seed=0):
    """逐音符自回归。mode=sample 从 softmax 采样 dx；mode=expect 用期望 dx。返回 (x_cum, dxs)"""
    W = Xw.shape[0]
    Xw = torch.from_numpy(Xw.astype(np.float32)).clone()
    rng = np.random.RandomState(seed)
    pred_x = np.zeros(W, dtype=np.float32)
    h = None
    dxs = np.zeros(W, dtype=np.float32)
    with torch.no_grad():
        for i in range(W):
            Xw[i, 6] = float(pred_x[i])
            xi = Xw[i].unsqueeze(0).unsqueeze(0)
            out, h = model.gru(xi, h)
            logits = model.head(out)[0, 0]
            probs = torch.softmax(logits, dim=0).numpy()
            if mode == "sample":
                dxs[i] = float(BIN_CENTERS[rng.choice(NB, p=probs)])
            else:
                dxs[i] = float(np.sum(probs * BIN_CENTERS))
    return np.cumsum(dxs), dxs


def main():
    ck = torch.load(os.path.join(BASE, "layout_ckpt_v4", "model.pth"), map_location="cpu")
    model = GRUCls(); model.load_state_dict(ck["model"]); model.eval()
    d = np.load(os.path.join(BASE, "train_data_v3", "test.npz"))
    X, y = d["X"], d["y"]
    yx = np.cumsum(y, axis=1)

    for mode in ["sample", "expect"]:
        # 每个窗口独立采样，取前 400 个窗口
        n_win = min(400, len(X))
        preds = []; dxs_all = []
        for i in range(n_win):
            p, dxs = gen_autoreg(model, X[i], mode=mode, seed=1000 + i)
            preds.append(p); dxs_all.append(dxs)
        pred = np.array(preds)
        mse = np.mean((pred - yx[:n_win]) ** 2)
        rmse = np.sqrt(mse) * X_SCALE
        r2 = 1 - np.sum((pred - yx[:n_win]) ** 2) / np.sum((yx[:n_win] - yx[:n_win].mean()) ** 2)
        corr = np.corrcoef(pred.ravel(), yx[:n_win].ravel())[0, 1]

        y_flat = y[:n_win].ravel()
        g_flat = np.array(dxs_all).ravel()
        def hist_norm(v):
            h, _ = np.histogram(v, bins=np.linspace(-1.2, 1.2, 16), density=True)
            return h + 1e-9
        h_true = hist_norm(y_flat); h_gen = hist_norm(g_flat)
        kl = np.sum(h_true * np.log(h_true / h_gen))
        js = 0.5 * np.sum(h_true * np.log(2 * h_true / (h_true + h_gen)) + h_gen * np.log(2 * h_gen / (h_true + h_gen)))
        def autocorr(v):
            v = v.reshape(-1)
            if len(v) < 5: return np.nan
            return np.corrcoef(v[:-1], v[1:])[0, 1]
        at = autocorr(y[:n_win]); ag = autocorr(g_flat)
        print(f"[v4 {mode}] RMSE={rmse:.1f}px R2={r2:.4f} corr={corr:.4f} | dx JS={js:.4f} KL={kl:.4f} | dx std 真实={y_flat.std():.3f} 生成={g_flat.std():.3f} | 自相关 真实={at:.3f} 生成={ag:.3f}")

    # 越界率：|x|>675 即 >1.0 归一化
    for mode in ["sample", "expect"]:
        n_win = min(400, len(X))
        preds = []
        for i in range(n_win):
            p, _ = gen_autoreg(model, X[i], mode=mode, seed=1000 + i)
            preds.append(p)
        pred = np.array(preds)
        oob = np.mean(np.abs(pred) > 1.0)
        real_oob = np.mean(np.abs(yx[:n_win]) > 1.0)
        print(f"[v4 {mode}] 越界率 |x|>675px: 生成={oob*100:.1f}% 真实={real_oob*100:.1f}%")


if __name__ == "__main__":
    main()
