#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v4 评估：softmax 期望累积生成 x。
指标：累积 RMSE、dx 分布 KL/JS（生成 vs 真实）、生成自相关 vs 真实自相关。"""
import json, os
import numpy as np
import torch
from train_layout_v4 import GRUCls, BIN_CENTERS, NB, to_bins

BASE = "/home/user/Doubao/chats/38445055719679234/phira_charts_dl"
X_SCALE = 675.0


def gen_autoreg(model, Xw):
    """逐音符自回归：期望 dx 累积。返回 pred_x (W,)"""
    W = Xw.shape[0]
    Xw = torch.from_numpy(Xw.astype(np.float32)).clone()
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
            dxs[i] = float(np.sum(probs * BIN_CENTERS))
            if i + 1 < W:
                pred_x[i + 1] = pred_x[i] + dxs[i]
    return np.cumsum(dxs), dxs


def main():
    ck = torch.load(os.path.join(BASE, "layout_ckpt_v4", "model.pth"), map_location="cpu")
    model = GRUCls(); model.load_state_dict(ck["model"]); model.eval()
    d = np.load(os.path.join(BASE, "train_data_v3", "test.npz"))
    X, y = d["X"], d["y"]
    yx = np.cumsum(y, axis=1)
    preds = []; dxs_all = []
    for i in range(len(X)):
        p, dxs = gen_autoreg(model, X[i])
        preds.append(p); dxs_all.append(dxs)
    pred = np.array(preds)
    mse = np.mean((pred - yx) ** 2)
    rmse = np.sqrt(mse) * X_SCALE
    r2 = 1 - np.sum((pred - yx) ** 2) / np.sum((yx - yx.mean()) ** 2)
    corr = np.corrcoef(pred.ravel(), yx.ravel())[0, 1]
    print(f"[v4] 自回归期望: RMSE={rmse:.1f}px R2={r2:.4f} corr={corr:.4f}")

    # dx 分布匹配：真实 dx vs 生成 dx
    y_flat = y.ravel()
    g_flat = np.array(dxs_all).ravel()
    def hist_norm(v):
        h, _ = np.histogram(v, bins=np.linspace(-1.2, 1.2, 16), density=True)
        return h + 1e-9
    h_true = hist_norm(y_flat); h_gen = hist_norm(g_flat)
    kl = np.sum(h_true * np.log(h_true / h_gen))
    js = 0.5 * np.sum(h_true * np.log(2 * h_true / (h_true + h_gen)) + h_gen * np.log(2 * h_gen / (h_true + h_gen)))
    print(f"dx 分布: KL={kl:.4f} JS={js:.4f}")
    print(f"真实 dx: std={y_flat.std():.3f} 生成 dx: std={g_flat.std():.3f}")
    # 自相关对比
    def autocorr(v):
        v = v.reshape(-1)
        if len(v) < 5: return np.nan
        return np.corrcoef(v[:-1], v[1:])[0, 1]
    at = autocorr(y[:2000]); ag = autocorr(g_flat[:2000])
    print(f"自相关: 真实={at:.3f} 生成={ag:.3f}")
    np.savez(os.path.join(BASE, "train_data_v3", "test_pred_v4.npz"), pred=pred, yx=yx)


if __name__ == "__main__":
    main()
