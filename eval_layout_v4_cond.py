#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对照实验：非累积逐位置采样 vs 真实。
用真实窗口特征（含真实 prev_x），逐位置从 softmax 采样 dx（不累积回特征、不做随机游走），
统计采样 dx 序列的分布 JS、std、自相关、逐音符 x 误差。"""
import os
import numpy as np
import torch
from train_layout_v4 import GRUCls, BIN_CENTERS, NB

BASE = "/home/user/Doubao/chats/38445055719679234/phira_charts_dl"
X_SCALE = 675.0


def main():
    ck = torch.load(os.path.join(BASE, "layout_ckpt_v4", "model.pth"), map_location="cpu")
    model = GRUCls(); model.load_state_dict(ck["model"]); model.eval()
    d = np.load(os.path.join(BASE, "train_data_v3", "test.npz"))
    X, y = d["X"], d["y"]
    yx = np.cumsum(y, axis=1)  # 真实 x 序列（窗口内累积）

    rng = np.random.RandomState(42)
    n_win = min(400, len(X))
    Xt = torch.from_numpy(X[:n_win].astype(np.float32))
    with torch.no_grad():
        logits = model(Xt)  # (B, W, NB)
        probs = torch.softmax(logits, dim=-1).numpy()

    # 逐位置采样 dx（每个位置独立，用真实特征）
    sampled_dx = np.zeros_like(y[:n_win])
    for b in range(n_win):
        for w in range(X.shape[1]):
            sampled_dx[b, w] = BIN_CENTERS[rng.choice(NB, p=probs[b, w])]

    y_flat = y[:n_win].ravel()
    g_flat = sampled_dx.ravel()

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
    at = autocorr(y[:n_win]); ag = autocorr(sampled_dx)
    print(f"[非累积采样] dx JS={js:.4f} KL={kl:.4f}")
    print(f"[非累积采样] dx std 真实={y_flat.std():.3f} 生成={g_flat.std():.3f}")
    print(f"[非累积采样] 自相关 真实={at:.3f} 生成={ag:.3f}")

    # 逐音符 x 预测误差：用采样 dx 重建 x（但用真实起点 0，窗口内累积）
    # 对比"完美知道分布但不知道具体值"的下界
    # 逐位置 argmax（最可能 dx）
    argmax_dx = BIN_CENTERS[np.argmax(probs, axis=-1)]
    pred_argmax = np.cumsum(argmax_dx, axis=1)
    mse = np.mean((pred_argmax - yx[:n_win]) ** 2)
    rmse = np.sqrt(mse) * X_SCALE
    corr = np.corrcoef(pred_argmax.ravel(), yx[:n_win].ravel())[0, 1]
    print(f"[逐位置 argmax dx 累积] RMSE={rmse:.1f}px corr={corr:.4f}")

    # 条件分布宽度：每个位置 dx 的条件 std（模型概率分布的 std）均值
    cond_std = np.sqrt(np.sum(probs * (BIN_CENTERS[None, None, :] - np.sum(probs * BIN_CENTERS[None, None, :], axis=-1, keepdims=True)) ** 2, axis=-1))
    print(f"[条件分布宽度] 平均条件 dx std={cond_std.mean():.3f} 归一化（≈{cond_std.mean()*X_SCALE:.0f}px），真实 dx std={y_flat.std():.3f}（≈{y_flat.std()*X_SCALE:.0f}px）")
    # 与常数预测比较：预测 dx=0（保持原位）
    pred0 = np.cumsum(np.zeros_like(y[:n_win]), axis=1)
    rmse0 = np.sqrt(np.mean((pred0 - yx[:n_win]) ** 2)) * X_SCALE
    print(f"[基线 dx=0 累积] RMSE={rmse0:.1f}px")


if __name__ == "__main__":
    main()
