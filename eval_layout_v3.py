#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v2 自回归推理评估：逐时间步生成 dx 并累积 x（prev_x 特征用模型自身预测）。
输出：test RMSE / R2 / corr + 逐歌样例预测（供可视化）。"""
import json, os, sys
import numpy as np
import torch
from train_layout_v3 import GRU

BASE = "/home/user/Doubao/chats/38445055719679234/phira_charts_dl"
X_SCALE = 675.0


def autoregressive_predict(model, Xw, use_teacher=False):
    """Xw: (W,7) 窗口特征（prev_x 列占位）。返回累积 pred_x (W,) 归一化。
    use_teacher=True 时用真实 prev_x（对照用）。"""
    W = Xw.shape[0]
    Xw = torch.from_numpy(Xw.astype(np.float32)).clone()
    if use_teacher:
        with torch.no_grad():
            o = model(Xw.unsqueeze(0))
        dx = o[0].numpy()
        return dx
    pred_x = np.zeros(W, dtype=np.float32)
    h = None
    dxs = np.zeros(W, dtype=np.float32)
    with torch.no_grad():
        for i in range(W):
            Xw[i, 6] = float(pred_x[i])  # prev_x = 上一步预测
            xi = Xw[i].unsqueeze(0).unsqueeze(0)  # (1,1,7)
            out, h = model.gru(xi, h)
            dxs[i] = model.head(out).item()
            if i + 1 < W:
                pred_x[i + 1] = pred_x[i] + dxs[i]
    return np.cumsum(dxs)  # 等价


def main():
    ck = torch.load(os.path.join(BASE, "layout_ckpt_v3", "model.pth"), map_location="cpu")
    st = json.load(open(os.path.join(BASE, "layout_ckpt_v3", "ckpt_state.json")))
    model = GRU(feat=7, hidden=64)
    model.load_state_dict(ck["model"])
    model.eval()

    for split in ["test"]:
        d = np.load(os.path.join(BASE, "train_data_v2", f"{split}.npz"))
        X, y = d["X"], d["y"]
        # y 是 dx；累积真实 x
        yx = np.cumsum(y, axis=1)
        preds = []
        for i in range(len(X)):
            preds.append(autoregressive_predict(model, X[i]))
        pred = np.array(preds)
        # 指标
        mse = np.mean((pred - yx) ** 2)
        rmse = np.sqrt(mse) * X_SCALE
        r2 = 1 - np.sum((pred - yx) ** 2) / np.sum((yx - yx.mean()) ** 2)
        corr = np.corrcoef(pred.ravel(), yx.ravel())[0, 1]
        print(f"[{split}] 自回归: RMSE={rmse:.1f}px  R2={r2:.4f}  corr={corr:.4f}")
        # teacher forcing 对照（前 200 窗口）
        tf = np.array([autoregressive_predict(model, X[i], use_teacher=True) for i in range(200)])
        tf_x = np.cumsum(tf, axis=1)
        mse_t = np.mean((tf_x - yx[:200]) ** 2)
        print(f"[{split}] teacher-forcing: RMSE={np.sqrt(mse_t)*X_SCALE:.1f}px")
        np.savez(os.path.join(BASE, "train_data_v2", f"{split}_pred.npz"),
                 pred=pred, yx=yx, X=X)
    # 存样例（前 6 窗口）供可视化
    print("样例已存", flush=True)


if __name__ == "__main__":
    main()
