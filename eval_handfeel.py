#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""手感评估：同一段音符，左画真谱 x，右画模型生成的 x，一眼比排键逻辑。"""
import os, json, argparse
import numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
from train_singleline import SeqNet
from train_delta_cls import DeltaNet, ar_eval as delta_ar
from train_parallel_cls import ParNet, parallel_eval

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"
COLORS = {1: (134, 233, 247), 2: (200, 242, 247), 3: (255, 213, 79), 4: (255, 127, 168)}


def font(sz):
    return ImageFont.truetype(FONT, sz) if os.path.exists(FONT) else ImageFont.load_default()


def gen(net, F):
    net.eval()
    f = torch.from_numpy(F.astype(np.float32).copy())
    y = torch.zeros(f.shape[0], 128)
    with torch.no_grad():
        for k in range(128):
            if k >= 1:
                y[:, k] = y[:, k - 1]
            f[:, k, 8] = y[:, k - 1] if k >= 1 else 0.0
            f[:, k, 9] = y[:, k - 2] if k >= 2 else 0.0
            f[:, k, 10] = f[:, k, 8] - f[:, k, 9]
            y[:, k] = net(f)[:, k]
    return y.numpy()


def panel(notes_types, xs, y0, h, w, label, draw, top, x_off=0):
    """画一条竖版音轨，xs 为归一化到 ±1 的相对位置"""
    cx = x_off + w // 2
    for i, (typ, xv) in enumerate(zip(notes_types, xs)):
        yy = top + i * (h / 128)
        px = cx + float(xv) * (w / 2 - 14)
        c = COLORS.get(int(typ), (200, 200, 200))
        if int(typ) == 2:
            draw.rounded_rectangle((px - 9, yy - 4, px + 9, yy + 4), radius=3, fill=c, outline=(255, 226, 132))
        elif int(typ) == 1:
            draw.rounded_rectangle((px - 11, yy - 2, px + 11, yy + 2), radius=2, fill=c)
        elif int(typ) == 3:
            draw.rounded_rectangle((px - 9, yy - 2, px + 9, yy + 2), radius=2, fill=c)
        else:
            draw.rounded_rectangle((px - 7, yy - 2, px + 7, yy + 2), radius=2, fill=c)
    draw.text((x_off + 10, y0 + 6), label, fill=(220, 220, 220), font=font(18))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ids", default="")
    ap.add_argument("--out", default="handfeel_compare.png")
    ap.add_argument("--arch", default="reg")
    a = ap.parse_args()

    d = np.load(a.data, allow_pickle=True)
    F, Y, M, fid = d["F"], d["Y"], d["M"], d["fid"].astype(str)
    if a.arch == "delta":
        ck = torch.load(a.model, map_location="cpu")
        net = DeltaNet(nb=ck.get("bins", 81))
        net.load_state_dict(ck["model"])
    elif a.arch == "parallel":
        ck = torch.load(a.model, map_location="cpu")
        net = ParNet(nf=ck.get("feat", 11), nb=ck.get("bins", 81))
        net.load_state_dict(ck["model"])
    else:
        net = SeqNet()
        net.load_state_dict(torch.load(a.model, map_location="cpu")["model"])

    want = [s for s in a.ids.split(",") if s]
    rows = []
    if want:
        for w in want:
            idx = np.where(fid == w)[0]
            if len(idx):
                mid = idx[len(idx) // 2]
                rows.append(int(mid))
    else:
        # 无指定时按谱分散抽几段
        uniq = list(dict.fromkeys(fid.tolist()))
        step = max(1, len(uniq) // 6)
        for u in uniq[::step][:6]:
            idx = np.where(fid == u)[0]
            if len(idx):
                rows.append(int(idx[len(idx) // 2]))

    if a.arch == "delta":
        pred = delta_ar(net, F[rows], Y[rows], M[rows], net.nb)[1]
    elif a.arch == "parallel":
        Fp = np.concatenate([F[rows][:, :, :8], F[rows][:, :, 11:]], axis=-1)
        pred = parallel_eval(net, Fp, Y[rows], M[rows], net.nb)[1]
    else:
        pred = gen(net, F[rows])
    types = F[rows][:, :, 1:5].argmax(-1) + 1

    W, H = 460, 300
    gap = 16
    img = Image.new("RGB", (W * 2 + gap * 3, H * len(rows) + gap * (len(rows) + 1) + 30), (26, 22, 26))
    dr = ImageDraw.Draw(img)
    dr.text((gap, 8), "左：真谱  右：模型生成（同一段 128 音）", fill=(235, 235, 235), font=font(18))
    for r, (idx, pts, prd) in enumerate(zip(rows, Y[rows], pred)):
        y0 = 34 + r * (H + gap)
        m = M[idx].astype(bool)
        info = "fid %s 窗口起点 %d" % (fid[idx], int(d["start"][idx]))
        panel(types[r], pts, y0, H, W, "真谱 " + info, dr, y0, 0)
        panel(types[r], prd, y0, H, W, "生成 (MAE %.0fpx)" % (np.mean(np.abs(prd[m] - pts[m]) * 675)), dr, y0, W + gap * 2)
    img.save(a.out)
    print("saved", a.out, img.size)
    for r, idx in enumerate(rows):
        m = M[idx].astype(bool)
        print(fid[idx], "MAE %.1f px" % (np.mean(np.abs(pred[r][m] - Y[idx][m]) * 675)))


if __name__ == "__main__":
    main()
