#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""竖版暗色游戏风谱面预览：时间向下，横向排键。"""
import os, json, argparse, collections
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"
COLORS = {
    1: (134, 233, 247),   # Tap 青蓝
    2: (200, 242, 247),   # Hold 浅蓝
    3: (255, 127, 168),   # Flick 粉（官方 type 3）
    4: (255, 213, 79),    # Drag 黄（官方 type 4）
}
NAMES = {1: "Tap", 2: "Hold", 3: "Flick", 4: "Drag"}


def font(size):
    return ImageFont.truetype(FONT_PATH, size) if os.path.exists(FONT_PATH) else ImageFont.load_default()


def dark_bg(w, h):
    img = Image.new("RGB", (w, h), (22, 18, 22))
    dr = ImageDraw.Draw(img)
    for y in range(h):
        t = y / max(1, h)
        r = int(22 + 18 * t)
        g = int(18 + 8 * t)
        b = int(22 + 12 * t)
        dr.line((0, y, w, y), fill=(r, g, b))
    return img


def glow_box(dr, box, base, edge):
    """像游戏里 Hold 那样的渐变方块。"""
    x0, y0, x1, y1 = [int(v) for v in box]
    hgt = max(1, y1 - y0)
    for i in range(hgt):
        t = i / hgt
        c = tuple(int(base[k] * (0.72 + 0.36 * (1 - t))) for k in range(3))
        dr.line((x0, y0 + i, x1, y0 + i), fill=c)
    dr.rounded_rectangle((x0, y0, x1, y1), radius=5, outline=edge, width=2)


def render_chart(d, title, out_path, mainline=None, time_scale=22, x_scale=190, by_speed=False, width=500):
    fid = d.get("fid")
    notes = d.get("notes", [])
    if mainline and fid in mainline:
        lid = int(mainline[fid])
        notes = [n for n in notes if int(n.get("line_id", 0)) == lid]
    notes = sorted(notes, key=lambda n: n["t"])
    if not notes:
        return None

    max_t = max(float(n["t"]) + float(n.get("dur", 0)) for n in notes)
    w = int(width)
    sc = w / 500.0
    top, bottom = int(86 * sc), int(44 * sc)

    # 按流速算每颗音该落多远：speed=10 对应 time_scale 像素/秒
    K = time_scale / 10.0
    ys = []
    if by_speed:
        acc = 0.0
        prev_t = None
        for n in notes:
            t = float(n["t"])
            if prev_t is not None:
                sp = float(n.get("speed", 10.0) or 10.0)
                sp = min(max(sp, 0.6), 25.0)
                acc += max(0.0, t - prev_t) * K * sp
            ys.append(acc)
            prev_t = t
        span = max(acc, 1.0)
        h = int(max(1800, top + bottom + span))
    else:
        h = int(max(1800, max_t * time_scale))
        span = h - top - bottom

    img = dark_bg(w, h)
    dr = ImageDraw.Draw(img)

    if by_speed:
        y_of = {i: top + ys[i] for i in range(len(notes))}
    else:
        y_of = None

    def ypos(i):
        if y_of is not None:
            return y_of[i]
        return top + float(notes[i]["t"]) / max_t * span

    def xy(i):
        n = notes[i]
        nx = float(n.get("abs_x", n["x"]))
        return w / 2 + nx / 675.0 * x_scale, ypos(i)

    def onscreen(i):
        return abs(float(notes[i].get("abs_x", notes[i]["x"]))) <= 675.0

    # 横向网格线，按秒铺
    if by_speed:
        ts = [float(n["t"]) for n in notes]
        for sec in range(0, int(max_t) + 1, 2):
            idx = int(np.searchsorted(np.array(ts), sec))
            if idx >= len(ts):
                continue
            if idx == 0:
                gy = top
            else:
                gy = ypos(idx - 1) + (ypos(idx) - ypos(idx - 1)) * (sec - ts[idx - 1]) / max(1e-9, ts[idx] - ts[idx - 1])
            if gy < h - bottom:
                dr.line((int(20*sc), gy, w - int(20*sc), gy), fill=(74, 68, 72), width=1)
    else:
        for sec in range(0, int(max_t) + 1, 2):
            y = top + sec / max_t * span
            dr.line((int(20*sc), y, w - int(20*sc), y), fill=(74, 68, 72), width=1)

    counts = {1: 0, 2: 0, 3: 0, 4: 0}
    drag_pts = []

    # 第一遍：Hold 的尾巴贴底画，细一点，不盖住别的音
    for i, n in enumerate(notes):
        typ = int(n["type"])
        if typ != 2 or not onscreen(i):
            continue
        x, y = xy(i)
        if by_speed:
            sp = min(max(float(n.get("speed", 10.0) or 10.0), 0.6), 25.0)
            hold_h = max(40, float(n.get("dur", 0)) * K * sp)
        else:
            hold_h = max(40, float(n.get("dur", 0)) * time_scale * 8 / 22)
        dr.rectangle((x - 5*sc, y + 22*sc, x + 5*sc, y + hold_h), fill=(96, 165, 178))
        dr.rectangle((x - 5*sc, y + hold_h - max(3,4*sc), x + 5*sc, y + hold_h), fill=(255, 226, 132))

    # 第二遍：Tap / Drag / Flick
    for i, n in enumerate(notes):
        typ = int(n["type"])
        counts[typ] += 1
        if typ == 2 or not onscreen(i):
            if typ == 2:
                counts[2] += 1
            continue
        x, y = xy(i)
        if typ == 1:
            dr.rounded_rectangle((x - 21*sc, y - 3*sc, x + 21*sc, y + 3*sc), radius=3, fill=COLORS[1])
        elif typ == 4:
            drag_pts.append((x, y))
            dr.rounded_rectangle((x - 18*sc, y - 3*sc, x + 18*sc, y + 3*sc), radius=3, fill=COLORS[4])
        elif typ == 3:
            dr.rounded_rectangle((x - 15*sc, y - 3*sc, x + 15*sc, y + 3*sc), radius=3, fill=COLORS[3])

    # 第三遍：Hold 方块头压在最上面，跟参考图一样是块的形状
    for i, n in enumerate(notes):
        if int(n["type"]) != 2 or not onscreen(i):
            continue
        x, y = xy(i)
        glow_box(dr, (x - 24*sc, y, x + 24*sc, y + 44*sc), COLORS[2], (255, 226, 132))

    # Drag 链状连线，阈值跟着纵向尺度走
    link_max = time_scale * 5.5
    for (x1, y1), (x2, y2) in zip(drag_pts, drag_pts[1:]):
        if abs(y2 - y1) < link_max and abs(x2 - x1) < 150:
            dr.line((x1, y1, x2, y2), fill=COLORS[4], width=2)

    dr.text((int(24*sc), int(18*sc)), title[:34], fill=(235, 235, 235), font=font(max(10,int(22*sc))))
    info = " ".join(f"{NAMES[k]}:{counts[k]}" for k in [1, 2, 3, 4])
    dr.text((int(24*sc), int(52*sc)), info, fill=(190, 190, 190), font=font(max(9,int(18*sc))))
    img.save(out_path)
    return counts, max_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--parsed", required=True)
    ap.add_argument("--mainline", default="")
    ap.add_argument("--time-scale", type=float, default=22)
    ap.add_argument("--x-scale", type=float, default=190)
    ap.add_argument("--by-speed", action="store_true")
    ap.add_argument("--width", type=float, default=500)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    meta = json.load(open(a.meta, encoding="utf-8"))
    mainline = json.load(open(a.mainline, encoding="utf-8")) if a.mainline else None

    thumbs = []
    for item in meta:
        fid = str(item["fid"])
        p = os.path.join(a.parsed, fid + ".json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p, encoding="utf-8"))
        out = os.path.join(a.out, f"{fid}_vertical.png")
        res = render_chart(d, item["title"], out, mainline, a.time_scale, a.x_scale, a.by_speed, a.width)
        if not res:
            continue
        item["vertical_counts"], item["duration"] = res
        thumb = Image.open(out)
        thumb.thumbnail((300, 760))
        thumbs.append((item, thumb))

    cw, ch = 320, 820
    sheet = Image.new("RGB", (cw * 2 + 24, ch * 4 + 24), (18, 15, 18))
    sd = ImageDraw.Draw(sheet)
    for i, (item, thumb) in enumerate(thumbs):
        col, row = i % 2, i // 2
        x, y = 12 + col * (cw + 12), 12 + row * (ch + 12)
        sd.text((x, y), f"{item['fid']}", fill=(230, 230, 230), font=font(18))
        sheet.paste(thumb, (x, y + 28))
    sheet.save(os.path.join(a.out, "vertical_contact.jpg"), quality=90)
    json.dump(meta, open(os.path.join(a.out, "vertical_meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("rendered", len(thumbs))


if __name__ == "__main__":
    main()
