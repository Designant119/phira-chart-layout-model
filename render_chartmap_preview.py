#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按静态谱面预览长图的样子，把训练样本画成 PNG。"""
import os, json, glob, argparse
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"
COLORS = {
    1: (66, 165, 245),   # Tap 蓝
    2: (77, 208, 225),   # Hold 青
    3: (255, 213, 79),   # Drag 黄
    4: (240, 98, 146),   # Flick 粉
}
NAMES = {1: "Tap", 2: "Hold", 3: "Drag", 4: "Flick"}


def font(size):
    return ImageFont.truetype(FONT_PATH, size) if os.path.exists(FONT_PATH) else ImageFont.load_default()


def render_chart(d, title, out_path, px_per_sec=42, row_h=96):
    notes = sorted(d["notes"], key=lambda n: n["t"])
    rows = sorted({int(n["line_id"]) for n in notes})
    row_index = {r: i for i, r in enumerate(rows)}
    max_t = max([float(n["t"]) + float(n.get("dur", 0)) for n in notes] or [1])

    left, right, top, bottom = 64, 64, 88, 32
    width = int(max(2200, left + right + max_t * px_per_sec))
    height = top + bottom + len(rows) * row_h
    img = Image.new("RGB", (width, height), "white")
    dr = ImageDraw.Draw(img)

    for i, r in enumerate(rows):
        y = top + i * row_h + row_h // 2
        dr.line((left, y, width - right, y), fill=(170, 170, 170), width=2)

    def tx(t):
        return left + t / max_t * (width - left - right)

    def ry(n):
        return top + row_index[int(n["line_id"])] * row_h + row_h // 2 + float(n["x"]) / 675.0 * 34

    counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for n in notes:
        typ = int(n["type"])
        counts[typ] += 1
        x0, y0 = tx(float(n["t"])), ry(n)
        if typ == 2:
            x1 = tx(min(max_t, float(n["t"]) + float(n.get("dur", 0))))
            dr.rectangle((x0, y0 - 6, x1, y0 + 6), fill=COLORS[2])
        elif typ == 1:
            dr.rounded_rectangle((x0 - 20, y0 - 5, x0 + 20, y0 + 5), radius=4, fill=COLORS[1])
        elif typ == 3:
            dr.polygon([(x0, y0 - 8), (x0 + 8, y0), (x0, y0 + 8), (x0 - 8, y0)], fill=COLORS[3])
        elif typ == 4:
            dr.polygon([(x0 - 8, y0 + 7), (x0 + 8, y0 + 7), (x0, y0 - 8)], fill=COLORS[4])

    dr.text((left, 18), title[:80], fill=(30, 30, 30), font=font(28))
    info = "  ".join(f"{NAMES[k]}:{counts[k]}" for k in [1, 2, 3, 4])
    dr.text((left, 52), info, fill=(70, 70, 70), font=font(22))
    img.save(out_path)
    return counts, max_t, len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--parsed", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    meta = json.load(open(a.meta, encoding="utf-8"))

    sheet_thumbs = []
    for item in meta:
        fid = str(item["fid"])
        p = os.path.join(a.parsed, fid + ".json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p, encoding="utf-8"))
        out = os.path.join(a.out, f"{fid}_chartmap.png")
        counts, dur, rows = render_chart(d, item["title"], out)
        item["render_counts"] = counts
        item["render_rows"] = rows
        thumb = Image.open(out)
        thumb.thumbnail((1500, 520))
        sheet_thumbs.append((item, thumb))

    # 总览长图：2 列 4 行
    cw = 1500
    ch = 560
    sheet = Image.new("RGB", (cw * 2 + 24, ch * 4 + 60), "white")
    sd = ImageDraw.Draw(sheet)
    for i, (item, thumb) in enumerate(sheet_thumbs):
        col, row = i % 2, i // 2
        x, y = 12 + col * (cw + 12), 12 + row * (ch + 12)
        sheet.paste(thumb, (x, y + 34))
        sd.text((x, y), f"{item['fid']} {item['title'][:50]}", fill=(20, 20, 20), font=font(22))
    sheet.save(os.path.join(a.out, "chartmap_contact.jpg"), quality=90)
    json.dump(meta, open(os.path.join(a.out, "render_meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("rendered", len(sheet_thumbs))


if __name__ == "__main__":
    main()
