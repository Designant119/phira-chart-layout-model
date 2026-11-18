#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 Phigros/Phira 静态谱面预览图里粗提取 note 相对位置。
注意：这是图面反推的弱数据，不是原始谱面；时间、Hold 时长、线移动/旋转事件都只能近似。
"""
import os, glob, json, argparse, math
import numpy as np
import cv2
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

X_SCALE = 675.0
ROW_BAND = 48.0
SCALE = 0.5


def detect_rows(gray_mask):
    row_frac = gray_mask.mean(axis=1)
    ys = np.where(row_frac > 0.20)[0]
    groups = []
    for y in ys:
        if not groups or y > groups[-1][-1] + 2:
            groups.append([int(y)])
        else:
            groups[-1].append(int(y))
    centers = [float(np.mean(g)) for g in groups]
    return centers


def nearest_row(y, centers):
    if not centers:
        return -1, 999.0
    d = [abs(y - c) for c in centers]
    i = int(np.argmin(d))
    return i, d[i]


def comps(mask, min_area=4):
    n, lab, stats, cen = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        out.append({
            "x": int(x), "y": int(y), "w": int(w), "h": int(h),
            "area": int(area), "cx": float(cen[i][0]), "cy": float(cen[i][1]),
        })
    return out


def overlaps_blue(c, blue_boxes, pad=6):
    for bx, by, bw, bh in blue_boxes:
        if c["x"] - pad < bx + bw and c["x"] + c["w"] + pad > bx and \
           c["y"] - pad < by + bh and c["y"] + c["h"] + pad > by:
            return True
    return False


def one_image(path, meta):
    im0 = np.array(Image.open(path).convert("RGB"))
    h0, w0 = im0.shape[:2]
    im = cv2.resize(im0, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    h, w = im.shape[:2]

    gray = ((hsv[:, :, 1] < 28) & (hsv[:, :, 2] > 70) & (hsv[:, :, 2] < 235)).astype(np.uint8) * 255
    rows = detect_rows(gray)
    cols = np.where(gray.any(axis=0))[0]
    left = int(cols[0]) if len(cols) else 64
    right = int(cols[-1]) if len(cols) else w - 64
    max_time = float((meta or {}).get("m", 0.0) or 0.0)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    blue = cv2.inRange(hsv, np.array((88, 50, 50)), np.array((115, 255, 255)))
    blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, kernel)
    yellow = cv2.inRange(hsv, np.array((18, 50, 80)), np.array((38, 255, 255)))
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, kernel)
    pink = cv2.inRange(hsv, np.array((160, 40, 60)), np.array((180, 255, 255))) | \
           cv2.inRange(hsv, np.array((0, 40, 60)), np.array((8, 255, 255)))
    pink = cv2.morphologyEx(pink, cv2.MORPH_CLOSE, kernel)

    blue_cs = comps(blue, min_area=8)
    blue_boxes = [(c["x"], c["y"], c["w"], c["h"]) for c in blue_cs]
    yellow_cs = [c for c in comps(yellow, min_area=8) if not overlaps_blue(c, blue_boxes)]
    pink_cs = comps(pink, min_area=6)

    notes = []

    def add(c, ntype, hold=0.0):
        rid, dy = nearest_row(c["cy"], rows)
        if dy > 60 * SCALE:
            rid = -1
        lane = max(-1.0, min(1.0, (c["cy"] - rows[rid]) / (ROW_BAND * SCALE))) if rid >= 0 else 0.0
        sec = 0.0 if right <= left or max_time <= 0 else (c["cx"] - left) / (right - left) * max_time
        bbox = [round(c["x"] / SCALE, 1), round(c["y"] / SCALE, 1),
                round(c["w"] / SCALE, 1), round(c["h"] / SCALE, 1)]
        notes.append({
            "t": round(float(sec), 6),
            "x": round(float(lane * X_SCALE), 3),
            "type": ntype,
            "dur": round(float(hold), 6),
            "line_id": rid,
            "src": "image",
            "bbox": bbox,
            "row_dist": round(float(dy / SCALE), 2),
        })

    for c in blue_cs:
        hold = 1.0 if c["h"] >= 12 else 0.0
        add(c, 2 if hold else 1, hold=0.0)
    for c in yellow_cs:
        add(c, 3)
    for c in pink_cs:
        add(c, 4)

    notes.sort(key=lambda n: n["t"])
    counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for n in notes:
        counts[n["type"]] += 1

    meta_t = (meta or {}).get("t", [0, 0, 0, 0])
    return {
        "notes": notes,
        "rows": len(rows),
        "image": {"width": w0, "height": h0, "left": round(left / SCALE, 1), "right": round(right / SCALE, 1), "maxTime": max_time},
        "extract_counts": counts,
        "meta_counts": {"tap": meta_t[0], "drag": meta_t[1], "hold": meta_t[2], "flick": meta_t[3]},
        "note": "时间轴和车道位置为图面反推；Hold 时长未可靠还原；线移动/旋转事件未提取。",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--info", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    info = json.load(open(a.info, encoding="utf-8"))
    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.images, "**", "*.png"), recursive=True))
    if a.limit:
        files = files[:a.limit]

    summary = []
    for p in files:
        dif = os.path.basename(os.path.dirname(p))
        fid = os.path.splitext(os.path.basename(p))[0]
        meta = (info.get(fid, {}) or {}).get(dif)
        res = one_image(p, meta)
        outp = os.path.join(a.out, f"{fid}__{dif}.json")
        json.dump(res, open(outp, "w", encoding="utf-8"), ensure_ascii=False)
        ec, mc = res["extract_counts"], res["meta_counts"]
        summary.append({
            "fid": fid, "difficulty": dif, "notes": len(res["notes"]),
            "rows": res["rows"],
            "extract": ec,
            "meta": {"tap": mc["tap"], "hold": mc["hold"], "drag": mc["drag"], "flick": mc["flick"]},
        })

    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("images", len(summary))
    print("notes", sum(x["notes"] for x in summary))


if __name__ == "__main__":
    main()
