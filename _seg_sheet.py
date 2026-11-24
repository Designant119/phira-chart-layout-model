#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把自动分段的结果做成一张对比图：每段一张截图 + 该段手感配置。"""
import json
import os
import subprocess
import sys
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, '.')
from phira_fetch import parse_chart
from seg_detect import beat_notes, detect, feats

FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc'


def fnt(size):
    return ImageFont.truetype(FONT, size) if os.path.exists(FONT) else ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chart', required=True)
    ap.add_argument('--video', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--cols', type=int, default=6)
    a = ap.parse_args()

    d = json.load(open(a.chart, encoding='utf-8'))
    _, bpm_segs = parse_chart(d)

    def b2s(beat):
        total = 0.0
        for i, (sb, bpm) in enumerate(bpm_segs):
            if i + 1 < len(bpm_segs):
                nb = bpm_segs[i + 1][0]
                if beat < nb:
                    return total + (beat - sb) * 60.0 / bpm
                total += (nb - sb) * 60.0 / bpm
            else:
                return total + (beat - sb) * 60.0 / bpm
        return beat * 0.5

    seq = beat_notes(a.chart)
    bounds = detect(seq)
    tmpdir = '/tmp/segframes'
    os.makedirs(tmpdir, exist_ok=True)

    cells = []
    for i, (lo, hi) in enumerate(bounds):
        mid = seq[(lo + hi) // 2]['b']
        sec = b2s(mid)
        p = os.path.join(tmpdir, 's%02d.png' % i)
        subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-ss', '%.2f' % sec,
                        '-i', a.video, '-frames:v', '1', p], check=False)
        f = feats(seq, lo, hi)
        ncount = sum(s['n'] for s in seq[lo:hi])
        cells.append({'i': i, 'img': p if os.path.exists(p) else None,
                      'txt': '#%d  %d音/%.0fs  每拍%.2f  dg%.0f%% fl%.0f%%  步%.0f 换%.2f'
                             % (i, ncount, b2s(seq[hi - 1]['b']) - b2s(seq[lo]['b']),
                                f[0], 100 * f[1], 100 * f[2], f[4], f[5])})

    cols = a.cols
    rows = int(np.ceil(len(cells) / cols))
    cw, ch = 320, 180
    pad, label_h, gap = 8, 24, 10
    W = cols * (cw + gap) + gap
    H = rows * (ch + label_h + gap) + gap + 40
    sheet = Image.new('RGB', (W, H), (14, 12, 16))
    dr = ImageDraw.Draw(sheet)
    dr.text((12, 10), '赴大荒 真谱 · 自动分段 %d 段（每段取中点一帧）' % len(cells),
            font=fnt(20), fill=(230, 230, 235))
    for k, c in enumerate(cells):
        x = gap + (k % cols) * (cw + gap)
        y = 40 + (k // cols) * (ch + label_h + gap)
        if c['img']:
            im = Image.open(c['img']).convert('RGB').resize((cw, ch))
            sheet.paste(im, (x, y))
        dr.rectangle([x, y + ch, x + cw, y + ch + label_h], fill=(24, 22, 28))
        dr.text((x + 4, y + ch + 4), c['txt'], font=fnt(12), fill=(200, 220, 235))
    sheet.save(a.out)
    print('段数', len(cells), '输出', a.out, sheet.size)


if __name__ == '__main__':
    main()
