#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从渲染帧里量 note 的实际像素宽度，反推 RPE 坐标下的宽度"""
import sys
import numpy as np
from PIL import Image
from scipy import ndimage

CYAN = lambda a: (a[:, :, 1] > 190) & (a[:, :, 2] > 190) & (a[:, :, 0] < 170)   # tap 青
GOLD = lambda a: (a[:, :, 0] > 180) & (a[:, :, 1] > 150) & (a[:, :, 2] < 150)   # drag 金


def report(path):
    im = np.array(Image.open(path).convert('RGB')).astype(int)
    H, W, _ = im.shape
    print('图像 %dx%d' % (W, H))
    for name, mask in (('tap', CYAN(im)), ('drag', GOLD(im))):
        lab, n = ndimage.label(mask)
        boxes = []
        for i in range(1, n + 1):
            ys, xs = np.where(lab == i)
            if len(xs) < 40:
                continue
            boxes.append((xs.min(), xs.max(), ys.min(), ys.max(), len(xs)))
        boxes.sort(key=lambda b: -(b[2] - b[3]))
        wide = [b[1] - b[0] + 1 for b in boxes]
        high = [b[2] - b[1] + 1 for b in boxes]
        if wide:
            print(' %s 块数%d 宽: 中位%.0f 最大%.0f | 高: 中位%.0f 最大%.0f' % (
                name, len(boxes), np.median(wide), max(wide), np.median(high), max(high)))
            for b in boxes[:3]:
                print('    bbox x[%d,%d] y[%d,%d] w=%d h=%d' % (
                    b[0], b[1], b[2], b[3], b[1] - b[0] + 1, b[3] - b[2] + 1))


if __name__ == '__main__':
    for p in sys.argv[1:]:
        print('===', p)
        report(p)
