#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把规则/模型输出的原始 chart json 转成长图脚本吃的 parsed 格式。
用法: python3 convert_to_parsed.py <chart.json> <out.json> [fid]"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_chart_from_model import collect


def main():
    chart = json.load(open(sys.argv[1], encoding='utf-8'))
    rec = collect(chart)
    notes = [dict(t=round(r['t'], 4), x=round(r['x'], 2), type=r['type'],
                  line_id=r['li'], dur=round(r['dur'], 4)) for r in rec]
    out = dict(fid=int(sys.argv[3]) if len(sys.argv) > 3 else 15831, notes=notes)
    json.dump(out, open(sys.argv[2], 'w', encoding='utf-8'), ensure_ascii=False)
    print('notes', len(notes))


if __name__ == '__main__':
    main()
