#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pec (Phigros legacy 文本) 谱面解析器 → 与 phira_fetch.parse_chart 同构输出。
指令格式（已验证 fid=40706 样本）：
  bp <beat> <bpm>                     BPM 段（beat 为拍）
  n1 <line> <time> <x> <type> <fake>  Tap（行内 type 恒 1，以前缀定类型）
  n2 <line> <time> <endTime> <x> <type> <fake>  Hold
  n3 <line> <time> <x> <type> <fake>  Drag（链，节点逐行）
  n4 <line> <time> <x> <type> <fake>  Flick
  #/& <speed>、cv/cp/cd/ca/cm/cr/cf <line> <time> <value>  判定线事件，忽略
  '0' 首行版本号，忽略
输出 notes[{t,x,type,dur,line_id}] + segs[[start_beat,bpm]]，与 parse_chart 完全同构。
"""
import math


def beat_to_sec(beat, segs):
    """逐段累计：拍 → 秒。与 phira_fetch.parse_chart 相同算法。"""
    if not segs:
        return beat * 0.5
    total_sec = 0.0
    for i, (sb, bpm) in enumerate(segs):
        if i + 1 < len(segs):
            nb = segs[i + 1][0]
            if beat < nb:
                return total_sec + (beat - sb) * 60.0 / bpm
            total_sec += (nb - sb) * 60.0 / bpm
        else:
            return total_sec + (beat - sb) * 60.0 / bpm
    return total_sec


def parse_pec(raw_text):
    """Pec 文本 → (notes, segs)。"""
    segs = []
    raw_notes = []  # (line, t_beat, x, type, isFake, end_beat)
    for line in raw_text.splitlines():
        s = line.strip()
        if not s:
            continue
        parts = s.split()
        tag = parts[0]
        try:
            if tag == "bp":
                segs.append([float(parts[1]), float(parts[2])])
            elif tag == "n1":
                raw_notes.append((int(parts[1]), float(parts[2]), float(parts[3]),
                                  1, int(parts[5]), None))
            elif tag == "n2":
                raw_notes.append((int(parts[1]), float(parts[2]), float(parts[4]),
                                  2, int(parts[6]), float(parts[3])))
            elif tag == "n3":
                raw_notes.append((int(parts[1]), float(parts[2]), float(parts[3]),
                                  3, int(parts[5]), None))
            elif tag == "n4":
                raw_notes.append((int(parts[1]), float(parts[2]), float(parts[3]),
                                  4, int(parts[5]), None))
            # 其余 tag（0/#/&/cv/cp/cd/ca/cm/cr/cf）忽略
        except (ValueError, IndexError):
            continue  # 畸形行跳过，不中断
    segs.sort(key=lambda x: x[0])
    if not segs:
        # 无 bp 段：按默认 120 BPM 处理，保证可解析
        segs = [[0.0, 120.0]]
    notes = []
    for line_i, t, x, typ, fake, end_beat in raw_notes:
        if fake:
            continue
        sec = beat_to_sec(t, segs)
        dur = 0.0
        if end_beat is not None:
            dur = max(0.0, beat_to_sec(end_beat, segs) - sec)
        notes.append({"t": round(sec, 6), "x": float(x), "type": typ,
                      "dur": round(dur, 6), "line_id": line_i})
    notes.sort(key=lambda n: n["t"])
    return notes, segs
