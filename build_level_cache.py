#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""扫描全部 pez，提取 META.level -> level_cache.json {fid: level_float}。
pez: subsets/unstable_*/*.pez；文件名前缀即 fid（如 15710_Circle-rombed oxyg.pez）。
zip 内含若干 .json，找含 META 键的那个，META.level 形如 'IN Lv.15'，正则 Lv\.?([\d.]+)。
"""
import glob, os, re, zipfile, json

BASE = "/home/user/Doubao/chats/38445055719679234/phira_charts_dl"
LVL_RE = re.compile(r"Lv\.?([\d.]+)")


def main():
    pezs = glob.glob(os.path.join(BASE, "subsets", "unstable_*", "*.pez"))
    cache = {}
    seen_basenames = set()
    n_read, n_fail = 0, 0
    for p in pezs:
        fn = os.path.basename(p)
        if fn in seen_basenames:
            continue
        seen_basenames.add(fn)
        fid = fn.split("_")[0]
        try:
            z = zipfile.ZipFile(p)
            js = [n for n in z.namelist() if n.endswith(".json")]
            lv = None
            for j in js:
                d = json.loads(z.read(j).decode("utf-8"))
                if not isinstance(d, dict):
                    continue
                meta = d.get("META")
                if not isinstance(meta, dict):
                    continue
                mm = LVL_RE.search(str(meta.get("level", "")))
                if mm:
                    lv = float(mm.group(1))
                    # 自制谱常出现离谱等级（Lv.514/108967/2024...），钳制到合理上限 20
                    lv = min(lv, 20.0)
                    break
            if lv is None:
                n_fail += 1
                continue
            # 无冲突；同一 fid 已存在则取较难者（max）保险
            cache[fid] = max(cache.get(fid, -1), lv)
            n_read += 1
        except Exception:
            n_fail += 1
    out = os.path.join(BASE, "level_cache.json")
    json.dump(cache, open(out, "w"), ensure_ascii=False, indent=0)
    print(f"pez 唯一 basename: {len(seen_basenames)} | 提取 level: {n_read} | 失败: {n_fail}")
    print(f"fid 覆盖: {len(cache)} -> {out}")
    for probe in ("15710", "15372"):
        print(f"  level[{probe}] = {cache.get(probe, 'MISSING')}")


if __name__ == "__main__":
    main()
