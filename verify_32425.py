import json, math, zipfile, sys
sys.path.insert(0, "/home/user/Doubao/chats/38445055719679234/phira_charts_dl")
from easing_lib import line_state_at, _beat
from verify_easing import build_segs, beat_to_sec

def parse_indep(pez_path):
    zf = zipfile.ZipFile(pez_path)
    jn = [n for n in zf.namelist() if n.lower().endswith(".json") and "AutoSave" not in n]
    chart = json.loads(zf.read(jn[0]).decode("utf-8", errors="replace"))
    segs = build_segs(chart.get("BPMList", []))
    out = []
    for jl in chart.get("judgeLineList", []):
        for n in jl.get("notes", []) or []:
            if n.get("isFake") or n.get("type", 1) not in (1,2,3,4):
                continue
            b = _beat(n["startTime"])
            px, py, theta = line_state_at(jl, chart, b)
            if abs(theta) > 10.0:
                continue
            t = beat_to_sec(segs, b)
            gx = px + (n.get("positionX", 0.0) or 0.0) * math.cos(math.radians(theta))
            dur = 0.0
            et = n.get("endTime")
            if et is not None:
                d = beat_to_sec(segs, _beat(et)) - t
                dur = d if d > 0 else 0.0
            out.append({"t": round(t,6), "x": round(gx,3), "type": n.get("type",1), "dur": round(dur,6)})
    out.sort(key=lambda k: (k["t"], k["x"]))
    return out

ind = parse_indep("subsets/unstable_0005/32425_Squidward Finishes Those Errands.pez")
print(f"独立官方语义: {len(ind)} 音符")
try:
    with open("parsed/32425.json") as f:
        old = json.load(f)
    old_notes = sorted(old.get("notes", []), key=lambda k: (k["t"], k["x"]))
    print(f"旧线性解析:   {len(old_notes)} 音符")
    diffs = maxdiff = 0.0
    n_diff = 0
    for a, b in zip(ind, old_notes):
        d = abs(a["x"] - b["x"])
        if d > 0.5:
            n_diff += 1
            maxdiff = max(maxdiff, d)
    print(f"x 差异>0.5: {n_diff}/{len(ind)}，最大差异 {maxdiff:.2f}")
    # 抽样：差异最大的几个
    diffs_list = []
    for a, b in zip(ind, old_notes):
        d = abs(a["x"] - b["x"])
        if d > 0.5:
            diffs_list.append((d, a["t"], a["x"], b["x"]))
    diffs_list.sort(reverse=True)
    print("最大差异样本 (diff, t, 新x, 旧x):")
    for d in diffs_list[:5]:
        print(f"  {d}")
except Exception as e:
    print("旧 parsed 读取失败:", e)

# 手算抽查：找几个非线性事件，验证官方 easing 值
import zipfile
zf = zipfile.ZipFile("subsets/unstable_0005/32425_Squidward Finishes Those Errands.pez")
jn = [n for n in zf.namelist() if n.lower().endswith(".json") and "AutoSave" not in n]
chart = json.loads(zf.read(jn[0]).decode("utf-8", errors="replace"))
print("\n手算抽查（官方语义）：")
shown = 0
for jl in chart.get("judgeLineList", []):
    for layer in (jl.get("eventLayers", []) or []):
        if isinstance(layer, dict):
            for key in ("rotateEvents", "moveXEvents", "moveYEvents"):
                for e in (layer.get(key) or []):
                    et = e.get("easingType", 1)
                    if et != 1 and shown < 6:
                        s, en = _beat(e["startTime"]), _beat(e["endTime"])
                        sv, ev = e.get("start", 0.0), e.get("end", 0.0)
                        # 取中间点
                        mid = (s + en) / 2
                        px, py, theta = line_state_at(jl, chart, mid)
                        # 官方公式：u=(mid-s)/(en-s), f(u)*(ev-sv)+sv
                        u = (mid - s) / (en - s)
                        from easing_lib import ease
                        expect = sv + (ev - sv) * ease(u, et)
                        print(f"  {key} easingType={et} u={u:.3f} 期望={expect:.3f} 独立计算={theta if 'rotate' in key else px:.3f}")
                        shown += 1
with open("verify_32425_official.json", "w") as f:
    json.dump({"fid":"32425","count":len(ind),"notes":ind}, f, ensure_ascii=False)
print("\n已保存 verify_32425_official.json")
