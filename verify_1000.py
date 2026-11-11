import json, math, zipfile, sys
sys.path.insert(0, "/home/user/Doubao/chats/38445055719679234/phira_charts_dl")
from easing_lib import line_state_at, interp_events, _beat, _layer_events

zf = zipfile.ZipFile("subsets/unstable_0002/1000_アイドル.pez")
jn = [n for n in zf.namelist() if n.lower().endswith(".json") and "AutoSave" not in n]
chart = json.loads(zf.read(jn[0]).decode("utf-8", errors="replace"))
jll = chart.get("judgeLineList", [])
print(f"总判定线数: {len(jll)}")

def lcount(jl):
    els = jl.get("eventLayers", []) or []
    return len(els) if isinstance(els, list) else 1

# 场景1：2+层且有音符
ml = [jl for jl in jll if lcount(jl) >= 2 and (jl.get("notes") or [])]
print(f"2+层且有音符的线: {len(ml)}")
# 场景2：父线且有音符
fa = [jl for jl in jll if jl.get("father", -1) != -1 and (jl.get("notes") or [])]
print(f"有父线且有音符的线: {len(fa)}")

def run(jl, tag):
    n0 = next((n for n in jl.get("notes", []) if not n.get("isFake")), None)
    if not n0: return
    b = _beat(n0["startTime"])
    px, py, theta = line_state_at(jl, chart, b)
    els = jl.get("eventLayers", []) or []
    if isinstance(els, dict): els = [els]
    old_th = interp_events([e for l in els for e in _layer_events(l, "rotateEvents")], b)
    old_px = interp_events([e for l in els for e in _layer_events(l, "moveXEvents")], b)
    px_nf = th_nf = 0.0
    for l in els:
        px_nf += interp_events(_layer_events(l, "moveXEvents"), b)
        th_nf += interp_events(_layer_events(l, "rotateEvents"), b)
    gx = px + (n0.get("positionX", 0.0) or 0.0) * math.cos(math.radians(theta))
    gx_old = old_px + (n0.get("positionX", 0.0) or 0.0) * math.cos(math.radians(old_th))
    print(f"\n[{tag}] beat={b:.3f} layers={lcount(jl)} father={jl.get('father')}")
    print(f"  θ: 官方={theta:.3f} 旧合并={old_th:.3f} 本线累加={th_nf:.3f}")
    print(f"  px: 官方={px:.3f} 旧合并={old_px:.3f} 本线累加={px_nf:.3f}")
    print(f"  x: 官方={gx:.3f} 旧={gx_old:.3f} 差异={gx-gx_old:+.3f}")
    if fa and jl.get("father", -1) != -1:
        print(f"  父线贡献 px: {px - px_nf:+.3f}")

if ml:
    run(ml[0], "多层累加")
if fa:
    run(fa[0], "父线叠加")
