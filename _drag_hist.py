import json, glob, numpy as np

allx = []
drag = []
for f in sorted(glob.glob('/home/user/Doubao/chats/38453392661469954/_run_real_traj/phira_all/parsed/*.json')):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    for n in d['notes']:
        if n['x'] == 0:
            continue
        allx.append(n['x'])
        if n['type'] == 3:
            drag.append(n['x'])

allx = np.array(allx); drag = np.array(drag)
edges = np.array([-675, -540, -405, -270, -135, -90, 0, 90, 135, 270, 405, 540, 675])


def h(x):
    c, _ = np.histogram(x, bins=edges)
    return np.round(100 * c / len(x), 1)


print('区间      ', ['-675~-540', '-540~-405', '-405~-270', '-270~-135', '-135~-90', '-90~0', '0~90', '90~135', '135~270', '270~405', '405~540', '540~675'])
print('全体音符  ', h(allx))
print('Drag      ', h(drag))
print()
print("全体 |x|中位", round(float(np.median(np.abs(allx))), 1), " Drag", round(float(np.median(np.abs(drag))), 1))
print("全体 |x|>400", round(100 * float((np.abs(allx) > 400).mean()), 1), "%  Drag", round(100 * float((np.abs(drag) > 400).mean()), 1), "%")
print("全体 |x|<135", round(100 * float((np.abs(allx) < 135).mean()), 1), "%  Drag", round(100 * float((np.abs(drag) < 135).mean()), 1), "%")
