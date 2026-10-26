"""
Keep the adopted.json entries of the retired test asteroids (retired.json) in adopted_retired.json, from the copy made
before the swap (backup_before_testswap/adopted.json). Used for the convex models of (1963) and (2134), which stay in
Table tab:damit (convex_only in retired.json), and for the (2150) light-curve template of amp_injection2.py.
"""
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
ret = {k: v for k, v in json.load(open(os.path.join(ROOT, "retired.json"), encoding="utf-8")).items() if not k.startswith("_")}
old = json.load(open(os.path.join(ROOT, "backup_before_testswap", "adopted.json"), encoding="utf-8"))
out = {n: {**old[n], "retired": True, "convex_only": bool(ret[n].get("convex_only"))} for n in sorted(ret, key=int)}
json.dump(out, open(os.path.join(ROOT, "adopted_retired.json"), "w", encoding="utf-8"), indent=1)
for n, a in out.items():
    print(n, a["P"], a.get("sigma_P"), a["quality"], "convex_only" if a["convex_only"] else "")
