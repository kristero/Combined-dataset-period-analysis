"""
Zero-point split (zp_split.split_one) for the six test asteroids added in the revision, with the same file layout as
zp_integrate.py left for the other 57: split workbooks and obs tables in data/ (and zp/data/ for the zp tree), the
unsplit originals in data_unsplit/. The split log is merged into zp/zp_split.json.
Usage: python testswap_split.py 249 486 870 1600 1700 7750
"""
import json
import os
import shutil
import sys

import zp_split as Z

ROOT = os.path.dirname(os.path.abspath(__file__))
nums = [int(x) for x in sys.argv[1:]]
logpath = os.path.join(Z.ZP, "zp_split.json")
alllog = json.load(open(logpath)) if os.path.exists(logpath) else {}
for n in nums:
    for sub, pat in (("workbooks", f"{n}-mpc.xlsx"), ("obs_tables", f"{n}_info.json"), ("obs_tables", f"{n}_obs.csv")):
        src = os.path.join(ROOT, "data", sub, pat)
        dst = os.path.join(ROOT, "data_unsplit", sub, pat)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
    alllog[str(n)] = Z.split_one(n)
    for sub, pat in (("workbooks", f"{n}-mpc.xlsx"), ("obs_tables", f"{n}_info.json"), ("obs_tables", f"{n}_obs.csv")):
        src = os.path.join(Z.ZP, "data", sub, pat)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(ROOT, "data", sub, pat))
    print(n, {k: v["kept"] for k, v in alllog[str(n)].items()})
json.dump(alllog, open(logpath, "w"), indent=1)
