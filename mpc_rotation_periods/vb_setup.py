"""
Blind run of the automatic period pipeline on the 120 validation asteroids (tests of the checks suggested in the
revision for slow rotators; the validation sample had so far only been folded at its LCDB periods).
Tree vb/ (a ROOT for the copied scripts, as zp/ in zp_setup.py):
  - the scripts of the period chain: mpc_redo.py (zp tree version, REPO = ../repo_snap), rebootstrap.py,
    adjudicate_pre.py, rebootstrap2.py, quality_rule.py;
  - the validation MPC data (validation/data) split at the 703/G96 zero-point change like the paper data (zp_split);
  - published.json without any literature value (no seeded candidates), SBDB records without the rotation period
    (data/sbdb_phys_orbit.json; the full records in data/sbdb_full.json, for the evaluation only).
Usage: python vb_setup.py
"""
import json
import os
import shutil
import time
import urllib.request

import zp_split as Z

ROOT = os.path.dirname(os.path.abspath(__file__))
VB = os.path.join(ROOT, "vb")
VAL = os.path.join(ROOT, "validation")
SAMPLE = json.load(open(os.path.join(VAL, "sample.json"), encoding="utf-8"))["sample"]
NUMS = [s["num"] for s in SAMPLE]

os.makedirs(os.path.join(VB, "data"), exist_ok=True)
txt = open(os.path.join(ROOT, "zp", "mpc_redo.py"), encoding="utf-8").read()
old = next(l for l in txt.splitlines() if l.startswith("REPO = "))
txt = txt.replace(old, 'REPO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "repo_snap")')
open(os.path.join(VB, "mpc_redo.py"), "w", encoding="utf-8", newline="\n").write(txt)
for s in ("rebootstrap.py", "adjudicate_pre.py", "rebootstrap2.py", "quality_rule.py"):
    shutil.copy(os.path.join(ROOT, s), os.path.join(VB, s))
shutil.copy(os.path.join(ROOT, "data", "mpc_obscodes.json"), os.path.join(VB, "data", "mpc_obscodes.json"))

# zero-point split of the validation data into vb/data
Z.ROOT, Z.ZP = VAL, VB
log = {n: Z.split_one(n) for n in NUMS}
json.dump(log, open(os.path.join(VB, "zp_split.json"), "w"), indent=1)
for n in NUMS:      # workbook path recorded in the info files: relative to the tree on any machine
    p = os.path.join(VB, "data", "obs_tables", f"{n}_info.json")
    info = json.load(open(p, encoding="utf-8"))
    info["workbook"] = f"data/workbooks/{n}-mpc.xlsx"
    json.dump(info, open(p, "w", encoding="utf-8"), indent=2)
print("split:", sum(1 for v in log.values() for L in v.values() if L["pre"] >= 30 and L["post"] >= 30), "datasets in two parts")

json.dump({n: {"table": 2} for n in NUMS}, open(os.path.join(VB, "published.json"), "w"), indent=1)

full_path = os.path.join(VB, "data", "sbdb_full.json")
full = json.load(open(full_path, encoding="utf-8")) if os.path.exists(full_path) else {}
for n in NUMS:
    if n in full:
        continue
    for attempt in range(5):
        try:
            with urllib.request.urlopen(f"https://ssd-api.jpl.nasa.gov/sbdb.api?sstr={n}&phys-par=1", timeout=60) as r:
                full[n] = json.load(r)
            break
        except OSError:
            time.sleep(5 * (attempt + 1))
    time.sleep(0.3)
json.dump(full, open(full_path, "w", encoding="utf-8"), indent=1)
blind = {}
for n, e in full.items():
    e = json.loads(json.dumps(e))
    e["phys_par"] = [p for p in e.get("phys_par", []) if p["name"] != "rot_per"]
    blind[n] = e
json.dump(blind, open(os.path.join(VB, "data", "sbdb_phys_orbit.json"), "w", encoding="utf-8"), indent=1)
print("SBDB records:", len(full), "of", len(NUMS))
