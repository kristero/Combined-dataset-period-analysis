"""Keep the outputs of the first rebootstrap2.py version (extended-range and weighted powers on 151-point fastchi2
grids; plain, non-exclusive +/-1.5 own windows) and the downstream files computed from them, before the corrected
rebootstrap2.py is re-run. Existing archive files are never overwritten.
  results/<N>/<N>_rebootstrap2.json -> results/<N>/<N>_rebootstrap2_v1.json
  quality_final.json -> quality_final_v1.json, adopted.json -> adopted_v1.json"""
import json
import os
import shutil

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
n = 0
for d in sorted(os.listdir(RES)):
    src = os.path.join(RES, d, f"{d}_rebootstrap2.json")
    dst = os.path.join(RES, d, f"{d}_rebootstrap2_v1.json")
    if os.path.exists(src) and not os.path.exists(dst):
        if "windows" in json.load(open(src, encoding="utf-8")):
            raise SystemExit(f"{src} was already written by the corrected version; not archiving it as v1")
        shutil.copy2(src, dst)
        n += 1
for src, dst in (("quality_final.json", "quality_final_v1.json"), ("adopted.json", "adopted_v1.json")):
    s, t = os.path.join(ROOT, src), os.path.join(ROOT, dst)
    if os.path.exists(s) and not os.path.exists(t):
        shutil.copy2(s, t)
        print("archived", src, "->", dst)
print(f"archived {n} rebootstrap2 results")
