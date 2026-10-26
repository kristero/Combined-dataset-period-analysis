"""Keep the outputs of the first rebootstrap.py version (overlapping +/-20 resolution-element search windows),
on which the preliminary proposals (proposals.json, evidence/) and the review were based, before the corrected
rebootstrap.py is re-run. Existing archive files are never overwritten. The code of that first version is kept
as rebootstrap_legacy.py (writes to a separate directory); it reproduces the archived results byte for byte."""
import json
import os
import shutil

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
n_copied = 0
for d in sorted(os.listdir(RES)):
    src = os.path.join(RES, d, f"{d}_rebootstrap.json")
    dst = os.path.join(RES, d, f"{d}_rebootstrap_legacy.json")
    if os.path.exists(src) and not os.path.exists(dst):
        if "windows" in json.load(open(src, encoding="utf-8")):
            raise SystemExit(f"{src} was already written by the corrected version; not archiving it as legacy")
        shutil.copy2(src, dst)
        n_copied += 1
for src, dst in (("proposals.json", "proposals_legacy.json"), ("evidence", "evidence_legacy")):
    s, t = os.path.join(ROOT, src), os.path.join(ROOT, dst)
    if os.path.exists(s) and not os.path.exists(t):
        (shutil.copytree if os.path.isdir(s) else shutil.copy2)(s, t)
        print("archived", src, "->", dst)
print(f"archived {n_copied} rebootstrap results")
