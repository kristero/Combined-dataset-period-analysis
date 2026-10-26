"""List asteroids whose corrected <N>_rebootstrap.json is complete (valid JSON in the corrected format) and whose
<N>_rebootstrap2.json has not yet been written by the corrected rebootstrap2.py.
Usage: python ready_for_rebootstrap2.py [N to exclude ...]"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
exclude = set(sys.argv[1:])
ready = []
for num in sorted(json.load(open(os.path.join(ROOT, "adopted.json"), encoding="utf-8")), key=int):
    if num in exclude:
        continue
    try:
        rb = json.load(open(os.path.join(RES, num, f"{num}_rebootstrap.json"), encoding="utf-8"))
    except Exception:  # noqa: BLE001  (missing, empty or being written)
        continue
    if rb.get("windows", {}).get("powers_from") != "wide grid, masked":
        continue
    try:
        r2 = json.load(open(os.path.join(RES, num, f"{num}_rebootstrap2.json"), encoding="utf-8"))
        if r2.get("windows", {}).get("powers_from") == "wide grid, masked":
            continue
    except Exception:  # noqa: BLE001
        pass
    ready.append(num)
print(" ".join(ready))
