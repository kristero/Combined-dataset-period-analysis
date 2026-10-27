"""
Blind validation run (vb/): adopted.json from the algorithmic proposals of adjudicate_pre.py, without any review, so
that rebootstrap2.py and quality_rule.py grade the fully automatic result.
Usage: python vb_autofinal.py [tree]      (default vb/)
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
T = os.path.join(ROOT, sys.argv[1] if len(sys.argv) > 1 else "vb")
props = json.load(open(os.path.join(T, "proposals.json"), encoding="utf-8"))
ad = {}
for n, p in props.items():
    if "error" in p:
        print(n, "error:", p["error"])
        continue
    ad[n] = {"P": p["P"], "sigma_P": p["sigma_P"], "quality": p["quality_auto"], "table": 2,
             "alternatives": [a["P"] for a in p.get("alternatives_auto", [])], "source": "algorithmic proposal (blind)"}
json.dump(ad, open(os.path.join(T, "adopted.json"), "w", encoding="utf-8"), indent=1)
print(len(ad), "asteroids written to", os.path.join(T, "adopted.json"))
