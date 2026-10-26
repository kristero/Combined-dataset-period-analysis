"""Recompute the spin-axis comparison of geometry_validate.py (section 2 of geometry_validation.json) with the corrected
published poles of lcdb_poles.py (2026-10-06 fix), without refitting the aspect models (geometry_results.json).
Only full published poles are used; asteroids with latitude-only solutions are skipped."""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, ROOT)
import amp_geometry as G  # noqa: E402

res = json.load(open(os.path.join(ROOT, "geometry_results.json")))
val = json.load(open(os.path.join(ROOT, "geometry_validation.json")))
poles = json.load(open(os.path.join(ROOT, "data", "lcdb_poles.json"), encoding="utf-8"))
rng = np.random.default_rng(3)
dist, rand, rows = [], [], []
for k, r in res.items():
    n = k.split(":")[1]
    if not r or not r.get("geometry") or n not in poles:
        continue
    sols = [p for e in poles[n] for p in e["poles"]]
    if not sols:
        continue
    g = r["geometry"]
    dd = min(G.pole_distance(g["lambda"], g["beta"], l, b) for l, b in sols)
    dist.append(dd)
    for _ in range(200):
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        lr, br = np.degrees(np.arctan2(v[1], v[0])) % 360, np.degrees(np.arcsin(v[2]))
        rand.append(min(G.pole_distance(lr, br, l, b) for l, b in sols))
    rows.append({"asteroid": k, "ours": [g["lambda"], g["beta"]], "published": sols, "distance": dd,
                 "delta_bic": g["delta_bic"]})
d = np.array(dist)
sig = np.array([r["distance"] for r in rows if r["delta_bic"] > 10])
val["poles_v1_buggy_reference"] = {k: v for k, v in val["poles"].items() if k != "rows"}
val["poles"] = {"n": len(d), "median_distance": float(np.median(d)), "random_median": float(np.median(rand)),
                "frac_within_30": float(np.mean(d <= 30)), "random_frac_within_30": float(np.mean(np.array(rand) <= 30)),
                "n_dbic_gt10": int(len(sig)), "median_distance_dbic_gt10": float(np.median(sig)) if len(sig) else None,
                "frac_within_30_dbic_gt10": float(np.mean(sig <= 30)) if len(sig) else None,
                "n_paper": sum(r["asteroid"].startswith("paper") for r in rows), "rows": rows}
json.dump(val, open(os.path.join(ROOT, "geometry_validation.json"), "w"), indent=1)
print({k: v for k, v in val["poles"].items() if k != "rows"})
