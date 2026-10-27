"""
Does a published albedo help with the dark facet? The dark facet of a convex model (an artificial facet that closes
the polyhedron; Kaasalainen & Torppa 2001) absorbs what a convex, uniformly reflecting body cannot reproduce: albedo
variegation, concavities, or a wrong pole or period. A single disk-integrated albedo (NEOWISE, AKARI, IRAS, as in
the JPL SBDB) says nothing about variegation, but it could correlate with surface type. Test: dark facet of the
convex models at the LCDB periods (validation sample) and at our periods (paper sample) against the SBDB geometric
albedo, with the light curve amplitude for comparison.
Usage: python darkfacet_albedo.py      Writes darkfacet_albedo.json
"""
import json
import os

import numpy as np
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.abspath(__file__))


def J(*p):
    return json.load(open(os.path.join(ROOT, *p), encoding="utf-8"))


SB = {}
for f in ("vb/data/sbdb_full.json", "data/sbdb_phys_orbit.json", "data/sbdb_testswap_full.json"):
    if os.path.exists(os.path.join(ROOT, f)):
        for n, r in J(f).items():
            SB.setdefault(n, r)
AMP = {s["num"]: s["amax"] for s in J("validation", "sample.json")["sample"]}


def phys(n, key):
    v = {p["name"]: p["value"] for p in SB.get(n, {}).get("phys_par", [])}.get(key)
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


rows = []
for d in sorted(os.listdir(os.path.join(ROOT, "damit_runs"))):
    parts = d.split("_")
    e = os.path.join(ROOT, "damit_runs", d, "eval.json")
    if len(parts) != 2 or not os.path.exists(e):
        continue
    kind, n = parts
    C = J("damit_runs", d, "eval.json")["C"]
    m = os.path.join(ROOT, "damit_runs", d, "poles", "mirrors.json")
    dark = J("damit_runs", d, "poles", "mirrors.json")["within"][0].get("dark_pct") if os.path.exists(m) else None
    dark = C["dark_pct"] if dark is None else dark
    rows.append({"run": d, "kind": kind, "num": n, "dark_pct": dark, "albedo": phys(n, "albedo"),
                 "diameter": phys(n, "diameter"), "amax": AMP.get(n), "scan_unique": C["scan"]["unique"]})

out = {}
for kind in ("val", "paper", "all"):
    r = [x for x in rows if (kind == "all" or x["kind"] == kind) and x["albedo"] is not None and x["dark_pct"] is not None]
    a, dk = np.array([x["albedo"] for x in r]), np.array([x["dark_pct"] for x in r])
    lo, hi = dk[a < 0.10], dk[a >= 0.20]
    out[kind] = {"n": len(r), "spearman_albedo_dark": [float(v) for v in spearmanr(a, dk)],
                 "median_dark_albedo_lt_0.10": float(np.median(lo)) if len(lo) else None, "n_lt_0.10": int(len(lo)),
                 "frac_dark_ge_1pct_albedo_lt_0.10": float(np.mean(lo >= 1)) if len(lo) else None,
                 "median_dark_albedo_ge_0.20": float(np.median(hi)) if len(hi) else None, "n_ge_0.20": int(len(hi)),
                 "frac_dark_ge_1pct_albedo_ge_0.20": float(np.mean(hi >= 1)) if len(hi) else None}
    ra = [x for x in r if x["amax"] is not None]
    if ra:
        out[kind]["spearman_amplitude_dark"] = [float(v) for v in spearmanr([x["amax"] for x in ra], [x["dark_pct"] for x in ra])]
json.dump({"summary": out, "rows": rows}, open(os.path.join(ROOT, "darkfacet_albedo.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
