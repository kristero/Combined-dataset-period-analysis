"""
Per-dataset values behind the dataset-power figure (make_figures.fig_dataset_power), saved so that the figure can be
redrawn in other forms without recomputing. For every asteroid and every dataset with at least 30 measurements:
number of measurements, time span, exact-chi2 Lomb-Scargle power at the adopted period (maximum within +/-1 %), the
candidate period the dataset prefers on its own (as in the subset test of the paper) and whether it belongs to the
adopted alias family. Same computation as make_figures.fig_dataset_power.
Runs on the LU HPC (hpc/ds_table.sbatch); writes dataset_power_table.json.
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import make_figures as MF


def in_family(Pc, P):
    return any(abs(Pc - r * P) <= 0.01 * Pc for r in (1.0, 2.0, 0.5))


def one(num):
    from astropy.timeseries import LombScargle
    import rebootstrap2 as r2
    a, _ = MF.load(num)
    Pad = MF.ADOPTED[str(num)]["P"]
    fc = 24.0 / Pad
    geo_rb = os.path.join(MF.RES_GEO, str(num), f"{num}_rebootstrap2.json")
    v = json.load(open(geo_rb if os.path.exists(geo_rb) else os.path.join(MF.RES, str(num), f"{num}_rebootstrap2.json"),
                       encoding="utf-8"))
    allc, prim = v["primary"] + v["extended"], v["primary"]
    rows = []
    for ds in np.unique(a["sheet"]):
        m = a["sheet"] == ds
        if np.sum(m) < 30:
            continue
        t, y = a["t"][m], a["y"][m]
        g = np.linspace(fc * 0.99, fc * 1.01, 3001)
        p = LombScargle(t, y, nterms=2).power(g, method="chi2")
        sub, src = v["subsets"].get(f"dataset:{ds}"), "rebootstrap2"
        if sub is None:          # datasets with fewer than 100 measurements: same exact-chi2 candidate test
            bb = max(float(t.max() - t.min()), 1.0)
            vals = [float(r2.power(t, y, r2.grid(c["f"], bb, 3.0, spp=20), exact=True).max()) for c in allc]
            sub = {"preferred_P": allc[int(np.argmax(vals))]["P"],
                   "preferred_primary_P": max(prim, key=lambda c: vals[allc.index(c)])["P"]}
            src = "exact"
        pp = sub["preferred_primary_P"] if Pad <= 240 else sub["preferred_P"]
        ok = in_family(pp, Pad) or (Pad > 240 and in_family(sub["preferred_primary_P"], Pad))
        rows.append({"num": int(num), "ds": str(ds), "n": int(np.sum(m)), "span_d": float(t.max() - t.min()),
                     "power": float(np.max(p)), "P_pref": float(pp), "P_pref_any": float(sub["preferred_P"]),
                     "P_pref_primary": float(sub["preferred_primary_P"]), "in_family": bool(ok), "pref_source": src})
    cands = sorted({float(c["P"]) for c in allc})
    print(num, len(rows), flush=True)
    return num, rows, cands


def main():
    nums = sorted(int(k) for k, v in MF.ADOPTED.items() if v.get("table") in (2, 3, 4))
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    ast, rows = {}, []
    with ProcessPoolExecutor(workers) as ex:
        for num, r, cands in ex.map(one, nums):
            ad = MF.ADOPTED[str(num)]
            ast[str(num)] = {"P": ad["P"], "quality": ad.get("quality"), "status": ad.get("status"),
                             "table": ad.get("table"), "candidates": cands}
            rows.extend(r)
    out = os.path.join(MF.ROOT, "dataset_power_table.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"asteroids": ast, "datasets": rows}, fh, indent=0)
    print("wrote", out, len(ast), "asteroids", len(rows), "datasets")


if __name__ == "__main__":
    main()
