"""
Effect of the 2018 zero-point jumps of stations 703 and G96 (station_zeropoint_check.py; Hoffmann et al. 2024) on
the aspect fit (amp_geometry). For every asteroid, the post-2018.5 measurements of 703 and G96 are shifted by the
asteroid's own jump estimate (mean over apparitions after minus before of median(y_station) - median(y_others) in
the same apparition; geometry cancels within an apparition), and the template fit + aspect fit are repeated.
Output: geometry_zeropoint_effect.json
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
EPOCH = 2458300.5
STATIONS = ("703", "G96")


def corrected(d, t0):
    import amp_methods as AM
    y = d["y"].copy()
    stn = np.array([s[:3] for s in d["sheet"].astype(str)])
    info = {}
    for s in STATIONS:
        pre, post = [], []
        for g in AM.apparitions(d["t"]):
            m, o = stn[g] == s, stn[g] != s
            if m.sum() >= 20 and o.sum() >= 20:
                dj = float(np.median(d["y"][g][m]) - np.median(d["y"][g][o]))
                (post if np.median(d["t"][g]) + t0 > EPOCH else pre).append(dj)
        if pre and post:
            jump = float(np.mean(post) - np.mean(pre))
            sel = (stn == s) & (d["t"] + t0 > EPOCH)
            y[sel] -= jump
            info[s] = {"jump": jump, "n_pre": len(pre), "n_post": len(post), "n_points": int(sel.sum())}
    return y, info


def one(job):
    import amp_data as D
    import amp_geometry as G
    import amp_methods as AM
    s, n = job
    d = D.load(s, n)
    a = np.load(os.path.join(ROOT, "results", n, f"{n}_arrays.npz")) if s == "paper" else \
        np.load(os.path.join(D.VAL, "results", f"{n}_arrays.npz"), allow_pickle=True)
    y, info = corrected(d, float(a["time_zero"]))
    if not info:
        return f"{s}:{n}", {"corrected": False}
    d2 = dict(d)
    d2["y"] = y
    tp = AM.template_fit(d2)
    par = G.fit(tp["rows"]) if tp else None
    return f"{s}:{n}", {"corrected": True, "jumps": info,
                        "geometry": None if par is None else {k: par[k] for k in
                                                              ("lambda", "beta", "a_b", "b_c", "m", "delta_bic", "chi2_M", "chi2_A")},
                        "windows_mu": [r["mu"] for r in tp["rows"]] if tp else []}


def main():
    import amp_data as D
    import amp_geometry as G
    with mp.get_context("spawn").Pool(6) as pool:
        res = dict(pool.imap_unordered(one, D.all_ids()))
    GEO = json.load(open(os.path.join(ROOT, "geometry_results.json")))
    poles = json.load(open(os.path.join(ROOT, "data", "lcdb_poles.json")))
    cmp_ = []
    for k, r in res.items():
        g0 = GEO[k].get("geometry") if GEO.get(k) else None
        g1 = r.get("geometry") if r.get("corrected") else g0
        if g0 is None or g1 is None:
            continue
        n = k.split(":")[1]
        row = {"key": k, "corrected": r["corrected"], "dbic_before": g0["delta_bic"], "dbic_after": g1["delta_bic"],
               "pole_shift": G.pole_distance(g0["lambda"], g0["beta"], g1["lambda"], g1["beta"])}
        if n in poles:
            sols = [p for e in poles[n] for p in e["poles"]]
            row["pole_dist_before"] = min(G.pole_distance(g0["lambda"], g0["beta"], l, b) for l, b in sols)
            row["pole_dist_after"] = min(G.pole_distance(g1["lambda"], g1["beta"], l, b) for l, b in sols)
        cmp_.append(row)
    sig0 = sum(r["dbic_before"] > 10 for r in cmp_)
    sig1 = sum(r["dbic_after"] > 10 for r in cmp_)
    paper0 = sorted(r["key"] for r in cmp_ if r["key"].startswith("paper") and r["dbic_before"] > 10)
    paper1 = sorted(r["key"] for r in cmp_ if r["key"].startswith("paper") and r["dbic_after"] > 10)
    pd0 = [r["pole_dist_before"] for r in cmp_ if "pole_dist_before" in r]
    pd1 = [r["pole_dist_after"] for r in cmp_ if "pole_dist_after" in r]
    jumps = [v["jump"] for r in res.values() if r.get("corrected") for v in r["jumps"].values()]
    summ = {"n_corrected": sum(r["corrected"] for r in cmp_), "n_fitted": len(cmp_),
            "dbic_gt10_before": sig0, "dbic_gt10_after": sig1,
            "paper_dbic_gt10_before": paper0, "paper_dbic_gt10_after": paper1,
            "median_pole_shift_corrected": float(np.median([r["pole_shift"] for r in cmp_ if r["corrected"]])),
            "pole_median_before": float(np.median(pd0)) if pd0 else None,
            "pole_median_after": float(np.median(pd1)) if pd1 else None,
            "median_abs_jump": float(np.median(np.abs(jumps))) if jumps else None}
    json.dump({"summary": summ, "rows": cmp_, "per_asteroid": res}, open(os.path.join(ROOT, "geometry_zeropoint_effect.json"), "w"),
              indent=1, default=float)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
