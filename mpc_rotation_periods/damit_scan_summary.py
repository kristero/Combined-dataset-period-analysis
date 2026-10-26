"""
Interim summary from the period scans alone (damit_runs/*/scan/out_periods):
  - uniqueness of the scan minimum: chi2 of the best minimum further than dP = 0.5 P^2/T from the best one, relative
    to chi2_min, against chi2_tr = 1 + 0.5 sqrt(2/nu) (Hanus et al. 2021; nu = N - 55)
  - for asteroids with published sidereal periods (LCDB spin-axis file, within 0.5 % of our period): distance of the
    DAMIT scan period and of the periodogram period of the MPC data (P_mpc) from the published period, in units of dP
Output: damit_scan_summary.json
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(ROOT, "damit_runs")
poles = json.load(open(os.path.join(ROOT, "data", "lcdb_poles.json")))
rows, uniq = [], {}
for d in sorted(os.listdir(R)):
    run = os.path.join(R, d)
    if not os.path.exists(os.path.join(run, "scan", "done")):
        continue
    info = json.load(open(os.path.join(run, "prep.json")))
    a = np.loadtxt(os.path.join(run, "scan", "out_periods"), ndmin=2)
    a = a[np.isfinite(a[:, 1])]
    i = int(np.argmin(a[:, 1]))
    far = np.abs(a[:, 0] - a[i, 0]) > info["dP"]
    thr = 1 + 0.5 * np.sqrt(2 / (info["n"] - 55))
    ratio = float((a[far, 1].min() / a[i, 1]) ** 2) if far.any() else np.nan
    uniq[d] = {"P_scan": float(a[i, 0]), "ratio": ratio, "threshold": thr, "unique": bool(ratio > thr),
               "dev_min": float(a[i, 1]), "dev_median": float(np.median(a[:, 1]))}
    n = info["num"]
    if n in poles:
        sids = [e["sid_per"] for e in poles[n] if e.get("sid_per")]
        if sids:
            sid = min(sids, key=lambda s: abs(s - info["P0"]))
            if abs(sid - info["P0"]) / sid <= 0.005:
                rows.append({"run": d, "sid": sid, "P_mpc": info["P0"], "P_scan": float(a[i, 0]), "dP": info["dP"],
                             "mpc_off_dP": abs(info["P0"] - sid) / info["dP"],
                             "scan_off_dP": abs(a[i, 0] - sid) / info["dP"], "scan_unique": uniq[d]["unique"]})
S = {"n_scans": len(uniq),
     "frac_unique": float(np.mean([u["unique"] for u in uniq.values()])),
     "frac_unique_val": float(np.mean([u["unique"] for k, u in uniq.items() if k.startswith("val")])),
     "frac_unique_paper": float(np.mean([u["unique"] for k, u in uniq.items() if k.startswith("paper")])),
     "sidereal": {"n": len(rows),
                  "scan_within_1dP": int(sum(r["scan_off_dP"] <= 1 for r in rows)),
                  "mpc_within_1dP": int(sum(r["mpc_off_dP"] <= 1 for r in rows)),
                  "scan_within_1dP_unique": [int(sum(r["scan_off_dP"] <= 1 for r in rows if r["scan_unique"])),
                                             int(sum(r["scan_unique"] for r in rows))],
                  "rows": rows},
     "per_run": uniq}
json.dump(S, open(os.path.join(ROOT, "damit_scan_summary.json"), "w"), indent=1)
print({k: v for k, v in S.items() if k not in ("per_run",)} | {"sidereal": {k: v for k, v in S["sidereal"].items() if k != "rows"}})
for r in rows:
    print(f"{r['run']:12s} sid {r['sid']:.6f}  P_mpc {r['P_mpc']:.6f} ({r['mpc_off_dP']:.1f} dP)  "
          f"DAMIT {r['P_scan']:.6f} ({r['scan_off_dP']:.1f} dP)  unique={r['scan_unique']}")
