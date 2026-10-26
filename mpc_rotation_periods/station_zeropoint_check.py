"""
Check for zero-point jumps of single stations (Hoffmann et al. 2024: 703 and G96 dropped in summer 2018 when the
reference catalogue changed from Gaia DR1 to DR2; G96 drifts again after 2022; 704 differs between galactic
hemispheres). For every asteroid and apparition (gap > 60 d) that contains the station and other datasets, the
offset d_j = median(y_station) - median(y_others) is formed (the geometry cancels within an apparition). A jump
appears as a difference between the mean offset of apparitions after and before the epoch.
Output: station_zeropoint_check.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import amp_data as D  # noqa: E402
import amp_methods as AM  # noqa: E402

EPOCHS = {"703": 2458300.5, "G96": 2458300.5, "E12": 2458300.5, "691": 2458300.5, "I41": 2458300.5,
          "T05": 2458300.5, "T08": 2458300.5, "F51": 2458300.5}


def main():
    rows = {s: [] for s in EPOCHS}
    for sample, num in D.all_ids():
        d = D.load(sample, num)
        a = np.load(os.path.join(ROOT, "results", num, f"{num}_arrays.npz")) if sample == "paper" else \
            np.load(os.path.join(D.VAL, "results", f"{num}_arrays.npz"), allow_pickle=True)
        t0 = float(a["time_zero"])
        sh = d["sheet"].astype(str)
        stn = np.array([s[:3] for s in sh])
        for g in AM.apparitions(d["t"]):
            jd_mid = float(np.median(d["t"][g])) + t0
            for s, ep in EPOCHS.items():
                m = stn[g] == s
                o = ~m
                if m.sum() >= 20 and o.sum() >= 20:
                    rows[s].append({"ast": f"{sample}:{num}", "jd": jd_mid, "post": jd_mid > ep,
                                    "d": float(np.median(d["y"][g][m]) - np.median(d["y"][g][o]))})
    out = {}
    for s, r in rows.items():
        per = {}
        for q in r:
            per.setdefault(q["ast"], {"pre": [], "post": []})["post" if q["post"] else "pre"].append(q["d"])
        jumps = [np.mean(v["post"]) - np.mean(v["pre"]) for v in per.values() if v["pre"] and v["post"]]
        if jumps:
            j = np.array(jumps)
            out[s] = {"n_asteroids": len(j), "median_jump": float(np.median(j)), "mean_jump": float(np.mean(j)),
                      "sem": float(np.std(j, ddof=1) / np.sqrt(len(j))) if len(j) > 1 else None,
                      "frac_negative": float(np.mean(j < 0))}
            print(s, out[s])
    json.dump(out, open(os.path.join(ROOT, "station_zeropoint_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
