"""
Common loader for the amplitude study: the combined, phase-reduced data of the paper sample (results/<N>/) and of
the validation sample (validation/results/), each measurement with its geometry (phase angle alpha, phase-angle
bisector longitude/latitude, r, Delta), the rotation period, and the LCDB amplitudes.

load(sample, num) -> dict(t, y, sheet, alpha, pab_lon, pab_lat, r, delta, P, quantized, lcdb)
  sample = "paper": P = adopted period; geometry matched from data/obs_tables by dataset and light-time-corrected time
  sample = "val":   P = LCDB period refined to the nearest periodogram peak (validation/build.py)
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
VAL = os.path.join(ROOT, "validation")
ADOPTED = json.load(open(os.path.join(ROOT, "adopted.json"), encoding="utf-8"))
LCDB_PAPER = json.load(open(os.path.join(ROOT, "data", "lcdb_amplitudes.json"), encoding="utf-8"))
VSAMPLE = {s["num"]: s for s in json.load(open(os.path.join(VAL, "sample.json"), encoding="utf-8"))["sample"]}
GEO = ("alpha", "pab_lon", "pab_lat", "r", "delta")


def _quantized(obs):
    q = set()
    for ds, g in obs.groupby("dataset"):
        m = g["mag_f"].to_numpy(float)
        if np.mean(np.abs(m * 10 - np.round(m * 10)) < 1e-6) >= 0.9:
            q.add(ds)
    return q


def _match_geometry(t, sheet, t0, obs):
    geo = {k: np.full(len(t), np.nan) for k in GEO}
    for ds in np.unique(sheet):
        o = obs[obs["dataset"] == ds].sort_values("jd_lt")
        m = np.where(sheet == ds)[0]
        jd = t[m] + t0
        x = o["jd_lt"].to_numpy()
        k = np.clip(np.searchsorted(x, jd), 1, len(o) - 1)
        kk = np.where(np.abs(jd - x[k - 1]) <= np.abs(jd - x[k]), k - 1, k)
        ok = np.abs(x[kk] - jd) < 1e-4
        for c in GEO:
            geo[c][m[ok]] = o[c].to_numpy()[kk[ok]]
    return geo


def load(sample, num):
    num = str(num)
    if sample == "paper":
        a = np.load(os.path.join(ROOT, "results", num, f"{num}_arrays.npz"))
        t, y, sheet, t0 = a["t"].astype(float), a["y"].astype(float), a["sheet"], float(a["time_zero"])
        obs = pd.read_csv(os.path.join(ROOT, "data", "obs_tables", f"{num}_obs.csv"))
        geo = _match_geometry(t, sheet, t0, obs)
        P = ADOPTED[num]["P"]
        l = LCDB_PAPER.get(num)
        lc = None
        if l and l["summary"]["amax"] is not None and l["summary"]["period"] and any(
                abs(l["summary"]["period"] - r * P) <= 0.03 * l["summary"]["period"] for r in (1, 2, 0.5)):
            amps = [d["amax"] for d in l["details"] if d["amax"] is not None and d["period"] and
                    abs(d["period"] - l["summary"]["period"]) <= 0.02 * l["summary"]["period"]]
            lc = {"amin": l["summary"]["amin"] if l["summary"]["amin"] is not None else l["summary"]["amax"],
                  "amax": l["summary"]["amax"], "ref_amplitudes": amps or [l["summary"]["amax"]]}
        info = {"snr": None, "quality": ADOPTED[num]["quality"]}
    else:
        a = np.load(os.path.join(VAL, "results", f"{num}_arrays.npz"), allow_pickle=True)   # our own files
        t, y, sheet, t0 = a["t"].astype(float), a["y"].astype(float), a["sheet"].astype(str), float(a["time_zero"])
        geo = {k: a[k].astype(float) for k in GEO}
        obs = pd.read_csv(os.path.join(VAL, "data", "obs_tables", f"{num}_obs.csv"), usecols=["dataset", "mag_f"])
        inf = json.load(open(os.path.join(VAL, "results", f"{num}_info.json"), encoding="utf-8"))
        P = inf.get("P_refined_wide", inf["P_refined"])     # strongest peak within +/-0.5 % of the LCDB period
        s = VSAMPLE[num]
        lc = {"amin": s["amin"] if s["amin"] is not None else s["amax"], "amax": s["amax"],
              "ref_amplitudes": s["ref_amplitudes"]}
        info = {"snr": inf["snr"], "quality": None}
    if sample == "paper":
        obs = obs[["dataset", "mag_f"]]
    return {"num": num, "sample": sample, "t": t, "y": y, "sheet": sheet, "P": P, **geo,
            "quantized": _quantized(obs), "lcdb": lc, **info}


def all_ids():
    return [("paper", n) for n in sorted(ADOPTED, key=int)] + [("val", n) for n in sorted(VSAMPLE, key=int)
                                                              if os.path.exists(os.path.join(VAL, "results", f"{n}_info.json"))]
