"""
Per-object forward-model correction of the noise bias of the per-apparition Fourier amplitude (APP_FS_med).
For every asteroid, its own light-curve shape (order-4 fit of the combined data, normalized to unit peak-to-peak) is
injected with amplitudes A_true on a grid into noise realizations built from its own residuals (amp_injection2.residuals:
noise level, outliers and night-to-night offsets kept, coherence with the rotation phase removed), and APP_FS_med is
measured. The median measured value m(A_true) is inverted at the observed APP_FS_med; the 16-84 % spread of the
simulations at that point gives the uncertainty. If the observed value is below m(A_min) the amplitude is an upper limit.
No literature amplitude enters this correction.
Output: amp_debias.json
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
GRID = np.array([0.03, 0.06, 0.10, 0.15, 0.20, 0.30, 0.45, 0.65, 0.90, 1.20])
N_REAL = 3


def run(job):
    import amp_data as D
    import amp_methods as M
    import amp_injection2 as I
    s, n = job
    d = D.load(s, n)
    obs = M.estimate(d)
    a_obs = obs["APP_FS_med"]
    X = np.hstack([np.ones((len(d["t"]), 1)), M.fourier_design(d["t"], d["P"], 4)])
    c, *_ = M.fit_clipped(X, d["y"])
    a4, _ = M.ptp_coef(c[1:], 4)
    shape_c = c[1:] / a4
    sims = np.full((len(GRID), N_REAL), np.nan)
    for k in range(N_REAL):
        dd = dict(d)
        dd["num"] = str(int(n) + 100000 * k)       # different permutation seed per realization
        res = I.residuals(dd)
        Xs = M.fourier_design(d["t"], d["P"], 4)
        for i, A in enumerate(GRID):
            dd["y"] = res + A * (Xs @ shape_c)
            sims[i, k] = M.estimate(dd)["APP_FS_med"]
    med = np.nanmedian(sims, axis=1)
    lo, hi = np.nanpercentile(sims, 16, axis=1), np.nanpercentile(sims, 84, axis=1)
    out = {"APP_FS_med_obs": a_obs, "grid": GRID.tolist(), "sim_median": med.tolist(), "sim_lo": lo.tolist(),
           "sim_hi": hi.tolist(), "COMB_F4": obs["COMB_F4"], "n_app": obs["n_app"]}
    if not np.isfinite(a_obs):
        out["A_corr"] = np.nan
    elif a_obs <= med[0]:
        out.update({"A_corr": float(GRID[0]), "upper_limit": True})
    else:
        mono = np.maximum.accumulate(med)          # enforce a monotonic mapping
        out["A_corr"] = float(np.interp(a_obs, mono, GRID))
        out["A_corr_lo"] = float(np.interp(a_obs, np.maximum.accumulate(hi), GRID))
        out["A_corr_hi"] = float(np.interp(a_obs, np.maximum.accumulate(lo), GRID))
        out["upper_limit"] = False
    return f"{s}:{n}", out


def main():
    import amp_data as D
    with mp.get_context("spawn").Pool(16) as pool:
        res = dict(pool.imap_unordered(run, D.all_ids()))
    json.dump(res, open(os.path.join(ROOT, "amp_debias.json"), "w"), indent=1)
    print("done", len(res), "upper limits:", sum(bool(r.get("upper_limit")) for r in res.values()))


if __name__ == "__main__":
    main()
