"""
Injection test for the amp_methods estimators. For every asteroid (paper + validation samples) the rotational signal
is removed apparition by apparition (residuals of the per-apparition Fourier fit with per-dataset zero points; data
outside usable apparitions: residuals of the combined order-4 fit), so the real sampling, noise, outliers and
night-to-night correlations remain. A light curve of known shape and amplitude is added at the rotation period
(templates: unit peak-to-peak order-8 shapes of 2150, 4303 and 1820, and a pure second harmonic; amplitudes 0.10,
0.25, 0.50 mag) and every estimator is applied. Note: removing a fitted model also removes a little noise
(parameters absorb it), so the residual noise is slightly underestimated in small apparitions.
Output: amp_injection2.json (recovered / injected).
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
AMPS = (0.10, 0.25, 0.50)
KEYS = ("COMB_F4", "APP_TPL_med", "APP_TPL_wmean", "APP_TPL_max10", "APP_FS_med", "APP_FS_max", "APP_VAR_med",
        "APP_VAR_max", "APP_VARs_med", "APP_VARs_max")


def templates():
    import amp_data as D
    import amp_methods as M
    T = {}
    for n in ("2150", "4303", "1820"):
        d = D.load("paper", n)
        X = np.hstack([np.ones((len(d["t"]), 1)), M.fourier_design(d["t"], d["P"], 8)])
        c, *_ = M.fit_clipped(X, d["y"])
        a, _ = M.ptp_coef(c[1:], 8)
        T[n] = c[1:] / a
    h = np.zeros(16)
    h[2] = 0.5           # sin(2 w t) with unit peak-to-peak (columns: sin1, cos1, sin2, ...)
    T["harm2"] = h
    return T


def residuals(d):
    """Residuals without the rotational signal, re-ordered so that the noise is incoherent with the period:
    removing a fitted model at P also removes the noise components at the harmonics of P, which would make the
    test unfairly favourable to Fourier estimators. Within every apparition and dataset the night means of the
    residuals are permuted among the nights and the deviations from the night means among the measurements, after
    scaling by sqrt(n / (n - p)) for the p fitted parameters. This keeps the noise level, its outliers and the
    night-to-night (zero-point) component, but removes any coherence with the rotation phase."""
    import amp_methods as M
    t, y, sheet, P = d["t"], d["y"], d["sheet"], d["P"]
    rng = np.random.default_rng(12345 + int(d["num"]))
    X = np.hstack([np.ones((len(t), 1)), M.fourier_design(t, P, 4)])
    c, _, r, _ = M.fit_clipped(X, y)
    res = r * np.sqrt(len(t) / (len(t) - X.shape[1]))
    blocks = [np.arange(len(t))]
    for g in M.apparitions(t):
        sg = sheet[g]
        sets = list(np.unique(sg))
        Z = np.vstack([(sg == s).astype(float) for s in sets]).T
        Xg = np.hstack([Z, M.fourier_design(t[g], P, 4)])
        cg, _, rg, _ = M.fit_clipped(Xg, y[g])
        res[g] = rg * np.sqrt(len(g) / max(len(g) - Xg.shape[1], 1))
        blocks.append(g)
    out = res.copy()
    in_app = np.zeros(len(t), bool)
    for g in blocks[1:]:
        in_app[g] = True
    for g in blocks[1:] + [np.where(~in_app)[0]]:
        for s in np.unique(sheet[g]):
            m = g[sheet[g] == s]
            night = np.floor(t[m]).astype(int)
            nights = np.unique(night)
            means = {k: np.mean(res[m][night == k]) for k in nights}
            dev = res[m] - np.array([means[k] for k in night])
            perm_n = dict(zip(nights, rng.permutation([means[k] for k in nights])))
            out[m] = np.array([perm_n[k] for k in night]) + rng.permutation(dev)
    return out


def run(job):
    import amp_data as D
    import amp_methods as M
    (s, n), T = job
    d = D.load(s, n)
    res = residuals(d)
    rng = np.random.default_rng(int(n))
    out = []
    for name, c in T.items():
        for amp in AMPS:
            off = rng.uniform()
            ph = (d["t"] * 24.0 / d["P"] + off) % 1.0
            X = np.vstack([f(2 * np.pi * k * ph) for k in range(1, 9) for f in (np.sin, np.cos)]).T
            dd = dict(d)
            dd["y"] = res + amp * (X @ c)
            e = M.estimate(dd)
            out.append({"template": name, "amp": amp, "ratio": {k: e[k] / amp for k in KEYS}})
    return f"{s}:{n}", out


def main():
    import amp_data as D
    T = templates()
    with mp.get_context("spawn").Pool(16) as pool:
        res = dict(pool.imap_unordered(run, [(i, T) for i in D.all_ids()]))
    json.dump(res, open(os.path.join(ROOT, "amp_injection2.json"), "w"), indent=1)
    print(f"{'estimator':14s} " + "   ".join(f"A={a:.2f}: median (16-84%)" for a in AMPS))
    summ = {}
    for k in KEYS:
        line, summ[k] = [], {}
        for a in AMPS:
            v = np.array([r["ratio"][k] for rows in res.values() for r in rows if r["amp"] == a and np.isfinite(r["ratio"][k])])
            q = np.percentile(v, [16, 50, 84])
            summ[k][str(a)] = [float(x) for x in q]
            line.append(f"{q[1]:.2f} ({q[0]:.2f}-{q[2]:.2f})")
        print(f"{k:14s} " + "    ".join(line))
    json.dump(summ, open(os.path.join(ROOT, "amp_injection2_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
