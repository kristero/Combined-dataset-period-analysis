"""
Injection test of the amplitude estimators: for every asteroid, a light curve of known shape and amplitude is added
to the real residuals (data minus the order-4 Fourier model at the adopted period), at the real observation times,
so that the real sampling, noise, outliers, and night-to-night correlations are kept, but the true amplitude is known.
Templates: unit-amplitude order-8 Fourier shapes of three well-observed asteroids (2150, 4303, 1820) and a pure
second harmonic; amplitudes 0.10, 0.25, 0.50 mag; random phase offset. The estimators and their noise floors are
those of amplitude_study.py. Output: amplitude_injection.json (recovered / true amplitude).
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os

import numpy as np

import amplitude_study as S

ROOT = os.path.dirname(os.path.abspath(__file__))
A = S.A
AMPS = (0.10, 0.25, 0.50)


def template_coefs():
    T = {}
    for num in ("2150", "4303", "1820"):
        a = np.load(os.path.join(ROOT, "results", num, f"{num}_arrays.npz"))
        coef, _, _ = S.fourier(a["t"], a["y"], A[num]["P"], 8)
        T[num] = coef / S.ptp_of(coef, 8)
    c = np.zeros(17)
    c[3] = 0.5          # pure second harmonic: sin(2 w t) with unit peak-to-peak
    T["harm2"] = c
    return T


def shape(coef, ph):
    N = (len(coef) - 1) // 2
    G = np.vstack([np.ones_like(ph)] + [f(2 * np.pi * k * ph) for k in range(1, N + 1) for f in (np.sin, np.cos)]).T
    return G @ coef


def run(job):
    num, T = job
    a = np.load(os.path.join(ROOT, "results", num, f"{num}_arrays.npz"))
    t, y = a["t"].astype(float), a["y"].astype(float)
    P = A[num]["P"]
    base = float(t.max() - t.min())
    _, _, m4 = S.fourier(t, y, P, 4)
    res = y - m4
    noise_floor = json.load(open(os.path.join(ROOT, "amplitude_study.json"), encoding="utf-8"))[num]["noise_floor"]
    rng = np.random.default_rng(1000 + int(num))
    out = []
    for tname, coef in T.items():
        for amp in AMPS:
            ph = (t * 24.0 / P + rng.uniform()) % 1.0
            ysim = res + amp * shape(coef, ph)
            est = S.estimators(t, ysim, P, base, refine_period=False)
            deb = {k + "d": float(np.sqrt(max(est[k] ** 2 - noise_floor[k] ** 2, 0.0))) for k in noise_floor}
            app = S.per_apparition(t, ysim, P, rng)
            pa = {}
            for N in (2, 4):
                pa[f"ALN{N}"] = S.aligned_amplitude(t, ysim, P, N=N)[0]
                vals = [r[f"A{N}d"] for r in app]
                pa[f"A{N}max"] = float(max(vals)) if vals else np.nan
                pa[f"A{N}med"] = float(np.median(vals)) if vals else np.nan
                rows = S.per_apparition_null(t, ysim, P, N=N)
                v = np.array([r["Ad"] for r in rows])
                w = np.array([r["n"] for r in rows], float)
                pa[f"N{N}med"] = float(np.median(v)) if len(v) else np.nan
                pa[f"N{N}max"] = float(v.max()) if len(v) else np.nan
                pa[f"N{N}p75"] = float(np.percentile(v, 75)) if len(v) else np.nan
                pa[f"N{N}wmean"] = float(np.sum(w * v) / np.sum(w)) if len(v) else np.nan
            rec = {k: v / amp for k, v in {**est, **deb, **pa}.items() if not k.startswith("P_")}
            out.append({"template": tname, "amp": amp, "ratio": rec})
    return num, out


def main():
    T = template_coefs()
    nums = sorted(A, key=int)
    with mp.get_context("spawn").Pool(16) as pool:
        res = dict(pool.imap_unordered(run, [(n, T) for n in nums]))
    json.dump(res, open(os.path.join(ROOT, "amplitude_injection.json"), "w"), indent=1)
    keys = list(res[nums[0]][0]["ratio"])
    print(f"{'estimator':10s} " + "  ".join(f"A={a:.2f}: med (16-84%)" for a in AMPS))
    summary = {}
    for k in keys:
        line, summary[k] = [], {}
        for amp in AMPS:
            v = np.array([r["ratio"][k] for n in nums for r in res[n] if r["amp"] == amp and np.isfinite(r["ratio"][k])])
            q = np.percentile(v, [16, 50, 84])
            summary[k][str(amp)] = [float(x) for x in q]
            line.append(f"{q[1]:.2f} ({q[0]:.2f}-{q[2]:.2f})")
        print(f"{k:10s} " + "   ".join(line))
    json.dump(summary, open(os.path.join(ROOT, "amplitude_injection_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
