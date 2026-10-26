"""
Which amplitude estimator reproduces the literature amplitudes best? All estimators work on the phase-reduced
combined data of results/<N>/<N>_arrays.npz at the adopted period (adopted.json); literature values are the LCDB
amplitudes (data/lcdb_amplitudes.json, from lcdb_amplitudes.py).

Estimators
  F<N>       peak-to-peak of a Fourier series of order N (2, 4, 6, 8) fitted at the adopted period, after iterative
             3-sigma clipping about the model of the same order
  F<N>r      the same at the period refined with the order-N Lomb-Scargle periodogram within +/-2 resolution
             elements of the adopted period
  B<w>       max - min of bin means in bins of w degrees of rotational phase (2, 5, 10, 18), 3-sigma clipping
             inside each bin (B18 with clipping about the 2-term model is the estimator of the submitted paper: "B18m2")
  B<w>e      max(mean + error) - min(mean - error) of the same bins (upper envelope)
  A<N>max / A<N>med   per-apparition amplitudes: every apparition (gap > 60 d) with >= 80 measurements and
             >= 8 of 10 phase bins covered is fitted separately with an order-N Fourier series (phase free);
             maximum and median over the apparitions
Noise floor: every estimator is also applied to pure noise (the residuals about the order-4 model, permuted in
phase), median of 20 realizations; "...d" = debiased estimate sqrt(max(A^2 - A_noise^2, 0)).
Output: amplitude_study.json
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
A = json.load(open(os.path.join(ROOT, "adopted.json"), encoding="utf-8"))
N_NOISE = 20
ORDERS = (2, 4, 6, 8)
BINS = (2, 5, 10, 18)


def design(t, P, N):
    w = 2 * np.pi * 24.0 / P
    return np.vstack([np.ones_like(t)] + [f(k * w * t) for k in range(1, N + 1) for f in (np.sin, np.cos)]).T


def ptp_of(coef, N):
    ph = np.linspace(0, 1, 2000, endpoint=False)
    G = np.vstack([np.ones_like(ph)] + [f(2 * np.pi * k * ph) for k in range(1, N + 1) for f in (np.sin, np.cos)]).T
    m = G @ coef
    return float(m.max() - m.min())


def fourier(t, y, P, N, clip=3.0):
    X = design(t, P, N)
    keep = np.ones(len(y), bool)
    for _ in range(6):
        coef, *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
        r = y - X @ coef
        s = np.std(r[keep], ddof=X.shape[1])
        new = np.abs(r) <= clip * s
        if np.array_equal(new, keep):
            break
        keep = new
    return coef, keep, X @ coef


def fourier_ptp(t, y, P, N):
    coef, keep, _ = fourier(t, y, P, N)
    return ptp_of(coef, N)


def binned_amp(phase, y, w_deg, clip=3.0, min_n=5):
    nb = int(round(360 / w_deg))
    idx = np.minimum((phase * nb).astype(int), nb - 1)
    means, errs = [], []
    for b in range(nb):
        v = y[idx == b]
        for _ in range(5):
            if len(v) < min_n:
                break
            med, s = np.median(v), np.std(v, ddof=1)
            nv = v[np.abs(v - med) <= clip * s]
            if len(nv) == len(v):
                break
            v = nv
        if len(v) >= min_n:
            means.append(np.mean(v))
            errs.append(np.std(v, ddof=1) / np.sqrt(len(v)))
    if len(means) < 0.6 * nb:
        return np.nan, np.nan
    means, errs = np.array(means), np.array(errs)
    return float(means.max() - means.min()), float((means + errs).max() - (means - errs).min())


def b18_m2(t, y, P):
    """Estimator of the submitted paper: 3-sigma clipping about the 2-term model, max - min of 18-deg bin means."""
    coef, _, model = fourier(t, y, P, 2, clip=1e9)
    res = y - model
    keep = np.ones(len(y), bool)
    for _ in range(5):
        s = np.std(res[keep], ddof=1)
        new = np.abs(res - np.median(res[keep])) <= 3.0 * s
        if np.array_equal(new, keep):
            break
        keep = new
    ph = (t * 24.0 / P) % 1.0
    m = [np.mean(y[keep & (ph >= a) & (ph < a + 0.05)]) for a in np.arange(0, 1, 0.05)
         if np.sum(keep & (ph >= a) & (ph < a + 0.05)) >= 5]
    return float(max(m) - min(m))


def refine(t, y, P, N, base):
    from astropy.timeseries import LombScargle
    f0 = 24.0 / P
    g = np.linspace(f0 - 2 / base, f0 + 2 / base, 161)
    p = LombScargle(t, y, nterms=N).power(g, method="fastchi2", assume_regular_frequency=True)
    return 24.0 / g[int(np.argmax(p))]


def estimators(t, y, P, base, refine_period=True):
    out = {}
    ph = (t * 24.0 / P) % 1.0
    for N in ORDERS:
        out[f"F{N}"] = fourier_ptp(t, y, P, N)
        if refine_period:
            Pr = refine(t, y, P, N, base)
            out[f"F{N}r"] = fourier_ptp(t, y, Pr, N)
            out[f"P_F{N}r"] = Pr
    for w in BINS:
        a, ae = binned_amp(ph, y, w)
        out[f"B{w}"], out[f"B{w}e"] = a, ae
    out["B18m2"] = b18_m2(t, y, P)
    return out


def apparitions(t, min_n=80, gap=60.0):
    o = np.argsort(t)
    cuts = np.where(np.diff(t[o]) > gap)[0]
    groups = np.split(o, cuts + 1)
    return [g for g in groups if len(g) >= min_n]


def per_apparition(t, y, P, rng):
    rows = []
    for g in apparitions(t):
        tg, yg = t[g], y[g]
        ph = (tg * 24.0 / P) % 1.0
        cov = len(np.unique(np.minimum((ph * 10).astype(int), 9)))
        if cov < 8:
            continue
        r = {"t_mid": float(np.median(tg)), "n": int(len(g)), "span_d": float(tg.max() - tg.min())}
        for N in (2, 4):
            coef, keep, model = fourier(tg, yg, P, N)
            a = ptp_of(coef, N)
            res = yg - model
            noise = [ptp_of(fourier(tg, rng.permutation(res), P, N)[0], N) for _ in range(N_NOISE)]
            an = float(np.median(noise))
            r[f"A{N}"], r[f"A{N}_noise"], r[f"A{N}d"] = a, an, float(np.sqrt(max(a * a - an * an, 0.0)))
        rows.append(r)
    return rows


def per_apparition_null(t, y, P, N=4, K=24, min_n=80):
    """Per-apparition order-N amplitudes with a noise floor that keeps the real time structure: the order-N model is
    fitted to the residuals (apparition data minus its order-N model at P), at their real times, at K null periods
    whose frequencies are offset by 3-12 resolution elements of the apparition from the rotation frequency (well
    inside the +/-0.5 c/d spacing of the daily aliases). Debiased amplitude sqrt(A^2 - A_null^2)."""
    rows = []
    for g in apparitions(t, min_n=min_n):
        tg, yg = t[g], y[g]
        ph = (tg * 24.0 / P) % 1.0
        if len(np.unique(np.minimum((ph * 10).astype(int), 9))) < 8:
            continue
        span = max(float(tg.max() - tg.min()), 5.0)
        coef, keep, model = fourier(tg, yg, P, N)
        a = ptp_of(coef, N)
        res = (yg - model)[keep]
        f0 = 24.0 / P
        offs = np.concatenate([np.linspace(3, 12, K // 2), -np.linspace(3, 12, K // 2)]) / span
        null = []
        for df in offs:
            f = f0 + df
            if f <= 0:
                continue
            c, _, _ = fourier(tg[keep], res, 24.0 / f, N)
            null.append(ptp_of(c, N))
        an = float(np.median(null))
        rows.append({"t_mid": float(np.median(tg)), "n": int(len(g)), "A": a, "A_null": an,
                     "Ad": float(np.sqrt(max(a * a - an * an, 0.0)))})
    return rows


def aligned_amplitude(t, y, P, N=4, min_n=80, n_iter=4, n_grid=72):
    """Amplitude of the common light curve after aligning the apparitions: the shifts of the light-curve phase between
    apparitions (changing viewing geometry, synodic-sidereal drift) smear the combined fold and lower its amplitude.
    Every apparition with >= min_n points and >= 8 of 10 phase bins gets a free phase shift and zero point (grid
    search against the current common order-N shape, linear least squares for zero point and scale); the common
    shape is refitted to the aligned data; iterated. Returns the peak-to-peak of the final common shape and the shifts."""
    ph0 = (t * 24.0 / P) % 1.0
    groups = []
    for g in apparitions(t, min_n=min_n):
        if len(np.unique(np.minimum((ph0[g] * 10).astype(int), 9))) >= 8:
            groups.append(g)
    shift = np.zeros(len(t))
    zp = np.zeros(len(t))
    grid = np.arange(n_grid) / n_grid
    deltas = [0.0] * len(groups)
    for _ in range(n_iter):
        ph = (ph0 + shift) % 1.0
        coef, keep, _ = fourier(ph * P / 24.0, y - zp, P, N)       # time-like argument so that design() gives phase
        for i, g in enumerate(groups):
            best = None
            for d in grid:
                Xs = design(((ph0[g] + d) % 1.0) * P / 24.0, P, N) @ np.r_[0.0, coef[1:]]
                X = np.vstack([np.ones(len(g)), Xs]).T
                c, *_ = np.linalg.lstsq(X, y[g], rcond=None)
                ss = float(np.sum((y[g] - X @ c) ** 2))
                if best is None or ss < best[0]:
                    best = (ss, d, c[0])
            deltas[i] = best[1]
            shift[g] = best[1]
            zp[g] = best[2] - coef[0]
    ph = (ph0 + shift) % 1.0
    coef, keep, _ = fourier(ph * P / 24.0, y - zp, P, N)
    return ptp_of(coef, N), deltas


def run(num):
    a = np.load(os.path.join(ROOT, "results", num, f"{num}_arrays.npz"))
    t, y = a["t"].astype(float), a["y"].astype(float)
    P = A[num]["P"]
    base = float(t.max() - t.min())
    rng = np.random.default_rng(int(num))
    est = estimators(t, y, P, base)
    # noise floor of every estimator: residuals about the order-4 model, permuted
    _, _, m4 = fourier(t, y, P, 4)
    res = y - m4
    noise = {}
    for _ in range(N_NOISE):
        e = estimators(t, rng.permutation(res), P, base, refine_period=False)
        for k, v in e.items():
            noise.setdefault(k, []).append(v)
    noise = {k: float(np.nanmedian(v)) for k, v in noise.items()}
    deb = {k + "d": float(np.sqrt(max(est[k] ** 2 - noise[k] ** 2, 0.0))) for k in noise if np.isfinite(est.get(k, np.nan))}
    app = per_apparition(t, y, P, rng)
    summ = {}
    for N in (2, 4):
        vals = [r[f"A{N}d"] for r in app]
        summ[f"A{N}max"] = float(max(vals)) if vals else np.nan
        summ[f"A{N}med"] = float(np.median(vals)) if vals else np.nan
    for N in (2, 4):
        summ[f"ALN{N}"], summ[f"ALN{N}_shifts"] = aligned_amplitude(t, y, P, N=N)
    appn = {}
    for N in (2, 4):
        rows = per_apparition_null(t, y, P, N=N)
        appn[N] = rows
        v = np.array([r["Ad"] for r in rows])
        w = np.array([r["n"] for r in rows], float)
        summ[f"N{N}med"] = float(np.median(v)) if len(v) else np.nan
        summ[f"N{N}max"] = float(v.max()) if len(v) else np.nan
        summ[f"N{N}p75"] = float(np.percentile(v, 75)) if len(v) else np.nan
        summ[f"N{N}wmean"] = float(np.sum(w * v) / np.sum(w)) if len(v) else np.nan
    return num, {"P": P, "n": int(len(t)), "estimates": est, "noise_floor": noise, "debiased": deb,
                 "apparitions": app, "apparitions_null": appn, "per_apparition": summ, "n_apparitions": len(app)}


def main():
    nums = sorted(A, key=int)
    with mp.get_context("spawn").Pool(16) as pool:
        res = dict(pool.imap_unordered(run, nums))
    json.dump({k: res[k] for k in nums}, open(os.path.join(ROOT, "amplitude_study.json"), "w"), indent=1)
    print("done", len(res))


if __name__ == "__main__":
    main()
