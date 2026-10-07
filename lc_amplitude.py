"""
Light-curve amplitude at a fixed period from the combined (reduced) magnitudes.

fourier_amplitude : peak-to-peak of an n-term Fourier series (default n = 4) fitted to all data folded at
                    the period, after iterative 3-sigma clipping; uncertainty from a bootstrap over
                    observing nights (one night of one dataset is resampled as a block).
template_amplitude: per-apparition amplitude. The data are split into apparitions at gaps longer than
                    60 days; an apparition is used if it has at least 60 measurements covering at least
                    8 of 10 intervals of rotational phase. A common light-curve shape S (n-term Fourier
                    series with unit peak-to-peak amplitude) is fitted to every apparition j as
                    y = Z_dj + g_j S(phi - dphi_j), with one zero point per dataset and apparition,
                    amplitude g_j >= 0 and phase shift dphi_j (grid search). S is then re-estimated from
                    the phase-aligned data of all apparitions and the fit is repeated (3 iterations).
                    A = mean of g_j weighted by the number of measurements; its uncertainty is the
                    weighted standard deviation of g_j divided by sqrt(number of apparitions).
                    This follows single-apparition Fourier fits of survey photometry (Waszczak et al. 2015,
                    AJ 150, 75; Chang et al. 2015, ApJS 219, 27), with a common shape to reduce the noise bias.
"""
from __future__ import annotations

import numpy as np

GAP_DAYS = 60.0
MIN_N_APP = 60


def _design(phase, nterms):
    cols = [f(2 * np.pi * k * phase) for k in range(1, nterms + 1) for f in (np.sin, np.cos)]
    return np.vstack(cols).T


def _ptp(coef, nterms):
    ph = np.linspace(0.0, 1.0, 1000, endpoint=False)
    m = _design(ph, nterms) @ coef
    return float(m.max() - m.min()), ph, m


def _fit_clipped(X, y, clip=3.0, iters=6):
    keep = np.ones(len(y), bool)
    c = None
    for _ in range(iters):
        c, *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
        r = y - X @ c
        s = np.std(r[keep], ddof=X.shape[1])
        new = np.abs(r) <= clip * s
        if np.array_equal(new, keep):
            break
        keep = new
    return c, keep


def fourier_amplitude(t, y, period_hours, nterms=4, sheet=None, n_boot=500, seed=1):
    """Peak-to-peak amplitude of the n-term Fourier fit of the folded data, with a night-block bootstrap error.
    t in days (any zero point), y in mag. Returns dict with amplitude, error, model phase/mag and kept mask."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    phase = (t * 24.0 / period_hours) % 1.0
    X = np.hstack([np.ones((len(t), 1)), _design(phase, nterms)])
    c, keep = _fit_clipped(X, y)
    amp, ph_grid, model = _ptp(c[1:], nterms)
    sheet = np.asarray(sheet if sheet is not None else np.zeros(len(t)), dtype=str)
    blocks = np.array([f"{s}|{int(np.floor(x))}" for s, x in zip(sheet, t)])
    ub, inv = np.unique(blocks[keep], return_inverse=True)
    idx_by_block = [np.where(inv == b)[0] for b in range(len(ub))]
    Xk, yk = X[keep], y[keep]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(int(n_boot)):
        pick = rng.integers(0, len(ub), len(ub))
        sel = np.concatenate([idx_by_block[b] for b in pick])
        cb, *_ = np.linalg.lstsq(Xk[sel], yk[sel], rcond=None)
        boots.append(_ptp(cb[1:], nterms)[0])
    return {"amplitude": amp, "amplitude_err": float(np.std(boots, ddof=1)), "nterms": nterms,
            "model_phase": ph_grid, "model_mag": model + c[0], "keep": keep, "coef": c,
            "n_used": int(keep.sum()), "n_clipped": int((~keep).sum())}


def _apparitions(t):
    o = np.argsort(t)
    cuts = np.where(np.diff(t[o]) > GAP_DAYS)[0]
    return [g for g in np.split(o, cuts + 1) if len(g) >= MIN_N_APP]


def template_amplitude(t, y, sheet, period_hours, nterms=4, n_iter=3, n_grid=200):
    """Per-apparition template amplitude (see module docstring). Returns None if no apparition qualifies."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    sheet = np.asarray(sheet, dtype=str)
    ph0 = (t * 24.0 / period_hours) % 1.0
    wins = [g for g in _apparitions(t) if len(np.unique(np.minimum((ph0[g] * 10).astype(int), 9))) >= 8]
    if not wins:
        return None
    X = np.hstack([np.ones((len(t), 1)), _design(ph0, nterms)])
    c, _ = _fit_clipped(X, y)
    a0 = _ptp(c[1:], nterms)[0]
    coef = c[1:] / a0
    grid = np.arange(n_grid) / n_grid
    shifts = np.zeros(len(wins))
    rows = []
    for it in range(n_iter + 1):
        rows = []
        for j, g in enumerate(wins):
            sg = sheet[g]
            sets = list(np.unique(sg))
            Z = np.vstack([(sg == s).astype(float) for s in sets]).T
            B = _design(ph0[g], nterms)
            # template shifted by every grid value: S(phi - d) has rotated Fourier coefficients
            C = np.zeros((2 * nterms, n_grid))
            for k in range(1, nterms + 1):
                a_k, b_k = coef[2 * (k - 1)], coef[2 * (k - 1) + 1]
                cd, sd = np.cos(2 * np.pi * k * grid), np.sin(2 * np.pi * k * grid)
                C[2 * (k - 1)] = a_k * cd + b_k * sd
                C[2 * (k - 1) + 1] = b_k * cd - a_k * sd
            F = B @ C
            yr, Fr = y[g].copy(), F.copy()
            for s in sets:                       # remove the zero points (per-dataset means)
                m = sg == s
                yr[m] -= yr[m].mean()
                Fr[m] -= Fr[m].mean(axis=0)
            num = Fr.T @ yr
            den = np.einsum("ij,ij->j", Fr, Fr)
            gg = num / den
            ss = yr @ yr - gg * num
            ss[gg <= 0] = np.inf
            jb = int(np.argmin(ss))
            Xg = np.hstack([Z, F[:, [jb]]])
            cg, *_ = np.linalg.lstsq(Xg, y[g], rcond=None)
            r = y[g] - Xg @ cg
            s_ = np.sqrt(float(r @ r) / max(len(g) - Xg.shape[1], 1))
            k = np.abs(r) <= 3 * s_
            cg, *_ = np.linalg.lstsq(Xg[k], y[g][k], rcond=None)
            shifts[j] = grid[jb]
            rows.append({"n": int(len(g)), "g": float(cg[-1]), "dphi": float(grid[jb]),
                         "zp": dict(zip(sets, cg[:-1])), "t_mid": float(np.median(t[g]))})
        if it == n_iter:
            break
        ta, ya = [], []
        for j, g in enumerate(wins):
            zp = rows[j]["zp"]
            ta.append((ph0[g] - shifts[j]) % 1.0)
            ya.append((y[g] - np.array([zp[s] for s in sheet[g]])) / rows[j]["g"])
        ta, ya = np.concatenate(ta), np.concatenate(ya)
        Xa = np.hstack([np.ones((len(ta), 1)), _design(ta, nterms)])
        ca, _ = _fit_clipped(Xa, ya)
        coef = ca[1:] / _ptp(ca[1:], nterms)[0]
    gs = np.array([r["g"] for r in rows])
    w = np.array([r["n"] for r in rows], float)
    A = float(np.sum(w * gs) / np.sum(w))
    sd = float(np.sqrt(np.sum(w * (gs - A) ** 2) / np.sum(w)))
    for r in rows:
        r.pop("zp")
    return {"amplitude": A, "amplitude_err": sd / np.sqrt(len(gs)), "n_apparitions": int(len(gs)),
            "apparitions": rows, "template_coef": coef.tolist()}
