"""
(3216) Harrington: further tests of rotation periods below 400 h (complement to inspect_3216_short.py).

Question (co-author): are there credible periods below or around 200 h that are not due to the window function?

Data: results/<N>/<N>_arrays.npz (original) and results_geo/<N>/<N>_arrays.npz (aspect-corrected); 5444 points,
of which 3466 are TESS (sheet C57G, 10-min cadence in 7 segments of 0.1-10.4 d in 2020 and 2021).

Detrending variants (linear least squares, the fitted part subtracted; the residuals are searched):
  raw  mean only
  Q    quadratic in time per dataset and apparition (degree rule of longperiod_tests.py: 2 if span > 10 d and n > 20,
       1 if n > 5, else 0) - the "slow trend" model of the paper
  QF   Q + global Fourier series at 1266.55 h with 4 harmonics (1266.55, 633.3, 422.2, 316.6 h): prewhitening of the
       long-period signal
  QFA  QF + per-apparition sinusoids at 633.3 and 316.7 h (apparitions with n >= 100 and span >= 30 d): lets the
       slow signal change amplitude/phase between apparitions
  QFT  QF + a constant and a linear trend per TESS segment (gaps > 1 d)
Tests
  1 periodograms (Lomb-Scargle, nterms = 2, unweighted; fastchi2 on regular grids, exact chi2 for refinement)
    2-24 h (step 1/5T) and 24-400 h (step 1/20T) of every variant
  2 permutation null per variant: nightly means permuted among the nights of the same dataset and apparition and the
    deviations within each night shuffled, then the variant's detrending re-applied; distribution of the highest power
    in 24-400 h, 24-200 h and 2-24 h; global p-values and local p-values at pre-specified periods
  3 "slow signal + noise" null: fitted slow model of QF or QFT (no short period) + permuted residuals, searched without
    detrending - how much of the raw 24-400 h periodogram the slow signal and the sampling reproduce
  4 candidates (strongest families of raw/QF/QFT and the periods named in the review: 185.0, 187.8, 214.6, 217.7,
    221.7, 255.1 h): power in each variant and in the aspect-corrected data, p-values, spectral window at P (all data,
    TESS, non-TESS) and nearest strong window peak, harmonic relation to 631.5/633.3/1266.55 h, subsets (odd/even
    apparitions, ATLAS/other, TESS/other), single apparitions, cross-prediction TESS <-> other stations, 4-term fold
    (peak-to-peak amplitude, number of maxima, compared with folds at random periods)
  5 TESS alone (2-360 h): variants appmean (one mean per apparition), segmean (one per segment), seglin (constant +
    linear trend per segment); permutation null (days permuted within segments); the 2020 and 2021 TESS data
    separately; point-to-point noise; injection-recovery
  6 injection-recovery in the combined data: double-peaked light curves (P = 30-380 h, peak-to-peak 0.05-0.73 mag;
    0.73 mag = per-apparition amplitude of (3216)) added to the real data, searched raw, after QF and after QFT;
    recovered if the strongest peak in 24-400 h is within 1 % of P_inj
  7 checks: calibration of the local p-values at 300 a-priori random periods; the strongest 2-24 h peak after QF
    (daily-alias relations, subsets, prewhitening of the residual long periods); TESS scatter after the global slow
    model compared with per-segment polynomials
  8 (added in the second pass)
    a validation of the vectorised periodogram against astropy LombScargle (chi2 and fastchi2)
    b binned data: one point per dataset and night (TESS: 6-h bins), so that every night has equal weight and the
      3466 TESS points no longer dominate; raw, Q, QF, QFT; permutation null (bins permuted within each dataset and
      apparition, detrending re-applied); injection-recovery
    c ground-based data only (all stations except C57G): raw and QF periodograms with permutation nulls, and the
      all-data QF slow model sampled at the ground epochs + permuted ground residuals
    d slow models WITHOUT freedom below 400 h, sampled at the real epochs: M3 = one constant per dataset and
      apparition + Fourier series at 1266.55 h with 3 harmonics (1266.6, 633.3, 422.2 h); M4 = M3 + 316.6 h harmonic;
      QF3 = Q + 3 harmonics; Q = quadratics alone. Q-type models can follow structure on the time scale of short
      blocks (TESS 2020: 21 d), M3 cannot, so M3 is the non-circular test of the sidelobe explanation
    e injection at the observed amplitude: the 4-harmonic raw-data fold at each review period (peak-to-peak
      0.28-0.35 mag) added to the data at a random phase, then QF/QFT detrending; would such a signal survive?
    f QP = QF + one 2-harmonic Fourier series per apparition with its own period in 560-720 h (the slow signal is not
      strictly periodic), for all data, ground-only and binned data; permutation null with the periods held fixed;
      plus QFA for the binned and ground-only data and alias checks of their strongest residual peaks
    g TESS test of the short ground-based residual peaks (24.84, 24.9, 46.3 h): amplitude in TESS vs ground,
      cross-prediction ground -> TESS, the ground signal injected into the TESS data
    h raw power at the review periods in subsets (odd/even, ATLAS/other, TESS/ground) and single apparitions
Output: inspect_3216_more.json, review/inspect_3216_more.png, review/inspect_3216_more_damit.png (convex-inversion
        chi2 landscape of damit_runs/paper_3216/longrange on a log scale; the panel of inspect_3216_short.png is empty
        because all ratios in 120-400 h exceed its y-limit of 1.6)
Usage: python inspect_3216_more.py [--workers 4]   (intermediate results cached in %TEMP%/inspect_3216_more_cache)
"""
import os

for _v in ("MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import ls_longperiod as L  # noqa: E402

NUM = "3216"
P_SYN, P_HALF, P_SID = 1266.55, 633.32, 631.5
P_LONGS = (P_SID, P_HALF, P_SYN)
REVIEW_P = (185.0, 187.83, 214.6, 217.7, 221.7, 255.1)
VARIANTS = ("raw", "Q", "QF", "QFA", "QFT")
TESS = "C57G"
CACHE = os.path.join(tempfile.gettempdir(), "inspect_3216_more_cache")
INJ_P = (30.0, 50.0, 80.0, 120.0, 160.0, 190.0, 215.0, 250.0, 300.0, 380.0)
INJ_A = (0.05, 0.1, 0.2, 0.3, 0.73)
INJ_N = 20
TINJ_P = (12.0, 24.0, 48.0, 96.0, 150.0, 200.0, 250.0, 300.0)
TINJ_A = (0.1, 0.2, 0.3)
TINJ_N = 20
TESS_VARIANTS = ("appmean", "segmean", "seglin", "segquad", "qf_appmean", "qf_seglin")
BIN_VARIANTS = ("raw", "Q", "QF", "QFT", "QFA")
GROUND_VARIANTS = ("raw", "Q", "QF", "QFA")
SLOW_MODELS = ("M3", "M4", "QF3", "Q", "QF")
BINJ_A = (0.1, 0.2, 0.3)
OBSAMP_N = 20


# ----------------------------------------------------------------------------------------------- helpers
def pfast(t, y, f, nterms=2):
    """Unweighted multi-term Lomb-Scargle power (standard normalisation, floating mean) on a regular grid f.
    Same algorithm and sums as astropy's fastchi2 (Palmer 2009 trig sums via FFT), but the per-frequency normal
    equations are solved in one vectorised call instead of a Python loop (identical result, ~20x faster)."""
    from astropy.timeseries.periodograms.lombscargle.implementations.utils import trig_sum
    t = np.asarray(t, float)
    y = np.asarray(y, float) - np.mean(y)
    Nf = len(f)
    kw = dict(f0=float(f[0]), df=float(f[1] - f[0]), N=Nf, use_fft=True)
    w = np.ones_like(t)
    Sw, Cw = [np.zeros(Nf)], [np.full(Nf, float(len(t)))]
    for i in range(1, 2 * nterms + 1):
        s, c = trig_sum(t, w, freq_factor=i, **kw)
        Sw.append(s)
        Cw.append(c)
    Sy, Cy = [np.zeros(Nf)], [np.zeros(Nf)]
    for i in range(1, nterms + 1):
        s, c = trig_sum(t, y, freq_factor=i, **kw)
        Sy.append(s)
        Cy.append(c)
    order = [("C", 0)] + [(X, i) for i in range(1, nterms + 1) for X in ("S", "C")]
    n = len(order)
    XTX = np.empty((Nf, n, n))
    XTy = np.empty((Nf, n))
    for a, (A, m) in enumerate(order):
        XTy[:, a] = Sy[m] if A == "S" else Cy[m]
        for b, (B, k) in enumerate(order):
            if A == "S" and B == "S":
                v = 0.5 * (Cw[abs(m - k)] - Cw[m + k])
            elif A == "C" and B == "C":
                v = 0.5 * (Cw[abs(m - k)] + Cw[m + k])
            elif A == "S" and B == "C":
                v = 0.5 * (np.sign(m - k) * Sw[abs(m - k)] + Sw[m + k])
            else:
                v = 0.5 * (np.sign(k - m) * Sw[abs(k - m)] + Sw[m + k])
            XTX[:, a, b] = v
    beta = np.linalg.solve(XTX, XTy[..., None])[..., 0]
    return np.einsum("ij,ij->i", XTy, beta) / np.dot(y, y)


def projector(X):
    U, s, _ = np.linalg.svd(X, full_matrices=False)
    return U[:, s > s.max() * 1e-9]


def resid(U, y):
    return y - U @ (U.T @ y)


def perm_index(groups, night):
    out = []
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        nn = night[idx]
        out.append([idx[nn == k] for k in np.unique(nn)])
    return out


def permute(rng, y, pidx):
    """nightly means permuted among the nights of each group, deviations shuffled within each night"""
    out = np.empty_like(y)
    for nights in pidx:
        means = np.array([y[ix].mean() for ix in nights])
        perm = rng.permutation(len(nights))
        for j, ix in enumerate(nights):
            out[ix] = means[perm[j]] + rng.permutation(y[ix] - means[j])
    return out


def shape(phase, A, rng):
    th1, th2 = rng.uniform(0, 2 * np.pi, 2)
    g = lambda ph: 0.85 * np.cos(4 * np.pi * ph + th2) + 0.25 * np.cos(2 * np.pi * ph + th1)  # noqa: E731
    return A * g(phase) / np.ptp(g(np.linspace(0, 1, 2000)))


def tess_segments(t, sh):
    seg = -np.ones(len(t), int)
    m = np.where(sh == TESS)[0]
    o = m[np.argsort(t[m])]
    for k, g in enumerate(np.split(o, np.where(np.diff(t[o]) > 1.0)[0] + 1)):
        seg[g] = k
    return seg


def masked(n, m, vals):
    c = np.zeros(n)
    c[m] = vals
    return c


def design(c, v):
    t, app, seg = c["t"], c["app"], c["seg"]
    n = len(t)
    if v == "raw":
        return np.ones((n, 1))
    cols = []
    for b in np.unique(c["block"]):
        m = c["block"] == b
        tt = t[m] - t[m].mean()
        span = float(np.ptp(tt))
        deg = 2 if span > 10 and m.sum() > 20 else (1 if m.sum() > 5 else 0)
        sc = max(span / 2.0, 1e-3)
        for d in range(deg + 1):
            cols.append(masked(n, m, (tt / sc) ** d))
    if "F" in v:
        w = 2 * np.pi * 24.0 / P_SYN
        for k in range(1, 5):
            cols += [np.cos(k * w * t), np.sin(k * w * t)]
    if "A" in v:
        w = 2 * np.pi * 24.0 / P_HALF
        for k in np.unique(app):
            m = app == k
            if m.sum() >= 100 and np.ptp(t[m]) >= 30:
                for h in (1, 2):
                    cols += [masked(n, m, np.cos(h * w * t[m])), masked(n, m, np.sin(h * w * t[m]))]
    if "T" in v:
        for k in np.unique(seg[seg >= 0]):
            m = seg == k
            if m.sum() >= 20 and np.ptp(t[m]) > 0.5:
                tt = t[m] - t[m].mean()
                cols += [masked(n, m, np.ones(m.sum())), masked(n, m, tt / max(np.ptp(tt) / 2, 1e-3))]
    return np.column_stack(cols)


_CTX = {}


def ctx(kind="original"):
    if kind in _CTX:
        return _CTX[kind]
    sub = "results" if kind == "original" else "results_geo"
    a = np.load(os.path.join(ROOT, sub, NUM, f"{NUM}_arrays.npz"), allow_pickle=True)
    t, y, sh, t0 = a["t"].astype(float), a["y"].astype(float), a["sheet"].astype(str), float(a["time_zero"])
    app = L.apparition_index(t)
    lon = L.station_lon(sh)
    night = np.floor(t + t0 + lon / 360.0).astype(int)
    T = float(np.ptp(t))
    c = dict(t=t, y=y, sh=sh, t0=t0, app=app, night=night, seg=tess_segments(t, sh), T=T,
             block=np.array([f"{s}|{k}" for s, k in zip(sh, app)]))
    c["proj"] = {v: projector(design(c, v)) for v in VARIANTS}
    c["pidx"] = perm_index(c["block"], night)
    c["FL"] = np.arange(24.0 / 400.0, 1.0, 1.0 / (20 * T))
    c["FI"] = np.arange(24.0 / 400.0, 1.0, 1.0 / (10 * T))
    c["FS"] = np.arange(1.0, 12.0, 1.0 / (5 * T))
    _CTX[kind] = c
    return c


def tctx():
    """TESS-only data. Variants: appmean / segmean / seglin / segquad on the original magnitudes; qf_appmean /
    qf_seglin on the residuals of the global QF fit (slow signal removed using all data)."""
    if "tess" in _CTX:
        return _CTX["tess"]
    c0 = ctx("original")
    m = c0["sh"] == TESS
    t, y, seg, app = c0["t"][m], c0["y"][m], c0["seg"][m], c0["app"][m]
    yqf = resid(c0["proj"]["QF"], c0["y"])[m]
    mslow = (c0["y"] - resid(c0["proj"]["QF"], c0["y"]))[m]
    day = np.floor(t + c0["t0"]).astype(int)
    n = len(t)
    X = {"appmean": np.column_stack([(app == k).astype(float) for k in np.unique(app)]),
         "segmean": np.column_stack([(seg == k).astype(float) for k in np.unique(seg)])}
    for name, deg in (("seglin", 1), ("segquad", 2)):
        cols = []
        for k in np.unique(seg):
            mm = seg == k
            cols.append(mm.astype(float))
            if mm.sum() >= 20 and np.ptp(t[mm]) > 0.5:
                tt = t[mm] - t[mm].mean()
                for d in range(1, deg + 1):
                    cols.append(masked(n, mm, (tt / max(np.ptp(tt) / 2, 1e-3)) ** d))
        X[name] = np.column_stack(cols)
    X["qf_appmean"], X["qf_seglin"] = X["appmean"], X["seglin"]
    Tt = float(np.ptp(t))
    c = dict(t=t, y=y, yqf=yqf, mslow=mslow, seg=seg, app=app, day=day, T=Tt,
             proj={v: projector(X[v]) for v in TESS_VARIANTS}, pidx=perm_index(seg, day),
             F=np.arange(24.0 / 360.0, 12.0, 1.0 / (5 * Tt)))
    c["base"] = {v: (yqf if v.startswith("qf_") else y) for v in TESS_VARIANTS}
    _CTX["tess"] = c
    return c


def block_degree(t, block):
    """polynomial degree per dataset+apparition block (rule of design())"""
    deg = {}
    for b in np.unique(block):
        m = block == b
        span = float(np.ptp(t[m]))
        deg[b] = 2 if span > 10 and m.sum() > 20 else (1 if m.sum() > 5 else 0)
    return deg


def design_x(t, block, seg, base, nh, tess_lin=False, deg=None, app=None, app_sin=()):
    """base 'M': one constant per block; 'Q': polynomial per block (degree from deg, capped at n-1);
    nh: harmonics of the 1266.55 h Fourier series; tess_lin: constant + linear trend per TESS segment;
    app_sin: apparitions that get their own 633.3/316.7 h sinusoids (as in design() variant QFA)"""
    n = len(t)
    if deg is None:
        deg = block_degree(t, block)
    cols = []
    for b in np.unique(block):
        m = block == b
        tt = t[m] - t[m].mean()
        span = float(np.ptp(tt))
        sc = max(span / 2.0, 1e-3)
        dmax = 0 if base == "M" else min(deg[b], int(m.sum()) - 1)
        for d in range(dmax + 1):
            cols.append(masked(n, m, (tt / sc) ** d))
    w = 2 * np.pi * 24.0 / P_SYN
    for k in range(1, nh + 1):
        cols += [np.cos(k * w * t), np.sin(k * w * t)]
    if tess_lin:
        for k in np.unique(seg[seg >= 0]):
            m = seg == k
            if m.sum() >= 3 and np.ptp(t[m]) > 0.5:
                tt = t[m] - t[m].mean()
                cols += [masked(n, m, np.ones(m.sum())), masked(n, m, tt / max(np.ptp(tt) / 2, 1e-3))]
    wa = 2 * np.pi * 24.0 / P_HALF
    for k in app_sin:
        m = app == k
        for h in (1, 2):
            cols += [masked(n, m, np.cos(h * wa * t[m])), masked(n, m, np.sin(h * wa * t[m]))]
    return np.column_stack(cols)


# (base, n harmonics, TESS segment trends, per-apparition 633.3/316.7 h sinusoids)
VSPEC = {"raw": None, "Q": ("Q", 0, False, False), "QF": ("Q", 4, False, False), "QFT": ("Q", 4, True, False),
         "QFA": ("Q", 4, False, True), "M3": ("M", 3, False, False), "M4": ("M", 4, False, False),
         "QF3": ("Q", 3, False, False)}


def qfa_apparitions():
    """apparitions with their own sinusoids in QFA (unbinned criterion of design(): n >= 100 and span >= 30 d)"""
    c0 = ctx("original")
    return [int(k) for k in np.unique(c0["app"]) if np.sum(c0["app"] == k) >= 100 and np.ptp(c0["t"][c0["app"] == k]) >= 30]


def bctx():
    """binned data: one point per dataset and night (station-local night of ctx); TESS (C57G) in 6-h bins"""
    if "bin" in _CTX:
        return _CTX["bin"]
    c0 = ctx("original")
    t, y, sh, t0 = c0["t"], c0["y"], c0["sh"], c0["t0"]
    is_t = sh == TESS
    kn = np.where(is_t, np.floor((t + t0) * 4.0).astype(np.int64), c0["night"].astype(np.int64))
    keys = np.array([f"{s}|{k}" for s, k in zip(sh, kn)])
    u, bi = np.unique(keys, return_inverse=True)
    cnt = np.bincount(bi).astype(float)
    tb = np.bincount(bi, t) / cnt
    yb = np.bincount(bi, y) / cnt
    first = np.zeros(len(u), int)
    first[bi[::-1]] = np.arange(len(t))[::-1]
    shb, appb, blockb = sh[first], c0["app"][first], c0["block"][first]
    segb = tess_segments(tb, shb)
    deg = block_degree(t, c0["block"])  # same polynomial degrees as for the unbinned data (capped at n_bins - 1)
    T = float(np.ptp(tb))
    c = dict(t=tb, y=yb, sh=shb, app=appb, block=blockb, seg=segb, bi=bi, cnt=cnt, T=T, deg=deg,
             F=np.arange(24.0 / 400.0, 1.0, 1.0 / (20 * T)), FI=np.arange(24.0 / 400.0, 1.0, 1.0 / (10 * T)))
    c["proj"] = {}
    for v in BIN_VARIANTS:
        if v == "raw":
            c["proj"][v] = projector(np.ones((len(tb), 1)))
        else:
            base, nh, tl, asin = VSPEC[v]
            c["proj"][v] = projector(design_x(tb, blockb, segb, base, nh, tl, deg, appb,
                                              qfa_apparitions() if asin else ()))
    # permutation groups: bins of the same dataset and apparition (each bin its own "night")
    c["pidx"] = perm_index(blockb, np.arange(len(tb)))
    _CTX["bin"] = c
    return c


def gctx():
    """ground-based data only (all datasets except C57G)"""
    if "ground" in _CTX:
        return _CTX["ground"]
    c0 = ctx("original")
    m = c0["sh"] != TESS
    c = {k: c0[k][m] for k in ("t", "y", "sh", "app", "night", "block", "seg")}
    c["T"] = float(np.ptp(c["t"]))
    c["t0"] = c0["t0"]
    c["mask"] = m
    c["proj"] = {v: projector(design(c, v)) for v in GROUND_VARIANTS}
    c["pidx"] = perm_index(c["block"], c["night"])
    c["F"] = c0["FL"]
    rq = resid(c0["proj"]["QF"], c0["y"])
    c["slow_all_QF"] = (c0["y"] - rq)[m]  # slow model fitted to all data, at the ground epochs
    c["resid_all_QF"] = rq[m]
    _CTX["ground"] = c
    return c


# ----------------------------------------------------------------------------------------------- tasks
def task_obs(s):
    c = ctx(s["data"])
    U = c["proj"][s["variant"]]
    yy = resid(U, c["y"])
    if s.get("subset"):
        m = subset_mask(c, s["subset"])
        return {"p": pfast(c["t"][m], yy[m], c[s["grid"]]).astype(np.float32)}
    return {"p": pfast(c["t"], yy, c[s["grid"]]).astype(np.float32)}


def task_null(s):
    c = ctx(s["data"])
    U = c["proj"][s["variant"]]
    r = resid(U, c["y"])
    rng = np.random.default_rng(s["seed"])
    f = c[s["grid"]]
    curves = []
    for _ in range(s["n"]):
        yp = resid(U, permute(rng, r, c["pidx"]))
        p = pfast(c["t"], yp, f)
        curves.append(p.astype(np.float16) if s["grid"] == "FL" else np.array([p.max(), 24.0 / f[np.argmax(p)]]))
    return {"curves": np.array(curves)}


def task_slownull(s):
    """fitted slow model (no short period) + permuted residuals, searched without detrending"""
    c = ctx("original")
    U = c["proj"][s["variant"]]
    r = resid(U, c["y"])
    M = c["y"] - r
    rng = np.random.default_rng(s["seed"])
    curves = [pfast(c["t"], M + permute(rng, r, c["pidx"]), c["FL"]).astype(np.float16) for _ in range(s["n"])]
    out = {"curves": np.array(curves)}
    if s.get("noise_free"):
        out["noise_free"] = pfast(c["t"], M, c["FL"]).astype(np.float32)
    return out


def task_inj(s):
    c = ctx("original")
    t, f = c["t"], c["FI"]
    rng = np.random.default_rng(s["seed"])
    res = {f"{A:g}": {k: [] for k in ("raw", "QF", "QFT")} for A in INJ_A}
    for _ in range(INJ_N):
        for A in INJ_A:
            Pj = s["P"] * (1 + rng.uniform(-0.02, 0.02))
            sig = shape(t * 24.0 / Pj, A, rng)
            for case in ("raw", "QF", "QFT"):
                yy = resid(c["proj"][case], c["y"] + sig)
                p = pfast(t, yy, f)
                k = int(np.argmax(p))
                w = np.abs(f - 24.0 / Pj) <= 1.0 / c["T"]
                res[f"{A:g}"][case].append((Pj, float(24.0 / f[k]), float(p[k]), float(p[w].max())))
    return {"P": s["P"], "res": res}


def task_tess_obs(s):
    c = tctx()
    out = {}
    for v in TESS_VARIANTS:
        out[v] = pfast(c["t"], resid(c["proj"][v], c["base"][v]), c["F"]).astype(np.float32)
    out["W"] = window(c["t"], c["F"]).astype(np.float32)
    out["slow_seglin_noise_free"] = pfast(c["t"], resid(c["proj"]["seglin"], c["mslow"]), c["F"]).astype(np.float32)
    for k in np.unique(c["app"]):
        m = c["app"] == k
        for v in ("appmean", "seglin", "qf_seglin"):
            out[f"app{k}_{v}"] = pfast(c["t"][m], resid(c["proj"][v], c["base"][v])[m], c["F"]).astype(np.float32)
    return out


def task_tess_null(s):
    c = tctx()
    v = s["variant"]
    rng = np.random.default_rng(s["seed"])
    if v == "slow_seglin":
        # fitted slow model (QF, all data) at the TESS times + permuted QF residuals, then seglin detrending
        U = c["proj"]["seglin"]
        return {"curves": np.array([pfast(c["t"], resid(U, c["mslow"] + permute(rng, c["yqf"], c["pidx"])), c["F"])
                                    for _ in range(s["n"])]).astype(np.float16)}
    U = c["proj"][v]
    r = resid(U, c["base"][v])
    return {"curves": np.array([pfast(c["t"], resid(U, permute(rng, r, c["pidx"])), c["F"])
                                for _ in range(s["n"])]).astype(np.float16)}


def task_tess_inj(s):
    c = tctx()
    rng = np.random.default_rng(s["seed"])
    res = {f"{A:g}": {v: [] for v in ("appmean", "seglin", "segquad")} for A in TINJ_A}
    for _ in range(TINJ_N):
        for A in TINJ_A:
            Pj = s["P"] * (1 + rng.uniform(-0.02, 0.02))
            sig = shape(c["t"] * 24.0 / Pj, A, rng)
            for v in ("appmean", "seglin", "segquad"):
                p = pfast(c["t"], resid(c["proj"][v], c["y"] + sig), c["F"])
                k = int(np.argmax(p))
                w = np.abs(c["F"] - 24.0 / Pj) <= 0.5 / 30.0
                res[f"{A:g}"][v].append((Pj, float(24.0 / c["F"][k]), float(p[k]), float(p[w].max())))
    return {"P": s["P"], "res": res}


QP_RANGE = (560.0, 720.0)


def qp_periods(t, yres, app, apps):
    """per apparition: the period in 560-720 h whose 2-harmonic fit (with a constant) reduces the variance of yres most
    (frequency step: 0.02 cycles over the apparition)"""
    out = {}
    for k in apps:
        m = app == k
        span = float(np.ptp(t[m]))
        fr = np.arange(24.0 / QP_RANGE[1], 24.0 / QP_RANGE[0], 0.02 / span)
        best = None
        for ff in fr:
            X = fourier_X(t[m], 24.0 / ff, 2)
            b, *_ = np.linalg.lstsq(X, yres[m], rcond=None)
            r = float(np.sum((yres[m] - X @ b) ** 2))
            if best is None or r < best[1]:
                best = (24.0 / ff, r)
        out[int(k)] = float(best[0])
    return out


def qp_context(kind):
    """QP = QF + one 2-harmonic Fourier series per apparition with its own period in 560-720 h (the slow signal is not
    strictly periodic: single apparitions give 623-646 h, and its synodic period changes with geometry).
    kind: 'original' (all data), 'ground', 'bin'"""
    key = "qp_" + kind
    if key in _CTX:
        return _CTX[key]
    if kind == "original":
        c = ctx("original")
        X0 = design(c, "QF")
        apps = qfa_apparitions()
    elif kind == "ground":
        c = gctx()
        X0 = design(c, "QF")
        apps = [int(k) for k in np.unique(c["app"]) if np.sum(c["app"] == k) >= 100 and np.ptp(c["t"][c["app"] == k]) >= 30]
    else:
        c = bctx()
        X0 = design_x(c["t"], c["block"], c["seg"], "Q", 4, False, c["deg"])
        apps = qfa_apparitions()
    t, app = c["t"], c["app"]
    yq = resid(projector(X0), c["y"])
    Pa = qp_periods(t, yq, app, apps)
    cols = [X0]
    n = len(t)
    for k, P in Pa.items():
        m = app == k
        w = 2 * np.pi * 24.0 / P
        for h in (1, 2):
            cols.append(np.column_stack([masked(n, m, np.cos(h * w * t[m])), masked(n, m, np.sin(h * w * t[m]))]))
    U = projector(np.column_stack(cols))
    q = {"U": U, "P_app": Pa, "t": t, "y": c["y"], "pidx": c["pidx"], "T": c["T"],
         "F": c["F"] if "F" in c else c["FL"], "n_par": int(U.shape[1])}
    _CTX[key] = q
    return q


def task_qp_obs(s):
    q = qp_context(s["data"])
    yy = resid(q["U"], q["y"])
    out = {"p": pfast(q["t"], yy, q["F"]).astype(np.float32), "P_app": q["P_app"], "n_par": q["n_par"],
           "rms": float(np.std(yy))}
    if s["data"] == "original":
        c = ctx("original")
        out["pS"] = pfast(q["t"], yy, c["FS"]).astype(np.float32)
    return out


def task_qp_null(s):
    """permutation null with the per-apparition periods held fixed (slightly anti-conservative)"""
    q = qp_context(s["data"])
    r = resid(q["U"], q["y"])
    rng = np.random.default_rng(s["seed"])
    return {"curves": np.array([pfast(q["t"], resid(q["U"], permute(rng, r, q["pidx"])), q["F"])
                                for _ in range(s["n"])]).astype(np.float16)}


def task_validate(s):
    """vectorised fastchi2 (pfast) vs astropy LombScargle chi2 (exact) and fastchi2, nterms = 2"""
    from astropy.timeseries import LombScargle
    c = ctx("original")
    out = {}
    rng = np.random.default_rng(s["seed"])
    for v in ("raw", "QF"):
        yy = resid(c["proj"][v], c["y"])
        p = pfast(c["t"], yy, c["FL"])
        idx = np.sort(rng.choice(len(c["FL"]), 1500, replace=False))
        ex = LombScargle(c["t"], yy, nterms=2).power(c["FL"][idx], method="chi2")
        fa = LombScargle(c["t"], yy, nterms=2).power(c["FL"], method="fastchi2", assume_regular_frequency=True)
        out[v] = {"max_abs_diff_vs_exact_1500_freqs": float(np.max(np.abs(p[idx] - ex))),
                  "max_abs_diff_vs_astropy_fastchi2": float(np.max(np.abs(p - fa))),
                  "max_power": float(p.max())}
    return out


def task_bin_obs(s):
    c = bctx()
    out = {v: pfast(c["t"], resid(c["proj"][v], c["y"]), c["F"]).astype(np.float32) for v in BIN_VARIANTS}
    out["W"] = window(c["t"], c["F"]).astype(np.float32)
    return out


def task_bin_null(s):
    c = bctx()
    U = c["proj"][s["variant"]]
    r = resid(U, c["y"])
    rng = np.random.default_rng(s["seed"])
    return {"curves": np.array([pfast(c["t"], resid(U, permute(rng, r, c["pidx"])), c["F"])
                                for _ in range(s["n"])]).astype(np.float16)}


def task_bin_inj(s):
    c = bctx()
    c0 = ctx("original")
    rng = np.random.default_rng(s["seed"])
    f = c["FI"]
    res = {f"{A:g}": {v: [] for v in ("QF", "QFT")} for A in BINJ_A}
    for _ in range(INJ_N):
        for A in BINJ_A:
            Pj = s["P"] * (1 + rng.uniform(-0.02, 0.02))
            sig = shape(c0["t"] * 24.0 / Pj, A, rng)
            sb = np.bincount(c["bi"], sig) / c["cnt"]
            for v in ("QF", "QFT"):
                p = pfast(c["t"], resid(c["proj"][v], c["y"] + sb), f)
                k = int(np.argmax(p))
                w = np.abs(f - 24.0 / Pj) <= 1.0 / c["T"]
                res[f"{A:g}"][v].append((Pj, float(24.0 / f[k]), float(p[k]), float(p[w].max())))
    return {"P": s["P"], "res": res}


def task_ground_obs(s):
    c = gctx()
    out = {v: pfast(c["t"], resid(c["proj"][v], c["y"]), c["F"]).astype(np.float32) for v in GROUND_VARIANTS}
    out["slow_noise_free"] = pfast(c["t"], c["slow_all_QF"], c["F"]).astype(np.float32)
    out["W"] = window(c["t"], c["F"]).astype(np.float32)
    return out


def task_ground_null(s):
    c = gctx()
    rng = np.random.default_rng(s["seed"])
    if s["variant"] == "slow":
        return {"curves": np.array([pfast(c["t"], c["slow_all_QF"] + permute(rng, c["resid_all_QF"], c["pidx"]), c["F"])
                                    for _ in range(s["n"])]).astype(np.float16)}
    U = c["proj"][s["variant"]]
    r = resid(U, c["y"])
    return {"curves": np.array([pfast(c["t"], resid(U, permute(rng, r, c["pidx"])), c["F"])
                                for _ in range(s["n"])]).astype(np.float16)}


def task_slowmodel(s):
    """slow model of type s['model'] fitted to all data, sampled at the real epochs: noise-free periodogram and
    realisations with permuted residuals (searched without detrending)"""
    c = ctx("original")
    base, nh, tl, _ = VSPEC[s["model"]]
    U = projector(design_x(c["t"], c["block"], c["seg"], base, nh, tl))
    r = resid(U, c["y"])
    M = c["y"] - r
    rng = np.random.default_rng(s["seed"])
    out = {"noise_free": pfast(c["t"], M, c["FL"]).astype(np.float32), "n_par": int(U.shape[1]),
           "rms_resid": float(np.std(r)), "frac_var_explained": float(1 - np.var(r) / np.var(c["y"]))}
    out["curves"] = np.array([pfast(c["t"], M + permute(rng, r, c["pidx"]), c["FL"]) for _ in range(s["n"])]).astype(np.float16)
    return out


def task_obsamp(s):
    """the raw-data 4-harmonic fold at P (observed shape and amplitude) injected at a random phase, then detrended"""
    c = ctx("original")
    t, y, f, T = c["t"], c["y"], c["FI"], c["T"]
    P = s["P"]
    X = fourier_X(t, P, 4)
    b, *_ = np.linalg.lstsq(X, y - y.mean(), rcond=None)
    ph = np.linspace(0, 1, 400, endpoint=False)
    amp = float(np.ptp(fourier_X(ph * P / 24.0, P, 4)[:, 1:] @ b[1:]))
    rng = np.random.default_rng(s["seed"])
    res = {v: [] for v in ("QF", "QFT")}
    w = np.abs(f - 24.0 / P) <= 1.0 / T
    for _ in range(OBSAMP_N):
        sh_ = rng.uniform(0, P / 24.0)
        sig = fourier_X(t + sh_, P, 4)[:, 1:] @ b[1:]
        for v in res:
            p = pfast(t, resid(c["proj"][v], y + sig), f)
            k = int(np.argmax(p))
            res[v].append((float(24.0 / f[k]), float(p[k]), float(p[w].max())))
    obs = {v: float(pfast(t, resid(c["proj"][v], y), f)[w].max()) for v in res}
    return {"P": P, "amp_pp": amp, "res": res, "obs_power_at_P": obs}


TASKS = {"obs": task_obs, "null": task_null, "slownull": task_slownull, "inj": task_inj,
         "tess_obs": task_tess_obs, "tess_null": task_tess_null, "tess_inj": task_tess_inj,
         "validate": task_validate, "bin_obs": task_bin_obs, "bin_null": task_bin_null, "bin_inj": task_bin_inj,
         "ground_obs": task_ground_obs, "ground_null": task_ground_null, "slowmodel": task_slowmodel,
         "obsamp": task_obsamp, "qp_obs": task_qp_obs, "qp_null": task_qp_null}


def run_task(s):
    path = os.path.join(CACHE, s["name"] + ".pkl")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return s["name"], pickle.load(fh)
    t1 = time.time()
    res = TASKS[s["kind"]](s)
    res["_sec"] = time.time() - t1
    with open(path + ".tmp", "wb") as fh:
        pickle.dump(res, fh)
    os.replace(path + ".tmp", path)
    return s["name"], res


def subset_mask(c, name):
    st = np.array([x[:3] for x in c["sh"]])
    return {"odd": c["app"] % 2 == 1, "even": c["app"] % 2 == 0, "atlas": np.isin(st, L.ATLAS),
            "not_atlas": ~np.isin(st, L.ATLAS), "tess": c["sh"] == TESS, "not_tess": c["sh"] != TESS}[name]


SUBSETS = ("odd", "even", "atlas", "not_atlas", "tess", "not_tess")


def specs():
    S = []
    for v in VARIANTS:
        S.append({"name": f"obs_orig_{v}_FL", "kind": "obs", "data": "original", "variant": v, "grid": "FL"})
    for v in ("raw", "QF", "QFT"):
        S.append({"name": f"obs_orig_{v}_FS", "kind": "obs", "data": "original", "variant": v, "grid": "FS"})
    for v in ("raw", "QF"):
        S.append({"name": f"obs_geo_{v}_FL", "kind": "obs", "data": "geometry_corrected", "variant": v, "grid": "FL"})
    for sub in SUBSETS:
        S.append({"name": f"obs_orig_QF_FL_{sub}", "kind": "obs", "data": "original", "variant": "QF", "grid": "FL",
                  "subset": sub})
    S.append({"name": "tess_obs", "kind": "tess_obs"})
    seed = 32160
    nperm = {"raw": 100, "Q": 60, "QF": 200, "QFA": 100, "QFT": 200}
    for v in VARIANTS:
        for j in range(nperm[v] // 20):
            seed += 1
            S.append({"name": f"null_orig_{v}_FL_{j}", "kind": "null", "data": "original", "variant": v, "grid": "FL",
                      "n": 20, "seed": seed})
    for j in range(3):
        seed += 1
        S.append({"name": f"null_geo_QF_FL_{j}", "kind": "null", "data": "geometry_corrected", "variant": "QF",
                  "grid": "FL", "n": 20, "seed": seed})
    for j in range(5):
        seed += 1
        S.append({"name": f"null_orig_QF_FS_{j}", "kind": "null", "data": "original", "variant": "QF", "grid": "FS",
                  "n": 20, "seed": seed})
    for v in ("QF", "QFT"):
        for j in range(5):
            seed += 1
            S.append({"name": f"slownull_{v}_{j}", "kind": "slownull", "variant": v, "n": 20, "seed": seed,
                      "noise_free": j == 0})
    for v in TESS_VARIANTS + ("slow_seglin",):
        for j in range(5):
            seed += 1
            S.append({"name": f"tess_null_{v}_{j}", "kind": "tess_null", "variant": v, "n": 40, "seed": seed})
    for P in INJ_P:
        seed += 1
        S.append({"name": f"inj_{P:g}", "kind": "inj", "P": P, "seed": seed})
    for P in TINJ_P:
        seed += 1
        S.append({"name": f"tess_inj_{P:g}", "kind": "tess_inj", "P": P, "seed": seed})
    # ---- second pass (section 8 of the docstring); separate seed base so that the first-pass tasks are unchanged
    seed = 42000
    S.append({"name": "validate", "kind": "validate", "seed": seed})
    S.append({"name": "bin_obs", "kind": "bin_obs"})
    S.append({"name": "ground_obs", "kind": "ground_obs"})
    for v in BIN_VARIANTS:
        for j in range(5):
            seed += 1
            S.append({"name": f"bin_null_{v}_{j}", "kind": "bin_null", "variant": v, "n": 60, "seed": seed})
    for v in GROUND_VARIANTS + ("slow",):
        for j in range(5):
            seed += 1
            S.append({"name": f"ground_null_{v}_{j}", "kind": "ground_null", "variant": v, "n": 40, "seed": seed})
    for mdl in SLOW_MODELS:
        if mdl == "QF":
            continue  # first pass (slownull_QF_*)
        seed += 1
        S.append({"name": f"slowmodel_{mdl}", "kind": "slowmodel", "model": mdl, "n": 40, "seed": seed})
    for P in INJ_P:
        seed += 1
        S.append({"name": f"bin_inj_{P:g}", "kind": "bin_inj", "P": P, "seed": seed})
    for P in REVIEW_P:
        seed += 1
        S.append({"name": f"obsamp_{P:g}", "kind": "obsamp", "P": P, "seed": seed})
    # ---- third pass: per-apparition free-period slow model (QP)
    seed = 43000
    for data, nchunk, n in (("original", 5, 20), ("ground", 5, 40), ("bin", 5, 60)):
        S.append({"name": f"qp_obs_{data}", "kind": "qp_obs", "data": data})
        for j in range(nchunk):
            seed += 1
            S.append({"name": f"qp_null_{data}_{j}", "kind": "qp_null", "data": data, "n": n, "seed": seed})
    return S


# ----------------------------------------------------------------------------------------------- analysis
def exact_local(t, y, f0, T, half=2.0, npts=401):
    g = f0 + np.linspace(-half, half, npts) / T
    g = g[g > 0]
    p = L.ls_power(t, y, g)
    k = int(np.argmax(p))
    return float(24.0 / g[k]), float(p[k])


def local_max(curve, f, f0, T, half=1.0):
    w = np.abs(f - f0) <= half / T
    return float(curve[w].max())


def fourier_X(t, P, nterms, offsets=None):
    w = 2 * np.pi * 24.0 / P
    cols = [np.cos(k * w * t) for k in range(1, nterms + 1)] + [np.sin(k * w * t) for k in range(1, nterms + 1)]
    if offsets is None:
        cols = [np.ones(len(t))] + cols
    else:
        cols = [(offsets == u).astype(float) for u in np.unique(offsets)] + cols
    return np.column_stack(cols)


def fold_stats(t, y, P, nterms=4, sheet=None):
    from scipy.signal import find_peaks
    X = fourier_X(t, P, nterms)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    ph = np.linspace(0, 1, 400, endpoint=False)
    mod = fourier_X(ph * P / 24.0, P, nterms) @ b
    A = float(np.ptp(mod))
    # magnitudes: maxima of brightness = minima of magnitude
    tiled = np.concatenate([-mod, -mod, -mod])
    pk, _ = find_peaks(tiled, prominence=0.15 * A)
    nmax = int(np.sum((pk >= 400) & (pk < 800)))
    out = {"P": float(P), "amp_pp": A, "n_maxima": nmax, "rms_before": float(np.std(y)),
           "rms_after": float(np.std(y - X @ b)), "model_phase": ph[::10].tolist(), "model": (mod[::10]).tolist()}
    if sheet is not None:
        phase = (t * 24.0 / P) % 1.0
        bins = np.linspace(0, 1, 21)
        for lab, m in (("tess", sheet == TESS), ("other", sheet != TESS)):
            idx = np.digitize(phase[m], bins) - 1
            out[f"bin_{lab}"] = [float(np.mean(y[m][idx == k])) if np.any(idx == k) else None for k in range(20)]
            out[f"bin_{lab}_n"] = [int(np.sum(idx == k)) for k in range(20)]
    return out


def xpred_r(t, y, sh, nkey, P, mA, mB):
    """fit nterms=2 at P (+ per-dataset offsets) to subset A, predict B; r between nightly means (per-dataset demeaned)"""
    XA = fourier_X(t[mA], P, 2, offsets=sh[mA])
    b, *_ = np.linalg.lstsq(XA, y[mA], rcond=None)
    nd = len(np.unique(sh[mA]))
    XB = fourier_X(t[mB], P, 2)[:, 1:]
    pred = XB @ b[nd:]
    obs = y[mB].copy()
    shB = sh[mB]
    for u in np.unique(shB):
        mm = shB == u
        obs[mm] -= obs[mm].mean()
        pred[mm] -= pred[mm].mean()
    _, inv = np.unique(nkey[mB], return_inverse=True)
    cnt = np.bincount(inv)
    o = np.bincount(inv, obs) / cnt
    p = np.bincount(inv, pred) / cnt
    if np.std(p) == 0:
        return 0.0
    return float(np.corrcoef(o, p)[0, 1])


def window(t, f, chunk=2000):
    w = np.empty(len(f))
    for s in range(0, len(f), chunk):
        ph = 2 * np.pi * np.outer(f[s:s + chunk], t)
        w[s:s + chunk] = (np.cos(ph).sum(1) ** 2 + np.sin(ph).sum(1) ** 2) / len(t) ** 2
    return w


def families(f, p, n=8, lo=24.0, hi=400.0):
    P = 24.0 / f
    m = (P >= lo) & (P <= hi)
    fam = L.peak_families(f[m], p[m], n=n)
    return [{"P": float(F["P"]), "power": float(F["power"]), "members": [round(x, 2) for x in F["members"][:6]]}
            for F in fam]


def harmonic(P):
    best = None
    for Pl in P_LONGS:
        for k in range(1, 13):
            d = (Pl / k - P) / P
            if best is None or abs(d) < abs(best[2]):
                best = (Pl, k, d)
    return {"P_long": best[0], "k": best[1], "rel_offset": float(best[2]), "within_1pct": bool(abs(best[2]) <= 0.01)}


def pval(obs, null_max):
    return float((1 + np.sum(null_max >= obs)) / (1 + len(null_max)))


def main(argv):
    workers = int(argv[argv.index("--workers") + 1]) if "--workers" in argv else 4
    workers = min(workers, 4)
    os.makedirs(CACHE, exist_ok=True)
    S = specs()
    todo = [s for s in S if not os.path.exists(os.path.join(CACHE, s["name"] + ".pkl"))]
    print(f"{len(S)} tasks, {len(todo)} to run, cache {CACHE}", flush=True)
    R = {}
    t_start = time.time()
    order = sorted(S, key=lambda s: {"obs": 0, "tess_obs": 0, "inj": 1, "null": 2, "slownull": 2, "tess_null": 3,
                                     "tess_inj": 3, "validate": 0, "bin_obs": 0, "ground_obs": 0, "slowmodel": 1,
                                     "obsamp": 1, "bin_inj": 2, "bin_null": 3, "ground_null": 3, "qp_obs": 0,
                                     "qp_null": 2}[s["kind"]])
    with mp.get_context("spawn").Pool(workers) as pool:
        for name, res in pool.imap_unordered(run_task, order):
            R[name] = res
            print(f"  {name} done ({res.get('_sec', 0):.0f} s, elapsed {time.time() - t_start:.0f} s)", flush=True)
    analyse(R)


def analyse(R):
    c = ctx("original")
    g = ctx("geometry_corrected")
    t, y, sh, T = c["t"], c["y"], c["sh"], c["T"]
    FL, FS, FI = c["FL"], c["FS"], c["FI"]
    PL = 24.0 / FL
    m200 = PL <= 200.0
    out = {"num": NUM, "n": int(len(t)), "n_tess": int(np.sum(sh == TESS)), "baseline_d": T,
           "variants": {"raw": "mean only", "Q": "quadratic per dataset and apparition",
                        "QF": "Q + Fourier 1266.55 h, 4 harmonics", "QFA": "QF + per-apparition 633.3/316.7 h sinusoids",
                        "QFT": "QF + constant and linear trend per TESS segment"},
           "grids": {"24-400 h": "step 1/(20T)", "2-24 h": "step 1/(5T)", "injection": "24-400 h, step 1/(10T)"}}
    Y = {v: resid(c["proj"][v], y) for v in VARIANTS}
    Yg = {v: resid(g["proj"][v], g["y"]) for v in ("raw", "QF")}
    out["rms_after_detrending"] = {v: float(np.std(Y[v])) for v in VARIANTS}
    # ---------------- nulls
    def null_curves(prefix):
        ks = sorted(k for k in R if k.startswith(prefix))
        return np.concatenate([R[k]["curves"] for k in ks]) if ks else None
    NULL = {v: null_curves(f"null_orig_{v}_FL_") for v in VARIANTS}
    NULL["geo_QF"] = null_curves("null_geo_QF_FL_")
    NULLS_FS = null_curves("null_orig_QF_FS_")
    SLOW = {v: null_curves(f"slownull_{v}_") for v in ("QF", "QFT")}
    OBS = {v: R[f"obs_orig_{v}_FL"]["p"] for v in VARIANTS}
    OBS["geo_raw"] = R["obs_geo_raw_FL"]["p"]
    OBS["geo_QF"] = R["obs_geo_QF_FL"]["p"]
    # ---------------- periodogram summaries
    pg = {}
    for v in list(VARIANTS) + ["geo_QF"]:
        p = OBS[v]
        nm = NULL[v]
        d = {"families_24_400": families(FL, p, n=8), "best_24_400": {"P": float(PL[np.argmax(p)]), "power": float(p.max())},
             "best_24_200": {"P": float(PL[m200][np.argmax(p[m200])]), "power": float(p[m200].max())}}
        if nm is not None:
            mx, mx200 = nm.max(1), nm[:, m200].max(1)
            d["null"] = {"n_perm": int(len(nm)), "max_24_400_median": float(np.median(mx)),
                         "max_24_400_95": float(np.percentile(mx, 95)), "max_24_400_99": float(np.percentile(mx, 99)),
                         "max_24_200_median": float(np.median(mx200)), "max_24_200_95": float(np.percentile(mx200, 95))}
            d["p_global_best_24_400"] = pval(p.max(), mx)
            d["p_global_best_24_200"] = pval(p[m200].max(), mx200)
        pg[v] = d
    for v in ("raw", "QF", "QFT"):
        p = R[f"obs_orig_{v}_FS"]["p"]
        pg[v]["best_2_24"] = {"P": float(24.0 / FS[np.argmax(p)]), "power": float(p.max()),
                              "families": families(FS, p, n=5, lo=2.0, hi=24.0)}
        if v == "QF" and NULLS_FS is not None:
            mx = NULLS_FS[:, 0]
            pg[v]["best_2_24"]["null"] = {"n_perm": int(len(mx)), "max_median": float(np.median(mx)),
                                          "max_95": float(np.percentile(mx, 95))}
            pg[v]["best_2_24"]["p_global"] = pval(p.max(), mx)
    out["periodograms"] = pg
    # ---------------- slow-signal null
    sl = {}
    for v in ("QF", "QFT"):
        cur = SLOW[v]
        nf = R[f"slownull_{v}_0"]["noise_free"]
        mx, mx200 = cur.max(1), cur[:, m200].max(1)
        med = np.median(cur, axis=0)
        sl[v] = {"n_real": int(len(cur)), "max_24_400_median": float(np.median(mx)), "max_24_400_95": float(np.percentile(mx, 95)),
                 "max_24_200_median": float(np.median(mx200)), "max_24_200_95": float(np.percentile(mx200, 95)),
                 "best_P_of_realisations_24_200": [round(float(PL[m200][np.argmax(x[m200])]), 1) for x in cur[:12]],
                 "p_raw_best_24_400": pval(OBS["raw"].max(), mx), "p_raw_best_24_200": pval(OBS["raw"][m200].max(), mx200),
                 "corr_raw_vs_median_realisation_24_400": float(np.corrcoef(OBS["raw"], med)[0, 1]),
                 "corr_raw_vs_median_realisation_24_200": float(np.corrcoef(OBS["raw"][m200], med[m200])[0, 1]),
                 "corr_raw_vs_noise_free_24_400": float(np.corrcoef(OBS["raw"], nf)[0, 1]),
                 "noise_free_families_24_400": families(FL, nf, n=8)}
    out["slow_signal_null"] = sl
    # ---------------- window
    W = window(t, FL)
    from scipy.signal import find_peaks
    kw, _ = find_peaks(W)
    kw = kw[np.argsort(W[kw])[::-1]][:25]
    wpk = [(float(W[i]), float(FL[i])) for i in kw]
    out["window_peaks_24_400"] = [{"P": round(24 / fw, 2), "W": round(wv, 4)} for wv, fw in wpk[:15]]
    tess_m = sh == TESS
    # ---------------- subsets: own best period (QF)
    SUBP = {sub: R[f"obs_orig_QF_FL_{sub}"]["p"] for sub in SUBSETS}
    out["subsets_QF_best_24_400"] = {sub: {"n": int(subset_mask(c, sub).sum()), "P": float(PL[np.argmax(p)]),
                                           "power": float(p.max()), "families": families(FL, p, n=4)}
                                     for sub, p in SUBP.items()}
    # single apparitions (QF)
    apps = {}
    for k in np.unique(c["app"]):
        m = c["app"] == k
        sp = float(np.ptp(t[m]))
        if m.sum() < 100 or sp < 30:
            continue
        ga = np.arange(24.0 / min(400.0, 2 * sp * 24), 1.0, 1.0 / (20 * sp))
        apps[int(k)] = (ga, L.ls_power(t[m], Y["QF"][m], ga), sp, int(m.sum()))
    # ---------------- candidates
    cand_P = []
    for src in (pg["raw"]["families_24_400"], pg["QF"]["families_24_400"], pg["QFT"]["families_24_400"],
                pg["QFA"]["families_24_400"]):
        cand_P += [F["P"] for F in src[:6]]
    cand_P += list(REVIEW_P)
    uniq = []
    for P in cand_P:
        if all(abs(P - Q) > 0.01 * Q for Q in uniq):
            uniq.append(P)
    rng = np.random.default_rng(99)
    randP = np.exp(rng.uniform(np.log(24.0), np.log(400.0), 300))
    dsi = {s: i for i, s in enumerate(np.unique(sh))}
    nkey = np.array([dsi[s] for s in sh], dtype=np.int64) * 100000000 + c["night"].astype(np.int64)
    NMAX = {k: v.max(1) for k, v in NULL.items() if v is not None}
    NMAX200 = {k: v[:, m200].max(1) for k, v in NULL.items() if v is not None}
    rows = []
    for P0 in uniq:
        f0 = 24.0 / P0
        row = {"P_seed": round(P0, 2), "review_named": bool(any(abs(P0 - q) <= 0.01 * q for q in REVIEW_P))}
        var = {}
        for v in VARIANTS:
            Pv, pv = exact_local(t, Y[v], f0, T)
            iv = int(np.argmin(np.abs(FL - 24.0 / Pv)))
            w = np.abs(FL - 24.0 / Pv) <= 1.0 / T
            d = {"P": Pv, "power": pv, "p_global_24_400": pval(OBS[v][iv], NMAX[v]),
                 "p_global_24_200": pval(OBS[v][iv], NMAX200[v]) if Pv <= 200 else None,
                 "p_local": pval(float(OBS[v][w].max()), NULL[v][:, w].max(1))}
            var[v] = d
        Pg, pgv = exact_local(g["t"], Yg["QF"], f0, T)
        ig = int(np.argmin(np.abs(FL - 24.0 / Pg)))
        var["geo_QF"] = {"P": Pg, "power": pgv, "p_global_24_400": pval(OBS["geo_QF"][ig], NMAX["geo_QF"])}
        Pgr, pgr = exact_local(g["t"], Yg["raw"], f0, T)
        var["geo_raw"] = {"P": Pgr, "power": pgr}
        row["variants"] = var
        # slow-signal null at this period (raw data)
        Pr = var["raw"]["P"]
        w = np.abs(FL - 24.0 / Pr) <= 1.0 / T
        obs_loc = float(OBS["raw"][w].max())
        row["slow_null_raw"] = {v: {"power_obs": obs_loc, "median_null": float(np.median(SLOW[v][:, w].max(1))),
                                    "p_local": pval(obs_loc, SLOW[v][:, w].max(1))} for v in ("QF", "QFT")}
        Pq = var["QF"]["P"]
        fq = 24.0 / Pq
        row["P_QF"] = Pq
        Wq = float(window(t, np.array([fq]))[0])
        row["window"] = {"W_all": Wq,
                         "W_all_percentile_24_400": float(np.mean(W <= Wq) * 100),
                         "W_tess": float(window(t[tess_m], np.array([fq]))[0]),
                         "W_other": float(window(t[~tess_m], np.array([fq]))[0]),
                         "near_window_peak": [(round(24 / fw, 1), dd, round(wv, 4)) for wv, fw in wpk for dd in (-2, -1, 0, 1, 2)
                                              if fw + dd > 0 and abs(fq - (fw + dd)) <= 1.0 / T]}
        row["harmonic"] = harmonic(Pq)
        row["subsets_QF"] = {}
        for sub in SUBSETS:
            m = subset_mask(c, sub)
            Ps, ps = exact_local(t[m], Y["QF"][m], fq, T)
            row["subsets_QF"][sub] = {"P_local": Ps, "power": ps, "subset_best_P": out["subsets_QF_best_24_400"][sub]["P"],
                                      "subset_best_power": out["subsets_QF_best_24_400"][sub]["power"]}
        row["apparitions_QF"] = []
        for k, (ga, pa, sp, nn) in apps.items():
            w = np.abs(ga - fq) <= 1.0 / sp
            row["apparitions_QF"].append({"k": k, "n": nn, "span_d": round(sp, 1), "best_P": round(float(24 / ga[np.argmax(pa)]), 1),
                                          "best_power": round(float(pa.max()), 3),
                                          "power_at_P": round(float(pa[w].max()), 3) if w.any() else None})
        r1 = xpred_r(t, Y["QF"], sh, nkey, Pq, ~tess_m, tess_m)
        r2 = xpred_r(t, Y["QF"], sh, nkey, Pq, tess_m, ~tess_m)
        n1 = np.array([xpred_r(t, Y["QF"], sh, nkey, q, ~tess_m, tess_m) for q in randP])
        n2 = np.array([xpred_r(t, Y["QF"], sh, nkey, q, tess_m, ~tess_m) for q in randP])
        row["xpred_QF"] = {"other_to_tess_r": r1, "other_to_tess_pct_random": float(np.mean(n1 < r1) * 100),
                           "tess_to_other_r": r2, "tess_to_other_pct_random": float(np.mean(n2 < r2) * 100)}
        fs = fold_stats(t, Y["QF"], Pq, 4, sh)
        famp = np.array([fold_stats(t, Y["QF"], q, 4)["amp_pp"] for q in randP[:100]])
        fs["amp_pct_random"] = float(np.mean(famp < fs["amp_pp"]) * 100)
        fs["amp_random_median"] = float(np.median(famp))
        fr = fold_stats(t, Y["raw"], var["raw"]["P"], 4)
        row["fold_QF"] = fs
        row["fold_raw"] = {k: fr[k] for k in ("P", "amp_pp", "n_maxima", "rms_before", "rms_after")}
        rows.append(row)
        print(f"cand {P0:7.2f}: QF P {Pq:7.2f} pow {var['QF']['power']:.3f} p {var['QF'].get('p_global_24_400')} "
              f"loc {var['QF'].get('p_local')} | raw {var['raw']['power']:.3f} QFT {var['QFT']['power']:.3f} "
              f"| W {row['window']['W_all']:.4f} | amp {fs['amp_pp']:.3f} ({fs['amp_pct_random']:.0f}%) nmax {fs['n_maxima']} "
              f"| xpred {r1:+.2f}/{r2:+.2f}", flush=True)
    out["candidates"] = rows
    # ---------------- calibration of the local p-values: a priori random periods (log-uniform 24-400 h)
    cal = {}
    rP = np.exp(np.random.default_rng(7).uniform(np.log(24.0), np.log(400.0), 300))
    for v in ("raw", "QF", "QFT"):
        pl = []
        for q in rP:
            w = np.abs(FL - 24.0 / q) <= 1.0 / T
            pl.append(pval(float(OBS[v][w].max()), NULL[v][:, w].max(1)))
        pl = np.array(pl)
        cal[v] = {"frac_local_p_lt_0.05": float(np.mean(pl < 0.05)), "frac_local_p_lt_0.01": float(np.mean(pl <= 0.01)),
                  "median_local_p": float(np.median(pl)),
                  "median_ratio_obs_to_null_median_24_400": float(np.median(OBS[v] / np.maximum(np.median(NULL[v], axis=0), 1e-6)))}
    out["local_p_calibration"] = cal
    # ---------------- 2-24 h: the strongest QF peak (11.54 h) - daily alias of residual slow structure?
    fLong = np.arange(24.0 / 5000.0, 24.0 / 300.0, 1.0 / (20 * T))
    pS = R["obs_orig_QF_FS"]["p"]
    Pb = float(24.0 / FS[np.argmax(pS)])
    chk = {"P": Pb, "power": float(pS.max()), "f_cpd": 24.0 / Pb,
           "alias_relations_h": {"f-1": 24.0 / (24.0 / Pb - 1) if 24.0 / Pb > 1 else None,
                                 "f-2": 24.0 / (24.0 / Pb - 2) if 24.0 / Pb > 2 else None,
                                 "f/2-1": 24.0 / (12.0 / Pb - 1) if 12.0 / Pb > 1 else None}}
    pr = pfast(t, Y["QF"], fLong)
    chk["QF_residual_families_300_5000"] = families(fLong, pr, n=5, lo=300.0, hi=5000.0)
    chk["subsets_power_at_P"] = {sub: exact_local(t[subset_mask(c, sub)], Y["QF"][subset_mask(c, sub)], 24.0 / Pb, T)[1]
                                 for sub in SUBSETS}
    yy = Y["QF"].copy()
    steps = []
    for _ in range(3):
        pl = pfast(t, yy, fLong)
        Pr_ = float(24.0 / fLong[np.argmax(pl)])
        X = fourier_X(t, Pr_, 2)
        b, *_ = np.linalg.lstsq(X, yy, rcond=None)
        yy = yy - X @ b
        ps = pfast(t, yy, FS)
        steps.append({"removed_P": Pr_, "best_2_24_P": float(24.0 / FS[np.argmax(ps)]), "best_2_24_power": float(ps.max()),
                      "power_at_P": exact_local(t, yy, 24.0 / Pb, T)[1]})
    chk["after_prewhitening_residual_long_periods"] = steps
    ps = pfast(t, Y["QFA"], FS)
    chk["QFA_best_2_24"] = {"P": float(24.0 / FS[np.argmax(ps)]), "power": float(ps.max()),
                            "power_at_P": exact_local(t, Y["QFA"], 24.0 / Pb, T)[1]}
    chk["fold_amp_pp"] = fold_stats(t, Y["QF"], Pb, 4)["amp_pp"]
    out["check_2_24h_best"] = chk
    # TESS within-segment variation: explained by the global slow model?
    tc0 = tctx()
    out["tess_rms_comparison"] = {"p2p_noise": None, "appmean": float(np.std(resid(tc0["proj"]["appmean"], tc0["y"]))),
                                  "seglin": float(np.std(resid(tc0["proj"]["seglin"], tc0["y"]))),
                                  "segquad": float(np.std(resid(tc0["proj"]["segquad"], tc0["y"]))),
                                  "global_QF_model_then_appmean": float(np.std(resid(tc0["proj"]["appmean"], tc0["yqf"]))),
                                  "n_par_QF_global": int(c["proj"]["QF"].shape[1])}
    # ---------------- injection
    inj = {}
    for P in INJ_P:
        rr = R[f"inj_{P:g}"]["res"]
        inj[f"{P:g}"] = {}
        for A, cases in rr.items():
            inj[f"{P:g}"][A] = {}
            for case, lst in cases.items():
                a = np.array(lst)
                rec = np.abs(a[:, 1] - a[:, 0]) <= 0.01 * a[:, 0]
                half = np.abs(a[:, 1] - a[:, 0] / 2) <= 0.01 * a[:, 0] / 2
                nul = NULL[case].max(1) if NULL.get(case) is not None else None
                det = rec & (a[:, 2] > np.percentile(nul, 95)) if nul is not None else rec
                inj[f"{P:g}"][A][case] = {"recovered": float(rec.mean()), "recovered_and_above_null95": float(det.mean()),
                                          "at_half": float(half.mean()), "median_power_at_Pinj": float(np.median(a[:, 3])),
                                          "median_best_P": float(np.median(a[:, 1]))}
    # amplitude whose median injected power equals the null 95 % level and the observed maximum (QF, log-log
    # interpolation between the two bracketing amplitudes)
    thr = {}
    for case in ("QF", "QFT"):
        n95 = float(np.percentile(NULL[case].max(1).astype(float), 95))
        omax = float(OBS[case].max())
        for P in INJ_P:
            A = np.array(INJ_A)
            mp_ = np.array([inj[f"{P:g}"][f"{a:g}"][case]["median_power_at_Pinj"] for a in INJ_A])
            def amp_for(level):
                k = int(np.searchsorted(mp_, level))
                if k == 0 or k >= len(A):
                    return None
                s = np.log(mp_[k] / mp_[k - 1]) / np.log(A[k] / A[k - 1])
                return float(A[k - 1] * (level / mp_[k - 1]) ** (1 / s))
            thr.setdefault(case, {})[f"{P:g}"] = {"amp_at_null95": amp_for(n95), "amp_at_observed_max": amp_for(omax)}
        thr[case]["null95"] = n95
        thr[case]["observed_max_24_400"] = omax
    out["injection"] = {"periods_h": list(INJ_P), "amplitudes_pp_mag": list(INJ_A), "trials": INJ_N, "results": inj,
                        "power_equivalent_amplitude": thr}
    # ---------------- TESS
    tc = tctx()
    TO = R["tess_obs"]
    PT = 24.0 / tc["F"]
    tess = {"n": int(len(tc["t"])), "segments": [], "variants": {}}
    for k in np.unique(tc["seg"]):
        m = tc["seg"] == k
        tess["segments"].append({"k": int(k), "n": int(m.sum()), "t_start": float(tc["t"][m].min()),
                                 "length_d": float(np.ptp(tc["t"][m])), "JD_start": float(tc["t"][m].min() + c["t0"])})
    # point-to-point noise (successive points < 0.5 h apart)
    o = np.argsort(tc["t"])
    dt = np.diff(tc["t"][o])
    dy = np.diff(tc["y"][o])
    tess["p2p_noise_mag"] = float(np.std(dy[dt < 0.5 / 24]) / np.sqrt(2))
    out["tess_rms_comparison"]["p2p_noise"] = tess["p2p_noise_mag"]
    for v in TESS_VARIANTS:
        tess[f"rms_{v}"] = float(np.std(resid(tc["proj"][v], tc["base"][v])))
    TNULL = {v: null_curves(f"tess_null_{v}_") for v in TESS_VARIANTS + ("slow_seglin",)}
    for v in TESS_VARIANTS:
        p = TO[v]
        nm = TNULL[v]
        mx = nm.max(1)
        yv = resid(tc["proj"][v], tc["base"][v])
        fam = families(tc["F"], p, n=6, lo=2.0, hi=360.0)
        for F in fam:
            fi = 24.0 / F["P"]
            Pex, pex = exact_local(tc["t"], yv, fi, tc["T"], half=2.0)
            F["P_exact"], F["power_exact"] = Pex, pex
            F["p_global"] = pval(F["power"], mx)
            wv = np.abs(tc["F"] - fi) <= 0.5 / 30.0
            F["p_local"] = pval(float(p[wv].max()), nm[:, wv].max(1))
            if v == "seglin":
                sn = TNULL["slow_seglin"][:, wv].max(1)
                F["slow_null_median_power"] = float(np.median(sn))
                F["slow_null_p_local"] = pval(float(p[wv].max()), sn)
        tess["variants"][v] = {"families_2_360": fam, "null_max_median": float(np.median(mx)),
                               "null_max_95": float(np.percentile(mx, 95)), "n_perm": int(len(mx))}
    smx = TNULL["slow_seglin"].max(1)
    sbest = 24.0 / tc["F"][np.argmax(TNULL["slow_seglin"].astype(np.float32), axis=1)]
    nf = TO["slow_seglin_noise_free"]
    tess["slow_signal_null_seglin"] = {
        "description": "QF slow model at the TESS times + permuted QF residuals, seglin detrending",
        "n_real": int(len(smx)), "max_median": float(np.median(smx)), "max_95": float(np.percentile(smx, 95)),
        "best_P_median": float(np.median(sbest)), "best_P_samples": [round(float(x), 1) for x in sbest[:15]],
        "frac_best_150_300": float(np.mean((sbest >= 150) & (sbest <= 300))),
        "noise_free_best_P": float(PT[np.argmax(nf)]), "noise_free_best_power": float(nf.max()),
        "p_obs_seglin_max": pval(float(TO["seglin"].max()), smx)}
    for k in np.unique(tc["app"]):
        for v in ("appmean", "seglin", "qf_seglin"):
            p = TO[f"app{k}_{v}"]
            tess[f"app{k}_{v}_best"] = {"P": float(PT[np.argmax(p)]), "power": float(p.max()),
                                        "families": families(tc["F"], p, n=4, lo=2.0, hi=360.0)}
    WT = TO["W"]
    kt, _ = find_peaks(WT)
    kt = kt[np.argsort(WT[kt])[::-1]][:10]
    tess["window_peaks"] = [{"P": round(float(PT[i]), 2), "W": round(float(WT[i]), 4)} for i in kt]
    ti = {}
    for P in TINJ_P:
        rr = R[f"tess_inj_{P:g}"]["res"]
        ti[f"{P:g}"] = {}
        for A, vs in rr.items():
            ti[f"{P:g}"][A] = {}
            for v, lst in vs.items():
                a = np.array(lst)
                rec = np.abs(a[:, 1] - a[:, 0]) <= 0.02 * a[:, 0]
                ti[f"{P:g}"][A][v] = {"recovered": float(rec.mean()), "median_best_P": float(np.median(a[:, 1])),
                                      "median_power_at_Pinj": float(np.median(a[:, 3]))}
    tess["injection"] = {"periods_h": list(TINJ_P), "amplitudes_pp_mag": list(TINJ_A), "trials": TINJ_N,
                         "criterion": "strongest peak in 2-360 h within 2 % of P_inj", "results": ti}
    # folds at TESS best periods
    tess["folds"] = {}
    for v in ("appmean", "seglin", "qf_seglin"):
        Pb = tess["variants"][v]["families_2_360"][0]["P_exact"]
        fsT = fold_stats(tc["t"], resid(tc["proj"][v], tc["base"][v]), Pb, 4)
        tess["folds"][v] = {k: fsT[k] for k in ("P", "amp_pp", "n_maxima", "rms_before", "rms_after")}
    out["tess_only"] = tess
    # ---------------- DAMIT (from inspect_3216_short.json)
    sj = json.load(open(os.path.join(ROOT, "inspect_3216_short.json")))
    dam = {}
    for name, v in sj["damit_longrange"].items():
        Pd, rd = np.array(v["curve"]["P"]), np.array(v["curve"]["chi2_ratio"])
        dd = {"global_best_P": v["global_best_P"]}
        for lo, hi in ((120, 200), (200, 400)):
            m = (Pd >= lo) & (Pd < hi)
            dd[f"min_chi2_ratio_{lo}_{hi}"] = float(rd[m].min())
            dd[f"P_at_min_{lo}_{hi}"] = float(Pd[m][np.argmin(rd[m])])
        dam[name] = dd
    lrj = json.load(open(os.path.join(ROOT, "longperiod_longrange.json")))[NUM]["subsets"]
    for name in dam:
        dam[name]["threshold"] = lrj[name]["threshold"]
    out["damit_longrange_summary"] = dam
    X2 = analyse2(R, out, OBS, SLOW)
    out["task_seconds"] = {k: round(v.get("_sec", 0), 1) for k, v in R.items()}
    with open(os.path.join(ROOT, "inspect_3216_more.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    # print summary
    for v in list(VARIANTS) + ["geo_QF"]:
        d = pg[v]
        print(f"== {v:6s} best 24-400 {d['best_24_400']['P']:.2f} h ({d['best_24_400']['power']:.3f}, p {d.get('p_global_best_24_400')}); "
              f"best 24-200 {d['best_24_200']['P']:.2f} h ({d['best_24_200']['power']:.3f}, p {d.get('p_global_best_24_200')}); "
              f"null95 {d.get('null', {}).get('max_24_400_95')} / {d.get('null', {}).get('max_24_200_95')}")
        print("        families:", [(round(F['P'], 1), round(F['power'], 3)) for F in d["families_24_400"]])
    for v in ("raw", "QF", "QFT"):
        print("2-24 h", v, pg[v]["best_2_24"]["P"], pg[v]["best_2_24"]["power"], pg[v]["best_2_24"].get("null"), pg[v]["best_2_24"].get("p_global"))
    print("slow null:", json.dumps({v: {k: x for k, x in d.items() if k != "noise_free_families_24_400"} for v, d in sl.items()}, default=float))
    print("TESS:", json.dumps({k: v for k, v in tess.items() if k not in ("injection",)}, default=float)[:4000])
    for P, d in inj.items():
        print("inj", P, {A: (x["raw"]["recovered"], x["QF"]["recovered"], x["QFT"]["recovered"]) for A, x in d.items()})
    for P, d in ti.items():
        print("tess inj", P, {A: (x["appmean"]["recovered"], x["seglin"]["recovered"]) for A, x in d.items()})
    print("SECOND PASS:", json.dumps(out["second_pass"], default=float, indent=0)[:12000])
    plot(out, OBS, NULL, SLOW, R, W, SUBP, sj, X2)


def _local(curve, f, P, T, half=1.0):
    w = np.abs(f - 24.0 / P) <= half / T
    return float(curve[w].max()), w


def analyse2(R, out, OBS, SLOW):
    """second-pass tests (docstring section 8)"""
    c = ctx("original")
    FL, T = c["FL"], c["T"]
    PL = 24.0 / FL
    m200 = PL <= 200.0

    def cat(prefix):
        ks = sorted(k for k in R if k.startswith(prefix))
        return np.concatenate([R[k]["curves"] for k in ks]).astype(np.float32) if ks else None

    sp = {"validation": {k: v for k, v in R["validate"].items() if k != "_sec"}}
    # ---------------- raw (not detrended) power at the review periods in subsets and single apparitions
    yraw = c["y"] - c["y"].mean()
    rs = {}
    for P in REVIEW_P:
        d = {"all": exact_local(c["t"], yraw, 24.0 / P, T)[1]}
        for sub in SUBSETS:
            m = subset_mask(c, sub)
            d[sub] = exact_local(c["t"][m], yraw[m] - yraw[m].mean(), 24.0 / P, T)[1]
        for k in np.unique(c["app"]):
            m = c["app"] == k
            if m.sum() >= 100 and np.ptp(c["t"][m]) >= 30:
                d[f"app{int(k)}"] = exact_local(c["t"][m], yraw[m] - yraw[m].mean(), 24.0 / P, float(np.ptp(c["t"][m])), half=1.0)[1]
        rs[f"{P:g}"] = d
    sp["raw_power_review_periods_subsets"] = rs
    # ---------------- b binned
    b = bctx()
    BO = R["bin_obs"]
    PB = 24.0 / b["F"]
    mb200 = PB <= 200.0
    BN = {v: cat(f"bin_null_{v}_") for v in BIN_VARIANTS}
    bd = {"n_bins": int(len(b["t"])), "n_tess_bins": int(np.sum(b["sh"] == TESS)),
          "n_par": {v: int(b["proj"][v].shape[1]) for v in BIN_VARIANTS},
          "rms": {v: float(np.std(resid(b["proj"][v], b["y"]))) for v in BIN_VARIANTS}, "variants": {}}
    for v in BIN_VARIANTS:
        p, nm = BO[v], BN[v]
        mx, mx200 = nm.max(1), nm[:, mb200].max(1)
        d = {"best_24_400": {"P": float(PB[np.argmax(p)]), "power": float(p.max())},
             "best_24_200": {"P": float(PB[mb200][np.argmax(p[mb200])]), "power": float(p[mb200].max())},
             "families_24_400": families(b["F"], p, n=8), "n_perm": int(len(nm)),
             "null_max_24_400_95": float(np.percentile(mx, 95)), "null_max_24_200_95": float(np.percentile(mx200, 95)),
             "p_global_best_24_400": pval(p.max(), mx), "p_global_best_24_200": pval(p[mb200].max(), mx200), "review": {}}
        for P in REVIEW_P:
            pw, w = _local(p, b["F"], P, b["T"])
            d["review"][f"{P:g}"] = {"power": pw, "p_global_24_400": pval(pw, mx), "p_local": pval(pw, nm[:, w].max(1))}
        bd["variants"][v] = d
    bd["window_at_review"] = {f"{P:g}": _local(BO["W"], b["F"], P, b["T"])[0] for P in REVIEW_P}
    binj = {}
    for P in INJ_P:
        rr = R[f"bin_inj_{P:g}"]["res"]
        binj[f"{P:g}"] = {}
        for A, vs in rr.items():
            binj[f"{P:g}"][A] = {}
            for v, lst in vs.items():
                a = np.array(lst)
                rec = np.abs(a[:, 1] - a[:, 0]) <= 0.01 * a[:, 0]
                n95 = np.percentile(BN[v].max(1), 95)
                binj[f"{P:g}"][A][v] = {"recovered": float(rec.mean()), "recovered_and_above_null95": float(np.mean(rec & (a[:, 2] > n95))),
                                        "median_power_at_Pinj": float(np.median(a[:, 3]))}
    bd["injection"] = {"amplitudes_pp_mag": list(BINJ_A), "trials": INJ_N, "results": binj}

    def alias_check(tt, yq, yqa, P, TT):
        """daily-alias relations of P, residual long periods (300-5000 h) of the QF residuals, power at P after QFA
        and after prewhitening the strongest residual long periods"""
        f0 = 24.0 / P
        fLong = np.arange(24.0 / 5000.0, 24.0 / 300.0, 1.0 / (20 * TT))
        d = {"P": P, "harmonic": harmonic(P),
             "alias_P_h": {"1-f": 24.0 / (1 - f0) if f0 < 1 else None, "2f-1": 24.0 / (2 * f0 - 1) if 2 * f0 > 1 else None},
             "residual_long_families": families(fLong, pfast(tt, yq, fLong), n=5, lo=300.0, hi=5000.0),
             "power_QF": exact_local(tt, yq, f0, TT)[1], "power_QFA": exact_local(tt, yqa, f0, TT)[1], "prewhitening": []}
        yy = yq.copy()
        F2 = np.arange(24.0 / 400.0, 1.0, 1.0 / (20 * TT))
        for _ in range(2):
            pl = pfast(tt, yy, fLong)
            Pr_ = float(24.0 / fLong[np.argmax(pl)])
            X = fourier_X(tt, Pr_, 2)
            bb, *_ = np.linalg.lstsq(X, yy, rcond=None)
            yy = yy - X @ bb
            ps = pfast(tt, yy, F2)
            d["prewhitening"].append({"removed_P": Pr_, "power_at_P": exact_local(tt, yy, f0, TT)[1],
                                      "best_24_400_P": float(24.0 / F2[np.argmax(ps)]), "best_24_400_power": float(ps.max())})
        return d

    ybq, ybqa = resid(b["proj"]["QF"], b["y"]), resid(b["proj"]["QFA"], b["y"])
    bd["alias_checks"] = {f"{P:.1f}": alias_check(b["t"], ybq, ybqa, P, b["T"])
                          for P in (bd["variants"]["QF"]["best_24_200"]["P"], bd["variants"]["QF"]["best_24_400"]["P"])}
    sp["binned"] = bd
    # ---------------- c ground only
    g = gctx()
    GO = R["ground_obs"]
    GN = {v: cat(f"ground_null_{v}_") for v in GROUND_VARIANTS + ("slow",)}
    gd = {"n": int(len(g["t"])), "variants": {}}
    for v in GROUND_VARIANTS:
        p, nm = GO[v], GN[v]
        mx, mx200 = nm.max(1), nm[:, m200].max(1)
        d = {"best_24_400": {"P": float(PL[np.argmax(p)]), "power": float(p.max())},
             "best_24_200": {"P": float(PL[m200][np.argmax(p[m200])]), "power": float(p[m200].max())},
             "families_24_400": families(FL, p, n=8), "n_perm": int(len(nm)),
             "null_max_24_400_95": float(np.percentile(mx, 95)), "null_max_24_200_95": float(np.percentile(mx200, 95)),
             "p_global_best_24_400": pval(p.max(), mx), "p_global_best_24_200": pval(p[m200].max(), mx200), "review": {}}
        for P in REVIEW_P:
            pw, w = _local(p, FL, P, T)
            d["review"][f"{P:g}"] = {"power": pw, "p_global_24_400": pval(pw, mx), "p_local": pval(pw, nm[:, w].max(1))}
        gd["variants"][v] = d
    sn = GN["slow"]
    med = np.median(sn, axis=0)
    gr = GO["raw"]
    gd["slow_null"] = {
        "description": "QF slow model fitted to all data, sampled at the ground epochs, + permuted ground residuals; no detrending",
        "n_real": int(len(sn)), "corr_raw_vs_median_24_400": float(np.corrcoef(gr, med)[0, 1]),
        "corr_raw_vs_median_24_200": float(np.corrcoef(gr[m200], med[m200])[0, 1]),
        "corr_raw_vs_noise_free_24_400": float(np.corrcoef(gr, GO["slow_noise_free"])[0, 1]),
        "noise_free_families_24_400": families(FL, GO["slow_noise_free"], n=6),
        "p_raw_best_24_400": pval(float(gr.max()), sn.max(1)), "p_raw_best_24_200": pval(float(gr[m200].max()), sn[:, m200].max(1)),
        "best_P_24_200_realisations": [round(float(PL[m200][np.argmax(x[m200])]), 1) for x in sn[:15]],
        "review": {f"{P:g}": {"power_obs": _local(gr, FL, P, T)[0],
                              "median_null": float(np.median(sn[:, _local(gr, FL, P, T)[1]].max(1))),
                              "p_local": pval(_local(gr, FL, P, T)[0], sn[:, _local(gr, FL, P, T)[1]].max(1))} for P in REVIEW_P}}
    ygq, ygqa = resid(g["proj"]["QF"], g["y"]), resid(g["proj"]["QFA"], g["y"])
    gd["alias_checks"] = {f"{P:.1f}": alias_check(g["t"], ygq, ygqa, P, g["T"])
                          for P in (gd["variants"]["QF"]["best_24_200"]["P"], gd["variants"]["QF"]["best_24_400"]["P"])}
    sp["ground_only"] = gd
    # ---------------- d slow models without freedom below 400 h
    raw = OBS["raw"]
    m_mid = (PL >= 150) & (PL <= 300)
    smd = {}
    for mdl in SLOW_MODELS:
        if mdl == "QF":
            cur = SLOW["QF"].astype(np.float32)
            nf = R["slownull_QF_0"]["noise_free"]
            info = {"n_par": int(c["proj"]["QF"].shape[1]), "rms_resid": float(np.std(resid(c["proj"]["QF"], c["y"])))}
            info["frac_var_explained"] = float(1 - np.var(resid(c["proj"]["QF"], c["y"])) / np.var(c["y"]))
        else:
            rr = R[f"slowmodel_{mdl}"]
            cur, nf = rr["curves"].astype(np.float32), rr["noise_free"]
            info = {k: rr[k] for k in ("n_par", "rms_resid", "frac_var_explained")}
        medc = np.median(cur, axis=0)
        bestP = PL[m200][np.argmax(cur[:, m200], axis=1)]
        near = np.zeros(len(bestP), bool)
        for q in (185.0, 187.83):
            near |= np.abs(bestP - q) <= 0.02 * q
        d = dict(info)
        d.update({"n_real": int(len(cur)),
                  "corr_raw_vs_noise_free_24_400": float(np.corrcoef(raw, nf)[0, 1]),
                  "corr_raw_vs_noise_free_24_200": float(np.corrcoef(raw[m200], nf[m200])[0, 1]),
                  "corr_raw_vs_noise_free_150_300": float(np.corrcoef(raw[m_mid], nf[m_mid])[0, 1]),
                  "corr_raw_vs_median_24_200": float(np.corrcoef(raw[m200], medc[m200])[0, 1]),
                  "noise_free_families_24_400": families(FL, nf, n=8),
                  "noise_free_best_24_200": {"P": float(PL[m200][np.argmax(nf[m200])]), "power": float(nf[m200].max())},
                  "realisations_best_24_200_P": [round(float(x), 1) for x in bestP[:15]],
                  "frac_realisations_best_24_200_within_2pct_of_185_or_187.8": float(near.mean()),
                  "p_raw_best_24_200": pval(float(raw[m200].max()), cur[:, m200].max(1)),
                  "p_raw_best_24_400": pval(float(raw.max()), cur.max(1)),
                  "review": {f"{P:g}": {"power_obs": _local(raw, FL, P, T)[0],
                                        "noise_free": _local(nf, FL, P, T)[0],
                                        "median_real": float(np.median(cur[:, _local(raw, FL, P, T)[1]].max(1))),
                                        "p_local": pval(_local(raw, FL, P, T)[0], cur[:, _local(raw, FL, P, T)[1]].max(1))}
                             for P in REVIEW_P}})
        smd[mdl] = d
    sp["slow_models"] = smd
    # ---------------- e injection at the observed amplitude
    oa = {}
    for P in REVIEW_P:
        rr = R[f"obsamp_{P:g}"]
        d = {"amp_pp_raw_fold": rr["amp_pp"], "trials": OBSAMP_N}
        for v, lst in rr["res"].items():
            a = np.array(lst)
            rec = np.abs(a[:, 0] - P) <= 0.01 * P
            d[v] = {"recovered_as_strongest_24_400": float(rec.mean()), "power_at_P_median": float(np.median(a[:, 2])),
                    "power_at_P_min": float(a[:, 2].min()), "observed_power_at_P": rr["obs_power_at_P"][v],
                    "ratio_min_injected_to_observed": float(a[:, 2].min() / rr["obs_power_at_P"][v])}
        oa[f"{P:g}"] = d
    sp["observed_amplitude_injection"] = oa
    # ---------------- TESS-only peaks vs TESS window peaks
    tw = out["tess_only"]["window_peaks"]
    co = {}
    for v in ("seglin", "segquad", "qf_seglin", "qf_appmean"):
        rows = []
        seen = []
        for F in out["tess_only"]["variants"][v]["families_2_360"]:
            Pp = F["P_exact"]
            if any(abs(Pp - q) <= 0.01 * q for q in seen):
                continue
            seen.append(Pp)
            nw = min(tw, key=lambda x: abs(x["P"] - Pp))
            rows.append({"P": round(Pp, 1), "power": round(F["power_exact"], 4), "p_global": F["p_global"],
                         "nearest_TESS_window_peak_P": nw["P"], "W": nw["W"], "rel_diff": round((Pp - nw["P"]) / nw["P"], 4)})
        co[v] = rows
    sp["tess_peaks_vs_tess_window"] = co
    # ---------------- f per-apparition free-period slow model (QP)
    named = sorted(set([24.84, 24.9, 46.3, 319.6, 328.4, 331.0] + list(REVIEW_P)))
    qd = {}
    QPC = {}
    for data in ("original", "ground", "bin"):
        o = R[f"qp_obs_{data}"]
        q = qp_context(data)
        F = q["F"]
        PP = 24.0 / F
        mm200 = PP <= 200.0
        nm = cat(f"qp_null_{data}_")
        p = o["p"]
        QPC[data] = (PP, p, nm)
        mx, mx200 = nm.max(1), nm[:, mm200].max(1)
        d = {"P_app": o["P_app"], "n_par": o["n_par"], "rms": o["rms"], "n_perm": int(len(nm)),
             "best_24_400": {"P": float(PP[np.argmax(p)]), "power": float(p.max())},
             "best_24_200": {"P": float(PP[mm200][np.argmax(p[mm200])]), "power": float(p[mm200].max())},
             "families_24_400": families(F, p, n=8), "null_max_24_400_95": float(np.percentile(mx, 95)),
             "null_max_24_200_95": float(np.percentile(mx200, 95)),
             "p_global_best_24_400": pval(p.max(), mx), "p_global_best_24_200": pval(p[mm200].max(), mx200), "named": {}}
        for P in named:
            pw, w = _local(p, F, P, q["T"])
            d["named"][f"{P:g}"] = {"power": pw, "p_global_24_400": pval(pw, mx), "p_local": pval(pw, nm[:, w].max(1))}
        if "pS" in o:
            FS = c["FS"]
            d["best_2_24"] = {"P": float(24.0 / FS[np.argmax(o["pS"])]), "power": float(o["pS"].max()),
                              "families": families(FS, o["pS"], n=5, lo=2.0, hi=24.0)}
        qd[data] = d
    sp["qp_free_period_per_apparition"] = qd
    # ---------------- g TESS test of the short ground-based residual peaks (24.8-24.9 h, 46.3 h)
    tc = tctx()
    TO = R["tess_obs"]
    TN = cat("tess_null_segquad_")
    ysq = resid(tc["proj"]["segquad"], tc["y"])
    yqs = resid(tc["proj"]["qf_seglin"], tc["base"]["qf_seglin"])
    gq = gctx()
    yga = resid(gq["proj"]["QFA"], gq["y"])
    ygq = resid(gq["proj"]["QF"], gq["y"])
    rngT = np.random.default_rng(5)
    randP = rngT.uniform(20.0, 60.0, 200)

    def amp2(tt, yy, P):
        X = fourier_X(tt, P, 2)
        bb, *_ = np.linalg.lstsq(X, yy, rcond=None)
        ph = np.linspace(0, 1, 200, endpoint=False)
        return float(np.ptp(fourier_X(ph * P / 24.0, P, 2)[:, 1:] @ bb[1:])), bb

    hk = np.floor(tc["t"] * 24.0).astype(np.int64)
    _, hinv = np.unique(hk, return_inverse=True)
    hcnt = np.bincount(hinv)

    def xr(P):
        _, bb = amp2(gq["t"], yga, P)
        pred = fourier_X(tc["t"], P, 2)[:, 1:] @ bb[1:]
        o_ = np.bincount(hinv, ysq) / hcnt
        p_ = np.bincount(hinv, pred) / hcnt
        return float(np.corrcoef(o_, p_)[0, 1])

    tt_rand_seg = np.array([amp2(tc["t"], ysq, q)[0] for q in randP])
    tt_rand_qfs = np.array([amp2(tc["t"], yqs, q)[0] for q in randP])
    xr_rand = np.array([xr(q) for q in randP])
    tst = {}
    for P0 in (24.84, 24.9, 46.3):
        Pg, pg_ = exact_local(gq["t"], yga, 24.0 / P0, gq["T"], half=1.0)
        ag, bg = amp2(gq["t"], yga, Pg)
        pw, w = _local(TO["segquad"], tc["F"], Pg, 30.0, half=0.5)
        d = {"P_ground_QFA": Pg, "power_ground_QFA": pg_, "amp_pp_ground_QFA": ag, "amp_pp_ground_QF": amp2(gq["t"], ygq, Pg)[0],
             "amp_pp_tess_segquad": amp2(tc["t"], ysq, Pg)[0],
             "amp_pp_tess_segquad_random_20_60h_median_95": [float(np.median(tt_rand_seg)), float(np.percentile(tt_rand_seg, 95))],
             "amp_pp_tess_qf_seglin": amp2(tc["t"], yqs, Pg)[0],
             "amp_pp_tess_qf_seglin_random_20_60h_median_95": [float(np.median(tt_rand_qfs)), float(np.percentile(tt_rand_qfs, 95))],
             "tess_segquad_power_at_P": pw, "tess_segquad_p_local": pval(pw, TN[:, w].max(1)),
             "xpred_ground_to_tess_r_hourly": xr(Pg),
             "xpred_r_random_20_60h_median_95": [float(np.median(xr_rand)), float(np.percentile(xr_rand, 95))]}
        # the ground-based signal (its amplitude, random phase) injected into the TESS data, segquad detrending
        rec, pws = [], []
        for _ in range(20):
            sig = fourier_X(tc["t"] + rngT.uniform(0, Pg / 24.0), Pg, 2)[:, 1:] @ bg[1:]
            pp = pfast(tc["t"], resid(tc["proj"]["segquad"], tc["y"] + sig), tc["F"])
            rec.append(abs(24.0 / tc["F"][np.argmax(pp)] - Pg) <= 0.02 * Pg)
            pws.append(float(pp[w].max()))
        d["inject_ground_signal_into_tess"] = {"recovered_as_strongest_2_360": float(np.mean(rec)),
                                               "power_at_P_median": float(np.median(pws)), "power_at_P_min": float(np.min(pws))}
        tst[f"{P0:g}"] = d
    sp["tess_test_of_ground_short_peaks"] = tst
    out["second_pass"] = sp
    return {"BO": BO, "BN": BN, "GO": GO, "GN": GN, "SM": {m: (R[f"slowmodel_{m}"]["noise_free"] if m != "QF" else
                                                              R["slownull_QF_0"]["noise_free"]) for m in SLOW_MODELS},
            "QP": QPC}


def plot(out, OBS, NULL, SLOW, R, W, SUBP, sj, X2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = ctx("original")
    FL = c["FL"]
    PL = 24.0 / FL
    ds = slice(None, None, 3)
    fig = plt.figure(figsize=(13, 24))
    gs = fig.add_gridspec(7, 2, height_ratios=[1.25, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], hspace=0.40, wspace=0.22)
    # (a) detrended periodograms 2-400 h
    ax = fig.add_subplot(gs[0, :])
    FS = c["FS"]
    for v, col, lab in (("QF", "#1a5276", "QF: trends + 1266.6 h Fourier series removed"),
                        ("QFT", "#b9770e", "QFT: QF + TESS segment trends removed")):
        ax.plot(PL[ds], OBS[v][ds], color=col, lw=0.6, label=lab)
        ps = R[f"obs_orig_{v}_FS"]["p"]
        ax.plot(24.0 / FS[::5], ps[::5], color=col, lw=0.5)
        ax.hlines(np.percentile(NULL[v].max(1).astype(float), 95), 24, 400, color=col, ls="--", lw=1)
    PPq, pq, _ = X2["QP"]["original"]
    ax.plot(PPq[ds], pq[ds], color="#7d3c98", lw=0.5, alpha=0.8, label="QP: QF + free 560-720 h period per apparition")
    ax.hlines(out["second_pass"]["qp_free_period_per_apparition"]["original"]["null_max_24_400_95"], 24, 400,
              color="#7d3c98", ls="--", lw=1)
    if out["periodograms"]["QF"]["best_2_24"].get("null"):
        ax.hlines(out["periodograms"]["QF"]["best_2_24"]["null"]["max_95"], 2, 24, color="#1a5276", ls="--", lw=1,
                  label="95 % of permuted maxima (dashed)")
    for P in REVIEW_P:
        ax.axvline(P, color="#c0392b", lw=0.6, ls=":", alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlim(2, 400)
    ax.set_ylim(0, 0.045)
    ax.set_ylabel("LS power (nterms = 2)")
    ax.set_xlabel("period (h)")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.text(0.99, 0.95, f"raw data for comparison: max {OBS['raw'].max():.3f} (24-400 h), see (b)", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color="0.3")
    ax.set_title("(a) (3216) after removing the slow signal (note the scale); red dotted: 185.0, 187.8, 214.6, 217.7, "
                 "221.7, 255.1 h", loc="left", fontsize=10)
    # (b) raw vs slow model + noise
    ax = fig.add_subplot(gs[1, :])
    ax.plot(PL[ds], OBS["raw"][ds], color="0.2", lw=0.7, label="observed (raw)")
    for v, col in (("QF", "#1a5276"), ("QFT", "#b9770e")):
        med = np.median(SLOW[v], axis=0)
        hi = np.percentile(SLOW[v], 95, axis=0)
        ax.plot(PL[ds], med[ds], color=col, lw=0.7, alpha=0.9, label=f"slow model {v} + permuted noise: median")
        if v == "QFT":
            ax.fill_between(PL[ds], 0, hi[ds], color=col, alpha=0.15, lw=0, label="QFT: 95 % per frequency")
    for P in REVIEW_P:
        ax.axvline(P, color="#c0392b", lw=0.6, ls=":", alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlim(24, 400)
    ax.set_ylabel("LS power")
    ax.set_xlabel("period (h)")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_title("(b) raw periodogram vs. periodograms of the fitted slow signal (no short period) sampled at the real times",
                 loc="left", fontsize=10)
    # (c) subsets QF
    ax = fig.add_subplot(gs[2, 0])
    for sub, col in (("tess", "#b9770e"), ("not_tess", "#1a5276"), ("odd", "#7d3c98"), ("even", "#239b56")):
        ax.plot(PL[ds], SUBP[sub][ds], color=col, lw=0.6, label=f"QF, {sub}")
    for P in REVIEW_P:
        ax.axvline(P, color="#c0392b", lw=0.6, ls=":", alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlim(24, 400)
    ax.set_ylabel("LS power")
    ax.set_xlabel("period (h)")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.set_title("(c) subsets after QF", loc="left", fontsize=10)
    # (d) TESS alone
    ax = fig.add_subplot(gs[2, 1])
    tc = tctx()
    PT = 24.0 / tc["F"]
    TO = R["tess_obs"]
    for v, col in (("appmean", "0.4"), ("seglin", "#b9770e"), ("segquad", "#7d3c98"), ("qf_seglin", "#1a5276")):
        ax.plot(PT, TO[v], color=col, lw=0.6, label=f"TESS, {v}")
        ax.axhline(out["tess_only"]["variants"][v]["null_max_95"], color=col, ls="--", lw=0.8)
    TS = np.concatenate([R[k]["curves"] for k in sorted(R) if k.startswith("tess_null_slow_seglin_")]).astype(np.float32)
    ax.plot(PT, np.median(TS, axis=0), color="#b9770e", lw=0.8, ls=(0, (1, 1)), label="slow model + noise, seglin: median")
    ax.plot(PT, TO["W"] * ax.get_ylim()[1] / max(TO["W"].max(), 1e-9) * 0.5, color="0.75", lw=0.5, label="TESS window (scaled)")
    for P in REVIEW_P:
        ax.axvline(P, color="#c0392b", lw=0.6, ls=":", alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlim(2, 360)
    ax.set_xlabel("period (h)")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.set_title("(d) TESS (C57G) alone; dashed: 95 % of permuted maxima", loc="left", fontsize=10)
    # (e) injection recovery
    ax = fig.add_subplot(gs[3, 0])
    inj = out["injection"]["results"]
    cols = {"0.05": "#d4ac0d", "0.1": "#ca6f1e", "0.2": "#c0392b", "0.3": "#7d3c98", "0.73": "#1a5276"}
    for A in [f"{a:g}" for a in INJ_A]:
        for case, ls in (("raw", ":"), ("QF", "-"), ("QFT", "--")):
            ax.plot(INJ_P, [inj[f"{P:g}"][A][case]["recovered"] for P in INJ_P], ls=ls, color=cols[A], marker="o", ms=3,
                    lw=1, label=f"{A} mag" if case == "QF" else None)
    ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("injected period (h)")
    ax.set_ylabel("fraction recovered")
    ax.legend(fontsize=7, frameon=False, title="peak-to-peak (solid QF, dashed QFT, dotted raw)", title_fontsize=7,
              loc="lower left")
    ax.set_title("(e) injection-recovery, combined data (24-400 h)", loc="left", fontsize=10)
    # (f) TESS time series with the global slow model (QF, fitted to all data)
    sub = gs[3, 1].subgridspec(1, 2, width_ratios=[1.0, 3.2], wspace=0.06)
    yT = resid(tc["proj"]["appmean"], tc["y"])
    mT = tc["mslow"] - (tc["y"] - yT)  # slow model with the same per-apparition means removed
    for j, (lo, hi) in enumerate(((5403, 5428), (5917, 5984))):
        ax = fig.add_subplot(sub[0, j])
        m = (tc["t"] >= lo) & (tc["t"] <= hi)
        ax.plot(tc["t"][m], yT[m], ".", ms=1, color="0.6", label="TESS (10 min)" if j else None)
        o = np.argsort(tc["t"][m])
        tt, mm = tc["t"][m][o], mT[m][o]
        brk = np.where(np.diff(tt) > 1.0)[0]
        for g_ in np.split(np.arange(len(tt)), brk + 1):
            ax.plot(tt[g_], mm[g_], color="#1a5276", lw=1.2, label="slow model (QF, all data)" if (j and g_[0] == 0) else None)
        ax.set_xlim(lo, hi)
        ax.set_ylim(0.9, -0.9)
        ax.set_xlabel("t (d)")
        if j == 0:
            ax.set_ylabel("mag (apparition mean removed)")
            ax.set_title("(f) TESS: 2020 | 2021", loc="left", fontsize=10)
        else:
            ax.set_yticklabels([])
            ax.legend(fontsize=7, frameon=False, loc="lower right")
    # (g),(h) folds at 187.83 and 214.6 h: raw data vs QF residuals (binned)
    rows = out["candidates"]
    pick = [min(rows, key=lambda r: abs(r["P_seed"] - q)) for q in (187.83, 214.6)]
    Yraw = resid(c["proj"]["raw"], c["y"])
    Yqf = resid(c["proj"]["QF"], c["y"])
    tess_m = c["sh"] == TESS
    for j, r in enumerate(pick):
        ax = fig.add_subplot(gs[4, j])
        x = (np.arange(20) + 0.5) / 20
        for lab, yy, P, mk in (("raw", Yraw, r["variants"]["raw"]["P"], "s"), ("QF residual", Yqf, r["P_QF"], "o")):
            ph = (c["t"] * 24.0 / P) % 1.0
            idx = np.minimum((ph * 20).astype(int), 19)
            for sl, col in (("TESS", "#b9770e"), ("other", "#1a5276")):
                m = tess_m if sl == "TESS" else ~tess_m
                b = np.array([yy[m][idx[m] == k].mean() if np.any(idx[m] == k) else np.nan for k in range(20)])
                ax.plot(np.r_[x, x + 1], np.r_[b, b], mk, ms=4 if mk == "o" else 3.5, color=col,
                        mfc=col if mk == "o" else "none", label=f"{lab}, {sl}")
        ax.invert_yaxis()
        ax.set_xlabel("phase")
        ax.set_ylabel("mag (binned, 20 bins)")
        ax.legend(fontsize=7, frameon=False, ncol=2)
        ax.set_title(f"({'gh'[j]}) fold at {r['P_seed']:.1f} h: raw amp {r['fold_raw']['amp_pp']:.2f} mag (power "
                     f"{r['variants']['raw']['power']:.3f}) -> QF amp {r['fold_QF']['amp_pp']:.2f} mag "
                     f"({r['variants']['QF']['power']:.3f})", loc="left", fontsize=9)
    # ---------------- second pass
    sp = out["second_pass"]
    # (i) binned data
    ax = fig.add_subplot(gs[5, 0])
    b = bctx()
    PB = 24.0 / b["F"]
    for v, col in (("raw", "0.25"), ("QF", "#1a5276")):
        ax.plot(PB[ds], X2["BO"][v][ds], color=col, lw=0.6, label=f"binned, {v}")
        ax.axhline(sp["binned"]["variants"][v]["null_max_24_400_95"], color=col, ls="--", lw=0.8)
    PPq, pq, _ = X2["QP"]["bin"]
    ax.plot(PPq[ds], pq[ds], color="#7d3c98", lw=0.6, label="binned, QP (free period per apparition)")
    ax.axhline(sp["qp_free_period_per_apparition"]["bin"]["null_max_24_400_95"], color="#7d3c98", ls="--", lw=0.8)
    for P in REVIEW_P:
        ax.axvline(P, color="#c0392b", lw=0.6, ls=":", alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlim(24, 400)
    ax.set_xlabel("period (h)")
    ax.set_ylabel("LS power")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.set_title(f"(i) binned: 1 point per dataset-night (TESS 6 h), n = {sp['binned']['n_bins']}; dashed: 95 % perm.",
                 loc="left", fontsize=9)
    # (j) ground-based only
    ax = fig.add_subplot(gs[5, 1])
    ax.plot(PL[ds], X2["GO"]["raw"][ds], color="0.25", lw=0.6, label="ground only, raw")
    ax.plot(PL[ds], np.median(X2["GN"]["slow"], axis=0)[ds].astype(float), color="#b9770e", lw=0.7,
            label="all-data slow model at ground epochs + noise: median")
    ax.plot(PL[ds], X2["GO"]["QF"][ds], color="#1a5276", lw=0.6, label="ground only, QF")
    ax.axhline(sp["ground_only"]["variants"]["QF"]["null_max_24_400_95"], color="#1a5276", ls="--", lw=0.8)
    PPq, pq, _ = X2["QP"]["ground"]
    ax.plot(PPq[ds], pq[ds], color="#7d3c98", lw=0.6, label="ground only, QP")
    ax.axhline(sp["qp_free_period_per_apparition"]["ground"]["null_max_24_400_95"], color="#7d3c98", ls="--", lw=0.8)
    for P in REVIEW_P:
        ax.axvline(P, color="#c0392b", lw=0.6, ls=":", alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlim(24, 400)
    ax.set_xlabel("period (h)")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.set_title(f"(j) ground-based stations only (n = {sp['ground_only']['n']})", loc="left", fontsize=9)
    # (k) slow models without freedom below 400 h, noise-free, vs observed raw
    ax = fig.add_subplot(gs[6, 0])
    ax.plot(PL[ds], OBS["raw"][ds], color="0.2", lw=0.8, label="observed (raw)")
    for mdl, col in (("M3", "#c0392b"), ("Q", "#7d3c98")):
        dd = sp["slow_models"][mdl]
        ax.plot(PL[ds], X2["SM"][mdl][ds], color=col, lw=0.7, alpha=0.85,
                label=f"model {mdl} only ({dd['n_par']} par.), r(24-200 h) = {dd['corr_raw_vs_noise_free_24_200']:.2f}")
    for P in REVIEW_P:
        ax.axvline(P, color="#c0392b", lw=0.6, ls=":", alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlim(24, 400)
    ax.set_xlabel("period (h)")
    ax.set_ylabel("LS power")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.set_title("(k) noise-free slow models at the real epochs; M3 has no term < 400 h", loc="left", fontsize=9)
    # (l) observed-amplitude injection
    ax = fig.add_subplot(gs[6, 1])
    xs = np.arange(len(REVIEW_P))
    for j, P in enumerate(REVIEW_P):
        rr = R[f"obsamp_{P:g}"]
        for v, dx, col in (("QF", -0.12, "#1a5276"), ("QFT", 0.12, "#b9770e")):
            a = np.array(rr["res"][v])
            ax.plot(np.full(len(a), j + dx), a[:, 2], "o", ms=3, color=col, alpha=0.6,
                    label=(f"{v}: raw-fold signal injected" if j == 0 else None))
            ax.plot(j + dx, rr["obs_power_at_P"][v], "*", ms=10, color=col, mec="k", mew=0.5,
                    label=(f"{v}: observed" if j == 0 else None))
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{P:g} h\n{R[f'obsamp_{P:g}']['amp_pp']:.2f} mag" for P in REVIEW_P], fontsize=7)
    ax.set_yscale("log")
    ax.set_ylabel("power at P after detrending")
    ax.legend(fontsize=7, frameon=False, loc="center right")
    ax.set_title("(l) raw-data fold at P (observed amplitude) injected, then detrended", loc="left", fontsize=9)
    fig.savefig(os.path.join(ROOT, "review", "inspect_3216_more.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)
    # second figure: DAMIT landscape (the panel of inspect_3216_short.png is empty: ratios > 1.6)
    fig, ax = plt.subplots(figsize=(9, 3.2))
    for name, col in (("all", "#1a5276"), ("odd", "#7d3c98"), ("even", "#b9770e")):
        cc = sj["damit_longrange"][name]["curve"]
        ax.plot(cc["P"], cc["chi2_ratio"], ".", ms=2, color=col, label=f"convex inversion, {name}")
    ax.axvspan(120, 400, color="0.9", zorder=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("period (h)")
    ax.set_ylabel("chi2 / chi2(best)")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("(3216) convex-inversion scan 120-3000 h (grey: 120-400 h)", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "review", "inspect_3216_more_damit.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1:])
