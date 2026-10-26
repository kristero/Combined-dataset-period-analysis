#!/usr/bin/env python
"""(1717) Arlon: prewhitening of the two rotation signals in the combined MPC data and search for the binary
orbital period.

Literature (see binary_1717.md): Cooney et al. 2006, CBET 369 and 504 -- two additive light-curve components
with P = 5.1484 h (0.08 mag) and 18.236 h (0.13 mag); mutual events repeating with a synodic orbital period of
117.0 h (Feb-Apr 2006); D2/D1 >= 0.5. Franco et al. 2019 (MPB 46, 350): 5.1448 h and 18.24 h, deep attenuations
in 5 of 13 sessions of 2019 Jan 22 - Mar 1 (orbital period not determined).

Steps
  1. refine f(18.21 h) and f(5.1475 h) jointly (non-linear least squares, variable projection) for the combined,
     ground-only and TESS (C57G) data; formal and night-block-bootstrap uncertainties;
  2. Lomb-Scargle (nterms=2, astropy 'standard' normalisation, 1-1500 h) of
       (a) data - Fourier model (orders 2-4) of 18.21 h,  (b) data - model of 5.1475 h,
       (c) data - joint model of both (orders 2-4), plus a variant with a linear trend per TESS segment;
     permutation significance (night blocks shuffled within each dataset+apparition);
  3. alias / commensurability checks of f2, f1 and the literature orbital frequency, and a test for
     combination frequencies m1*f1+m2*f2 (expected for non-principal-axis rotation, not for two bodies);
  4. mutual-event (dip) searches on the double-prewhitened residuals: BLS on TESS, folded-binned maximum of the
     night means stacked over apparitions for the ground data (all apparitions and the subsets observed within
     5 and 10 deg of the phase-angle-bisector directions of the reported event windows of 2006 and 2019, or their
     antipodes), permutation significance and injection-recovery of synthetic events at 117.0 h; a tail test
     (excess of faint night means in the event geometry) and the list of event-geometry nights;
  5. per-apparition amplitudes of the two components.

Inputs (read only): results/1717/1717_arrays.npz, data/horizons/1717_geocentric_daily.csv
Outputs: binary_1717.json, review/binary_1717.png
Usage:   python binary_1717.py [--quick]      (--quick: few permutations, for testing)
At most 4 worker processes are used.
"""
import os
import sys
import json
import time
import argparse
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(ROOT, "results", "1717", "1717_arrays.npz")
HOR = os.path.join(ROOT, "data", "horizons", "1717_geocentric_daily.csv")
OUT_JSON = os.path.join(ROOT, "binary_1717.json")
OUT_PNG = os.path.join(ROOT, "review", "binary_1717.png")

P2_START, P1_START = 18.2137, 5.14753          # adopted.json: P and the alternative
LIT = {"P1_CBET369": 5.1484, "A1_CBET369": 0.08, "P2_CBET369": 18.236, "A2_CBET369": 0.13,
       "P1_CBET504": 5.148, "P2_CBET504": 18.23, "Porb_syn_CBET504": 117.0, "P1_LCDB": 5.1484}
PORB = 117.0
SIG_PORB = 0.5     # h; NOT given in CBET 504 -- assumed (covers rounding and synodic-sidereal differences)
SIDEREAL_CPD = 1.0027379
P_MIN, P_MAX = 1.0, 1500.0
SPP = 5
N_WORKERS = 4

# ----------------------------------------------------------------------------------------------------------
# fast multi-term Lomb-Scargle (identical to astropy LombScargle(nterms).power(method='chi2'), ~10x faster)
# ----------------------------------------------------------------------------------------------------------


def ls2_power(t, y, f0, df, nf, nterms=2, chunk=200000):
    from astropy.timeseries.periodograms.lombscargle.implementations.utils import trig_sum
    t = np.asarray(t, float)
    y = np.asarray(y, float) - np.mean(y)
    chi2_ref = float(np.dot(y, y))
    ones = np.ones_like(t)
    S, C, YS, YC = {}, {}, {}, {}
    for m in range(1, 2 * nterms + 1):
        S[m], C[m] = trig_sum(t, ones, df, nf, f0=f0, freq_factor=m, use_fft=True)
    for k in range(1, nterms + 1):
        YS[k], YC[k] = trig_sum(t, y, df, nf, f0=f0, freq_factor=k, use_fft=True)
    n = float(len(t))
    nb = 1 + 2 * nterms
    out = np.empty(nf)
    for a in range(0, nf, chunk):
        sl = slice(a, min(nf, a + chunk))
        L = sl.stop - sl.start

        def Cm(m):
            return np.full(L, n) if m == 0 else C[abs(m)][sl]

        def Sm(m):
            return np.zeros(L) if m == 0 else np.sign(m) * S[abs(m)][sl]
        A = np.empty((L, nb, nb))
        b = np.zeros((L, nb))
        A[:, 0, 0] = n
        for j in range(1, nterms + 1):
            A[:, 0, 2 * j - 1] = A[:, 2 * j - 1, 0] = C[j][sl]
            A[:, 0, 2 * j] = A[:, 2 * j, 0] = S[j][sl]
            b[:, 2 * j - 1] = YC[j][sl]
            b[:, 2 * j] = YS[j][sl]
            for k in range(1, nterms + 1):
                A[:, 2 * j - 1, 2 * k - 1] = 0.5 * (Cm(j - k) + Cm(j + k))
                A[:, 2 * j, 2 * k] = 0.5 * (Cm(j - k) - Cm(j + k))
                sc = 0.5 * (Sm(j + k) + Sm(j - k))
                A[:, 2 * j, 2 * k - 1] = sc
                A[:, 2 * k - 1, 2 * j] = sc
        beta = np.linalg.solve(A, b[..., None])[..., 0]
        out[sl] = np.einsum("ij,ij->i", b, beta) / chi2_ref
    return out


def grid(t, pmin=P_MIN, pmax=P_MAX, spp=SPP):
    T = float(np.max(t) - np.min(t))
    df = 1.0 / (spp * T)
    f0 = 24.0 / pmax
    nf = int((24.0 / pmin - f0) / df) + 1
    return f0, df, nf


def periodogram(t, y, **kw):
    f0, df, nf = grid(t, **kw)
    return f0 + df * np.arange(nf), ls2_power(t, y, f0, df, nf)


def families(f, p, top=10, frac=0.01, pmax=None):
    from scipy.signal import find_peaks
    P = 24.0 / f
    pk, _ = find_peaks(p)
    if pmax is not None:
        pk = pk[P[pk] <= pmax]
    pk = pk[np.argsort(p[pk])[::-1]]
    fams = []
    for i in pk:
        if any(abs(P[i] - q["P"]) <= frac * q["P"] for q in fams):
            continue
        fams.append({"P": float(P[i]), "f": float(f[i]), "power": float(p[i])})
        if len(fams) >= top:
            break
    return fams


def local_max(f, p, fc, hw):
    m = np.abs(f - fc) <= hw
    if not m.any():
        return None, None
    i = np.where(m)[0][np.argmax(p[m])]
    return float(24.0 / f[i]), float(p[i])


# ----------------------------------------------------------------------------------------------------------
# data, blocks, groups, geometry
# ----------------------------------------------------------------------------------------------------------


def load():
    import pandas as pd
    d = np.load(NPZ, allow_pickle=True)
    t, y, sheet = d["t"].astype(float), d["y"].astype(float), d["sheet"].astype(str)
    tz = float(d["time_zero"])
    # apparitions: gaps > 100 d in the combined data
    o = np.argsort(t)
    gap = np.diff(t[o]) > 100.0
    app_sorted = np.concatenate([[0], np.cumsum(gap)])
    app = np.empty(len(t), int)
    app[o] = app_sorted
    # night blocks (ground: gaps > 0.25 d within a dataset; TESS: 1-d chunks)
    blk = np.empty(len(t), int)
    nb = 0
    for sh in np.unique(sheet):
        idx = np.where(sheet == sh)[0]
        idx = idx[np.argsort(t[idx])]
        if sh == "C57G":
            key = np.floor(t[idx] - t[idx].min()).astype(int)
            _, inv = np.unique(key, return_inverse=True)
        else:
            inv = np.concatenate([[0], np.cumsum(np.diff(t[idx]) > 0.25)])
        blk[idx] = nb + inv
        nb += int(inv.max()) + 1
    # shuffle groups: dataset x apparition
    keys = np.array([f"{s}|{a}" for s, a in zip(sheet, app)])
    _, grp = np.unique(keys, return_inverse=True)
    # TESS segments (gaps > 1.5 d) for the linear-trend nuisance terms
    seg = np.full(len(t), -1)
    mt = sheet == "C57G"
    it = np.where(mt)[0]
    it = it[np.argsort(t[it])]
    seg[it] = np.concatenate([[0], np.cumsum(np.diff(t[it]) > 1.5)])
    h = pd.read_csv(HOR)
    lon = np.degrees(np.unwrap(np.radians(h.pab_lon.values)))
    pab = np.interp(t + tz, h.jd.values, lon) % 360.0
    pablat = np.interp(t + tz, h.jd.values, h.pab_lat.values)
    alpha = np.interp(t + tz, h.jd.values, h.alpha.values)
    return dict(t=t, y=y, sheet=sheet, tz=tz, app=app, blk=blk, grp=grp, seg=seg, pab=pab, pablat=pablat,
                alpha=alpha, hor=h)


EVENT_WINDOWS = {"events_2006_Feb24_Apr17_CBET504": ("2006-02-24", "2006-04-17"),
                 "events_2019_Jan22_Mar01_Franco2019": ("2019-01-22", "2019-03-01")}
NOEVENT_WINDOWS = {"no_events_2005_Dec04_2006_Jan11_CBET369": ("2005-12-04", "2006-01-11")}


def event_season(hor):
    """PAB geometry during the windows with reported mutual events (CBET 504: 2006 Feb 24 - Apr 17; Franco et al.
    2019, MPB 46, 350: deep attenuations in 5 of 13 sessions in 2019 Jan 22 - Mar 1) and during the 2005 Dec 4 -
    2006 Jan 11 run without events (CBET 369). Returns the summary and the daily PAB unit vectors of the event
    windows."""
    from astropy.time import Time

    def sel(a, b):
        ja, jb = Time(a).jd, Time(b).jd
        return (hor.jd >= ja - 0.5) & (hor.jd <= jb + 0.5)

    def rng(m):
        return [float(hor.pab_lon[m].min()), float(hor.pab_lon[m].max()),
                float(hor.pab_lat[m].min()), float(hor.pab_lat[m].max()),
                float(hor.alpha[m].min()), float(hor.alpha[m].max())]
    out = {}
    vecs = []
    for k, (a, b) in EVENT_WINDOWS.items():
        m = sel(a, b)
        out[k] = rng(m)
        vecs.append(unitvec(hor.pab_lon[m].values, hor.pab_lat[m].values))
    for k, (a, b) in NOEVENT_WINDOWS.items():
        out[k] = rng(sel(a, b))
    out["columns"] = "pab_lon_min, pab_lon_max, pab_lat_min, pab_lat_max, alpha_min, alpha_max (deg)"
    return out, np.vstack(vecs)


def unitvec(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def event_distance(pab_lon, pab_lat, ev_vecs):
    """angular distance (deg) of the PAB direction from the nearest PAB direction of the reported event windows or
    its antipode (mutual-event geometry is the same for opposite lines of sight)."""
    u = unitvec(pab_lon, pab_lat)
    c = np.abs(u @ ev_vecs.T).max(axis=1)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


# ----------------------------------------------------------------------------------------------------------
# linear models
# ----------------------------------------------------------------------------------------------------------


def design(t, comps, offset_lab=None, nuis=None):
    """comps: list of (f [c/d], order, coef_lab or None). coef_lab: int labels -> separate Fourier
    coefficients per label (label -1 = shared)."""
    cols = []
    if offset_lab is None:
        cols.append(np.ones_like(t))
    else:
        for L in np.unique(offset_lab):
            cols.append((offset_lab == L).astype(float))
    for f, k, lab in comps:
        for j in range(1, k + 1):
            c, s = np.cos(2 * np.pi * j * f * t), np.sin(2 * np.pi * j * f * t)
            if lab is None:
                cols += [c, s]
            else:
                for L in np.unique(lab):
                    m = (lab == L).astype(float)
                    cols += [c * m, s * m]
    if nuis is not None:
        cols += [nuis[:, i] for i in range(nuis.shape[1])]
    return np.column_stack(cols)


def lsfit(A, y):
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return A @ c, c


def tess_nuisance(t, seg):
    """offset + linear slope per TESS segment (zero for ground data)."""
    cols = []
    for s in np.unique(seg[seg >= 0]):
        m = seg == s
        tm = np.mean(t[m])
        cols.append(m.astype(float))
        cols.append(np.where(m, t - tm, 0.0))
    return np.column_stack(cols) if cols else None


def refine(t, y, freqs, orders, labs=None, offset_lab=None, nuis=None):
    """joint non-linear least squares for the frequencies (variable projection). Returns f, formal sigma,
    rms, BIC."""
    from scipy.optimize import least_squares
    labs = labs or [None] * len(freqs)
    T = float(t.max() - t.min())

    def res(fr):
        A = design(t, [(fr[i], orders[i], labs[i]) for i in range(len(fr))], offset_lab, nuis)
        return y - lsfit(A, y)[0]
    step = (1e-3 / (2 * np.pi * T)) / np.asarray(freqs)
    r = least_squares(res, np.asarray(freqs, float), diff_step=step, x_scale=1.0 / T, xtol=1e-12, ftol=1e-12)
    A = design(t, [(r.x[i], orders[i], labs[i]) for i in range(len(freqs))], offset_lab, nuis)
    npar = np.linalg.matrix_rank(A) + len(freqs)
    rss = float(np.sum(r.fun ** 2))
    s2 = rss / (len(y) - npar)
    J = r.jac
    cov = np.linalg.inv(J.T @ J) * s2
    bic = len(y) * np.log(rss / len(y)) + npar * np.log(len(y))
    return r.x, np.sqrt(np.diag(cov)), float(np.sqrt(s2)), float(bic)


# ----------------------------------------------------------------------------------------------------------
# permutation machinery (night blocks shuffled within dataset x apparition)
# ----------------------------------------------------------------------------------------------------------


def block_structure(blk, grp):
    out = []
    for g in np.unique(grp):
        bl = np.unique(blk[grp == g])
        out.append([np.where(blk == b)[0] for b in bl])
    return out


def relabel_times(t, struct, rng):
    """move every night block (with its internal time offsets) to the start time of another night of the same
    dataset+apparition: destroys inter-night coherence, keeps within-night structure, nightly offsets and the
    set of observing epochs."""
    tn = t.copy()
    for blocks in struct:
        if len(blocks) < 2:
            continue
        starts = np.array([t[b].min() for b in blocks])
        perm = rng.permutation(len(blocks))
        for i, b in enumerate(blocks):
            tn[b] = starts[perm[i]] + (t[b] - starts[i])
    return tn


_W = {}


def _init_worker(payload):
    _W.update(payload)


def _perm_ls(args):
    """one permutation of one residual series: returns max power over (1-1500 h), (1-240 h), orbital windows,
    and the given target windows."""
    key, seed = args
    d = _W[key]
    rng = np.random.default_rng(seed)
    tn = relabel_times(d["t"], d["struct"], rng)
    f0, df, nf = d["grid"]
    p = ls2_power(tn, d["r"], f0, df, nf)
    f = f0 + df * np.arange(nf)
    P = 24.0 / f
    out = {"global": float(p.max()), "primary": float(p[P <= 240.0].max())}
    if d.get("daily", True):
        pm = (P <= 240.0) & ~daycomm_mask(f)
        out["primary_excl_daycomm"] = float(p[pm].max())
    else:
        out["primary_excl_daycomm"] = out["primary"]
    om = np.zeros(nf, bool)
    for lo, hi in d["orbwin"]:
        om |= (f >= lo) & (f <= hi)
    out["orbit_windows"] = float(p[om].max()) if om.any() else None
    for nm, (fc, hw) in d.get("targets", {}).items():
        mm = np.abs(f - fc) <= hw
        out["target_" + nm] = float(p[mm].max())
    return out


def _perm_bls(seed):
    d = _W["bls"]
    rng = np.random.default_rng(seed)
    tn = relabel_times(d["t"], d["struct"], rng)
    s_ = np.asarray(bls_run(tn, d["r"], d["per"], d["durs"], d["sig"]).power)
    return float(s_.max()), float(s_[d["win"]].max())


def orbit_windows(porb=PORB, sig=SIG_PORB, nmax=4, kmax=2):
    """frequency windows (c/d) of the literature orbital period, its harmonics n*forb (n<=4) and their daily
    aliases n*forb +- k (k<=2, n<=2). Half-width: 3*sigma(Porb) mapped to frequency, at least 0.5 resolution
    element of a single apparition is irrelevant here (coherent combined data)."""
    flo, fhi = 24.0 / (porb + 3 * sig), 24.0 / (porb - 3 * sig)
    wins = {}
    for n in range(1, nmax + 1):
        wins[f"{n}forb"] = (n * flo, n * fhi)
    for n in (1, 2):
        for k in range(1, kmax + 1):
            for sgn in (+1, -1):
                lo, hi = n * flo + sgn * k, n * fhi + sgn * k
                if lo > 24.0 / P_MAX:
                    wins[f"{n}forb{'+' if sgn > 0 else '-'}{k}"] = (lo, hi)
    return wins


# ----------------------------------------------------------------------------------------------------------
# dip searches
# ----------------------------------------------------------------------------------------------------------


def running_median(t, r, width):
    o = np.argsort(t)
    ts, rs = t[o], r[o]
    lo = np.searchsorted(ts, ts - width / 2)
    hi = np.searchsorted(ts, ts + width / 2)
    med = np.array([np.median(rs[a:b]) for a, b in zip(lo, hi)])
    out = np.empty_like(r)
    out[o] = med
    return out


def bls_run(t, r, periods_d, durations_d, sig):
    from astropy.timeseries import BoxLeastSquares
    bls = BoxLeastSquares(t, -r, dy=np.full(len(t), sig))     # magnitude residual -> fading = flux dip
    res = bls.power(periods_d, durations_d, objective="snr")
    return res


def night_table(t, r, blk, sel):
    """night means of residuals for the selected points; weights from pooled within-night scatter plus a
    night-to-night jitter term."""
    bl = np.unique(blk[sel])
    tb, mb, nb, ib = [], [], [], []
    within = []
    for b in bl:
        m = sel & (blk == b)
        tb.append(t[m].mean())
        mb.append(r[m].mean())
        nb.append(m.sum())
        ib.append(np.where(m)[0])
        if m.sum() >= 2:
            within.append(r[m] - r[m].mean())
    tb, mb, nb = np.array(tb), np.array(mb), np.array(nb)
    w_all = np.concatenate(within) if within else np.array([0.0])
    s_pt = float(np.sqrt(np.sum(w_all ** 2) / max(1, len(w_all) - len(within))))
    s_night2 = max(0.0, float(np.var(mb) - np.mean(s_pt ** 2 / nb)))
    w = 1.0 / (s_night2 + s_pt ** 2 / nb)
    return dict(t=tb, m=mb, n=nb, w=w, idx=ib, s_pt=s_pt, s_night=float(np.sqrt(s_night2)))


def period_grid_dip(pmin, pmax, Tmax, wbin):
    P = [pmin]
    while P[-1] < pmax:
        P.append(P[-1] + (wbin / 4.0) * P[-1] ** 2 / (24.0 * Tmax))
    return np.array(P)


def fold_index(tb, P_h, wbin, off):
    nbins = int(round(1.0 / wbin))
    ph = (tb[None, :] * 24.0 / P_h[:, None] + off) % 1.0
    bi = np.minimum((ph / wbin).astype(np.int32), nbins - 1)
    return (bi + nbins * np.arange(len(P_h), dtype=np.int32)[:, None]).ravel(), nbins


def fold_zmax(flat, nbins, nP, m, w, min_n=2):
    sw = np.bincount(flat, weights=np.tile(w, nP), minlength=nP * nbins)
    swm = np.bincount(flat, weights=np.tile(w * m, nP), minlength=nP * nbins)
    cnt = np.bincount(flat, minlength=nP * nbins)
    z = np.where(cnt >= min_n, swm / np.sqrt(np.maximum(sw, 1e-30)), -np.inf)
    z = z.reshape(nP, nbins)
    k = np.argmax(z, axis=1)
    return z[np.arange(nP), k], k


class DipStack:
    """folded-binned maximum of night means (positive = fainter), maximised over phase per apparition and summed
    over apparitions; offsets 0 and half a bin."""

    def __init__(self, nt, app_of_night, P_h, wbin=0.05, min_nights=15):
        self.P, self.wbin = P_h, wbin
        self.apps = [a for a in np.unique(app_of_night) if np.sum(app_of_night == a) >= min_nights]
        self.sel = {a: np.where(app_of_night == a)[0] for a in self.apps}
        self.idx = {}
        for a in self.apps:
            tb = nt["t"][self.sel[a]]
            self.idx[a] = [fold_index(tb, P_h, wbin, off) for off in (0.0, wbin / 2)]

    def per_app(self, m, w):
        out = {}
        for a in self.apps:
            s = self.sel[a]
            zs = [fold_zmax(fl, nb, len(self.P), m[s], w[s])[0] for fl, nb in self.idx[a]]
            out[a] = np.maximum(zs[0], zs[1])
        return out

    def stack(self, m, w):
        pa = self.per_app(m, w)
        return np.sum([pa[a] for a in self.apps], axis=0), pa


def permute_nights(m, w, grp_night, rng):
    mp, wp = m.copy(), w.copy()
    for g in np.unique(grp_night):
        i = np.where(grp_night == g)[0]
        if len(i) > 1:
            j = rng.permutation(i)
            mp[i], wp[i] = m[j], w[j]
    return mp, wp


def inject_events(t, P_h, phase0, depth, dur_h, second=True):
    ph = (t * 24.0 / P_h - phase0) % 1.0
    hw = 0.5 * dur_h / P_h
    inev = (ph < hw) | (ph > 1 - hw)
    if second:
        inev |= np.abs(ph - 0.5) < hw
    return depth * inev


# ----------------------------------------------------------------------------------------------------------
# misc. relations
# ----------------------------------------------------------------------------------------------------------


def nearest_ratio(x, qmax=10):
    best = []
    for q in range(1, qmax + 1):
        p = int(round(x * q))
        if p > 0:
            best.append((abs(x - p / q), p, q))
    best.sort()
    return [{"p": b[1], "q": b[2], "ratio": b[1] / b[2], "rel_diff": float((x - b[1] / b[2]) / x)} for b in best[:3]]


def alias_relations(f_a, f_b, tol, names=("a", "b"), amax=4, kmax=4):
    """all |m_a f_a + m_b f_b - k c| < tol for c in (1 solar, 1 sidereal day^-1)"""
    rel = []
    for ma in range(-amax, amax + 1):
        for mb in range(-amax, amax + 1):
            if ma == 0 and mb == 0:
                continue
            v = ma * f_a + mb * f_b
            for k in range(0, kmax + 1):
                for c, cn in ((1.0, "solar"), (SIDEREAL_CPD, "sidereal")):
                    if abs(abs(v) - k * c) < tol:
                        rel.append({"m_" + names[0]: ma, "m_" + names[1]: mb, "k": k, "day": cn,
                                    "residual_cpd": float(abs(v) - k * c)})
    return rel


def classify_peak(P, f1, f2, T, daily=True):
    """labels: day-commensurate (|f - j/2| < 0.012 c/d), harmonics n*f (n<=8) and half-integer multiples
    (n/2)*f (= the nterms=2 image of the harmonic n*f), yearly sidelobes (within 1/yr of n*f) and, for
    ground-based data, daily aliases n*f +- k c/d."""
    f = 24.0 / P
    tol = max(3.0 / T, 0.002)
    lab = []
    if daily and abs(2 * f - round(2 * f)) < 0.024 and round(2 * f) > 0:
        lab.append(f"day-commensurate (~{round(2 * f) / 2:g} c/d)")
    for nm, fx in (("f1", f1), ("f2", f2)):
        for n2 in range(1, 17):           # n2/2 * f
            v = 0.5 * n2 * fx
            nn = f"{n2 // 2}" if n2 % 2 == 0 else f"{n2}/2"
            if abs(f - v) < tol:
                lab.append(f"{nn}{nm}" + ("" if n2 % 2 == 0 else f" (nterms=2 image of {n2}{nm})"))
            elif abs(f - v) < 1.0 / 365.25 + tol and n2 % 2 == 0:
                lab.append(f"sidelobe of {nn}{nm} (within 1/yr)")
            if daily and n2 % 2 == 0 and n2 <= 8:
                for k in range(1, 4):
                    for sgn in (+1, -1):
                        for c in (1.0, SIDEREAL_CPD):
                            if abs(f - abs(v + sgn * k * c)) < tol:
                                lab.append(f"{nn}{nm}{'+' if sgn > 0 else '-'}{k} c/d (daily alias)")
    if P > 240:
        lab.append("long-period (>240 h; trends / geometry)")
    return sorted(set(lab))


def daycomm_mask(f, tol=0.012):
    return np.abs(2 * f - np.round(2 * f)) < 2 * tol


# ----------------------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--outdir", default=None, help="write the json/png to this directory instead (for tests)")
    args = ap.parse_args()
    global OUT_JSON, OUT_PNG
    if args.outdir:
        OUT_JSON = os.path.join(args.outdir, "binary_1717.json")
        OUT_PNG = os.path.join(args.outdir, "binary_1717.png")
    N_PERM_LS_BC = 10 if args.quick else 100
    N_PERM_LS_D = 10 if args.quick else 200
    N_PERM_DIP = 20 if args.quick else 300
    N_BOOT = 10 if args.quick else 100
    N_INJ = 5 if args.quick else 40
    t0 = time.time()
    rng = np.random.default_rng(1717)
    D = load()
    t, y, sheet, app, blk, grp, seg = D["t"], D["y"], D["sheet"], D["app"], D["blk"], D["grp"], D["seg"]
    tess = sheet == "C57G"
    ground = ~tess
    T_all = float(t.max() - t.min())
    out = {"asteroid": 1717, "inputs": {"npz": NPZ, "horizons": HOR}, "literature_values_used": LIT,
           "sigma_Porb_assumed_h": SIG_PORB, "settings": {"P_range_h": [P_MIN, P_MAX], "samples_per_peak": SPP,
           "nterms": 2, "n_perm_ls_prewhitened_single": N_PERM_LS_BC, "n_perm_ls_double": N_PERM_LS_D,
           "n_perm_dip": N_PERM_DIP, "n_boot_freq": N_BOOT, "n_inject_per_depth": N_INJ,
           "blocks": "ground: gaps > 0.25 d within a dataset; TESS: 1-d chunks",
           "shuffle_groups": "dataset x apparition (apparitions separated by gaps > 100 d)"}}
    ev, ev_vecs = event_season(D["hor"])
    out["event_geometry"] = ev
    evdist = event_distance(D["pab"], D["pablat"], ev_vecs)
    D["evdist"] = evdist
    evgeo5 = evdist <= 5.0
    evgeo = evdist <= 10.0          # loose subset (used for the LS search)
    out["event_geometry"]["subset_definition"] = (
        "angular distance of the phase-angle-bisector direction (or its antipode) from the PAB directions of the "
        "reported mutual-event windows (2006 Feb 24-Apr 17, CBET 504; 2019 Jan 22-Mar 1, Franco et al. 2019) "
        "<= 5 deg (strict) or <= 10 deg (loose). Note: the 2005 Dec-2006 Jan run without events (CBET 369) lies "
        "3-4 deg from the 2019 event direction, so the event zone is narrow and these subsets are approximate.")
    out["event_geometry"]["tess_min_distance_deg"] = float(evdist[tess].min())
    out["event_geometry"]["noevent_run_distance_deg"] = [float(x) for x in event_distance(
        np.array([114.2, 119.9]), np.array([6.49, 7.16]), ev_vecs)]
    out["data"] = {"N": int(len(t)), "N_ground": int(ground.sum()), "N_tess": int(tess.sum()),
                   "baseline_d": T_all, "time_zero_jd": D["tz"],
                   "apparitions": [], "tess_pab_lon_range": [float(D["pab"][tess].min()), float(D["pab"][tess].max())],
                   "N_ground_event_geometry_10deg": int((ground & evgeo).sum()),
                   "N_ground_event_geometry_5deg": int((ground & evgeo5).sum())}
    from astropy.time import Time
    for a in np.unique(app):
        m = app == a
        out["data"]["apparitions"].append({
            "id": int(a), "start": Time(t[m].min() + D["tz"], format="jd").iso[:10],
            "end": Time(t[m].max() + D["tz"], format="jd").iso[:10], "n": int(m.sum()),
            "n_tess": int((m & tess).sum()), "n_nights_ground": int(len(np.unique(blk[m & ground]))),
            "pab_lon": [float(D["pab"][m].min()), float(D["pab"][m].max())],
            "pab_lat": [float(D["pablat"][m].min()), float(D["pablat"][m].max())],
            "min_event_distance_deg": float(evdist[m].min()),
            "n_event_geometry_ground_10deg": int((m & ground & evgeo).sum()),
            "n_event_geometry_ground_5deg": int((m & ground & evgeo5).sum())})
    print(f"loaded N={len(t)}; event-geometry ground points: {(ground & evgeo).sum()} (10 deg), "
          f"{(ground & evgeo5).sum()} (5 deg)  [{time.time() - t0:.0f}s]")

    sets = {"all": np.ones(len(t), bool), "ground": ground, "tess": tess}

    # ---------------- 1. refined frequencies ---------------------------------------------------------------
    F = {}
    out["frequencies"] = {}
    for name, m in sets.items():
        tt, yy = t[m], y[m]
        res_k = {}
        for k in (2, 3, 4):
            fr, sg, rms, bic = refine(tt, yy, [24 / P2_START, 24 / P1_START], [k, k])
            res_k[k] = (fr, sg, rms, bic)
        kbest = min(res_k, key=lambda k: res_k[k][3])
        fr, sg, rms, bic = res_k[kbest]
        F[name] = {"f2": fr[0], "f1": fr[1], "k": kbest}
        out["frequencies"][name] = {
            "order_by_BIC": kbest, "BIC": {str(k): res_k[k][3] for k in res_k},
            "rms": rms,
            "f2_cpd": fr[0], "sig_f2_formal": sg[0], "P2_h": 24 / fr[0], "sig_P2_formal_h": 24 / fr[0] ** 2 * sg[0],
            "f1_cpd": fr[1], "sig_f1_formal": sg[1], "P1_h": 24 / fr[1], "sig_P1_formal_h": 24 / fr[1] ** 2 * sg[1],
            "by_order": {str(k): {"P2_h": 24 / res_k[k][0][0], "P1_h": 24 / res_k[k][0][1]} for k in res_k}}
        print(f"[{name}] P2={24 / fr[0]:.6f} P1={24 / fr[1]:.6f} (k={kbest})  [{time.time() - t0:.0f}s]")
    # night-block bootstrap of the frequencies
    for name, m in sets.items():
        idx_all = np.where(m)[0]
        struct = block_structure(blk[m], grp[m])
        tt, yy = t[m], y[m]
        k = F[name]["k"]
        bs = []
        for b in range(N_BOOT):
            pick = []
            for blocks in struct:
                sel = rng.integers(0, len(blocks), len(blocks))
                pick += [blocks[s] for s in sel]
            pick = np.concatenate(pick)
            try:
                fr, *_ = refine(tt[pick], yy[pick], [F[name]["f2"], F[name]["f1"]], [k, k])
                bs.append(fr)
            except Exception:
                pass
        bs = np.array(bs)
        o = out["frequencies"][name]
        o["n_boot"] = int(len(bs))
        o["sig_P2_boot_h"] = float(np.std(24 / bs[:, 0], ddof=1))
        o["sig_P1_boot_h"] = float(np.std(24 / bs[:, 1], ddof=1))
        o["P2_boot_median_h"] = float(np.median(24 / bs[:, 0]))
        o["P1_boot_median_h"] = float(np.median(24 / bs[:, 1]))
        _ = idx_all
    print(f"bootstrap done  [{time.time() - t0:.0f}s]")
    f2, f1 = F["all"]["f2"], F["all"]["f1"]

    # ---------------- 2. prewhitening + periodograms --------------------------------------------------------
    nuis_all = tess_nuisance(t, seg)
    spectra = {}      # for the figure
    out["prewhitening"] = {}
    perm_jobs = {}
    payload = {}
    owin = orbit_windows()
    out["orbit_windows_cpd"] = {k: list(v) for k, v in owin.items()}
    for name, m in sets.items():
        tt, yy = t[m], y[m]
        T = float(tt.max() - tt.min())
        hw = 1.5 / T
        ff2, ff1 = F[name]["f2"], F[name]["f1"]
        nu = nuis_all[m] if name != "ground" else None
        if nu is not None:
            nu = nu[:, np.any(nu != 0, axis=0)]
        o = {"N": int(m.sum()), "baseline_d": T}
        fgrid, p0 = periodogram(tt, yy)
        spectra[(name, "orig")] = (fgrid, p0)
        o["original"] = {"top": families(fgrid, p0), "power_f2": local_max(fgrid, p0, ff2, hw),
                         "power_f1": local_max(fgrid, p0, ff1, hw)}
        for lab, comps_fn in (("minus_P2", lambda k: [(ff2, k, None)]),
                              ("minus_P1", lambda k: [(ff1, k, None)]),
                              ("minus_both", lambda k: [(ff2, k, None), (ff1, k, None)])):
            for k in (2, 3, 4):
                A = design(tt, comps_fn(k))
                mod, c = lsfit(A, yy)
                r = yy - mod
                fg, p = periodogram(tt, r)
                e = {"rms": float(np.std(r)), "top": families(fg, p, top=10),
                     "top_primary_range": families(fg, p, top=6, pmax=240.0),
                     "power_f1": local_max(fg, p, ff1, hw), "power_f2": local_max(fg, p, ff2, hw)}
                for fam in e["top"] + e["top_primary_range"]:
                    fam["labels"] = classify_peak(fam["P"], ff1, ff2, T, daily=(name != "tess"))
                if lab == "minus_both":
                    e["orbit_windows"] = {kk: local_max(fg, p, 0.5 * (lo + hi), 0.5 * (hi - lo))
                                          for kk, (lo, hi) in owin.items()}
                o[f"{lab}_k{k}"] = e
                if k == F[name]["k"]:
                    spectra[(name, lab)] = (fg, p)
                    if lab in ("minus_P2", "minus_P1") or (lab == "minus_both" and nu is None):
                        key = f"{name}:{lab}"
                        payload[key] = {"t": tt, "r": r, "struct": block_structure(blk[m], grp[m]),
                                        "grid": grid(tt), "orbwin": list(owin.values()), "daily": name != "tess",
                                        "targets": {"f1": (ff1, hw), "f2": (ff2, hw)}}
                        perm_jobs[key] = N_PERM_LS_BC if lab != "minus_both" else N_PERM_LS_D
        # double prewhitening with linear trends per TESS segment (nuisance), only where TESS is included;
        # order k (BIC) and, as a diagnostic, order 6
        if nu is not None:
            for k in sorted({F[name]["k"], 6}):
                A = design(tt, [(ff2, k, None), (ff1, k, None)], nuis=nu)
                mod, c = lsfit(A, yy)
                r = yy - mod
                fg, p = periodogram(tt, r)
                e = {"rms": float(np.std(r)), "top": families(fg, p, top=10),
                     "top_primary_range": families(fg, p, top=6, pmax=240.0),
                     "orbit_windows": {kk: local_max(fg, p, 0.5 * (lo + hi), 0.5 * (hi - lo)) for kk, (lo, hi) in owin.items()}}
                for fam in e["top"] + e["top_primary_range"]:
                    fam["labels"] = classify_peak(fam["P"], ff1, ff2, T, daily=(name != "tess"))
                o[f"minus_both_tessdetrended_k{k}"] = e
                if k == F[name]["k"]:
                    spectra[(name, "minus_both")] = (fg, p)
                    key = f"{name}:minus_both"
                    payload[key] = {"t": tt, "r": r, "struct": block_structure(blk[m], grp[m]), "grid": grid(tt),
                                    "orbwin": list(owin.values()), "daily": name != "tess",
                                    "targets": {"f1": (ff1, hw), "f2": (ff2, hw)}}
                    perm_jobs[key] = N_PERM_LS_D
        out["prewhitening"][name] = o
        print(f"prewhitening [{name}] done  [{time.time() - t0:.0f}s]")
    # ground data in the event geometry
    mge = ground & evgeo
    tt, yy = t[mge], y[mge]
    k = F["ground"]["k"]
    A = design(tt, [(F["ground"]["f2"], k, None), (F["ground"]["f1"], k, None)])
    r = yy - lsfit(A, yy)[0]
    fg, p = periodogram(tt, r)
    Tg = float(tt.max() - tt.min())
    e = {"N": int(mge.sum()), "rms": float(np.std(r)), "top": families(fg, p, top=10),
         "orbit_windows": {kk: local_max(fg, p, 0.5 * (lo + hi), 0.5 * (hi - lo)) for kk, (lo, hi) in owin.items()}}
    for fam in e["top"]:
        fam["labels"] = classify_peak(fam["P"], F["ground"]["f1"], F["ground"]["f2"], Tg)
    out["prewhitening"]["ground_event_geometry_minus_both"] = e
    spectra[("ground_ev", "minus_both")] = (fg, p)
    payload["ground_ev:minus_both"] = {"t": tt, "r": r, "struct": block_structure(blk[mge], grp[mge]),
                                       "grid": grid(tt), "orbwin": list(owin.values()), "daily": True,
                                       "targets": {}}
    perm_jobs["ground_ev:minus_both"] = N_PERM_LS_D

    # permutations (multiprocessing, <= 4 workers)
    from multiprocessing import Pool
    jobs = [(key, int(1000 * i + j)) for i, key in enumerate(sorted(perm_jobs)) for j in range(perm_jobs[key])]
    print(f"LS permutations: {len(jobs)} periodograms on {N_WORKERS} workers ...", flush=True)
    with Pool(N_WORKERS, initializer=_init_worker, initargs=(payload,)) as pool:
        res = pool.map(_perm_ls, jobs, chunksize=2)
    null = {}
    for (key, _), r_ in zip(jobs, res):
        null.setdefault(key, []).append(r_)
    out["permutation_ls"] = {}
    for key, lst in null.items():
        name, lab = key.split(":")
        g = np.array([x["global"] for x in lst])
        pr = np.array([x["primary"] for x in lst])
        ow = np.array([x["orbit_windows"] for x in lst], dtype=float)
        fg, p = spectra[(name, lab)]
        P = 24.0 / fg
        om = np.zeros(len(fg), bool)
        for lo, hi in owin.values():
            om |= (fg >= lo) & (fg <= hi)
        px = np.array([x["primary_excl_daycomm"] for x in lst])
        pexm = (P <= 240) & (~daycomm_mask(fg) if payload[key].get("daily", True) else True)

        def pv(nullv, o_):
            return float((1 + np.sum(nullv >= o_)) / (1 + len(nullv)))
        obs = {"global": float(p.max()), "global_P": float(P[np.argmax(p)]),
               "primary": float(p[P <= 240].max()), "primary_P": float(P[P <= 240][np.argmax(p[P <= 240])]),
               "primary_excl_daycomm": float(p[pexm].max()), "primary_excl_daycomm_P": float(P[pexm][np.argmax(p[pexm])]),
               "orbit_windows": float(p[om].max()), "orbit_windows_P": float(P[om][np.argmax(p[om])])}
        e = {"n_perm": len(lst), "observed": obs,
             "null_global_q50_q95_q99": np.percentile(g, [50, 95, 99]).tolist(),
             "null_primary_q50_q95_q99": np.percentile(pr, [50, 95, 99]).tolist(),
             "null_primary_excl_daycomm_q50_q95_q99": np.percentile(px, [50, 95, 99]).tolist(),
             "null_orbitwin_q50_q95_q99": np.percentile(ow, [50, 95, 99]).tolist(),
             "p_global": pv(g, obs["global"]), "p_primary": pv(pr, obs["primary"]),
             "p_primary_excl_daycomm": pv(px, obs["primary_excl_daycomm"]),
             "p_orbit_windows": pv(ow, obs["orbit_windows"])}
        if lab in ("minus_P2", "minus_P1"):
            tn = "f1" if lab == "minus_P2" else "f2"
            ftarget, hwt = payload[key]["targets"][tn]
            Pt, pt = local_max(fg, p, ftarget, hwt)
            loc = np.array([x["target_" + tn] for x in lst])
            e["target"] = {"name": tn, "P": Pt, "power": pt,
                           "p_vs_null_primary_max": pv(pr, pt),
                           "ratio_to_null_primary_q99": float(pt / np.percentile(pr, 99)),
                           "null_local_q50_q99": np.percentile(loc, [50, 99]).tolist(),
                           "p_local": pv(loc, pt)}
        out["permutation_ls"][key] = e
        spectra[(name, lab, "q99")] = (np.percentile(pr, 99), np.percentile(ow, 99), np.percentile(g, 99))
    print(f"LS permutations done  [{time.time() - t0:.0f}s]", flush=True)

    # ---------------- 3. relations ---------------------------------------------------------------------------
    rel = {}
    fo = 24.0 / PORB
    sig_fo = 24.0 / PORB ** 2 * SIG_PORB
    sig_f2 = out["frequencies"]["all"]["sig_f2_formal"]
    res_cpd = 1.0 / T_all
    rel["f2_cpd"], rel["f1_cpd"], rel["forb_lit_cpd"], rel["sig_forb_assumed_cpd"] = f2, f1, fo, sig_fo
    rel["P2_over_P1"] = (24 / f2) / (24 / f1)
    rel["P2_over_P1_nearest_ratios"] = nearest_ratio((24 / f2) / (24 / f1))
    rel["Porb_over_P2"] = PORB / (24 / f2)
    rel["Porb_over_P2_nearest"] = nearest_ratio(PORB / (24 / f2), qmax=4)
    rel["Porb_over_P1"] = PORB / (24 / f1)
    rel["Porb_over_P1_nearest"] = nearest_ratio(PORB / (24 / f1), qmax=4)
    rel["Porb_for_exact_13_2_with_P2"] = 6.5 * 24 / f2
    rel["f1_minus_f2"] = f1 - f2
    rel["f1_minus_2f2"] = f1 - 2 * f2
    rel["f1_minus_3f2"] = f1 - 3 * f2
    rel["alias_relations_f1_f2_within_0.01cpd"] = alias_relations(f1, f2, 0.01, names=("f1", "f2"))
    # f2, f1 vs harmonics of forb and their daily aliases
    best = {}
    for nm, fx in (("f2", f2), ("f1", f1)):
        cands = []
        for n in range(1, 31):
            for k in range(-4, 5):
                v = n * fo + k
                if v <= 0:
                    continue
                cands.append({"n": n, "k": k, "f": v, "diff_cpd": fx - v, "n_sigma_forb": abs(fx - v) / (n * sig_fo)})
        cands.sort(key=lambda c: abs(c["diff_cpd"]))
        best[nm] = cands[:5]
    rel["nearest_n_forb_plus_k"] = best
    rel["resolution_cpd_combined"] = res_cpd
    out["relations"] = rel

    # combination frequencies (NPA test) on the double-prewhitened residuals (combined data, TESS detrended;
    # and TESS alone)
    comb = {}
    for name in ("all", "tess", "ground"):
        m = sets[name]
        tt, yy = t[m], y[m]
        k = F[name]["k"]
        nu = nuis_all[m] if name != "ground" else None
        if nu is not None:
            nu = nu[:, np.any(nu != 0, axis=0)]
        A = design(tt, [(F[name]["f2"], k, None), (F[name]["f1"], k, None)], nuis=nu)
        r = yy - lsfit(A, yy)[0]
        T = float(tt.max() - tt.min())

        def semi_amp(fc):
            B = np.column_stack([np.ones_like(tt), np.cos(2 * np.pi * fc * tt), np.sin(2 * np.pi * fc * tt)])
            c = lsfit(B, r)[1]
            return float(np.hypot(c[1], c[2]))
        lst = []
        for m1 in range(-3, 4):
            for m2 in range(-3, 4):
                if m1 == 0 or m2 == 0:
                    continue
                fc = m1 * F[name]["f1"] + m2 * F[name]["f2"]
                if fc <= 0.05 or fc > 24.0:
                    continue
                if any(abs(fc - x["f"]) < 1e-9 for x in lst):
                    continue
                lst.append({"m1": m1, "m2": m2, "f": float(fc), "P_h": float(24 / fc), "semi_amp": semi_amp(fc),
                            "day_commensurate": bool(abs(2 * fc - round(2 * fc)) < 0.02)})
        # reference: random frequencies 0.1-12 c/d away from harmonics, combinations and half-integer c/d
        forb_list = [x["f"] for x in lst] + [j * F[name]["f1"] for j in range(1, 6)] + [j * F[name]["f2"] for j in range(1, 10)]
        rr = np.random.default_rng(7)
        ref = []
        while len(ref) < 1500:
            fc = rr.uniform(0.1, 12.0)
            if min(abs(fc - x) for x in forb_list) < 5.0 / T or abs(2 * fc - round(2 * fc)) < 0.02:
                continue
            ref.append(semi_amp(fc))
        ref = np.array(ref)
        for x in lst:
            x["percentile_vs_random"] = float(np.mean(ref < x["semi_amp"]) * 100)
        lst.sort(key=lambda x: -x["semi_amp"])
        pmax = max(x["percentile_vs_random"] for x in lst) / 100.0
        # semi-amplitudes of the fitted harmonics of the two signals, for comparison
        A_ = design(tt, [(F[name]["f2"], k, None), (F[name]["f1"], k, None)], nuis=nu)
        cc = lsfit(A_, yy)[1]
        h2 = [float(np.hypot(cc[1 + 2 * j], cc[2 + 2 * j])) for j in range(k)]
        h1 = [float(np.hypot(cc[1 + 2 * k + 2 * j], cc[2 + 2 * k + 2 * j])) for j in range(k)]
        comb[name] = {"n_random": len(ref), "random_semi_amp_q50_q95_q99": np.percentile(ref, [50, 95, 99]).tolist(),
                      "n_combinations": len(lst),
                      "p_any_combination_by_chance": float(1 - pmax ** len(lst)),
                      "semi_amp_harmonics_f2": h2, "semi_amp_harmonics_f1": h1,
                      "combinations": lst}
    out["combination_frequencies"] = comb
    print(f"relations done  [{time.time() - t0:.0f}s]", flush=True)

    # ---------------- 4. dip searches -------------------------------------------------------------------------
    dips = {}
    # 4a TESS: BLS on the double-prewhitened, detrended residuals
    m = tess
    tt, yy = t[m], y[m]
    k = F["tess"]["k"]
    A = design(tt, [(F["tess"]["f2"], k, None), (F["tess"]["f1"], k, None)])
    r = yy - lsfit(A, yy)[0]
    trend = running_median(tt, r, 1.5)
    rt = r - trend
    sig_t = float(1.4826 * np.median(np.abs(rt - np.median(rt))))
    Tt = float(tt.max() - tt.min())
    durs = np.array([2.0, 3.0, 4.0, 6.0, 8.0]) / 24.0
    fmin, fmax = 24.0 / 250.0, 24.0 / 40.0
    dfreq = durs.min() / Tt ** 2
    freqs_bls = np.arange(fmin, fmax, dfreq)
    per_bls = 1.0 / freqs_bls
    res_b = bls_run(tt, rt, per_bls, durs, sig_t)
    Ph = per_bls * 24.0
    win_orb = ((Ph >= PORB - 3 * SIG_PORB - 4) & (Ph <= PORB + 3 * SIG_PORB + 4))
    win_half = ((Ph >= PORB / 2 - 1.5 * SIG_PORB - 2) & (Ph <= PORB / 2 + 1.5 * SIG_PORB + 2))
    win = win_orb | win_half
    snr = np.asarray(res_b.power)
    ib = int(np.argmax(snr))
    iw = np.where(win)[0][np.argmax(snr[win])]
    tess_obs = {"global_max_snr": float(snr[ib]), "global_P_h": float(Ph[ib]),
                "global_depth_mag": float(res_b.depth[ib]), "global_duration_h": float(res_b.duration[ib] * 24),
                "global_t0": float(res_b.transit_time[ib]),
                "window_max_snr": float(snr[iw]), "window_P_h": float(Ph[iw]),
                "window_depth_mag": float(res_b.depth[iw]), "window_duration_h": float(res_b.duration[iw] * 24),
                "window_t0": float(res_b.transit_time[iw])}
    # permutation: 1-d chunks relabelled (<= 4 worker processes)
    struct_t = block_structure(blk[m], grp[m])
    from multiprocessing import Pool
    bls_payload = {"bls": {"t": tt, "r": rt, "struct": struct_t, "per": per_bls, "durs": durs, "sig": sig_t,
                           "win": win}}
    with Pool(N_WORKERS, initializer=_init_worker, initargs=(bls_payload,)) as pool:
        resb = pool.map(_perm_bls, [50000 + i for i in range(N_PERM_DIP)], chunksize=4)
    gnull = np.array([x[0] for x in resb])
    wnull = np.array([x[1] for x in resb])
    tess_obs["n_perm"] = int(len(gnull))
    tess_obs["null_global_q50_q95_q99"] = np.percentile(gnull, [50, 95, 99]).tolist()
    tess_obs["null_window_q50_q95_q99"] = np.percentile(wnull, [50, 95, 99]).tolist()
    tess_obs["p_global"] = float((1 + np.sum(gnull >= tess_obs["global_max_snr"])) / (1 + len(gnull)))
    tess_obs["p_window"] = float((1 + np.sum(wnull >= tess_obs["window_max_snr"])) / (1 + len(wnull)))
    tess_obs["sigma_point_mag"] = sig_t
    tess_obs["detrend"] = "running median, 1.5 d"
    tess_obs["durations_h"] = (durs * 24).tolist()
    tess_obs["P_range_h"] = [40.0, 250.0]
    tess_obs["windows_h"] = [[float(Ph[win_orb].min()), float(Ph[win_orb].max())],
                             [float(Ph[win_half].min()), float(Ph[win_half].max())]]
    # injection-recovery (two equal box events per orbit, 4 h, P = 117.0 h, random phase)
    thr = float(np.percentile(wnull, 99))
    inj = {}
    per_w = per_bls[win]
    for depth in (0.03, 0.05, 0.08, 0.10, 0.15):
        rec = []
        for i in range(N_INJ):
            ph0 = rng.uniform()
            tn = relabel_times(tt, struct_t, rng)          # chunk-permuted background
            ri = rt + inject_events(tn, PORB, ph0, depth, 4.0)
            rb = bls_run(tn, ri, per_w, durs, sig_t)
            rec.append(float(np.max(rb.power)))
        rec = np.array(rec)
        inj[f"{depth:.2f}"] = {"median_window_snr": float(np.median(rec)), "frac_above_null_q99": float(np.mean(rec > thr))}
    tess_obs["injection"] = {"threshold_null_window_q99": thr,
                             "events": "two equal 4-h box dips per 117.0-h orbit, added to chunk-permuted residuals",
                             "by_depth_mag": inj}
    dips["tess_bls"] = tess_obs
    spectra["tess_bls"] = (Ph, snr, np.percentile(gnull, 99), thr)
    spectra["tess_resid"] = (tt, rt, res_b, ib, iw)
    print(f"TESS BLS done  [{time.time() - t0:.0f}s]", flush=True)

    # 4b ground: per-apparition Fourier coefficients, offsets per dataset x apparition, night means
    m = ground
    tt, yy = t[m], y[m]
    k = 2
    napp = {a: np.sum(app[m] == a) for a in np.unique(app[m])}
    lab = np.array([a if napp[a] >= 60 else -1 for a in app[m]])
    A = design(tt, [(F["ground"]["f2"], k, lab), (F["ground"]["f1"], k, lab)], offset_lab=grp[m])
    rg = yy - lsfit(A, yy)[0]
    rfull = np.zeros(len(t))
    rfull[m] = rg
    gsel = {"all_ground": ground, "event_geometry_10deg": ground & evgeo, "event_geometry_5deg": ground & evgeo5}
    dips["ground_stack"] = {}
    Pgrid = None
    for nm, sel in gsel.items():
        nt = night_table(t, rfull, blk, sel)
        bl = np.unique(blk[sel])
        app_n = np.array([app[blk == b][0] for b in bl])
        grp_n = np.array([grp[blk == b][0] for b in bl])
        spans = [nt["t"][app_n == a].max() - nt["t"][app_n == a].min() for a in np.unique(app_n)]
        if Pgrid is None:
            Pgrid = period_grid_dip(50.0, 150.0, max(spans), 0.05)
        ds = DipStack(nt, app_n, Pgrid, wbin=0.05, min_nights=15 if nm == "all_ground" else 8)
        S, pa = ds.stack(nt["m"], nt["w"])
        wtarget = ((Pgrid >= PORB - 3 * SIG_PORB) & (Pgrid <= PORB + 3 * SIG_PORB)) | \
                  ((Pgrid >= (PORB - 3 * SIG_PORB) / 2) & (Pgrid <= (PORB + 3 * SIG_PORB) / 2))
        wbroad = (Pgrid >= 100) & (Pgrid <= 135)
        obs = {"n_nights": int(len(bl)), "apparitions_used": [int(a) for a in ds.apps],
               "nights_per_apparition": {str(int(a)): int(np.sum(app_n == a)) for a in ds.apps},
               "s_point_mag": nt["s_pt"], "s_night_mag": nt["s_night"],
               "global_max": float(S.max()), "global_P_h": float(Pgrid[np.argmax(S)]),
               "target_max": float(S[wtarget].max()), "target_P_h": float(Pgrid[wtarget][np.argmax(S[wtarget])]),
               "broad_max": float(S[wbroad].max()), "broad_P_h": float(Pgrid[wbroad][np.argmax(S[wbroad])]),
               "S_at_117h": float(np.interp(PORB, Pgrid, S)),
               "per_apparition_best": {str(int(a)): {"P_h": float(Pgrid[np.argmax(pa[a])]), "zmax": float(pa[a].max()),
                                                     "z_at_117h": float(np.interp(PORB, Pgrid, pa[a]))} for a in ds.apps}}
        gn, tn_, bn_ = [], [], []
        for i in range(N_PERM_DIP):
            mp, wp = permute_nights(nt["m"], nt["w"], grp_n, rng)
            Sp, _ = ds.stack(mp, wp)
            gn.append(Sp.max())
            tn_.append(Sp[wtarget].max())
            bn_.append(Sp[wbroad].max())
        gn, tn_, bn_ = map(np.array, (gn, tn_, bn_))
        obs.update({"n_perm": int(len(gn)),
                    "null_global_q50_q95_q99": np.percentile(gn, [50, 95, 99]).tolist(),
                    "null_target_q50_q95_q99": np.percentile(tn_, [50, 95, 99]).tolist(),
                    "null_broad_q50_q95_q99": np.percentile(bn_, [50, 95, 99]).tolist(),
                    "p_global": float((1 + np.sum(gn >= obs["global_max"])) / (1 + len(gn))),
                    "p_target": float((1 + np.sum(tn_ >= obs["target_max"])) / (1 + len(tn_))),
                    "p_broad": float((1 + np.sum(bn_ >= obs["broad_max"])) / (1 + len(bn_)))})
        # injection-recovery: synthetic events (night means of the point-wise box model) added to a night-permuted
        # realisation of the residual night means (so that any real signal in the data does not bias the recovery)
        thr_t = float(np.percentile(tn_, 99))
        injr = {}
        for depth in (0.05, 0.10, 0.15, 0.20):
            rec = []
            for i in range(N_INJ):
                ph0 = rng.uniform()
                ev_pt = inject_events(t, PORB, ph0, depth, 4.0)
                inj_n = np.array([ev_pt[ix].mean() for ix in nt["idx"]])
                mp, wp = permute_nights(nt["m"], nt["w"], grp_n, rng)
                Si, _ = ds.stack(mp + inj_n, wp)
                rec.append(float(Si[wtarget].max()))
            rec = np.array(rec)
            injr[f"{depth:.2f}"] = {"median_target_S": float(np.median(rec)), "frac_above_null_q99": float(np.mean(rec > thr_t))}
        obs["injection"] = {"threshold_null_target_q99": thr_t, "events": "two equal 4-h box dips per 117.0-h orbit, "
                            "present in every included apparition (optimistic), added to night-permuted residuals",
                            "by_depth_mag": injr}
        dips["ground_stack"][nm] = obs
        spectra[("dip", nm)] = (Pgrid, S, float(np.percentile(gn, 99)), thr_t)
        if nm != "all_ground":
            Pb = float(Pgrid[wtarget][np.argmax(S[wtarget])])
            spectra[("dipfold", nm)] = (nt, app_n, ds.apps, Pb, obs["p_target"])
        print(f"ground dip stack [{nm}] done  [{time.time() - t0:.0f}s]", flush=True)

    # 4c ground: tail test -- mutual events add faint outliers to the night means in the event geometry
    from scipy.stats import mannwhitneyu, fisher_exact
    nt_all = night_table(t, rfull, blk, ground)
    bl_all = np.unique(blk[ground])
    dist_n = np.array([np.median(evdist[blk == b]) for b in bl_all])
    z_n = nt_all["m"] * np.sqrt(nt_all["w"])
    jd_n = nt_all["t"] + D["tz"]
    tail = {"definition": "z = night-mean residual / its sigma (pooled within-night scatter + night jitter); positive = "
                          "fainter; residuals of the per-apparition two-period fit (Sect. 4b)",
            "expected_event_fraction": "two ~4-h events per 117-h orbit -> ~7-9% of nights (each ~1 h long) inside an "
                                       "event, if the whole subset is in the event zone"}
    out_ref = dist_n > 20.0
    tail["n_nights_reference_gt20deg"] = int(out_ref.sum())
    for lab_, lim in (("5deg", 5.0), ("10deg", 10.0)):
        ins = dist_n <= lim
        e = {"n_nights": int(ins.sum())}
        for zc in (2.0, 2.5, 3.0):
            a_, b_ = int(np.sum(z_n[ins] > zc)), int(np.sum(z_n[out_ref] > zc))
            e[f"n_z_gt_{zc}"] = a_
            e[f"frac_z_gt_{zc}"] = a_ / max(1, ins.sum())
            e[f"frac_z_gt_{zc}_reference"] = b_ / max(1, out_ref.sum())
            e[f"fisher_p_one_sided_{zc}"] = float(fisher_exact([[a_, ins.sum() - a_], [b_, out_ref.sum() - b_]],
                                                               alternative="greater")[1])
            # same for the bright side (control)
            a2, b2 = int(np.sum(z_n[ins] < -zc)), int(np.sum(z_n[out_ref] < -zc))
            e[f"n_z_lt_-{zc}"] = a2
            e[f"frac_z_lt_-{zc}_reference"] = b2 / max(1, out_ref.sum())
        e["mannwhitney_p_one_sided_fainter"] = float(mannwhitneyu(z_n[ins], z_n[out_ref], alternative="greater")[1])
        e["median_z"] = float(np.median(z_n[ins]))
        e["median_z_reference"] = float(np.median(z_n[out_ref]))
        tail[lab_] = e
    # list of the faintest event-geometry nights (candidate events) and the nights of the 2019 event window
    from astropy.time import Time as _T
    cand = []
    for i in np.where((dist_n <= 10.0))[0]:
        bi = bl_all[i]
        cand.append({"date_utc": _T(jd_n[i], format="jd").iso[:16], "dataset": str(sheet[blk == bi][0]),
                     "n": int(nt_all["n"][i]), "mean_resid_mag": float(nt_all["m"][i]), "z": float(z_n[i]),
                     "event_distance_deg": float(dist_n[i])})
    cand.sort(key=lambda c: -c["z"])
    tail["event_geometry_10deg_nights_sorted_by_z"] = cand
    j1, j2 = _T("2019-01-22").jd, _T("2019-03-02").jd
    w19 = (jd_n >= j1) & (jd_n <= j2)
    tail["franco2019_window_nights"] = {"n_nights": int(w19.sum()), "n_points": int(nt_all["n"][w19].sum()),
                                        "z": [float(x) for x in z_n[w19]],
                                        "n_z_gt_2": int(np.sum(z_n[w19] > 2)), "max_z": float(z_n[w19].max()) if w19.any() else None}
    dips["ground_tail_test"] = tail
    spectra["tail"] = (jd_n, z_n, dist_n)
    print(f"tail test done  [{time.time() - t0:.0f}s]", flush=True)
    dips["ground_stack"]["method"] = ("residuals of a fit with per-apparition Fourier coefficients (order 2, both periods; "
                                      "apparitions with >= 60 points) and offsets per dataset x apparition; night means "
                                      "weighted by pooled within-night scatter + night jitter; folded at each trial period "
                                      "into 0.05-phase bins (two bin offsets), z = weighted mean / its sigma, maximum over "
                                      "phase (positive = fainter) summed over apparitions; target windows 117 +- 1.5 h and half")
    out["dip_search"] = dips

    # ---------------- 5. per-apparition amplitudes ------------------------------------------------------------
    amps = []
    for a in np.unique(app):
        mm = app == a
        if mm.sum() < 80:
            continue
        for part, sel in (("ground", mm & ground), ("tess", mm & tess)):
            if sel.sum() < 60:
                continue
            tt, yy = t[sel], y[sel]
            A = design(tt, [(f2, 2, None), (f1, 2, None)], offset_lab=sheet[sel])
            mod, c = lsfit(A, yy)
            r = yy - mod
            nof = len(np.unique(sheet[sel]))
            ph = np.linspace(0, 1, 400, endpoint=False)
            s2 = np.sum(r ** 2) / (len(r) - A.shape[1])
            cov = np.linalg.pinv(A.T @ A) * s2
            draws = np.random.default_rng(3).multivariate_normal(c, cov, 300)

            def p2p(cc, i0):
                v = cc[i0] * np.cos(2 * np.pi * ph) + cc[i0 + 1] * np.sin(2 * np.pi * ph) + \
                    cc[i0 + 2] * np.cos(4 * np.pi * ph) + cc[i0 + 3] * np.sin(4 * np.pi * ph)
                return v.max() - v.min()
            A2 = p2p(c, nof)
            A1 = p2p(c, nof + 4)
            A2s = np.std([p2p(d_, nof) for d_ in draws])
            A1s = np.std([p2p(d_, nof + 4) for d_ in draws])
            amps.append({"apparition": int(a), "part": part, "n": int(sel.sum()),
                         "start": Time(t[sel].min() + D["tz"], format="jd").iso[:10],
                         "pab_lon_mean": float(np.degrees(np.angle(np.mean(np.exp(1j * np.radians(D["pab"][sel]))))) % 360),
                         "min_event_distance_deg": float(evdist[sel].min()),
                         "alpha_mean": float(np.mean(D["alpha"][sel])),
                         "A2_18h": float(A2), "sig_A2": float(A2s), "A1_5h": float(A1), "sig_A1": float(A1s),
                         "rms": float(np.std(r))})
    out["per_apparition_amplitudes"] = {"note": "peak-to-peak of order-2 Fourier fits of both periods with dataset offsets; "
                                        "not corrected for the noise bias (about +16% at 0.1 mag, Sect. ampresults)",
                                        "rows": amps}

    # ---------------- dilution (which body?) -----------------------------------------------------------------
    def intrinsic(dm_obs, frac):
        x = 1 - (1 - 10 ** (-0.4 * dm_obs)) / frac
        return float(-2.5 * np.log10(x)) if x > 0 else None
    dil = []
    for q in (0.5, 0.7, 1.0):
        fs = q ** 2 / (1 + q ** 2)
        dil.append({"D2_over_D1": q, "flux_fraction_secondary": fs,
                    "A_intrinsic_if_18h_on_secondary": intrinsic(0.12, fs),
                    "A_intrinsic_if_5h_on_primary": intrinsic(0.09, 1 - fs),
                    "A_intrinsic_if_18h_on_primary": intrinsic(0.12, 1 - fs),
                    "A_intrinsic_if_5h_on_secondary": intrinsic(0.09, fs)})
    out["dilution_estimates"] = {"note": "equal albedos and phase behaviour; observed amplitudes 0.12 (18.21 h) and "
                                 "0.09 mag (5.15 h) from Table 2; intrinsic peak-to-peak amplitude needed",
                                 "rows": dil}
    out["runtime_s"] = time.time() - t0

    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(_jsonable(out), fh, indent=1)
    print(f"wrote {OUT_JSON}")
    make_figure(D, F, spectra, out)
    print(f"wrote {OUT_PNG}  total {time.time() - t0:.0f}s")


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    return o


# ----------------------------------------------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------------------------------------------


def make_figure(D, F, spectra, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker  # noqa: F401
    BLUE, ORANGE, AQUA, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#8c8b86", "#0b0b0b", "#52514e"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#b5b4ae", "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False})
    t, y, sheet = D["t"], D["y"], D["sheet"]
    f2, f1 = F["all"]["f2"], F["all"]["f1"]
    P2, P1 = 24 / f2, 24 / f1
    fig, axs = plt.subplots(7, 2, figsize=(15, 27))

    def spec(ax, key, title, color=BLUE, marks=(), q99=None, ylim=None):
        fg, p = spectra[key][:2]
        P = 24.0 / fg
        ax.plot(P, p, color=color, lw=0.5)
        ax.set_xscale("log")
        ax.set_xlim(1, 1500)
        for Pm, lab, c in marks:
            ax.axvline(Pm, color=c, lw=0.8, ls="--", alpha=0.8)
            ax.text(Pm, ax.get_ylim()[1] * 0.97 if ylim is None else ylim[1] * 0.97, " " + lab, color=MUTED,
                    fontsize=7.5, va="top", rotation=90)
        if q99 is not None:
            ax.axhline(q99, color=ORANGE, lw=0.9, ls=":")
            ax.text(1.05, q99, "permutation 99% (max 1-240 h)", color=MUTED, fontsize=7, va="bottom")
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=9.5, loc="left", color=INK)
        ax.set_xlabel("period (h)")
        ax.set_ylabel("LS power (nterms=2)")
    orbm = [(PORB, "117 h (P_orb lit.)", AQUA), (PORB / 2, "P_orb/2", AQUA)]
    spec(axs[0, 0], ("all", "orig"), "(a) combined data (N=8398), original",
         marks=[(P2, "18.214 h", GREY), (P1, "5.1475 h", GREY)])
    q = spectra.get(("all", "minus_P2", "q99"))
    spec(axs[0, 1], ("all", "minus_P2"), f"(b) combined, 18.214 h subtracted (order {F['all']['k']})",
         marks=[(P1, "5.1475 h", GREY)], q99=q[0] if q else None)
    q = spectra.get(("all", "minus_P1", "q99"))
    spec(axs[1, 0], ("all", "minus_P1"), f"(c) combined, 5.1475 h subtracted (order {F['all']['k']})",
         marks=[(P2, "18.214 h", GREY)], q99=q[0] if q else None)
    q = spectra.get(("all", "minus_both", "q99"))
    spec(axs[1, 1], ("all", "minus_both"), "(d) combined, both subtracted (+ linear trend per TESS segment)",
         marks=orbm, q99=q[0] if q else None)
    q = spectra.get(("ground", "minus_both", "q99"))
    spec(axs[2, 0], ("ground", "minus_both"), "(e) ground-based only (N=2921), both subtracted", marks=orbm,
         q99=q[0] if q else None)
    q = spectra.get(("tess", "minus_both", "q99"))
    spec(axs[2, 1], ("tess", "minus_both"), "(f) TESS C57G only (N=5477), both subtracted (+ segment trends)",
         marks=orbm, q99=q[0] if q else None)

    # folds
    def binned(ph, v, nb=30):
        e = np.linspace(0, 1, nb + 1)
        i = np.digitize(ph, e) - 1
        mu = np.array([np.mean(v[i == j]) if np.sum(i == j) > 2 else np.nan for j in range(nb)])
        se = np.array([np.std(v[i == j]) / np.sqrt(np.sum(i == j)) if np.sum(i == j) > 2 else np.nan for j in range(nb)])
        return 0.5 * (e[1:] + e[:-1]), mu, se
    k = F["all"]["k"]
    for ax, (fa, fb, Pa, lab) in zip((axs[3, 0], axs[3, 1]), ((f2, f1, P2, "18.2137 h (5.1475 h subtracted)"),
                                                               (f1, f2, P1, "5.1475 h (18.2137 h subtracted)"))):
        for part, sel, c, off in (("ground", sheet != "C57G", BLUE, 0.0), ("TESS", sheet == "C57G", ORANGE, 0.25)):
            tt, yy = t[sel], y[sel]
            A = design(tt, [(fb, k, None)])
            r = yy - lsfit(A, yy)[0]
            ph = (tt * fa) % 1.0
            x, mu, se = binned(ph, r - np.nanmean(r))
            ax.errorbar(np.r_[x, x + 1], np.r_[mu, mu] + off, yerr=np.r_[se, se], fmt="o", ms=3.5, color=c,
                        lw=0.8, label=f"{part} (offset {off:+.2f})")
            B = design(tt, [(fa, k, None)])
            mod = lsfit(B, r)[0]
            o = np.argsort(ph)
            ax.plot(np.r_[ph[o], ph[o] + 1], np.r_[mod[o], mod[o]] - np.mean(mod) + off, color=INK, lw=0.8)
        ax.invert_yaxis()
        ax.set_xlabel("phase")
        ax.set_ylabel("reduced magnitude (mag)")
        ax.set_title(f"(g{'1' if ax is axs[3, 0] else '2'}) fold at {lab}; 30 phase bins, mean +- s.e.", fontsize=9.5,
                     loc="left")
        ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    # dip searches
    Ph, snr, g99, w99 = spectra["tess_bls"]
    ax = axs[4, 0]
    ax.plot(Ph, snr, color=BLUE, lw=0.6)
    ax.axhline(g99, color=ORANGE, ls=":", lw=0.9)
    ax.axhline(w99, color=AQUA, ls=":", lw=0.9)
    ax.text(41, g99, "perm. 99% (40-250 h)", fontsize=7, color=MUTED, va="bottom")
    ax.text(41, w99, "perm. 99% (windows at 117 and 58.5 h)", fontsize=7, color=MUTED, va="top")
    for Pm in (PORB, PORB / 2):
        ax.axvspan(Pm - 3 * SIG_PORB, Pm + 3 * SIG_PORB, color=AQUA, alpha=0.15, lw=0)
    ax.set_xscale("log")
    ax.set_xticks([40, 60, 80, 100, 117, 150, 200, 250])
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("trial period (h)")
    ax.set_ylabel("BLS depth S/N (fading)")
    ax.set_title("(h) TESS: box (mutual-event) search, double-prewhitened, 1.5-d running median removed",
                 fontsize=9.5, loc="left")
    ax = axs[4, 1]
    for off, (nm, c) in zip((0.0, 0.6, 1.2), (("all_ground", BLUE), ("event_geometry_10deg", ORANGE),
                                               ("event_geometry_5deg", AQUA))):
        Pg, S, g99, t99 = spectra[("dip", nm)]
        ax.plot(Pg, S / t99 + off, color=c, lw=0.6,
                label=f"{nm.replace('_', ' ')} (+{off:.1f}); 99%: {t99:.1f} target, {g99:.1f} global")
        ax.axhline(1.0 + off, color=c, ls=":", lw=0.9)
    for Pm in (PORB, PORB / 2):
        ax.axvspan(Pm - 3 * SIG_PORB, Pm + 3 * SIG_PORB, color=AQUA, alpha=0.15, lw=0)
    ax.set_xlabel("trial period (h)")
    ax.set_ylabel("stack statistic / permutation 99% (target windows) + offset")
    ax.set_title("(i) ground: folded-binned night means (fading), stacked over apparitions", fontsize=9.5, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    # TESS residual fold at the best period in the windows
    tt, rt, res_b, ib, iw = spectra["tess_resid"]
    ax = axs[5, 0]
    P = float(res_b.period[iw])
    t0_ = float(res_b.transit_time[iw])
    ph = ((tt - t0_) / P + 0.5) % 1.0
    ax.plot(ph, rt, ",", color=GREY, alpha=0.5)
    x, mu, se = binned(ph, rt, nb=int(round(P * 24 / 2)))
    ax.errorbar(x, mu, yerr=se, fmt="o", ms=3, color=BLUE, lw=0.8)
    ax.set_ylim(0.3, -0.3)
    ax.set_xlabel("phase (box centre at 0.5)")
    ax.set_ylabel("residual (mag)")
    ax.set_title(f"(j1) TESS residuals folded at {P * 24:.2f} h (best BLS period in the 117/58.5-h windows); 2-h bins",
                 fontsize=9.5, loc="left")
    # ground event-geometry night means folded at the best target-window period of the stack
    ax = axs[5, 1]
    nt, app_n, apps, Pb, pt = spectra[("dipfold", "event_geometry_5deg")]
    cols = [BLUE, ORANGE, AQUA, "#7a5cc4"]
    for j, a in enumerate(apps):
        s = app_n == a
        ph = (nt["t"][s] * 24.0 / Pb) % 1.0
        ax.errorbar(ph, nt["m"][s], yerr=1 / np.sqrt(nt["w"][s]), fmt="o", ms=3.5, lw=0.7, color=cols[j % 4],
                    label=f"apparition {a} ({s.sum()} nights)")
    ax.invert_yaxis()
    ax.set_xlabel(f"phase at P = {Pb:.2f} h (arbitrary zero point, per-apparition phases independent)")
    ax.set_ylabel("night-mean residual (mag)")
    ax.set_title(f"(j2) ground, event geometry (<=5 deg): night means at best target-window period (p={pt:.3f})",
                 fontsize=9.5, loc="left")
    ax.legend(fontsize=7.5, frameon=False)
    # tail test panels
    jd_n, z_n, dist_n = spectra["tail"]
    ax = axs[6, 0]
    ax.scatter(dist_n, z_n, s=6, color=GREY, alpha=0.6, lw=0)
    sel5 = dist_n <= 5
    sel10 = (dist_n > 5) & (dist_n <= 10)
    ax.scatter(dist_n[sel5], z_n[sel5], s=12, color=AQUA, lw=0, label="<= 5 deg")
    ax.scatter(dist_n[sel10], z_n[sel10], s=12, color=ORANGE, lw=0, label="5-10 deg")
    ax.axhline(0, color=INK, lw=0.6)
    ax.axhline(2.5, color=ORANGE, lw=0.7, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("angular distance of PAB direction from reported event geometry (deg)")
    ax.set_ylabel("night-mean residual z (positive = fainter)")
    ax.set_title("(k1) ground night means vs. distance from the mutual-event geometry", fontsize=9.5, loc="left")
    ax.legend(fontsize=7.5, frameon=False)
    ax = axs[6, 1]
    from astropy.time import Time as _T
    yr = _T(jd_n, format="jd").decimalyear
    w = (yr > 2017.3) & (yr < 2019.6)
    ax.scatter(yr[w], z_n[w], s=10, color=GREY, lw=0)
    ax.scatter(yr[w & sel5], z_n[w & sel5], s=16, color=AQUA, lw=0, label="<= 5 deg")
    ax.scatter(yr[w & sel10], z_n[w & sel10], s=16, color=ORANGE, lw=0, label="5-10 deg")
    a19, b19 = _T("2019-01-22").decimalyear, _T("2019-03-01").decimalyear
    ax.axvspan(a19, b19, color=AQUA, alpha=0.15, lw=0)
    ax.text(b19, -3.6, " events reported\n (Franco et al. 2019)", fontsize=7, va="bottom", color=MUTED)
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xlabel("year")
    ax.set_ylabel("night-mean residual z (positive = fainter)")
    ax.set_title("(k2) ground night means in the 2017 and 2018-19 apparitions", fontsize=9.5, loc="left")
    ax.legend(fontsize=7.5, frameon=False)
    fig.suptitle("(1717) Arlon: prewhitening of the 18.21-h and 5.15-h signals and search for the 117-h orbital period",
                 fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
