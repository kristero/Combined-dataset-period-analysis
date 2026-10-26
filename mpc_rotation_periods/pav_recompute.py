"""
Separate-dataset periods P_av recomputed with the CORRECT light-time sign.

Background (NOTES_FOR_AUTHORS.md, section 4 item 1): the separate-dataset analysis (manuscript Section
"Separate-dataset approach", Eq. eq:pav) used the JDc2 column of the "*-excelB?.xlsx" workbooks,
    JDc2 = epoch + (Delta_i - Delta_0)/c          (wrong sign; formula "=B2+(D2-$D$2)*4.99/36/24"),
instead of epoch + (Delta_0 - Delta_i)/c (Eq. eq:lighttime_rel).  This script recomputes the per-dataset
periods p_i and P_av with both signs, keeping the ORIGINAL peak choice, so that the effect of the sign is isolated.

What the original analysis did (established from the workbooks, see pav_recompute.md):
  * per dataset sheet: magred = mag - 5 log10(r Delta) (daily ephemeris), linear magnitude-phase fit
    (slope k, coefficient of determination R^2) and magph7 = magred - (alpha - alpha_0) k;
  * the "a" sheets (columns JDc2, magph7) were analysed with a TWO-term Lomb-Scargle periodogram (nterms = 2,
    standard normalization) at the rotation frequency (the per-dataset periods p_i of the "Periods" sheets
    are reproduced with nterms = 2; nterms = 1 does not reproduce them);
  * the per-dataset (p_i, w_i, N_i, R_i^2) of a "Periods"/"PeriodsD" sheet were combined with Eq. eq:pav.

Approaches
  S  (12 asteroids with workbooks): the same rows, same magph7 values and the original peak of every dataset:
     the peak nearest to the original p_i is located in the periodogram of the wrong-sign JDc2 (validation:
     reproduces p_i, w_i, P_av), then followed to the nearest peak with the corrected times
     t = epoch - (Delta - Delta_0)/c (= 2 epoch - JDc2 up to a constant).
  R  (all 44 main-sample asteroids, our MPC data data/obs_tables/<N>_obs.csv): relative reduction of every
     dataset (alpha > 7 deg, linear magnitude-phase relation, R^2 of that fit), wrong-sign emulation
     t_w = jd + (Delta - Delta_first)/c and correct t_c = jd_lt; the peak nearest to the original P_av is taken
     with the wrong sign and followed to the nearest peak with the correct sign.
     Dataset choice "rule3": the manuscript rule (largest dataset of each station with >= 150 measurements at
     alpha > 7 deg, the 2-3 largest stations); "all": every dataset with >= 50 measurements at alpha > 7 deg
     (the workbooks show that the original P_av combined all datasets, 8-14 per asteroid).  In both choices a
     dataset must span at least three reference periods.
Delta_sign = P_av(correct) - P_av(wrong) on identical data; P_av_new = P_av_old + Delta_sign
(Delta_sign from S where a workbook exists, otherwise from R/rule3).
Sensitivity variants (json): "emul" = highest peak within +-1 % (FAM_REL) where the original peak was the highest there;
"diapmax_consistent" (S only) = highest peak of the original zoom interval, if within the range of the original p_i.
Recounts: pav_recompute_recount.py; report: pav_recompute_report.py -> pav_recompute.md.

Run: python pav_recompute.py            -> pav_recompute.json (+ cache in pav_recompute_cache/)
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

ROOT = os.path.dirname(os.path.abspath(__file__))
AST_DATA = os.path.join(ROOT, "..", "Asteroid_data")
CACHE = os.path.join(ROOT, "pav_recompute_cache")
OUT_JSON = os.path.join(ROOT, "pav_recompute.json")
C_SHEET = 4.99 / 36 / 24          # light time for 1 AU in days, as in the workbooks
C_DAY = 0.0057755183              # 1 AU / c in days (our data)
ALPHA_MIN = 7.0
NMIN_RULE = 150
NMIN_ALL = 50
FAM_REL = 0.01                    # +-1 % in frequency: comb teeth and nearby peaks of the same alias family
                                  # (daily aliases of the 2nd harmonic are 0.5 c/d away in rotation frequency)
NPROC = int(os.environ.get("NPROC", "4"))

# ------------------------------------------------------------------------------------------------------------
# Original "Periods" blocks of the workbooks (pandas, header=None, 0-based rows; columns ds, p, w, N, R2).
# The block chosen for the main sample is the one whose Eq. (pav) value equals the published P_av
# (2607: no block reproduces the published 2.943; 3716: the published 10.743 is the block value 10.473 with
# transposed digits).  Test asteroids: P_av is not published in this manuscript (Table tab:known has none).
BLOCKS = {
    2607: dict(wb="12asteroidudatiicaruspubl/2607-excelBF.xlsx", sheet="Periods", rows=(1, 10), cols=(0, 2, 3, 5, 6),
               other=[]),
    2968: dict(wb="12asteroidudatiicaruspubl/2968-excelBD.xlsx", sheet="PeriodsD", rows=(1, 9), cols=(0, 2, 3, 5, 6),
               other=["12asteroidudatiicaruspubl/2968-excelBF.xlsx"]),
    2971: dict(wb="12asteroidudatiicaruspubl/2971-excelBD.xlsx", sheet="PeriodsD", rows=(1, 12), cols=(0, 2, 3, 5, 6),
               other=["12asteroidudatiicaruspubl/2971-excelBF.xlsx"]),
    3081: dict(wb="12asteroidudatiicaruspubl/3081-excelBD.xlsx", sheet="PeriodsD", rows=(1, 8), cols=(0, 2, 3, 5, 6),
               other=[]),
    3173: dict(wb="12asteroidudatiicaruspubl2/3173-excelBD.xlsx", sheet="PeriodsD", rows=(1, 10), cols=(0, 2, 3, 5, 6),
               other=["12asteroidudatiicaruspubl2/3173-excelBF.xlsx"]),
    3473: dict(wb="12asteroidudatiicaruspubl2/3473-excelBD.xlsx", sheet="PeriodsD", rows=(1, 13), cols=(0, 2, 3, 5, 6),
               other=["12asteroidudatiicaruspubl2/3473-excelBF.xlsx"]),
    3716: dict(wb="12asteroidudatiicaruspubl2/3716-excelBD.xlsx", sheet="Periods", rows=(61, 73), cols=(0, 2, 3, 5, 6),
               other=["12asteroidudatiicaruspubl2/3716-excelBF.xlsx"]),
    4303: dict(wb="12asteroidudatiicaruspubl2/4303-excelBD.xlsx", sheet="PeriodsD", rows=(1, 14), cols=(0, 2, 3, 5, 6),
               other=["12asteroidudatiicaruspubl2/4303-excelBF.xlsx"]),
    # test asteroids
    1951: dict(wb="12asteroidudatiicaruspubl/1951-excelBF1.xlsx", sheet="Periodi", rows=(21, 33), cols=(0, 2, 3, 5, 6),
               other=["12asteroidudatiicaruspubl/1951-excelBF.xlsx"]),
    1963: dict(wb="12asteroidudatiicaruspubl/1963-excelBF.xlsx", sheet="Periods", rows=(16, 26), cols=(0, 2, 3, 4, 5),
               other=[]),
    2134: dict(wb="12asteroidudatiicaruspubl/2134-excelBD.xlsx", sheet="Periods", rows=(1, 8), cols=(0, 2, 3, 5, 6),
               other=["12asteroidudatiicaruspubl/2134-excelBF.xlsx"]),
    2150: dict(wb="12asteroidudatiicaruspubl/2150-excelBF.xlsx", sheet="Periodi", rows=(1, 13), cols=(0, 2, 3, 5, 6),
               other=[]),
}
P_AV_TYPO = {3716: (10.743, "published 10.743 h is the workbook value 10.473 h (Periods sheet, 10.47 h block, "
                            "cell 10.473132 +- 0.002521) with transposed digits")}


# ------------------------------------------------------------------------------------------------------------
# Exact two-term Lomb-Scargle power (standard normalization, floating mean), vectorized.
def ls2_power(t, y, f):
    """Power of the model a0 + sum_{k=1,2} (a_k cos 2pi k f t + b_k sin 2pi k f t); f in 1/day, t in days."""
    t = np.asarray(t, float) - np.min(t)
    y = np.asarray(y, float)
    yc = y - y.mean()
    yy = np.sum(yc ** 2)
    f = np.atleast_1d(np.asarray(f, float))
    out = np.empty(len(f))
    n = len(t)
    chunk = max(1, int(1.5e6 // n))
    for a in range(0, len(f), chunk):
        ff = f[a:a + chunk]
        z = np.exp(2j * np.pi * ff[:, None] * t[None, :])
        z2 = z * z
        S1, S2, S3, S4 = z.sum(1), z2.sum(1), (z2 * z).sum(1), (z2 * z2).sum(1)
        Y1, Y2 = (z * yc).sum(1), (z2 * yc).sum(1)
        c1, s1, c2, s2, c3, s3, c4, s4 = S1.real, S1.imag, S2.real, S2.imag, S3.real, S3.imag, S4.real, S4.imag
        m = len(ff)
        A = np.empty((m, 5, 5))
        A[:, 0, 0] = n
        A[:, 0, 1] = A[:, 1, 0] = c1
        A[:, 0, 2] = A[:, 2, 0] = s1
        A[:, 0, 3] = A[:, 3, 0] = c2
        A[:, 0, 4] = A[:, 4, 0] = s2
        A[:, 1, 1] = (n + c2) / 2
        A[:, 2, 2] = (n - c2) / 2
        A[:, 1, 2] = A[:, 2, 1] = s2 / 2
        A[:, 1, 3] = A[:, 3, 1] = (c1 + c3) / 2
        A[:, 1, 4] = A[:, 4, 1] = (s3 + s1) / 2
        A[:, 2, 3] = A[:, 3, 2] = (s3 - s1) / 2
        A[:, 2, 4] = A[:, 4, 2] = (c1 - c3) / 2
        A[:, 3, 3] = (n + c4) / 2
        A[:, 4, 4] = (n - c4) / 2
        A[:, 3, 4] = A[:, 4, 3] = s4 / 2
        b = np.zeros((m, 5))
        b[:, 1], b[:, 2], b[:, 3], b[:, 4] = Y1.real, Y1.imag, Y2.real, Y2.imag
        try:
            beta = np.linalg.solve(A, b[..., None])[..., 0]
        except np.linalg.LinAlgError:
            beta = np.array([np.linalg.lstsq(A[i], b[i], rcond=None)[0] for i in range(m)])
        out[a:a + chunk] = np.sum(beta * b, axis=1) / yy
    return out


def _refine(t, y, f0, step):
    fg = np.linspace(f0 - 1.5 * step, f0 + 1.5 * step, 121)
    pg = ls2_power(t, y, fg)
    i = int(np.argmax(pg))
    if 0 < i < len(fg) - 1:            # parabolic interpolation
        y0, y1, y2 = pg[i - 1], pg[i], pg[i + 1]
        den = y0 - 2 * y1 + y2
        d = 0.5 * (y0 - y2) / den if den < 0 else 0.0
        fb = fg[i] + d * (fg[1] - fg[0])
    else:
        fb = fg[i]
    return float(fb), float(ls2_power(t, y, [fb])[0])


def find_peak(t, y, p_ref, prom_frac=0.0, rel=0.01, osf=20):
    """Local maximum of the nterms=2 periodogram nearest to p_ref (hours).

    prom_frac > 0: only maxima with prominence >= prom_frac * (highest power in the window) are considered
    (avoids noise ripples when the reference is not a peak of this dataset).
    Returns the peak, its power, the distance from p_ref in resolution elements (1/T), and the highest peak
    within +-FAM_REL (relative, in frequency) of it ("family maximum")."""
    T = float(np.max(t) - np.min(t))
    f_ref = 24.0 / p_ref
    half = max(rel * f_ref, 2.5 * FAM_REL * f_ref, 10.0 / T)
    step = 1.0 / (osf * T)
    f = np.arange(max(f_ref - half, step), f_ref + half, step)
    p = ls2_power(t, y, f)
    idx_all = find_peaks(p)[0]
    idx = find_peaks(p, prominence=prom_frac * p.max())[0] if prom_frac > 0 else idx_all
    if len(idx) == 0:
        return None
    j = idx[np.argmin(np.abs(f[idx] - f_ref))]
    fb, wb = _refine(t, y, f[j], step)
    # highest local maximum within +-FAM_REL of the chosen peak
    near = idx_all[np.abs(f[idx_all] - fb) <= FAM_REL * fb]
    k = near[np.argmax(p[near])] if len(near) else j
    if k != j and p[k] > p[j]:
        fk, wk = _refine(t, y, f[k], step)
    else:
        k, fk, wk = j, fb, wb
    return dict(p=24.0 / fb, w=wb, f=fb, dist_res=float((fb - f_ref) * T), T=T,
                p_fam=24.0 / fk, w_fam=wk, fam_same=bool(k == j))


def pav(p, w, N, R2):
    p, w, N, R2 = (np.asarray(x, float) for x in (p, w, N, R2))
    R2 = np.maximum(R2, 0.0)
    return float((np.sum(w * p) / w.sum() + np.sum(N * p) / N.sum() + np.sum(R2 * p) / R2.sum()) / 3.0)


def linfit_r2(x, y):
    c = np.polyfit(x, y, 1)
    res = y - np.polyval(c, x)
    return float(c[0]), float(1 - np.sum(res ** 2) / np.sum((y - y.mean()) ** 2))


# ------------------------------------------------------------------------------------------------------------
# Workbooks
def _norm(s):
    return str(s).strip().lower()


def _read_workbooks(num):
    os.makedirs(CACHE, exist_ok=True)
    fn = os.path.join(CACHE, f"sheets_{num}.pkl")
    if os.path.exists(fn):
        return pickle.load(open(fn, "rb"))
    import openpyxl
    cfg = BLOCKS[num]
    out = {}
    for rel in [cfg["wb"]] + cfg["other"]:
        path = os.path.join(AST_DATA, rel)
        wb = openpyxl.load_workbook(path, read_only=True)
        names = wb.sheetnames
        wb.close()
        dss = set()
        if rel == cfg["wb"]:
            blk = pd.read_excel(path, sheet_name=cfg["sheet"], header=None)
            r0, r1 = cfg["rows"]
            for i in range(r0, r1 + 1):
                dss.add(str(blk.iloc[i, cfg["cols"][0]]).strip())
        else:
            blk = None
        out[rel] = dict(names=names, block=blk)
    # read the sheets whose names match a dataset of the block
    blkdf = out[cfg["wb"]]["block"]
    r0, r1 = cfg["rows"]
    dss = [str(blkdf.iloc[i, cfg["cols"][0]]).strip() for i in range(r0, r1 + 1)]
    for rel in out:
        path = os.path.join(AST_DATA, rel)
        want = []
        for s in out[rel]["names"]:
            for ds in dss:
                if _base(s) == _base(ds):
                    want.append(s)
        want = sorted(set(want))
        out[rel]["sheets"] = pd.read_excel(path, sheet_name=want, header=None) if want else {}
        if "mpj" in out[rel]["names"]:      # daily ephemeris: year, month, day, INT(JD), Delta, r, phase angle
            mpj = pd.read_excel(path, sheet_name="mpj", header=None)
            out[rel]["mpj"] = mpj.apply(pd.to_numeric, errors="coerce").to_numpy(float)
    pickle.dump(out, open(fn, "wb"))
    return out


def _parse_diap(v):
    try:
        a, b = str(v).replace(",", ".").split("-")
        a, b = float(a), float(b)
        return (a, b) if 0 < a < b <= 60 else None
    except Exception:
        return None


def diap_max(t, y, diap, osf=20):
    """Highest peak of the nterms=2 periodogram in the frequency interval diap (c/d)."""
    T = float(np.max(t) - np.min(t))
    step = 1.0 / (osf * T)
    f = np.arange(diap[0], diap[1], step)
    p = ls2_power(t, y, f)
    j = int(np.argmax(p))
    fb, wb = _refine(t, y, f[j], step)
    return dict(p=24.0 / fb, w=wb, f=fb)


def _base(name):
    s = str(name).strip()
    for suf in ("kor",):
        if s.endswith(suf):
            s = s[: -len(suf)]
    if len(s) > 4 and s.endswith("a"):
        s = s[:-1]
    return s.lower()


def _sheet_arrays(df):
    """Return dict of arrays from a dataset sheet (epoch, D, ph, magred, mph, jdc2) or an 'a' sheet (jdc2, mph)."""
    if df is None or len(df) < 2:
        return None
    hdr = [_norm(x) for x in df.iloc[0].tolist()]
    def idx(pred):
        for j, h in enumerate(hdr):
            if pred(h):
                return j
        return None
    j_ep = idx(lambda h: h == "epoch")
    j_jd = idx(lambda h: h in ("jdc2", "jdc2a"))
    j_m = idx(lambda h: h.startswith("magph") or h.startswith("mgph"))
    if j_jd is None or j_m is None:
        return None
    body = df.iloc[1:]
    num = lambda j: pd.to_numeric(body.iloc[:, j], errors="coerce").to_numpy(float)
    d = dict(jdc2=num(j_jd), mph=num(j_m))
    if j_ep is not None:
        d["epoch"] = num(j_ep)
        d["D"] = num(j_ep + 2)
        j_ph = idx(lambda h: h == "ph")
        d["ph"] = num(j_ph) if j_ph is not None else num(j_ep + 4)
        j_mr = idx(lambda h: h == "magred")
        d["magred"] = num(j_mr) if j_mr is not None else np.full(len(body), np.nan)
        ok = np.isfinite(d["epoch"]) & np.isfinite(d["jdc2"]) & np.isfinite(d["mph"]) & np.isfinite(d["D"])
        d["kind"] = "dataset"
    else:
        ok = np.isfinite(d["jdc2"]) & np.isfinite(d["mph"])
        d["kind"] = "a"
    for k in list(d):
        if isinstance(d[k], np.ndarray):
            d[k] = d[k][ok]
    return d


def block_sources(num):
    """Rows of the original block with the data actually analysed for every dataset."""
    cfg = BLOCKS[num]
    W = _read_workbooks(num)
    blk = W[cfg["wb"]]["block"]
    r0, r1 = cfg["rows"]
    cd, cp, cw, cn, cr = cfg["cols"]
    rows = []
    diap0 = None
    for i in range(r0, r1 + 1):
        ds = str(blk.iloc[i, cd]).strip()
        vals = [pd.to_numeric(blk.iloc[i, c], errors="coerce") for c in (cp, cw, cn, cr)]
        dp = _parse_diap(blk.iloc[i, 1])        # frequency interval (c/d) of the zoomed periodogram, if given
        if dp is not None and diap0 is None:
            diap0 = dp
        rows.append(dict(ds=ds, p=float(vals[0]), w=float(vals[1]), N=float(vals[2]), R2=float(vals[3]),
                         diap=dp if dp is not None else diap0))
    # candidate sources
    cands = []
    for rel in [cfg["wb"]] + cfg["other"]:
        for s, df in W[rel]["sheets"].items():
            a = _sheet_arrays(df)
            if a is not None:
                cands.append((rel, s, a))
    dsets = [(rel, s, a) for rel, s, a in cands if a["kind"] == "dataset"]
    for r in rows:
        mine = [(rel, s, a) for rel, s, a in cands if _base(s) == _base(r["ds"])]
        if not mine:
            r["source"] = None
            continue
        # prefer exact N, dataset sheets, the block's own workbook
        def score(c):
            rel, s, a = c
            return (abs(len(a["mph"]) - r["N"]), 0 if a["kind"] == "dataset" else 1, 0 if rel == cfg["wb"] else 1)
        rel, s, a = sorted(mine, key=score)[0]
        a = dict(a)
        if a["kind"] == "a":
            # recover epoch, Delta, alpha from a dataset sheet of the same name: the observation with the same
            # magph value within 0.03 d of JDc2 (the "a" sheets are value copies of columns JDc2 and magph,
            # sometimes of an earlier version of the sheet whose Delta_0 differed, i.e. JDc2 offset by a constant)
            ep = np.full(len(a["jdc2"]), np.nan); D = ep.copy(); ph = ep.copy(); mr = ep.copy()
            for rel2, s2, b in dsets:
                if _base(s2) != _base(r["ds"]):
                    continue
                order = np.argsort(b["epoch"])
                be = b["epoch"][order]
                for k in range(len(a["jdc2"])):
                    if np.isfinite(ep[k]):
                        continue
                    lo, hi = np.searchsorted(be, a["jdc2"][k] - 0.03), np.searchsorted(be, a["jdc2"][k] + 0.03)
                    if hi <= lo:
                        continue
                    qs = order[lo:hi]
                    dm = np.abs(b["mph"][qs] - a["mph"][k])
                    q = qs[int(np.argmin(dm))]
                    if dm.min() < 1e-6:
                        ep[k], D[k], ph[k], mr[k] = b["epoch"][q], b["D"][q], b["ph"][q], b["magred"][q]
            a.update(epoch=ep, D=D, ph=ph, magred=mr)
            # Delta of unmatched rows from the daily ephemeris (VLOOKUP of INT(epoch) as in the sheets;
            # |JDc2 - epoch| < 0.02 d, so INT(JDc2) gives the same or the adjacent day: < 0.1 min in light time)
            miss = ~np.isfinite(a["D"])
            if miss.any():
                for rel2 in [rel] + [x for x in [cfg["wb"]] + cfg["other"] if x != rel]:
                    mpj = W[rel2].get("mpj")
                    if mpj is None:
                        continue
                    jd_int = mpj[:, 3]
                    ok = np.isfinite(jd_int)
                    lut = dict(zip(jd_int[ok].astype(np.int64), mpj[ok, 4]))
                    for k in np.where(miss)[0]:
                        d0 = int(np.floor(a["jdc2"][k]))
                        for dd in (d0, d0 - 1, d0 + 1):     # adjacent day if the day is missing in mpj
                            if dd in lut and np.isfinite(lut[dd]):
                                a["D"][k] = lut[dd]
                                break
                    a["D_from_ephemeris"] = int(miss.sum())
                    break
        a["matched"] = float(np.mean(np.isfinite(a["epoch"])))
        r["source"] = dict(workbook=os.path.basename(rel), sheet=s, kind=a["kind"], n=int(len(a["mph"])),
                           matched_epoch=a["matched"], delta_from_ephemeris=int(a.get("D_from_ephemeris", 0)))
        r["_a"] = a
    return rows


def run_S(num):
    rows = block_sources(num)
    out = []
    for r in rows:
        rec = {k: v for k, v in r.items() if k != "_a"}
        a = r.get("_a")
        if a is None:
            rec["flag"] = "no data sheet"
            out.append(rec)
            continue
        m = np.isfinite(a["D"])
        if m.sum() < 10:
            rec["flag"] = "Delta of the 'a' sheet rows could not be recovered"
            out.append(rec)
            continue
        t_w = a["jdc2"][m]
        y = a["mph"][m]
        # corrected times: JDc2 = epoch + (Delta - Delta_ref)/c  ->  JDc2 - 2 Delta/c = epoch - (Delta + Delta_ref)/c,
        # i.e. the correct epoch - (Delta - Delta_0)/c up to a constant, which does not affect a periodogram
        t_c = t_w - 2 * a["D"][m] * C_SHEET
        rec["N_data"] = int(m.sum())
        ph, mr = a["ph"][m], a["magred"][m]
        okr = np.isfinite(ph) & np.isfinite(mr)
        if okr.sum() >= 0.9 * m.sum() and okr.sum() > 5:
            rec["R2_data"] = linfit_r2(ph[okr], mr[okr])[1]
        else:
            rec["R2_data"] = r["R2"]
            rec["R2_note"] = "magred/alpha not available for this copy; original R^2 used"
        dt = (t_w - t_c) * 1440
        rec["timing_error_spread_min"] = float(dt.max() - dt.min())     # = 2 (Delta_max - Delta_min)/c
        pw = find_peak(t_w, y, r["p"])
        pc = find_peak(t_c, y, pw["p"]) if pw else None
        rec["wrong"], rec["correct"] = pw, pc
        if r.get("diap") and pw:
            # was the original p_i the highest peak of the zoomed interval?  then an analyst would again take
            # the highest peak there with the corrected times
            dw, dc = diap_max(t_w, y, r["diap"]), diap_max(t_c, y, r["diap"])
            res_el = 1.0 / pw["T"]
            rec["diap_max"] = dict(interval=r["diap"], wrong=dw, correct=dc,
                                   original_was_max=bool(abs(dw["f"] - pw["f"]) < 0.5 * res_el),
                                   max_moved=bool(abs(dc["f"] - pc["f"]) >= 0.5 * res_el))
        out.append(rec)
    return out


# ------------------------------------------------------------------------------------------------------------
# Our data (MPC re-analysis obs tables)
def load_obs(num):
    df = pd.read_csv(os.path.join(ROOT, "data", "obs_tables", f"{num}_obs.csv"))
    df = df[np.isfinite(df["mag_f"]) & np.isfinite(df["alpha"]) & np.isfinite(df["jd"])]
    out = {}
    for ds, g in df.groupby("dataset"):
        g = g.sort_values("jd")
        g7 = g[g["alpha"] > ALPHA_MIN]
        if len(g7) < 10:
            continue
        mred = g7["mag_f"].to_numpy() - 5 * np.log10(g7["r"].to_numpy() * g7["delta"].to_numpy())
        al = g7["alpha"].to_numpy()
        k, R2 = linfit_r2(al, mred)
        y = mred - (al - al[0]) * k
        jd = g7["jd"].to_numpy()
        de = g7["delta"].to_numpy()
        out[ds] = dict(stn=str(g7["stn"].iloc[0]), N=int(len(g7)), N_all=int(len(g)), R2=R2, k=k, y=y,
                       t_w=jd + (de - de[0]) * C_DAY, t_c=g7["jd_lt"].to_numpy(),
                       jd_min=float(jd.min()), jd_max=float(jd.max()))
    return out


def select_datasets(obs, p_ref):
    """rule3 and all-dataset choices; a dataset must span at least 3 reference periods (e.g. a single TESS visit
    cannot measure a 447 h period)."""
    obs = {d: o for d, o in obs.items() if (o["jd_max"] - o["jd_min"]) >= 3 * p_ref / 24.0}
    by_n = sorted(obs, key=lambda d: -obs[d]["N"])
    best_per_stn = {}
    for d in by_n:
        best_per_stn.setdefault(obs[d]["stn"], d)
    elig = [d for d in best_per_stn.values() if obs[d]["N"] >= NMIN_RULE]
    elig = sorted(elig, key=lambda d: -obs[d]["N"])
    flag = ""
    if len(elig) >= 2:
        rule3 = elig[:3]
    else:
        rest = [d for d in sorted(best_per_stn.values(), key=lambda d: -obs[d]["N"]) if d not in elig and obs[d]["N"] >= 70]
        rule3 = (elig + rest)[:3] if len(elig) == 1 else rest[:3]
        flag = f"only {len(elig)} station(s) with >= {NMIN_RULE} measurements at alpha > {ALPHA_MIN:g} deg"
    alls = [d for d in by_n if obs[d]["N"] >= NMIN_ALL]
    return rule3, alls, flag


def run_R(num, p_ref, datasets, obs):
    res = []
    for d in datasets:
        o = obs[d]
        pw = find_peak(o["t_w"], o["y"], p_ref, prom_frac=0.1)
        pc = find_peak(o["t_c"], o["y"], pw["p"]) if pw else None
        dt = (o["t_w"] - o["t_c"]) * 1440
        res.append(dict(ds=d, N=o["N"], R2=o["R2"], k=o["k"], jd_range=[o["jd_min"], o["jd_max"]], wrong=pw, correct=pc,
                        timing_error_spread_min=float(dt.max() - dt.min())))
    return res


def summarize(rows, key_p="p", src="R"):
    """P_av for the wrong and the correct sign from per-dataset results."""
    ok = [r for r in rows if r.get("wrong") and r.get("correct")]
    if len(ok) < 1:
        return None
    N = [r["N_data"] if src == "S" else r["N"] for r in ok]
    R2 = [r["R2_data"] if src == "S" else r["R2"] for r in ok]
    res = {}
    def _moved(p1, p2, r):           # different peaks: more than half a resolution element apart
        return abs(24 / p1 - 24 / p2) > 0.5 / r["wrong"]["T"]
    for sign in ("wrong", "correct"):
        res[f"P_av_{sign}"] = pav([r[sign]["p"] for r in ok], [r[sign]["w"] for r in ok], N, R2)
        res[f"P_av_{sign}_fam"] = pav([r[sign]["p_fam"] for r in ok], [r[sign]["w_fam"] for r in ok], N, R2)
    res["delta_sign"] = res["P_av_correct"] - res["P_av_wrong"]
    res["delta_sign_rel"] = res["delta_sign"] / res["P_av_wrong"]
    res["n_datasets"] = len(ok)
    res["delta_sign_fam"] = res["P_av_correct_fam"] - res["P_av_wrong_fam"]
    # analyst emulation: where the original (wrong-sign) peak was the highest peak within +-1 % of itself, take the
    # highest peak within +-1 % also with the correct sign; otherwise keep following the same peak
    pe = [r["correct"]["p_fam"] if r["wrong"]["fam_same"] else r["correct"]["p"] for r in ok]
    we = [r["correct"]["w_fam"] if r["wrong"]["fam_same"] else r["correct"]["w"] for r in ok]
    res["P_av_correct_emul"] = pav(pe, we, N, R2)
    res["delta_sign_emul"] = res["P_av_correct_emul"] - res["P_av_wrong"]
    res["emul_peak_changed"] = [r["ds"] for r, p in zip(ok, pe) if _moved(p, r["correct"]["p"], r)]
    res["n_original_was_fammax"] = int(sum(r["wrong"]["fam_same"] for r in ok))
    if src == "S" and any("diap_max" in r for r in ok):
        # second emulation: where the original p_i was the highest peak of the zoomed interval ("diap"),
        # take the highest peak of that interval with the corrected times
        pd_, wd_, moved = [], [], []
        for r in ok:
            dm = r.get("diap_max")
            if dm and dm["original_was_max"]:
                pd_.append(dm["correct"]["p"]); wd_.append(dm["correct"]["w"])
                if dm["max_moved"]:
                    moved.append(r["ds"])
            else:
                pd_.append(r["correct"]["p"]); wd_.append(r["correct"]["w"])
        res["P_av_correct_diapmax"] = pav(pd_, wd_, N, R2)
        res["delta_sign_diapmax"] = res["P_av_correct_diapmax"] - res["P_av_wrong"]
        res["n_original_was_diapmax"] = int(sum(1 for r in ok if r.get("diap_max", {}).get("original_was_max")))
        res["diapmax_moved"] = moved
        # the same, but a new highest peak is accepted only if it lies within the range of the original p_i of the
        # block (the analyst kept only peaks consistent with the other datasets)
        lo, hi = min(r["p"] for r in ok), max(r["p"] for r in ok)
        pc_, wc_, acc = [], [], []
        for r, p, w in zip(ok, pd_, wd_):
            if lo <= p <= hi:
                pc_.append(p); wc_.append(w)
                if abs(24 / p - 24 / r["correct"]["p"]) > 1e-6 and r["ds"] in moved:
                    acc.append(r["ds"])
            else:
                pc_.append(r["correct"]["p"]); wc_.append(r["correct"]["w"])
        res["P_av_correct_diapmax_consistent"] = pav(pc_, wc_, N, R2)
        res["delta_sign_diapmax_consistent"] = res["P_av_correct_diapmax_consistent"] - res["P_av_wrong"]
        res["diapmax_consistent_moved"] = acc
    # datasets whose highest peak within +-1 % of the followed peak changes with the sign
    res["fammax_moved"] = [r["ds"] for r in ok if _moved(r["correct"]["p_fam"], r["wrong"]["p_fam"], r)]
    res["max_abs_dp_rel"] = float(max(abs(r["correct"]["p"] - r["wrong"]["p"]) / r["wrong"]["p"] for r in ok))
    return res


# ------------------------------------------------------------------------------------------------------------
def process(num):
    t0 = time.time()
    pub = json.load(open(os.path.join(ROOT, "published.json"), encoding="utf-8"))
    ad = json.load(open(os.path.join(ROOT, "adopted.json"), encoding="utf-8"))
    key = str(num)
    rec = dict(asteroid=num, table=pub[key]["table"], P_adopted=ad[key]["P"], status=ad[key].get("status", "determined"),
               quality=ad[key].get("quality"))
    P_old_pub = pub[key].get("P_av")
    rec["P_av_old_published"] = P_old_pub
    P_old = P_old_pub
    flags = []
    if num in P_AV_TYPO:
        flags.append(P_AV_TYPO[num][1])
    # ---- S approach
    if num in BLOCKS:
        rows = run_S(num)
        cfg = BLOCKS[num]
        orig = [r for r in rows if np.isfinite(r["p"])]
        P_sheet = pav([r["p"] for r in orig], [r["w"] for r in orig], [r["N"] for r in orig], [r["R2"] for r in orig])
        sS = summarize(rows, src="S")
        repro = []
        for r in rows:
            if r.get("wrong"):
                repro.append(dict(ds=r["ds"], dp=r["wrong"]["p"] - r["p"], dw=r["wrong"]["w"] - r["w"],
                                  dN=r["N_data"] - r["N"], dR2=r["R2_data"] - r["R2"]))
        # hybrid: reproduced p_i with the original weights
        okr = [r for r in rows if r.get("wrong")]
        P_hyb = pav([r["wrong"]["p"] for r in okr], [r["w"] for r in okr], [r["N"] for r in okr], [r["R2"] for r in okr])
        rec["S"] = dict(workbook=cfg["wb"], sheet=cfg["sheet"], rows_0based=cfg["rows"], P_av_sheet_eq=P_sheet,
                        P_av_repro_pi_only=P_hyb, summary=sS, reproduction=repro, datasets=rows)
        if P_old is None:
            P_old = P_sheet
            rec["P_av_old_source"] = "workbook block (not published in this manuscript)"
        elif num in P_AV_TYPO:
            P_old = P_sheet
            rec["P_av_old_source"] = "workbook block (published value is a typo)"
        else:
            rec["P_av_old_source"] = "published"
            if abs(round(P_sheet, 3) - P_old_pub) > 0.0006:
                flags.append(f"published P_av {P_old_pub} not reproduced by the workbook block (Eq. pav {P_sheet:.5f})")
    else:
        rec["P_av_old_source"] = "published"
    rec["P_av_old"] = P_old
    # ---- R approach (our data)
    if P_old is not None:
        obs = load_obs(num)
        rule3, alls, fl = select_datasets(obs, P_old)
        if fl:
            flags.append(fl)
        rec["R_rule3"] = dict(datasets=run_R(num, P_old, rule3, obs))
        rec["R_rule3"]["summary"] = summarize(rec["R_rule3"]["datasets"])
        rec["R_all"] = dict(datasets=run_R(num, P_old, alls, obs))
        rec["R_all"]["summary"] = summarize(rec["R_all"]["datasets"])
        rec["datasets_available"] = {d: dict(N_alpha7=obs[d]["N"], N_all=obs[d]["N_all"], stn=obs[d]["stn"]) for d in obs}
    # ---- adopted new value
    if P_old is not None:
        if "S" in rec and rec["S"]["summary"]:
            src = "S"
            dlt = rec["S"]["summary"]["delta_sign"]
        else:
            src = "R_rule3"
            dlt = rec["R_rule3"]["summary"]["delta_sign"]
        rec["delta_sign_source"] = src
        rec["delta_sign"] = dlt
        rec["P_av_new"] = P_old + dlt
        rec["rel_change"] = dlt / P_old
        sm = rec["S"]["summary"] if src == "S" else rec["R_rule3"]["summary"]
        dsl = rec["S"]["datasets"] if src == "S" else rec["R_rule3"]["datasets"]
        prim = []
        for d in dsl:
            if not (d.get("wrong") and d.get("correct")):
                continue
            prim.append(dict(ds=d["ds"], N_i=d.get("N_data", d.get("N")), R2_i=d.get("R2_data", d.get("R2")),
                             p_i_original=d["p"] if src == "S" else None, w_i_original=d["w"] if src == "S" else None,
                             p_i_wrong=d["wrong"]["p"], w_i_wrong=d["wrong"]["w"],
                             p_i_new=d["correct"]["p"], w_i_new=d["correct"]["w"],
                             far_from_reference=bool(abs(d["wrong"]["p"] / (d["p"] if src == "S" else P_old) - 1) > 0.02)))
        rec["primary"] = dict(source=src, P_av_wrong_repro=sm["P_av_wrong"], P_av_correct_direct=sm["P_av_correct"],
                              datasets=prim)
        if any(x["far_from_reference"] for x in prim):
            flags.append("peak more than 2% from the reference in: " +
                         ", ".join(x["ds"] for x in prim if x["far_from_reference"]))
        rec["delta_sign_emul"] = sm["delta_sign_emul"]
        rec["rel_change_emul"] = sm["delta_sign_emul"] / P_old
        alt = [rec[v]["summary"]["delta_sign"] / P_old for v in ("S", "R_rule3", "R_all") if v in rec and rec[v]["summary"]]
        alt += [rec[v]["summary"]["delta_sign_emul"] / P_old for v in ("S", "R_rule3", "R_all") if v in rec and rec[v]["summary"]]
        if "S" in rec and rec["S"]["summary"] and "delta_sign_diapmax" in rec["S"]["summary"]:
            # the unconstrained diap-max emulation is reported separately (it jumps to inconsistent aliases)
            rec["rel_change_diapmax_unconstrained"] = rec["S"]["summary"]["delta_sign_diapmax"] / P_old
            rec["rel_change_diapmax_consistent"] = rec["S"]["summary"]["delta_sign_diapmax_consistent"] / P_old
            alt.append(rec["rel_change_diapmax_consistent"])
        rec["rel_change_max_abs_all_variants"] = float(max(abs(x) for x in alt))
    rec["flags"] = flags
    rec["runtime_s"] = time.time() - t0
    return rec


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items() if not k.startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def main(argv):
    pub = json.load(open(os.path.join(ROOT, "published.json"), encoding="utf-8"))
    nums = [int(a) for a in argv if not a.startswith("-")]
    if not nums:
        nums = [int(k) for k in pub if not k.startswith("_") and (pub[k].get("P_av") is not None or int(k) in BLOCKS)]
    for n in nums:                       # build workbook caches serially (openpyxl/pandas I/O)
        if n in BLOCKS:
            _read_workbooks(n)
    with ProcessPoolExecutor(max_workers=NPROC) as ex:
        recs = list(ex.map(process, nums))
    out = {}
    if os.path.exists(OUT_JSON) and argv:
        out = json.load(open(OUT_JSON, encoding="utf-8")).get("asteroids", {})
    for r in recs:
        out[str(r["asteroid"])] = _clean(r)
    for k in pub:                       # test asteroids without any separate-dataset value
        if not k.startswith("_") and k not in out:
            out[k] = dict(asteroid=int(k), table=pub[k]["table"], P_av_old=None, P_av_new=None,
                          flags=["no separate-dataset value: P_av not published in this manuscript and no workbook"])
        s = r.get("P_av_new")
        print(f"{r['asteroid']}: P_av_old={r.get('P_av_old')}  P_av_new={s}  rel={r.get('rel_change')}  flags={r['flags']}",
              flush=True)
    meta = dict(script="pav_recompute.py", created=time.strftime("%Y-%m-%d %H:%M"),
                method="nterms=2 exact Lomb-Scargle (standard normalization); peak nearest to the reference followed "
                       "from the wrong-sign to the correct-sign times; Eq. eq:pav",
                constants=dict(C_SHEET=C_SHEET, C_DAY=C_DAY, ALPHA_MIN=ALPHA_MIN, NMIN_RULE=NMIN_RULE, NMIN_ALL=NMIN_ALL,
                               FAM_REL=FAM_REL))
    tmp = OUT_JSON + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(dict(meta=meta, asteroids=dict(sorted(out.items()))), fh, indent=1)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, OUT_JSON)


if __name__ == "__main__":
    main(sys.argv[1:])
