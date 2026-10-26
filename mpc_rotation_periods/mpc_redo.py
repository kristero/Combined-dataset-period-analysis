"""
Combined-dataset rotation-period analysis of MPC photometry.

The data reduction and the Lomb-Scargle analysis use the unmodified classes of the
Combined-dataset-period-analysis repository:
    Asteroid_pickle.DatasetGenerator   (HG1G2 reference fit, shift of every dataset to the
                                        reference phase curve, removal of the phase dependence)
    asteroid_ls.AsteroidLSPipeline      (combination, outlier removal, periodogram figure,
                                        folding, spectral window, peak-width fit)
Input workbooks are produced by mpc_fetch.py (MPC observations + JPL Horizons geometry,
light-time-corrected epochs in the "Jdc2" column).

Added here (diagnostics for the choice between alias peaks):
  * dataset screening of Step 7 of the paper (free-fit boundary values, chi2 > 3 chi2_ref);
  * two-term Lomb-Scargle over 0.8-1500 h (primary range 0.8-240 h), scan with 5 samples per
    peak, candidate peaks refined with 50 samples per peak;
  * peak families (local maxima within +/-1 % in period = inter-apparition comb of one solution);
  * candidate periods: strongest families, harmonic partners of the best peak, all periods quoted
    in the submitted manuscript;
  * night-block bootstrap: fraction of resamplings in which each candidate gives the highest
    power, scatter of the peak position (period uncertainty);
  * independent checks: longitude groups, TESS alone, single datasets, leave-one-dataset-out,
    noise-weighted periodogram;
  * fold statistics (residual rms, extrema of the four-term Fourier model).

Usage: python mpc_redo.py [numbers...] [--workers N]
"""
from __future__ import annotations

import contextlib
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
import warnings

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = r"C:\Users\kn18001\Documents\Asteroids\Combined-dataset-period-analysis"
ROOT = os.path.dirname(os.path.abspath(__file__))
WB_DIR = os.path.join(ROOT, "data", "workbooks")
OBS_DIR = os.path.join(ROOT, "data", "obs_tables")
RES_DIR = os.path.join(ROOT, "results")
os.makedirs(RES_DIR, exist_ok=True)

P_MIN = 0.8            # h (lower limit stated in the paper)
P_MAX_PRIMARY = 240.0  # h
P_MAX_EXT = 1500.0     # h
SPP_SCAN = 5           # samples per peak for the full-range scan
SPP = 50               # samples per peak for the refinement windows around candidates
FAMILY_FRAC = 0.01     # local maxima within +/-1 % in period form one peak family (inter-apparition comb)
PEAK_SIGMA = 5
N_BOOT = 300
SIDEREAL_CPD = 1.0027379
REF_PREFERENCE = ["T08o", "T05o", "G96G", "F51w", "703G", "703V", "C57G"]

LON_GROUP = {
    "americas": ["T05", "T08", "T07", "F51", "F52", "703", "G96", "691", "I41", "689", "704", "H45",
                 "G45", "W68", "Y00", "V00", "W84", "I52", "V06", "I11", "U68", "U55", "T12", "T14", "645",
                 "E10", "807", "809", "Z84"],
    "europe_africa": ["M22", "R17", "Z22", "069", "J43", "A50", "L01", "K91", "J04"],
    "asia_australia": ["D29", "P07", "Q62", "E12", "N50", "O75", "Q63", "Q64", "D35"],
    "space": ["C57", "C51", "258", "274"],
}
GROUP_OF = {stn: g for g, lst in LON_GROUP.items() for stn in lst}

with open(os.path.join(ROOT, "published.json"), "r", encoding="utf-8") as _fh:
    PUBLISHED = json.load(_fh)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.floating):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, float):
        return None if not np.isfinite(o) else o
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    return o


def ls_power(t, y, freq, nterms=2, dy=None):
    from astropy.timeseries import LombScargle
    ls = LombScargle(t, y, dy=dy, nterms=nterms)
    return ls.power(freq, method="fastchi2", assume_regular_frequency=True)


def window_grid(f0, baseline_days, half_width_res=20.0, spp=SPP):
    df = 1.0 / (spp * baseline_days)
    hw = min(half_width_res / baseline_days, 0.5 * f0)   # never reach zero frequency
    n = int(np.ceil(2 * hw / df)) + 1
    return (f0 - hw) + df * np.arange(n)


def relation_to(P, P_ref, baseline_days):
    """Harmonic / daily-alias relation of period P to P_ref (hours)."""
    f, fr = 24.0 / P, 24.0 / P_ref
    tol = max(3.0 / baseline_days, 0.004)
    rel = []
    for num, den, name in [(2, 1, "P/2"), (1, 2, "2P"), (3, 2, "2P/3"), (2, 3, "3P/2"), (3, 1, "P/3"),
                           (1, 3, "3P"), (4, 1, "P/4"), (1, 4, "4P")]:
        if abs(f - fr * num / den) < tol:
            rel.append(name)
    if abs(f - fr) < tol:
        rel.append("same")
    d2 = abs(2 * f - 2 * fr)       # modulation (second harmonic) frequencies
    for day in (1.0, SIDEREAL_CPD):
        k = int(round(d2 / day))
        if k >= 1 and abs(d2 - k * day) < tol:
            rel.append(f"daily alias of 2f (k={k}; rotation frequency differs by {k / 2:g} c/d)")
            break
    d1 = abs(f - fr)
    for day in (1.0, SIDEREAL_CPD):
        k = int(round(d1 / day))
        if k >= 1 and abs(d1 - k * day) < tol and k % 2 == 1:
            rel.append(f"daily alias of f (k={k})")
            break
    return rel


def fourier_fit(t, y, P, nterms):
    w = 2 * np.pi * 24.0 / P
    cols = [np.ones_like(t)]
    for k in range(1, nterms + 1):
        cols += [np.sin(k * w * t), np.cos(k * w * t)]
    A = np.vstack(cols).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    model = A @ coef
    phg = np.linspace(0, 1, 1000, endpoint=False)
    colsg = [np.ones_like(phg)]
    for k in range(1, nterms + 1):
        colsg += [np.sin(2 * np.pi * k * phg), np.cos(2 * np.pi * k * phg)]
    mg = np.vstack(colsg).T @ coef
    return model, mg


def count_extrema(mg, prominence=0.02):
    from scipy.signal import find_peaks
    n = len(mg)
    ext = np.concatenate([mg, mg, mg])
    pk, _ = find_peaks(ext, prominence=prominence)
    tr, _ = find_peaks(-ext, prominence=prominence)
    return int(np.sum((pk >= n) & (pk < 2 * n))), int(np.sum((tr >= n) & (tr < 2 * n)))


def fold_stats(t, y, P):
    m2, g2 = fourier_fit(t, y, P, 2)
    m4, g4 = fourier_fit(t, y, P, 4)
    n_max, n_min = count_extrema(g4, prominence=0.02)
    return {"rms_n2": float(np.std(y - m2)), "rms_n4": float(np.std(y - m4)),
            "amp_n2": float(g2.max() - g2.min()), "amp_n4": float(g4.max() - g4.min()),
            "n4_maxima": n_max, "n4_minima": n_min}


def peak_families(periods, power, idx_sorted_by_power, frac=FAMILY_FRAC, n_max=15):
    fams = []
    for i in idx_sorted_by_power:
        P = periods[i]
        for fam in fams:
            if abs(P - fam["P"]) <= frac * fam["P"]:
                fam["teeth"].append((float(P), float(power[i])))
                break
        else:
            fams.append({"P": float(P), "power": float(power[i]), "i": int(i), "teeth": [(float(P), float(power[i]))]})
            if len(fams) >= n_max:
                break
    for fam in fams:
        t = sorted(fam["teeth"], key=lambda x: -x[1])
        fam["n_teeth"] = len(t)
        fam["second_tooth"] = t[1] if len(t) > 1 else None
        fam["second_tooth_ratio"] = float(t[1][1] / t[0][1]) if len(t) > 1 else 0.0
        fam["teeth"] = t[:6]
    return fams


def analyse(num: int) -> dict:
    t_start = time.time()
    sys.path.insert(0, REPO)
    from astropy.timeseries import LombScargle
    from scipy.signal import find_peaks
    from Asteroid_pickle import DatasetGenerator
    from asteroid_ls import AsteroidLSPipeline

    out_dir = os.path.join(RES_DIR, str(num))
    os.makedirs(out_dir, exist_ok=True)
    out = out_dir + os.sep
    with open(os.path.join(OBS_DIR, f"{num}_info.json"), "r", encoding="utf-8") as fh:
        info = json.load(fh)
    datasets = list(info["datasets"].keys())
    ref = next((r for r in REF_PREFERENCE if r in datasets), datasets[0])
    sheets = [ref] + [d for d in datasets if d != ref]
    res = {"asteroid": num, "reference": ref, "datasets_available": info["datasets"],
           "mpc_retrieval": {k: info[k] for k in ("n_records", "n_with_mag", "n_kept", "utc_first", "utc_last")},
           "settings": {"P_min": P_MIN, "P_max_primary": P_MAX_PRIMARY, "P_max_ext": P_MAX_EXT,
                        "spp_scan": SPP_SCAN, "spp_refine": SPP, "n_boot": N_BOOT, "nterms": 2}}

    gen = DatasetGenerator(WB_DIR + os.sep, f"{num}-mpc.xlsx", num, sheets, base_dir=REPO)

    # --- 1) reference HG1G2 fit --------------------------------------------------------------
    H, G1, G2, H_e, G1_e, G2_e = gen.reference_obs_comp(ref, method="HG1G2", return_errors=True)
    plt.close("all")
    res["reference_fit"] = {"H": H, "G1": G1, "G2": G2, "H_err": H_e, "G1_err": G1_e, "G2_err": G2_e}

    # --- 2) dataset screening (paper Step 7) ---------------------------------------------------
    screen = []
    for s in sheets:
        ph, Hs, _t = gen._load_sheet_for_calibration(s, outlier_sigma=1.8)
        fixed = gen.fit(ph, Hs, method="HG1G2", G1=G1, G2=G2)
        free = gen.fit(ph, Hs, method="HG1G2")
        g1f, g2f = float(free.params["G1"].value), float(free.params["G2"].value)
        screen.append({"dataset": s, "n": int(len(Hs)), "alpha_min": float(np.min(ph)), "alpha_max": float(np.max(ph)),
                       "H_fixed": float(fixed.params["H"].value), "redchi_fixed": float(fixed.redchi),
                       "G1_free": g1f, "G2_free": g2f, "redchi_free": float(free.redchi),
                       "boundary": bool((g1f < 0.01 and g2f > 0.99) or (g1f > 0.99 and g2f < 0.01))})
    ref_redchi = next(r["redchi_fixed"] for r in screen if r["dataset"] == ref)
    for r in screen:
        r["chi2_ratio"] = r["redchi_fixed"] / ref_redchi
        r["keep"] = bool(r["dataset"] == ref or (not r["boundary"] and r["chi2_ratio"] <= 3.0))
        r["reason"] = "reference" if r["dataset"] == ref else (
            "boundary G1/G2" if r["boundary"] else ("chi2 > 3 chi2_ref" if r["chi2_ratio"] > 3.0 else "kept"))
    kept = [r["dataset"] for r in screen if r["keep"]]
    res["screening"] = screen
    res["ref_redchi"] = ref_redchi
    res["datasets_used"] = kept

    # --- 3) combination with the repository code ------------------------------------------------
    gen.reduced_obs = kept
    dict_sheets = gen.all_obs_comb(H, G1_val=G1, G2_val=G2, method="HG1G2", save_figure=True, save_figures=out,
                                   save_file=False, ref_redchi=ref_redchi)
    plt.close("all")
    pipe = AsteroidLSPipeline(dict_sheets, num, base_dir=REPO)
    ts, ys = pipe.reading_data()
    plt.close("all")
    sheet_kept = np.delete(pipe.sheet, pipe.remove_idx)
    baseline = float(np.max(ts) - np.min(ts))
    res.update({"n_combined": int(len(pipe.y)), "n_after_outlier_removal": int(len(ys)),
                "n_per_dataset": {k: int(np.sum(sheet_kept == k)) for k in kept},
                "time_zero_jd": float(pipe.time_zero), "baseline_days": baseline})
    # per-dataset scatter of the reduced magnitudes -> weights for the weighted periodogram
    sd = {ds: float(np.std(ys[sheet_kept == ds], ddof=1)) for ds in kept if np.sum(sheet_kept == ds) > 2}
    dy = np.array([sd.get(s, np.nan) for s in sheet_kept])
    dy[~np.isfinite(dy)] = np.nanmedian(dy)
    res["dataset_scatter_mag"] = sd

    # --- 4) periodogram scan 0.8 - 1500 h ---------------------------------------------------------
    ls = LombScargle(ts, ys, nterms=2)
    fmin, fmax = 24.0 / P_MAX_EXT, 24.0 / P_MIN
    freq = ls.autofrequency(samples_per_peak=SPP_SCAN, minimum_frequency=fmin, maximum_frequency=fmax)
    power = ls.power(freq, method="fastchi2", assume_regular_frequency=True)
    pipe.ls, pipe.nterms = ls, 2
    periods = 24.0 / freq
    prim = periods <= P_MAX_PRIMARY

    peaks_fig = pipe.find_peaks(freq[prim], power[prim], height=None, distance=200, number_of_sigma=PEAK_SIGMA,
                                save_fig=True, save_figures=out, xscale="log", merge_frac=0.02, max_peaks=8)
    plt.close("all")
    threshold = float(pipe.last_peak_threshold)
    res["threshold_5sigma"] = threshold

    pk_all, _ = find_peaks(power, distance=3)
    pk_prim = pk_all[prim[pk_all]]
    pk_ext = pk_all[~prim[pk_all]]
    fam_prim = peak_families(periods, power, pk_prim[np.argsort(power[pk_prim])[::-1]][:400])
    fam_ext = peak_families(periods, power, pk_ext[np.argsort(power[pk_ext])[::-1]][:100], n_max=5)

    def refine(P_seed, frac=0.003, spp=SPP):
        g = np.arange(24.0 / (P_seed * (1 + frac)), 24.0 / (P_seed * (1 - frac)), 1.0 / (spp * baseline))
        p = ls_power(ts, ys, g)
        k = int(np.argmax(p))
        return float(24.0 / g[k]), float(g[k]), float(p[k])

    for fam in fam_prim + fam_ext:
        fam["P"], fam["f"], fam["power"] = refine(fam["P"], frac=0.0015)
    fam_prim.sort(key=lambda x: -x["power"])
    P_best = fam_prim[0]["P"]
    for fam in fam_prim:
        fam["relation_to_best"] = relation_to(fam["P"], P_best, baseline)
    res["families_primary"] = [{k: v for k, v in f.items() if k != "i"} for f in fam_prim]
    res["families_extended"] = [{k: v for k, v in f.items() if k != "i"} for f in fam_ext]
    res["max_power_primary"] = fam_prim[0]["power"]

    # weighted scan (diagnostic)
    ls_w = LombScargle(ts, ys, dy=dy, nterms=2)
    power_w = ls_w.power(freq, method="fastchi2", assume_regular_frequency=True)
    pkw, _ = find_peaks(power_w, distance=3)
    pkw = pkw[prim[pkw]]
    fam_w = peak_families(periods, power_w, pkw[np.argsort(power_w[pkw])[::-1]][:300], n_max=6)
    res["families_weighted"] = [{"P": f["P"], "power": f["power"]} for f in fam_w]

    # --- 5) candidate periods ---------------------------------------------------------------------
    cands = []

    def add(P, source):
        if not (P_MIN <= P <= P_MAX_EXT):
            return
        for c in cands:
            if abs(c["P_seed"] - P) <= 0.003 * P:
                if source not in c["sources"]:
                    c["sources"].append(source)
                return
        cands.append({"P_seed": float(P), "sources": [source]})

    for rank, fam in enumerate(fam_prim[:8]):
        if fam["power"] >= 0.5 * fam_prim[0]["power"] or rank < 4:
            add(fam["P"], f"family#{rank + 1}")
    for fam in fam_ext[:2]:
        add(fam["P"], "extended-range family")
    for fam in fam_w[:2]:
        add(fam["P"], "weighted-scan family")
    for base_rank in (0, 1):
        if base_rank < len(fam_prim):
            Pb = fam_prim[base_rank]["P"]
            add(Pb / 2.0, f"family#{base_rank + 1}/2")
            add(Pb * 2.0, f"2*family#{base_rank + 1}")
    pub = PUBLISHED.get(str(num), {})
    for key in ("P_c", "P_av", "P_fig", "P_fig_periodogram", "P_fig_twin", "P_text", "P_tess_text", "P_lit"):
        if key in pub:
            add(float(pub[key]), f"manuscript:{key}")

    for c in cands:
        c["P"], c["f"], c["power"] = refine(c["P_seed"])
    merged = []
    for c in sorted(cands, key=lambda x: -x["power"]):
        dP = c["P"] ** 2 / 24.0 / baseline
        dup = next((m for m in merged if abs(m["P"] - c["P"]) <= 3 * dP), None)
        if dup:
            dup["sources"] += [s for s in c["sources"] if s not in dup["sources"]]
        else:
            merged.append(c)
    cands = merged
    for c in cands:
        c["dP_resolution"] = float(c["P"] ** 2 / 24.0 / baseline)
        c["relation_to_best"] = relation_to(c["P"], P_best, baseline)
        c["fold"] = fold_stats(ts, ys, c["P"])
        c["power_weighted"] = float(np.max(ls_power(ts, ys, window_grid(c["f"], baseline, 3.0), dy=dy)))

    # --- 6) night-block bootstrap --------------------------------------------------------------------
    rng = np.random.default_rng(num)
    t_abs = ts + pipe.time_zero
    night_key = np.array([f"{s}|{int(np.floor(x))}" for s, x in zip(sheet_kept, t_abs)])
    _, inv = np.unique(night_key, return_inverse=True)
    order_inv = np.argsort(inv, kind="stable")
    bounds = np.searchsorted(inv[order_inv], np.arange(inv.max() + 2))
    groups = [order_inv[bounds[g]:bounds[g + 1]] for g in range(inv.max() + 1)]
    grids = [window_grid(c["f"], baseline) for c in cands]
    boot_best = np.zeros((N_BOOT, len(cands)))
    boot_loc = np.zeros((N_BOOT, len(cands)))
    for b in range(N_BOOT):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[g] for g in pick])
        tb, yb = ts[idx], ys[idx]
        for j, g in enumerate(grids):
            p = ls_power(tb, yb, g)
            k = int(np.argmax(p))
            boot_best[b, j] = p[k]
            boot_loc[b, j] = 24.0 / g[k]
    winners = np.argmax(boot_best, axis=1)
    for j, c in enumerate(cands):
        loc = boot_loc[:, j]
        c["boot_win_fraction"] = float(np.mean(winners == j))
        c["boot_sigma_P"] = float(np.std(loc, ddof=1))
        c["boot_median_P"] = float(np.median(loc))
        in_tooth = np.abs(loc - c["P"]) <= 1.5 * c["dP_resolution"]
        c["boot_same_tooth_fraction"] = float(np.mean(in_tooth))
        c["boot_sigma_P_tooth"] = float(np.std(loc[in_tooth], ddof=1)) if np.sum(in_tooth) > 2 else None
        c["boot_mean_power"] = float(np.mean(boot_best[:, j]))
    # win fractions when harmonic partners (P, 2P, P/2) are pooled into one solution family
    fam_id = list(range(len(cands)))
    for a in range(len(cands)):
        for b2 in range(a):
            Pa, Pb = cands[a]["P"], cands[b2]["P"]
            if (abs(2 * Pa - Pb) <= 3 * cands[b2]["dP_resolution"]
                    or abs(Pa - 2 * Pb) <= 3 * cands[a]["dP_resolution"]):
                fam_id[a] = fam_id[b2]
                break
    for j, c in enumerate(cands):
        c["harmonic_family"] = int(fam_id[j])
        c["boot_family_win_fraction"] = float(np.mean([fam_id[w] == fam_id[j] for w in winners]))

    # --- 7) independent checks ------------------------------------------------------------------------
    stn_of = np.array([s[:3] for s in sheet_kept])
    grp_of = np.array([GROUP_OF.get(s, "other") for s in stn_of])

    def cand_powers(tb, yb, dyb=None):
        base_b = max(float(np.max(tb) - np.min(tb)), 1.0)
        vals = []
        for c in cands:
            g = window_grid(c["f"], base_b, half_width_res=3.0, spp=20)
            vals.append(float(np.max(ls_power(tb, yb, g, dy=dyb))))
        return vals, base_b

    def summarise(vals):
        order = np.argsort(vals)[::-1]
        return {"power": vals, "preferred_P": cands[int(order[0])]["P"],
                "ratio_second_to_first": float(vals[order[1]] / vals[order[0]]) if len(vals) > 1 and vals[order[0]] > 0 else None}

    subsets = {}
    for gname in ["americas", "europe_africa", "asia_australia", "space", "other"]:
        m = grp_of == gname
        if np.sum(m) >= 150:
            subsets[f"group:{gname}"] = m
    if np.sum(grp_of == "space") >= 100:
        subsets["all_but_TESS"] = grp_of != "space"
    for ds in kept:
        m = sheet_kept == ds
        if np.sum(m) >= 100:
            subsets[f"dataset:{ds}"] = m
        if len(kept) > 2:
            subsets[f"without:{ds}"] = sheet_kept != ds
    sub_res = {}
    for name, m in subsets.items():
        vals, base_b = cand_powers(ts[m], ys[m])
        sub_res[name] = {"n": int(np.sum(m)), "baseline_days": base_b, **summarise(vals)}
    res["subsets"] = sub_res
    res["weighted_all"] = summarise([c["power_weighted"] for c in cands])

    m_tess = stn_of == "C57"
    if np.sum(m_tess) >= 100:
        tb, yb = ts[m_tess], ys[m_tess]
        base_b = float(np.max(tb) - np.min(tb))
        ls_t = LombScargle(tb, yb, nterms=2)
        f_t = ls_t.autofrequency(samples_per_peak=20, minimum_frequency=24.0 / P_MAX_PRIMARY, maximum_frequency=fmax)
        p_t = ls_t.power(f_t, method="fastchi2", assume_regular_frequency=True)
        pk_t, _ = find_peaks(p_t, distance=5)
        pk_t = pk_t[np.argsort(p_t[pk_t])[::-1]][:8]
        spans = []
        tt = np.sort(tb)
        cuts = np.where(np.diff(tt) > 30)[0]
        starts = np.r_[0, cuts + 1]
        ends = np.r_[cuts, len(tt) - 1]
        for s0, e0 in zip(starts, ends):
            spans.append([float(tt[s0] + pipe.time_zero), float(tt[e0] + pipe.time_zero), int(e0 - s0 + 1)])
        res["tess_only"] = {"n": int(np.sum(m_tess)), "baseline_days": base_b, "visits": spans,
                            "peaks": [{"P": float(24.0 / f_t[i]), "power": float(p_t[i])} for i in pk_t]}

    # --- 8) figures / diagnostics from the repository code ------------------------------------------
    order_c = sorted(range(len(cands)), key=lambda j: (-cands[j]["boot_win_fraction"], -cands[j]["power"]))
    P_top = cands[order_c[0]]["P"]
    try:
        pipe_freq_hi = np.arange(24.0 / (P_top * 1.1), 24.0 / (P_top * 0.9), 1.0 / (20 * baseline))
        pipe_pow_hi = ls_power(ts, ys, pipe_freq_hi)
        g = pipe.plot_peak_zoom_gaussian(pipe_freq_hi, pipe_pow_hi, P_top, wide_frac=0.10, n_resolution=3.0,
                                         save_fig=True, save_figures=out)
        res["gaussian_top"] = g
    except Exception as exc:
        res["gaussian_top_error"] = f"{type(exc).__name__}: {exc}"
    plt.close("all")
    try:
        sw = pipe.spectral_window(candidates_hours=[cands[j]["P"] for j in order_c[:4]], samples_per_peak=5,
                                  save_fig=True, save_figures=out)
        res["spectral_window"] = {"window_peaks": sw["window_peaks"], "relations": sw["relations"]}
    except Exception as exc:
        res["spectral_window_error"] = f"{type(exc).__name__}: {exc}"
    plt.close("all")
    fold_list = list(order_c[:3])
    for j, c in enumerate(cands):
        if any(s.startswith("manuscript:P_c") for s in c["sources"]) and j not in fold_list:
            fold_list.append(j)
    for j in fold_list:
        try:
            li = pipe.visualize_ls(np.array([cands[j]["f"]]), np.array([0]), 0, save_fig=True, save_figures=out,
                                   bin_deg=10.0, amp_sigma_clip=3.0, min_points_per_bin=5)
            cands[j]["fold_figure"] = li.get("file")
            cands[j]["amplitude_binned"] = li.get("amplitude_mag")
            cands[j]["amplitude_binned_err"] = li.get("amplitude_err_mag")
        except Exception as exc:
            cands[j]["fold_figure_error"] = f"{type(exc).__name__}: {exc}"
        plt.close("all")

    ds_pow = {}
    for ds in kept:
        m = sheet_kept == ds
        if np.sum(m) < 30:
            continue
        fc = 24.0 / P_top
        g = np.linspace(fc * 0.99, fc * 1.01, 4001)
        ds_pow[ds] = float(np.max(ls_power(ts[m], ys[m], g)))
    res["dataset_power_at_top"] = ds_pow

    res["candidates"] = [cands[j] for j in order_c]
    res["suggested_P"] = P_top

    nblk = 60000
    order_p = np.argsort(periods)
    ps, pw, pww = periods[order_p], power[order_p], power_w[order_p]
    edges = np.linspace(0, len(ps), nblk + 1).astype(int)
    keep_i = [a + int(np.argmax(pw[a:b])) for a, b in zip(edges[:-1], edges[1:]) if b > a]
    np.savez_compressed(os.path.join(out_dir, f"{num}_arrays.npz"), period=ps[keep_i],
                        power=pw[keep_i].astype(np.float32), power_weighted=pww[keep_i].astype(np.float32),
                        t=ts, y=ys, sheet=sheet_kept.astype(str), time_zero=pipe.time_zero, threshold=threshold)
    res["runtime_s"] = round(time.time() - t_start, 1)
    return res


def run_one(num: int) -> str:
    out_dir = os.path.join(RES_DIR, str(num))
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, f"{num}_log.txt")
    with open(log_path, "w", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        warnings.simplefilter("ignore")
        try:
            res = analyse(num)
        except Exception as exc:
            traceback.print_exc()
            res = {"asteroid": num, "error": f"{type(exc).__name__}: {exc}"}
    with open(os.path.join(out_dir, f"{num}_result.json"), "w", encoding="utf-8") as fh:
        json.dump(_jsonable(res), fh, indent=1)
    if res.get("error"):
        return f"[{num}] ERROR {res['error']}"
    return f"[{num}] P_top={res.get('suggested_P'):.4f} h in {res.get('runtime_s')} s"


def main(argv):
    workers = 6
    if "--workers" in argv:
        k = argv.index("--workers")
        workers = int(argv[k + 1])
        argv = argv[:k] + argv[k + 2:]
    nums = [int(a) for a in argv if not a.startswith("--")]
    if not nums:
        nums = sorted(int(k) for k in PUBLISHED if not k.startswith("_"))
    if workers <= 1 or len(nums) == 1:
        for n in nums:
            print(run_one(n), flush=True)
        return
    with mp.get_context("spawn").Pool(workers) as pool:
        for msg in pool.imap_unordered(run_one, nums):
            print(msg, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
