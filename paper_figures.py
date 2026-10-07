"""
Regenerate the figures of the paper
"Combination of multi-observatory photometry to determine asteroid rotation periods"
after the first review round, plus the extra diagnostic figures promised in the
response to the reviewers.

Run (from the repository folder):
    python paper_figures.py                 # all asteroids
    python paper_figures.py 4303 3173       # selected asteroids
    python paper_figures.py 4303 --skip-opposition   # skip the Horizons-based per-opposition plots

Outputs (all under ./paper_figures/):
    paper/<N>/      figures that replace the ones in the manuscript (same file names as main.tex)
    response/<N>/   diagnostics for the reviewer response only
    summary.json / summary.csv   numbers needed for captions and tables (epochs, N, H/G1/G2, peaks)

Per-asteroid choices (observatory lists, reference sheet, periods, Fourier order) live in CONFIG.
The observatory lists reproduce the legends of the submitted figures.
"""
from __future__ import annotations

import json
import os
import sys
import time
import pickle
import traceback

import matplotlib

matplotlib.use("Agg")  # headless; plt.show() inside the pipeline becomes a no-op
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.time import Time
from astropy.timeseries import LombScargle

from Asteroid_pickle import DatasetGenerator
from asteroid_ls import AsteroidLSPipeline
import lc_amplitude

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR_1 = os.path.join(REPO_DIR, "..", "Asteroid_data", "12asteroidudatiicaruspubl")
DATA_DIR_2 = os.path.join(REPO_DIR, "..", "Asteroid_data", "12asteroidudatiicaruspubl2")
OUT_DIR = os.path.join(REPO_DIR, "paper_figures")

# Run options (set from the command line in main):
#   bias       : "band" (one band offset per dataset) or "dephocus" (per-measurement DePhOCUS offsets, V band)
#   band_phase : fit G1, G2 separately for every band with enough data (wavelength dependent phase curves)
#   tag        : outputs go to paper_figures/runs/<tag>/ (paper figures, response, summaries, ls_cache)
#   lc_nterms  : Fourier order of the model drawn in the light-curve figures and used for the amplitude
OPTIONS = dict(bias="band", band_phase=False, tag=None, lc_nterms=4, extras=True, amp_source="fourier")

# Lomb-Scargle search interval used for the combined datasets (Section 3.4 of the paper).
P_MIN_HOURS = 0.5
P_MAX_HOURS = 240.0
SAMPLES_PER_PEAK = 15
PEAK_SIGMA = 5          # "5 sigma above the mean power" (Section 3.2); override per asteroid with peak_height
PEAK_DISTANCE = 1000    # minimum separation of detected peaks in grid points
PEAK_MERGE_FRAC = 0.07  # peaks within 7 % in period of a stronger peak are not labelled (readability)
PEAK_MAX = 6            # at most this many (strongest) peaks are labelled
LS_CACHE_DIR = os.path.join(OUT_DIR, "ls_cache")   # periodograms are cached here (npz) to speed up re-plotting

# ----------------------------------------------------------------------------
# Per-asteroid configuration.
#   sheets     : observatory sheets in the workbook, in legend order of the submitted figure
#   reference  : reference observatory for the HG1G2 fit (paper: T08o unless stated otherwise)
#   periods    : rotation periods (hours) at which folded light curves are produced
#   nterms_lc  : Fourier order for the folded light-curve figure(s)
#   single_obs : sheet for single-observatory figures (Figs. 2-3 of the paper)
#   window     : candidate periods (hours) for the spectral-window alias check (response only)
#   opposition : make per-opposition light curves (response only)
#   peak_height: manual power threshold for the peaks marked in the periodogram figure.
#                None -> 5 sigma above the background (used when no peak dominates, e.g. 2968).
#                Where a few peaks dominate, the threshold is raised so that only the adopted
#                period, its half and the strongest aliases are labelled; the daily aliases of
#                the main peak (same "family") are discussed in the text instead of being labelled.
# ----------------------------------------------------------------------------
CONFIG = {
    2607: dict(
        workbook=os.path.join(DATA_DIR_1, "2607-excelBF.xlsx"),
        sheets=["G96G", "G96V", "T05c", "T05o", "T08o", "F51w", "G45r", "D29R", "703G", "703V"],
        reference="T08o", periods=[2.936], nterms_lc=[2],
        peak_height=0.10,   # 2.94 h and 1.47 h (w = 0.13-0.14 after the light-time fix); their aliases (2.62/1.31 h at w = 0.07) are not labelled
        phase_fit_comparison=True, summary_2x2=True,
    ),
    2968: dict(
        workbook=os.path.join(DATA_DIR_1, "2968-excelBF.xlsx"),
        sheets=["G45G", "G45r", "G96G", "F51w", "T08c", "T08o", "703G", "703V", "T05c", "T05o"],
        reference="T08o", periods=[4.560], nterms_lc=[2],
    ),
    2971: dict(
        workbook=os.path.join(DATA_DIR_1, "2971-excelBF.xlsx"),
        sheets=["703G", "703V", "M22o", "T05c", "T05o", "G96G", "G45r", "F51w", "I41r", "T08o1", "W68o"],
        reference="T08o1", periods=[4.491], nterms_lc=[2],
        peak_height=0.07,   # 4.49 h and 2.25 h (w = 0.08); daily aliases 4.95/2.48/4.11 h (w <= 0.06) not labelled
    ),
    3081: dict(
        workbook=os.path.join(DATA_DIR_1, "3081-excelBD.xlsx"),   # only a BD workbook exists for 3081
        sheets=["703G", "703V", "T05c", "T05o", "G96G", "G96V", "D29R", "T08o", "W68o", "M22o"],
        reference="T08o", periods=[8.007], nterms_lc=[4], nterms_ps=[2],
        peak_height=0.30,   # 8.01 h and 4.00 h (w = 0.40); daily aliases (w <= 0.19) and 24/12 h (w = 0.11) not labelled
        window=[8.007, 4.00, 24.0],
    ),
    3173: dict(
        workbook=os.path.join(DATA_DIR_2, "3173-excelBF.xlsx"),
        sheets=["T05c", "T05o", "703G", "703V", "T08o", "D29R", "G45r", "G96G", "W68o"],
        reference="T08o", periods=[45.983], nterms_lc=[2],
        peak_height=0.15,   # keeps 45.98/22.99 h (w = 0.28), the 49.9/24.9 h alternative (w = 0.19) and 12.2 h (w = 0.18)
        window=[45.983, 49.98, 24.98, 22.99], opposition=dict(periods=[45.983, 49.98], min_period_hours=60.0),
    ),
    3473: dict(
        workbook=os.path.join(DATA_DIR_2, "3473-excelBF.xlsx"),
        # I41r removed: its measurements form a separate branch 0.5 mag below the other datasets
        sheets=["C57G", "T08o", "703V", "703G", "G96V", "T05o", "T05c", "W68o", "D29R", "G96G", "G45r", "M22o", "691V"],
        reference="T08o", periods=[9.074], nterms_lc=[2],
        peak_height=0.05,   # 9.07 h and 4.54 h (w = 0.18); daily aliases 7.63/3.81 h (w = 0.04) not labelled
    ),
    3716: dict(
        workbook=os.path.join(DATA_DIR_2, "3716-excelBF.xlsx"),
        sheets=["P07G", "G45r", "D29R", "W68o", "T08o", "703G", "703V", "M22o", "G96G", "G96V", "F52w", "T05c", "T05o"],
        reference="T08o", periods=[10.474], nterms_lc=[2],
        peak_height=0.15,   # 10.47/5.24 h (w = 0.21) and the daily alias 13.41/6.70 h (w = 0.17); 8.59/4.30 h (w = 0.13) not labelled
        window=[10.474, 13.407, 5.24, 6.70],
    ),
    4303: dict(
        workbook=os.path.join(DATA_DIR_2, "4303-excelBF.xlsx"),
        sheets=["P07G", "D29R", "703G", "703V", "M22o", "C57G", "W68o", "G96G", "G96V", "G45r", "F51w", "T08o", "T05c", "T05o"],
        reference="T08o", periods=[6.136], nterms_lc=[2],
        peak_height=0.20,   # 6.14 h and 3.07 h (w = 0.55-0.57); all daily aliases have w <= 0.05
        single_obs="T08o", gaussian_peak=True,
        peak_height_single=0.5,   # manual threshold used for the T08o periodogram in the submitted Fig. 3
        opposition=dict(periods=[6.136], min_period_hours=55.0),
    ),
    # --- asteroids with known rotation periods, used to test the method (Table 3) ---------
    # Their figures go to the response folder only (paper=False).
    1951: dict(
        workbook=os.path.join(DATA_DIR_1, "1951-excelBF.xlsx"), paper=False,
        sheets=["703G", "703V", "C57G", "H45R", "I41g", "I41r", "M22o", "T05c", "T05o", "T05w", "T08c", "T08o", "W68o"],
        reference="T08o", periods=[5.300], nterms_lc=[2],
    ),
    1963: dict(
        workbook=os.path.join(DATA_DIR_1, "1963-excelBF.xlsx"), paper=False,
        # C57G2 sheet of this workbook has no phase-angle column and cannot be used;
        # I41g (69) and M22c (68) have fewer than 70 measurements in the workbook and are not used
        sheets=["689V", "703G", "703V", "I41r", "M22o", "T05c", "T05o", "T05w", "T08o", "W68c", "W68o"],
        reference="T08o", periods=[18.164], nterms_lc=[2],
    ),
    2134: dict(
        workbook=os.path.join(DATA_DIR_1, "2134-excelBF.xlsx"), paper=False,
        sheets=["703G", "703V", "C57G", "G45r", "I41g", "I41r", "T05c", "T05o", "T08o", "W68o"],
        reference="T08o", periods=[4.114], nterms_lc=[2],
    ),
    2150: dict(
        workbook=os.path.join(DATA_DIR_1, "2150-excelBF.xlsx"), paper=False,
        sheets=["703G", "703V", "C57G", "G45r", "I41r", "M22o", "T05c", "T05o", "T08c", "T08o", "W68o"],
        reference="T08o", periods=[6.125], nterms_lc=[2],
    ),
}


def _dir(*parts) -> str:
    path = os.path.join(*parts)
    os.makedirs(path, exist_ok=True)
    return path + os.sep


def _jd_to_utc(jd: float) -> str:
    return Time(float(jd), format="jd").utc.iso.split(".")[0]


def _select_peak_index(frequency, power, period_hours, tol_frac=0.005):
    """Index of the highest periodogram point within +/- tol_frac of the requested period."""
    periods = 24.0 / np.asarray(frequency, dtype=float)
    mask = np.abs(periods - period_hours) <= tol_frac * period_hours
    if not np.any(mask):
        raise ValueError(f"No grid point within {tol_frac*100:.1f}% of P={period_hours} h")
    idx = np.where(mask)[0]
    return int(idx[np.argmax(np.asarray(power)[idx])])


def _peak_table(frequency, power, peaks):
    return [
        {"period_hours": float(24.0 / frequency[i]), "power": float(power[i])}
        for i in sorted(peaks, key=lambda i: -power[i])
    ]


def _amplitudes(pipe, f_best):
    """Amplitudes at frequency f_best (cycles/day): n-term Fourier fit of the combined fold (night-block
    bootstrap error) and the per-apparition template amplitude (lc_amplitude.py)."""
    t_parts, y_parts, s_parts = [], [], []
    for sheet, (H_s, t_s, _) in pipe.all_sheets.items():
        H_s, t_s = np.asarray(H_s, float), np.asarray(t_s, float)
        ok = np.isfinite(H_s) & np.isfinite(t_s)
        t_parts.append(t_s[ok]); y_parts.append(H_s[ok]); s_parts.append(np.full(ok.sum(), sheet))
    t = np.concatenate(t_parts) - float(pipe.time_zero)
    y = np.concatenate(y_parts)
    s = np.concatenate(s_parts)
    P = 24.0 / f_best
    four = lc_amplitude.fourier_amplitude(t, y, P, nterms=OPTIONS["lc_nterms"], sheet=s, n_boot=300)
    tpl = lc_amplitude.template_amplitude(t, y, s, P, nterms=4)
    out = {"fourier_n": OPTIONS["lc_nterms"], "fourier_amp": four["amplitude"], "fourier_err": four["amplitude_err"],
           "fourier_n_clipped": four["n_clipped"]}
    if tpl is not None:
        out.update(template_amp=tpl["amplitude"], template_err=tpl["amplitude_err"],
                   template_n_app=tpl["n_apparitions"],
                   template_apparitions=[{k: r[k] for k in ("n", "g", "dphi", "t_mid")} for r in tpl["apparitions"]])
    return out


def _light_curves(pipe, frequency, power, periods, nterms, out_paper, summary, tag="", ts=None, ys=None):
    """Folded light curves at the requested periods with the n = lc_nterms Fourier model (all points shown).
    The frequency is that of the n = 2 periodogram peak; the amplitude of the model and the per-apparition
    template amplitude are stored in the summary."""
    for P in periods:
        idx = _select_peak_index(frequency, power, P)
        f_best = float(frequency[idx])
        amps = _amplitudes(pipe, f_best)
        if ts is not None:
            pipe.nterms = OPTIONS["lc_nterms"]
            pipe.ls = LombScargle(ts, ys, nterms=OPTIONS["lc_nterms"])
        if OPTIONS["amp_source"] == "template" and "template_amp" in amps:
            amp_box = (amps["template_amp"], amps["template_err"], "per-apparition template")
        else:
            amp_box = (amps["fourier_amp"], amps["fourier_err"], f"n={OPTIONS['lc_nterms']} model")
        info = pipe.visualize_ls(frequency, np.array([idx]), 0, save_fig=True, save_figures=out_paper,
                                 bin_deg=10.0, amp_sigma_clip=3.0, min_points_per_bin=5,
                                 show_bins=False, amplitude_box=amp_box)
        plt.close("all")
        info.update(amps)
        summary[f"light_curve{tag}_P={P}_n={pipe.nterms}"] = info


def _ls_cached(pipe, ts, ys, nterms, cache_name):
    """Lomb-Scargle over the full search interval, cached as npz (the grid has ~5 M points)."""
    os.makedirs(LS_CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(LS_CACHE_DIR, cache_name)
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        frequency = np.asarray(data["frequency"], dtype=float)
        power = np.asarray(data["power"], dtype=float)
        if int(data["n_points"]) == len(ys) and abs(float(data["y_sum"]) - float(np.sum(ys))) < 1e-6:
            pipe.nterms = nterms
            pipe.ls = LombScargle(ts, ys, nterms=nterms)   # model() needs the fitted object
            print("Loaded cached periodogram:", cache_path)
            return frequency, power
        print("Cache does not match the current dataset; recomputing.")
    frequency, power, _ = pipe.LS_initiate(ts, ys, nterms=nterms, maxf=24.0 / P_MIN_HOURS,
                                           minf=24.0 / P_MAX_HOURS, samples_per_peak=SAMPLES_PER_PEAK,
                                           save_bool=False)
    plt.close("all")
    np.savez(cache_path, frequency=frequency, power=power.astype(np.float32),
             n_points=len(ys), y_sum=float(np.sum(ys)))
    return frequency, power


def run_asteroid(num: int, cfg: dict, skip_opposition: bool = False) -> dict:
    t_start = time.time()
    out_resp = _dir(OUT_DIR, "response", str(num))
    out_paper = _dir(OUT_DIR, "paper", str(num)) if cfg.get("paper", True) else out_resp
    summary: dict = {"asteroid": num, "workbook": os.path.abspath(cfg["workbook"]),
                     "sheets": cfg["sheets"], "reference": cfg["reference"]}

    workbook_dir, workbook_name = os.path.split(cfg["workbook"])
    sheets = cfg["sheets"] + cfg.get("extra_sheets", [])
    gen = DatasetGenerator(workbook_dir + os.sep, workbook_name, num, sheets, base_dir=REPO_DIR,
                           bias_mode=OPTIONS["bias"])
    summary["options"] = dict(OPTIONS)
    summary["sheets"] = sheets

    # --- 1) reference phase curve and cross-observatory shift (Section 3.1) -----------------
    H, G1, G2, H_err, G1_err, G2_err = gen.reference_obs_comp(cfg["reference"], method="HG1G2", return_errors=True)
    plt.close("all")
    summary["reference_fit"] = {"H": H, "G1": G1, "G2": G2, "H_err": H_err, "G1_err": G1_err, "G2_err": G2_err}
    if OPTIONS["band_phase"]:
        summary["band_slopes"] = gen.fit_band_slopes(H, G1, G2, cfg["reference"])

    dict_sheets = gen.all_obs_comb(H, G1_val=G1, G2_val=G2, method="HG1G2",
                                   save_figure=True, save_figures=out_resp,
                                   save_file=True, save_path=out_resp)
    plt.close("all")
    counts = {k: int(len(np.asarray(v[0], dtype=float))) for k, v in dict_sheets.items()}
    summary["n_per_observatory"] = counts
    summary["n_total"] = int(sum(counts.values()))
    summary["delta_H"] = gen.delta_H
    summary["bias_log"] = gen.bias_log
    dH = np.array([v["dH"] for v in gen.delta_H.values()])
    nH = np.array([v["n"] for v in gen.delta_H.values()], float)
    summary["H_levels"] = {"H_ref": H, "H_weighted_mean": float(H + np.sum(nH * dH) / np.sum(nH)),
                           "dH_rms": float(np.sqrt(np.mean(dH ** 2))), "dH_std": float(np.std(dH, ddof=1)),
                           "dH_min": float(dH.min()), "dH_max": float(dH.max())}

    # --- 2) Lomb-Scargle on the combined dataset ------------------------------------------
    pipe = AsteroidLSPipeline(dict_sheets, num, base_dir=REPO_DIR)
    ts, ys = pipe.reading_data()
    plt.close("all")
    summary["n_after_outlier_removal"] = int(len(ys))
    summary["time_zero_jd"] = float(pipe.time_zero)
    summary["time_zero_utc"] = _jd_to_utc(pipe.time_zero)
    summary["baseline_days"] = float(np.max(pipe.t) - np.min(pipe.t))
    summary["date_range_utc"] = [_jd_to_utc(np.min(pipe.t)), _jd_to_utc(np.max(pipe.t))]

    ls_cache = {}
    for nterms in sorted(set(cfg.get("nterms_ps", [2]))):
        frequency, power = _ls_cached(pipe, ts, ys, nterms, f"{num}_LS_n={nterms}.npz")
        ls_cache[nterms] = (frequency, power)

        # Periodogram figure for the paper (full search range, log period axis) ...
        if nterms in cfg.get("nterms_ps", [2]):
            peaks = pipe.find_peaks(frequency, power, height=cfg.get("peak_height"), distance=PEAK_DISTANCE,
                                    number_of_sigma=PEAK_SIGMA, save_fig=True, save_figures=out_paper,
                                    xscale="log", merge_frac=PEAK_MERGE_FRAC, max_peaks=PEAK_MAX)
            plt.close("all")
            summary[f"periodogram_n={nterms}"] = {
                "range_hours": [P_MIN_HOURS, P_MAX_HOURS], "threshold": pipe.last_peak_threshold,
                "peaks": _peak_table(frequency, power, peaks),
            }
            # ... and a linear 0.5-50 h view for the response (same range as the submitted figures).
            peaks50 = pipe.find_peaks(frequency, power, height=cfg.get("peak_height"), distance=PEAK_DISTANCE,
                                      number_of_sigma=PEAK_SIGMA, save_fig=True, save_figures=out_resp,
                                      period_range=(P_MIN_HOURS, 50.0), filename_suffix="_0.5-50h",
                                      merge_frac=PEAK_MERGE_FRAC, max_peaks=PEAK_MAX)
            plt.close("all")
            summary[f"periodogram_n={nterms}_0.5-50h"] = {
                "threshold": pipe.last_peak_threshold, "peaks": _peak_table(frequency, power, peaks50),
            }

        # Folded light curves for the paper (n = lc_nterms model at the n = 2 peak frequency).
        if nterms == 2:
            _light_curves(pipe, frequency, power, cfg["periods"], OPTIONS["lc_nterms"], out_paper, summary,
                          ts=ts, ys=ys)
            pipe.nterms = 2
            pipe.ls = LombScargle(ts, ys, nterms=2)

    # --- paper Fig. 1: pipeline panels with the step 2 outlier limits (3 sigma reference fit, 1.8 sigma combination)
    if cfg.get("summary_2x2"):
        gen.plot_phase_curve_summary_2x2(reference_sheet=cfg["reference"], comparison_sheet="G96V",
                                         outlier_param=1.8, ref_outlier_param=3.0, save_figures=True,
                                         save_dir=out_paper.rstrip(os.sep))
        plt.close("all")

    # --- paper Fig. 2: single-observatory light curve (4303 / T08o)
    if cfg.get("single_obs"):
        sheet = cfg["single_obs"]
        pipe1 = AsteroidLSPipeline({sheet: dict_sheets[sheet]}, f"{num}_{sheet}", base_dir=REPO_DIR)
        ts1, ys1 = pipe1.reading_data()
        plt.close("all")
        f1, p1 = _ls_cached(pipe1, ts1, ys1, 2, f"{num}_{sheet}_LS_n=2.npz")
        pk1 = pipe1.find_peaks(f1, p1, height=cfg.get("peak_height_single"), distance=PEAK_DISTANCE,
                               number_of_sigma=PEAK_SIGMA, save_fig=True, save_figures=out_resp,
                               xscale="log", merge_frac=PEAK_MERGE_FRAC, max_peaks=PEAK_MAX)
        plt.close("all")
        s1 = {"n_points": int(len(ys1)), "time_zero_jd": float(pipe1.time_zero),
              "time_zero_utc": _jd_to_utc(pipe1.time_zero), "threshold": pipe1.last_peak_threshold,
              "peaks": _peak_table(f1, p1, pk1)}
        _light_curves(pipe1, f1, p1, cfg["periods"], OPTIONS["lc_nterms"], out_paper, s1, tag="", ts=ts1, ys=ys1)
        if cfg.get("gaussian_peak"):   # paper Fig. 3: power spectrum around the peak with the fitted Gaussian
            pipe1.nterms = 2
            pipe1.ls = LombScargle(ts1, ys1, nterms=2)
            s1["gaussian"] = pipe1.plot_peak_zoom_gaussian(f1, p1, cfg["periods"][0], wide_frac=0.10,
                                                            n_resolution=3.0, save_fig=True, save_figures=out_paper)
            plt.close("all")
        summary[f"single_observatory_{sheet}"] = s1

    # --- 3) response-only diagnostics -----------------------------------------------------
    if not OPTIONS["extras"]:
        summary["runtime_s"] = round(time.time() - t_start, 1)
        return summary
    pipe.nterms = min(ls_cache)
    # Width of the adopted periodogram peak (Gaussian fit) as a resolution indicator.
    try:
        f2, p2 = ls_cache[min(ls_cache)]
        g = pipe.plot_peak_zoom_gaussian(f2, p2, cfg["periods"][0], wide_frac=0.10, n_resolution=3.0,
                                         save_fig=True, save_figures=out_resp)
        plt.close("all")
        summary["combined_peak_gaussian"] = g
    except Exception as exc:
        summary["combined_peak_gaussian_error"] = f"{type(exc).__name__}: {exc}"

    if cfg.get("window"):
        info = pipe.spectral_window(candidates_hours=cfg["window"], save_fig=True, save_figures=out_resp)
        plt.close("all")
        summary["spectral_window"] = {"window_peaks": info["window_peaks"], "relations": info["relations"],
                                      "file": info["file"]}

    if cfg.get("phase_fit_comparison"):
        df_cmp = gen.compare_phase_curve_fit_strategies(reference_sheet=cfg["reference"], outlier_param=1.8,
                                                        save_figures=True, save_dir=out_resp)
        plt.close("all")
        df_cmp.to_csv(os.path.join(out_resp, f"{num}_phase_fit_free_vs_fixed.csv"), index=False)
        summary["phase_fit_comparison_csv"] = os.path.join(out_resp, f"{num}_phase_fit_free_vs_fixed.csv")

    # Per-opposition light curves (Reviewer 2), Horizons-based opposition epochs.
    if cfg.get("opposition") and not skip_opposition:
        opp_cfg = cfg["opposition"]
        try:
            pipe_o = AsteroidLSPipeline(dict_sheets, num, base_dir=REPO_DIR)
            opp_data = pipe_o.reading_data(split_oppositions=True, opposition_method="horizons",
                                           min_obs_per_opposition=50)
            plt.close("all")
            opp_ls = pipe_o.LS_initiate_per_opposition(opposition_data=opp_data, nterms=2, peak_idx=0,
                                                       minf=24.0 / float(opp_cfg.get("min_period_hours", 55.0)),
                                                       double_period_tol=0.12, calibrate_each_opposition=False)
            plt.close("all")
            pipe_o.visualize_ls_per_opposition(opp_ls, save_fig=True, save_figures=out_resp)
            plt.close("all")
            for P in opp_cfg["periods"]:
                pipe_o.visualize_time_series_all_oppositions(ls_results=opp_ls, best_period_hours=P,
                                                             use_calibrated=False, align_phase_offsets=True,
                                                             save_fig=True, save_figures=out_resp)
                plt.close("all")
            summary["oppositions"] = {
                str(g): {"date_utc": r.get("opposition_date_utc"), "n_points": int(len(r["t"])),
                         "selected_period_hours": r.get("selected_period_hours"),
                         "primary_period_hours": r.get("primary_period_hours"),
                         "selection_reason": r.get("selection_reason")}
                for g, r in opp_ls.items()
            }
        except Exception as exc:  # Horizons/network problems should not kill the whole run
            summary["oppositions_error"] = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()

    summary["runtime_s"] = round(time.time() - t_start, 1)
    return summary


def _flatten_for_csv(summary: dict) -> dict:
    row = {"asteroid": summary["asteroid"], "reference": summary["reference"],
           "n_total": summary.get("n_total"), "n_after_outlier_removal": summary.get("n_after_outlier_removal"),
           "time_zero_jd": summary.get("time_zero_jd"), "time_zero_utc": summary.get("time_zero_utc"),
           "baseline_days": summary.get("baseline_days"),
           "observatories": ", ".join(f"{k}({v})" for k, v in summary.get("n_per_observatory", {}).items())}
    ref = summary.get("reference_fit", {})
    for k in ("H", "G1", "G2", "H_err", "G1_err", "G2_err"):
        row[k] = ref.get(k)
    lcs = [v for k, v in summary.items() if k.startswith("light_curve_P=")]
    if lcs:
        lc = lcs[0]
        row["P_light_curve_hours"] = lc.get("period_hours")
        row["caption_epoch"] = lc.get("caption_epoch")
        row["amplitude_mag"] = lc.get("amplitude_mag")
        row["amplitude_err_mag"] = lc.get("amplitude_err_mag")
        row["amplitude_model_mag"] = lc.get("model_amplitude_mag")
        row["amplitude_n6_fit_mag"] = lc.get("amplitude_n6_fit_mag")
    g = summary.get("combined_peak_gaussian", {})
    row["peak_fwhm_hours"] = g.get("gauss_fwhm_hours")
    row["period_resolution_hours"] = g.get("period_resolution_hours")
    ps = summary.get("periodogram_n=2", {})
    row["peak_threshold_n=2"] = ps.get("threshold")
    row["peaks_n=2_hours"] = "; ".join(f"{p['period_hours']:.3f} ({p['power']:.3f})" for p in ps.get("peaks", []))
    return row


def _json_default(o):
    return o.tolist() if hasattr(o, "tolist") else str(o)


def merge_summaries():
    """Collect paper_figures/summaries/<N>.json into summary.json and summary.csv."""
    sdir = os.path.join(OUT_DIR, "summaries")
    all_summary = {}
    if os.path.isdir(sdir):
        for name in sorted(os.listdir(sdir)):
            if name.endswith(".json"):
                with open(os.path.join(sdir, name), "r", encoding="utf-8") as fh:
                    s = json.load(fh)
                all_summary[str(s["asteroid"])] = s
    summary_path = os.path.join(OUT_DIR, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(all_summary, fh, indent=2, default=_json_default)
    rows = [_flatten_for_csv(s) for s in all_summary.values() if "error" not in s]
    if rows:
        pd.DataFrame(rows).sort_values("asteroid").to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)
    print("Merged", len(all_summary), "summaries into", summary_path)


def _set_options(argv):
    """--bias=band|dephocus  --band-phase  --tag=NAME  --no-extras  --amp=fourier|template  --lc-nterms=N"""
    global OUT_DIR, LS_CACHE_DIR
    for a in argv:
        if a.startswith("--bias="):
            OPTIONS["bias"] = a.split("=", 1)[1]
        elif a == "--band-phase":
            OPTIONS["band_phase"] = True
        elif a.startswith("--tag="):
            OPTIONS["tag"] = a.split("=", 1)[1]
        elif a == "--no-extras":
            OPTIONS["extras"] = False
        elif a.startswith("--amp="):
            OPTIONS["amp_source"] = a.split("=", 1)[1]
        elif a.startswith("--lc-nterms="):
            OPTIONS["lc_nterms"] = int(a.split("=", 1)[1])
    if OPTIONS["tag"]:
        OUT_DIR = os.path.join(REPO_DIR, "paper_figures", "runs", OPTIONS["tag"])
        LS_CACHE_DIR = os.path.join(OUT_DIR, "ls_cache")
    os.makedirs(OUT_DIR, exist_ok=True)


def main(argv):
    _set_options(argv)
    args = [a for a in argv if not a.startswith("--")]
    skip_opp = "--skip-opposition" in argv
    if "--merge" in argv:
        merge_summaries()
        return
    numbers = [int(a) for a in args] if args else list(CONFIG)
    sdir = os.path.join(OUT_DIR, "summaries")
    os.makedirs(sdir, exist_ok=True)

    for num in numbers:
        print("\n" + "=" * 78 + f"\nAsteroid {num}\n" + "=" * 78)
        try:
            s = run_asteroid(num, CONFIG[num], skip_opposition=skip_opp)
        except Exception as exc:
            s = {"asteroid": num, "error": f"{type(exc).__name__}: {exc}"}
            traceback.print_exc()
        with open(os.path.join(sdir, f"{num}.json"), "w", encoding="utf-8") as fh:
            json.dump(s, fh, indent=2, default=_json_default)

    merge_summaries()
    print("\nDone.")


if __name__ == "__main__":
    main(sys.argv[1:])
