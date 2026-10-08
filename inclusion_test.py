"""
Inclusion test of the datasets that are not used in the C-method (second revision).

Every dataset with at least 70 measurements that is not in the combined data of the paper is added to
the combined data on its own, and the C-method is run with the same settings as in paper_figures.py
(DePhOCUS offsets, T08o reference fit with free H, G1, G2, fixed slopes for the other datasets, outlier
limits of step 2, Lomb-Scargle with n = 2 over 0.5-240 h, 15 points per peak width). Recorded:
  * criterion (1) of step 7: the free HG1G2 fit of the dataset alone converges to G1 = 0, G2 = 1 (or vice versa)
  * criterion (2) of step 7: reduced chi^2 of the fixed-slope fit relative to the reference observatory
  * the power of the C-method period (highest point within +/- 0.5 %) with and without the dataset,
    and the strongest peaks of the whole search range
  * a figure for the visual check of criterion (3): phase curve of the dataset and the combined light
    curve folded with the C-method period, with the dataset highlighted

The datasets that are used are tested in the same way by leaving them out one at a time ('-<dataset>').

Run:  python inclusion_test.py <N> [base|<dataset>|-<dataset> ...]
      -> paper_figures/inclusion/<N>_<base|dataset|minus_dataset>.json (+ .png for added datasets)
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks

import paper_figures as pf
from Asteroid_pickle import DatasetGenerator
from asteroid_ls import AsteroidLSPipeline

REPO_DIR = pf.REPO_DIR
WB_DIR = os.path.join(REPO_DIR, "..", "Asteroid_data", "rev3")
OUT = os.path.join(REPO_DIR, "paper_figures", "inclusion_local" if os.environ.get("INC_LOCAL") else "inclusion")

# Datasets with at least 70 measurements that are not used in the C-method of the paper.
CANDIDATES = {
    1951: ["Y00R"], 1963: ["C57G", "T08w", "704V"], 2134: ["704V"], 2150: ["704V", "T05w"],
    2607: ["704V"], 2968: ["704V"], 2971: ["C57G", "704V"], 3081: ["C57G", "704V", "I41r"],
    3173: ["C57G", "I41r", "M22o", "704V", "F51w", "T05w"], 3473: ["I41r", "704V"],
    3716: ["704V", "C57G", "F51w"], 4303: ["704V"],
}
# Period of the C-method used for the comparison (3173: the strongest candidate of the 0.5-240 h search).
PERIOD = {1951: 5.2995, 1963: 18.1644, 2134: 4.1149, 2150: 6.1241, 2607: 2.9356, 2968: 4.5596,
          2971: 4.4908, 3081: 8.0073, 3173: 45.9825, 3473: 9.0740, 3716: 10.4743, 4303: 6.1363}


def combined(num: int, sheets: list, ref: str):
    gen = DatasetGenerator(WB_DIR + os.sep, f"{num}-excelRB.xlsx", num, sheets, base_dir=REPO_DIR,
                           bias_mode="dephocus")
    H, G1, G2, He, G1e, G2e = gen.reference_obs_comp(ref, method="HG1G2", return_errors=True)
    plt.close("all")
    ds = gen.all_obs_comb(H, G1_val=G1, G2_val=G2, method="HG1G2")
    plt.close("all")
    pipe = AsteroidLSPipeline(ds, num, base_dir=REPO_DIR)
    ts, ys = pipe.reading_data()
    plt.close("all")
    ls = LombScargle(ts, ys, nterms=2)
    if os.environ.get("INC_LOCAL"):
        # Fast mode: power only within +/- 1 % of the C-method frequency, its double and (3173) the other
        # candidates, sampled with 50 points per peak width (no full-range search, no top_peaks).
        T = float(np.max(ts) - np.min(ts))
        f0s = [24.0 / PERIOD[num], 48.0 / PERIOD[num]] + ([24.0 / 22.991, 24.0 / 49.98] if num == 3173 else [])
        f = np.unique(np.concatenate([np.arange(0.99 * f0, 1.01 * f0, 1.0 / (50 * T)) for f0 in f0s]))
        p = ls.power(f)
    else:
        f, p = ls.autopower(minimum_frequency=24.0 / pf.P_MAX_HOURS, maximum_frequency=24.0 / pf.P_MIN_HOURS,
                            samples_per_peak=int(os.environ.get("INC_SPP", pf.SAMPLES_PER_PEAK)))
    return gen, ds, pipe, np.asarray(ts, float), np.asarray(ys, float), f, p, (H, G1, G2, He, G1e, G2e)


def power_at(f, p, P, tol=0.005):
    per = 24.0 / f
    m = np.abs(per - P) <= tol * P
    i = np.where(m)[0][np.argmax(p[m])]
    return float(per[i]), float(p[i])


def top_peaks(f, p, n=8):
    idx, _ = find_peaks(p, distance=pf.PEAK_DISTANCE)
    idx = idx[np.argsort(-p[idx])][:n]
    return [{"P": float(24.0 / f[i]), "w": float(p[i])} for i in idx]


def free_fit(gen: DatasetGenerator, sheet: str):
    """Free HG1G2 fit of one dataset (criterion 1), on the same data as in step 2."""
    import pandas as pd
    df = pd.read_excel(gen.path + gen.file_name, sheet_name=sheet).dropna(subset=["magred"])
    H = df["mag"].to_numpy(float) - 5 * np.log10(df.iloc[:, 3].to_numpy(float) * df.iloc[:, 4].to_numpy(float))
    H = H + gen._bias(sheet, df)
    ph, h, _ = gen.removeOutliers(df["Ph"].to_numpy(float), H, 1.8)
    r = gen.fit(np.asarray(ph), np.asarray(h), method="HG1G2")
    G1, G2 = float(r.params["G1"].value), float(r.params["G2"].value)
    boundary = bool((G1 < 0.01 and G2 > 0.99) or (G1 > 0.99 and G2 < 0.01))
    return {"H": float(r.params["H"].value), "G1": G1, "G2": G2, "redchi": float(r.redchi), "boundary": boundary,
            "n": int(len(h)), "phase_min": float(np.min(ph)), "phase_max": float(np.max(ph))}


def run(num: int, variant: str):
    """variant: 'base' (datasets of the paper), a dataset name (added) or '-<dataset>' (left out)."""
    cfg = pf.CONFIG[num]
    if variant == "base":
        sheets = list(cfg["sheets"])
    elif variant.startswith("-"):
        sheets = [s for s in cfg["sheets"] if s != variant[1:]]
    else:
        sheets = list(cfg["sheets"]) + [variant]
    P = PERIOD[num]
    gen, ds, pipe, ts, ys, f, p, ref = combined(num, sheets, cfg["reference"])
    out = {"asteroid": num, "variant": variant, "sheets": sheets, "n_total": int(len(ys)),
           "n_per_sheet": {k: int(len(np.asarray(v[0], float))) for k, v in ds.items()},
           "reference_fit": dict(zip(["H", "G1", "G2", "H_err", "G1_err", "G2_err"], map(float, ref))),
           "delta_H": gen.delta_H, "bias_log": gen.bias_log, "P_C": P}
    out["peak_P"], out["peak_w"] = power_at(f, p, P)
    out["half_P"], out["half_w"] = power_at(f, p, P / 2)
    if num == 3173:
        out["P_22.99"], out["w_22.99"] = power_at(f, p, 22.991)
        out["P_49.98"], out["w_49.98"] = power_at(f, p, 49.98)
    out["top_peaks"] = top_peaks(f, p)
    ref_chi = gen.delta_H.get(cfg["reference"], {}).get("redchi")
    if variant == "base":
        out["criterion2_ratios"] = {k: float(v["redchi"] / ref_chi) for k, v in gen.delta_H.items()}
    elif not variant.startswith("-"):
        dh = gen.delta_H.get(variant, {})
        out["criterion2_ratio"] = float(dh["redchi"] / ref_chi) if dh and ref_chi else None
        out["free_fit"] = free_fit(gen, variant)
        # Residual rms of every dataset about the n = 4 model of the combined data at the C-method period.
        f_c = 24.0 / out["peak_P"]
        t0 = float(pipe.time_zero)
        model = LombScargle(ts, ys, nterms=4)
        rms = {}
        for k, v in pipe.all_sheets.items():
            y_k, t_k = np.asarray(v[0], float), np.asarray(v[1], float) - t0
            ok = np.isfinite(y_k) & np.isfinite(t_k)
            if ok.sum() > 5:
                rms[k] = float(np.std(y_k[ok] - model.model(t_k[ok], f_c)))
        out["residual_rms"] = rms
        # Figure for the visual check (criterion 3).
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        for k, v in pipe.all_sheets.items():
            y_k, t_k, ph_k = (np.asarray(v[i], float) for i in range(3))
            t_k = t_k - t0
            sel = k == variant
            ax[0].scatter(ph_k, y_k, s=6 if sel else 2, c="crimson" if sel else "0.75", zorder=3 if sel else 1)
            phase = (t_k * f_c) % 1.0
            ax[1].scatter(phase, y_k, s=6 if sel else 2, c="crimson" if sel else "0.75", zorder=3 if sel else 1)
        grid = np.linspace(0, 1, 400)
        ax[1].plot(grid, model.model(grid / f_c, f_c), c="k", lw=1)
        for a in ax:
            a.invert_yaxis()
        ax[0].set_xlabel("Phase angle (deg)"); ax[0].set_ylabel("Reduced magnitude")
        ax[1].set_xlabel(f"Rotational phase (P = {out['peak_P']:.4f} h)")
        ax[0].set_title(f"{num} {variant} (red) after steps 1-6")
        ax[1].set_title(f"w = {out['peak_w']:.3f}")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"{num}_{variant}.png"), dpi=110)
        plt.close("all")
    name = variant if not variant.startswith("-") else "minus_" + variant[1:]
    json.dump(out, open(os.path.join(OUT, f"{num}_{name}.json"), "w"), indent=1, default=float)
    print(json.dumps({k: out[k] for k in ("asteroid", "variant", "n_total", "peak_P", "peak_w")}))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    num = int(sys.argv[1])
    variants = sys.argv[2:] or ["base"] + CANDIDATES[num]
    for v in variants:
        run(num, v)
