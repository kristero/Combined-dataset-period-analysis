"""
Appendix analysis of (3173) McNaught: does adding the TESS (C57G), I41r and M22o datasets clarify the period,
and how probable are the long-period candidates?

For every dataset selection, the combined dataset is built with the paper pipeline (paper_figures.OPTIONS,
DePhOCUS offsets and band-dependent phase curves when --final is given), then
  * Lomb-Scargle (n = 2) over 0.5-240 h and over 240-1500 h, and the power at the candidate periods;
  * bootstrap over observing nights (one night of one dataset is one block, resampled with replacement):
    fraction of resamples in which each candidate has the highest power among the candidates;
  * false-alarm probability of the strongest long-period peak: the night means of the residuals are
    permuted among the nights (within-night structure and sampling kept), and the highest power over
    240-1500 h of 300 permuted datasets is compared with the observed one.
Writes paper_figures/runs/<tag>/appendix_3173.json, the combined data (npz) and the appendix figure.
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
import lc_amplitude

NUM = 3173
CANDS = [22.99, 24.93, 45.98, 49.9, 571.7, 1143.0]
VARIANTS = {"paper": [], "tess": ["C57G"], "all": ["C57G", "I41r", "M22o"]}


def build(extra):
    cfg = pf.CONFIG[NUM]
    wdir, wname = os.path.split(cfg["workbook"])
    gen = DatasetGenerator(wdir + os.sep, wname, NUM, cfg["sheets"] + extra, base_dir=pf.REPO_DIR,
                           bias_mode=pf.OPTIONS["bias"])
    H, G1, G2 = gen.reference_obs_comp(cfg["reference"], method="HG1G2")
    if pf.OPTIONS["band_phase"]:
        gen.fit_band_slopes(H, G1, G2, cfg["reference"])
    ds = gen.all_obs_comb(H, G1_val=G1, G2_val=G2, method="HG1G2")
    plt.close("all")
    pipe = AsteroidLSPipeline(ds, NUM, base_dir=pf.REPO_DIR)
    t, y = pipe.reading_data()
    plt.close("all")
    sheet = np.delete(np.asarray(pipe.sheet, dtype=str), pipe.remove_idx)
    return t, y, sheet, float(pipe.time_zero), gen.delta_H


def local_power(ls, P, base, frac=0.01):
    f = np.arange(24.0 / (P * (1 + frac)), 24.0 / (P * (1 - frac)), 1.0 / (40 * base))
    p = ls.power(f, method="fastchi2", assume_regular_frequency=True)
    k = int(np.argmax(p))
    return float(24.0 / f[k]), float(p[k])


def top_peaks(ls, lo, hi, base, spp, n=8):
    f = np.arange(24.0 / hi, 24.0 / lo, 1.0 / (spp * base))
    p = ls.power(f, method="fastchi2", assume_regular_frequency=True)
    pk, _ = find_peaks(p, distance=3)
    pk = pk[np.argsort(p[pk])[::-1]][:n]
    return f, p, [{"P": float(24.0 / f[i]), "w": float(p[i])} for i in pk]


def bootstrap_probabilities(t, y, sheet, cands, n_boot=300, seed=7):
    """Fraction of night-block bootstrap resamples in which each candidate is the strongest candidate."""
    base = float(t.max() - t.min())
    blocks = np.array([f"{s}|{int(np.floor(x))}" for s, x in zip(sheet, t)])
    ub, inv = np.unique(blocks, return_inverse=True)
    idx = [np.where(inv == b)[0] for b in range(len(ub))]
    grids = [np.arange(24.0 / (P * 1.01), 24.0 / (P * 0.99), 1.0 / (20 * base)) for P in cands]
    rng = np.random.default_rng(seed)
    wins = np.zeros(len(cands), int)
    for _ in range(n_boot):
        sel = np.concatenate([idx[b] for b in rng.integers(0, len(ub), len(ub))])
        ls = LombScargle(t[sel], y[sel], nterms=2)
        best = [np.max(ls.power(g, method="fastchi2", assume_regular_frequency=True)) for g in grids]
        wins[int(np.argmax(best))] += 1
    return {str(P): float(w / n_boot) for P, w in zip(cands, wins)}


def fap_long(t, y, sheet, obs_power, n_perm=300, seed=11):
    """FAP of the strongest 240-1500 h peak from permutations of the night means among the nights."""
    base = float(t.max() - t.min())
    blocks = np.array([f"{s}|{int(np.floor(x))}" for s, x in zip(sheet, t)])
    ub, inv = np.unique(blocks, return_inverse=True)
    means = np.array([np.mean(y[inv == b]) for b in range(len(ub))])
    resid = y - means[inv]
    f = np.arange(24.0 / 1500.0, 24.0 / 240.0, 1.0 / (10 * base))
    rng = np.random.default_rng(seed)
    pmax = []
    for _ in range(n_perm):
        ys = resid + rng.permutation(means)[inv]
        pmax.append(np.max(LombScargle(t, ys, nterms=2).power(f, method="fastchi2", assume_regular_frequency=True)))
    pmax = np.array(pmax)
    return {"fap": float((np.sum(pmax >= obs_power) + 1) / (n_perm + 1)), "perm_max_median": float(np.median(pmax)),
            "perm_max_99": float(np.percentile(pmax, 99)), "n_perm": n_perm}


def main(argv):
    pf._set_options(argv)
    out_dir = pf.OUT_DIR
    res = {"options": dict(pf.OPTIONS)}
    for name, extra in VARIANTS.items():
        t, y, sheet, t0, dH = build(extra)
        np.savez_compressed(os.path.join(out_dir, f"3173_{name}_data.npz"), t=t, y=y, sheet=sheet, t0=t0)
        base = float(t.max() - t.min())
        ls = LombScargle(t, y, nterms=2)
        r = {"n": int(len(t)), "datasets": sorted(set(sheet.tolist())), "baseline_days": base,
             "delta_H": dH}
        _, _, r["peaks_0.5-240"] = top_peaks(ls, 0.5, 240.0, base, 15)
        fl, pl, r["peaks_240-1500"] = top_peaks(ls, 240.0, 1500.0, base, 20)
        r["at"] = {str(P): local_power(ls, P, base) for P in CANDS}
        r["bootstrap_prob"] = bootstrap_probabilities(t, y, sheet, CANDS)
        r["fap_long"] = fap_long(t, y, sheet, r["peaks_240-1500"][0]["w"])
        for P in (r["at"]["45.98"][0], r["at"]["1143.0"][0]):
            a = lc_amplitude.fourier_amplitude(t, y, P, nterms=4, sheet=sheet, n_boot=200)
            r.setdefault("amplitude_n4", {})[f"{P:.2f}"] = [a["amplitude"], a["amplitude_err"]]
        res[name] = r
        print(name, json.dumps({k: r[k] for k in ("n", "at", "bootstrap_prob", "fap_long")}, indent=1))
        json.dump(res, open(os.path.join(out_dir, "appendix_3173.json"), "w"), indent=1)
    figure(out_dir)


def figure(out_dir):
    """Appendix figure: (a) periodogram 0.5-1500 h, (b) light curve folded at the double-wave long period,
    (c) the 2020 TESS measurements (C57G) against time."""
    d = np.load(os.path.join(out_dir, "3173_paper_data.npz"))
    t, y = d["t"], d["y"]
    res = json.load(open(os.path.join(out_dir, "appendix_3173.json")))
    base = float(t.max() - t.min())
    ls = LombScargle(t, y, nterms=2)
    f1, f2 = np.arange(24 / 1500, 24 / 240, 1 / (20 * base)), np.arange(24 / 240, 48, 1 / (5 * base))
    p = np.concatenate([ls.power(g, method="fastchi2", assume_regular_frequency=True) for g in (f1, f2)])
    f = np.concatenate([f1, f2])
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42})
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5), constrained_layout=True)
    ax[0].plot(24 / f, p, lw=0.4, color="#8c510a")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("Period (hours)")
    ax[0].set_ylabel("Power")
    for P, dx, ha in ((22.99, -4, "right"), (45.98, 4, "left"), (49.9, 4, "left"), (571.7, -4, "right"), (1143.0, 4, "left")):
        Pb, wb = res["paper"]["at"][str(P)]
        ax[0].annotate(f"{Pb:.0f}" if Pb > 100 else f"{Pb:.1f}", (Pb, wb), textcoords="offset points",
                       xytext=(dx, 2), ha=ha, fontsize=6)
    ax[0].set_ylim(0, 0.5)
    ax[0].set_title("(a)", loc="left")
    P_long = res["paper"]["at"]["1143.0"][0]
    ph = ((t * 24 / P_long) % 1) * 360
    a = lc_amplitude.fourier_amplitude(t, y, P_long, nterms=4, n_boot=2)
    ax[1].scatter(ph, y, s=2, color="#d8b365", alpha=0.6, edgecolors="none")
    ax[1].plot(a["model_phase"] * 360, a["model_mag"], "--", color="black", lw=0.9)
    ax[1].invert_yaxis()
    ax[1].set_xlim(0, 360)
    ax[1].set_xlabel("Rotational phase (deg)")
    ax[1].set_ylabel("Reduced magnitude")
    ax[1].set_title(f"(b) P = {P_long:.0f} h", loc="left")
    dt = np.load(os.path.join(out_dir, "3173_tess_data.npz"))
    jd = dt["t"] + float(dt["t0"])
    m = (dt["sheet"] == "C57G") & (jd < 2459300)
    ax[2].scatter(jd[m] - 2459100, dt["y"][m], s=3, color="#35978f", edgecolors="none")
    ax[2].invert_yaxis()
    ax[2].set_xlabel("JD - 2459100")
    ax[2].set_ylabel("Reduced magnitude")
    ax[2].set_title("(c) TESS 2020", loc="left")
    fig.savefig(os.path.join(out_dir, "3173_long_period_appendix.pdf"))
    fig.savefig(os.path.join(out_dir, "3173_long_period_appendix.png"), dpi=200)


if __name__ == "__main__":
    if "--figure" in sys.argv:
        pf._set_options(sys.argv[1:])
        figure(pf.OUT_DIR)
    else:
        main(sys.argv[1:])
