"""
Appendix A figure: the four asteroids with known rotation periods (1951, 1963, 2134, 2150) analyzed with the
C-method. Left: power spectrum (n = 2, 0.5-240 h) with the ALCDEF period marked. Right: light curve folded at
the C-method period with the n = 4 Fourier model. Uses the cached periodograms and the options of a run
(python appendix_validation.py --bias=dephocus --tag=deph).
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

import paper_figures as pf
from Asteroid_pickle import DatasetGenerator
from asteroid_ls import AsteroidLSPipeline

ALCDEF = {1951: 5.302, 1963: 18.160, 2134: 4.114, 2150: 6.125}


def combined(num):
    cfg = pf.CONFIG[num]
    wdir, wname = os.path.split(cfg["workbook"])
    gen = DatasetGenerator(wdir + os.sep, wname, num, cfg["sheets"], base_dir=pf.REPO_DIR, bias_mode=pf.OPTIONS["bias"])
    H, G1, G2 = gen.reference_obs_comp(cfg["reference"], method="HG1G2")
    ds = gen.all_obs_comb(H, G1_val=G1, G2_val=G2, method="HG1G2")
    plt.close("all")
    pipe = AsteroidLSPipeline(ds, num, base_dir=pf.REPO_DIR)
    t, y = pipe.reading_data()
    plt.close("all")
    return t, y


def main(argv):
    pf._set_options(argv)
    # Build the combined data first: the pipeline draws diagnostic plots on the current axes.
    data = {}
    for num in ALCDEF:
        data[num] = combined(num)
        plt.close("all")
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "pdf.fonttype": 42})
    fig, axs = plt.subplots(4, 2, figsize=(7.0, 8.6), constrained_layout=True,
                            gridspec_kw={"width_ratios": [1.5, 1]})
    info = {}
    for i, num in enumerate(ALCDEF):
        d = np.load(os.path.join(pf.LS_CACHE_DIR, f"{num}_LS_n=2.npz"))
        f, p = np.asarray(d["frequency"], float), np.asarray(d["power"], float)
        s = json.load(open(os.path.join(pf.OUT_DIR, "summaries", f"{num}.json")))
        lc = next(v for k, v in s.items() if k.startswith("light_curve_P="))
        Pc = lc["period_hours"]
        ax = axs[i, 0]
        ax.plot(24.0 / f, p, lw=0.35, color="#8c510a")
        ax.set_xscale("log")
        ax.set_xlim(0.5, 240)
        ax.axvline(ALCDEF[num], ls="--", lw=0.7, color="0.3")
        ax.set_ylabel("Power")
        ax.text(0.01, 0.95, f"({num})  ALCDEF {ALCDEF[num]:.3f} h, C-method {Pc:.3f} h", transform=ax.transAxes,
                ha="left", va="top", fontsize=7, bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1))
        if i == 3:
            ax.set_xlabel("Period (hours)")
        t, y = data[num]
        fb = 24.0 / Pc
        ls4 = LombScargle(t, y, nterms=4)
        ph = (t * fb) % 1.0
        grid = np.linspace(0, 1, 400)
        ax = axs[i, 1]
        ax.scatter(ph * 360, y, s=1.5, color="#d8b365", alpha=0.6, edgecolors="none")
        ax.plot(grid * 360, ls4.model(grid / fb, fb), "--", color="black", lw=0.9)
        ax.set_xlim(0, 360)
        lo, hi = np.percentile(y, [1, 99])
        ax.set_ylim(hi + 0.1, lo - 0.1)
        ax.set_ylabel("Reduced magnitude")
        if i == 3:
            ax.set_xlabel("Rotational phase (deg)")
        info[num] = {"P_C": Pc, "t0": s["time_zero_jd"]}
    fig.savefig(os.path.join(pf.OUT_DIR, "appendix_validation.pdf"))
    fig.savefig(os.path.join(pf.OUT_DIR, "appendix_validation.png"), dpi=200)
    json.dump(info, open(os.path.join(pf.OUT_DIR, "appendix_validation.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
