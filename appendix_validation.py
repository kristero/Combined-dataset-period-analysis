"""
Appendix A figure: the four asteroids with known rotation periods (1951, 1963, 2134, 2150) analyzed with the
C-method. Left: power spectrum (n = 2, 0.5-240 h) with the peaks of the C-method run and the ALCDEF period
(dashed line). Right: light curve folded at the C-method period with the n = 4 Fourier model. Same style as the
power spectrum and light curve figures of the paper (fig_style.py). Uses the cached periodograms, the summaries
and the options of a run (python appendix_validation.py --bias=dephocus --tag=rev3).
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

import fig_style
import paper_figures as pf
from Asteroid_pickle import DatasetGenerator
from asteroid_ls import AsteroidLSPipeline

ALCDEF = {1951: 5.302, 1963: 18.160, 2134: 4.114, 2150: 6.125}
NAMES = {1951: "Lick", 1963: "Bezovec", 2134: "Dennispalm", 2150: "Nyctimene"}


def combined(num):
    """Combined C-method data of one asteroid: the pipeline object (datasets) and the times and magnitudes."""
    cfg = pf.CONFIG[num]
    wdir, wname = os.path.split(cfg["workbook"])
    gen = DatasetGenerator(wdir + os.sep, wname, num, cfg["sheets"], base_dir=pf.REPO_DIR, bias_mode=pf.OPTIONS["bias"])
    H, G1, G2 = gen.reference_obs_comp(cfg["reference"], method="HG1G2")
    ds = gen.all_obs_comb(H, G1_val=G1, G2_val=G2, method="HG1G2")
    plt.close("all")
    pipe = AsteroidLSPipeline(ds, num, base_dir=pf.REPO_DIR)
    t, y = pipe.reading_data()
    plt.close("all")
    return pipe, np.asarray(t, float), np.asarray(y, float)


def main(argv):
    pf._set_options(argv)
    # Build the combined data first: the pipeline draws diagnostic plots on the current axes.
    data = {}
    for num in ALCDEF:
        data[num] = combined(num)
        plt.close("all")
    fig_style.set_rc()
    fig, axs = plt.subplots(4, 2, figsize=(7.0, 10.2), constrained_layout=True,
                            gridspec_kw={"width_ratios": [1.45, 1]})
    info = {}
    for i, num in enumerate(ALCDEF):
        s = json.load(open(os.path.join(pf.OUT_DIR, "summaries", f"{num}.json")))
        lc = next(v for k, v in s.items() if k.startswith("light_curve_P="))
        Pc = lc["period_hours"]
        ps = s["periodogram_n=2"]
        d = np.load(os.path.join(pf.LS_CACHE_DIR, f"{num}_LS_n=2.npz"))
        thr = ps["threshold"]
        thr_txt = f"{thr:.2f}" if thr >= 0.1 else f"{thr:.3f}"
        ax = axs[i, 0]
        fig_style.spectrum_panel(
            fig, ax, d["frequency"], d["power"],
            [p["period_hours"] for p in ps["peaks"]], [p["power"] for p in ps["peaks"]], f"Peaks > {thr_txt}",
            extra=lambda a, P=ALCDEF[num]: a.axvline(P, ls="--", lw=0.7, color="0.3", label="ALCDEF"),
            legend_fs=6, colsp=1.0)
        ax.set_title(f"({num}) {NAMES[num]}", loc="left", fontsize=8)
        ax.set_ylabel("Power")
        pipe, t, y = data[num]
        fb = 24.0 / Pc
        t0 = float(pipe.time_zero)
        groups = []
        for name, (H_s, t_s, _) in pipe.all_sheets.items():
            H_s, t_s = np.asarray(H_s, float), np.asarray(t_s, float)
            ok = np.isfinite(H_s) & np.isfinite(t_s)
            groups.append((name.replace("T08o1", "T08o"), (((t_s[ok] - t0) * fb) % 1.0) * 360.0, H_s[ok]))
        grid = np.linspace(0, 1, 500)
        model = LombScargle(t, y, nterms=4).model(grid / fb, fb)
        ax = axs[i, 1]
        fig_style.light_curve_panel(
            ax, groups, grid * 360.0, model,
            f"P = {Pc:.3f} h\nA = {lc['template_amp']:.2f} $\\pm$ {lc['template_err']:.2f} mag", ncol=5, size=6)
        ax.set_ylabel("Reduced magnitude")
        if i == 3:
            axs[i, 0].set_xlabel("Rotational period (hours)")
            ax.set_xlabel("Rotational phase (deg)")
        info[num] = {"P_C": Pc, "t0": s["time_zero_jd"]}
    fig.savefig(os.path.join(pf.OUT_DIR, "appendix_validation.pdf"))
    fig.savefig(os.path.join(pf.OUT_DIR, "appendix_validation.png"), dpi=200)
    json.dump(info, open(os.path.join(pf.OUT_DIR, "appendix_validation.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
