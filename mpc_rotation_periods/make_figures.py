"""
Figures of the revised manuscript, all produced from the same analysis run as the tables
(results/<N>/<N>_arrays.npz, results/<N>/<N>_result.json) and the adopted periods (adopted.json).

  fig_pipeline_2607.pdf        combination pipeline for (2607) (Fig. 1)
  fig_single_dataset.pdf       single-dataset periodogram and fold (Fig. 2; example chosen with
                               search_single_dataset_example.py)
  fig_lc_01.pdf ... fig_lc_03.pdf   periodograms and folded light curves, 15 asteroids per page
  fig_dataset_power.pdf        periodogram power at the adopted period and preferred period of every dataset
                               (no longer in the paper: Figure 7 is drawn by fig_dataset_matrix.py from
                               dataset_power_table.json, same values)
  fig_period_diameter.pdf      period-diameter diagram

Datasets whose magnitudes are reported only to 0.1 mag (all TESS photometry in the MPC database and several
ground-based datasets; data/dataset_precision.json) are used in the analysis, the binned means and the models, but
their individual points are not drawn in the folded light curves: dense quantized photometry forms horizontal
streaks at the discrete magnitude levels, which the smooth distance and phase corrections turn into slanted tracks.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = r"C:\Users\kn18001\Documents\Asteroids\Combined-dataset-period-analysis"
sys.path.insert(0, REPO)
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "manuscript_revised")
C_AU_PER_DAY = 173.1446326846693

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5, "axes.linewidth": 0.6,
    "xtick.major.size": 3, "ytick.major.size": 3, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
})

with open(os.path.join(ROOT, "adopted.json"), "r", encoding="utf-8") as fh:
    ADOPTED = json.load(fh)
with open(os.path.join(ROOT, "data", "sbdb_phys_orbit.json"), "r", encoding="utf-8") as fh:
    SBDB = json.load(fh)
with open(os.path.join(ROOT, "data", "dataset_precision.json"), "r", encoding="utf-8") as fh:
    PRECISION = json.load(fh)["by_asteroid"]
Q_LIMIT = 0.9          # dataset counts as quantized (0.1-mag steps) if >= 90 % of its magnitudes lie on that grid
MODEL_COLOR = "#1a5276"   # Fourier model (navy), distinct from the YlOrBr data points and the black bin means
PLOT_NTERMS = 6           # order of the Fourier curve drawn in the folded light curves (the outlier clipping and the
                          # combined-fold amplitude keep the two-term model of the period search)
SINGLE_EXAMPLE = (1949, "C57G")     # Fig. 2, chosen with search_tess_example.py (no daily aliases)


def quantized(num):
    return {ds for ds, f in PRECISION[str(num)].items() if f >= Q_LIMIT}


def label(num):
    full = SBDB[str(num)]["object"]["fullname"]          # "4239 Goodman (1980 OE)"
    name = full.split(" ", 1)[1].split(" (")[0]
    return f"({num}) {name}"


RES_GEO = os.path.join(ROOT, "results_geo")      # geometry-corrected data (geometry_reanalysis.py)


def load(num):
    geo = os.path.join(RES_GEO, str(num), f"{num}_arrays.npz")
    a = np.load(geo if os.path.exists(geo) else os.path.join(RES, str(num), f"{num}_arrays.npz"), allow_pickle=False)
    with open(os.path.join(RES, str(num), f"{num}_result.json"), "r", encoding="utf-8") as fh:
        r = json.load(fh)
    return a, r


def fourier(t, y, P, nterms):
    w = 2 * np.pi * 24.0 / P
    cols = [np.ones_like(t)] + [f(k * w * t) for k in range(1, nterms + 1) for f in (np.sin, np.cos)]
    A = np.vstack(cols).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    ph = np.linspace(0, 1, 400)
    colsg = [np.ones_like(ph)] + [f(2 * np.pi * k * ph) for k in range(1, nterms + 1) for f in (np.sin, np.cos)]
    return A @ coef, ph, np.vstack(colsg).T @ coef


def binned(phase, y, model, nbins=20, clip=3.0):
    res = y - model
    keep = np.ones_like(y, dtype=bool)
    for _ in range(5):
        s = np.std(res[keep], ddof=1)
        new = np.abs(res - np.median(res[keep])) <= clip * s
        if np.array_equal(new, keep):
            break
        keep = new
    edges = np.linspace(0, 1, nbins + 1)
    c, m, e = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = keep & (phase >= a) & (phase < b)
        if np.sum(sel) >= 5:
            c.append(0.5 * (a + b))
            m.append(np.mean(y[sel]))
            e.append(np.std(y[sel], ddof=1) / np.sqrt(np.sum(sel)))
    c, m, e = map(np.asarray, (c, m, e))
    i_hi, i_lo = int(np.argmax(m)), int(np.argmin(m))
    amp = float(m[i_hi] - m[i_lo])
    amp_err = float(np.hypot(e[i_hi], e[i_lo]))
    return c, m, e, amp, amp_err, keep


def peak_envelope(t, y, f0):
    """Gaussian envelope of the periodogram peak at frequency f0 (c/d), fitted to the heights of its comb teeth.
    Returns (centre, sigma) in c/d; the peak region is |f - centre| <= 3 sigma (capped at 0.25 c/d, half the
    spacing of the daily aliases of the rotation frequency, so that no daily alias is ever part of the region)."""
    from astropy.timeseries import LombScargle
    from scipy.optimize import curve_fit
    from scipy.signal import find_peaks
    base = float(t.max() - t.min())
    hw = min(0.03, 0.45 * f0)
    g = np.arange(f0 - hw, f0 + hw, 1.0 / (10 * base))
    p = LombScargle(t, y, nterms=2).power(g, method="fastchi2", assume_regular_frequency=True)
    k, _ = find_peaks(p)
    k = k[p[k] >= 0.1 * p.max()]
    sig = None
    if len(k) >= 3:
        try:
            gauss = lambda x, A, mu, s: A * np.exp(-0.5 * ((x - mu) / s) ** 2)
            (A_, mu, s), _ = curve_fit(gauss, g[k], p[k], p0=[p[k].max(), g[k][np.argmax(p[k])], 0.002],
                                       maxfev=5000)
            if 0 < abs(s) < hw and abs(mu - f0) < hw:
                sig, cen = abs(s), mu
        except RuntimeError:
            pass
    if sig is None:      # fallback: half-width at 10 % of the envelope (running maximum over ~3 teeth)
        from scipy.ndimage import maximum_filter1d
        env = maximum_filter1d(p, size=max(3, int(0.004 * 10 * base)))
        i0 = int(np.argmin(np.abs(g - f0)))
        lo, hi = i0, i0
        while lo > 0 and env[lo] > 0.1 * env[i0]:
            lo -= 1
        while hi < len(g) - 1 and env[hi] > 0.1 * env[i0]:
            hi += 1
        cen, sig = f0, max(g[i0] - g[lo], g[hi] - g[i0]) / 3.0
    return cen, min(sig, 0.25 / 3.0)


def competing(num, a, ad, pmax):
    """Alternatives drawn as open circles: above the 5-sigma threshold, inside the plotted range, and outside the
    Gaussian envelope (+/-3 sigma) of the adopted peak and of its harmonics P/2 and 2P, i.e. not a comb tooth or
    harmonic of the adopted period. Returns [(P, power)] and the list of alternatives left out."""
    P, pw = a["period"], a["power"]
    Pad = ad["P"]
    t, y = a["t"], a["y"]
    env = []
    for r in (1.0, 0.5, 2.0):
        if 0.8 <= r * Pad <= pmax:
            env.append(peak_envelope(t, y, 24.0 / (r * Pad)))
    shown, left_out = [], []
    for alt in ad.get("alternatives", []):
        if alt > pmax or alt < 0.8:
            continue
        jj = np.where(np.abs(P - alt) <= 0.003 * alt)[0]
        if not jj.size or float(pw[jj].max()) <= float(a["threshold"]):
            continue
        fa = 24.0 / alt
        if any(abs(fa - c) <= 3 * s for c, s in env) or any(abs(alt - r * Pad) <= 0.01 * alt for r in (0.5, 1.0, 2.0)):
            left_out.append(alt)
            continue
        shown.append((alt, float(pw[jj].max())))
    return shown, left_out


def periodogram_axis(ax, a, ad, num, show_ylabel):
    P, pw = a["period"], a["power"]
    Pad = ad["P"]
    pmax = 1500.0 if Pad > 240 else 240.0
    m = (P >= 0.8) & (P <= pmax)
    ax.plot(P[m], pw[m], lw=0.45, color="black")
    ax.axhline(float(a["threshold"]), ls="--", lw=0.5, color="0.5")
    j = np.where(np.abs(P - Pad) <= 0.003 * Pad)[0]
    y_ad = float(pw[j].max()) if j.size else float(pw[np.argmin(np.abs(P - Pad))])
    ax.scatter([Pad], [y_ad], s=18, color="#c0392b", zorder=4, lw=0)
    shown, left_out = competing(num, a, ad, pmax)
    for alt, h in shown:
        ax.scatter([alt], [h], s=18, facecolors="none", edgecolors="#2471a3", lw=0.9, zorder=4)
    ax.set_xscale("log")
    ax.set_xlim(0.8, pmax)
    ticks = [t for t in (1, 3, 10, 30, 100, 300, 1000) if 0.8 <= t <= pmax]
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", pad=1.5)
    top = max(float(pw[m].max()), y_ad) * 1.18
    ax.set_ylim(0, top)
    ax.set_yticks([0, float(f"{top / 1.18:.1g}")])
    if show_ylabel:
        ax.set_ylabel("Power", labelpad=1)
    return shown, left_out


def fold_axis(ax, a, ad, num, show_xlabels, show_ylabel):
    t, y, sheet = a["t"], a["y"], a["sheet"]
    P = ad["P"]
    nterms = int(ad.get("nterms", 2))
    model, _, _ = fourier(t, y, P, nterms)
    phase = ((t * 24.0 / P) % 1.0)
    c, m, e, amp, amp_err, keep = binned(phase, y, model)
    _, phg, mg = fourier(t[keep], y[keep], P, PLOT_NTERMS)      # curve drawn: six-term series of the kept points
    fine = keep & ~np.isin(sheet, list(quantized(num)))     # individual points only for 0.01-mag data
    # "autumn" theme of the method (as in the original code): datasets coloured with the darker part of YlOrBr,
    # drawn faintly in the background
    sets = sorted(set(sheet[fine]))
    pal = plt.get_cmap("YlOrBr")(np.linspace(0.45, 0.95, max(len(sets), 2)))
    for ds, col in zip(sets, pal):
        sel = fine & (sheet == ds)
        ax.scatter(phase[sel], y[sel], s=1.2, color=col, alpha=0.45, lw=0, rasterized=True)
    ax.plot(phg, mg, color=MODEL_COLOR, lw=1.1)
    ax.errorbar(c, m, yerr=e, fmt="s", ms=2.3, color="black", elinewidth=0.6, capsize=0)
    lo, hi = np.percentile(y[keep], [1.0, 99.0])
    pad = 0.08 * (hi - lo)
    ax.set_ylim(hi + pad, lo - pad)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0", "0.5", "1"] if show_xlabels else [])
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    if show_xlabels:
        ax.set_xlabel("Rotational phase", labelpad=1)
    if show_ylabel:
        ax.set_ylabel("Reduced mag", labelpad=1)
    return amp, amp_err, int(len(t)), int(fine.sum())


def fmt_period(P, s):
    """Same rounding as the tables (make_tables.fmt_p): uncertainty to two significant digits."""
    if s is None or not np.isfinite(s) or s <= 0:
        return f"{P:.4f}"
    dec = min(max(0, -int(np.floor(np.log10(s))) + 1), 6)
    s_r = round(s, dec) or 10 ** (-dec)
    return f"{P:.{dec}f} $\\pm$ {s_r:.{dec}f}"


def fmt_compact(P, s):
    """Period with its uncertainty in units of the last digit, e.g. 5.356140(74); same rounding as the tables."""
    dec = min(max(0, -int(np.floor(np.log10(s))) + 1), 6)
    s_r = round(s, dec) or 10 ** (-dec)
    return f"{P:.{dec}f}({int(round(s_r * 10 ** dec))})"


def lc_pages(order, per_page=15, ncols=3):
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    pages = [order[i:i + per_page] for i in range(0, len(order), per_page)]
    amps, marks = {}, {}
    nrows = int(np.ceil(per_page / ncols))
    with plt.rc_context({"font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                         "axes.titlesize": 7.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5}):
        for k, nums in enumerate(pages, start=1):
            fig = plt.figure(figsize=(7.2, 9.6))
            outer = fig.add_gridspec(nrows, ncols, left=0.07, right=0.99, top=0.975, bottom=0.045,
                                     hspace=0.34, wspace=0.25)
            titles = []
            last_row_in_col = {col: max(i // ncols for i in range(len(nums)) if i % ncols == col)
                               for col in range(min(ncols, len(nums)))}
            for i, num in enumerate(nums):
                row, col = divmod(i, ncols)
                inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[row, col], height_ratios=[1, 1.65],
                                                hspace=0.28)
                axp, axf = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])
                a, _ = load(num)
                ad = ADOPTED[str(num)]
                shown, left_out = periodogram_axis(axp, a, ad, num, show_ylabel=(col == 0))
                amp, amp_err, n, n_pts = fold_axis(axf, a, ad, num, show_xlabels=(row == last_row_in_col[col]),
                                                   show_ylabel=(col == 0))
                amps[str(num)] = {"amp": amp, "amp_err": amp_err, "n_fold": n, "n_points_drawn": n_pts}
                marks[str(num)] = {"competing_shown": [p for p, _ in shown], "same_peak_left_out": left_out}
                tl = axp.text(0.0, 1.04, label(num), transform=axp.transAxes, ha="left", va="bottom", fontsize=7.8)
                hdr = (f"no period (fold {fmt_short(ad['P'])} h)" if ad.get("status") == "undetermined"
                       else f"{fmt_compact(ad['P'], ad['sigma_P'])} h  {ad.get('quality', '')}")
                tr = axp.text(1.0, 1.04, hdr,
                              transform=axp.transAxes, ha="right", va="bottom", fontsize=7.8)
                titles.append((tl, tr))
            # shrink the header fonts where the name and the period would overlap
            fig.canvas.draw()
            rend = fig.canvas.get_renderer()
            for tl, tr in titles:
                size = 7.8
                while size > 5.8 and tl.get_window_extent(rend).x1 + 4 > tr.get_window_extent(rend).x0:
                    size -= 0.2
                    tl.set_fontsize(size)
                    tr.set_fontsize(size)
            out = os.path.join(FIG, f"fig_lc_{k:02d}.pdf")
            fig.savefig(out, bbox_inches=None)
            fig.savefig(out[:-4] + ".png", dpi=110, bbox_inches=None)
            plt.close(fig)
            print("wrote", out, "with", nums[0], "-", nums[-1])
    for old in range(len(pages) + 1, 12):        # remove pages of the former four-per-page layout
        for ext in (".pdf", ".png"):
            f = os.path.join(FIG, f"fig_lc_{old:02d}{ext}")
            if os.path.exists(f):
                os.remove(f)
    return amps, marks, [(p[0], p[-1]) for p in pages]


def exact_power(t, y, f, chunk=20000):
    """nterms=2 Lomb-Scargle power with exact trigonometric sums (fastchi2 without the FFT approximation, which is
    unreliable for small datasets), computed in chunks of a regular frequency grid."""
    from astropy.timeseries import LombScargle
    ls = LombScargle(t, y, nterms=2)
    out = np.empty(len(f))
    for idx in np.array_split(np.arange(len(f)), max(1, int(np.ceil(len(f) / chunk)))):
        out[idx] = ls.power(f[idx], method="fastchi2", assume_regular_frequency=True, method_kwds={"use_fft": False})
    return out


def fig_single_dataset(num=None, ds=None):
    """Separate-dataset reduction of one dataset (Eqs. 3-5 of the paper): magnitudes and times referred to the
    first observation, linear phase coefficient from alpha > 7 deg, observations at alpha < 7 deg excluded,
    3-sigma (MAD) clipping. The example is the clearest single-dataset periodogram among the quality-A asteroids
    (search_single_dataset_example.py)."""
    from astropy.timeseries import LombScargle
    import pandas as pd
    num, ds = (num, ds) if num else SINGLE_EXAMPLE
    obs = pd.read_csv(os.path.join(ROOT, "data", "obs_tables", f"{num}_obs.csv"))
    d = obs[(obs["dataset"] == ds) & (obs["alpha"] > 7.0)].sort_values("jd")
    D0, r0, a0 = d["delta"].iloc[0], d["r"].iloc[0], d["alpha"].iloc[0]
    m_r = d["mag_f"] - 5 * np.log10((d["delta"] / D0) * (d["r"] / r0))
    k_lin = np.polyfit(d["alpha"], m_r, 1)[0]
    y = (m_r - (d["alpha"] - a0) * k_lin).to_numpy()
    t_abs = (d["jd"] + (D0 - d["delta"]) / C_AU_PER_DAY).to_numpy()
    keep0 = np.abs(y - np.median(y)) <= 3 * 1.4826 * np.median(np.abs(y - np.median(y)))
    y, t_abs = y[keep0], t_abs[keep0]
    t = t_abs - t_abs[0]
    ls = LombScargle(t, y, nterms=2)
    f = ls.autofrequency(samples_per_peak=10, minimum_frequency=24 / 240.0, maximum_frequency=24 / 0.8)
    p = exact_power(t, y, f)
    chk = np.linspace(0, len(f) - 1, 7).astype(int)
    assert np.allclose(p[chk], ls.power(f[chk], method="chi2"), atol=1e-6), "exact sums disagree with chi2"
    P = 24 / f
    Pad = ADOPTED[str(num)]["P"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.5), gridspec_kw={"width_ratios": [1.25, 1]},
                                   constrained_layout=True)
    ax1.plot(P, p, lw=0.5, color="black")
    j = np.where(np.abs(P - Pad) <= 0.003 * Pad)[0]
    k = j[np.argmax(p[j])]
    ax1.scatter([P[k]], [p[k]], s=18, color="#c0392b", zorder=3, lw=0)
    ax1.set_xscale("log")
    ax1.set_xlim(0.8, 240)
    ax1.set_xticks([1, 2, 5, 10, 20, 50, 100, 200])
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax1.xaxis.set_minor_formatter(NullFormatter())
    ax1.set_ylim(0, 1.15 * p.max())
    ax1.set_xlabel("Period (h)")
    ax1.set_ylabel("LS power")
    ds_name = f"TESS ({ds})" if ds.startswith("C57") else ds
    ax1.set_title(f"{label(num)}, {ds_name} alone ({len(t)} measurements)", loc="left")
    jh = np.where(np.abs(P - P[k] / 2) <= 0.003 * P[k] / 2)[0]      # label the harmonic P/2 of the same solution
    if jh.size and p[jh].max() > 0.3 * p[k]:
        kh = jh[np.argmax(p[jh])]
        ax1.annotate("$P/2$", (P[kh], p[kh]), xytext=(-4, 2), textcoords="offset points", ha="right", fontsize=7)
    model, _, _ = fourier(t, y, float(P[k]), 2)
    phase = (t * 24.0 / float(P[k])) % 1.0
    c, mm, e, amp, amp_err, keep = binned(phase, y, model)
    _, phg, mg = fourier(t[keep], y[keep], float(P[k]), PLOT_NTERMS)
    quant = ds in quantized(num)
    if quant:      # 0.1-mag steps: individual points would form streaks; show the binned means only
        lo, hi = float(np.min(mm - 3 * e)), float(np.max(mm + 3 * e))
    else:
        ax2.scatter(phase[keep], y[keep], s=3, color="0.45", alpha=0.7, lw=0)
        lo, hi = np.percentile(y[keep], [0.5, 99.5])
    ax2.plot(phg, mg, color=MODEL_COLOR, lw=1.0)
    ax2.errorbar(c, mm, yerr=e, fmt="s", ms=2.8, color="black", elinewidth=0.6, capsize=0)
    ax2.set_ylim(hi + 0.1 * (hi - lo), lo - 0.1 * (hi - lo))
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Rotational phase")
    ax2.set_ylabel(r"$m_{\alpha}$ (mag)")
    ax2.set_title(f"P = {P[k]:.4f} h", loc="right", fontsize=7)
    out = os.path.join(FIG, "fig_single_dataset.pdf")
    fig.savefig(out)
    fig.savefig(out[:-4] + ".png", dpi=150)
    plt.close(fig)
    for old in ("fig_3858_T08o.pdf", "fig_3858_T08o.png"):
        if os.path.exists(os.path.join(FIG, old)):
            os.remove(os.path.join(FIG, old))
    print("wrote", out)
    from scipy.signal import find_peaks
    pk, _ = find_peaks(p)
    peaks = []
    for i in pk[np.argsort(p[pk])[::-1]]:
        if all(abs(P[i] - q) > 0.02 * q for q, _ in peaks):
            peaks.append((float(P[i]), float(p[i])))
        if len(peaks) == 6:
            break
    return {"asteroid": num, "dataset": ds, "P_single": float(P[k]), "power_single": float(p[k]), "n": int(len(t)),
            "k_lin": float(k_lin), "strongest_peaks": peaks, "jd_first": float(t_abs[0]),
            "baseline_yr": float(t.max() / 365.25), "amplitude": amp, "quantized_0.1mag": bool(quant)}


def fig_pipeline(num=2607, comparison="G96V"):
    """Four panels: (a) reference data before/after distance correction, (b) reference and
    comparison dataset with their HG1G2 fits, (c) all datasets shifted to the reference phase curve,
    (d) after removal of the phase dependence."""
    from Asteroid_pickle import DatasetGenerator
    _, r = load(num)
    used = r["datasets_used"]
    ref = r["reference"]
    gen = DatasetGenerator(os.path.join(ROOT, "data", "workbooks") + os.sep, f"{num}-mpc.xlsx", num, used, base_dir=REPO)
    import pandas as pd

    def sheet(s):
        df = pd.read_excel(gen.path + gen.file_name, sheet_name=s).dropna(subset=["magred"])
        mag = df["mag"].to_numpy(float)
        ph = df["Ph"].to_numpy(float)
        Hr = mag - 5 * np.log10(df.iloc[:, 3].to_numpy(float) * df.iloc[:, 4].to_numpy(float))
        Hr = Hr + gen.filter_bias.get(s[-1], 0.0)
        _, _, rm = gen.removeOutliers(ph, Hr, 1.8)
        keep = np.ones(len(ph), bool)
        keep[rm] = False
        return ph, mag, Hr, keep

    H, G1, G2 = r["reference_fit"]["H"], r["reference_fit"]["G1"], r["reference_fit"]["G2"]
    cmap = plt.get_cmap("tab20")
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 6.2), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d = axs.ravel()
    ph, mag, Hr, keep = sheet(ref)
    ax_a.scatter(ph, mag, s=4, color="0.65", lw=0, label=f"{ref}: observed magnitude")
    ax_a.scatter(ph[keep], Hr[keep], s=4, color="#1f4e79", lw=0, label=f"{ref}: reduced to 1 AU, band-corrected")
    ax_a.scatter(ph[~keep], Hr[~keep], s=12, marker="x", color="#c0392b", lw=0.8, label="outliers")
    grid = np.linspace(0, max(ph) + 2, 200)
    ax_b.scatter(ph[keep], Hr[keep], s=4, color="#1f4e79", lw=0, label=f"{ref} (reference)")
    ax_b.plot(grid, gen.hg1g2_phase_function(grid, H, G1, G2), color="black", lw=1.0,
              label=f"HG$_1$G$_2$: H={H:.2f}, G$_1$={G1:.2f}, G$_2$={G2:.2f}")
    if comparison in used:
        phc, _, Hc, kc = sheet(comparison)
        fc = gen.fit(phc[kc], Hc[kc], method="HG1G2", G1=G1, G2=G2)
        Hcf = float(fc.params["H"].value)
        ax_b.scatter(phc[kc], Hc[kc], s=5, color="#e67e22", lw=0, label=f"{comparison} before shift")
        ax_b.plot(grid, gen.hg1g2_phase_function(grid, Hcf, G1, G2), color="#e67e22", lw=1.0,
                  label=f"{comparison} fit, H={Hcf:.2f} (G$_1$, G$_2$ fixed)")
    for i, s in enumerate(used):
        phs, _, Hs, ks = sheet(s)
        fs = gen.fit(phs[ks], Hs[ks], method="HG1G2", G1=G1, G2=G2)
        Hs_fit = float(fs.params["H"].value)
        shifted = Hs[ks] + (H - Hs_fit)
        flat = shifted - (gen.hg1g2_phase_function(phs[ks], Hs_fit, G1, G2) - Hs_fit)
        ax_c.scatter(phs[ks], shifted, s=3, color=cmap(i % 20), lw=0, label=s)
        ax_d.scatter(phs[ks], flat, s=3, color=cmap(i % 20), lw=0, label=s)
    for ax, lab in zip((ax_a, ax_b, ax_c, ax_d), "abcd"):
        ax.invert_yaxis()
        ax.set_xlabel(r"Solar phase angle $\alpha$ (deg)")
        ax.set_ylabel("Magnitude")
        ax.set_title(f"({lab})", loc="left")
        ax.minorticks_on()
    ax_a.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.07), fontsize=6, markerscale=2.0, ncol=1)
    ax_b.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.07), fontsize=6, markerscale=2.0, ncol=1)
    ax_d.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=6, markerscale=2.5)
    out = os.path.join(FIG, "fig_pipeline_2607.pdf")
    fig.savefig(out)
    fig.savefig(out[:-4] + ".png", dpi=150)
    plt.close(fig)
    print("wrote", out)


def fmt_short(P):
    return f"{P:.2f}" if P < 10 else (f"{P:.1f}" if P < 100 else f"{P:.0f}")


def fig_dataset_power(order):
    """Rows: asteroids; columns: dataset types retained for at least five asteroids. Colour: exact-chi2 power at the
    adopted period (maximum within +/-1 %). Text: the candidate period (rebootstrap2.py candidates) preferred by the
    dataset alone, as in the subset checks of the paper (primary-range candidates for adopted periods <= 240 h);
    black if it belongs to the adopted alias family (within 1 % of P, 2P or P/2), red otherwise."""
    from astropy.timeseries import LombScargle
    import rebootstrap2 as r2

    def in_family(Pc, P):
        return any(abs(Pc - r * P) <= 0.01 * Pc for r in (1.0, 2.0, 0.5))

    rows, pref = {}, {}
    for num in order:
        a, r = load(num)
        Pad = ADOPTED[str(num)]["P"]
        fc = 24.0 / Pad
        geo_rb = os.path.join(RES_GEO, str(num), f"{num}_rebootstrap2.json")
        v = json.load(open(geo_rb if os.path.exists(geo_rb) else os.path.join(RES, str(num), f"{num}_rebootstrap2.json"),
                           encoding="utf-8"))
        allc, prim = v["primary"] + v["extended"], v["primary"]
        for ds in np.unique(a["sheet"]):
            m = a["sheet"] == ds
            if np.sum(m) < 30:
                continue
            g = np.linspace(fc * 0.99, fc * 1.01, 3001)
            # exact chi2: the approximate fastchi2 method is unreliable for small datasets
            p = LombScargle(a["t"][m], a["y"][m], nterms=2).power(g, method="chi2")
            rows.setdefault(str(ds), {})[num] = float(np.max(p))
            sub = v["subsets"].get(f"dataset:{ds}")
            if sub is None:          # datasets with fewer than 100 measurements: same exact-chi2 candidate test
                tb, yb = a["t"][m], a["y"][m]
                bb = max(float(tb.max() - tb.min()), 1.0)
                vals = [float(r2.power(tb, yb, r2.grid(c["f"], bb, 3.0, spp=20), exact=True).max()) for c in allc]
                sub = {"preferred_P": allc[int(np.argmax(vals))]["P"],
                       "preferred_primary_P": max(prim, key=lambda c: vals[allc.index(c)])["P"]}
            pp = sub["preferred_primary_P"] if Pad <= 240 else sub["preferred_P"]
            pref.setdefault(str(ds), {})[num] = (pp, in_family(pp, Pad) or (Pad > 240 and in_family(sub["preferred_primary_P"], Pad)))
    counts = {k: len(v) for k, v in rows.items()}
    names = [k for k in sorted(rows, key=lambda k: (-counts[k], k)) if counts[k] >= 5]
    M = np.full((len(order), len(names)), np.nan)
    for j, ds in enumerate(names):
        for i, num in enumerate(order):
            if num in rows[ds]:
                M[i, j] = rows[ds][num]
    fig, ax = plt.subplots(figsize=(7.2, 9.4), constrained_layout=True)
    ax.set_facecolor("0.9")
    from matplotlib.colors import LinearSegmentedColormap
    greens = LinearSegmentedColormap.from_list("greens_light", plt.get_cmap("Greens")(np.linspace(0.0, 0.7, 256)))
    im = ax.imshow(np.ma.masked_invalid(M), aspect="auto", cmap=greens, vmin=0, vmax=1, interpolation="nearest")
    n_agree = n_all = 0
    for i, num in enumerate(order):
        for j, ds in enumerate(names):
            if num in pref.get(ds, {}):
                pp, ok = pref[ds][num]
                n_all += 1
                n_agree += ok
                ax.text(j, i, fmt_short(pp), ha="center", va="center", fontsize=4.6,
                        color="black" if ok else "#b03a2e", fontweight="normal" if ok else "bold")
    ax.set_xticks(range(len(names)))
    # 703bG, G96bG: the parts from 2018 July 1 of the 703G and G96G datasets (zero-point split, zp_split.py)
    ax.set_xticklabels([n[:3] + n[-1] + "'" if len(n) == 5 and n[3] == "b" else n for n in names], rotation=90, fontsize=6.5)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"({n})  no period" if ADOPTED[str(n)].get("status") == "undetermined"
                        else f"({n})  {fmt_short(ADOPTED[str(n)]['P'])} h" for n in order], fontsize=5.6)
    ax.set_xlabel("Dataset (observatory code + band)")
    ax.set_ylabel("Asteroid and adopted period")
    ax.set_xticks(np.arange(-0.5, len(names)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(order)), minor=True)
    ax.grid(which="minor", color="white", lw=0.4)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.025, shrink=0.5, location="right")
    cb.set_label("LS power at the adopted period")
    out = os.path.join(FIG, "fig_dataset_power.pdf")
    fig.savefig(out)
    fig.savefig(out[:-4] + ".png", dpi=150)
    plt.close(fig)
    med = {ds: float(np.nanmedian(M[:, j])) for j, ds in enumerate(names)}
    print("wrote", out)
    return {"median_power": med, "n_asteroids": {ds: counts[ds] for ds in names},
            "preferred_in_adopted_family": [n_agree, n_all]}


def fig_period_diameter(order_new, order_known):
    with open(os.path.join(ROOT, "data", "sbdb_rotper_diameter.json"), "r", encoding="utf-8") as fh:
        q = json.load(fh)
    D = np.array([float(x[2]) for x in q["data"]])
    P = np.array([float(x[3]) for x in q["data"]])
    ok = (D > 0) & (P > 0)
    fig, ax = plt.subplots(figsize=(3.5, 3.0), constrained_layout=True)
    ax.scatter(D[ok], P[ok], s=0.6, color="0.7", lw=0, rasterized=True, label="SBDB (LCDB periods)")
    d_line = np.logspace(-3, 3.5, 50)
    ax.plot(d_line, np.full_like(d_line, 2.2), ls="--", color="#c0392b", lw=0.8, label="2.2 h spin barrier")
    for tau, ls_ in ((0.1, ":"), (4.5, "-.")):
        ax.plot(d_line, 17.0 * tau ** (1 / 3) * d_line ** (2 / 3), ls=ls_, color="#6c3483", lw=0.8,
                label=f"NPA damping time {tau:g} Gyr")

    def ours(nums, marker, color, lab, quality_c=False):
        xs, ys = [], []
        for n in nums:
            if (ADOPTED[str(n)]["quality"] == "C") != quality_c or ADOPTED[str(n)].get("status") == "undetermined":
                continue
            d = SBDB[str(n)].get("phys_par", [])
            diam = next((float(p["value"]) for p in d if p.get("name") == "diameter"), None)
            if diam is None:
                continue
            xs.append(diam)
            ys.append(ADOPTED[str(n)]["P"])
        if quality_c:
            ax.scatter(xs, ys, s=14, marker=marker, facecolors="white", edgecolors=color, lw=0.8, zorder=5, label=lab)
        else:
            ax.scatter(xs, ys, s=14, marker=marker, color=color, edgecolors="black", lw=0.3, zorder=5, label=lab)
        return len(xs)

    n_new = ours(order_new, "D", "#2471a3", "This work: new periods (quality A, B)")
    n_new_c = ours(order_new, "D", "#2471a3", "This work: new periods (quality C)", quality_c=True)
    n_known = ours(order_known, "o", "#d4ac0d", "This work: known periods")
    n_known += ours(order_known, "o", "#d4ac0d", "_nolegend_", quality_c=True)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.1, 1000)
    ax.set_ylim(1000 * 1.5, 0.5)
    ax.set_xlabel("Diameter (km)")
    ax.set_ylabel("Rotation period (h)")
    ax.legend(frameon=False, fontsize=5.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, markerscale=1.0)
    leg = ax.get_legend()
    leg.legend_handles[0].set_sizes([8.0])      # make the SBDB background marker visible in the legend
    out = os.path.join(FIG, "fig_period_diameter.pdf")
    fig.savefig(out)
    fig.savefig(out[:-4] + ".png", dpi=150)
    plt.close(fig)
    print("wrote", out)
    return {"n_background": int(np.sum(ok)), "n_new_plotted": n_new + n_new_c, "n_new_quality_c": n_new_c,
            "n_known_plotted": n_known}


def main():
    table34 = [int(k) for k, v in ADOPTED.items() if v.get("table") in (3, 4)]
    table2 = [int(k) for k, v in ADOPTED.items() if v.get("table") == 2]
    table34.sort()
    table2.sort()
    amps, marks, page_ranges = lc_pages(table34)
    info = {"lc_amplitudes": amps, "lc_competing_marks": marks, "lc_page_ranges": page_ranges}
    amps_all = {}
    import amp_data
    import amp_methods
    for num in table2 + table34:
        a, _ = load(num)
        ad = ADOPTED[str(num)]
        model, _, _ = fourier(a["t"], a["y"], ad["P"], int(ad.get("nterms", 2)))
        phase = (a["t"] * 24.0 / ad["P"]) % 1.0
        _, _, _, amp_c, amp_c_err, keep = binned(phase, a["y"], model)
        # light-curve amplitude: per-apparition template fit (amp_methods.template_fit), amplitude-weighted mean over
        # the apparitions; uncertainty = scatter of the apparition amplitudes / sqrt(number of apparitions)
        d = amp_data.load("paper", num)
        d["y"] = a["y"].astype(float)            # geometry-corrected magnitudes where the correction is applied
        tp = amp_methods.template_fit(d)
        if tp is not None:
            g = np.array([r["g"] for r in tp["rows"]])
            w = np.array([r["n"] for r in tp["rows"]], float)
            amp = tp["wmean"]
            amp_err = float(np.sqrt(np.sum(w * (g - amp) ** 2) / np.sum(w)) / np.sqrt(len(g)))
            app = {"n_apparitions": int(len(g)), "app_min": float(g.min()), "app_max": float(g.max())}
        else:
            amp, amp_err, app = amp_c, amp_c_err, {"n_apparitions": 0}
        amps_all[str(num)] = {"amp": amp, "amp_err": amp_err, "amp_combined_fold": amp_c,
                              "n_fold": int(np.sum(keep)), **app}
    info["lc_amplitudes_all"] = amps_all
    info["single_dataset"] = fig_single_dataset()
    # fig_pipeline() (the former Figure 1) is no longer part of the paper: the steps are illustrated in Nagainis & Eglitis
    info["dataset_power"] = fig_dataset_power(sorted(table2 + table34))
    info["period_diameter"] = fig_period_diameter(table34, table2)
    with open(os.path.join(ROOT, "figure_info.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=1)


if __name__ == "__main__":
    main()
