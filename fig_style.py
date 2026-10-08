"""
Panel versions of the power spectrum and light curve figures of the paper (AsteroidLSPipeline.find_peaks and
visualize_ls), with the same fonts, colors, markers and peak labels, for the multi-panel appendix figures.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, NullFormatter

from asteroid_ls import AsteroidLSPipeline

MARKERS = ['o', 's', '^', 'D', 'v', '>', '<', 'p', '*', 'H', 'X', 'd', 'P', '8']


def set_rc():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6,
        "xtick.major.size": 3, "ytick.major.size": 3, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "savefig.bbox": "tight", "savefig.dpi": 600, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def palette(n):
    """Dataset colors of the light curve figures: the darker part of YlOrBr."""
    cmap = plt.get_cmap("YlOrBr")
    return np.asarray([cmap(0.65)]) if n <= 1 else cmap(np.linspace(0.45, 0.95, n))


def _decimate(period, power, max_points=400000):
    """Block min/max decimation of long spectra for plotting only (as in find_peaks)."""
    if len(period) <= max_points:
        return period, power
    edges = np.linspace(0, len(period), max_points // 2 + 1).astype(int)
    sel = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        seg = power[a:b]
        i_max, i_min = a + int(np.argmax(seg)), a + int(np.argmin(seg))
        sel.append(min(i_min, i_max))
        if i_min != i_max:
            sel.append(max(i_min, i_max))
    sel = np.asarray(sel, dtype=int)
    return period[sel], power[sel]


def spectrum_panel(fig, ax, frequency, power, peak_period, peak_power, peak_label,
                   ticks=(0.5, 1, 2, 5, 10, 20, 50, 100, 240), extra=None, legend_fs=6.5, colsp=2.0):
    """Power spectrum on a log period axis: black line, red peak markers with period labels."""
    period = 24.0 / np.asarray(frequency, float)
    order = np.argsort(period)
    per, pw = _decimate(period[order], np.asarray(power, float)[order])
    ax.plot(per, pw, lw=0.8, color="black", label="Power")
    ax.scatter(peak_period, peak_power, s=14, color="red", zorder=3, label=peak_label)
    if extra is not None:
        extra(ax)
    ax.set_ylim(0, float(np.max(pw)) * 1.40)
    ax.set_xscale("log")
    lo, hi = float(per.min()), float(per.max())
    # the grid ends can fall a hair inside the nominal search limits (0.5 and 240 h)
    tk = [t for t in ticks if lo * (1 - 2e-3) <= t <= hi * (1 + 2e-3)]
    ax.set_xlim(min([lo] + tk), max([hi] + tk))
    ax.set_xticks(tk)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.minorticks_on()
    ax.grid(which="both", axis="y", linestyle=":", linewidth=0.4, alpha=0.6)
    ax.legend(frameon=False, handlelength=1.2, borderpad=0.2, labelspacing=0.3, ncol=3, fontsize=legend_fs,
              columnspacing=colsp, loc="lower right", bbox_to_anchor=(1.0, 1.01))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    AsteroidLSPipeline._place_peak_labels(ax, fig, renderer, ax.get_window_extent(renderer=renderer),
                                          np.asarray(peak_period, float), np.asarray(peak_power, float),
                                          fontsize=6.5)


def light_curve_panel(ax, groups, model_phase_deg, model_mag, box, nterms=4, ncol=4, size=9, legend=True):
    """Folded light curve: one color and marker per dataset, dashed Fourier model, P and A in a box.
    groups: list of (dataset name, phase in degrees, magnitude)."""
    col = palette(len(groups))
    for i, (name, ph, mag) in enumerate(groups):
        ax.scatter(ph, mag, marker=MARKERS[i % len(MARKERS)], s=size, color=col[i], edgecolors="none",
                   alpha=0.9, label=name, zorder=1)
    ax.plot(model_phase_deg, model_mag, "--", lw=0.9, color="black", label=f"n={nterms} model", zorder=2)
    ax.text(0.02, 0.97, box, transform=ax.transAxes, fontsize=6.5, ha="left", va="top", zorder=6,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.5))
    ax.set_xlim(0, 360)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.invert_yaxis()
    ax.minorticks_on()
    ax.grid(which="both", axis="y", linestyle=":", linewidth=0.4, alpha=0.6)
    if legend:
        ax.legend(frameon=False, handlelength=1.0, borderpad=0.2, labelspacing=0.15, columnspacing=0.8,
                  fontsize=5.5, ncol=ncol, loc="lower center", bbox_to_anchor=(0.5, 1.01), markerscale=1.2)
