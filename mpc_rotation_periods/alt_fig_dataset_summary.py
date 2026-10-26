"""
Alternative to the dataset-power figure (Figure 7): one row per dataset type (observatory code + band, retained for at
least five asteroids) instead of one cell per asteroid and dataset.
  (a) number of asteroids with such a dataset, split by the period that the dataset prefers on its own: the adopted
      alias family (as in Figure 7), a daily alias of it, a period commensurate with the day, or another period;
      the percentage is the fraction in the adopted family
  (b) Lomb-Scargle power at the adopted period, one point per asteroid, median marked
  (c) number of measurements per dataset, one point per asteroid, median marked
Reads dataset_power_table.json (dataset_power_table.py, run on the LU HPC); writes alt_figures/fig_dataset_summary.*
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "alt_figures")
TOL = 0.01          # cycles per day, daily-alias test
MIN_AST = 5
CATS = ("family", "alias", "day", "other")

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8.5, "legend.fontsize": 8.5, "axes.linewidth": 0.6,
    "xtick.major.size": 3, "ytick.major.size": 0, "xtick.minor.size": 1.5,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
})

# Okabe-Ito colours (colour-blind safe)
COL = {"family": "#0072B2", "alias": "#E69F00", "day": "#CC79A7", "other": "#BBBBBB"}
LAB = {"family": "adopted alias family", "alias": "daily alias", "day": "commensurate with the day",
       "other": "other period"}
SURVEY = [("ATLAS", ("T05", "T08", "M22", "W68", "R17")), ("Catalina,\nMt. Lemmon", ("703", "G96")),
          ("Pan-STARRS", ("F51", "F52")), ("ZTF", ("I41",)), ("TESS", ("C57",)), ("SST", ("G45", "P07")),
          ("Purple\nMountain", ("D29",))]
DAY_P = [48.0 / n for n in range(1, 13)] + [72.0, 96.0]


def category(r, P):
    """family: the flag of Figure 7 (within 1 % of P, 2P or P/2). alias: the dominant (second-harmonic) frequency of
    the light curve, 48/P cycles per day, and that of the preferred period Pp (48/Pp, or 24/Pp if Pp is a half
    period) differ by 1 or 2 cycles per day or add up to 1 or 2 cycles per day (within TOL). day: Pp within 1 % of
    48/n h (n = 1-12), 72 h or 96 h."""
    if r["in_family"]:
        return "family"
    Pp, F2 = r["P_pref"], 48.0 / P
    for d in (48.0 / Pp, 24.0 / Pp):
        for k in (1, 2):
            if abs(abs(d - F2) - k) < TOL or abs(d + F2 - k) < TOL:
                return "alias"
    if any(abs(Pp - q) <= 0.01 * q for q in DAY_P):
        return "day"
    return "other"


def label(ds):
    stn, band = ds[:3], ds[-1]
    if len(ds) == 5 and ds[3] == "b":
        return f"{stn} {band} (2018–)"
    if stn in ("703", "G96") and band == "G":
        return f"{stn} {band} (–2018)"
    return f"{stn} {band}"


def strip(ax, by, order, y, key, rng):
    for ds in order:
        v = by[ds]
        x = np.array([r[key] for r in v])
        jit = rng.uniform(-0.24, 0.24, len(v))
        ax.scatter(x, y[ds] + jit, s=8, c=[COL[r["cat"]] for r in v], lw=0, alpha=0.9, zorder=3)
        ax.plot([np.median(x)] * 2, [y[ds] - 0.38, y[ds] + 0.38], color="k", lw=1.5, zorder=4)


def main():
    T = json.load(open(os.path.join(ROOT, "dataset_power_table.json"), encoding="utf-8"))
    A = T["asteroids"]
    by = defaultdict(list)
    for r in T["datasets"]:
        r["cat"] = category(r, A[str(r["num"])]["P"])
        by[r["ds"]].append(r)
    keep = {ds for ds, v in by.items() if len({r["num"] for r in v}) >= MIN_AST}
    rows_all = [r for ds in keep for r in by[ds]]
    c = Counter(r["cat"] for r in rows_all)
    print(f"{len(rows_all)} datasets of {len(keep)} types; {dict(c)}; outside the family "
          f"{len(rows_all) - c['family']} = {(len(rows_all) - c['family']) / len(rows_all):.3f}")
    # chance rate of the alias and day tests: preferred periods drawn log-uniformly over 2-2000 h
    rng0 = np.random.default_rng(0)
    sim = Counter()
    for a in A.values():
        for Pp in np.exp(rng0.uniform(np.log(2), np.log(2000), 2000)):
            sim[category({"in_family": False, "P_pref": Pp}, a["P"])] += 1
    tot = sum(sim.values())
    print("chance rate for random periods:", {k: round(v / tot, 3) for k, v in sim.items()})
    if "-v" in sys.argv:
        for r in sorted((r for r in rows_all if r["cat"] == "other"), key=lambda r: (r["num"], r["ds"])):
            print("  other", r["num"], r["ds"], f"P={A[str(r['num'])]['P']:.4f}", f"pref={r['P_pref']:.4f}")

    order, groups = [], []
    for name, codes in SURVEY:
        sel = sorted((ds for ds in keep if ds[:3] in codes), key=lambda ds: (codes.index(ds[:3]), ds[-1], len(ds)))
        if sel:
            groups.append((name, len(order), len(order) + len(sel) - 1))
            order += sel
    rest = sorted(keep - set(order))
    if rest:
        groups.append(("Other", len(order), len(order) + len(rest) - 1))
        order += rest
    ny = len(order)
    y = {ds: i for i, ds in enumerate(order)}

    fig = plt.figure(figsize=(7.1, 0.205 * ny + 1.1))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.2, 0.8], wspace=0.08, left=0.265, right=0.985, top=0.89,
                          bottom=0.09)
    axa, axb, axc = (fig.add_subplot(gs[0, i]) for i in range(3))
    for ax in (axa, axb, axc):
        ax.set_ylim(ny - 0.4, -0.6)
        for k, (name, i0, i1) in enumerate(groups):
            if k % 2 == 0:
                ax.axhspan(i0 - 0.5, i1 + 0.5, color="0.94", zorder=0, lw=0)
        ax.set_yticks(range(ny))
        ax.tick_params(axis="y", length=0)
        ax.spines[["top", "right"]].set_visible(False)
    axa.set_yticklabels([label(ds) for ds in order])
    for ax in (axb, axc):
        ax.set_yticklabels([])
    for name, i0, i1 in groups:
        axa.annotate(name, xy=(0.0, 0.5 * (i0 + i1)), xycoords=("figure fraction", "data"), xytext=(2, 0),
                     textcoords="offset points", ha="left", va="center", fontsize=8.5, fontweight="bold",
                     linespacing=1.0)

    # (a) number of asteroids by category of the preferred period
    for ds in order:
        n = Counter(r["cat"] for r in by[ds])
        left = 0
        for cat in CATS:
            axa.barh(y[ds], n[cat], left=left, height=0.7, color=COL[cat], lw=0, zorder=2)
            left += n[cat]
        axa.text(left + 0.8, y[ds], f"{100 * n['family'] / len(by[ds]):.0f}%", va="center", ha="left", fontsize=8,
                 color=COL["family"])
    axa.set_xlim(0, 67)
    axa.set_xticks([0, 20, 40, 57])
    axa.set_xlabel("Number of asteroids")
    axa.set_title("(a) Preferred period", loc="left")

    rng = np.random.default_rng(1)
    strip(axb, by, order, y, "power", rng)
    axb.set_xlim(0, 1)
    axb.set_xlabel("Lomb–Scargle power")
    axb.set_title("(b) Power at $P$", loc="left")

    strip(axc, by, order, y, "n", rng)
    axc.set_xscale("log")
    axc.set_xlim(25, 7000)
    axc.set_xlabel("Measurements")
    axc.set_title("(c) Dataset size", loc="left")

    handles = [Patch(color=COL[k], label=LAB[k]) for k in CATS]
    handles.append(plt.Line2D([], [], color="k", lw=1.5, label="median"))
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.6, 1.0),
               handlelength=1.1, handletextpad=0.5, columnspacing=1.2)
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, "fig_dataset_summary.pdf")
    fig.savefig(out)
    fig.savefig(out[:-4] + ".png", dpi=200)
    print("wrote", out)
    stats = {ds: {"n_ast": len(by[ds]), **Counter(r["cat"] for r in by[ds]),
                  "median_power": float(np.median([r["power"] for r in by[ds]])),
                  "median_n": float(np.median([r["n"] for r in by[ds]]))} for ds in order}
    json.dump({"tol_cpd": TOL, "categories": dict(c), "n_datasets": len(rows_all), "chance_rates": dict(sim),
               "by_type": stats},
              open(os.path.join(OUT, "fig_dataset_summary_info.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
