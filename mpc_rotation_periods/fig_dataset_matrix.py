"""
Figure of the individual datasets (fig:dspower; replaces the fig_dataset_power.pdf drawn by make_figures.py): one cell
per asteroid (rows) and dataset type (columns, retained for at least five asteroids). Cell colour: Lomb-Scargle power
at the adopted period. Number: the period that the dataset prefers on its own; its colour gives the kind of period
(adopted period, i.e. within 1 % of P, 2P or P/2; daily alias; commensurate with the day; other; see category()).
Bottom rows: share of the asteroids with an adopted period for which the dataset prefers it, and the number of these
asteroids (review C04: the six asteroids without a period are drawn faded and not counted). Header: survey, station code, band (G' = 703G and G96G
from 2018 July 1 on, separate datasets after the zero-point split).
Drawn at the printed size (full text width, no scaling). Reads dataset_power_table.json (dataset_power_table.py, run
on the LU HPC with hpc/ds_table.sbatch); writes manuscript_revised/fig_dataset_matrix.pdf (+ .png preview) and
fig_dataset_matrix_info.json (counts quoted in the Discussion).
"""
from __future__ import annotations

import json
import os
from collections import Counter

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
FIG = os.environ.get("FIG_DIR", os.path.join(ROOT, "manuscript_revised"))
MIN_AST = 5
W_UNDET = 0.35                       # cell colour weight of the asteroids without an adopted period
TOL = 0.01                           # cycles per day, daily-alias test
W, H = 7.1, 8.25                     # inches, printed size (leaves room for a short caption on the page)
L_IN, R_IN, T_IN, B_IN = 0.98, 0.04, 0.56, 0.60
GAP_C, GAP_R = 0.36, 0.28            # extra space between survey groups / after every fifth row (cell units)
FS = 7.3                             # cell text
FL = 7.6                             # labels
TXT = {"family": ("black", "normal"), "alias": ("#C44E00", "bold"), "day": ("#8B2A6E", "bold"),
       "other": ("#5A5A5A", "bold")}
SURVEY = [("ATLAS", ("T05", "T08", "M22", "W68", "R17")), ("Catalina", ("703", "G96")),
          ("Pan-STARRS", ("F51", "F52")), ("ZTF", ("I41",)), ("TESS", ("C57",)), ("SST", ("G45", "P07")),
          ("PMO", ("D29",))]

plt.rcParams.update({
    # Times-like STIX (bundled with matplotlib): narrower digits than DejaVu Serif, same face as the AASTeX text
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "axes.linewidth": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def category(r, P):
    """family: the flag of dataset_power_table.py (within 1 % of P, 2P or P/2). alias: the dominant (second-harmonic)
    frequency of the light curve, 48/P cycles per day, and that of the preferred period Pp (48/Pp, or 24/Pp if Pp is
    a half period) differ by 1 or 2 cycles per day or add up to 1 or 2 cycles per day (within TOL). day: commensurate
    with the day as defined in the paper (Section sec:combined): the second harmonic 48/Pp lies within TOL of an
    integer number of cycles per day. other: everything else."""
    if r["in_family"]:
        return "family"
    Pp, F2 = r["P_pref"], 48.0 / P
    for d in (48.0 / Pp, 24.0 / Pp):
        for k in (1, 2):
            if abs(abs(d - F2) - k) < TOL or abs(d + F2 - k) < TOL:
                return "alias"
    k = round(48.0 / Pp)
    if k >= 1 and abs(48.0 / Pp - k) < TOL:          # the definition of Section sec:combined
        return "day"
    return "other"


def fmt_short(P):
    return f"{P:.2f}" if P < 10 else (f"{P:.1f}" if P < 100 else f"{P:.0f}")


def band(ds):
    return ds[-1] + ("′" if len(ds) == 5 and ds[3] == "b" else "")


def runs(keys):
    """[key, first index, last index] of consecutive equal keys"""
    out = []
    for j, k in enumerate(keys):
        if out and out[-1][0] == k:
            out[-1][2] = j
        else:
            out.append([k, j, j])
    return out


def chance_rates(A, n=2000, seed=0):
    """fraction of periods drawn log-uniformly over 2-2000 h that pass the alias and day tests"""
    rng = np.random.default_rng(seed)
    c = Counter()
    for a in A.values():
        for Pp in np.exp(rng.uniform(np.log(2), np.log(2000), n)):
            c[category({"in_family": False, "P_pref": Pp}, a["P"])] += 1
    tot = sum(c.values())
    return {k: v / tot for k, v in c.items()}


def main():
    T = json.load(open(os.path.join(ROOT, "dataset_power_table.json"), encoding="utf-8"))
    A = T["asteroids"]
    cells, n_ast = {}, {}
    for r in T["datasets"]:
        r["cat"] = category(r, A[str(r["num"])]["P"])
        cells[(r["num"], r["ds"])] = r
        n_ast.setdefault(r["ds"], set()).add(r["num"])
    keep = {ds for ds, s in n_ast.items() if len(s) >= MIN_AST}
    cols, groups = [], []
    for name, codes in SURVEY:
        sel = sorted((ds for ds in keep if ds[:3] in codes), key=lambda ds: (codes.index(ds[:3]), ds[-1], len(ds)))
        if sel:
            groups.append((name, len(cols), len(cols) + len(sel) - 1))
            cols += sel
    assert set(cols) == keep, keep - set(cols)
    rows = sorted(int(k) for k in A)
    undet = {n for n in rows if A[str(n)].get("status") == "undetermined"}
    gi ={j: g for g, (_, j0, j1) in enumerate(groups) for j in range(j0, j1 + 1)}
    xc = np.array([j + GAP_C * gi[j] for j in range(len(cols))])
    yc = np.array([i + GAP_R * (i // 5) for i in range(len(rows))])

    fig = plt.figure(figsize=(W, H))
    ax_h = H - T_IN - B_IN
    ax = fig.add_axes([L_IN / W, B_IN / H, (W - L_IN - R_IN) / W, ax_h / H])
    cmap = LinearSegmentedColormap.from_list("g", plt.get_cmap("Greens")(np.linspace(0.02, 0.62, 256)))
    norm = Normalize(0, 1)
    for i, num in enumerate(rows):
        for j, ds in enumerate(cols):
            r = cells.get((num, ds))
            fc = cmap(norm(r["power"])) if r else "#EDEDED"
            if r and num in undet:          # candidate period only: faded, excluded from the bottom rows
                fc = tuple(W_UNDET * np.array(fc[:3]) + (1 - W_UNDET) * np.array([0.93, 0.93, 0.93]))
            ax.add_patch(Rectangle((xc[j] - 0.47, yc[i] - 0.44), 0.94, 0.88, facecolor=fc, edgecolor="none"))
            if r:
                col, wt = TXT[r["cat"]] if num not in undet else ("0.45", "normal")
                ax.text(xc[j], yc[i] + 0.04, fmt_short(r["P_pref"]), ha="center", va="center", fontsize=FS,
                        color=col, fontweight=wt)
    # bottom rows: share of the asteroids with an adopted period for which the dataset prefers it, and their number
    yb = yc[-1] + 1.45
    yn = yb + 1.0
    by_type = {}
    for j, ds in enumerate(cols):
        v = [cells[(n, ds)] for n in rows if (n, ds) in cells and n not in undet]
        cnt = Counter(r["cat"] for r in v)
        by_type[ds] = {"n_ast": len(v), **cnt}
        ax.text(xc[j], yb, f"{100 * cnt['family'] / len(v):.0f}%", ha="center", va="center", fontsize=FS,
                color="black")
        ax.text(xc[j], yn, f"{len(v)}", ha="center", va="center", fontsize=FS, color="0.3")
    ax.set_xlim(xc[0] - 0.55, xc[-1] + 0.55)
    ax.set_ylim(yn + 0.6, yc[0] - 0.6)
    ax.set_yticks(list(yc) + [yb, yn])
    ax.set_yticklabels([f"({n})  no period" if n in undet
                        else f"({n})  {fmt_short(A[str(n)]['P'])} h" for n in rows] + ["adopted period", "asteroids"],
                       fontsize=FL)
    for t, n in zip(ax.get_yticklabels(), rows):
        if n in undet:
            t.set_color("0.45")
    ax.get_yticklabels()[-2].set_fontstyle("italic")
    ax.get_yticklabels()[-1].set_fontstyle("italic")
    ax.set_xticks([])
    ax.tick_params(length=0, pad=2)
    for s in ax.spines.values():
        s.set_visible(False)

    # three header rows: survey, station code, band (heights in inches above the matrix)
    tr = blended_transform_factory(ax.transData, ax.transAxes)

    def yax(inch):
        return 1.0 + inch / ax_h

    for j, ds in enumerate(cols):
        ax.text(xc[j], yax(0.05), band(ds), transform=tr, ha="center", va="bottom", fontsize=FL)
    for code, j0, j1 in runs([ds[:3] for ds in cols]):
        ax.text(0.5 * (xc[j0] + xc[j1]), yax(0.19), code, transform=tr, ha="center", va="bottom", fontsize=FL)
        if j1 > j0:
            ax.plot([xc[j0] - 0.38, xc[j1] + 0.38], [yax(0.175)] * 2, transform=tr, color="0.55", lw=0.5,
                    clip_on=False)
    for name, j0, j1 in groups:
        ax.plot([xc[j0] - 0.45, xc[j1] + 0.45], [yax(0.335)] * 2, transform=tr, color="0.2", lw=0.7, clip_on=False)
        ax.text(0.5 * (xc[j0] + xc[j1]), yax(0.35), name, transform=tr, ha="center", va="bottom", fontsize=FL,
                fontweight="bold")

    # colour bar and key below the matrix
    cax = fig.add_axes([L_IN / W, 0.17 / H, 1.9 / W, 0.075 / H])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cb.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cb.ax.tick_params(labelsize=7, length=2, pad=1)
    cb.outline.set_linewidth(0.4)
    fig.text(L_IN / W, 0.28 / H, "Power at the adopted period", fontsize=FL, va="bottom")
    key = [("family", "adopted family ($P$, $2P$ or $P/2$)"), ("alias", "daily alias"),
           ("day", "commensurate with the day"), ("other", "other period")]
    x0 = (L_IN + 2.2) / W
    for k, (cat, text) in enumerate(key):
        xx = x0 + (k // 2) * 1.95 / W
        yy = (0.34 - 0.17 * (k % 2)) / H
        fig.patches.append(Rectangle((xx, yy - 0.045 / H), 0.2 / W, 0.09 / H, transform=fig.transFigure,
                                     facecolor=TXT[cat][0], edgecolor="none"))
        fig.text(xx + 0.3 / W, yy, text, fontsize=FL, va="center")
    fig.text(x0, 0.28 / H + 0.17 / H, "Colour of the preferred period", fontsize=FL, va="bottom")
    out = os.path.join(FIG, "fig_dataset_matrix.pdf")
    fig.savefig(out, dpi=300)          # no bbox="tight": the page size is the printed size
    fig.savefig(out[:-4] + ".png", dpi=200)
    plt.close(fig)
    allc = Counter(r["cat"] for (n, ds), r in cells.items() if ds in keep)
    resc = Counter(r["cat"] for (n, ds), r in cells.items() if ds in keep and n not in undet)
    info = {"n_datasets": sum(allc.values()), "categories": dict(allc), "n_datasets_resolved": sum(resc.values()),
            "categories_resolved": dict(resc), "n_datasets_all_types": len(cells),
            "chance_rates": chance_rates({k: a for k, a in A.items() if int(k) not in undet}), "by_type": by_type}
    with open(os.path.join(ROOT, "fig_dataset_matrix_info.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=1)
    print("wrote", out, len(rows), "rows", len(cols), "columns", info["n_datasets"], "cells", dict(allc),
          {k: round(v, 3) for k, v in info["chance_rates"].items()})


if __name__ == "__main__":
    main()
