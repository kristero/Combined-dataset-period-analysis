"""
Figure for the convex-vs-ellipsoid comparison (review/damit_comparison.png):
 (a) distance to published poles (axis, as in the paper) for E1, C, E2 and random axes
 (b) leave-one-apparition-out errors of the per-apparition amplitude g_j and mean level mu_j
 (c) brightness correction before the period search: power at P (validation), E1 vs C
 (d,e) example: two apparitions of one asteroid with the C and E2 model lightcurves
Usage: python damit_figures.py [example_run]
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
RUNS = os.path.join(ROOT, "damit_runs")
COL = {"E1": "#b9770e", "C": "#1a5276", "E2": "#7d3c98", "null": "#7f8c8d", "rand": "#bbbbbb"}
plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12})


def load_evals():
    E = {}
    for d in sorted(os.listdir(RUNS)):
        p = os.path.join(RUNS, d, "eval.json")
        if os.path.exists(p):
            r = json.load(open(p))
            E[r["key"]] = r
    return E


def example_panels(axes, run):
    import damit_eval as V
    import damit_model as M
    info = json.load(open(os.path.join(run, "prep.json")))
    ev = json.load(open(os.path.join(run, "eval.json")))
    prep = dict(np.load(os.path.join(run, "prep.npz"), allow_pickle=True))
    sh = prep["sheet"].astype(str)
    pj = json.load(open(os.path.join(run, "poles", "poles.json")))
    best = pj.get("best") or min(pj["solutions"], key=lambda r: r["dev"])
    mC, _ = V.convex_model(run, "poles", V.sol_index(best))
    ell = json.load(open(os.path.join(run, "ellipsoid.json")))
    mE = V.ellipsoid_model([ell["best"][k] for k in ell["names"]], ell["jd0"])
    H, jd = prep["H_red"], prep["jd_lt"]
    mi = -2.5 * np.log10(mC.brightness(jd, prep["sun"], prep["earth"]))
    Z = {s: np.mean((H - mi)[sh == s]) for s in np.unique(sh)}
    yd = H - np.array([Z[s] for s in sh])
    # apparitions: gap > 60 d; pick the two with the most points and the largest amplitude contrast
    o = np.argsort(jd)
    cuts = np.where(np.diff(jd[o]) > 60)[0]
    groups = [g for g in np.split(o, cuts + 1) if len(g) >= 60]
    amps = [np.mean(mC.amplitude(prep["sun"][g[::max(1, len(g) // 5)]], prep["earth"][g[::max(1, len(g) // 5)]])) for g in groups]
    gi = [int(np.argmin(amps)), int(np.argmax(amps))]
    P, t0 = mC.par["P"], mC.par["t0"]
    for ax, k in zip(axes, gi):
        g = groups[k]
        ph = ((jd[g] - t0) * 24 / P) % 1
        ax.plot(ph, yd[g], ".", ms=2.5, color="#e0a96d", alpha=0.6, rasterized=True)
        tm = np.median(jd[g])
        tt = tm + np.linspace(0, P / 24, 300)
        sun = np.repeat(prep["sun"][g].mean(0)[None], 300, 0)
        ear = np.repeat(prep["earth"][g].mean(0)[None], 300, 0)
        php = ((tt - t0) * 24 / P) % 1
        srt = np.argsort(php)
        for mod, name in ((mC, "C"), (mE, "E2")):
            mm = -2.5 * np.log10(mod.brightness(tt, sun, ear))
            mm = mm - mm.mean() + np.median(yd[g])
            ax.plot(php[srt], mm[srt], "-", color=COL[name], lw=2.2 if name == "C" else 1.6,
                    ls="-" if name == "C" else "--", label="convex (DAMIT)" if name == "C" else "ellipsoid, same fit")
        yr = np.percentile(yd[g], [2, 98])
        ax.set_ylim(yr[1] + 0.1, yr[0] - 0.1)
        ax.set_xlim(0, 1)
        y0 = jd[g].min()
        from astropy.time import Time
        ax.set_title(f"({info['num']}) {Time(np.median(jd[g]), format='jd').datetime.year}: model amplitude {amps[k]:.2f} mag", fontsize=11)
        ax.set_xlabel(f"rotational phase (P = {P:.6f} h)")
        ax.set_ylabel("reduced magnitude")
    axes[0].legend(loc="lower left", fontsize=9, frameon=False)


def main(argv):
    S = json.load(open(os.path.join(ROOT, "damit_comparison.json")))
    E = load_evals()
    ex = argv[0] if argv else None
    fig = plt.figure(figsize=(14, 9.5))
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.3)
    # (a) poles
    ax = fig.add_subplot(gs[0, 0])
    rows = S["poles"]["rows"]
    for i, (m, lab) in enumerate((("E1", "ellipsoid aspect fit\n(paper)"), ("C", "convex\n(DAMIT)"), ("E2", "ellipsoid,\npoint level"))):
        v = [r[m]["axis_distance"] for r in rows if m in r and "axis_distance" in r[m]]
        ax.scatter(np.full(len(v), i) + np.random.default_rng(1).uniform(-0.12, 0.12, len(v)), v, s=22,
                   color=COL[m], alpha=0.8)
        ax.hlines(np.median(v), i - 0.3, i + 0.3, color="k", lw=2)
        ax.text(i, 93, f"median {np.median(v):.0f} deg", ha="center", fontsize=9)
    ax.axhline(S["poles"]["random_axis"]["median"], color="grey", ls=":", lw=1.5)
    ax.text(2.45, S["poles"]["random_axis"]["median"] + 1.5, "random", color="grey", fontsize=9, ha="right")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["ellipsoid aspect fit\n(paper)", "convex\n(DAMIT)", "ellipsoid,\npoint level"], fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_ylabel("distance to published pole (deg)")
    ax.set_title(f"(a) spin axes, {sum(1 for r in rows if r['published'])} asteroids")
    # (b) loao errors
    ax = fig.add_subplot(gs[0, 1])
    L = S["windows_loao_common"]
    names = [("null_loao", "no geometry"), ("E1_loao", "ellipsoid\n(paper)"), ("C_loao", "convex"), ("E2_loao", "ellipsoid,\npoint level")]
    x = np.arange(len(names))
    cols = [COL["null"], COL["E1"], COL["C"], COL["E2"]]
    ax.bar(x - 0.2, [L[k]["g_median_abs"] for k, _ in names], 0.4, color=cols, alpha=0.9, label="amplitude $g_j$")
    ax.bar(x + 0.2, [L[k]["mu_median_abs"] for k, _ in names], 0.4, color=cols, alpha=0.45, hatch="//",
           label="mean level $\\mu_j$")
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in names], fontsize=9)
    ax.set_ylabel("median |error| of withheld apparition (mag)")
    ax.set_title(f"(b) withheld apparitions (n = {L['C_loao']['n']})")
    ax.legend(fontsize=9, frameon=False)
    # (c) correction: power at P
    ax = fig.add_subplot(gs[0, 2])
    vals = [(r["periodogram"]["none"]["power_at_P"], r["periodogram"].get("E1", {}).get("power_at_P", np.nan),
             r["periodogram"]["C_full"]["power_at_P"]) for k, r in E.items() if k.startswith("val:")]
    v = np.array(vals)
    ax.scatter(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0], s=14, color=COL["C"], alpha=0.7)
    lim = np.nanmax(np.abs(v[:, 1:] - v[:, :1])) * 1.05
    ax.plot([-lim, lim], [-lim, lim], color="grey", lw=1)
    ax.axhline(0, color="grey", lw=0.6)
    ax.axvline(0, color="grey", lw=0.6)
    ax.set_xlabel("power gain at P, ellipsoid correction")
    ax.set_ylabel("power gain at P, convex correction")
    ax.set_title(f"(c) period search, {len(v)} test asteroids")
    # (d,e) example
    if ex:
        axes = [fig.add_subplot(gs[1, i]) for i in range(2)]
        example_panels(axes, os.path.join(RUNS, ex))
        ax = fig.add_subplot(gs[1, 2])
    else:
        ax = fig.add_subplot(gs[1, :])
    # (f) fit improvement C vs E2
    dC = np.array([r["C"]["dev"] for r in E.values()])
    dE = np.array([r["E2"]["dev"] for r in E.values()])
    ax.hist(100 * (dE / dC - 1), bins=30, color=COL["C"], alpha=0.8)
    ax.set_xlabel("rms of ellipsoid / rms of convex model - 1 (%)")
    ax.set_ylabel("asteroids")
    ax.set_title("(f) fit to all measurements")
    os.makedirs(os.path.join(ROOT, "review"), exist_ok=True)
    out = os.path.join(ROOT, "review", "damit_comparison.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main(sys.argv[1:])
