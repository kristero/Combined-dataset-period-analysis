"""
Figures documenting the test failure of (870) Manto (adopted 171.72 h, quality B; published 122.3 h, sidereal 122.166 h).
All values are read from the pipeline outputs; nothing is fitted to the literature. Writes review/manto/fig*.png and
review/manto/manto_numbers.json.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.timeseries import LombScargle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "review", "manto")
os.makedirs(OUT, exist_ok=True)
J = lambda *p: json.load(open(os.path.join(ROOT, *p), encoding="utf-8"))
N = "870"
P_ADOPT, P_LIT_FAM, P_SID_PUB = 171.7215, 122.648, 122.166
C_AD, C_LIT, C_OTHER, C_TESS, C_GROUND = "#b03a2e", "#1f6f8b", "0.55", "#2e86c1", "0.25"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 110, "savefig.bbox": "tight"})
NUM = {}

geo = np.load(os.path.join(ROOT, "results_geo", N, f"{N}_arrays.npz"), allow_pickle=True)
t, y, sheet = geo["t"].astype(float), geo["y"].astype(float), geo["sheet"].astype(str)
t0 = float(geo["time_zero"])
tess = np.array([s.startswith("C57") for s in sheet])
rb2 = J("results_geo", N, f"{N}_rebootstrap2.json")
res = J("results", N, f"{N}_result.json")


def mark(ax, label=True):
    for P, c, ls, lab in ((P_ADOPT, C_AD, "-", "adopted family (171.72 h, 85.86 h)"), (P_ADOPT / 2, C_AD, "-", None),
                          (P_LIT_FAM, C_LIT, "-", "literature family (122.65 h, 61.32 h)"), (P_LIT_FAM / 2, C_LIT, "-", None)):
        ax.axvline(P, color=c, ls=ls, lw=0.8, alpha=0.6, label=lab if label else None)


def ls_power(tt, yy, Pmin=40.0, Pmax=260.0, dy=None):
    T = tt.max() - tt.min()
    f = np.arange(24.0 / Pmax, 24.0 / Pmin, 1.0 / (10 * T))
    p = LombScargle(tt, yy, dy=dy, nterms=2).power(f, method="fastchi2")
    return 24.0 / f, p


# 1. combined periodogram (stored by the pipeline: aspect-corrected data)
P, pw, pww = geo["period"], geo["power"], geo["power_weighted"]
fig, axs = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True)
for ax, p, lab in ((axs[0], pw, "unweighted"), (axs[1], pww, "variance-weighted")):
    ax.plot(P, p, color="k", lw=0.5)
    mark(ax, label=(ax is axs[0]))
    ax.set_ylabel(f"power ({lab})")
    ax.axhline(float(geo["threshold"]), color="0.6", ls=":", lw=0.8)
axs[0].legend(loc="upper left", fontsize=7.5, frameon=False)
axs[1].set_xscale("log")
axs[1].set_xlim(20, 1500)
axs[1].set_xlabel("period (h)")
axs[0].set_title("(870) Manto: combined periodogram of the aspect-corrected MPC data (8541 measurements)")
fig.savefig(os.path.join(OUT, "fig01_periodogram.png"))
plt.close(fig)

# 2. subsets: TESS only, all but TESS, longitude groups
import importlib.util
spec = importlib.util.spec_from_file_location("mr", os.path.join(ROOT, "zp", "mpc_redo.py"))
src = open(os.path.join(ROOT, "zp", "mpc_redo.py"), encoding="utf-8").read()
LON = eval(src.split("LON_GROUP = ", 1)[1].split("\n}", 1)[0] + "\n}")
grp = np.array([next((g for g, l in LON.items() if s[:3] in l), "other") for s in sheet])
subs = [("TESS only (C57G)", tess, "dataset:C57G"), ("all data without TESS", ~tess, "without:C57G")] + \
       [(f"longitude group: {g.replace('_', '/')}", grp == g, k) for g, k in
        (("americas", "group:americas_hawaii"), ("europe_africa", "group:europe_africa"), ("asia_australia", "group:asia_australia"))]
fig, axs = plt.subplots(len(subs), 1, figsize=(8, 9.5), sharex=True, constrained_layout=True)
sub_best = {}
for ax, (lab, m, key) in zip(axs, subs):
    Pp, pp = ls_power(t[m], y[m])
    ax.plot(Pp, pp, color="k", lw=0.6)
    mark(ax, label=(ax is axs[0]))
    best = Pp[np.argmax(pp)]
    pa = pp[np.argmin(np.abs(Pp - P_ADOPT))]
    pl = max(pp[np.argmin(np.abs(Pp - P_LIT_FAM))], pp[np.argmin(np.abs(Pp - P_LIT_FAM / 2))])
    pref = (rb2["subsets"].get(key) or {}).get("preferred_primary_P")
    sub_best[lab] = {"n": int(m.sum()), "best_P": float(best), "power_171.7": float(pa), "power_lit_family": float(pl),
                     "pipeline_preferred_candidate": pref}
    ax.set_title(f"{lab}: n = {m.sum()}; strongest peak {best:.2f} h; among the candidates the pipeline's subset test "
                 f"selects {pref:.2f} h" if pref else f"{lab}: n = {m.sum()}; strongest peak {best:.2f} h", fontsize=8.5, loc="left")
    ax.set_ylabel("power")
axs[0].legend(loc="upper right", fontsize=7, frameon=False)
axs[-1].set_xlabel("period (h)")
fig.savefig(os.path.join(OUT, "fig02_subsets.png"))
plt.close(fig)
NUM["subsets"] = sub_best

# 3. folded light curves at both periods
def fourier(tt, yy, Pq, n=4):
    w = 2 * np.pi * 24.0 / Pq
    A = np.vstack([np.ones_like(tt)] + [f(k * w * tt) for k in range(1, n + 1) for f in (np.sin, np.cos)]).T
    c, *_ = np.linalg.lstsq(A, yy, rcond=None)
    ph = np.linspace(0, 1, 400)
    G = np.vstack([np.ones_like(ph)] + [f(2 * np.pi * k * ph) for k in range(1, n + 1) for f in (np.sin, np.cos)]).T
    return ph, G @ c, float(np.std(yy - A @ c))


fig, axs = plt.subplots(2, 2, figsize=(9.5, 7), sharey=True, constrained_layout=True)
for j, (Pq, col, name) in enumerate(((P_ADOPT, C_AD, "171.72 h"), (P_LIT_FAM, C_LIT, "122.65 h"))):
    for i, (m, lab) in enumerate(((~tess, "ground-based"), (tess, "TESS"))):
        ax = axs[i, j]
        ph = (t[m] * 24.0 / Pq) % 1.0
        ax.scatter(ph, y[m], s=1.5, color=C_GROUND if i == 0 else C_TESS, alpha=0.25, lw=0)
        bins = np.linspace(0, 1, 26)
        k = np.digitize(ph, bins) - 1
        mb = [np.median(y[m][k == b]) if np.sum(k == b) > 5 else np.nan for b in range(25)]
        ax.plot(0.5 * (bins[1:] + bins[:-1]), mb, "o-", color=col, ms=3, lw=1)
        ph4, f4, rms = fourier(t[m], y[m], Pq)
        ax.plot(ph4, f4, color="k", lw=0.8)
        ax.set_title(f"{lab} data at {name}: rms {rms:.3f} mag", fontsize=9)
        ax.set_xlabel("rotational phase")
        NUM.setdefault("fold_rms", {})[f"{lab} {Pq:.2f}"] = rms
axs[0, 0].set_ylabel("reduced magnitude")
axs[1, 0].set_ylabel("reduced magnitude")
axs[0, 0].invert_yaxis()
fig.suptitle("(870) Manto: aspect-corrected data folded at the adopted (left) and the literature-family (right) period")
fig.savefig(os.path.join(OUT, "fig03_folds.png"))
plt.close(fig)

# 4. bootstrap family fractions
prim = rb2["primary"]
fams = {}
for c in prim:
    f = c["family"]
    if f not in fams or c["power"] > fams[f]["power"]:
        fams[f] = c
fl = sorted(fams.values(), key=lambda c: -max(c["famwin_u"], c["famwin_w"]))[:6]
fig, ax = plt.subplots(figsize=(7, 3.2))
x = np.arange(len(fl))
ax.bar(x - 0.2, [c["famwin_u"] for c in fl], 0.4, label="unweighted", color="0.35")
ax.bar(x + 0.2, [c["famwin_w"] for c in fl], 0.4, label="variance-weighted", color="0.7")
ax.set_xticks(x)
ax.set_xticklabels([f"{c['P']:.2f} h" for c in fl])
for xi, c in zip(x, fl):
    col = C_AD if abs(c["P"] - P_ADOPT) / P_ADOPT < 0.02 or abs(c["P"] - P_ADOPT / 2) / P_ADOPT < 0.02 else \
        (C_LIT if abs(c["P"] - P_LIT_FAM) / P_LIT_FAM < 0.02 or abs(c["P"] - P_LIT_FAM / 2) / P_LIT_FAM < 0.02 else "k")
    ax.get_xticklabels()[xi].set_color(col)
ax.set_ylabel("fraction of 300 bootstrap samples")
ax.set_title("(870) Manto: bootstrap of observatory nights, share won by each period family")
ax.legend(frameon=False, fontsize=8)
fig.savefig(os.path.join(OUT, "fig04_bootstrap.png"))
plt.close(fig)
NUM["bootstrap"] = [{"P": c["P"], "famwin_u": c["famwin_u"], "famwin_w": c["famwin_w"]} for c in fl]

# 5. time coverage and TESS visits
jd = t + t0
yr = 2000 + (jd - 2451544.5) / 365.25
fig, axs = plt.subplots(2, 1, figsize=(8.5, 5.4), gridspec_kw={"height_ratios": [1, 1.4]}, constrained_layout=True)
axs[0].scatter(yr[~tess], np.zeros((~tess).sum()), s=4, marker="|", color=C_GROUND, label="ground-based")
axs[0].scatter(yr[tess], np.ones(tess.sum()) * 0.3, s=4, marker="|", color=C_TESS, label="TESS")
axs[0].set_yticks([])
axs[0].set_ylim(-0.3, 0.6)
axs[0].set_xlabel("year")
axs[0].legend(frameon=False, fontsize=8, loc="upper left")
axs[0].set_title(f"(870) Manto: {(~tess).sum()} ground-based and {tess.sum()} TESS measurements (55 %)", fontsize=9)
cuts = np.where(np.diff(np.sort(jd[tess])) > 15)[0]      # visits as in the pipeline (sector gaps of a few days kept)
tj = np.sort(jd[tess])
starts, ends = np.r_[tj[0], tj[cuts + 1]], np.r_[tj[cuts], tj[-1]]
visits = []
for k, (a, b) in enumerate(zip(starts, ends)):
    m = tess & (jd >= a) & (jd <= b)
    visits.append({"start_jd": float(a), "days": float(b - a), "n": int(m.sum()),
                   "cycles_171.7": float((b - a) * 24 / P_ADOPT), "cycles_122.65": float((b - a) * 24 / P_LIT_FAM)})
    off = sum(v["days"] + 6 for v in visits[:-1])
    axs[1].scatter(jd[m] - a + off, y[m], s=1, color=C_TESS, alpha=0.4)
    for Pq, c in ((P_ADOPT, C_AD), (P_LIT_FAM, C_LIT)):
        for cyc in np.arange(0, (b - a) * 24 / Pq + 1e-9, 1.0):
            axs[1].axvline(off + cyc * Pq / 24, color=c, lw=0.6, alpha=0.5)
    axs[1].text(off, np.nanmin(y[tess]) - 0.04, f"{b - a:.1f} d: {visits[-1]['cycles_171.7']:.1f} x 171.7 h, "
                f"{visits[-1]['cycles_122.65']:.1f} x 122.65 h", fontsize=7, va="bottom")
axs[1].invert_yaxis()
axs[1].set_ylim(np.nanmax(y[tess]) + 0.05, np.nanmin(y[tess]) - 0.12)
axs[1].set_xlabel("days since the start of each TESS visit (visits placed side by side); vertical lines: rotations at "
                  "171.7 h (red) and 122.65 h (blue)", fontsize=8)
axs[1].set_ylabel("reduced mag")
fig.savefig(os.path.join(OUT, "fig05_tess_coverage.png"))
plt.close(fig)
NUM["tess_visits"] = visits

# 6. per-dataset preferences (Figure 7 of the paper, row of 870)
dpt = J("dataset_power_table.json")
rows = [r for r in dpt["datasets"] if str(r.get("num", r.get("asteroid", ""))) == N]
if rows:
    fig, ax = plt.subplots(figsize=(8.5, 3.8), constrained_layout=True)
    rs = sorted(rows, key=lambda r: -r.get("n", 0))
    for i, r in enumerate(rs):
        Pp = r.get("P_pref") or r.get("P_pref_any")
        fam = "adopted" if Pp and any(abs(Pp - q * P_ADOPT) <= 0.01 * Pp for q in (1, 2, 0.5)) else \
            ("literature" if Pp and any(abs(Pp - q * P_LIT_FAM) <= 0.01 * Pp for q in (1, 2, 0.5)) else "other")
        ax.scatter(i, Pp, s=20 + r.get("n", 100) / 15, color={"adopted": C_AD, "literature": C_LIT, "other": C_OTHER}[fam])
    for Pq, c in ((P_ADOPT, C_AD), (P_ADOPT / 2, C_AD), (P_LIT_FAM, C_LIT), (P_LIT_FAM / 2, C_LIT)):
        ax.axhline(Pq, color=c, lw=0.6, alpha=0.5)
    ax.set_yscale("log")
    yt = []
    for v in sorted({round(r.get("P_pref") or r.get("P_pref_any"), 1) for r in rs} | {round(P_ADOPT, 1)}):
        if not yt or v > 1.05 * yt[-1]:
            yt.append(v)
        elif abs(v - P_ADOPT) < abs(yt[-1] - P_ADOPT):
            yt[-1] = v        # of neighbouring periods label the one closest to the adopted period
    ax.set_yticks(yt)
    ax.set_yticklabels([f"{v:.1f}" for v in yt], fontsize=7)
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_xticks(range(len(rs)))
    ax.set_xticklabels([f"{r['ds']}\n{r['n']}" for r in rs], fontsize=7)
    ax.set_ylabel("period preferred by the dataset alone (h)")
    ax.set_title("(870) Manto: period preferred by each dataset among the candidates (size: number of measurements)")
    fig.savefig(os.path.join(OUT, "fig06_datasets.png"))
    plt.close(fig)
    NUM["datasets"] = [{k: r.get(k) for k in ("ds", "n", "P_pref", "power", "in_family")} for r in rows]

# 7. single-apparition best periods (longperiod_tests.py)
lpt = J("longperiod_tests.json").get(N, {})
apps = lpt.get("apparitions") or lpt.get("single_apparition") or []
if apps:
    # apparition mid-times with the same 60-day gap rule as longperiod_tests.py (on the uncorrected arrays it uses)
    au = np.load(os.path.join(ROOT, "results", N, f"{N}_arrays.npz"), allow_pickle=True)
    tu = np.sort(au["t"].astype(float)) + float(au["time_zero"])
    cut = np.where(np.diff(tu) > 60)[0]
    mids = [0.5 * (s + e) for s, e in zip(np.r_[tu[0], tu[cut + 1]], np.r_[tu[cut], tu[-1]])]
    yrs = [2000 + (m - 2451544.5) / 365.25 for m in mids]
    fig, ax = plt.subplots(figsize=(8, 3.4), constrained_layout=True)
    for key, col, lab in (("171.722", C_AD, "171.72 h"), ("85.8608", "#e59866", "85.86 h (P/2)"), ("122.16", C_LIT, "122.16 h")):
        xs = [yrs[a["k"]] for a in apps if key in a["cand"] and "power" in a["cand"][key] and a["k"] < len(yrs)]
        ys = [a["cand"][key]["power"] for a in apps if key in a["cand"] and "power" in a["cand"][key] and a["k"] < len(yrs)]
        ax.plot(xs, ys, "o", color=col, label=f"power at {lab}")
    for a in apps:
        if a.get("best_P") and a["k"] < len(yrs):
            ax.annotate(f"{a['best_P']:.0f}", (yrs[a["k"]], a["best_power"]), textcoords="offset points", xytext=(0, 6),
                        fontsize=7, ha="center")
    ax.set_xlabel("apparition (mid-year)")
    ax.set_ylabel("Lomb-Scargle power")
    ax.set_title("(870) Manto: single apparitions on their own; numbers: strongest period (h) of the apparition", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(os.path.join(OUT, "fig07_apparitions.png"))
    plt.close(fig)
NUM["apparitions"] = apps

# 8. convex window scan over the candidate periods (damit_longscan.py)
lpd = J("longperiod_damit.json")[N]
rws = lpd.get("rows") or lpd.get("windows") or []
fig, ax = plt.subplots(figsize=(7.5, 3.4))
for r in rws:
    Pb, cr = r.get("P_best"), r.get("chi2_ratio_to_best") or r.get("chi2_ratio")
    if Pb is None or cr is None:
        continue
    col = C_AD if any(abs(Pb - q * P_ADOPT) <= 0.01 * Pb for q in (1, 2, 0.5)) else \
        (C_LIT if any(abs(Pb - q * P_LIT_FAM) <= 0.01 * Pb for q in (1, 2, 0.5)) else C_OTHER)
    ax.scatter(Pb, cr, color=col, s=30, zorder=3)
    ax.annotate(f"{Pb:.2f}", (Pb, cr), textcoords="offset points", xytext=(3, 3), fontsize=7)
ax.axhline(lpd["threshold"], color="k", ls=":", lw=0.8, label="uniqueness threshold")
ax.axvline(P_SID_PUB, color=C_LIT, ls="--", lw=0.8, label="published sidereal 122.166 h")
ax.set_xscale("log")
ax.set_xlabel("best period in each scan window (h)")
ax.set_ylabel(r"$\chi^2$ / $\chi^2_{\min}$")
ax.set_title("(870) Manto: convex inversion scanned around every candidate period")
ax.legend(frameon=False, fontsize=8)
fig.savefig(os.path.join(OUT, "fig08_convex_windows.png"))
plt.close(fig)
NUM["convex_windows"] = rws

# 9. the two convex models with their spin and maximum-inertia axes
from review_checks import inertia


def read_shape(p):
    v = open(p).read().split()
    nv, nf = int(v[0]), int(v[1])
    V = np.array(v[2:2 + 3 * nv], float).reshape(nv, 3)
    F = np.array(v[2 + 3 * nv:2 + 3 * nv + 3 * nf], int).reshape(nf, 3) - 1
    return V, F


def axes_of(V):
    from scipy.spatial import ConvexHull
    h = ConvexHull(V)
    c0 = V[h.vertices].mean(axis=0)
    tri = V[h.simplices] - c0
    vol6 = np.abs(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])))
    cen = c0 + np.sum(vol6[:, None] * tri.sum(axis=1), axis=0) / (4 * vol6.sum())
    t2 = V[h.simplices] - cen
    d2 = np.abs(np.einsum("ij,ij->i", t2[:, 0], np.cross(t2[:, 1], t2[:, 2])))
    ss = t2.sum(axis=1)
    Cm = (np.einsum("k,kij,kil->jl", d2, t2, t2) + np.einsum("k,kj,kl->jl", d2, ss, ss)) / 120.0
    I = np.trace(Cm) * np.eye(3) - Cm
    ev, evec = np.linalg.eigh(I)
    return cen, evec[:, 2]


shp = os.path.join(OUT, "shapes")
models = [("ts_870_areas_best.shape", "Model at the adopted period\nP$_{\\rm sid}$ = 171.731 h, pole (14°, +20°)", "rejected"),
          ("alt870_areas_best.shape", "Model at the literature-family period\nP$_{\\rm sid}$ = 122.163 h, pole (96°, +25°)", "accepted")]
fig = plt.figure(figsize=(10, 5.2))
for k, (f, title, verdict) in enumerate(models):
    p = os.path.join(shp, f)
    if not os.path.exists(p):
        continue
    V, F = read_shape(p)
    V = V / np.max(np.abs(V))
    cen, ax_max = axes_of(V)
    ang = float(np.degrees(np.arccos(min(1.0, abs(ax_max[2])))))
    ax = fig.add_subplot(1, 2, k + 1, projection="3d")
    light = np.array([1, -1, 1]) / np.sqrt(3)
    nrm = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    nrm /= np.linalg.norm(nrm, axis=1)[:, None]
    shade = np.clip(nrm @ light, 0.05, 1)
    pc = Poly3DCollection(V[F], facecolors=plt.cm.Greys(0.25 + 0.6 * shade), edgecolor="none", alpha=0.55)
    ax.add_collection3d(pc)
    for sgn in (1, -1):
        ax.plot([cen[0], cen[0]], [cen[1], cen[1]], [cen[2], cen[2] + sgn * 1.5], color="#c0392b", lw=2.5)
        e = cen + sgn * 1.5 * ax_max
        ax.plot([cen[0], e[0]], [cen[1], e[1]], [cen[2], e[2]], color="#2471a3", lw=2.5, ls="--")
    # view perpendicular to the plane of the two axes, so that both are seen at their true angle
    nview = np.cross([0, 0, 1], ax_max)
    if np.linalg.norm(nview) < 0.2:
        nview = np.array([1.0, 0.0, 0.0])
    nview /= np.linalg.norm(nview)
    ax.view_init(elev=float(np.degrees(np.arcsin(np.clip(nview[2], -1, 1)))), azim=float(np.degrees(np.arctan2(nview[1], nview[0]))))
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_zlim(-1.2, 1.2)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.set_title(f"{title}\nspin axis (red) {ang:.0f}° from the axis of maximum\nmoment of inertia (blue, dashed): {verdict}",
                 fontsize=9)
    NUM.setdefault("convex_models", {})[f] = {"angle_spin_maxI_deg": ang}
fig.savefig(os.path.join(OUT, "fig09_convex_models.png"))
plt.close(fig)

# 10. synodic-sidereal comparison of the test asteroids (testswap_litcompare.json)
LC = J("testswap_litcompare.json")
fig, ax = plt.subplots(figsize=(8, 3.8), constrained_layout=True)
pts = sorted([(n, v) for n, v in LC.items() if v.get("dP_sid_in_P2Porb") is not None and n != N], key=lambda x: int(x[0]))
q = LC[N]
alt_ratio = (P_LIT_FAM - q["P_sid_lit"]) / (P_LIT_FAM ** 2 / q["P_orb_h"])
labels = []
for i, (n, v) in enumerate(pts):
    lat = np.mean([p[1] for s in v["spin_axis"] for p in s["poles"]]) if v["spin_axis"] else 0
    ax.scatter(i, v["dP_sid_in_P2Porb"], color="#7d3c98" if lat > 0 else "#117a65", s=40, zorder=3)
    labels.append(f"({n})\n{'prograde' if lat > 0 else 'retrograde'}")
i = len(pts)
ax.scatter(i, alt_ratio, color=C_LIT, s=70, marker="D", zorder=4)
ax.annotate(f"{alt_ratio:+.2f}", (i, alt_ratio), textcoords="offset points", xytext=(8, -4), fontsize=8, color=C_LIT)
labels.append("(870) at 122.65 h\nprograde")
ax.annotate(f"(870) at the adopted 171.72 h: {q['dP_sid_in_P2Porb']:+.0f} (off scale)", (i - 0.2, 1.75), fontsize=8,
            color=C_AD, ha="right")
ax.axhline(1, color="#7d3c98", lw=0.7, alpha=0.5); ax.axhline(-1, color="#117a65", lw=0.7, alpha=0.5)
ax.axhline(0, color="0.85", lw=0.6)
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylim(-2, 2)
ax.set_ylabel(r"$(P - P_{\rm sid,pub})\,/\,(P^2/P_{\rm orb})$")
ax.set_title("Test asteroids with published convex models: our period minus the published sidereal period "
             "(expected +1 prograde, -1 retrograde)", fontsize=9)
fig.savefig(os.path.join(OUT, "fig10_synodic_sidereal.png"))
plt.close(fig)
NUM["alt_ratio"] = alt_ratio

# 11. aspect model of 870 (geometry_results.json): amplitude per apparition and fitted model
gr = J("geometry_results.json").get(f"paper:{N}")
if gr and gr.get("windows"):
    w = gr["windows"]
    tm = np.array([x["t_mid"] for x in w])
    g = np.array([x["g"] for x in w]); sg = np.array([x["sigma_g"] for x in w])
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.errorbar(2000 + (tm + t0 - 2451544.5) / 365.25 if tm.max() < 1e5 else 2000 + (tm - 2451544.5) / 365.25, g, sg,
                fmt="o", color="k", ms=4, label="amplitude per apparition (template fit)")
    if gr.get("window_model_A"):
        ax.plot(2000 + (tm + t0 - 2451544.5) / 365.25 if tm.max() < 1e5 else 2000 + (tm - 2451544.5) / 365.25,
                gr["window_model_A"], "s", color=C_AD, mfc="none", label="aspect model")
    ax.set_xlabel("year"); ax.set_ylabel("peak-to-peak amplitude (mag)")
    ax.set_title(f"(870) Manto: amplitude per apparition at 171.72 h; aspect model $\\Delta$BIC = {gr['geometry']['delta_bic']:.0f}")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(os.path.join(OUT, "fig11_aspect.png"))
    plt.close(fig)

json.dump(NUM, open(os.path.join(OUT, "manto_numbers.json"), "w"), indent=1, default=float)
print("written", sorted(f for f in os.listdir(OUT) if f.endswith(".png")))
