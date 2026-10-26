"""
Worked example of daily aliasing for one asteroid (default (4239) Goodman, P = 23.13818 h; submitted value 15.612 h).
 (a) spectral window W(f) = |sum_k exp(2 pi i f t_k)|^2 / N^2 of the actual observation times
 (b) combined Lomb-Scargle periodogram (nterms = 2, as in mpc_redo.py) against rotation frequency, peaks labelled
 (c) the same for the three longitude groups of rebootstrap2.py
 (d) the data folded at the adopted period and at the alias (binned means, residual rms)
Output: review/alias_example_<N>.png
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from astropy.timeseries import LombScargle  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from rebootstrap2 import GROUP_OF  # noqa: E402

num = sys.argv[1] if len(sys.argv) > 1 else "4239"
P_true = json.load(open(os.path.join(ROOT, "adopted.json")))[num]["P"]
P_alias = float(sys.argv[2]) if len(sys.argv) > 2 else 15.61252
a = np.load(os.path.join(ROOT, "results", num, f"{num}_arrays.npz"), allow_pickle=True)
t, y, sheet = a["t"].astype(float), a["y"].astype(float), a["sheet"].astype(str)
grp = np.array([GROUP_OF.get(s[:3], "other") for s in sheet])
T = np.ptp(t)
f = np.arange(0.3, 5.0, 1.0 / (10 * T))             # rotation frequency, cycles/day
plt.rcParams.update({"font.size": 10})
fig = plt.figure(figsize=(11, 9))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1.2, 1.2], hspace=0.45, wspace=0.22)

# (a) spectral window
fw = np.arange(0.0, 3.05, 1.0 / (10 * T))
ph = 2 * np.pi * np.outer(fw, t)
W = (np.cos(ph).sum(1) ** 2 + np.sin(ph).sum(1) ** 2) / len(t) ** 2
ax = fig.add_subplot(gs[0, :])
ax.plot(fw, W, color="0.25", lw=0.6)
for k in (1, 2, 3):
    ax.annotate(f"{k} c/d (daily sampling)" if k == 1 else f"{k} c/d", (k, W[np.argmin(np.abs(fw - k))]),
                xytext=(k + 0.08, 0.75), fontsize=9, arrowprops=dict(arrowstyle="-", color="0.5"))
ax.set_xlim(0, 3.05)
ax.set_ylim(0, 1.05)
ax.set_xlabel("frequency offset (cycles/day)")
ax.set_ylabel("window power")
ax.set_title(f"(a) Spectral window of the {len(t)} observation times of ({num}): every signal is copied at +-1, 2 c/d",
             loc="left")

# (b) combined periodogram
ls = LombScargle(t, y, nterms=2)
p = ls.power(f, method="fastchi2", assume_regular_frequency=True)
ax = fig.add_subplot(gs[1, :])
ax.plot(f, p, color="#1a5276", lw=0.6)
f1 = 24.0 / P_true
labels = [(f1, f"P = {P_true:.3f} h (true)", "#c0392b"), (2 * f1, f"P/2 = {P_true / 2:.3f} h (same family)", "#c0392b"),
          (24.0 / P_alias, f"{P_alias:.3f} h = 2f$_{{rot}}$+1 c/d alias", "#7d3c98"),
          (f1 + 1.0, f"{24 / (f1 + 1):.3f} h = f$_{{rot}}$+1 c/d", "0.4"),
          (2 * f1 + 2.0, f"{24 / (2 * f1 + 2):.3f} h = 2f$_{{rot}}$+2 c/d", "0.4")]
for fx, lab, col in labels:
    k = np.abs(f - fx) < 0.01
    if k.any():
        py = p[k].max()
        dx, dy = {0: (0.05, 0.03), 1: (0.05, 0.05), 2: (-0.55, 0.06), 3: (0.10, -0.075), 4: (0.05, 0.03)}[labels.index((fx, lab, col))]
        ax.annotate(lab, (fx, py), xytext=(fx + dx, py + dy), fontsize=8.5, color=col,
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.6))
ax.set_xlim(0.3, 5.0)
ax.set_ylim(0, p.max() * 1.35)
ax.set_xlabel("rotation frequency (cycles/day)")
ax.set_ylabel("LS power (nterms = 2)")
ax.set_title("(b) Combined periodogram of all observatories: the true family dominates, its daily aliases are weaker",
             loc="left")

# (c) power at the true period and at the alias, per longitude group and combined (exact chi2)
ax = fig.add_subplot(gs[2, 0])
cols = {"americas_hawaii": "Americas/Hawaii", "europe_africa": "Europe/Africa", "asia_australia": "Asia/Australia"}
names, pt, pa = [], [], []
for g, lab in list(cols.items()) + [("all", "all combined")]:
    m = np.ones(len(t), bool) if g == "all" else grp == g
    if m.sum() < 50:
        continue
    lsg = LombScargle(t[m], y[m], nterms=2)
    names.append(f"{lab}\n({m.sum()} points)")
    pt.append(lsg.power(np.array([24.0 / P_true]), method="chi2")[0])
    pa.append(lsg.power(np.array([24.0 / P_alias]), method="chi2")[0])
xb = np.arange(len(names))
ax.bar(xb - 0.2, pt, 0.4, color="#c0392b", label=f"{P_true:.3f} h (true)")
ax.bar(xb + 0.2, pa, 0.4, color="#7d3c98", label=f"{P_alias:.3f} h (alias)")
for i in range(len(names)):
    ax.text(xb[i], max(pt[i], pa[i]) + 0.02, f"{pa[i] / pt[i]:.2f}", ha="center", fontsize=8.5)
ax.set_xticks(xb)
ax.set_xticklabels(names, fontsize=8)
ax.set_ylabel("LS power")
ax.set_ylim(0, max(pt) * 1.2)
ax.legend(fontsize=8, frameon=False)
ax.set_title("(c) Alias/true power ratio per longitude group", loc="left")
# (d) folds
ax = fig.add_subplot(gs[2, 1])
for P, c, off in ((P_true, "#c0392b", 0.0), (P_alias, "#7d3c98", 0.9)):
    phs = (t * 24.0 / P) % 1.0
    bins = np.linspace(0, 1, 21)
    idx = np.digitize(phs, bins) - 1
    med = np.array([np.median(y[idx == i]) for i in range(20)])
    X = np.vstack([np.ones_like(phs)] + [fn(2 * np.pi * k * phs) for k in (1, 2) for fn in (np.sin, np.cos)]).T
    c_, *_ = np.linalg.lstsq(X, y, rcond=None)
    rms = np.sqrt(np.mean((y - X @ c_) ** 2))
    ax.plot(phs, y + off, ".", ms=1, color=c, alpha=0.15, rasterized=True)
    ax.plot(0.5 * (bins[1:] + bins[:-1]), med + off, "o-", color=c, ms=3, lw=1.2,
            label=f"P = {P:.3f} h: residual rms {rms:.3f} mag")
ax.invert_yaxis()
ax.set_xlabel("rotational phase")
ax.set_ylabel("reduced magnitude (offset)")
ax.legend(fontsize=8, frameon=False, loc="lower center")
ax.set_title("(d) Folded data: true period double-peaked and tighter", loc="left")
out = os.path.join(ROOT, "review", f"alias_example_{num}.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print(out)
# numbers for the text
print("f_rot true %.5f c/d, 2f = %.5f; alias f %.5f, 2f = %.5f; 2f difference %.5f c/d" %
      (f1, 2 * f1, 24 / P_alias, 48 / P_alias, 48 / P_alias - 2 * f1))
for g in cols:
    m = grp == g
    if m.sum() < 50:
        continue
    lsg = LombScargle(t[m], y[m], nterms=2)
    pt = lsg.power(np.array([f1]), method="chi2")[0]
    pa = lsg.power(np.array([24 / P_alias]), method="chi2")[0]
    print(f"{g}: n={m.sum()} power true {pt:.3f} alias {pa:.3f}")
