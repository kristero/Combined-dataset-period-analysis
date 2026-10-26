"""Figures and numbers for the amplitude-method report: (a) our amplitudes vs LCDB for the validation and paper samples,
(b) injection recovery, (c) new vs current amplitudes of the 57 paper asteroids. Writes review/amplitude_methods.png
and amp_report2.json."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(ROOT, "amp_eval_results.json"), encoding="utf-8"))
DEB = json.load(open(os.path.join(ROOT, "amp_debias.json"), encoding="utf-8"))
INJ = json.load(open(os.path.join(ROOT, "amp_injection2_summary.json"), encoding="utf-8"))
FI = json.load(open(os.path.join(ROOT, "figure_info.json"), encoding="utf-8"))
for k, r in R.items():
    r["APP_FS_med_deb"] = DEB[k].get("A_corr", np.nan)

fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
for ax, key, title in ((axs[0], "COMB_F4", "Combined fold, Fourier N=4 (current practice)"),
                       (axs[1], "APP_TPL_wmean", "Per-apparition template fit (amplitude-weighted mean)")):
    for samp, col, mk in (("val:", "#b9770e", "o"), ("paper:", "#1a5276", "D")):
        rows = [r for k, r in R.items() if k.startswith(samp) and r["lcdb"] and np.isfinite(r[key])]
        lo = np.array([r["lcdb"]["amin"] for r in rows])
        hi = np.array([r["lcdb"]["amax"] for r in rows])
        v = np.array([r[key] for r in rows])
        mid = 0.5 * (lo + hi)
        ax.errorbar(mid, v, xerr=[mid - lo, hi - mid], fmt=mk, ms=3.5, color=col, ecolor=col, elinewidth=0.5, alpha=0.8,
                    label=f"{'validation' if samp == 'val:' else 'paper'} (n={len(rows)}), median ratio {np.median(v / mid):.2f}")
    ax.plot([0, 1.2], [0, 1.2], "k-", lw=0.7)
    ax.set_xlim(0, 1.15)
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("LCDB amplitude (mag): midpoint, bar = Amin-Amax")
    ax.set_ylabel("Our amplitude (mag)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
ax = axs[2]
amps = [0.10, 0.25, 0.50]
for i, (k, lab, col) in enumerate((("COMB_F4", "combined fold N=4", "0.3"), ("APP_TPL_wmean", "per-apparition template (recommended)", "#b9770e"), ("APP_FS_med", "per-apparition free Fourier, median", "#d4ac0d"),
                                   ("APP_FS_max", "per-apparition Fourier, maximum", "#7e5109"), ("APP_VARs_med", "per-apparition dispersion (shape)", "#2471a3"))):
    q = np.array([INJ[k][str(a)] for a in amps])
    ax.errorbar(np.array(amps) * (1 + (i - 2) * 0.04), q[:, 1], yerr=[q[:, 1] - q[:, 0], q[:, 2] - q[:, 1]], fmt="o-",
                color=col, capsize=2, label=lab, lw=0.8)
ax.axhline(1, color="k", lw=0.6)
ax.set_xscale("log")
ax.set_xticks(amps)
ax.set_xticklabels([f"{a:.2f}" for a in amps])
ax.set_ylim(0.8, 3.2)
ax.set_xlabel("Injected amplitude (mag)")
ax.set_ylabel("Recovered / injected (median, 16-84 %)")
ax.set_title("Injection test (real sampling, shuffled real noise)", fontsize=10)
ax.legend(fontsize=7.5)
fig.tight_layout()
os.makedirs(os.path.join(ROOT, "review"), exist_ok=True)
fig.savefig(os.path.join(ROOT, "review", "amplitude_methods.png"), dpi=110)

# paper sample: current table amplitudes vs new
cur = FI["lc_amplitudes_all"]
rows = []
for k, r in R.items():
    if k.startswith("paper:"):
        n = k.split(":")[1]
        rows.append((n, cur[n]["amp"], r["COMB_F4"], r["APP_FS_med_deb"], DEB[k].get("upper_limit"), r["n_app"]))
ratio = np.array([b / a for _, a, _, b, ul, _ in rows if np.isfinite(b) and not ul])
print("paper sample: new (per-apparition, corrected) / current amplitude: median %.2f, 16-84%% %.2f-%.2f; upper limits: %s; "
      "asteroids without usable apparitions: %s" % (np.median(ratio), *np.percentile(ratio, [16, 84]),
      [n for n, *_, ul, na in rows if ul], [n for n, a, c, b, ul, na in rows if not np.isfinite(b)]))
json.dump({"paper_amplitudes": [{"num": n, "current": a, "comb_f4": c, "per_apparition_corrected": b, "upper_limit": ul,
                                 "n_apparitions": na} for n, a, c, b, ul, na in rows]},
          open(os.path.join(ROOT, "amp_report2.json"), "w"), indent=1)
print("wrote review/amplitude_methods.png")
