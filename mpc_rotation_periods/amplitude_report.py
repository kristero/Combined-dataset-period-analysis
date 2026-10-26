"""Summary of the amplitude study for the authors: validation table against the LCDB, injection recovery, and the
change of the amplitudes of all 57 asteroids if the order-4 Fourier amplitude replaces the current estimator.
Writes amplitude_report.json and review/amplitude_comparison.png."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(ROOT, "amplitude_study.json"), encoding="utf-8"))
E = json.load(open(os.path.join(ROOT, "amplitude_evaluation.json"), encoding="utf-8"))
INJ = json.load(open(os.path.join(ROOT, "amplitude_injection_summary.json"), encoding="utf-8"))
A = json.load(open(os.path.join(ROOT, "adopted.json"), encoding="utf-8"))
ref = E["reference"]
K = 1 / 0.75      # empirical scale between the combined-data and the literature (single-apparition) amplitudes

print("Validation (LCDB range | current B18m2 | F2 | F4 | F8 | B2 debiased | F4 x 1.33):")
rows = []
for n in sorted(ref, key=int):
    e = S[n]["estimates"]
    d = S[n]["debiased"]
    lo, hi = ref[n]
    r = {"asteroid": n, "lcdb": [lo, hi], "B18m2": e["B18m2"], "F2": e["F2"], "F4": e["F4"], "F8": e["F8"],
         "B2d": d["B2d"], "F4cal": e["F4"] * K}
    rows.append(r)
    print(f"  ({n})  {lo:.2f}-{hi:.2f} | {e['B18m2']:.2f} | {e['F2']:.2f} | {e['F4']:.2f} | {e['F8']:.2f} | "
          f"{d['B2d']:.2f} | {e['F4'] * K:.2f}")
for key in ("B18m2", "F4", "F4cal"):
    v = np.array([r[key] for r in rows])
    mid = np.array([0.5 * sum(r["lcdb"]) for r in rows])
    inr = sum(r["lcdb"][0] - 0.03 <= r[key] <= r["lcdb"][1] + 0.03 for r in rows)
    print(f"  {key}: in range {inr}/{len(rows)}, median ratio to LCDB mid {np.median(v / mid):.2f}, "
          f"rms {np.sqrt(np.mean(np.log10(v / mid) ** 2)):.3f} dex")
ratio = np.array([S[n]["estimates"]["F4"] / S[n]["estimates"]["B18m2"] for n in A])
print(f"F4 / current amplitude over all 57: median {np.median(ratio):.3f}, range {ratio.min():.2f}-{ratio.max():.2f}")
json.dump({"validation": rows, "F4_over_current": {n: S[n]["estimates"]["F4"] / S[n]["estimates"]["B18m2"] for n in A},
           "injection": {k: INJ[k] for k in ("B18m2", "F2", "F4", "F6", "F8", "B2", "B2d", "B5", "B10", "N4wmean", "ALN4")}},
          open(os.path.join(ROOT, "amplitude_report.json"), "w"), indent=1)

fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
ax = axs[0]
keys = [("B18m2", "current (18° bins, N=2 clip)"), ("F4", "Fourier N=4"), ("F8", "Fourier N=8"), ("B2d", "2° bins, noise-corrected")]
cols = ["0.4", "#b9770e", "#7e5109", "#2471a3"]
for i, (k, lab) in enumerate(keys):
    for j, r in enumerate(rows):
        x = j + (i - 1.5) * 0.17
        ax.plot([j - 0.4, j + 0.4], [r["lcdb"][0]] * 2, color="0.75", lw=0.8) if i == 0 else None
        ax.plot([j - 0.4, j + 0.4], [r["lcdb"][1]] * 2, color="0.75", lw=0.8) if i == 0 else None
        ax.fill_between([j - 0.4, j + 0.4], r["lcdb"][0], r["lcdb"][1], color="0.9", zorder=0) if i == 0 else None
        ax.scatter([x], [r[k]], color=cols[i], s=18, zorder=3, label=lab if j == 0 else None)
ax.set_xticks(range(len(rows)))
ax.set_xticklabels([r["asteroid"] for r in rows], rotation=60)
ax.set_ylabel("Amplitude (mag)")
ax.set_title("Grey: LCDB amplitude range")
ax.legend(fontsize=7, loc="upper left")
ax = axs[1]
amps = [0.10, 0.25, 0.50]
for i, (k, lab) in enumerate(keys):
    q = np.array([INJ[k][str(a)] for a in amps])
    ax.errorbar(np.array(amps) * (1 + (i - 1.5) * 0.05), q[:, 1], yerr=[q[:, 1] - q[:, 0], q[:, 2] - q[:, 1]],
                fmt="o", color=cols[i], capsize=2, label=lab)
ax.axhline(1, color="k", lw=0.6)
ax.set_xscale("log")
ax.set_xticks(amps)
ax.set_xticklabels([f"{a:.2f}" for a in amps])
ax.set_xlabel("Injected amplitude (mag)")
ax.set_ylabel("Recovered / injected (median, 16-84%)")
ax.set_title("Injection test with the real noise and sampling")
fig.tight_layout()
os.makedirs(os.path.join(ROOT, "review"), exist_ok=True)
fig.savefig(os.path.join(ROOT, "review", "amplitude_comparison.png"), dpi=110)
print("wrote review/amplitude_comparison.png")
