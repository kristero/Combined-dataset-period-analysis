"""Diagnose non-rotational structure in the folded light curves: magnitude quantization of each dataset (reported
precision 0.1 mag vs 0.01 mag) and the TESS visits. Writes data/dataset_precision.json and review/diag_<N>.png
(fold at the adopted period, coloured by dataset class)."""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
A = json.load(open(os.path.join(ROOT, "adopted.json"), encoding="utf-8"))


def precision_class(m):
    m = np.asarray(m, float)
    m = m[np.isfinite(m)]
    if len(m) == 0:
        return None, None
    f01 = float(np.mean(np.abs(m * 10 - np.round(m * 10)) < 1e-6))
    f001 = float(np.mean(np.abs(m * 100 - np.round(m * 100)) < 1e-6))
    return f01, f001


prec = {}
for f in sorted(glob.glob(os.path.join(ROOT, "data", "obs_tables", "*_obs.csv"))):
    num = os.path.basename(f).split("_")[0]
    df = pd.read_csv(f, usecols=["dataset", "mag_f"])
    for ds, g in df.groupby("dataset"):
        f01, f001 = precision_class(g["mag_f"].values)
        prec.setdefault(ds, {"n": 0, "n01": 0.0})
        prec[ds]["n"] += len(g)
        prec[ds]["n01"] += f01 * len(g)
        prec.setdefault("_by_ast", {}).setdefault(num, {})[ds] = round(f01, 3)
summary = {ds: round(v["n01"] / v["n"], 3) for ds, v in prec.items() if not ds.startswith("_")}
json.dump({"fraction_at_0.1mag_steps": summary, "by_asteroid": prec["_by_ast"]},
          open(os.path.join(ROOT, "data", "dataset_precision.json"), "w"), indent=1)
print("fraction of magnitudes on a 0.1-mag grid, per dataset (all asteroids):")
for ds, v in sorted(summary.items(), key=lambda kv: -kv[1]):
    print(f"  {ds:6s} {v:.2f}  (n={prec[ds]['n']})")

os.makedirs(os.path.join(ROOT, "review"), exist_ok=True)
for num in sys.argv[1:]:
    a = np.load(os.path.join(ROOT, "results", num, f"{num}_arrays.npz"))
    t, y, sheet = a["t"], a["y"], a["sheet"]
    P = A[num]["P"]
    ph = ((t * 24.0 / P) % 1.0)
    q = prec["_by_ast"][num]
    cls = np.array(["TESS" if s.startswith("C57") else ("0.1-mag steps" if q.get(s, 0) > 0.9 else "other") for s in sheet])
    fig, axs = plt.subplots(1, 4, figsize=(16, 3.6), sharey=True)
    for ax, c, col in zip(axs, ["other", "0.1-mag steps", "TESS", None], ["0.4", "tab:red", "tab:blue", None]):
        if c is None:
            for cc, col2 in (("other", "0.6"), ("0.1-mag steps", "tab:red"), ("TESS", "tab:blue")):
                m = cls == cc
                ax.scatter(ph[m], y[m], s=1, c=col2, alpha=0.4, label=f"{cc} ({m.sum()})")
            ax.legend(markerscale=6, fontsize=8)
            ax.set_title(f"({num}) all, P={P:.4f} h")
        else:
            m = cls == c
            ax.scatter(ph[m], y[m], s=1, c=col, alpha=0.5)
            ax.set_title(f"{c}: {m.sum()} pts; " + ", ".join(sorted(set(sheet[m])))[:60], fontsize=8)
    axs[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "review", f"diag_{num}.png"), dpi=80)
    plt.close(fig)
    print("wrote", f"review/diag_{num}.png")
