"""
Reproducibility check of the HPC Python 3.12 environment: (1706) rerun on the HPC (zp/results_hpc_testswap/1706)
against the stored result of the local run (results/1706). Compares the screening, the combined data, the threshold,
every candidate (period, power, bootstrap statistics, fold) and the night-block bootstrap of rebootstrap.py.
Writes testswap_repro_check.json.
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(ROOT, "results", "1706")
B = os.path.join(ROOT, "zp", "results_hpc_testswap", "1706")
out = {}


def num_diff(a, b, path=""):
    """largest relative difference of all numbers in two nested structures, and the paths that differ"""
    worst, paths = 0.0, []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            if k in ("runtime_s", "fold_figure", "workbook", "figures") or k.endswith("_figure"):
                continue
            if k not in a or k not in b:
                paths.append(f"{path}.{k} missing")
                continue
            w, p = num_diff(a[k], b[k], f"{path}.{k}")
            worst, paths = max(worst, w), paths + p
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return 1.0, [f"{path} length {len(a)} vs {len(b)}"]
        for i, (x, y) in enumerate(zip(a, b)):
            w, p = num_diff(x, y, f"{path}[{i}]")
            worst, paths = max(worst, w), paths + p
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        d = abs(a - b) / max(abs(a), abs(b), 1e-300)
        if d > 1e-9:
            paths.append(f"{path}: {a} vs {b}")
        worst = d
    elif a != b and not (isinstance(a, str) and isinstance(b, str) and ("\\" in a or "/" in a)):
        paths.append(f"{path}: {a!r} vs {b!r}")
        worst = 1.0
    return worst, paths


def rel(x):
    """relative size of one difference line 'path: a vs b' (1 for non-numeric differences)"""
    try:
        u, v = (float(s) for s in x.rsplit(": ", 1)[1].split(" vs "))
        return abs(u - v) / max(abs(u), abs(v), 1e-300)
    except (ValueError, IndexError):
        return 1.0


for f in ("1706_result.json", "1706_rebootstrap.json"):
    a = json.load(open(os.path.join(A, f), encoding="utf-8"))
    b = json.load(open(os.path.join(B, f), encoding="utf-8"))
    w, p = num_diff(a, b)
    big = sorted(((rel(x), x) for x in p), reverse=True)
    out[f] = {"max_relative_difference": w, "n_differences": len(p),
              "n_above_1e-5": sum(r > 1e-5 for r, _ in big), "largest": [x for _, x in big[:20]]}
    print(f, "max rel diff", f"{w:.2e}", "differences", len(p), "above 1e-5:", out[f]["n_above_1e-5"])
    for r, x in big[:12]:
        print(f"   {r:.1e} {x[:180]}")
ya = np.load(os.path.join(A, "1706_arrays.npz"))
yb = np.load(os.path.join(B, "1706_arrays.npz"))
for k in ("t", "y"):
    out[f"arrays_{k}"] = float(np.max(np.abs(ya[k] - yb[k]))) if ya[k].shape == yb[k].shape else f"shape {ya[k].shape} vs {yb[k].shape}"
    print("arrays", k, out[f"arrays_{k}"])
json.dump(out, open(os.path.join(ROOT, "testswap_repro_check.json"), "w"), indent=1)
