"""Accuracy of astropy's approximate 'fastchi2' power against the exact 'chi2' method, for the full combined
datasets (windows around the adopted period and its alternatives) and for the single-dataset / subset checks."""
import json
import os

import numpy as np
from astropy.timeseries import LombScargle

ROOT = os.path.dirname(os.path.abspath(__file__))
A = json.load(open(os.path.join(ROOT, "adopted.json"), encoding="utf-8"))
out = {}
worst_full, worst_sub = [], []
for num, ad in A.items():
    a = np.load(os.path.join(ROOT, "results", num, f"{num}_arrays.npz"))
    t, y, sheet = a["t"], a["y"], a["sheet"]
    base = t.max() - t.min()
    Ps = [ad["P"]] + [p for p in ad.get("alternatives", []) if 0.8 <= p <= 1500]
    dmax = 0.0
    for P in Ps:
        f0 = 24.0 / P
        g = np.arange(f0 - 3 / base, f0 + 3 / base, 1 / (50 * base))
        g = g[g > 0]
        ls = LombScargle(t, y, nterms=2)
        pf = ls.power(g, method="fastchi2", assume_regular_frequency=True)
        pe = ls.power(g, method="chi2")
        dmax = max(dmax, float(np.max(np.abs(pf - pe))))
        worst_full.append((float(np.max(np.abs(pf - pe))), num, P, float(pe.max()), float(g[np.argmax(pe)]) - float(g[np.argmax(pf)])))
    sub = {}
    for ds in np.unique(sheet):
        m = sheet == ds
        if m.sum() < 30:
            continue
        g = np.linspace(24.0 / ad["P"] * 0.99, 24.0 / ad["P"] * 1.01, 1501)
        ls = LombScargle(t[m], y[m], nterms=2)
        pf = float(ls.power(g, method="fastchi2", assume_regular_frequency=True).max())
        pe = float(ls.power(g, method="chi2").max())
        sub[str(ds)] = (pf, pe)
        worst_sub.append((abs(pf - pe), num, str(ds), pf, pe, int(m.sum())))
    out[num] = {"full_max_abs_diff": dmax, "datasets": sub}
json.dump(out, open(os.path.join(ROOT, "fastchi2_check.json"), "w"), indent=1)
worst_full.sort(reverse=True)
worst_sub.sort(reverse=True)
print("FULL DATA, worst |fastchi2 - chi2| (diff, asteroid, P, exact max power, peak shift in c/d):")
for w in worst_full[:8]:
    print("  ", tuple(round(x, 5) if isinstance(x, float) else x for x in w))
print("SINGLE DATASETS, worst (diff, asteroid, dataset, fast, exact, n):")
for w in worst_sub[:15]:
    print("  ", tuple(round(x, 4) if isinstance(x, float) else x for x in w))
print("n single-dataset cases with |diff|>0.02:", sum(1 for w in worst_sub if w[0] > 0.02), "of", len(worst_sub))
