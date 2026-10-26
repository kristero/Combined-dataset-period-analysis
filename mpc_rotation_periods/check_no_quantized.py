"""Robustness of the adopted periods to the datasets whose magnitudes are reported only to 0.1 mag (all TESS data
in the MPC database and several ground-based datasets; data/dataset_precision.json from diag_quantization.py).
For every asteroid the combined data without these datasets are searched over the candidate periods of
rebootstrap2.py with the exact chi2 method (as the subset checks of the paper), and the preferred candidate is
compared with the adopted alias family. Output: no_quantized_check.json"""
import json
import os

import numpy as np

import rebootstrap2 as r2

ROOT = os.path.dirname(os.path.abspath(__file__))
A = json.load(open(os.path.join(ROOT, "adopted.json"), encoding="utf-8"))
PREC = json.load(open(os.path.join(ROOT, "data", "dataset_precision.json"), encoding="utf-8"))["by_asteroid"]
Q_LIMIT = 0.9     # a dataset counts as quantized if >= 90 % of its magnitudes lie on the 0.1-mag grid


def in_family(Pc, P, tol=0.01):
    return any(abs(Pc - r * P) <= tol * Pc for r in (1.0, 2.0, 0.5))


out = {}
for num in sorted(A, key=int):
    # geometry-corrected data where the viewing-geometry correction is applied (geometry_reanalysis.py)
    rd = os.path.join(ROOT, "results_geo", num)
    if not os.path.exists(os.path.join(rd, f"{num}_rebootstrap2.json")):
        rd = os.path.join(ROOT, "results", num)
    a = np.load(os.path.join(rd, f"{num}_arrays.npz"))
    v = json.load(open(os.path.join(rd, f"{num}_rebootstrap2.json"), encoding="utf-8"))
    t, y, sheet = a["t"], a["y"], a["sheet"]
    quant = sorted(ds for ds in set(sheet) if PREC[num].get(ds, 0.0) >= Q_LIMIT)
    m = ~np.isin(sheet, quant)
    P = A[num]["P"]
    allc = v["primary"] + v["extended"]
    prim = v["primary"]
    tb, yb = t[m], y[m]
    bb = float(tb.max() - tb.min())
    vals = [float(r2.power(tb, yb, r2.grid(c["f"], bb, 3.0, spp=20), exact=True).max()) for c in allc]
    best = allc[int(np.argmax(vals))]["P"]
    best_prim = max(prim, key=lambda c: vals[allc.index(c)])["P"]
    pref = best_prim if P <= 240 else best
    ok = in_family(pref, P) or (P > 240 and in_family(best_prim, P))
    o = np.argsort(vals)[::-1]
    out[num] = {"quantized_datasets": quant, "n_quantized": int((~m).sum()), "n_rest": int(m.sum()),
                "preferred_P": pref, "agrees_with_adopted_family": bool(ok), "quality": A[num]["quality"],
                "ratio_second_to_first": float(vals[o[1]] / vals[o[0]])}
    print(f"{num} [{A[num]['quality']}] without {quant} ({(~m).sum()} of {len(t)} points): preferred {pref:.4f} h "
          f"vs adopted {P:.4f} h -> {'same family' if ok else 'DIFFERENT'}", flush=True)
json.dump(out, open(os.path.join(ROOT, "no_quantized_check.json"), "w"), indent=1)
agree = [n for n, r in out.items() if r["agrees_with_adopted_family"]]
print(f"\nadopted family preferred without the 0.1-mag datasets: {len(agree)} of {len(out)}")
for q in "ABC":
    nq = [n for n, r in out.items() if r["quality"] == q]
    print(f"  quality {q}: {sum(out[n]['agrees_with_adopted_family'] for n in nq)} of {len(nq)}; "
          f"different: {[n for n in nq if not out[n]['agrees_with_adopted_family']]}")
