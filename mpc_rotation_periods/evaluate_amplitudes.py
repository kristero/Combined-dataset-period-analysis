"""Compare the amplitude estimators of amplitude_study.py with the LCDB amplitudes for the asteroids whose LCDB period
belongs to our adopted alias family. Output: amplitude_evaluation.json and a printed ranking."""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(ROOT, "amplitude_study.json"), encoding="utf-8"))
L = json.load(open(os.path.join(ROOT, "data", "lcdb_amplitudes.json"), encoding="utf-8"))
A = json.load(open(os.path.join(ROOT, "adopted.json"), encoding="utf-8"))
TOL = 0.03


def same_family(Pl, P):
    return Pl is not None and any(abs(Pl - r * P) <= 0.03 * Pl for r in (1.0, 2.0, 0.5))


ref = {}
for n, l in L.items():
    s = l["summary"]
    if s["amax"] is None or not same_family(s["period"], A[n]["P"]):
        continue
    lo = s["amin"] if s["amin"] is not None else s["amax"]
    ref[n] = (lo, s["amax"])
print("validation asteroids:", {n: ref[n] for n in sorted(ref, key=int)})


def value(n, key):
    r = S[n]
    if key in r["per_apparition"]:
        return r["per_apparition"][key]
    if key.endswith("d"):
        return r["debiased"].get(key, np.nan)
    return r["estimates"].get(key, np.nan)


keys = []
for k in S[next(iter(S))]["estimates"]:
    if not k.startswith("P_"):
        keys += [k, k + "d"]
keys += ["A2max", "A2med", "A4max", "A4med", "ALN2", "ALN4"] + [f"N{N}{s}" for N in (2, 4) for s in ("med", "max", "p75", "wmean")]
rows = []
for k in keys:
    vals = {n: value(n, k) for n in ref}
    ok = {n: v for n, v in vals.items() if np.isfinite(v)}
    if len(ok) < len(ref) - 1:
        continue
    inr = sum(ref[n][0] - TOL <= v <= ref[n][1] + TOL for n, v in ok.items())
    rat = np.array([v / (0.5 * (ref[n][0] + ref[n][1])) for n, v in ok.items()])
    lr = np.log10(np.clip(rat, 1e-3, None))
    rows.append({"estimator": k, "n": len(ok), "in_range": int(inr), "median_ratio_to_mid": float(np.median(rat)),
                 "scatter_dex": float(1.4826 * np.median(np.abs(lr - np.median(lr)))),
                 "rms_dex": float(np.sqrt(np.mean(lr ** 2))), "values": {n: round(v, 3) for n, v in ok.items()}})
rows.sort(key=lambda r: (-r["in_range"], r["rms_dex"]))
json.dump({"reference": ref, "ranking": rows}, open(os.path.join(ROOT, "amplitude_evaluation.json"), "w"), indent=1)
print(f"\n{'estimator':10s} in-range  median ratio  scatter(dex)  rms(dex)")
for r in rows:
    print(f"{r['estimator']:10s} {r['in_range']:3d}/{r['n']:<3d}   {r['median_ratio_to_mid']:6.2f}        {r['scatter_dex']:.3f}        {r['rms_dex']:.3f}")
