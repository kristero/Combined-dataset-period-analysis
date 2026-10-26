"""Run amp_methods.estimate on the paper and validation samples and compare with the LCDB amplitudes.
Output: amp_eval_results.json (all estimates) and amp_eval_summary.json (metrics, bootstrap 68 % intervals)."""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
KEYS = ["COMB_F4", "APP_TPL_med", "APP_TPL_wmean", "APP_TPL_max10", "APP_FS_med", "APP_FS_med_deb", "APP_FS_max", "APP_VAR_med", "APP_VAR_max", "APP_VARs_med", "APP_VARs_max",
        "APP_SF_med", "APP_FS_med_apr", "APP_VAR_med_apr", "APP_VARs_med_apr"]
TOL = 0.03


def run(job):
    import amp_data as D
    import amp_methods as M
    s, n = job
    d = D.load(s, n)
    r = M.estimate(d)
    return f"{s}:{n}", {"lcdb": d["lcdb"], "snr": d["snr"], "quality": d["quality"], "P": d["P"], **r}


def metrics(rows, key, rng, ref="lcdb"):
    """ref = "lcdb": LCDB summary Amin/Amax (all publications); "dense": dense-photometry publications only
    (lcdb_dense_refs.py)."""
    v, lo, hi, refmed = [], [], [], []
    for r in rows:
        a = r.get(key)
        R_ = r.get(ref)
        if a is None or not np.isfinite(a) or not R_ or (ref == "dense" and not R_.get("amps")):
            continue
        v.append(a)
        if ref == "dense":
            lo.append(R_["min"])
            hi.append(R_["max"])
            refmed.append(R_["median"])
        else:
            lo.append(R_["amin"])
            hi.append(R_["amax"])
            refmed.append(float(np.median(R_["ref_amplitudes"])))
    v, lo, hi, refmed = map(np.array, (v, lo, hi, refmed))
    if len(v) < 5:
        return None
    mid = 0.5 * (lo + hi)
    rat = v / mid
    inr = (v >= lo - TOL) & (v <= hi + TOL)
    boot = [np.median(rat[rng.integers(0, len(rat), len(rat))]) for _ in range(2000)]
    lr = np.log10(np.clip(rat, 1e-3, None))
    out = {"n": int(len(v)), "in_range": float(np.mean(inr)), "median_ratio_mid": float(np.median(rat)),
           "median_ratio_mid_68": [float(np.percentile(boot, 16)), float(np.percentile(boot, 84))],
           "median_ratio_refmed": float(np.median(v / refmed)),
           "scatter_dex": float(1.4826 * np.median(np.abs(lr - np.median(lr))))}
    for name, sel in (("A<0.25", mid < 0.25), ("0.25-0.45", (mid >= 0.25) & (mid < 0.45)), ("A>=0.45", mid >= 0.45)):
        if sel.sum() >= 3:
            out["ratio_" + name] = [float(np.median(rat[sel])), int(sel.sum())]
    return out


def main(argv):
    import amp_data as D
    if "--eval-only" not in argv:
        with mp.get_context("spawn").Pool(16) as pool:
            res = dict(pool.imap_unordered(run, D.all_ids()))
        json.dump(res, open(os.path.join(ROOT, "amp_eval_results.json"), "w"), indent=1)
    res = json.load(open(os.path.join(ROOT, "amp_eval_results.json"), encoding="utf-8"))
    deb_path = os.path.join(ROOT, "amp_debias.json")
    if os.path.exists(deb_path):          # forward-model noise correction of APP_FS_med (amp_debias.py)
        deb = json.load(open(deb_path, encoding="utf-8"))
        for k, r in res.items():
            if k in deb:
                r["APP_FS_med_deb"] = deb[k].get("A_corr", np.nan)
    dense = json.load(open(os.path.join(ROOT, "data", "lcdb_dense_refs.json"), encoding="utf-8"))
    for k, r in res.items():
        r["dense"] = dense.get(k.split(":")[1])
    groups = {"validation (all), LCDB summary": ([r for k, r in res.items() if k.startswith("val:")], "lcdb"),
              "validation (snr>=5), LCDB summary": ([r for k, r in res.items() if k.startswith("val:") and (r["snr"] or 0) >= 5], "lcdb"),
              "validation (all), dense-photometry refs only": ([r for k, r in res.items() if k.startswith("val:")], "dense"),
              "paper (LCDB-matched), LCDB summary": ([r for k, r in res.items() if k.startswith("paper:") and r["lcdb"]], "lcdb")}
    rng = np.random.default_rng(1)
    summary = {}
    for g, (rows, ref) in groups.items():
        summary[g] = {}
        print(f"\n=== {g}: {len(rows)} asteroids")
        print(f"{'estimator':18s} n   in-range  ratio(mid) [68%]      ratio(ref median)  scatter  | A<0.25  0.25-0.45  >=0.45")
        for k in KEYS:
            m = metrics(rows, k, rng, ref)
            if not m:
                continue
            summary[g][k] = m
            b = "  ".join(f"{m.get('ratio_' + x, [np.nan])[0]:.2f}" for x in ("A<0.25", "0.25-0.45", "A>=0.45"))
            print(f"{k:18s} {m['n']:3d}  {m['in_range']:.2f}     {m['median_ratio_mid']:.2f} [{m['median_ratio_mid_68'][0]:.2f}-"
                  f"{m['median_ratio_mid_68'][1]:.2f}]      {m['median_ratio_refmed']:.2f}             {m['scatter_dex']:.3f}   | {b}")
    json.dump(summary, open(os.path.join(ROOT, "amp_eval_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
