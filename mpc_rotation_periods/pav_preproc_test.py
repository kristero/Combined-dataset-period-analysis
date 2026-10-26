"""
Sensitivity of the separate-dataset periods to the pre-processing.

The separate-dataset approach (Eglitis 2025) reduces every dataset with a linear magnitude-phase relation fitted at
alpha > 7 deg (Equations eq:reduced_rel - eq:linphase); the combined approach uses the HG1G2 phase curve of the
reference dataset at all phase angles, with outlier removal and dataset screening (results/<N>/<N>_arrays.npz).
For the 44 main-sample asteroids every dataset of the separate-dataset analysis that also passed the screening of the
combined approach is analysed with both reductions,
  L: linear magnitude-phase relation at alpha > 7 deg, correct light-time sign (pav_recompute.load_obs);
  H: the phase-reduced magnitudes of the combined approach (all phase angles; no aspect correction),
and for each dataset
  (a) the periodogram peak nearest to the published P_av is followed (pav_recompute.find_peak), as in the light-time
      check of the separate-dataset periods;
  (b) the period that the dataset prefers among the candidate periods of the combined approach (exact chi2 power
      around every candidate, as in Figure 7; primary-range candidates when P <= 240 h).
P_av follows Eq. eq:pav with the weights w (peak power), N and R^2 (R^2 of the linear magnitude-phase fit in both
cases). Writes pav_preproc_test.json. Run on the LU HPC: hpc/pav_preproc.sbatch
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import pav_recompute as PR

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")


def preferred(t, y, cands, primary_only):
    import rebootstrap2 as r2
    bb = max(float(t.max() - t.min()), 1.0)
    use = [c for c in cands if (not primary_only) or c["primary"]]
    vals = [float(r2.power(t, y, r2.grid(c["f"], bb, 3.0, spp=20), exact=True).max()) for c in use]
    k = int(np.argmax(vals))
    return float(use[k]["P"]), float(vals[k])


def same_period(p1, p2, tol=1e-3):
    return abs(p1 - p2) <= tol * p2


def process(num):
    pub = json.load(open(os.path.join(ROOT, "published.json"), encoding="utf-8"))[str(num)]
    ad = json.load(open(os.path.join(ROOT, "adopted.json"), encoding="utf-8"))[str(num)]
    P_av, P = pub["P_av"], ad["P"]
    if num == 3716:
        P_av = 10.473132                      # transcription error of the published value (make_tables.PAV_FIX)
    obs = PR.load_obs(num)
    _, alls, _ = PR.select_datasets(obs, P_av)
    a = np.load(os.path.join(RES, str(num), f"{num}_arrays.npz"), allow_pickle=False)
    v = json.load(open(os.path.join(RES, str(num), f"{num}_rebootstrap2.json"), encoding="utf-8"))
    cands = [dict(c, primary=True) for c in v["primary"]] + [dict(c, primary=False) for c in v["extended"]]
    sheets = a["sheet"].astype(str)
    rows, screened_out = [], []
    for ds in alls:
        m = sheets == ds
        if m.sum() < 30:
            screened_out.append(ds)
            continue
        o = obs[ds]
        L = dict(t=o["t_c"], y=o["y"], N=o["N"])
        Hd = dict(t=a["t"][m].astype(float), y=a["y"][m].astype(float), N=int(m.sum()))
        rec = dict(ds=ds, R2=o["R2"])
        for key, d in (("L", L), ("H", Hd)):
            fp = PR.find_peak(d["t"], d["y"], P_av, prom_frac=0.1)
            pp, pw = preferred(d["t"], d["y"], cands, primary_only=P <= 240)
            rec[key] = dict(N=d["N"], follow_p=fp["p"] if fp else None, follow_w=fp["w"] if fp else None,
                            pref_p=pp, pref_w=pw)
        rec["pref_same"] = same_period(rec["L"]["pref_p"], rec["H"]["pref_p"])
        rows.append(rec)
    out = dict(asteroid=num, P=P, P_av_published=pub["P_av"], P_av_ref=P_av, status=ad.get("status"),
               quality=ad.get("quality"), datasets=rows, screened_out=screened_out)
    ok = [r for r in rows if r["L"]["follow_p"] and r["H"]["follow_p"]]
    R2 = [r["R2"] for r in ok]
    for key in ("L", "H"):
        if ok:
            out[f"P_av_follow_{key}"] = PR.pav([r[key]["follow_p"] for r in ok], [r[key]["follow_w"] for r in ok],
                                               [r[key]["N"] for r in ok], R2)
        if rows:
            out[f"P_av_pref_{key}"] = PR.pav([r[key]["pref_p"] for r in rows], [r[key]["pref_w"] for r in rows],
                                             [r[key]["N"] for r in rows], [r["R2"] for r in rows])
    print(num, len(rows), "datasets", {k: round(out[k], 5) for k in out if k.startswith("P_av_")}, flush=True)
    return out


def stats(x):
    x = np.asarray(x, float)
    q = np.percentile(x, [50, 84, 100])
    return dict(n=int(len(x)), median=float(q[0]), p84=float(q[1]), max=float(q[2]))


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    pub = json.load(open(os.path.join(ROOT, "published.json"), encoding="utf-8"))
    nums = sorted(int(k) for k, v in pub.items() if not k.startswith("_") and v.get("P_av") is not None)
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(process, nums))
    rel = lambda a, b: abs(a - b) / b
    S = {}
    fol = [rel(r["P_av_follow_H"], r["P_av_follow_L"]) for r in res if "P_av_follow_H" in r and "P_av_follow_L" in r]
    pre = [rel(r["P_av_pref_H"], r["P_av_pref_L"]) for r in res if "P_av_pref_H" in r and "P_av_pref_L" in r]
    S["follow_rel_change"] = stats(fol)
    S["pref_rel_change"] = stats(pre)
    S["pref_changed_gt_0p5pct"] = int(sum(x > 0.005 for x in pre))
    ds = [d for r in res for d in r["datasets"]]
    S["n_datasets"] = len(ds)
    S["n_datasets_pref_changed"] = int(sum(not d["pref_same"] for d in ds))
    S["n_screened_out"] = int(sum(len(r["screened_out"]) for r in res))
    det = [r for r in res if r.get("status") != "undetermined"]
    for key in ("pref_L", "pref_H"):
        S[f"agree_P_within_0p5pct_{key}"] = int(sum(rel(r[f"P_av_{key}"], r["P"]) <= 0.005 for r in det
                                                      if f"P_av_{key}" in r))
    S["agree_P_within_0p5pct_published"] = int(sum(rel(r["P_av_ref"], r["P"]) <= 0.005 for r in det))
    S["n_determined"] = len(det)
    json.dump(dict(summary=S, asteroids=res), open(os.path.join(ROOT, "pav_preproc_test.json"), "w",
                                                   encoding="utf-8"), indent=1)
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
