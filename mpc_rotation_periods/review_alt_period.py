"""
Convex models at competing candidate periods (check 2 of the revision: a model at the adopted period that fails the
principal-axis check should prompt models at the competing periods). For asteroid N and each competing period P_alt:
a copy of damit_runs/paper_N with the scan interval of damit_prep.scan_interval centred on P_alt (sigma from the
bootstrap of that candidate), period_scan, convexinv from the 12 initial poles, refinement of the best and mirror
solutions (damit_mirrors.do_run), the principal-axis check of the refined solutions, and the fit relative to the
model at the adopted period. Run on the HPC (Linux DAMIT build).
Usage: python review_alt_period.py N P_alt [P_alt ...] [--collect]    Writes review_alt_period.json (merged per asteroid)
"""
import glob
import json
import os
import shutil
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import damit_mirrors as DM  # noqa: E402
import damit_run as R  # noqa: E402
from damit_prep import scan_interval  # noqa: E402
from review_checks import inertia  # noqa: E402
from review_inertia2 import vertices  # noqa: E402

OUT = os.path.join(ROOT, "review_alt_period.json")


def candidate_sigma(num, P, tree=ROOT, wide=False):
    for sub in ("results_geo", "results"):
        p = os.path.join(tree, sub, num, f"{num}_rebootstrap2.json")
        if os.path.exists(p):
            v = json.load(open(p))
            c = min(v["primary"] + v["extended"], key=lambda c: abs(c["P"] - P))
            if abs(c["P"] - P) <= 0.01 * P:
                s = c.get("sigma_tooth") or c.get("sigma_all") or 1.5e-5 * P
                if wide:   # as the uncertainty of quality_rule.py, so the window matches the runs at adopted periods
                    s = max(c["sigma_tooth"] if (c.get("same_tooth", 0) >= 0.9 and c.get("sigma_tooth")) else c["sigma_all"],
                            c["P"] ** 2 / 24.0 / v["baseline_days"] / 2.0)
                return s, c["P"]
    return 1.5e-5 * P, P


def run_alt(num, P_alt, collect=False, kind="paper", tree=ROOT, wide=False):
    """kind/tree: 'paper' with the rebootstrap2 candidates of ROOT, or 'val' with those of the blind run (vb/).
    wide: scan window from the quality-rule uncertainty instead of the bootstrap scatter within the tooth."""
    src = os.path.join(ROOT, "damit_runs", f"{kind}_{num}")
    sig, P_c = candidate_sigma(num, P_alt, tree, wide)
    run = os.path.join(ROOT, "damit_runs", f"{kind}_{num}_alt{P_c:.4f}".replace(".", "p"))
    os.makedirs(run, exist_ok=True)
    for f in ("lcs", "prep.npz"):
        shutil.copy(os.path.join(src, f), os.path.join(run, f))
    info = json.load(open(os.path.join(src, "prep.json")))
    z = np.load(os.path.join(src, "prep.npz"))
    r_mean = float(np.mean(np.linalg.norm(z["sun"], axis=1)))
    span_h = info["span_days"] * 24
    P_lo, P_hi, dP = scan_interval(P_c, span_h, r_mean, sig)
    info.update({"P0": P_c, "sigma_P": sig, "P_lo": P_lo, "P_hi": P_hi, "dP": dP,
                 "n_trial_periods": int(np.ceil((P_hi - P_lo) / (0.8 * dP))) + 1, "alt_of": num})
    json.dump(info, open(os.path.join(run, "prep.json"), "w"), indent=1)
    if not (collect and os.path.exists(os.path.join(run, "poles", "mirrors.json"))):
        print(num, P_c, "scan", R.do_scan(run), flush=True)
        print(num, P_c, "poles", R.do_poles(run), flush=True)
        print(num, P_c, "mirrors", DM.do_run(run), flush=True)
    mj = json.load(open(os.path.join(run, "poles", "mirrors.json")))
    sc = R.read_scan(run)
    ok = np.isfinite(sc["dev"])
    out = {"P_candidate": P_c, "sigma": sig, "scan": [P_lo, P_hi], "P_scan_best": float(sc["P"][ok][np.argmin(sc["dev"][ok])]),
           "within": mj.get("within"), "only_mirror_ambiguity": mj.get("only_mirror_ambiguity"), "principal_axis": {}}
    for areas in sorted(glob.glob(os.path.join(run, "poles", "areas_best")) + glob.glob(os.path.join(run, "poles", "areas_mirror*"))):
        k = os.path.basename(areas)
        try:
            out["principal_axis"][k] = inertia(vertices(areas, f"alt{num}_{P_c:.3f}_{k}"))
        except Exception as e:  # noqa: BLE001
            out["principal_axis"][k] = {"error": str(e)[:200]}
        print(num, P_c, k, out["principal_axis"][k], flush=True)
    m0p = os.path.join(src, "poles", "mirrors.json")
    dev0 = (json.load(open(m0p)).get("within") or [{}])[0].get("dev") if os.path.exists(m0p) else None
    dev0 = dev0 or json.load(open(os.path.join(src, "eval.json")))["C"]["dev"]
    if mj.get("within"):
        out["chi2_ratio_alt_over_adopted"] = (mj["within"][0]["dev"] / dev0) ** 2
        out["dark_pct"] = mj["within"][0].get("dark_pct")
    from damit_eval import scan_unique
    out["scan_test"] = scan_unique(run, info["n"], 49 + 3 + 3 - 1)  # same uniqueness test and n_par as damit_eval
    return out


if __name__ == "__main__":
    collect = "--collect" in sys.argv          # reuse finished runs, only redo the checks and the summary
    args = [a for a in sys.argv[1:] if a != "--collect"]
    num, alts = args[0], [float(x) for x in args[1:]]
    for P in alts:
        r = run_alt(num, P, collect)
        res = json.load(open(OUT)) if os.path.exists(OUT) else {}   # reread: several asteroids may run in parallel
        res.setdefault(num, {})[f"{P:g}"] = r
        json.dump(res, open(OUT, "w"), indent=1, default=float)
    print(json.dumps({k: {kk: (vv if not isinstance(vv, dict) else "...") for kk, vv in v.items()} for k, v in res[num].items()},
                     default=float)[:3000])
