"""
(2564) and (2993): the blind periods (2.9585 and 4.4082 h) agree with the sidereal periods of the convex models of
Durech et al. (2018, 2020), 2.958216 and 4.409 h, whereas the LCDB summary periods are 3.01 and 4.456 h. Convex
inversion of our MPC data from the fine grid of 60 initial poles (fine_checks.fine_poles) started at both periods,
and the chi2 of the two compared. Run on the HPC (Linux DAMIT build).
Usage: python lcdb_doubt_check.py [--workers 4]      Writes lcdb_doubt_check.json
"""
import concurrent.futures as cf
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
CASES = {"2564": {"blind": 2.958527, "lcdb": 3.01}, "2993": {"blind": 4.408213, "lcdb": 4.456}}


def job(args):
    n, which, P = args
    import fine_checks as F
    run = os.path.join(ROOT, "damit_runs", f"val_{n}_lc{which}")
    F.clone(os.path.join(ROOT, "damit_runs", f"val_{n}"), run, scan=False)
    pj = F.fine_poles(run, P)
    b = pj["best"]
    ang = F.angle(run, "areas_best", f"lcd_{n}_{which}")
    return args, {"P_start": P, "P_final": b["P"], "dev": b["dev"], "lambda": b["lambda"], "beta": b["beta"],
                  "dark_pct": b.get("dark_pct"), "angle_best": ang}


if __name__ == "__main__":
    from damit_eval import chi2_threshold
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4
    tasks = [(n, w, c[w]) for n, c in CASES.items() for w in ("blind", "lcdb")]
    res = {}
    with cf.ProcessPoolExecutor(workers) as ex:
        for (n, w, P), r in ex.map(job, tasks):
            res.setdefault(n, {})[w] = r
            print(n, w, r, flush=True)
    for n, v in res.items():
        nd = json.load(open(os.path.join(ROOT, "damit_runs", f"val_{n}", "prep.json")))["n"]
        v["threshold"] = chi2_threshold(nd - (49 + 3 + 3 - 1))
        v["chi2_ratio_lcdb_over_blind"] = (v["lcdb"]["dev"] / v["blind"]["dev"]) ** 2
    json.dump(res, open(os.path.join(ROOT, "lcdb_doubt_check.json"), "w"), indent=1, default=float)
    print(json.dumps({n: (v["chi2_ratio_lcdb_over_blind"], v["threshold"]) for n, v in res.items()}))
