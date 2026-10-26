"""
Review checks of review_checks.py repeated for the revised test sample (retired.json; six new test asteroids):
  harmonic_test     P vs P/2 test (A02, A03) for every asteroid with an adopted period
  principal_axis    angle between the spin axis and the axis of maximum moment of inertia (A05) for every model of
                    damit_paper_table.json (accepted and rejected) and for every refined solution (areas_*) of the six
  phase_positivity  phase function at the largest observed phase angle (B11) for the accepted models
  damit_membership  published DAMIT models of the accepted models and of the six (C05)
Run on the HPC (minkowski needs the Linux DAMIT build). Writes testswap_checks.json; the parts of review_checks.json
that do not depend on the sample ((2503) half-period model, (4482) injection) are not repeated.
"""
import glob
import json
import os

import review_checks as RC
from review_inertia2 import vertices      # minkowski with the facet-area floor retry

ROOT = RC.ROOT
NEW6 = ("249", "486", "870", "1600", "1700", "7750")
T = json.load(open(os.path.join(ROOT, "damit_paper_table.json"), encoding="utf-8"))
OUT = {}


def step(name, fn):
    try:
        OUT[name] = fn()
    except Exception as e:  # noqa: BLE001
        OUT[name] = {"error": repr(e)[:500]}
    print(name, "done", flush=True)


def principal_axis():
    tmpd = os.path.join(ROOT, "review_tmp")
    os.makedirs(tmpd, exist_ok=True)
    items = [(str(m["num"]), os.path.join(ROOT, "damit_runs", f"paper_{m['num']}", "poles", m["areas"]))
             for m in T["models"] + T.get("rejected", [])]
    for n in NEW6:      # refined solutions of the six (best and refined mirror)
        for areas in sorted(glob.glob(os.path.join(ROOT, "damit_runs", f"paper_{n}", "poles", "areas_best")) +
                            glob.glob(os.path.join(ROOT, "damit_runs", f"paper_{n}", "poles", "areas_mirror*"))):
            items.append((f"{n}:{os.path.basename(areas)}", areas))
    out = {}
    for key, areas in items:
        try:
            r = RC.inertia(vertices(areas, "ts_" + key.replace(":", "_")))
            par = areas.replace("areas_", "par_")
            if os.path.exists(par):
                x = open(par).read().split()
                r["pole_P"] = [float(x[0]), float(x[1]), float(x[2])]
        except Exception as e:  # noqa: BLE001
            r = {"error": str(e)[:200]}
        out[key] = r
        print(key, r, flush=True)
    return out


def membership():
    want = {str(m["num"]) for m in T["models"]} | set(NEW6)
    T2 = {"models": [{"num": n} for n in sorted(want, key=int)]}
    json.dump(T2, open(os.path.join(ROOT, "review_tmp", "membership_nums.json"), "w"))
    orig = RC.json.load
    RC.json.load = lambda fh, *a, **k: T2 if getattr(fh, "name", "").endswith("damit_paper_table.json") else orig(fh, *a, **k)
    try:
        return RC.damit_membership()
    finally:
        RC.json.load = orig


step("harmonic_test", RC.harmonic_test)
step("principal_axis", principal_axis)
step("phase_positivity", RC.phase_positivity)
step("damit_membership", membership)
json.dump(OUT, open(os.path.join(ROOT, "testswap_checks.json"), "w", encoding="utf-8"), indent=1, default=float)
print("written testswap_checks.json")
