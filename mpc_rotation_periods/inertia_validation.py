"""
Check 2 of the revision, specificity: principal-axis check of the convex models of the 120 validation asteroids, which
were inverted at their LCDB (correct) periods. For every run: the best refined solution and the refined mirror
solutions within the threshold (poles/mirrors.json), the angle between the spin axis and the axis of maximum moment of
inertia, and whether the model passes the acceptance criteria of the paper (unique period in the scan, at most the
mirror ambiguity, dark facet below 1 %). Run on the HPC (Linux DAMIT build).
Usage: python inertia_validation.py [--workers 16]       Writes inertia_validation.json
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ROOT, "damit_runs")


def one(d):
    sys.path.insert(0, ROOT)
    from review_checks import inertia
    from review_inertia2 import vertices
    run = os.path.join(RUNS, d)
    e = json.load(open(os.path.join(run, "eval.json")))
    C = e["C"]
    mp = os.path.join(run, "poles", "mirrors.json")
    M = json.load(open(mp)) if os.path.exists(mp) else None
    W = M["within"] if M else [{"areas": "areas_best", "dark_pct": C["dark_pct"], "lambda": C["lambda"], "beta": C["beta"]}]
    only_mirror = M["only_mirror_ambiguity"] if M else C["poles"]["only_mirror_ambiguity"]
    dark = W[0].get("dark_pct") if W[0].get("dark_pct") is not None else C["dark_pct"]
    out = {"P": C["P"], "accepted": bool(C["scan"]["unique"] and only_mirror and dark < 1.0), "dark_pct": dark,
           "scan_unique": bool(C["scan"]["unique"]), "only_mirror": bool(only_mirror), "solutions": []}
    for w in W:
        areas = os.path.join(run, "poles", w["areas"])
        try:
            r = inertia(vertices(areas, f"valinert_{d}_{w['areas']}"))
            ang = r["angle_maxI_to_spin_deg"]
        except Exception as ex:  # noqa: BLE001
            ang, r = None, {"error": str(ex)[:120]}
        out["solutions"].append({"areas": w["areas"], "lambda": w.get("lambda"), "beta": w.get("beta"), "angle": ang,
                                 "dark_pct": w.get("dark_pct"), **({"error": r["error"]} if "error" in r else {})})
    return d, out


if __name__ == "__main__":
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 16
    dirs = sorted(d for d in os.listdir(RUNS) if d.startswith("val_") and len(d.split("_")) == 2
                  and os.path.exists(os.path.join(RUNS, d, "eval.json")))
    res = {}
    with ProcessPoolExecutor(workers) as ex:
        for d, o in ex.map(one, dirs):
            res[d] = o
            print(d, o["accepted"], [s["angle"] if s["angle"] is None else round(s["angle"], 1) for s in o["solutions"]], flush=True)
    json.dump(res, open(os.path.join(ROOT, "inertia_validation.json"), "w"), indent=1)
    acc = [o for o in res.values() if o["accepted"]]
    best = [o["solutions"][0]["angle"] for o in acc if o["solutions"] and o["solutions"][0]["angle"] is not None]
    print(f"{len(res)} validation models, {len(acc)} accepted; best solution angle > 45 deg for "
          f"{sum(a > 45 for a in best)} of {len(best)} accepted models with a polyhedron")
