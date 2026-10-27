"""
Check 2 of the revision, sensitivity: convex models of the validation asteroids at a wrong period. For every validation
asteroid with a convex run at its LCDB period (damit_runs/val_<n>), the strongest candidate of the blind run (vb/,
rebootstrap2 primary candidates by unweighted power) outside the alias family of the LCDB period (P, 2P, P/2 within 1 %)
is inverted as in review_alt_period.run_alt: scan window centred on it, convexinv from 12 poles, refinement of the best
and mirror solutions, principal-axis check, chi2 relative to the model at the LCDB period. Run on the HPC.
Compare with inertia_validation.json (the same models at the correct periods).
Usage: python inertia_sensitivity.py [--workers 30] [--collect]      Writes inertia_sensitivity.json
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.abspath(__file__))
VB = os.path.join(ROOT, "vb")


def in_family(Pc, P, tol=0.01):
    return any(abs(Pc - r * P) <= tol * Pc for r in (1.0, 2.0, 0.5))


def wrong_candidate(n, P_lcdb):
    v = json.load(open(os.path.join(VB, "results", n, f"{n}_rebootstrap2.json")))
    c = [c for c in v["primary"] if not in_family(c["P"], P_lcdb)]
    return max(c, key=lambda c: c["power"]) if c else None


def one(args):
    n, P_lcdb, collect = args
    sys.path.insert(0, ROOT)
    from review_alt_period import run_alt
    c = wrong_candidate(n, P_lcdb)
    if c is None:
        return n, {"P_lcdb": P_lcdb, "error": "no candidate outside the LCDB family"}
    try:
        r = run_alt(n, c["P"], collect, kind="val", tree=VB, wide=True)
    except Exception as e:  # noqa: BLE001
        return n, {"P_lcdb": P_lcdb, "P_wrong": c["P"], "error": str(e)[:200]}
    W = r.get("within") or []
    best = W[0]["areas"] if W else None
    pa = r["principal_axis"]
    ang = [pa[w["areas"]].get("angle_maxI_to_spin_deg") for w in W if w["areas"] in pa]
    return n, {"P_lcdb": P_lcdb, "P_wrong": c["P"], "power_wrong": c["power"], "chi2_ratio": r.get("chi2_ratio_alt_over_adopted"),
               "scan_unique": (r.get("scan_test") or {}).get("unique"), "only_mirror": r.get("only_mirror_ambiguity"),
               "dark_pct": r.get("dark_pct"), "angle_best": pa.get(best, {}).get("angle_maxI_to_spin_deg") if best else None,
               "angles": ang,
               # unique None: no other minimum farther than dP in the window, so the uniqueness criterion cannot fail
               "accepted": bool((r.get("scan_test") or {}).get("unique") is not False and r.get("only_mirror_ambiguity")
                                and (r.get("dark_pct") if r.get("dark_pct") is not None else 99) < 1.0)}


if __name__ == "__main__":
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 30
    collect = "--collect" in sys.argv
    truth = {s["num"]: s["period"] for s in json.load(open(os.path.join(ROOT, "validation", "sample.json")))["sample"]}
    nums = sorted((n for n in truth if os.path.exists(os.path.join(ROOT, "damit_runs", f"val_{n}", "eval.json"))
                   and os.path.exists(os.path.join(VB, "results", n, f"{n}_rebootstrap2.json"))), key=int)
    res = {}
    with ProcessPoolExecutor(workers) as ex:
        for n, o in ex.map(one, [(n, truth[n], collect) for n in nums]):
            res[n] = o
            print(n, {k: o.get(k) for k in ("P_lcdb", "P_wrong", "chi2_ratio", "accepted", "angle_best", "error")}, flush=True)
    json.dump(res, open(os.path.join(ROOT, "inertia_sensitivity.json"), "w"), indent=1, default=float)
    ok = [o for o in res.values() if "error" not in o]
    acc = [o for o in ok if o["accepted"]]
    print(f"{len(ok)} wrong-period models; {sum(o['chi2_ratio'] > 1.0077 for o in ok if o['chi2_ratio'])} fit worse than the "
          f"correct-period model; {len(acc)} pass the acceptance criteria; of these, "
          f"{sum((o['angle_best'] or 0) > 45 for o in acc)} fail the inertia check")
