"""
Effect of the 703/G96 zero-point correction (datasets split at 2018 July 1; zp_split.py) on the period analysis.
For every asteroid, the adopted period is matched to its comb tooth in the candidate list of rebootstrap2.py for the
unsplit data (results_geo_unsplit/ where the aspect model was significant, else results_unsplit/) and for the corrected
data (results_geo/ where the aspect model is significant now, else results/), and compared: refined period (shift in
units of the unsplit uncertainty), family fractions of the unweighted and weighted bootstrap, subsets, uncertainty and
quality by the rule of quality_rule.py.
Usage: python compare_zp.py  ->  compare_zp.json and a printed table
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import quality_rule as Q  # noqa: E402

A = json.load(open(os.path.join(ROOT, "adopted.json"), encoding="utf-8"))
A0 = json.load(open(os.path.join(ROOT, "unsplit_snapshot", "adopted.json"), encoding="utf-8"))


def load(subs, num):
    for sub in subs:
        f = os.path.join(ROOT, sub, num, f"{num}_rebootstrap2.json")
        if os.path.exists(f):
            return json.load(open(f, encoding="utf-8")), sub
    return None, None


def tooth(v, P):
    best = None
    for c in v["primary"] + v["extended"]:
        dP = c.get("dP") or P ** 2 / 24.0 / v["baseline_days"]
        if abs(c["P"] - P) <= 1.5 * dP and (best is None or abs(c["P"] - P) < abs(best["P"] - P)):
            best = c
    return best


out = {}
for num in sorted(A, key=int):
    ad, ad0 = A[num], A0[num]
    v0, s0 = load(("results_geo_unsplit", "results_unsplit"), num)
    v1, s1 = load(("results_geo", "results"), num)
    row = {"P0": ad0["P"], "P": ad["P"], "status": ad.get("status"), "src0": s0, "src": s1}
    for lab, v, a in (("orig", v0, ad0), ("zp", v1, ad)):
        g = Q.grade(num, a, v)
        c = tooth(v, a.get("P_review", a["P"]))
        row[lab] = {"P_tooth": c["P"] if c else None, "fam_u": g["fam_u"], "fam_w": g["fam_w"], "groups": g["groups"],
                    "loo": g["loo"], "tess": g["tess"], "sigma": g["sigma_P"], "q_rule": g["quality_rule"],
                    "q_final": g["quality_final"]}
    o, z = row["orig"], row["zp"]
    row["dP_sigma"] = (z["P_tooth"] - o["P_tooth"]) / max(o["sigma"], 1e-12) if o["P_tooth"] and z["P_tooth"] else None
    row["changed"] = [k for k in ("q_rule", "q_final") if o[k] != z[k]] + (
        ["family_fraction"] if abs(o["fam_u"] - z["fam_u"]) > 0.05 or abs(o["fam_w"] - z["fam_w"]) > 0.05 else []) + (
        ["tooth"] if row["dP_sigma"] is not None and abs(row["dP_sigma"]) > 1 else []) + (
        ["sigma"] if abs(z["sigma"] / o["sigma"] - 1) > 0.25 else []) + (["aspect_set"] if (s0 == "results_geo_unsplit") != (s1 == "results_geo") else [])
    out[num] = row
json.dump(out, open(os.path.join(ROOT, "compare_zp.json"), "w"), indent=1)
for num, r in out.items():
    o, z = r["orig"], r["zp"]
    if not r["changed"] and "--all" not in sys.argv:
        continue
    print(f"{num:>5} P {r['P0']}->{r['P']:.6g} {r.get('status') or '':<12} Q {o['q_final']}->{z['q_final']} (rule {o['q_rule']}->{z['q_rule']}) "
          f"fam_u {o['fam_u']:.2f}->{z['fam_u']:.2f} fam_w {o['fam_w']:.2f}->{z['fam_w']:.2f} loo {o['loo']}->{z['loo']} "
          f"grp {o['groups']}->{z['groups']} sig {o['sigma']:.2g}->{z['sigma']:.2g} dP/sig {r['dP_sigma'] if r['dP_sigma'] is None else round(r['dP_sigma'], 2)} "
          f"[{r['src0']} -> {r['src']}]  <-- {','.join(r['changed'])}")
print(len([r for r in out.values() if r["changed"]]), "of", len(out), "with changes")
