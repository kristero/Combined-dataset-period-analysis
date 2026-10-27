"""
Check 1 of the revision: blind run of the automatic pipeline on the 120 validation asteroids (tree vb/, vb_setup.py,
hpc/vb_blind.sbatch; adopted = algorithmic proposal without review) and the dominant-dataset rule.

Recovery: the blind period P against the LCDB summary period P_lcdb (validation/sample.json, U = 3 or 3-):
  same     within 1 % of P_lcdb
  harmonic within 1 % of 2 P_lcdb or P_lcdb/2 (the same alias family in the sense of the quality rule)
  wrong    otherwise
Dominant-dataset rule (formulated after the failure for (870), before this run): when one dataset holds more than half
of the measurements, grade B or better requires that either the leave-one-out subset without it or a majority of the
longitude groups selects the alias family of the adopted period; otherwise the grade is capped at C.
Revised rule (co-author comment, 27 October): a dominant dataset that by itself covers at least k rotations in each of
at least two visits (k = 3 or 5; visits split at gaps > 15 d) is sufficient for B; the rule then applies only to
datasets whose visits are short compared with the period, as the TESS visits of (870).
The rule is applied to the blind validation grades and, for comparison, to the 58 asteroids of the paper.
Usage: python vb_evaluate.py      Writes vb_evaluate.json
"""
import json
import os
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
VB = os.path.join(ROOT, "vb")


def J(*p):
    return json.load(open(os.path.join(*p), encoding="utf-8"))


def in_family(Pc, P, tol=0.01):
    return any(abs(Pc - r * P) <= tol * Pc for r in (1.0, 2.0, 0.5))


def recovery(P, P_lit):
    if abs(P - P_lit) <= 0.01 * P_lit:
        return "same"
    return "harmonic" if in_family(P, P_lit) else "wrong"


def dominant(v, P):
    """Largest dataset (from the leave-one-out subsets), its share, and the two tests of the rule."""
    subs, n = v["subsets"], v["n"]
    loo = {k.split(":", 1)[1]: s for k, s in subs.items() if k.startswith("without:")}
    if not loo:
        return None
    D = min(loo, key=lambda d: loo[d]["n"])
    share = 1 - loo[D]["n"] / n
    key = "preferred_primary_P" if P <= 240 else "preferred_P"
    groups = [s for k, s in subs.items() if k.startswith("group:") and k != "group:space"]
    g_ok = sum(in_family(s[key], P) for s in groups)
    return {"dataset": D, "share": share, "without_ok": bool(in_family(loo[D][key], P)),
            "without_P": loo[D][key], "groups": f"{g_ok}/{len(groups)}", "groups_majority": g_ok > len(groups) / 2}


def capped(q, d, k=None):
    """Grade under the rule; with k, a dominant dataset that covers at least k rotations in each of at least two
    visits counts as sufficient on its own (the revised rule)."""
    if d is None or d["share"] <= 0.5 or q not in ("A", "B"):
        return q
    if k is not None and sum(r >= k for r in d.get("visit_rotations", [])) >= 2:
        return q
    return q if (d["without_ok"] or d["groups_majority"]) else "C"


def visit_rotations(tree, n, D, P, gap=15.0):
    """Rotations of period P covered by each visit of dataset D (visits split at gaps > gap days)."""
    for sub in ("results_geo", "results"):
        p = os.path.join(tree, sub, n, f"{n}_arrays.npz")
        if os.path.exists(p):
            a = np.load(p)
            t = np.sort(a["t"][a["sheet"].astype(str) == D])
            if len(t) < 2:
                return []
            return [float((v[-1] - v[0]) * 24.0 / P) for v in np.split(t, np.where(np.diff(t) > gap)[0] + 1)]
    return None


def rb2(tree, n):
    for sub in ("results_geo", "results"):
        p = os.path.join(tree, sub, n, f"{n}_rebootstrap2.json")
        if os.path.exists(p):
            return J(p)
    return None


# blind validation run
truth = {s["num"]: s["period"] for s in J(ROOT, "validation", "sample.json")["sample"]}
AD, QF = J(VB, "adopted.json"), J(VB, "quality_final.json")
rows = []
for n, q in QF.items():
    v = rb2(VB, n)
    P = q["P"]
    d = dominant(v, P) if v else None
    if d and d["share"] > 0.5:
        d["visit_rotations"] = visit_rotations(VB, n, d["dataset"], P)
    rows.append({"num": n, "P": P, "sigma_P": q["sigma_P"], "P_lcdb": truth[n], "recovery": recovery(P, truth[n]),
                 "quality": q["quality_final"], "quality_capped": capped(q["quality_final"], d),
                 "quality_capped_k3": capped(q["quality_final"], d, 3), "quality_capped_k5": capped(q["quality_final"], d, 5),
                 "dominant": d,
                 "fam_u": q["fam_u"], "fam_w": q["fam_w"], "groups": q["groups"], "loo": q["loo"], "tess": q["tess"]})
missing = sorted(set(truth) - set(QF), key=int)

by_grade = {}
for g in "ABC":
    r = [x for x in rows if x["quality"] == g]
    by_grade[g] = {"n": len(r), **Counter(x["recovery"] for x in r)}
by_grade_capped = {}
for g in "ABC":
    r = [x for x in rows if x["quality_capped"] == g]
    by_grade_capped[g] = {"n": len(r), **Counter(x["recovery"] for x in r)}
changed = [x for x in rows if x["quality_capped"] != x["quality"]]
dom = [x for x in rows if x["dominant"] and x["dominant"]["share"] > 0.5]
summary = {"n": len(rows), "missing": missing, "recovery": dict(Counter(x["recovery"] for x in rows)),
           "by_grade": by_grade, "by_grade_with_rule": by_grade_capped,
           "n_dominant": len(dom), "dominant_datasets": dict(Counter(x["dominant"]["dataset"] for x in dom)),
           "downgraded": [(x["num"], x["quality"], x["recovery"], x["dominant"]["dataset"], round(x["dominant"]["share"], 2))
                          for x in changed],
           "downgraded_correct": sum(x["recovery"] != "wrong" for x in changed),
           "downgraded_wrong": sum(x["recovery"] == "wrong" for x in changed),
           "AB_wrong_not_caught": [(x["num"], x["quality_capped"], x["P"], x["P_lcdb"]) for x in rows
                                   if x["quality_capped"] in ("A", "B") and x["recovery"] == "wrong"]}

# paper sample, for comparison
PA, PQ = J(ROOT, "adopted.json"), J(ROOT, "quality_final.json")
paper = []
for n, q in PQ.items():
    v = rb2(ROOT, n)
    P = q["P"]
    d = dominant(v, P) if v else None
    if d and d["share"] > 0.5:
        d["visit_rotations"] = visit_rotations(ROOT, n, d["dataset"], P)
    paper.append({"num": n, "table": PA[n]["table"], "P": P, "quality": q["quality_final"],
                  "quality_capped": capped(q["quality_final"], d), "quality_capped_k3": capped(q["quality_final"], d, 3),
                  "quality_capped_k5": capped(q["quality_final"], d, 5), "dominant": d})
summary["paper_n_dominant"] = sum(1 for x in paper if x["dominant"] and x["dominant"]["share"] > 0.5)
summary["paper_dominant"] = [(x["num"], x["quality"], x["dominant"]["dataset"], round(x["dominant"]["share"], 2),
                              x["dominant"]["without_ok"], x["dominant"]["groups"]) for x in paper
                             if x["dominant"] and x["dominant"]["share"] > 0.5]
summary["paper_changed"] = [(x["num"], x["quality"], x["quality_capped"]) for x in paper if x["quality_capped"] != x["quality"]]
for k in (3, 5):
    summary[f"revised_k{k}"] = {
        "validation_changed": [(x["num"], x["quality"], x[f"quality_capped_k{k}"], x["recovery"]) for x in rows
                               if x[f"quality_capped_k{k}"] != x["quality"]],
        "paper_changed": [(x["num"], x["quality"], x[f"quality_capped_k{k}"]) for x in paper
                          if x[f"quality_capped_k{k}"] != x["quality"]],
        "validation_exempt": sum(1 for x in rows if x["dominant"] and x["dominant"]["share"] > 0.5
                                 and sum(r >= k for r in x["dominant"].get("visit_rotations") or []) >= 2),
        "paper_exempt": [x["num"] for x in paper if x["dominant"] and x["dominant"]["share"] > 0.5
                         and sum(r >= k for r in x["dominant"].get("visit_rotations") or []) >= 2]}
summary["paper_visit_rotations"] = {x["num"]: [round(r, 1) for r in x["dominant"]["visit_rotations"] or []]
                                    for x in paper if x["dominant"] and x["dominant"]["share"] > 0.5}
summary["validation_visit_rotations"] = {x["num"]: [round(r, 1) for r in x["dominant"]["visit_rotations"] or []]
                                         for x in rows if x["dominant"] and x["dominant"]["share"] > 0.5}

json.dump({"summary": summary, "validation": rows, "paper": paper},
          open(os.path.join(ROOT, "vb_evaluate.json"), "w"), indent=1, default=float)
print(json.dumps(summary, indent=1, default=float))
print("\nwrong or harmonic blind periods:")
for x in sorted(rows, key=lambda x: int(x["num"])):
    if x["recovery"] != "same":
        d = x["dominant"] or {}
        print(f"  ({x['num']}) P={x['P']:.5f} lcdb={x['P_lcdb']} {x['recovery']:8} Q={x['quality']}->{x['quality_capped']} "
              f"famU={x['fam_u']} famW={x['fam_w']} groups={x['groups']} loo={x['loo']} tess={x['tess']} "
              f"dom={d.get('dataset')} {d.get('share', 0):.2f}")
