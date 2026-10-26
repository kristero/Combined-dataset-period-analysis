"""Compare the corrected rebootstrap2.py (equal-length grids, exclusive own windows) with its first version
(results/<N>/<N>_rebootstrap2_v1.json, quality_final_v1.json, adopted_v1.json), and re-evaluate the numbers in
the manuscript that are derived from the rebootstrap2 results. Run after rebootstrap2.py, quality_rule.py and
finalize.py."""
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
A1 = json.load(open(os.path.join(ROOT, "adopted_v1.json"), encoding="utf-8"))
A2 = json.load(open(os.path.join(ROOT, "adopted.json"), encoding="utf-8"))
Q1 = json.load(open(os.path.join(ROOT, "quality_final_v1.json"), encoding="utf-8"))
Q2 = json.load(open(os.path.join(ROOT, "quality_final.json"), encoding="utf-8"))


def rb(num, v1):
    return json.load(open(os.path.join(RES, num, f"{num}_rebootstrap2{'_v1' if v1 else ''}.json"), encoding="utf-8"))


def in_family(Pc, P, tol=0.01):
    return any(abs(Pc - r * P) <= tol * Pc for r in (1.0, 2.0, 0.5))


problems, rows = [], {}
for num in sorted(A2, key=int):
    o, n = rb(num, True), rb(num, False)
    if n.get("windows", {}).get("powers_from") != "wide grid, masked":
        problems.append(f"{num}: rebootstrap2.json is not from the corrected version")
        continue
    if [c["P"] for c in o["primary"]] != [c["P"] for c in n["primary"]] or \
            [c["P"] for c in o["extended"]] != [c["P"] for c in n["extended"]]:
        problems.append(f"{num}: candidate lists differ")
    if o["subsets"] != n["subsets"]:
        problems.append(f"{num}: subset results differ")
    for a, b in zip(o["primary"], n["primary"]):
        if (a["same_tooth"], a["sigma_all"], a["sigma_tooth"]) != (b["same_tooth"], b["sigma_all"], b["sigma_tooth"]):
            problems.append(f"{num}: position statistics differ for P={a['P']:.5f}")
    P = A2[num]["P"]
    beat = lambda v: max((c["frac_above_best_primary"] for c in v["extended"] if in_family(c["P"], P)), default=None)
    d = {"max_dwin_u": max((abs(a["win_u"] - b["win_u"]) for a, b in zip(o["primary"], n["primary"])), default=0),
         "max_dwin_w": max((abs(a["win_w"] - b["win_w"]) for a, b in zip(o["primary"], n["primary"])), default=0),
         "max_dfrac": max((abs(a["frac_above_best_primary"] - b["frac_above_best_primary"])
                           for a, b in zip(o["extended"], n["extended"])), default=0),
         "beat": [beat(o), beat(n)]}
    rows[num] = d

print("structural checks:", "all passed" if not problems else "")
for p in problems:
    print("  PROBLEM", p)
wu = max(rows.values(), key=lambda r: r["max_dwin_u"])
print("largest change of an unweighted win fraction: %.3f" % max(r["max_dwin_u"] for r in rows.values()))
print("largest change of a weighted win fraction:   %.3f" % max(r["max_dwin_w"] for r in rows.values()))
print("largest change of frac_above_best_primary:   %.3f" % max(r["max_dfrac"] for r in rows.values()))
print("\nquality-rule inputs and results that changed:")
nchg = 0
for num in sorted(Q2, key=int):
    a, b = Q1[num], Q2[num]
    keys = ("fam_u", "fam_w", "groups", "loo", "tess", "quality_rule", "quality_final", "sigma_P")
    diff = {k: (a[k], b[k]) for k in keys if a[k] != b[k]}
    if rows[num]["beat"][0] != rows[num]["beat"][1]:
        diff["beat"] = tuple(rows[num]["beat"])
    if diff:
        nchg += 1
        print(f"  {num}: {diff}")
print(f"  ({nchg} asteroids)")
adiff = {n: [k for k in set(A1[n]) | set(A2[n]) if A1[n].get(k) != A2[n].get(k)] for n in A2}
adiff = {n: k for n, k in adiff.items() if k}
print("\nadopted.json differences:", adiff or "none")

print("\nmanuscript numbers derived from rebootstrap2 (first version -> corrected):")


def cand(v, P, tol=1e-4):
    return next(c for c in v["primary"] + v["extended"] if abs(c["P"] - P) <= tol * P)


def show(label, f):
    print(f"  {label}: {f(True)} -> {f(False)}")


show("(1798) weighted win of 3.5067 h (text: 99%)", lambda v1: round(cand(rb("1798", v1), 3.50668)["win_w"], 3))
show("(2134) unweighted wins of 24.04+48.09 h (text: 19%)",
     lambda v1: round(cand(rb("2134", v1), 24.0441, 1e-3)["win_u"] + cand(rb("2134", v1), 48.0862, 1e-3)["win_u"], 3))
show("(4482) family fraction, unweighted (text: 90%)", lambda v1: Q1["4482"]["fam_u"] if v1 else Q2["4482"]["fam_u"])
show("(3459) 547.4 h beats the primary range (text: all samples)",
     lambda v1: cand(rb("3459", v1), 547.431, 1e-3)["frac_above_best_primary"])
show("(4129) family fractions u/w (text: all samples)",
     lambda v1: ((Q1 if v1 else Q2)["4129"]["fam_u"], (Q1 if v1 else Q2)["4129"]["fam_w"]))
show("six alias asteroids A/B: fam_u, fam_w all 1.0 (text: all samples of both periodograms)",
     lambda v1: all((Q1 if v1 else Q2)[n][k] == 1.0 for n in ("2539", "2817", "3891", "4239", "4478", "4556")
                    for k in ("fam_u", "fam_w")))
for num, Pa, Pn in (("2503", 102.8388, 102.6126), ("3858", 7.08809, 7.09003), ("4556", 36.0794, 36.1327)):
    show(f"({num}) weighted power ratio neighbour/adopted (text: comparable)",
         lambda v1, num=num, Pa=Pa, Pn=Pn: round(cand(rb(num, v1), Pn, 2e-4)["power_weighted"] /
                                                 cand(rb(num, v1), Pa, 2e-4)["power_weighted"], 3))
show("(1944, 3173, 3216) rule grade (text: the rule alone would give B)",
     lambda v1: [(Q1 if v1 else Q2)[n]["quality_rule"] for n in ("1944", "3173", "3216")])
show("grade counts main sample (text: 23 A, 11 B, 10 C)",
     lambda v1: sorted(__import__("collections").Counter((Q1 if v1 else Q2)[n]["quality_final"]
                                                         for n in A2 if A2[n]["table"] in (3, 4)).items()))
