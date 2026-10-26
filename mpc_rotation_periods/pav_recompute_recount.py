"""
Recounts of the manuscript statements that use the separate-dataset periods P_av (Section sec:compare, Table tab:slow),
for the published P_av, for the published P_av with the (3716) typo corrected, and for the P_av recomputed with the
correct light-time sign (pav_recompute.json).  Combined periods from adopted.json (candidate P for status
"undetermined", as in the manuscript).

Classification (main sample, 44 asteroids):
  agree        |P_av / P - 1| <= 0.5 %
  daily alias  otherwise, if the second-harmonic frequencies 2/P_av and 2/P (cycles/day) differ by an integer k >= 1
               or add up to 1 cycle/day, within ALIAS_TOL; |deviation| > 0.01 c/d is called "approximate"
  other        the rest
Run: python pav_recompute_recount.py   -> pav_recompute_recount.json (printed summary)
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
ALIAS_TOL = 0.02          # c/d
AGREE = 0.005
SLOW = [2497, 2503, 4331, 3459, 3891, 4478, 4129, 4482, 1944, 3173, 3216, 4371]


def classify(P_av, P):
    d = P_av / P - 1
    if abs(d) <= AGREE:
        return dict(cls="agree", rel=d)
    f2a, f2 = 48.0 / P_av, 48.0 / P
    diff, s = f2a - f2, f2a + f2
    k = round(abs(diff))
    if k >= 1 and abs(abs(diff) - k) <= ALIAS_TOL:
        return dict(cls="alias", rel=d, k=int(k), dev=float(abs(diff) - k), kind="difference",
                    approx=bool(abs(abs(diff) - k) > 0.01))
    if abs(s - 1) <= ALIAS_TOL:          # (4129): the two second-harmonic frequencies add up to 1 c/d
        return dict(cls="alias", rel=d, k=1, dev=float(s - 1), kind="sum", approx=bool(abs(s - 1) > 0.01))
    return dict(cls="other", rel=d, d2f=float(diff))


def recount(values, adopted):
    out = {}
    for k, P_av in values.items():
        out[k] = classify(P_av, adopted[k]["P"])
    agree = sorted(k for k, v in out.items() if v["cls"] == "agree")
    alias = sorted(k for k, v in out.items() if v["cls"] == "alias")
    other = sorted(k for k, v in out.items() if v["cls"] == "other")
    med = float(np.median([abs(out[k]["rel"]) for k in agree])) if agree else None
    return dict(per_asteroid=out, n_agree=len(agree), median_abs_rel_agree=med, agree=agree, n_alias=len(alias),
                alias=alias, approximate_alias=[k for k in alias if out[k]["approx"]], n_other=len(other), other=other,
                other_rel=[(k, out[k]["rel"]) for k in other])


def main():
    pub = json.load(open(os.path.join(ROOT, "published.json"), encoding="utf-8"))
    ad = json.load(open(os.path.join(ROOT, "adopted.json"), encoding="utf-8"))
    J = json.load(open(os.path.join(ROOT, "pav_recompute.json"), encoding="utf-8"))["asteroids"]
    main_sample = [k for k in pub if not k.startswith("_") and pub[k].get("P_av") is not None]
    sets = {
        "published": {k: pub[k]["P_av"] for k in main_sample},
        "published_typo_fixed": {k: (J[k]["P_av_old"] if k == "3716" else pub[k]["P_av"]) for k in main_sample},
        "new": {k: J[k]["P_av_new"] for k in main_sample},
        # the sign correction alone, applied to the published values (3716 keeps the typo)
        "sign_only": {k: pub[k]["P_av"] * (1 + J[k]["rel_change"]) for k in main_sample},
    }
    # worst case over all variants: the variant with the largest |change| for each asteroid
    worst = {}
    for k in main_sample:
        r = J[k]
        cands = [r["rel_change"], r.get("rel_change_emul", 0.0)]
        if "rel_change_diapmax_consistent" in r:
            cands.append(r["rel_change_diapmax_consistent"])
        for v in ("S", "R_rule3", "R_all"):
            if v in r and r[v].get("summary"):
                cands += [r[v]["summary"]["delta_sign"] / r["P_av_old"], r[v]["summary"]["delta_sign_emul"] / r["P_av_old"]]
        c = max(cands, key=abs)
        worst[k] = r["P_av_old"] * (1 + c)
    sets["new_worst_variant"] = worst
    res = {name: recount(vals, ad) for name, vals in sets.items()}
    res["slow_table"] = {str(k): dict(P_av_published=pub[str(k)]["P_av"], P_av_old=J[str(k)]["P_av_old"],
                                      P_av_new=J[str(k)]["P_av_new"], rounded_new=round(J[str(k)]["P_av_new"], 3))
                         for k in SLOW}
    res["table3_last_digit_changes"] = {k: dict(published=pub[k]["P_av"], new_rounded=round(J[k]["P_av_new"], 3))
                                        for k in main_sample if round(J[k]["P_av_new"], 3) != pub[k]["P_av"]}
    res["meta"] = dict(ALIAS_TOL=ALIAS_TOL, AGREE=AGREE)
    with open(os.path.join(ROOT, "pav_recompute_recount.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    for name in sets:
        r = res[name]
        print(f"{name:22s} agree {r['n_agree']:2d} (median {100*r['median_abs_rel_agree']:.3f} %)  alias {r['n_alias']:2d} "
              f"{r['alias']} approx {r['approximate_alias']}  other {r['n_other']} "
              f"{[(k, round(100*v, 2)) for k, v in r['other_rel']]}")
    print("Table 3 last-digit changes:", res["table3_last_digit_changes"])
    print("slow:", {k: (v["P_av_published"], v["rounded_new"]) for k, v in res["slow_table"].items()})


if __name__ == "__main__":
    main()
