"""
Refined positions of the adopted periods in the final data. The adopted periods of adopted.json are the comb teeth
chosen in the review; their positions are taken from the candidate list of rebootstrap2.py for the data the grade is
based on (results_geo/<N>/ where the aspect model is significant, otherwise results/<N>/): the candidate in the same
comb tooth (within 1.5 resolution elements). finalize.py applies the refined positions (period_refine.json).
Usage: python refine_periods.py
"""
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
A = json.load(open(os.path.join(ROOT, "adopted.json"), encoding="utf-8"))
out = {}
for num, ad in sorted(A.items(), key=lambda kv: int(kv[0])):
    for sub in ("results_geo", "results"):
        f = os.path.join(ROOT, sub, num, f"{num}_rebootstrap2.json")
        if os.path.exists(f):
            v = json.load(open(f, encoding="utf-8"))
            break
    P = ad.get("P_review", ad["P"])
    best = None
    for c in v["primary"] + v["extended"]:
        dP = c.get("dP") or P ** 2 / 24.0 / v["baseline_days"]
        if abs(c["P"] - P) <= 1.5 * dP and (best is None or abs(c["P"] - P) < abs(best["P"] - P)):
            best = c
    if best is None:
        print(f"{num}: no candidate in the tooth of {P}")
        continue
    out[num] = {"P_review": P, "P": best["P"], "source": sub, "shift": best["P"] - P,
                "shift_res": (best["P"] - P) / (P ** 2 / 24.0 / v["baseline_days"])}
json.dump(out, open(os.path.join(ROOT, "period_refine.json"), "w"), indent=1)
for num, r in out.items():
    flag = "  <--" if abs(r["shift_res"]) > 0.5 else ""
    print(f"{num}: {r['P_review']} -> {r['P']:.6f} ({r['shift_res']:+.2f} resolution elements, {r['source']}){flag}")
