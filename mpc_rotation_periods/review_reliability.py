"""
Review A03: the three kinds of reliability of every adopted period, for the new columns of Tables 2 and 3.
  f_u, f_w  share of the bootstrap samples in which the period family (P, P/2, 2P and their comb teeth) outperforms
            the other candidates, unweighted and weighted periodogram (quality_final.json)
  f_tooth   share of the samples whose peak lies in the adopted comb tooth (for P > 240 h: that of P/2)
  harmonic  Delta BIC of the four-term fit at P against the two-term fit at P/2 with the number of nights as sample
            size (review_checks.json, harmonic_test; positive favours P)
Writes review_reliability.json
"""
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
AD = json.load(open(os.path.join(ROOT, "adopted.json"), encoding="utf-8"))
Q = json.load(open(os.path.join(ROOT, "quality_final.json"), encoding="utf-8"))
RC = os.path.join(ROOT, "review_checks.json")
H = json.load(open(RC, encoding="utf-8")).get("harmonic_test", {}) if os.path.exists(RC) else {}

out = {}
for n, ad in sorted(AD.items(), key=lambda kv: int(kv[0])):
    if ad.get("status") == "undetermined":
        continue
    geo = os.path.join(ROOT, "results_geo", n, f"{n}_rebootstrap2.json")
    v = json.load(open(geo if os.path.exists(geo) else os.path.join(ROOT, "results", n, f"{n}_rebootstrap2.json"),
                       encoding="utf-8"))
    P = ad["P"]
    tooth = None
    for r in (1.0, 0.5):
        c0 = next((c for c in v["primary"] if abs(c["P"] - r * P) <= 1.5 * c["dP"]), None)
        if c0:
            tooth = c0["same_tooth"]
            break
    h = H.get(n, {})
    f_u, f_w = Q[n]["fam_u"], Q[n]["fam_w"]
    if P > 240 and tooth is None:      # P/2 outside the primary range: share of samples above the best primary peak
        f_u = max((c["frac_above_best_primary"] for c in v["extended"] if abs(c["P"] - P) <= 0.01 * P), default=None)
        f_w = None
    out[n] = dict(table=ad["table"], P=P, Q=Q[n]["quality_final"], f_u=f_u, f_w=f_w,
                  f_tooth=tooth, night_blocks=v["night_blocks"], dbic_harm=h.get("dbic_eff"))
json.dump(out, open(os.path.join(ROOT, "review_reliability.json"), "w"), indent=1)
for n, r in out.items():
    print(n, r["Q"], f"{r['P']:.4f}", r["f_u"], r["f_w"], None if r["f_tooth"] is None else round(r["f_tooth"], 2),
          r["night_blocks"], r["dbic_harm"])
