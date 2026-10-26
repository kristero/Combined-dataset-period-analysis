"""
Final, uniformly applied quality rule (as stated in the paper), evaluated on the corrected bootstrap and exact
subset checks of rebootstrap2.py.

Alias family of the adopted period P: all candidates within 1% of P, 2P or P/2 (harmonics and their
inter-apparition comb teeth; the choice of tooth is reflected in the uncertainty).
  A: the alias family wins >= 95% of the bootstrap samples of both the unweighted and the weighted periodogram;
     all leave-one-dataset-out subsets select it; the daily-alias choice is confirmed independently, i.e. by at
     least two longitude groups (>= 2/3 of the groups agreeing) or by TESS; TESS does not contradict it; the
     period is not commensurate with the day; P <= 240 h.
  B: the alias family wins >= 80% of the samples in at least one of the two periodograms, and the majority of
     the longitude-group and leave-one-out tests agree; day-commensurate periods only with TESS support;
     P > 240 h only if the long-period family beats the primary-range candidates in >= 90% of the samples (or its
     P/2 member wins >= 90% within the primary range), is selected by all leave-one-out subsets, and is confirmed
     independently (longitude groups or TESS as for A, or dense photometry from the literature).
  C: otherwise.
Documented exceptions (review judgement, see paper): EXCEPTIONS below.
"""
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
A = json.load(open(os.path.join(ROOT, "adopted.json"), encoding="utf-8"))
EXCEPTIONS = {
    "1717": ("C", "two independent periodicities (18.21 and 5.1475 h); the literature period is the weaker one"),
    "1798": ("B", "unweighted periodogram dominated by day-commensurate zero-point artifacts; weighted periodogram "
                  "(family fraction 1.00) and TESS alone give the period"),
    "4482": ("C", "comb-tooth ambiguity; conflicts with the period from dense photometry (Fekete et al. 2024)"),
    "1944": ("C", "long-period signal comparable to the apparition length; rotation not established"),
    "3173": ("C", "long-period signal comparable to the apparition length; rotation not established"),
    "3216": ("C", "long-period signal comparable to the apparition length; rotation not established"),
}
INDEPENDENT_SUPPORT_LONG ={"3459": "Polakis (2020), dense photometry, 532.9 h"}


def in_family(Pc, P, tol=0.01):
    return any(abs(Pc - r * P) <= tol * Pc for r in (1.0, 2.0, 0.5))


def day_comm(P, tol=0.01):
    f2 = 48.0 / P
    return round(f2) >= 1 and abs(f2 - round(f2)) < tol


def grade(num, ad, v):
    """Quality grade and uncertainty of asteroid num with adopted entry ad from rebootstrap2 output v."""
    P = ad["P"]
    prim, ext = v["primary"], v["extended"]
    fam_u = sum(c["win_u"] for c in prim if in_family(c["P"], P))
    fam_w = sum(c["win_w"] for c in prim if in_family(c["P"], P))
    subs = v["subsets"]
    key = "preferred_primary_P" if P <= 240 else "preferred_P"
    ok = lambda s: in_family(s[key], P) or (P > 240 and in_family(s["preferred_primary_P"], P))
    groups = [s for k, s in subs.items() if k.startswith("group:") and k != "group:space"]
    loo = [s for k, s in subs.items() if k.startswith("without:")]
    tess = subs.get("group:space") or subs.get("dataset:C57G")
    g_ok, loo_ok = sum(map(ok, groups)), sum(map(ok, loo))
    t_ok = None if tess is None else ok(tess)
    dc = day_comm(P)
    indep = (len(groups) >= 2 and g_ok >= 2 / 3 * len(groups)) or t_ok is True
    if P <= 240:
        if fam_u >= 0.95 and fam_w >= 0.95 and loo_ok == len(loo) and indep and t_ok is not False and not dc:
            q = "A"
        elif max(fam_u, fam_w) >= 0.80 and (g_ok + loo_ok) > 0.5 * (len(groups) + len(loo)) and (not dc or t_ok):
            q = "B"
        else:
            q = "C"
    else:
        beat = max((c["frac_above_best_primary"] for c in ext if in_family(c["P"], P)), default=0.0)
        half_prim = max((c["win_u"] for c in prim if in_family(c["P"], P)), default=0.0)
        strong = beat >= 0.90 or fam_u >= 0.90 or half_prim >= 0.90
        q = "B" if (strong and loo_ok == len(loo) and (num in INDEPENDENT_SUPPORT_LONG or indep)) else "C"
    rule_q = q
    reason = "rule"
    if num in EXCEPTIONS:
        q, reason = EXCEPTIONS[num]
    # uncertainty: bootstrap scatter of the peak position within the adopted tooth (over all selected teeth if
    # < 90% of the samples select it), but not less than P^2/(2 dT); enlarged where neighbouring teeth compete
    # (for P > 240 h the bootstrap scatter of the P/2 member, doubled, is used when P/2 lies in the primary range)
    half_res = P ** 2 / 24.0 / v["baseline_days"] / 2.0
    s_boot = 0.0
    for r in (1.0, 0.5):
        c0 = next((c for c in prim if abs(c["P"] - r * P) <= 1.5 * c["dP"]), None)
        if c0:
            s = c0["sigma_tooth"] if (c0["same_tooth"] >= 0.9 and c0["sigma_tooth"]) else c0["sigma_all"]
            s_boot = s / r
            break
    sigma = float(f"{max(s_boot, half_res):.2g}")
    return {"P": P, "sigma_P": sigma, "sigma_boot": s_boot, "half_resolution": half_res,
            "quality_adopted_before": ad["quality"], "quality_rule": rule_q, "quality_final": q,
            "reason": reason, "fam_u": round(fam_u, 3), "fam_w": round(fam_w, 3),
            "groups": f"{g_ok}/{len(groups)}", "loo": f"{loo_ok}/{len(loo)}", "tess": t_ok, "day_comm": dc,
            "independent_alias_test": indep}


if __name__ == "__main__":
    out = {}
    for num, ad in sorted(A.items(), key=lambda kv: int(kv[0])):
        # asteroids with a significant viewing-geometry (aspect) model: the bootstrap of the corrected data
        geo = os.path.join(ROOT, "results_geo", num, f"{num}_rebootstrap2.json")
        src = geo if os.path.exists(geo) else os.path.join(ROOT, "results", num, f"{num}_rebootstrap2.json")
        v = json.load(open(src, encoding="utf-8"))
        out[num] = grade(num, ad, v)
        out[num]["geometry_corrected"] = os.path.exists(geo)
    json.dump(out, open(os.path.join(ROOT, "quality_final.json"), "w"), indent=1)
    for num, r in out.items():
        ch = "  <-- changes" if r["quality_final"] != r["quality_adopted_before"] else ""
        ex = f"  [exception: {r['reason'][:60]}]" if r["reason"] != "rule" else ""
        print(f"{num}: P={r['P']:.4f} before {r['quality_adopted_before']} rule {r['quality_rule']} final {r['quality_final']} "
              f"famU {r['fam_u']:.2f} famW {r['fam_w']:.2f} groups {r['groups']} loo {r['loo']} tess {r['tess']} "
              f"dc {r['day_comm']} indep {r['independent_alias_test']}{ch}{ex}")
    from collections import Counter
    t34 = [r for n, r in out.items() if A[n]["table"] in (3, 4)]
    t2 = [r for n, r in out.items() if A[n]["table"] == 2]
    print("main sample:", Counter(r["quality_final"] for r in t34), " test sample:", Counter(r["quality_final"] for r in t2))
