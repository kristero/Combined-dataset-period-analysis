"""
Check 3 of the revision: consistency of the periodogram period (a mean synodic period) with the sidereal period and
pole of the convex model. Over observations spread around the orbit, the mean synodic period differs from the sidereal
period by about P^2/P_orb, positive for prograde and negative for retrograde rotation, and by about zero when the spin
axis lies near the orbital plane. A ratio r = (P_mpc - P_sid)/(P^2/P_orb) far from sign(cos obliquity) points to a wrong
comb tooth of the periodogram or of the convex scan (one tooth is a fraction of P^2/P_orb, e.g. 0.76 for (1600)).

Validation: the 78 asteroids of damit_comparison.json with published sidereal periods (LCDB spin-axis table), for which
the right tooth is known. For each: r_own with our convex P_sid and the obliquity of our pole, r_pub with the published
P_sid and the obliquities of the published poles. The flag uses only our own data:
    |cos eps_own| >= COS_MIN  and  |r_own - sign(cos eps_own)| > TOL
and is compared with (a) a wrong convex period (|P_C - P_sid,pub| > dP, the separation of neighbouring minima) and
(b) a periodogram tooth inconsistent with the published solution (|r_pub - sign(cos eps_pub)| > TOL).
Paper sample: the same flag for the models of Table 5 and the three rejected models (damit_paper_table.json).
Usage: python synsid_check.py [--tol 0.3] [--cosmin 0.5]      Writes synsid_check.json
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
TOL = float(sys.argv[sys.argv.index("--tol") + 1]) if "--tol" in sys.argv else 0.3
COS_MIN = float(sys.argv[sys.argv.index("--cosmin") + 1]) if "--cosmin" in sys.argv else 0.5


def J(*p):
    return json.load(open(os.path.join(ROOT, *p), encoding="utf-8"))


ORB = {}
for f in ("vb/data/sbdb_full.json", "data/sbdb_phys_orbit.json", "data/sbdb_testswap_full.json"):
    if os.path.exists(os.path.join(ROOT, f)):
        for n, r in J(f).items():
            ORB.setdefault(n, r)


def elem(n, name):
    return float(next(e["value"] for e in ORB[n]["orbit"]["elements"] if e["name"] == name))


def cos_obliquity(n, lam, bet):
    """Cosine of the angle between the spin axis (ecliptic lambda, beta) and the orbit normal."""
    i, om = np.radians(elem(n, "i")), np.radians(elem(n, "om"))
    l, b = np.radians(lam), np.radians(bet)
    s = np.array([np.cos(b) * np.cos(l), np.cos(b) * np.sin(l), np.sin(b)])
    nrm = np.array([np.sin(i) * np.sin(om), -np.sin(i) * np.cos(om), np.cos(i)])
    return float(s @ nrm)


def scale(n, P):
    return P ** 2 / (elem(n, "per") * 24.0)


def expected(cosines):
    """sign(cos eps) when all poles give the same sign with |cos eps| >= COS_MIN, else None (axis near the plane)."""
    if not cosines or min(abs(c) for c in cosines) < COS_MIN or len({np.sign(c) for c in cosines}) > 1:
        return None
    return float(np.sign(cosines[0]))


def own_pole(asteroid):
    kind, n = asteroid.split(":")
    m = os.path.join(ROOT, "damit_runs", f"{kind}_{n}", "poles", "mirrors.json")
    if os.path.exists(m):
        w = J(m)["within"][0]
        return w["lambda"], w["beta"], w["P"]
    C = J("damit_runs", f"{kind}_{n}", "eval.json")["C"]
    return C["lambda"], C["beta"], C["P"]


S = J("damit_comparison.json")
poles = {r["asteroid"]: r for r in S["poles"]["rows"]}
val = []
for r in S["sidereal_period"]["rows"]:
    a = r["asteroid"]
    n = a.split(":")[1]
    lam, bet, P_C = own_pole(a)
    sc = scale(n, r["P_mpc"])
    c_own = cos_obliquity(n, lam, bet)
    pub = [p for p in poles.get(a, {}).get("published", [])]
    c_pub = [cos_obliquity(n, *p) for p in pub]
    e_own, e_pub = expected([c_own]), expected(c_pub)
    r_own, r_pub = (r["P_mpc"] - P_C) / sc, (r["P_mpc"] - r["sid"]) / sc
    dP = abs(r["P_C"] - r["sid"]) / r["dP_C_in_dP"] if r["dP_C_in_dP"] else None
    val.append({"asteroid": a, "P_mpc": r["P_mpc"], "P_C": P_C, "P_sid_pub": r["sid"], "P2_over_Porb": sc,
                "dP_neighbour": dP, "own_pole": [lam, bet], "cos_eps_own": c_own, "published_poles": pub,
                "cos_eps_pub": c_pub, "r_own": r_own, "r_pub": r_pub, "r_C_vs_pub": (P_C - r["sid"]) / sc,
                "expected_own": e_own, "expected_pub": e_pub,
                "flag": e_own is not None and abs(r_own - e_own) > TOL,
                "convex_wrong": abs(P_C - r["sid"]) > (dP or np.inf),
                "periodogram_tooth_off": None if e_pub is None else bool(abs(r_pub - e_pub) > TOL)})


def table(rows, key):
    t = {(f, w): 0 for f in (True, False) for w in (True, False)}
    for x in rows:
        if x[key] is not None:
            t[(x["flag"], x[key])] += 1
    return {"flag_and_wrong": t[(True, True)], "flag_not_wrong": t[(True, False)],
            "noflag_wrong": t[(False, True)], "noflag_not_wrong": t[(False, False)]}


# offset of the periodogram period from the convex-implied mean synodic period P_sid + sign(cos eps) P^2/P_orb, in units
# of the quality-rule uncertainty of the blind validation run (vb/quality_final.json), as sigma_P of the paper sample
QB = J("vb", "quality_final.json") if os.path.exists(os.path.join(ROOT, "vb", "quality_final.json")) else {}
for x in val:
    q = QB.get(x["asteroid"].split(":")[1])
    if q and x["expected_own"] is not None and abs(q["P"] - x["P_mpc"]) < 0.001 * x["P_mpc"]:
        x["sigma_P_blind"] = q["sigma_P"]
        x["offset_in_sigma_blind"] = (x["P_mpc"] - x["P_C"] - x["expected_own"] * x["P2_over_Porb"]) / q["sigma_P"]

testable = [x for x in val if x["expected_own"] is not None]
summary = {"tol": TOL, "cos_min": COS_MIN, "n": len(val), "n_testable_own": len(testable),
           "n_near_plane_own": len(val) - len(testable),
           "r_own_testable": {"median_abs_dev": float(np.median([abs(x["r_own"] - x["expected_own"]) for x in testable])),
                              "frac_within_tol": float(np.mean([abs(x["r_own"] - x["expected_own"]) <= TOL for x in testable]))},
           "vs_convex_wrong": table(val, "convex_wrong"),
           "vs_periodogram_tooth_off": table(val, "periodogram_tooth_off"),
           "n_pub_testable": sum(x["expected_pub"] is not None for x in val),
           "pub_tooth_off": [x["asteroid"] for x in val if x["periodogram_tooth_off"]],
           "convex_wrong": [x["asteroid"] for x in val if x["convex_wrong"]],
           "flagged": [x["asteroid"] for x in val if x["flag"]],
           "flagged_offset_in_sigma_blind": {x["asteroid"]: x.get("offset_in_sigma_blind") for x in val if x["flag"]}}

# paper sample: models of Table 5 and the rejected models
T = J("damit_paper_table.json")
paper = []
for m in T["models"] + T.get("rejected", []):
    n = m["num"]
    P = m["P_adopted"]
    sc = scale(n, P)
    lam, bet = m["poles"][0]
    c_own = cos_obliquity(n, lam, bet)
    e_own = expected([c_own])
    pub = m.get("published_poles") or []
    c_pub = [cos_obliquity(n, *p) for p in pub]
    r_own = (P - m["P_sid"]) / sc
    paper.append({"num": n, "rejected": "reason" in m, "quality": m["quality"], "P": P, "P_sid": m["P_sid"],
                  "offset_in_sigma": None if e_own is None else (r_own - e_own) * sc / m["sigma_adopted"],
                  "P2_over_Porb": sc, "pole": [lam, bet], "cos_eps_own": c_own,
                  "obliquity_deg": float(np.degrees(np.arccos(np.clip(c_own, -1, 1)))), "r_own": r_own,
                  "expected_own": e_own, "flag": e_own is not None and abs(r_own - e_own) > TOL,
                  "published_poles": pub, "cos_eps_pub": c_pub, "expected_pub": expected(c_pub)})
summary["paper_flagged"] = [x["num"] for x in paper if x["flag"]]
summary["paper_near_plane"] = [x["num"] for x in paper if x["expected_own"] is None]

json.dump({"summary": summary, "validation": val, "paper": paper},
          open(os.path.join(ROOT, "synsid_check.json"), "w"), indent=1, default=float)
print(json.dumps(summary, indent=1))
print("\nvalidation rows (flag, convex wrong or periodogram tooth off):")
for x in val:
    if x["flag"] or x["convex_wrong"] or x["periodogram_tooth_off"]:
        print(f"  {x['asteroid']:>12} r_own={x['r_own']:+.2f} (exp {x['expected_own']}, cos {x['cos_eps_own']:+.2f}) "
              f"r_pub={x['r_pub']:+.2f} (exp {x['expected_pub']}) r_C-pub={x['r_C_vs_pub']:+.2f} flag={x['flag']} "
              f"convex_wrong={x['convex_wrong']} tooth_off={x['periodogram_tooth_off']}")
print("\npaper models:")
for x in paper:
    print(f"  ({x['num']}) {'REJ ' if x['rejected'] else ''}r_own={x['r_own']:+.2f} obliquity={x['obliquity_deg']:.0f} "
          f"exp={x['expected_own']} flag={x['flag']} pub_exp={x['expected_pub']}")
