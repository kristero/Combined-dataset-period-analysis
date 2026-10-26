"""
Literature comparison of the six test asteroids added in the revision, made after their periods were final (blind
review: review_results.json batch 'testswap'; geometry, refinement and convex inversion: hpc/testswap_*.out).
P_lit: LCDB period as given in the JPL SBDB (data/sbdb_testswap_full.json, rot_per; the pipeline files had it removed).
Sidereal periods and poles of published convex models: LCDB spin-axis table (data/lcdb/lc_spinaxis_pub.txt).
Expected difference between a mean synodic period measured near opposition and the sidereal period: about P^2/P_orb,
positive for prograde and negative for retrograde rotation.
Writes testswap_litcompare.json
"""
import json
import os
import re

ROOT = os.path.dirname(os.path.abspath(__file__))
NEW6 = ["249", "486", "870", "1600", "1700", "7750"]
SB = json.load(open(os.path.join(ROOT, "data", "sbdb_testswap_full.json"), encoding="utf-8"))
AD = json.load(open(os.path.join(ROOT, "adopted.json"), encoding="utf-8"))
QF = json.load(open(os.path.join(ROOT, "quality_final.json"), encoding="utf-8"))


def spin_axis(num):
    """Reference lines below the asteroid's header in the LCDB spin-axis table: (reference, poles, sidereal P)."""
    L = open(os.path.join(ROOT, "data", "lcdb", "lc_spinaxis_pub.txt"), encoding="latin-1").read().split("\n")
    head = next(x for x in L if x.startswith("Number") and "Sid Per" in x)
    c_l1, c_sid, c_mod = head.index("L1") - 4, head.index("Sid Per") - 3, head.index("Mod") - 1   # fixed columns
    out = []
    for i, ln in enumerate(L):
        if re.match(rf"^\s*{num}\s", ln):
            for x in L[i + 1:]:
                if not x.strip() or re.match(r"^\s*\d+\s", x):
                    break
                v = [float(s) for s in x[c_l1:c_sid].split()]
                sid = x[c_sid:c_mod].strip()
                out.append({"ref": x[10:41].strip(), "poles": [v[k:k + 2] for k in range(0, len(v) - 1, 2)],
                            "P_sid": float(sid) if sid else None})
            break
    return out


def elem(r, name):
    return float(next(e["value"] for e in r["orbit"]["elements"] if e["name"] == name))


SBALL = json.load(open(os.path.join(ROOT, "data", "sbdb_phys_orbit.json"), encoding="utf-8"))
PUB = json.load(open(os.path.join(ROOT, "published.json"), encoding="utf-8"))
TEST = sorted((n for n, a in AD.items() if a.get("table") == 2), key=int)


def own_sidereal(n):
    """Sidereal period of our convex model (best refined solution), if the run exists."""
    p = os.path.join(ROOT, "damit_runs", f"paper_{n}", "eval.json")
    if not os.path.exists(p):
        return None
    m = os.path.join(ROOT, "damit_runs", f"paper_{n}", "poles", "mirrors.json")
    if os.path.exists(m):
        return json.load(open(m))["within"][0]["P"]
    return json.load(open(p))["C"]["P"]


res = {}
for n in TEST:
    r = SB.get(n) or SBALL[n]
    phys = {p["name"]: p["value"] for p in r.get("phys_par", [])}
    P_lit = float(phys["rot_per"]) if "rot_per" in phys else PUB.get(n, {}).get("P_lit")
    P_orb_h = elem(r, "per") * 24.0
    P = AD[n]["P"]
    d_exp = P ** 2 / P_orb_h
    sa = spin_axis(n)
    psid = [s["P_sid"] for s in sa if s["P_sid"]]
    out = {"P": P, "sigma_P": AD[n]["sigma_P"], "quality": AD[n]["quality"], "P_lit": P_lit, "P_orb_h": P_orb_h,
           "P2_over_Porb": d_exp, "spin_axis": sa}
    if P_lit:
        out["dP_lit"] = P - P_lit
        out["dP_lit_pct"] = 100 * (P - P_lit) / P_lit
        out["dP_lit_in_P2Porb"] = (P - P_lit) / d_exp
    if psid:
        ps = sum(psid) / len(psid)
        out["P_sid_lit"] = ps
        out["dP_sid"] = P - ps
        out["dP_sid_in_P2Porb"] = (P - ps) / d_exp
    own = own_sidereal(n)
    if own:
        out["P_sid_own"] = own
        out["dP_sid_own_in_P2Porb"] = (P - own) / d_exp
    res[n] = out
    print(f"({n}) P={P:.6f}±{AD[n]['sigma_P']} {AD[n]['quality']}  P_lit={P_lit}  dP={out.get('dP_lit', float('nan')):+.5f} h "
          f"({out.get('dP_lit_pct', float('nan')):+.3f}%, {out.get('dP_lit_in_P2Porb', float('nan')):+.2f} P2/Porb)  "
          f"P2/Porb={d_exp:.5f} h  P_sid={out.get('P_sid_lit')}  P-P_sid={out.get('dP_sid', float('nan')):+.5f} h "
          f"({out.get('dP_sid_in_P2Porb', float('nan')):+.2f} P2/Porb)  own P_sid={own} "
          f"({out.get('dP_sid_own_in_P2Porb', float('nan')):+.2f} P2/Porb)  poles {[s['poles'] for s in sa]}")
json.dump(res, open(os.path.join(ROOT, "testswap_litcompare.json"), "w", encoding="utf-8"), indent=1)
