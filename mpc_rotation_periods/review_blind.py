"""
Review A06: the 13 test asteroids rerun without any candidate that was seeded from a published value.
Candidates of mpc_redo.py whose seed equals a value of published.json (P_lit or the earlier manuscript period P_c, or
twice or half of these) are removed from the candidate list of rebootstrap.py, unless an unseeded candidate lies in
the same comb tooth. The adopted period and its alternatives (adopted.json) are not added as seeds either.
rebootstrap2.py is then rerun on the same data (corrected data for the asteroids with an aspect model), and the
quality rule is applied to the adopted period. Run on the LU HPC (hpc/review_blind.sbatch).
Writes review_blind/<N>/<N>_rebootstrap2.json and review_blind.json
"""
import json
import multiprocessing as mp
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
BLIND = os.path.join(ROOT, "review_blind")
AD = json.load(open(os.path.join(ROOT, "adopted.json"), encoding="utf-8"))
PUB = json.load(open(os.path.join(ROOT, "published.json"), encoding="utf-8"))
TEST = sorted((n for n, a in AD.items() if a["table"] == 2), key=int)


def seeded(num):
    """refined periods of the candidates of mpc_redo.py that were seeded from published values, and of the others"""
    r = json.load(open(os.path.join(ROOT, "results", num, f"{num}_result.json"), encoding="utf-8"))
    vals = [PUB[num][k] * f for k in ("P_lit", "P_c") if k in PUB[num] for f in (1.0, 2.0, 0.5)]
    lit, own = [], []
    for c in r["candidates"]:
        (lit if any(abs(c["P_seed"] - v) <= 1e-6 * v for v in vals) else own).append((c["P"], c["dP_resolution"]))
    return lit, own


def prepare(num):
    src = os.path.join(ROOT, "results_geo", num)
    if not os.path.exists(os.path.join(src, f"{num}_arrays.npz")):
        src = os.path.join(ROOT, "results", num)
    os.makedirs(os.path.join(BLIND, num), exist_ok=True)
    shutil.copy(os.path.join(src, f"{num}_arrays.npz"), os.path.join(BLIND, num))
    rb = json.load(open(os.path.join(src, f"{num}_rebootstrap.json"), encoding="utf-8"))
    lit, own = seeded(num)
    near = lambda P, lst: any(abs(P - q) <= 1.5 * max(d, q ** 2 / 24.0 / rb["baseline_days"]) for q, d in lst)
    removed = []
    for key in ("primary", "extended"):
        keep = []
        for c in rb[key]:
            if near(c["P"], lit) and not near(c["P"], own):
                removed.append(c["P"])
            else:
                keep.append(c)
        rb[key] = keep
    json.dump(rb, open(os.path.join(BLIND, num, f"{num}_rebootstrap.json"), "w"), indent=1)
    first = (rb["primary"] + rb["extended"])[0]["P"]
    return num, {"table": 2, "P": first, "quality": AD[num]["quality"], "alternatives": []}, removed


def boot(num):
    import rebootstrap2 as r2
    r2.ROOT, r2.RES = BLIND, BLIND
    return r2.run(int(num))


def main():
    os.makedirs(BLIND, exist_ok=True)
    blind_ad, removed = {}, {}
    for n in TEST:
        _, ad, rm = prepare(n)
        blind_ad[n], removed[n] = ad, rm
    json.dump(blind_ad, open(os.path.join(BLIND, "adopted.json"), "w"), indent=1)
    with mp.get_context("spawn").Pool(len(TEST)) as pool:
        for msg in pool.imap_unordered(boot, TEST):
            print(msg, flush=True)
    import quality_rule as QR
    QF = json.load(open(os.path.join(ROOT, "quality_final.json"), encoding="utf-8"))
    out = {}
    for n in TEST:
        v = json.load(open(os.path.join(BLIND, n, f"{n}_rebootstrap2.json"), encoding="utf-8"))
        P = AD[n]["P"]
        present = any(abs(c["P"] - P) <= 1.5 * c["dP"] for c in v["primary"] + v["extended"])
        best = max(v["primary"], key=lambda c: c["famwin_u"])
        g = QR.grade(n, AD[n], v)
        out[n] = dict(removed_candidates=removed[n], adopted_in_candidates=present,
                      best_family_P=best["P"], best_family_win_u=best["famwin_u"],
                      best_family_is_adopted=QR.in_family(best["P"], P),
                      fam_u=g["fam_u"], fam_w=g["fam_w"], groups=g["groups"], loo=g["loo"], tess=g["tess"],
                      rule_blind=g["quality_rule"], rule_final=QF[n]["quality_rule"], final=QF[n]["quality_final"],
                      fam_u_final=QF[n]["fam_u"], fam_w_final=QF[n]["fam_w"])
        print(n, out[n], flush=True)
    json.dump(out, open(os.path.join(ROOT, "review_blind.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
