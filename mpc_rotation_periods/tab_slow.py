"""
Rows of Table tab:slow (periods of the slow rotators from the different methods) from adopted.json, the separate-
dataset periods (published.json, with the correction of make_tables.PAV_FIX), the DAMIT window scans
(longperiod_damit.json) and the 120-3000 h scans (longperiod_longrange.json). Writes the rows into the manuscript
between the \\startdata and \\enddata of tab:slow (or only prints them with --dry).
Usage: python tab_slow.py [--dry] [--snapshot DIR]   (DIR: read the JSON files from DIR, e.g. unsplit_snapshot)
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, sys.argv[sys.argv.index("--snapshot") + 1]) if "--snapshot" in sys.argv else ROOT
TEX = os.path.join(ROOT, "manuscript_revised", "mpc_rotation_periods.tex")
A = json.load(open(os.path.join(SRC, "adopted.json"), encoding="utf-8"))
PUB = json.load(open(os.path.join(ROOT, "published.json"), encoding="utf-8"))
LD = json.load(open(os.path.join(SRC, "longperiod_damit.json"), encoding="utf-8"))
LR = json.load(open(os.path.join(SRC, "longperiod_longrange.json"), encoding="utf-8"))
sys.path.insert(0, ROOT)
from make_tables import PAV_FIX, fmt_p, tex_name  # noqa: E402

QORD = {"A": 0, "B": 1, "C": 2}


def rel(a, b):
    return abs(a - b) / b


def fmt_h(P, nd):
    s = f"{P + 1e-9:.{nd}f}"
    return s[:-2] if s.endswith(".0") else s


rows = []
nums = [n for n, a in A.items() if a["P"] > 70 and a.get("table") in (3, 4)]
nums.sort(key=lambda n: (A[n].get("status") == "undetermined", QORD.get(A[n]["quality"], 3), int(n)))
for n in nums:
    ad = A[n]
    extra_P = None
    und = ad.get("status") == "undetermined"
    P = ad["P"]
    pav = PAV_FIX.get(n, PUB[n].get("P_av"))
    Ps = f"({fmt_h(P, 1)})" if und else fmt_p(P, ad["sigma_P"])[0]
    if n in LR:
        sub = LR[n]["subsets"]
        Psid = sub["all"]["P_best"]
        minima = sub["all"]["minima_within_threshold"]
        others = [m for m in minima if rel(m, Psid) > 0.01]
    else:
        win = [w for w in LD[n]["windows"] if w["within_threshold"]]
        main = min(win, key=lambda w: min(rel(w["P_best"], P), rel(w["P_best"], P / 2), rel(w["P_best"], 2 * P)))
        # prefer the solution in the family member of P itself when it is below the threshold
        fam_P = [w for w in win if rel(w["P_best"], P) < 0.05]
        main = fam_P[0] if fam_P else main
        Psid = main["P_best"]
        others = [w["P_best"] for w in win if w is not main]
        # the family member at P itself is above the threshold: give its chi2 excess
        if not fam_P and not und:
            wP = [w for w in LD[n]["windows"] if rel(w["P_best"], P) < 0.05]
            if wP:
                extra_P = (wP[0]["P_best"], wP[0]["chi2_ratio_to_best"])
    ref, mark = P, ""
    if rel(Psid, P / 2) < rel(Psid, P):
        ref, mark = P / 2, "^{a}"
    d = (Psid - ref) / ref * 100
    remarks = []
    if n not in LR and extra_P:
        remarks.append(f"{fmt_h(extra_P[0], 2)}\\,h: $\\chi^2$ higher by {100 * (extra_P[1] - 1):.1f}\\%")
    if n in LR and not und:
        halves = {h: sub[h] for h in ("odd", "even")}
        if len(minima) == 1 and all(rel(s["P_best"], Psid) < 0.01 and s["n_minima_within_threshold"] == 1
                                    for s in halves.values()):
            remarks.append("unique in 120--3000\\,h, also in both halves")
    elif n in LR:
        halves = {h: sub[h] for h in ("odd", "even")}
        agree = [h for h, s in halves.items() if rel(s["P_best"], Psid) < 0.01 and s["n_minima_within_threshold"] == 1]
        if len(agree) == 2:
            remarks.append("same in both halves")
        else:
            for h, s in halves.items():
                if h in agree:
                    continue
                ms = sorted(s["minima_within_threshold"], key=lambda m: rel(m, s["P_best"]))
                txt = " and ".join(fmt_h(m, 1) for m in sorted(ms))
                remarks.append(f"{txt}\\,h in the {h} apparitions")
    for o in others:
        remarks.append(f"{fmt_h(o, 1 if o >= 100 else 2)}\\,h fits equally well")
    alts = [x for x in ad.get("alternatives", []) if rel(x, ref) < 0.03 and rel(x, ref) > 1e-4]
    # neighbouring comb tooth closer to the sidereal period: only for the asteroids without a period (for the others
    # the synodic-sidereal difference of up to 0.3 deg/day x P is comparable to the tooth spacing; see the text)
    closer = [x for x in alts if abs(Psid - x) < abs(Psid - ref)]
    if closer and und:
        x = min(closer, key=lambda x: abs(Psid - x))
        remarks.append(f"comb tooth {fmt_h(x, 1)}\\,h")
    if not remarks:
        remarks.append("unique")

    nm, _ = tex_name(int(n))
    rows.append(f"{nm} & {'\\nodata' if und else ad['quality']} & {pav:.3f} & {Ps} & {fmt_h(Psid, 1 if und else 2)} & "
                f"${d:+.2f}{mark}$ & {'; '.join(remarks)} \\\\")
body = "\n".join(rows)
print(body)
if "--dry" not in sys.argv:
    s = open(TEX, encoding="utf-8").read()
    m = re.search(r"(\\label\{tab:slow\}.*?\\startdata\n)(.*?)(\n\\enddata)", s, re.S)
    assert m, "tab:slow not found"
    s = s[:m.start(2)] + body + s[m.end(2):]
    open(TEX, "w", encoding="utf-8", newline="\n").write(s)
    print("tab:slow rows written")
