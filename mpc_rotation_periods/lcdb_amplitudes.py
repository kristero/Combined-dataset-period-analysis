"""Literature light-curve amplitudes of the 57 asteroids from the Asteroid Lightcurve Database (LCDB, Warner et al.
2009; revision 2023 October, data/lcdb/lc_summary_pub.txt and lc_details_pub.txt). Fixed-width columns are located
from the header line. Output: data/lcdb_amplitudes.json with the summary entry (period, Amin, Amax, U) and every
publication entry (reference, period, Amin, Amax, U)."""
import json
import os
import re

ROOT = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(ROOT, "data", "lcdb")
A = json.load(open(os.path.join(ROOT, "adopted.json"), encoding="utf-8"))
NUMS = set(int(n) for n in A)


def columns(header):
    names = ["NUMBER", "NAME", "PERIOD", "P DESC", "AMIN", "AMAX", " U ", "NOTES"]
    pos = {n.strip(): header.index(n) for n in names}
    return pos


def fnum(s):
    s = s.strip().lstrip("<>")
    try:
        return float(s)
    except ValueError:
        return None


def parse(path):
    lines = open(path, encoding="latin-1").read().splitlines()
    hdr = next(l for l in lines if l.startswith("NUMBER"))
    pos = columns(hdr)
    # fields: PERIOD starts at pos["PERIOD"] up to "P DESC"; AMIN at pos["AMIN"] (4-5 chars), AMAX, U
    out, cur = {}, None
    for l in lines[lines.index(hdr) + 2:]:
        if not l.strip():
            continue
        numtxt = l[:pos["NAME"]].strip()
        rec = {"period": fnum(l[pos["PERIOD"]:pos["P DESC"]]),
               "amin": fnum(l[pos["AMIN"]:pos["AMIN"] + 5]), "amax": fnum(l[pos["AMAX"]:pos["AMAX"] + 5]),
               "U": l[pos["U"] + 1:pos["U"] + 4].strip()}
        if numtxt.isdigit():
            n = int(numtxt)
            cur = n if n in NUMS else None
            if cur is not None:
                out[cur] = {"summary": rec, "name": l[pos["NAME"]:pos["NAME"] + 30].strip(), "details": []}
        elif cur is not None and l[:pos["NAME"]].strip() == "":
            rec["ref"] = l[pos["NAME"]:pos["NAME"] + 40].strip()
            out[cur]["details"].append(rec)
    return out


summ = parse(os.path.join(D, "lc_summary_pub.txt"))
det = parse(os.path.join(D, "lc_details_pub.txt"))
res = {}
for n in sorted(NUMS):
    if n in det:
        res[str(n)] = {"name": det[n]["name"], "summary": det[n]["summary"], "details": det[n]["details"]}
    elif n in summ:
        res[str(n)] = {"name": summ[n]["name"], "summary": summ[n]["summary"], "details": []}
json.dump(res, open(os.path.join(ROOT, "data", "lcdb_amplitudes.json"), "w"), indent=1)
print(f"{len(res)} of {len(NUMS)} asteroids have an LCDB entry")
for n, r in res.items():
    s = r["summary"]
    amps = [(d["ref"][:22], d["period"], d["amin"], d["amax"], d["U"]) for d in r["details"] if d["amin"] or d["amax"]]
    print(f"{n} ({A[n]['quality']}, table {A[n]['table']}) P_lcdb={s['period']} Amin={s['amin']} Amax={s['amax']} U={s['U']} "
          f"| ours P={A[n]['P']:.4f} | refs with amplitude: {amps}")
