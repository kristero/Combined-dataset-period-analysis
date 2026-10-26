"""Published spin-axis solutions (ecliptic lambda, beta) for the paper and validation asteroids, from the LCDB spin-axis
file (data/lcdb/lc_spinaxis_pub.txt, revision 2023 October). Every asteroid has a summary line (number, possibly
flagged with '*') followed by reference lines with up to four pole solutions (L1 B1 ... L4 B4) and a sidereal period.

The values are right-aligned and start up to two characters before their header labels, so the pole fields are read
as whitespace-separated tokens between the start of the L1 column (minus 4 characters) and the start of the 'Sid Per'
column (minus 3). lambda = -1 or beta = -99 means "not determined": such solutions are kept separately as
latitude-only ("lat_only") and are not used as full poles.
(Corrected 2026-10-06: the first version cut the fields at the header positions, which dropped leading digits, did
not recognise '*'-flagged asteroid lines, and treated lambda = -1 as a longitude.)
Output: data/lcdb_poles.json  {num: [{ref, poles: [[lam, bet], ...] (full poles only), lat_only: [...], sid_per}]}
"""
import json
import os
import re

ROOT = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(ROOT, "data", "lcdb", "lc_spinaxis_pub.txt")
lines = open(path, encoding="latin-1").read().splitlines()
hdr = next(l for l in lines if l.startswith("Number"))
iL1, iSid, iMod, iQ = hdr.index("L1"), hdr.index("Sid Per"), hdr.index("Mod"), hdr.index(" Q ") + 1
nums = set(json.load(open(os.path.join(ROOT, "adopted.json"), encoding="utf-8")))
nums |= {s["num"] for s in json.load(open(os.path.join(ROOT, "validation", "sample.json"), encoding="utf-8"))["sample"]}


def num_or_none(s):
    try:
        return float(s)
    except ValueError:
        return None


out, cur = {}, None
for l in lines:
    if l.startswith(("ASTEROID", "GENERATED", "Number", "---")) or not l.strip():
        continue
    if l[:7].strip().isdigit():                     # summary line: number in columns 1-7 (then "*" flag or name)
        cur = l[:7].strip() if l[:7].strip() in nums else None
        continue
    if cur is None or l[:10].strip():
        continue
    toks = [num_or_none(t) for t in l[iL1 - 4:iSid - 3].split()]
    pairs = [(toks[i], toks[i + 1]) for i in range(0, len(toks) - 1, 2)]
    full = [[a, b] for a, b in pairs if a is not None and b is not None and a != -1.0 and -90.0 <= b <= 90.0]
    lat = [b for a, b in pairs if a == -1.0 and b is not None and -90.0 <= b <= 90.0]
    sid = num_or_none(l[iSid - 3:iMod - 1].strip())
    e = {"ref": l[10:iQ].strip(), "poles": full, "lat_only": lat, "sid_per": sid}
    if (full or lat or sid) and e not in out.get(cur, []):    # the file lists every asteroid twice
        out.setdefault(cur, []).append(e)
json.dump(out, open(os.path.join(ROOT, "data", "lcdb_poles.json"), "w"), indent=1)
n_full = sum(any(e["poles"] for e in v) for v in out.values())
n_sid = sum(any(e["sid_per"] for e in v) for v in out.values())
print(f"{len(out)} asteroids with LCDB spin-axis entries; {n_full} with a full pole; {n_sid} with a sidereal period")
for n in ("933", "1405", "1963", "3706", "6361", "4563"):
    print(n, out.get(n))
