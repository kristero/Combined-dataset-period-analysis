"""Count how the expert review changed the algorithmic proposals (proposals -> adopted.json).

By default the proposals the review actually saw are used: proposals_legacy.json, made by adjudicate_pre.py from
the first rebootstrap.py version (overlapping +/-20 resolution-element search windows); these are the numbers
quoted in the paper. With --proposals FILE another set can be compared, e.g. proposals.json regenerated from the
corrected rebootstrap.py (output review_changes_<FILE stem>.json).
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
prop_file = "proposals_legacy.json"
if "--proposals" in sys.argv:
    prop_file = sys.argv[sys.argv.index("--proposals") + 1]
if prop_file == "proposals_legacy.json" and not os.path.exists(os.path.join(ROOT, prop_file)):
    raise SystemExit("proposals_legacy.json (the proposals the review saw; quoted in the paper) is required; "
                     "run archive_legacy_bootstrap.py before re-running rebootstrap.py/adjudicate_pre.py, "
                     "or pass --proposals FILE to compare another set (writes review_changes_<stem>.json)")
PROP = json.load(open(os.path.join(ROOT, prop_file), encoding="utf-8"))
A = json.load(open(os.path.join(ROOT, "adopted.json"), encoding="utf-8"))
out_file = "review_changes.json" if "--proposals" not in sys.argv \
    else f"review_changes_{os.path.splitext(os.path.basename(prop_file))[0]}.json"


def relation(Pa, Pb, base):
    if abs(Pa - Pb) <= 1.5 * Pb ** 2 / 24.0 / base:
        return "same"
    for r, name in ((2.0, "harmonic 2P"), (0.5, "harmonic P/2")):
        if abs(Pa - r * Pb) <= 0.01 * Pa:
            return name
    if abs(Pa - Pb) <= 0.01 * Pa:
        return "comb tooth"
    return "other peak (daily alias or unrelated)"


out = {}
for n, ad in A.items():
    p = PROP[n]
    rel = relation(ad["P"], p["P"], p["baseline_days"])
    out[n] = {"P_auto": p["P"], "P_adopted": ad["P"], "relation": rel, "Q_auto": p["quality_auto"],
              "Q_final": ad["quality"], "table": ad["table"]}
print(f"proposals: {prop_file}")
changed_P = {n: r for n, r in out.items() if r["relation"] != "same"}
print(f"period changed for {len(changed_P)} of {len(out)}:")
for n, r in changed_P.items():
    print(f"  {n} (table {r['table']}): {r['P_auto']:.5f} -> {r['P_adopted']:.5f}  [{r['relation']}]  Q {r['Q_auto']}->{r['Q_final']}")
same = {n: r for n, r in out.items() if r["relation"] == "same"}
qch = {n: r for n, r in same.items() if r["Q_auto"] != r["Q_final"]}
print(f"same period, quality changed for {len(qch)}:", {n: f"{r['Q_auto']}->{r['Q_final']}" for n, r in qch.items()})
json.dump(out, open(os.path.join(ROOT, out_file), "w"), indent=1)
