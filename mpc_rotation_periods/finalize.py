"""
Merge the review / verification verdicts (review_results.json, saved from the alias-review workflow)
with the algorithmic proposals into adopted.json, the single source of the periods used in the
tables and figures. Manual overrides (overrides.json, with a reason for each) are applied last.
"""
from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
PUB = json.load(open(os.path.join(ROOT, "published.json"), encoding="utf-8"))
PROP = json.load(open(os.path.join(ROOT, "proposals.json"), encoding="utf-8"))
REV = json.load(open(os.path.join(ROOT, "review_results.json"), encoding="utf-8"))
OVR_PATH = os.path.join(ROOT, "overrides.json")
OVR = json.load(open(OVR_PATH, encoding="utf-8")) if os.path.exists(OVR_PATH) else {}

adopted = {}
for item in REV:
    if not item or not item.get("review"):
        continue
    ver = {v["asteroid"]: v for v in (item.get("verify") or {}).get("verdicts", [])}
    for r in item["review"]["results"]:
        n = str(r["asteroid"])
        e = {"table": PUB[n]["table"], "P": r["adopted_P"], "sigma_P": r["sigma_P"], "quality": r["quality"],
             "alternatives": [a["P"] for a in r["alternatives"]],
             "alternative_notes": {str(a["P"]): a["note"] for a in r["alternatives"]},
             "review_rationale": r["rationale"], "paper_note": r["paper_note"], "concerns": r["concerns"],
             "agrees_with_algorithm": r["agrees_with_algorithm"], "nterms": 2, "source": "review"}
        if int(n) in ver:
            v = ver[int(n)]
            e.update({"P": v["final_P"], "sigma_P": v["final_sigma_P"], "quality": v["final_quality"],
                      "alternatives": v["final_alternatives"], "verify_verdict": v["verdict"],
                      "verify_reason": v["reason"], "source": "verify"})
        p = PROP.get(n, {})
        e["syn_sid_scale"] = e["P"] ** 2 / (p.get("orbital_period_days", float("nan")) * 24.0)
        e["baseline_days"] = p.get("baseline_days")
        adopted[n] = e
for n, o in OVR.items():
    adopted[n].update(o)
    adopted[n]["source"] = "override"
# refined position of the adopted comb tooth in the final data (refine_periods.py); applied only to the tooth that
# the review and the overrides chose
PR_PATH = os.path.join(ROOT, "period_refine.json")
if os.path.exists(PR_PATH):
    for n, r in json.load(open(PR_PATH, encoding="utf-8")).items():
        if abs(r["P_review"] - adopted[n]["P"]) <= 1e-9 * max(1.0, r["P_review"]):
            adopted[n]["P_review"] = adopted[n]["P"]
            adopted[n]["P"] = r["P"]
# quality grades: the uniform rule of quality_rule.py (evaluated on the corrected bootstrap of rebootstrap2.py)
# plus the documented exceptions listed there; quality_rule.py only reads the adopted periods
QF_PATH = os.path.join(ROOT, "quality_final.json")
if os.path.exists(QF_PATH):
    for n, q in json.load(open(QF_PATH, encoding="utf-8")).items():
        adopted[n].update({"quality": q["quality_final"], "quality_rule": q["quality_rule"],
                           "quality_reason": q["reason"], "sigma_P": q["sigma_P"]})

missing = [k for k in PUB if not k.startswith("_") and k not in adopted]
if missing:
    raise SystemExit(f"missing verdicts for {missing}")
json.dump(dict(sorted(adopted.items(), key=lambda kv: int(kv[0]))),
          open(os.path.join(ROOT, "adopted.json"), "w", encoding="utf-8"), indent=1)
for n, e in sorted(adopted.items(), key=lambda kv: int(kv[0])):
    pc = PUB[n].get("P_c")
    print(f"{n}: P={e['P']:.5f} +/- {e['sigma_P']:.2g} Q={e['quality']} alt={e['alternatives']} "
          f"[{e['source']}{'/' + e.get('verify_verdict', '') if 'verify_verdict' in e else ''}] (P_c={pc})")
