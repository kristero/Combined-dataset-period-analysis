"""
Register the six test asteroids added in the revision and retire the five they replace:
  * the blind review verdicts (workflow output, see testswap notes) are appended to review_results.json as batch
    'testswap' in the format finalize.py reads;
  * the retired asteroids (retired.json) move from published.json to published_retired.json;
  * finalize.py rebuilds adopted.json.
Usage: python testswap_register.py <workflow output json>
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
wf = json.load(open(sys.argv[1], encoding="utf-8"))["result"]["results"]
REV_PATH = os.path.join(ROOT, "review_results.json")
rev = json.load(open(REV_PATH, encoding="utf-8"))
rev = [it for it in rev if (it or {}).get("batch") != "testswap"]
results, verdicts = [], []
for r in wf:
    a, v = r["review"], r["verify"]
    results.append({"asteroid": a["asteroid"], "adopted_P": a["adopted_P"], "sigma_P": a["sigma_P"], "quality": a["quality"],
                    "alternatives": a["alternatives"], "agrees_with_algorithm": a["agrees_with_algorithm"],
                    "differs_from_manuscript": False, "rationale": a["rationale"], "paper_note": a["paper_note"],
                    "concerns": a["concerns"], "no_period": a["no_period"]})
    verdicts.append({"asteroid": v["asteroid"], "verdict": v["verdict"], "final_P": v["final_P"],
                     "final_sigma_P": v["final_sigma_P"], "final_quality": v["final_quality"],
                     "final_alternatives": v["final_alternatives"], "reason": v["reason"]})
rev.append({"batch": "testswap", "review": {"results": results}, "verify": {"verdicts": verdicts}})
json.dump(rev, open(REV_PATH, "w", encoding="utf-8"), indent=1)
print("review_results.json: batch 'testswap' with", len(results), "asteroids")

ret = {k for k in json.load(open(os.path.join(ROOT, "retired.json"), encoding="utf-8")) if not k.startswith("_")}
pub = json.load(open(os.path.join(ROOT, "published.json"), encoding="utf-8"))
rp = os.path.join(ROOT, "published_retired.json")
old = json.load(open(rp, encoding="utf-8")) if os.path.exists(rp) else {}
old.update({k: pub[k] for k in ret if k in pub})
json.dump(old, open(rp, "w", encoding="utf-8"), indent=1)
pub = {k: v for k, v in pub.items() if k not in ret}
json.dump(pub, open(os.path.join(ROOT, "published.json"), "w", encoding="utf-8"), indent=1)
print("published.json:", len([k for k in pub if not k.startswith("_")]), "asteroids; retired:", sorted(old, key=int))
p = subprocess.run([sys.executable, os.path.join(ROOT, "finalize.py")], capture_output=True, text=True)
print(p.stdout[-1500:], p.stderr[-1500:])
