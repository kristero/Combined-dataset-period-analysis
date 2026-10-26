"""Review C03: amplitude statistics of Section 3.4 recomputed without the six asteroids that have no adopted period.
Writes review_amp_resolved.json"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(ROOT, "amp_eval_results.json"), encoding="utf-8"))
AD = json.load(open(os.path.join(ROOT, "adopted.json"), encoding="utf-8"))
UNDET = sorted(n for n, a in AD.items() if a.get("status") == "undetermined" or a.get("quality") in (None, "-", ""))


def stats(keys):
    comb = np.array([R[k]["COMB_F4"] for k in keys])
    app = np.array([R[k]["APP_TPL_wmean"] for k in keys])
    ok = np.isfinite(comb) & np.isfinite(app) & (app > 0)
    q = comb[ok] / app[ok]
    return dict(n=int(ok.sum()), comb_over_app_median=float(np.median(q)),
                app_over_comb_median=float(np.median(1 / q)), app_over_comb_16_84=[float(x) for x in np.percentile(1 / q, [16, 84])])


val = [k for k in R if k.startswith("val:")]
paper = [k for k in R if k.startswith("paper:")]
resolved = [k for k in paper if k.split(":")[1] not in UNDET]
out = dict(undetermined=UNDET, all_177=stats(val + paper), resolved_171=stats(val + resolved),
           paper_57=stats(paper), paper_resolved_51=stats(resolved),
           n_apparitions_resolved=[int(np.min([R[k]["n_app"] for k in resolved])), int(np.median([R[k]["n_app"] for k in resolved])),
                                   int(np.max([R[k]["n_app"] for k in resolved]))])
json.dump(out, open(os.path.join(ROOT, "review_amp_resolved.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
