"""Numbers and table of the DAMIT convex models for the paper.
 - period accuracy on the validation sample: |P_DAMIT - P_published| in units of dP = 0.5 P^2 / T (74 asteroids)
 - reliable models of our sample (unique period in the scan window, at most the mirror pole ambiguity, dark facet
   below 1 %), without the long-period asteroids under re-analysis ((3173), (4371))
Output: damit_paper_table.json and manuscript_revised/table_damit.tex (deluxetable body)"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from make_tables import tex_name  # noqa: E402
R = os.path.join(ROOT, "damit_runs")
S = json.load(open(os.path.join(ROOT, "damit_comparison.json")))
A = json.load(open(os.path.join(ROOT, "adopted.json")))
poles = json.load(open(os.path.join(ROOT, "data", "lcdb_poles.json")))
EXCLUDE = {n for n, a in A.items() if a.get("status") == "undetermined"}   # no period determined

rows = S["sidereal_period"]["rows"]
x = np.array([r["dP_C_in_dP"] for r in rows])
acc = {"n": len(x), "median_dP": float(np.median(x)), "p68_dP": float(np.percentile(x, 68)),
       "p68_dP_within1": float(np.percentile(x[x <= 1], 68)), "n_within1": int(np.sum(x <= 1))}
print("period accuracy (units of dP):", acc)
k_unc = round(max(acc["p68_dP_within1"], 0.1), 2)

tab = []
for d in sorted(os.listdir(R), key=lambda s: (s.split("_")[0], int(s.split("_")[1]) if s.split("_")[1].isdigit() else 0)):
    if not d.startswith("paper_"):
        continue
    n = d.split("_")[1]
    e = json.load(open(os.path.join(R, d, "eval.json")))
    C = e["C"]
    # pole solutions: the refined (polished) best solution and the refined competing solutions within the threshold
    # (damit_mirrors.py); without mirrors.json the unrefined start solutions of damit_eval.pole_clusters
    mpath = os.path.join(R, d, "poles", "mirrors.json")
    M = json.load(open(mpath)) if os.path.exists(mpath) else None
    only_mirror = M["only_mirror_ambiguity"] if M else C["poles"]["only_mirror_ambiguity"]
    # the model of the table and the figure: the best refined solution (it may be a refined mirror solution)
    W0 = M["within"][0] if M else {"P": C["P"], "dark_pct": C["dark_pct"], "dev": C["dev"], "areas": "areas_best"}
    dark = W0["dark_pct"] if W0.get("dark_pct") is not None else C["dark_pct"]
    ok = C["scan"]["unique"] and only_mirror and dark < 1.0
    if not ok or n in EXCLUDE:
        continue
    info = json.load(open(os.path.join(R, d, "prep.json")))
    jd = np.load(os.path.join(R, d, "prep.npz"))["jd_lt"]
    T = (jd.max() - jd.min()) * 24.0
    dP = 0.5 * C["P"] ** 2 / T
    sols = [(w["lambda"], w["beta"]) for w in (M["within"] if M else C["poles"]["within"])]
    pub = [p for x_ in poles.get(n, []) for p in x_["poles"]]
    tab.append({"num": n, "table": A[n]["table"], "quality": A[n]["quality"], "P_adopted": A[n]["P"],
                "sigma_adopted": A[n].get("sigma_P"), "P_sid": W0["P"], "sigma_sid": k_unc * dP, "dP": dP,
                "poles": sols, "dark_pct": dark, "rms": W0["dev"], "published_poles": pub, "areas": W0["areas"],
                "rel_diff": (W0["P"] - A[n]["P"]) / A[n]["P"], "poles_source": "polished" if M else "start solutions"})
json.dump({"accuracy": acc, "k_unc": k_unc, "models": tab}, open(os.path.join(ROOT, "damit_paper_table.json"), "w"), indent=1)


def fmt_P(P, s):
    nd = max(0, int(np.ceil(-np.log10(s))) + 1) if s > 0 else 6
    return f"{P:.{nd}f}", f"{s:.{nd}f}"


def name(n):
    return {"1798": "Watts", "1820": "Lohmann", "1942": "Jablunka", "1963": "Bezovec", "2134": "Dennispalm",
            "2150": "Nyctimene", "2539": "Ningxia", "2817": "Perec", "3716": "Petzval", "4048": "Samwestfall",
            "4239": "Goodman", "4270": "Juanvictoria", "4618": "Shakhovskoj", "4761": "Urrutia"}.get(n, "")


lines = []
for t in tab:
    p, s = fmt_P(t["P_sid"], t["sigma_sid"])
    pl = "; ".join(f"$({l:.0f}, {b:+.0f})$" for l, b in t["poles"])
    pa = fmt_P(t["P_adopted"], t["sigma_adopted"])[0] if t["sigma_adopted"] else f"{t['P_adopted']}"
    lines.append(f"{tex_name(int(t['num']))[0]} & {t['quality']} & {pa} & ${p}\\pm{s}$ & {pl} & "
                 f"{t['dark_pct']:.1f} \\\\")
open(os.path.join(ROOT, "manuscript_revised", "table_damit_rows.tex"), "w", newline="\n").write("\n".join(lines) + "\n")
for t in tab:
    print(t["num"], t["quality"], t["P_adopted"], round(t["P_sid"], 6), "+-", round(t["sigma_sid"], 6),
          "rel", f"{t['rel_diff']:.1e}", t["poles"], "pub", t["published_poles"], "dark", t["dark_pct"])
