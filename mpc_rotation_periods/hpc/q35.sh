ssh -q -o BatchMode=yes luhpc 'cd ~/mpc_redo && $HOME/micromamba/envs/py312/bin/python - <<EOF
import json, os, sys
sys.path.insert(0, ".")
from damit_eval import chi2_threshold
for a in ["paper_4618", "paper_4761", "val_553", "val_594", "val_905", "val_1608"]:
    r = {}
    for w in "CA":
        pj = json.load(open(f"damit_runs/{a}_fp{w}/poles/poles.json"))
        b = pj["best"]
        r[w] = (b["P"], b["dev"], b["lambda"], b["beta"], b.get("dark_pct"))
    n = json.load(open(f"damit_runs/{a}/prep.json"))["n"]
    thr = chi2_threshold(n - 54)
    print(a, "C:", r["C"], "A:", r["A"], "ratio A/C %.3f thr %.4f" % ((r["A"][1] / r["C"][1]) ** 2, thr))
F = json.load(open("fine_checks.json"))
for k, v in F.get("inertia", {}).items():
    print("INERTIA", k, "ratio_fine_over_12 %.4f" % v.get("chi2_ratio_fine_over_12", float("nan")), "only_mirror", v.get("only_mirror_ambiguity"),
          [(round(w["lambda"]), round(w["beta"]), round(w["ratio"], 4), w.get("dark_pct"), round(w["angle"], 1) if isinstance(w["angle"], float) else w["angle"]) for w in v["within"]],
          "starts within:", [(round(s["ratio"], 4), round(s["angle"], 1) if isinstance(s["angle"], float) else s["angle"]) for s in v["starts_within"]][:8])
for k, v in F.get("synsid", {}).items():
    if "periodogram" in v:
        p = v["periodogram"]
        print("PGRAM", k, {kk: (round(vv, 6) if isinstance(vv, float) else vv) for kk, vv in p.items()})
EOF' 2>&1 | grep -v "micromamba/bin/micromamba: No such"
