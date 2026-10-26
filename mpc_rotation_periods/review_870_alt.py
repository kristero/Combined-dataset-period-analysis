"""
(870) Manto: convex model at the competing period, as for (2503) in review_checks.half_period_model. The window scan of
damit_longscan.py has its best minimum in the window of the second candidate family at 122.1631 h (chi2 ratio 1.058
against 171.73 h); convexinv from the 12 initial poles at that period, refinement of the best and mirror solutions
(damit_mirrors.do_run), and the principal-axis check of the refined solutions. Run on the HPC (Linux DAMIT build).
Writes damit_runs/paper_870_alt122/ and review_870_alt.json
"""
import glob
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import damit_mirrors as DM  # noqa: E402
import damit_run as R  # noqa: E402
from review_checks import inertia  # noqa: E402
from review_inertia2 import vertices  # noqa: E402

P_ALT = 122.1631
src = os.path.join(ROOT, "damit_runs", "paper_870")
run = os.path.join(ROOT, "damit_runs", "paper_870_alt122")
os.makedirs(os.path.join(run, "scan"), exist_ok=True)
for f in ("lcs", "prep.json", "prep.npz"):
    shutil.copy(os.path.join(src, f), os.path.join(run, f))
open(os.path.join(run, "scan", "out_periods"), "w").write(f"{P_ALT:.6f} 0.120110 1.0 50 0.5\n")
print("poles:", R.do_poles(run), flush=True)
print("mirrors:", DM.do_run(run), flush=True)
pj = json.load(open(os.path.join(run, "poles", "poles.json")))
mj = json.load(open(os.path.join(run, "poles", "mirrors.json")))
out = {"P_alt": P_ALT, "best": pj["best"], "mirrors": mj, "principal_axis": {}}
for areas in sorted(glob.glob(os.path.join(run, "poles", "areas_best")) + glob.glob(os.path.join(run, "poles", "areas_mirror*"))):
    k = os.path.basename(areas)
    try:
        r = inertia(vertices(areas, "alt870_" + k))
    except Exception as e:  # noqa: BLE001
        r = {"error": str(e)[:200]}
    out["principal_axis"][k] = r
    print(k, r, flush=True)
# fit quality relative to the adopted-period model of paper_870 (same data, same settings)
m0 = json.load(open(os.path.join(src, "poles", "mirrors.json")))
out["dev_ratio_alt_over_adopted"] = (mj["within"][0]["dev"] / m0["within"][0]["dev"]) if mj.get("within") else None
out["chi2_ratio_alt_over_adopted"] = out["dev_ratio_alt_over_adopted"] ** 2 if out["dev_ratio_alt_over_adopted"] else None
json.dump(out, open(os.path.join(ROOT, "review_870_alt.json"), "w"), indent=1, default=float)
print("chi2 ratio (alt / adopted):", out["chi2_ratio_alt_over_adopted"])
