"""Single-dataset example without daily aliases: the TESS data (C57G) of the quality-A asteroids, reduced with the
separate-dataset method (Eqs. 3-5). Ground-based single datasets always show daily aliases with 90-98 % of the
power of the true peak (search_single_dataset_example.py), so only the continuous TESS photometry can give one
distinct peak. Ranks by the power of the adopted family relative to the strongest other peak (exact chi2).
Output: tess_example_candidates.json"""
import json
import multiprocessing as mp
import os

import numpy as np

from search_single_dataset_example import evaluate

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    A = json.load(open(os.path.join(ROOT, "adopted.json"), encoding="utf-8"))
    jobs = []
    for num, ad in A.items():
        if ad["quality"] != "A" or ad["P"] > 100:
            continue
        a = np.load(os.path.join(ROOT, "results", num, f"{num}_arrays.npz"))
        if "C57G" in set(a["sheet"]):       # TESS retained in the combined analysis
            jobs.append((int(num), "C57G", ad["P"]))
    print(len(jobs), "TESS datasets", flush=True)
    with mp.get_context("spawn").Pool(min(16, len(jobs))) as pool:
        out = [r for r in pool.imap_unordered(evaluate, jobs) if r]
    out.sort(key=lambda r: -r["dominance"])
    json.dump(out, open(os.path.join(ROOT, "tess_example_candidates.json"), "w"), indent=1)
    for r in out:
        print(f"({r['asteroid']}) n={r['n']} dominance {r['dominance']:.2f} family global max {r.get('family_is_global_max')} "
              f"P_family {r.get('P_best_family', 0):.4f} (adopted {r.get('P_adopted', 0):.4f}) power {r.get('power_family', 0):.3f} "
              f"full period {r.get('P_is_full_period')}")


if __name__ == "__main__":
    main()
