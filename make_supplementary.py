"""
Supplementary Table S1: the S-method values of every dataset and candidate period (Section 3.3), from the
result files of separate_analysis.py.

python make_supplementary.py [results_dir]   (default paper_figures/rev3_results/separate)
  -> <results_dir>/../supplementary_table_S1.csv
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SEP = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "paper_figures", "rev3_results", "separate")
ORDER = [1951, 1963, 2134, 2150, 2607, 2968, 2971, 3081, 3173, 3473, 3716, 4303]
COLS = ["asteroid", "candidate_h", "dataset", "N", "p_h", "w", "R2", "sigma_above_mean", "maxima", "minima",
        "used", "reason_not_used", "P_S_h", "sigma_kv_h", "log_xi"]


def rows():
    for n in ORDER:
        path = os.path.join(SEP, f"{n}.json")
        if not os.path.exists(path):
            continue
        s = json.load(open(path, encoding="utf-8"))
        for cand, c in s["candidates"].items():
            per = {r["ds"]: r for r in c.get("per_dataset", [])}
            for ds in sorted(s["datasets"]):
                r = per.get(ds)
                if r is None:
                    continue
                used = ds in c.get("used", [])
                yield {"asteroid": n, "candidate_h": cand, "dataset": ds, "N": r.get("N"),
                       "p_h": round(r["p"], 5), "w": round(r["w"], 3), "R2": round(r.get("R2", float("nan")), 3),
                       "sigma_above_mean": round(r.get("sigma_above_mean", float("nan")), 1),
                       "maxima": r.get("n_max"), "minima": r.get("n_min"), "used": "yes" if used else "no",
                       "reason_not_used": "" if used else c.get("rejected", {}).get(ds, ""),
                       "P_S_h": round(c["P"], 5) if "P" in c else "", "sigma_kv_h": round(c["sigma_kv"], 5) if "sigma_kv" in c else "",
                       "log_xi": round(c["log_xi"], 2) if "log_xi" in c else ""}


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.normpath(SEP)), "supplementary_table_S1.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        k = 0
        for r in rows():
            w.writerow(r)
            k += 1
    print(out, k, "rows")
