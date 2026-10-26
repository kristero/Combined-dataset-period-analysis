"""
Fine DAMIT period scans for the asteroids with adopted periods longer than one day (P > 24 h).

Candidate windows per asteroid (from longperiod_ls.json, original data):
  - the adopted period P and its harmonics P/2 and 2P
  - the six strongest peak families of the fine-grid Lomb-Scargle periodogram (24-5000 h)
  - the strongest short period (2-24 h)
Each window is P_c +- max(1.5 P_c^2 / P_orb, 3 P_c^2 / T) (synodic-sidereal range, or three minimum separations
dP = 0.5 P^2 / T), scanned with period_scan at a step of 0.2 dP (DAMIT's default is 0.8 dP), six initial poles per
trial period, degree/order 4, 6 rows, 30 iterations, phase slope k fitted (a, d fixed). Windows that overlap are
merged. All windows of an asteroid use the same data and model, so their chi^2 values are directly comparable.
Output: damit_runs/<run>/longscan/<label>/out_periods, longperiod_damit.json
Usage: python damit_longscan.py [--workers 64] [--summary]
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import damit_wsl as W  # noqa: E402

RUNS = os.path.join(ROOT, "damit_runs")
STEP = 0.2
LM, ROWS, ITER = 4, 6, 30


def windows(num, ls, info):
    jd = np.load(os.path.join(RUNS, f"paper_{num}", "prep.npz"))["jd_lt"]
    T = (jd.max() - jd.min()) * 24.0
    r = np.load(os.path.join(RUNS, f"paper_{num}", "prep.npz"))["sun"]
    P_orb = float(np.mean(np.linalg.norm(r, axis=1))) ** 1.5 * 365.25 * 24
    v = ls["versions"][0]
    P0 = ls["adopted_P"]
    cands = [("adopted", P0), ("adopted_half", P0 / 2), ("adopted_double", 2 * P0)]
    for i, c in enumerate(sorted(v["candidates"], key=lambda c: -c["power"])[:6]):
        cands.append((f"ls{i + 1}", c["P"]))
    cands.append(("short_best", v["short_best"]["P"]))
    wins = []
    for label, P in cands:
        half = max(1.5 * P * P / P_orb, 3 * P * P / T)
        lo, hi = P - half, P + half
        for w in wins:
            if lo <= w["hi"] and hi >= w["lo"]:
                w["lo"], w["hi"] = min(w["lo"], lo), max(w["hi"], hi)
                w["labels"].append(label)
                break
        else:
            wins.append({"lo": lo, "hi": hi, "labels": [label], "P_c": P})
    for w in wins:
        w["dP"] = 0.5 * w["P_c"] ** 2 / T
        w["n_trials"] = int(np.ceil((1 / w["lo"] - 1 / w["hi"]) / (STEP * 0.5 / (T / 24.0) / 24.0))) + 1
    return wins, T, P_orb


def scan_one(job):
    run, label, lo, hi = job
    d = os.path.join(run, "longscan", label)
    out = os.path.join(d, "out_periods")
    if os.path.exists(os.path.join(d, "done")):
        return run, label, "cached"
    os.makedirs(d, exist_ok=True)
    try:
        open(os.path.join(d, "scan_in"), "w", newline="\n").write(
            W.scan_text(lo, hi, coef=STEP, lm=LM, rows=ROWS, stop=ITER))
        _, sec = W.run(run, "period_scan", f"longscan/{label}/scan_in longscan/{label}/out_periods", timeout=12 * 3600)
        open(os.path.join(d, "done"), "w").write(f"{sec:.0f}\n")
        return run, label, f"{sec / 60:.1f} min"
    except Exception:  # noqa: BLE001
        return run, label, "FAILED " + traceback.format_exc()[-300:]


def summary():
    LS = json.load(open(os.path.join(ROOT, "longperiod_ls.json")))
    plan = json.load(open(os.path.join(ROOT, "longperiod_damit_plan.json")))
    out = {}
    for num, p in plan.items():
        run = os.path.join(RUNS, f"paper_{num}")
        n = json.load(open(os.path.join(run, "prep.json")))["n"]
        thr = 1 + 0.5 * np.sqrt(2 / (n - (LM + 1) ** 2 - 6))
        rows = []
        for w in p["windows"]:
            f = os.path.join(run, "longscan", w["name"], "out_periods")
            if not os.path.exists(os.path.join(run, "longscan", w["name"], "done")):
                continue
            a = np.array([ln.split() for ln in open(f) if len(ln.split()) == 5], float).reshape(-1, 5)
            a = a[np.isfinite(a[:, 1])]
            if not len(a):
                continue
            i = int(np.argmin(a[:, 1]))
            rows.append({"window": w["name"], "labels": w["labels"], "lo": w["lo"], "hi": w["hi"],
                         "P_best": float(a[i, 0]), "dev_min": float(a[i, 1]), "dev_median": float(np.median(a[:, 1])),
                         "n_trials": int(len(a))})
        if not rows:
            continue
        rows.sort(key=lambda r: r["dev_min"])
        best = rows[0]
        for r in rows:
            r["chi2_ratio_to_best"] = (r["dev_min"] / best["dev_min"]) ** 2
            r["within_threshold"] = bool(r["chi2_ratio_to_best"] <= thr)
        adopted = [r for r in rows if "adopted" in r["labels"]]
        out[num] = {"adopted_P": LS[num]["adopted_P"], "quality": LS[num]["quality"], "threshold": thr,
                    "best_window": best["window"], "best_labels": best["labels"], "P_best": best["P_best"],
                    "adopted_is_best": bool(adopted and adopted[0] is best),
                    "adopted_within_threshold": bool(adopted and adopted[0]["within_threshold"]),
                    "n_within_threshold": int(sum(r["within_threshold"] for r in rows)), "windows": rows}
    json.dump(out, open(os.path.join(ROOT, "longperiod_damit.json"), "w"), indent=1)
    for num, o in sorted(out.items(), key=lambda x: -x[1]["adopted_P"]):
        print(f"({num}) adopted {o['adopted_P']:g} h {o['quality']}: DAMIT best {o['P_best']:.4f} h "
              f"[{','.join(o['best_labels'])}], adopted best={o['adopted_is_best']}, "
              f"within threshold={o['n_within_threshold']} windows")
        for r in o["windows"][:5]:
            print(f"     {r['window']:12s} {','.join(r['labels']):30s} P {r['P_best']:10.4f}  dev {r['dev_min']:.5f} "
                  f"(median {r['dev_median']:.5f})  chi2 ratio {r['chi2_ratio_to_best']:.4f}")


def main(argv):
    if "--summary" in argv:
        summary()
        return
    workers = int(argv[argv.index("--workers") + 1]) if "--workers" in argv else 64
    LS = json.load(open(os.path.join(ROOT, "longperiod_ls.json")))
    plan, jobs = {}, []
    for num, ls in LS.items():
        run = os.path.join(RUNS, f"paper_{num}")
        info = json.load(open(os.path.join(run, "prep.json")))
        wins, T, P_orb = windows(num, ls, info)
        for k, w in enumerate(wins):
            w["name"] = f"w{k}_" + w["labels"][0]
            jobs.append((run, w["name"], w["lo"], w["hi"]))
        plan[num] = {"T_h": T, "P_orb_h": P_orb, "windows": wins}
    json.dump(plan, open(os.path.join(ROOT, "longperiod_damit_plan.json"), "w"), indent=1)
    print(f"{len(jobs)} windows, {sum(w['n_trials'] for p in plan.values() for w in p['windows'])} trial periods",
          flush=True)
    # longest first
    cost = {(j[0], j[1]): json.load(open(os.path.join(j[0], "prep.json")))["n"] *
            next(w["n_trials"] for w in plan[os.path.basename(j[0]).split("_")[1]]["windows"] if w["name"] == j[1])
            for j in jobs}
    jobs.sort(key=lambda j: -cost[(j[0], j[1])])
    with mp.get_context("spawn").Pool(workers) as pool:
        for k, (run, label, msg) in enumerate(pool.imap_unordered(scan_one, jobs)):
            print(f"[{k + 1}/{len(jobs)}] {os.path.basename(run)} {label}: {msg}", flush=True)
    summary()


if __name__ == "__main__":
    main(sys.argv[1:])
