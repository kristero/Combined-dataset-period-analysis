"""
Broad DAMIT period scans of the longest periods (adopted P > 250 h): 120-3000 h, on all data and on two independent
halves (odd and even apparitions, gap > 60 d), at DAMIT's default step 0.8 dP (dP = 0.5 P^2 / T of each subset), six
initial poles per trial period, degree/order 4, 6 rows, 30 iterations, phase slope k fitted. The range is split into
chunks that run in parallel. A real rotation period should give a unique chi^2 minimum that is the same in both halves.
Output: damit_runs/paper_<N>/longrange/<subset>/chunk_<k>/out_periods, longperiod_longrange.json
Usage: python damit_longrange.py [--workers 112] [--summary]
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
TARGETS = ["3216", "3173", "1944", "3459", "4371", "4129"]
P_MIN, P_MAX, STEP, N_CHUNK = 120.0, 3000.0, 0.8, 24
LM, ROWS, ITER = 4, 6, 30


def subsets(num):
    """Write lcs files for all data and for odd / even apparitions; return {name: (dir, T_h, n)}."""
    run = os.path.join(RUNS, f"paper_{num}")
    prep = np.load(os.path.join(run, "prep.npz"), allow_pickle=True)
    jd = prep["jd_lt"]
    order = np.argsort(jd)
    lab = np.zeros(len(jd), int)
    cuts = np.where(np.diff(jd[order]) > 60.0)[0]
    for k, g in enumerate(np.split(order, cuts + 1)):
        lab[g] = k
    lines = open(os.path.join(run, "lcs")).read().split("\n")
    nlc, i, blocks = int(lines[0]), 1, []
    for _ in range(nlc):
        n = int(lines[i].split()[0])
        blocks.append(lines[i + 1:i + 1 + n])
        i += 1 + n
    fo = prep["file_order"]
    out = {}
    for name, keep in (("all", np.ones(len(jd), bool)), ("odd", lab % 2 == 1), ("even", lab % 2 == 0)):
        d = os.path.join(run, "longrange", name)
        os.makedirs(d, exist_ok=True)
        k, kept_blocks = 0, []
        for bl in blocks:
            kb = [ln for q, ln in enumerate(bl) if keep[fo[k + q]]]
            k += len(bl)
            if len(kb) >= 10:
                kept_blocks.append(kb)
        txt = [str(len(kept_blocks))]
        for bl in kept_blocks:
            txt += [f"{len(bl)} 0"] + bl
        open(os.path.join(d, "lcs"), "w", newline="\n").write("\n".join(txt) + "\n")
        T = float(np.ptp(jd[keep]) * 24.0)
        out[name] = (d, T, int(keep.sum()))
    return out


def chunks(T):
    """Split [P_MIN, P_MAX] into N_CHUNK pieces with equal numbers of trial periods (equal steps in frequency)."""
    f = np.linspace(1.0 / P_MAX, 1.0 / P_MIN, N_CHUNK + 1)
    P = np.sort(1.0 / f)
    return [(float(P[i]), float(P[i + 1])) for i in range(N_CHUNK)]


def scan(job):
    d, k, lo, hi = job
    c = os.path.join(d, f"chunk_{k}")
    if os.path.exists(os.path.join(c, "done")):
        return d, k, "cached"
    os.makedirs(c, exist_ok=True)
    try:
        open(os.path.join(c, "scan_in"), "w", newline="\n").write(
            W.scan_text(lo, hi, coef=STEP, lm=LM, rows=ROWS, stop=ITER))
        _, sec = W.run(d, "period_scan", f"chunk_{k}/scan_in chunk_{k}/out_periods", timeout=24 * 3600)
        open(os.path.join(c, "done"), "w").write(f"{sec:.0f}\n")
        return d, k, f"{sec / 60:.1f} min"
    except Exception:  # noqa: BLE001
        return d, k, "FAILED " + traceback.format_exc()[-300:]


def summary():
    out = {}
    for num in TARGETS:
        ad = json.load(open(os.path.join(ROOT, "adopted.json")))[num]
        res = {}
        for name in ("all", "odd", "even"):
            d = os.path.join(RUNS, f"paper_{num}", "longrange", name)
            rows = []
            for k in range(N_CHUNK):
                f = os.path.join(d, f"chunk_{k}", "out_periods")
                if os.path.exists(os.path.join(d, f"chunk_{k}", "done")):
                    rows += [ln.split() for ln in open(f) if len(ln.split()) == 5]
            if not rows:
                continue
            a = np.array(rows, float)
            a = a[np.isfinite(a[:, 1])]
            a = a[np.argsort(a[:, 0])]
            n = int(open(os.path.join(d, "lcs")).read().split("\n")[0]) and sum(
                1 for ln in open(os.path.join(d, "lcs")) if len(ln.split()) == 8)
            thr = 1 + 0.5 * np.sqrt(2.0 / (n - (LM + 1) ** 2 - 6))
            i = int(np.argmin(a[:, 1]))
            chi = (a[:, 1] / a[i, 1]) ** 2
            # distinct local minima within the threshold (separated by more than 2 %)
            mins = []
            for j in np.argsort(a[:, 1]):
                if chi[j] > thr:
                    break
                if all(abs(a[j, 0] - m) > 0.02 * m for m in mins):
                    mins.append(float(a[j, 0]))
            near = {f"{P:g}": float(chi[np.abs(a[:, 0] - P) <= 0.01 * P].min()) if np.any(np.abs(a[:, 0] - P) <= 0.01 * P)
                    else None for P in (ad["P"], ad["P"] / 2, 2 * ad["P"])}
            res[name] = {"n_points": n, "n_trials": int(len(a)), "P_best": float(a[i, 0]), "dev_min": float(a[i, 1]),
                         "dev_median": float(np.median(a[:, 1])), "threshold": float(thr),
                         "minima_within_threshold": mins[:20], "n_minima_within_threshold": len(mins),
                         "chi2_ratio_at_adopted_family": near}
        out[num] = {"adopted_P": ad["P"], "quality": ad["quality"], "subsets": res}
    json.dump(out, open(os.path.join(ROOT, "longperiod_longrange.json"), "w"), indent=1)
    for num, o in out.items():
        print(f"({num}) adopted {o['adopted_P']:g} h {o['quality']}")
        for name, r in o["subsets"].items():
            print(f"   {name:4s} n={r['n_points']:5d} trials {r['n_trials']:5d} best {r['P_best']:8.2f} h  "
                  f"minima within threshold: {r['n_minima_within_threshold']} {[round(x) for x in r['minima_within_threshold'][:8]]}  "
                  f"chi2 ratio at P, P/2, 2P: {r['chi2_ratio_at_adopted_family']}")


def main(argv):
    if "--summary" in argv:
        summary()
        return
    workers = int(argv[argv.index("--workers") + 1]) if "--workers" in argv else 112
    jobs = []
    for num in TARGETS:
        for name, (d, T, n) in subsets(num).items():
            for k, (lo, hi) in enumerate(chunks(T)):
                jobs.append((d, k, lo, hi))
    print(f"{len(jobs)} scan chunks", flush=True)
    with mp.get_context("spawn").Pool(workers) as pool:
        for i, (d, k, msg) in enumerate(pool.imap_unordered(scan, jobs)):
            print(f"[{i + 1}/{len(jobs)}] {os.path.relpath(d, RUNS)} chunk {k}: {msg}", flush=True)
    summary()


if __name__ == "__main__":
    main(sys.argv[1:])
