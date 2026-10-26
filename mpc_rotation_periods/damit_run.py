"""
DAMIT convex inversion of the MPC data (inputs from damit_prep.py), run with the C programs of damit/build in WSL.

Stages (each skips work that is already done):
  scan   period_scan over [P_lo, P_hi] (damit_prep: synodic-sidereal range + 3 sigma), step 0.8 dP, six initial
         poles per trial period (the program's default), resolution SCAN_LM / SCAN_ROWS, SCAN_ITER iterations
  poles  convexinv at the best scan period from 12 isotropic initial poles (icosahedron vertices), degree/order 6,
         8 rows, START_ITER iterations; every solution kept (par_k, areas_k, dev); the best one polished with
         POLISH_ITER iterations (par_best, areas_best)
Phase function: a = 0.5 and d = 0.1 fixed, slope k fitted (damit_wsl.PHASE_FREE). Inputs: light-time corrected
epochs, intensities reduced to unit distances without phase correction, asteroid-centric ecliptic J2000 Sun and Earth
vectors, one relative lightcurve per dataset (checked by damit_input_check.py).
  loao   leave-one-apparition-out: for every template window j (amp_methods.template_fit) the best solution is
         refitted without the points of window j (convexinv, 50 iterations, started from the best solution)
Usage: python damit_run.py scan|poles|loao [--workers 18] [--only val|paper] [--nums a,b]
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import re
import sys
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import damit_wsl as W  # noqa: E402

RUNS = os.path.join(ROOT, "damit_runs")
SCAN_LM, SCAN_ROWS, SCAN_ITER = 4, 6, 30
FINAL_LM, FINAL_ROWS, START_ITER, POLISH_ITER, LOAO_ITER = 6, 8, 50, 150, 50
# 12 isotropic initial poles: the vertices of an icosahedron (near-polar ones at |beta| = 85 deg, so that lambda is
# defined); the literature uses about 10 isotropic starts (Durech et al. 2016; Hanus et al. 2021)
POLES = [(0.0, 85.0), (180.0, -85.0)] + [(72.0 * k, 26.57) for k in range(5)] + \
        [(36.0 + 72.0 * k, -26.57) for k in range(5)]


def run_dirs(only=None, nums=None):
    out = []
    for d in sorted(os.listdir(RUNS)):
        p = os.path.join(RUNS, d)
        if not os.path.isfile(os.path.join(p, "prep.json")):
            continue
        s, n = d.split("_", 1)
        if only and s != only:
            continue
        if nums and n not in nums:
            continue
        out.append(p)
    return out


def last_dev(stdout):
    devs = re.findall(r"dev\s+([0-9.eE+-]+)", stdout)
    dark = re.findall(r"dark facet with area\s+([0-9.]+)%", stdout)
    return (float(devs[-1]) if devs else np.nan), (float(dark[-1]) if dark else np.nan)


# ----------------------------------------------------------------------------------------------------------- scan
def expected_trials(run):
    """Number of trial periods period_scan computes (same formula as period_scan.c)."""
    info = json.load(open(os.path.join(run, "prep.json")))
    jd = np.load(os.path.join(run, "prep.npz"))["jd_lt"]
    fstep = 0.5 / (jd.max() - jd.min()) / 24 * 0.8
    return int((1 / info["P_lo"] - 1 / info["P_hi"]) / fstep) + 1


def n_lines(path):
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return sum(1 for ln in f if len(ln.split()) == 5)


def do_scan(run):
    import time
    out = os.path.join(run, "scan", "out_periods")
    done = os.path.join(run, "scan", "done")
    if os.path.exists(out) and os.path.getsize(out) > 0 and os.path.exists(done):
        return run, "cached"
    os.makedirs(os.path.join(run, "scan"), exist_ok=True)
    # a scan left running by an interrupted orchestrator: wait for it, then accept it if complete
    waited = False
    while W.running_in(run):
        waited = True
        time.sleep(30)
    if n_lines(out) >= expected_trials(run):
        open(done, "w").write("completed by an earlier process\n")
        return run, "completed earlier" + (" (waited)" if waited else "")
    # lock against a second orchestrator
    lock = os.path.join(run, "scan", "lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
    except FileExistsError:
        if W.running_in(run):
            return run, "locked by another process"
        os.remove(lock)
        return do_scan(run)
    info = json.load(open(os.path.join(run, "prep.json")))
    open(os.path.join(run, "scan", "scan_in"), "w", newline="\n").write(
        W.scan_text(info["P_lo"], info["P_hi"], lm=SCAN_LM, rows=SCAN_ROWS, stop=SCAN_ITER))
    try:
        _, sec = W.run(run, "period_scan", "scan/scan_in scan/out_periods", timeout=6 * 3600)
        # phase setting of this scan (the first 171 scans were run with a, d, k free; reruns with k only)
        open(done, "w").write(f"{sec:.1f}\nphase_free {W.PHASE_FREE}\n")
    finally:
        os.remove(lock)
    return run, f"{sec / 60:.1f} min"


def read_scan(run):
    a = np.loadtxt(os.path.join(run, "scan", "out_periods"), ndmin=2)
    return {"P": a[:, 0], "dev": a[:, 1], "chi2": a[:, 2], "iter": a[:, 3], "dark": a[:, 4]}


# ---------------------------------------------------------------------------------------------------------- poles
def do_poles(run):
    pdir = os.path.join(run, "poles")
    summ = os.path.join(pdir, "poles.json")
    if os.path.exists(summ):
        return run, "cached"
    os.makedirs(pdir, exist_ok=True)
    sc = read_scan(run)
    ok = np.isfinite(sc["dev"])
    P_best = float(sc["P"][ok][int(np.argmin(sc["dev"][ok]))])
    n = json.load(open(os.path.join(run, "prep.json")))["n"]
    tmax = max(900, 0.1 * n)                      # typical run: ~0.005 s per point for 100 iterations
    res, failed = [], []
    for k, (lam, bet) in enumerate(POLES):
        open(os.path.join(pdir, f"in_{k}"), "w", newline="\n").write(
            W.par_text(lam, bet, P_best, lm=FINAL_LM, rows=FINAL_ROWS, stop=START_ITER))
        try:
            so, sec = W.run(run, "convexinv", f"-v -o poles/areas_{k} -p poles/par_{k} poles/in_{k} /dev/null",
                            timeout=tmax)
            dev, dark = last_dev(so)
            par = W.read_par(os.path.join(pdir, f"par_{k}"))
            if not (np.isfinite(dev) and np.isfinite(par["P"])):
                raise RuntimeError("non-finite solution")
            res.append({**par, "idx": k, "start": [lam, bet], "dev": dev, "dark_pct": dark, "sec": sec})
        except Exception as exc:  # noqa: BLE001
            failed.append({"idx": k, "start": [lam, bet], "error": str(exc)[:200]})
    if not res:
        raise RuntimeError("all pole starts failed")
    # polish the best start: convexinv again from its pole, period and phase slope, POLISH_ITER iterations
    b = min(res, key=lambda r: r["dev"])
    open(os.path.join(pdir, "in_best"), "w", newline="\n").write(
        W.par_text(b["lambda"], b["beta"], b["P"], lm=FINAL_LM, rows=FINAL_ROWS, stop=POLISH_ITER,
                   phase=(b["a"], b["d"], b["k"])))
    so, sec = W.run(run, "convexinv", "-v -o poles/areas_best -p poles/par_best poles/in_best /dev/null",
                    timeout=3 * tmax)
    dev, dark = last_dev(so)
    best = {**W.read_par(os.path.join(pdir, "par_best")), "idx": "best", "from_start": b["idx"], "dev": dev,
            "dark_pct": dark, "sec": sec}
    if not (np.isfinite(dev) and dev <= b["dev"]):          # polishing did not improve: keep the start
        best = {**b, "from_start": b["idx"], "polish_dev": dev}
        dev = b["dev"]
    json.dump({"P_scan_best": P_best, "phase_free": W.PHASE_FREE, "start_iter": START_ITER,
               "polish_iter": POLISH_ITER, "solutions": res, "failed": failed, "best": best},
              open(summ, "w"), indent=1)
    return run, f"best dev {dev:.5f}" + (f" ({len(failed)} starts failed)" if failed else "")


def best_solution(run):
    s = json.load(open(os.path.join(run, "poles", "poles.json")))
    b = s.get("best") or min(s["solutions"], key=lambda r: r["dev"])
    return b, s


# ----------------------------------------------------------------------------------------------------------- loao
def do_loao(run):
    import amp_data as D
    import amp_methods as M
    ldir = os.path.join(run, "loao")
    summ = os.path.join(ldir, "loao.json")
    if os.path.exists(summ):
        return run, "cached"
    os.makedirs(ldir, exist_ok=True)
    info = json.load(open(os.path.join(run, "prep.json")))
    d = D.load(info["sample"], info["num"])
    tp = M.template_fit(d)
    if tp is None:
        json.dump({"windows": []}, open(summ, "w"))
        return run, "no windows"
    prep = np.load(os.path.join(run, "prep.npz"), allow_pickle=True)
    src = prep["src_index"]                         # prep row -> amp_data index
    order = prep["file_order"]                      # lcs line order -> prep row
    pos_of_src = np.full(len(d["t"]), -1)
    pos_of_src[src] = np.arange(len(src))
    lines = open(os.path.join(run, "lcs")).read().split("\n")
    # parse lcs blocks
    nlc, i = int(lines[0]), 1
    blocks = []
    for _ in range(nlc):
        n = int(lines[i].split()[0])
        blocks.append(lines[i + 1:i + 1 + n])
        i += 1 + n
    row_of_line = order                             # k-th data line (over all blocks) is prep row order[k]
    b, _ = best_solution(run)
    out = []
    for j, r in enumerate(tp["rows"]):
        drop = set(pos_of_src[r["idx"]].tolist())
        k = 0
        keep_blocks = []
        for bl in blocks:
            kept = [ln for q, ln in enumerate(bl) if row_of_line[k + q] not in drop]
            k += len(bl)
            if len(kept) >= 3:
                keep_blocks.append(kept)
        txt = [str(len(keep_blocks))]
        for bl in keep_blocks:
            txt.append(f"{len(bl)} 0")
            txt.extend(bl)
        open(os.path.join(ldir, f"lcs_{j}"), "w", newline="\n").write("\n".join(txt) + "\n")
        open(os.path.join(ldir, f"in_{j}"), "w", newline="\n").write(
            W.par_text(b["lambda"], b["beta"], b["P"], lm=FINAL_LM, rows=FINAL_ROWS, stop=LOAO_ITER,
                       phase=(b["a"], b["d"], b["k"])))
        try:
            so, sec = W.run(run, "convexinv", f"-v -o loao/areas_{j} -p loao/par_{j} loao/in_{j} /dev/null",
                            lcs=f"loao/lcs_{j}", timeout=max(900, 0.05 * info["n"]))
            dev, dark = last_dev(so)
            out.append({"j": j, "n_dropped": len(drop), "dev": dev, "dark_pct": dark,
                        **W.read_par(os.path.join(ldir, f"par_{j}"))})
        except Exception as exc:  # noqa: BLE001
            out.append({"j": j, "n_dropped": len(drop), "failed": str(exc)[:200]})
        os.remove(os.path.join(ldir, f"lcs_{j}"))
    json.dump({"windows": out}, open(summ, "w"), indent=1)
    return run, f"{len(out)} windows"


def main(argv):
    stage = argv[0]
    workers = int(argv[argv.index("--workers") + 1]) if "--workers" in argv else 18
    only = argv[argv.index("--only") + 1] if "--only" in argv else None
    nums = set(argv[argv.index("--nums") + 1].split(",")) if "--nums" in argv else None
    runs = run_dirs(only, nums)
    if stage == "scanpoles":
        # unfinished scans first (longest first), then the pole searches of the finished ones
        def key(r):
            info = json.load(open(os.path.join(r, "prep.json")))
            return (os.path.exists(os.path.join(r, "scan", "done")), -info["n"] * info["n_trial_periods"])
        runs.sort(key=key)
    if stage == "scan":
        # longest jobs first
        cost = {r: json.load(open(os.path.join(r, "prep.json"))) for r in runs}
        runs.sort(key=lambda r: -cost[r]["n"] * cost[r]["n_trial_periods"])
    jobs = [(stage, r) for r in runs]
    if stage == "loao":
        # loao needs the Python template fit -> processes, each launching its own WSL jobs
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(workers) as pool:
            for k, (r, msg) in enumerate(pool.imap_unordered(_safe, jobs)):
                print(f"[{k + 1}/{len(runs)}] {os.path.basename(r)}: {msg}", flush=True)
    else:
        with cf.ThreadPoolExecutor(workers) as ex:
            for k, (r, msg) in enumerate(ex.map(_safe, jobs)):
                print(f"[{k + 1}/{len(runs)}] {os.path.basename(r)}: {msg}", flush=True)


def do_scanpoles(run):
    _, m1 = do_scan(run)
    _, m2 = do_poles(run)
    return run, f"scan {m1}; poles {m2}"


def _safe(job):
    stage, r = job
    try:
        return {"scan": do_scan, "poles": do_poles, "loao": do_loao, "scanpoles": do_scanpoles}[stage](r)
    except Exception:  # noqa: BLE001
        return r, "FAILED " + traceback.format_exc()[-400:]


if __name__ == "__main__":
    main(sys.argv[1:])
