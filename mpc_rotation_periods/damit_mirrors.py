"""
Polished competing (mirror) pole solutions of the convex models.

damit_run.py refines only the best of the 12 start solutions (POLISH_ITER iterations), while damit_eval.pole_clusters
applies the uniqueness threshold to the unrefined 50-iteration starts. Here the best start of every other pole cluster
that comes close to the threshold (chi2 ratio to the best start below 1 + 1.5 (thr - 1)) is refined in the same way
(same resolution, POLISH_ITER iterations, from its own pole, period and phase slope), and the threshold of Hanus et al.
(2021), chi2 <= (1 + 0.5 sqrt(2/nu)) chi2_min, is applied to the refined deviations relative to the refined best
solution. The pole list of the paper table then consists of refined solutions only.
Output: damit_runs/<run>/poles/mirrors.json (+ par_mirror_<idx>, areas_mirror_<idx>)
Usage: python damit_mirrors.py [--only paper] [--nums a,b] [--workers 6] [--force] [--reorder]
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import damit_eval as E  # noqa: E402
import damit_run as R  # noqa: E402
import damit_wsl as W  # noqa: E402

N_PAR = 49 + 3 + 3 - 1          # as damit_eval: shape coefficients + phase + pole/period - one scale
FORCE = "--force" in sys.argv
MERGE_SEP = 10.0                 # deg


def clusters(sols):
    out = []
    for r in sorted(sols, key=lambda r: r["dev"]):
        lam, bet = E.norm_pole(r["lambda"], r["beta"])
        v = E.vec(lam, bet)
        for c in out:
            if E.ang(v, c["v"]) <= E.POLE_SEP:
                c["members"].append(r["idx"])
                break
        else:
            out.append({"v": v, "sol": r, "members": [r["idx"]]})
    return out


def do_run(run):
    pdir = os.path.join(run, "poles")
    out_path = os.path.join(pdir, "mirrors.json")
    if os.path.exists(out_path) and os.path.getmtime(out_path) > os.path.getmtime(os.path.join(pdir, "poles.json")) \
            and not FORCE:
        return run, "cached"
    pj = json.load(open(os.path.join(pdir, "poles.json")))
    best = pj["best"]
    n_data = len(np.load(os.path.join(run, "prep.npz"))["y"])
    thr = E.chi2_threshold(n_data - N_PAR)
    cl = clusters(pj["solutions"])
    best_start = min(pj["solutions"], key=lambda r: r["dev"])
    n = json.load(open(os.path.join(run, "prep.json")))["n"]
    tmax = 3 * max(900, 0.1 * n)
    rows = []
    for c in cl:
        s = c["sol"]
        lam, bet = E.norm_pole(s["lambda"], s["beta"])
        row = {"idx": s["idx"], "members": c["members"], "start": {"lambda": lam, "beta": bet, "dev": s["dev"]},
               "start_ratio": (s["dev"] / best_start["dev"]) ** 2}
        if best.get("from_start") in c["members"]:
            bl, bb = E.norm_pole(best["lambda"], best["beta"])
            row.update({"is_best": True, "polished": {"lambda": bl, "beta": bb, "dev": best["dev"], "P": best["P"]}})
        elif row["start_ratio"] <= 1 + 1.5 * (thr - 1):
            k = s["idx"]
            open(os.path.join(pdir, f"in_mirror_{k}"), "w", newline="\n").write(
                W.par_text(s["lambda"], s["beta"], s["P"], lm=R.FINAL_LM, rows=R.FINAL_ROWS, stop=R.POLISH_ITER,
                           phase=(s["a"], s["d"], s["k"])))
            try:
                so, sec = W.run(run, "convexinv",
                                f"-v -o poles/areas_mirror_{k} -p poles/par_mirror_{k} poles/in_mirror_{k} /dev/null",
                                timeout=tmax)
                dev, dark = R.last_dev(so)
                par = W.read_par(os.path.join(pdir, f"par_mirror_{k}"))
                if not (np.isfinite(dev) and dev <= s["dev"]):
                    raise RuntimeError(f"polishing did not improve (dev {dev})")
                pl, pb = E.norm_pole(par["lambda"], par["beta"])
                row["polished"] = {"lambda": pl, "beta": pb, "dev": dev, "P": par["P"], "dark_pct": dark, "sec": sec}
            except Exception as exc:  # noqa: BLE001
                row["polish_error"] = str(exc)[:200]
        rows.append(row)
    if "dark_pct" in best:
        for r in rows:
            if r.get("is_best"):
                r["polished"]["dark_pct"] = best["dark_pct"]
    res = summarize(rows, thr, n_data)
    json.dump(res, open(out_path, "w"), indent=1, default=float)
    return run, message(res)


def summarize(rows, thr, n_data):
    """Threshold and order relative to the best REFINED solution (a refined competing solution can fit better than
    the refined best start): within = refined solutions with (dev/dev_min)^2 <= thr, best first; each with the
    areas file of its shape."""
    pol = [r for r in rows if "polished" in r]
    dev_min = min(r["polished"]["dev"] for r in pol)
    for r in pol:
        r["ratio"] = (r["polished"]["dev"] / dev_min) ** 2
        r["within"] = bool(r["ratio"] <= thr)
        r["areas"] = "areas_best" if r.get("is_best") else f"areas_mirror_{r['idx']}"
    ok, kept_v = [], []
    for r in sorted([r for r in pol if r["within"]], key=lambda r: r["ratio"]):
        # refined solutions that converged to (practically) the pole of a better one are the same solution; distinct
        # local minima, such as mirror poles at high latitude, can lie closer than POLE_SEP and are kept
        v = E.vec(r["polished"]["lambda"], r["polished"]["beta"])
        if any(E.ang(v, u) <= MERGE_SEP for u in kept_v):
            r["merged_into_better"] = True
            continue
        kept_v.append(v)
        ok.append(r)
    b = ok[0]["polished"]
    mirror = [r for r in ok[1:] if abs(((r["polished"]["lambda"] - b["lambda"] + 180) % 360) - 180) >= 120
              and abs(r["polished"]["beta"] - b["beta"]) < 50]
    return {"threshold": thr, "n_data": n_data, "clusters": rows,
            "within": [{"lambda": r["polished"]["lambda"], "beta": r["polished"]["beta"], "dev": r["polished"]["dev"],
                        "ratio": r["ratio"], "P": r["polished"]["P"], "dark_pct": r["polished"].get("dark_pct"),
                        "areas": r["areas"], "is_best_start": bool(r.get("is_best"))} for r in ok],
            "only_mirror_ambiguity": bool(len(ok) == 1 or (len(ok) == 2 and len(mirror) == 1))}


def message(res):
    return f"{len(res['within'])} within threshold; " + ", ".join(
        f"({w['lambda']:.0f},{w['beta']:+.0f}) r={w['ratio']:.4f}{'' if w['is_best_start'] else ' [refined mirror]'}"
        for w in res["within"]) + f" thr={res['threshold']:.4f}"


def reorder(run):
    """Recompute the summary of an existing mirrors.json (no new refinement)."""
    path = os.path.join(run, "poles", "mirrors.json")
    m = json.load(open(path))
    pj = json.load(open(os.path.join(run, "poles", "poles.json")))
    for r in m["clusters"]:
        if r.get("is_best") and "dark_pct" in pj["best"]:
            r["polished"]["dark_pct"] = pj["best"]["dark_pct"]
    res = summarize(m["clusters"], m["threshold"], m["n_data"])
    json.dump(res, open(path, "w"), indent=1, default=float)
    return run, message(res)


def main(argv):
    only = argv[argv.index("--only") + 1] if "--only" in argv else "paper"
    nums = set(argv[argv.index("--nums") + 1].split(",")) if "--nums" in argv else None
    workers = int(argv[argv.index("--workers") + 1]) if "--workers" in argv else 6
    runs = [r for r in R.run_dirs(only, nums) if os.path.exists(os.path.join(r, "poles", "poles.json"))]
    fn = reorder if "--reorder" in argv else do_run
    if fn is reorder:
        runs = [r for r in runs if os.path.exists(os.path.join(r, "poles", "mirrors.json"))]
    with cf.ThreadPoolExecutor(workers) as ex:
        for fut in cf.as_completed([ex.submit(fn, r) for r in runs]):
            try:
                run, msg = fut.result()
                print(os.path.basename(run), msg, flush=True)
            except Exception as exc:  # noqa: BLE001
                print("ERROR", exc, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
