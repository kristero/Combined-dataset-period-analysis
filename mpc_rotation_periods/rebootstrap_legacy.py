"""
FIRST VERSION of rebootstrap.py, kept unchanged for provenance: it produced the results on which the
preliminary proposals (proposals_legacy.json, evidence_legacy/) and the review were based. Do not use it
for new work: its +/-20 resolution-element search windows overlap for neighbouring comb teeth, and the
shared peak is credited to one of them (see the docstring of the corrected rebootstrap.py). Only this
docstring and the output location differ from the original: results are written as
<N>_rebootstrap_legacy.json into the directory given with --out (required), never into results/<N>/.

Night-block bootstrap of the candidate periods, recomputed from the saved arrays of mpc_redo.py.

Differences from the bootstrap inside mpc_redo.py:
  * the competition is restricted to candidates in the primary range (P <= 240 h), as prescribed in
    the paper (longer periods are considered only when no satisfactory primary-range solution exists);
    candidates with P > 240 h are bootstrapped separately against the best primary candidate;
  * the bootstrap is done for the unweighted and for the noise-weighted periodogram;
  * extra candidate periods can be supplied on the command line (asteroid:P1,P2,...).

Usage: python rebootstrap_legacy.py --out DIR [N ...] [N:P1,P2 ...] [--workers 8]
Output: DIR/<N>_rebootstrap_legacy.json
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
N_BOOT = 300
SPP = 50
P_PRIMARY = 240.0


def ls_power(t, y, freq, dy=None):
    from astropy.timeseries import LombScargle
    return LombScargle(t, y, dy=dy, nterms=2).power(freq, method="fastchi2", assume_regular_frequency=True)


def grid(f0, base, hw_res=20.0):
    df = 1.0 / (SPP * base)
    hw = min(hw_res / base, 0.5 * f0)
    return (f0 - hw) + df * np.arange(int(np.ceil(2 * hw / df)) + 1)


def families(Ps, dPs):
    fam = list(range(len(Ps)))
    for a in range(len(Ps)):
        for b in range(a):
            if abs(2 * Ps[a] - Ps[b]) <= 3 * dPs[b] or abs(Ps[a] - 2 * Ps[b]) <= 3 * dPs[a] or abs(Ps[a] - Ps[b]) <= 3 * dPs[a]:
                fam[a] = fam[b]
                break
    return fam


def run(job):
    num, extra, OUT_DIR = job
    r = json.load(open(os.path.join(RES, str(num), f"{num}_result.json"), encoding="utf-8"))
    a = np.load(os.path.join(RES, str(num), f"{num}_arrays.npz"))
    t, y, sheet = a["t"], a["y"], a["sheet"]
    base = float(t.max() - t.min())
    sd = {ds: np.std(y[sheet == ds], ddof=1) for ds in np.unique(sheet)}
    dy = np.array([sd[s] for s in sheet])
    Ps = [c["P"] for c in r["candidates"]]
    for P in extra:
        if all(abs(P - q) > 0.002 * P for q in Ps):
            Ps.append(float(P))
    # refine every candidate on the full data (unweighted)
    cands = []
    for P in Ps:
        g = grid(24.0 / P, base, 3.0)
        p = ls_power(t, y, g)
        k = int(np.argmax(p))
        Pk = 24.0 / g[k]
        if any(abs(Pk - c["P"]) <= 3 * c["dP"] for c in cands):
            continue
        pw = ls_power(t, y, grid(24.0 / Pk, base, 3.0), dy=dy)
        cands.append({"P": float(Pk), "f": float(g[k]), "power": float(p[k]), "power_weighted": float(pw.max()),
                      "dP": float(Pk ** 2 / 24.0 / base)})
    prim = [c for c in cands if c["P"] <= P_PRIMARY]
    ext = [c for c in cands if c["P"] > P_PRIMARY]
    fam = families([c["P"] for c in prim], [c["dP"] for c in prim])
    for c, fid in zip(prim, fam):
        c["family"] = int(fid)
    night = np.array([f"{s}|{int(np.floor(x + float(a['time_zero'])))}" for s, x in zip(sheet, t)])
    _, inv = np.unique(night, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    bnd = np.searchsorted(inv[order], np.arange(inv.max() + 2))
    groups = [order[bnd[i]:bnd[i + 1]] for i in range(inv.max() + 1)]
    rng = np.random.default_rng(1000 + num)
    grids_p = [grid(c["f"], base) for c in prim]
    grids_e = [grid(c["f"], base) for c in ext]
    bu = np.zeros((N_BOOT, len(prim)))
    bw = np.zeros((N_BOOT, len(prim)))
    loc = np.zeros((N_BOOT, len(prim)))
    be = np.zeros((N_BOOT, len(ext)))
    for b in range(N_BOOT):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        tb, yb, db = t[idx], y[idx], dy[idx]
        for j, g in enumerate(grids_p):
            p = ls_power(tb, yb, g)
            k = int(np.argmax(p))
            bu[b, j], loc[b, j] = p[k], 24.0 / g[k]
            bw[b, j] = float(ls_power(tb, yb, g[max(0, k - 3 * SPP):k + 3 * SPP + 1], dy=db).max())
        for j, g in enumerate(grids_e):
            be[b, j] = float(ls_power(tb, yb, g).max())
    wu, ww = np.argmax(bu, axis=1), np.argmax(bw, axis=1)
    for j, c in enumerate(prim):
        c["win_u"] = float(np.mean(wu == j))
        c["win_w"] = float(np.mean(ww == j))
        c["famwin_u"] = float(np.mean([prim[w]["family"] == c["family"] for w in wu]))
        c["famwin_w"] = float(np.mean([prim[w]["family"] == c["family"] for w in ww]))
        l = loc[:, j]
        tooth = np.abs(l - c["P"]) <= 1.5 * c["dP"]
        c["same_tooth"] = float(np.mean(tooth))
        c["sigma_tooth"] = float(np.std(l[tooth], ddof=1)) if tooth.sum() > 2 else None
        c["sigma_all"] = float(np.std(l, ddof=1))
    best_prim = np.max(bu, axis=1)
    for j, c in enumerate(ext):
        c["frac_above_best_primary"] = float(np.mean(be[:, j] > best_prim))
    out = {"asteroid": num, "baseline_days": base, "n": int(len(t)), "primary": prim, "extended": ext,
           "n_boot": N_BOOT}
    json.dump(out, open(os.path.join(OUT_DIR, f"{num}_rebootstrap_legacy.json"), "w", encoding="utf-8"), indent=1)
    best = max(prim, key=lambda c: c["famwin_u"])
    return f"[{num}] {len(prim)} primary, {len(ext)} extended candidates; best family P={best['P']:.4f} famwin_u={best['famwin_u']:.2f}"


def main(argv):
    if "--out" not in argv:
        raise SystemExit("--out DIR is required (the legacy results must not overwrite results/<N>/)")
    k = argv.index("--out")
    out_dir = os.path.abspath(argv[k + 1])
    argv = argv[:k] + argv[k + 2:]
    if os.path.commonpath([out_dir, RES]) == RES:
        raise SystemExit("--out must not be inside results/")
    os.makedirs(out_dir, exist_ok=True)
    workers = 8
    if "--workers" in argv:
        k = argv.index("--workers")
        workers = int(argv[k + 1])
        argv = argv[:k] + argv[k + 2:]
    jobs = []
    for a in argv:
        if ":" in a:
            n, ps = a.split(":")
            jobs.append((int(n), [float(x) for x in ps.split(",") if x], out_dir))
        else:
            jobs.append((int(a), [], out_dir))
    if not jobs:
        jobs = [(int(d), [], out_dir) for d in sorted(os.listdir(RES)) if d.isdigit()
                and os.path.exists(os.path.join(RES, d, f"{d}_arrays.npz"))]
    if workers <= 1 or len(jobs) == 1:
        for j in jobs:
            print(run(j), flush=True)
        return
    with mp.get_context("spawn").Pool(workers) as pool:
        for msg in pool.imap_unordered(run, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
