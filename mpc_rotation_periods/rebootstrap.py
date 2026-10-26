"""
Night-block bootstrap of the candidate periods, recomputed from the saved arrays of mpc_redo.py.

Differences from the bootstrap inside mpc_redo.py:
  * the competition is restricted to candidates in the primary range (P <= 240 h), as prescribed in
    the paper (longer periods are considered only when no satisfactory primary-range solution exists);
    candidates with P > 240 h are bootstrapped separately against the best primary candidate;
  * the bootstrap is done for the unweighted and for the noise-weighted periodogram;
  * extra candidate periods can be supplied on the command line (asteroid:P1,P2,...).

Search windows (corrected version). Every candidate owns the part of the frequency axis that lies within
TOOTH_RES resolution elements (1/baseline) of its frequency and closer to it than to any other candidate
(primary or extended), so that no frequency belongs to two candidates. The win competition, the weighted power
and the extended-range comparison use only these own windows. All compared powers are evaluated with fastchi2 on
the candidate's +/-WIDE_RES grid and then restricted to the own window, so that every candidate's window sits at
the same position of an equally long grid: the fastchi2 approximation error depends on the grid length and grows
along the grid, and powers from grids of different lengths or positions must not be compared.

The first version (rebootstrap_legacy.py) searched +/-20 resolution elements around every candidate. Neighbouring
comb teeth (3-40 resolution elements apart; 61 pairs in 30 of the 57 asteroids) then found the same peak, each on
its own grid. Because the fastchi2 error grows along the grid (relative +1e-3 at index 1000, +4e-3 at 1400), the
shared peak read highest on the grid of the lowest-frequency (longest-period) neighbour, which took the wins
whatever the list order; the weighted power was taken +/-3 resolution elements around that unweighted maximum
and inherited the same attribution. (There were no exact ties: the candidate grids lie on different lattices.)
The legacy results, on which the preliminary proposals and the review were based, are kept as
results/<N>/<N>_rebootstrap_legacy.json, proposals_legacy.json and evidence_legacy/.
The peak position is still followed within +/-WIDE_RES resolution elements, deliberately across neighbouring
teeth, so that sigma_all and same_tooth describe the scatter between comb teeth (cycle-count ambiguity); these
statistics are identical in both versions (same seeds, same resamplings, same grids).

Usage: python rebootstrap.py [N ...] [N:P1,P2 ...] [--workers 8]
Output: results/<N>/<N>_rebootstrap.json
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
TOOTH_RES = 1.5    # half-width of a candidate's own window (win competition), in resolution elements
WIDE_RES = 20.0    # half-width of the window in which the peak position is followed, in resolution elements
NEAR_TIE = 3e-3    # relative power difference below which two candidates count as tied in a resampling


def ls_power(t, y, freq, dy=None):
    from astropy.timeseries import LombScargle
    return LombScargle(t, y, dy=dy, nterms=2).power(freq, method="fastchi2", assume_regular_frequency=True)


def grid(f0, base, hw_res=WIDE_RES):
    df = 1.0 / (SPP * base)
    hw = min(hw_res / base, 0.5 * f0)
    return (f0 - hw) + df * np.arange(int(np.ceil(2 * hw / df)) + 1)


def write_json(path, obj):
    """Write atomically and fail loudly. Passing an unclosed file object to json.dump leaves the final flush to the
    garbage collector, which ignores errors: on 2026-09-29 four result files were left empty that way."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def own_window(g, j, freqs, base):
    """Mask of the points of grid g that belong to candidate j: within TOOTH_RES resolution elements of its
    frequency and strictly closer to it than to every other candidate frequency (a point exactly half-way
    between two candidates belongs to neither). In one dimension this is a contiguous interval, so the
    masked grid stays regular, as required by fastchi2 with assume_regular_frequency=True."""
    d = np.abs(g - freqs[j])
    m = d <= TOOTH_RES / base * (1.0 + 1e-9)
    for k, f in enumerate(freqs):
        if k != j:
            m &= d < np.abs(g - f)
    if m.sum() < 2:     # fastchi2 needs at least two regularly spaced frequencies
        raise ValueError(f"own window of candidate {j} (f={freqs[j]:.8f}) has {int(m.sum())} grid points")
    return m


def families(Ps, dPs):
    fam = list(range(len(Ps)))
    for a in range(len(Ps)):
        for b in range(a):
            if abs(2 * Ps[a] - Ps[b]) <= 3 * dPs[b] or abs(Ps[a] - 2 * Ps[b]) <= 3 * dPs[a] or abs(Ps[a] - Ps[b]) <= 3 * dPs[a]:
                fam[a] = fam[b]
                break
    return fam


def candidates(num, extra, t, y, base):
    """Refine every candidate of mpc_redo.py (plus extra periods) on the full data (unweighted), dropping
    candidates that fall within three resolution elements of an earlier one."""
    r = json.load(open(os.path.join(RES, str(num), f"{num}_result.json"), encoding="utf-8"))
    Ps = [c["P"] for c in r["candidates"]]
    for P in extra:
        if all(abs(P - q) > 0.002 * P for q in Ps):
            Ps.append(float(P))
    cands = []
    for P in Ps:
        g = grid(24.0 / P, base, 3.0)
        p = ls_power(t, y, g)
        k = int(np.argmax(p))
        Pk = 24.0 / g[k]
        if any(abs(Pk - c["P"]) <= 3 * c["dP"] for c in cands):
            continue
        cands.append({"P": float(Pk), "f": float(g[k]), "power": float(p[k]), "dP": float(Pk ** 2 / 24.0 / base)})
    return cands


def run(job):
    num, extra = job
    a = np.load(os.path.join(RES, str(num), f"{num}_arrays.npz"))
    t, y, sheet = a["t"], a["y"], a["sheet"]
    base = float(t.max() - t.min())
    sd = {ds: np.std(y[sheet == ds], ddof=1) for ds in np.unique(sheet)}
    dy = np.array([sd[s] for s in sheet])
    cands = candidates(num, extra, t, y, base)
    freqs = [c["f"] for c in cands]
    # own windows on the wide grids; the own window lies inside the wide window (same lattice, same centre).
    # Every compared power is computed on the wide grid and then masked (see the module docstring).
    for j, c in enumerate(cands):
        g = grid(c["f"], base)
        c["_grid"], c["_own"] = g, own_window(g, j, freqs, base)
        c["power_weighted"] = float(ls_power(t, y, g, dy=dy)[c["_own"]].max())
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
    bu = np.zeros((N_BOOT, len(prim)))
    bw = np.zeros((N_BOOT, len(prim)))
    loc = np.zeros((N_BOOT, len(prim)))
    be = np.zeros((N_BOOT, len(ext)))
    for b in range(N_BOOT):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        tb, yb, db = t[idx], y[idx], dy[idx]
        for j, c in enumerate(prim):
            g, m = c["_grid"], c["_own"]
            p = ls_power(tb, yb, g)
            loc[b, j] = 24.0 / g[int(np.argmax(p))]
            bu[b, j] = float(p[m].max())
            bw[b, j] = float(ls_power(tb, yb, g, dy=db)[m].max())
        for j, c in enumerate(ext):
            be[b, j] = float(ls_power(tb, yb, c["_grid"])[c["_own"]].max())
    wu, ww = np.argmax(bu, axis=1), np.argmax(bw, axis=1)

    def near_ties(bb, rel=NEAR_TIE):
        """Resamplings in which the two best candidates differ by less than rel (relative): below the fastchi2
        approximation error, so the winner of such a resampling is not decided by the data."""
        if bb.shape[1] < 2:
            return 0
        s = np.sort(bb, axis=1)
        return int(np.sum(s[:, -2] >= s[:, -1] * (1.0 - rel)))
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
    best_prim = np.max(bu, axis=1) if len(prim) else np.full(N_BOOT, -np.inf)
    for j, c in enumerate(ext):
        c["frac_above_best_primary"] = float(np.mean(be[:, j] > best_prim))
    for c in cands:
        own = c.pop("_grid")[c.pop("_own")]
        c["own_window_P"] = [float(24.0 / own.max()), float(24.0 / own.min())]
    out = {"asteroid": num, "baseline_days": base, "n": int(len(t)), "primary": prim, "extended": ext,
           "n_boot": N_BOOT, "windows": {"own_half_width_res": TOOTH_RES, "position_half_width_res": WIDE_RES,
                                         "own_windows_disjoint": True, "powers_from": "wide grid, masked"},
           "near_ties": {"relative": NEAR_TIE, "unweighted": near_ties(bu), "weighted": near_ties(bw)}}
    write_json(os.path.join(RES, str(num), f"{num}_rebootstrap.json"), out)
    best = max(prim, key=lambda c: c["famwin_u"])
    return f"[{num}] {len(prim)} primary, {len(ext)} extended candidates; best family P={best['P']:.4f} famwin_u={best['famwin_u']:.2f}"


def main(argv):
    workers = 8
    if "--workers" in argv:
        k = argv.index("--workers")
        workers = int(argv[k + 1])
        argv = argv[:k] + argv[k + 2:]
    jobs = []
    for a in argv:
        if ":" in a:
            n, ps = a.split(":")
            jobs.append((int(n), [float(x) for x in ps.split(",") if x]))
        else:
            jobs.append((int(a), []))
    if not jobs:
        jobs = [(int(d), []) for d in sorted(os.listdir(RES)) if d.isdigit()
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
