"""
Second version of the candidate bootstrap and subset checks (replaces rebootstrap.py / the subset part of
mpc_redo.py), fixing three problems found in the verification:
  * shared peaks: a candidate "wins" a resampling if the highest power within its OWN comb tooth is the largest
    of all candidates; the own window is the part of the frequency axis within +/-1.5 resolution elements of the
    candidate that is closer to it than to any other candidate (primary or extended), so that no frequency
    belongs to two candidates. The peak position (period scatter) is still measured within +/-20 resolution
    elements;
  * night blocks are local nights of each station (floor(JD + east longitude/360)), not integer-JD days;
  * all subset periodograms (longitude groups, single datasets, leave-one-dataset-out, TESS) use the exact chi2
    method; astropy's approximate fastchi2 is inaccurate for small datasets (powers > 1). The full-data bootstrap
    keeps fastchi2, whose power differs from the exact value by < 0.01 with identical peak positions.
Candidates: all candidates of rebootstrap.py (primary and extended) plus the adopted period and the alternatives
of adopted.json between 0.8 and 1500 h; candidates above 240 h are bootstrapped separately against the best
primary candidate.
(rebootstrap.py has since been corrected for its search windows, with disjoint own windows: within 1.5
resolution elements and closer to the candidate than to any other. It still uses integer-JD night blocks
(floor(JD), split at 12:00 UT) and has no subset checks. Its candidate periods, used here as seeds, are identical
before and after the correction; see check_rebootstrap_fix.py.)
Equal grids: every compared bootstrap power (unweighted, weighted, extended range) and the full-data weighted
power is evaluated with fastchi2 on the candidate's +/-20 resolution-element grid and then restricted to its own
window. The fastchi2 approximation error depends on the grid length and grows along the grid, so powers from grids
of different lengths or positions must not be compared. (The first version of this script evaluated the
extended-range powers and the weighted powers on 151-point grids and compared the extended ones with primary
powers from the 2001-point grid, which shifted frac_above_best_primary by up to about 0.02; its outputs are kept
as results/<N>/<N>_rebootstrap2_v1.json.)
Output: results/<N>/<N>_rebootstrap2.json
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
OBS = json.load(open(os.path.join(ROOT, "data", "mpc_obscodes.json"), encoding="utf-8"))
GROUPS = {
    "americas_hawaii": ["T05", "T08", "T07", "F51", "F52", "703", "G96", "691", "I41", "689", "704", "H45", "G45",
                        "W68", "Y00", "V00", "644", "E10", "807", "809"],
    "europe_africa": ["M22", "R17", "Z22", "069", "J43", "A50", "L01", "K91", "J04"],
    "asia_australia": ["D29", "P07", "Q62", "E12", "N50", "O75", "O18", "Q63", "Q64", "D35"],
    "space": ["C57"],
}
GROUP_OF = {s: g for g, lst in GROUPS.items() for s in lst}


def power(t, y, f, dy=None, exact=False):
    from astropy.timeseries import LombScargle
    ls = LombScargle(t, y, dy=dy, nterms=2)
    if exact:
        return ls.power(f, method="chi2")
    return ls.power(f, method="fastchi2", assume_regular_frequency=True)


def grid(f0, base, hw_res, spp=SPP):
    df = 1.0 / (spp * base)
    hw = min(hw_res / base, 0.5 * f0)
    return (f0 - hw) + df * np.arange(int(np.ceil(2 * hw / df)) + 1)


def write_json(path, obj):
    """Write atomically and fail loudly. Passing an unclosed file object to json.dump leaves the final flush to the
    garbage collector, which ignores errors: on 2026-09-29 four bootstrap result files were left empty that way."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def own_window(g, j, freqs, base, hw_res=1.5):
    """Mask of the points of grid g that belong to candidate j: within hw_res resolution elements of its frequency
    and strictly closer to it than to every other candidate frequency. In one dimension this is a contiguous
    interval, so the masked grid stays regular (fastchi2 with assume_regular_frequency=True)."""
    d = np.abs(g - freqs[j])
    m = d <= hw_res / base * (1.0 + 1e-9)
    for k, f in enumerate(freqs):
        if k != j:
            m &= d < np.abs(g - f)
    if m.sum() < 2:
        raise ValueError(f"own window of candidate {j} (f={freqs[j]:.8f}) has {int(m.sum())} grid points")
    return m


def families(Ps, dPs):
    fam = list(range(len(Ps)))
    for a in range(len(Ps)):
        for b in range(a):
            if abs(2 * Ps[a] - Ps[b]) <= 3 * dPs[b] or abs(Ps[a] - 2 * Ps[b]) <= 3 * dPs[a]:
                fam[a] = fam[b]
                break
    return fam


def run(num):
    ad = json.load(open(os.path.join(ROOT, "adopted.json"), encoding="utf-8"))[str(num)]
    rb = json.load(open(os.path.join(RES, str(num), f"{num}_rebootstrap.json"), encoding="utf-8"))
    a = np.load(os.path.join(RES, str(num), f"{num}_arrays.npz"))
    t, y, sheet = a["t"], a["y"], a["sheet"]
    t0 = float(a["time_zero"])
    base = float(t.max() - t.min())
    sd = {ds: np.std(y[sheet == ds], ddof=1) for ds in np.unique(sheet)}
    dy = np.array([sd[s] for s in sheet])
    seeds = [c["P"] for c in rb["primary"]] + [c["P"] for c in rb["extended"]] + [ad["P"]] + \
        [p for p in ad.get("alternatives", []) if 0.8 <= p <= 1500]
    cands = []
    for P in seeds:
        g = grid(24.0 / P, base, 3.0)
        p = power(t, y, g)
        k = int(np.argmax(p))
        Pk, dP = 24.0 / g[k], (24.0 / g[k]) ** 2 / 24.0 / base
        if any(abs(Pk - c["P"]) <= 1.5 * c["dP"] for c in cands):
            continue
        cands.append({"P": float(Pk), "f": float(g[k]), "power": float(p[k]), "dP": float(dP)})
    # wide (+/-20) grid and exclusive own window of every candidate; all compared powers come from the wide grid
    freqs = [c["f"] for c in cands]
    wide_of, own_of = [], []
    for j, c in enumerate(cands):
        g = grid(c["f"], base, 20.0)
        wide_of.append(g)
        own_of.append(own_window(g, j, freqs, base))
    prim = [c for c in cands if c["P"] <= P_PRIMARY]
    ext = [c for c in cands if c["P"] > P_PRIMARY]
    iprim = [j for j, c in enumerate(cands) if c["P"] <= P_PRIMARY]
    iext = [j for j, c in enumerate(cands) if c["P"] > P_PRIMARY]
    fam = families([c["P"] for c in prim], [c["dP"] for c in prim])
    for c, fid, j in zip(prim, fam, iprim):
        c["family"] = int(fid)
        c["power_weighted"] = float(power(t, y, wide_of[j], dy=dy)[own_of[j]].max())
    # local-night blocks
    lon = np.array([(OBS.get(s[:3], {}).get("lon") or 0.0) for s in sheet], dtype=float)
    night = np.array([f"{s}|{int(np.floor(x + t0 + L / 360.0))}" for s, x, L in zip(sheet, t, lon)])
    _, inv = np.unique(night, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    bnd = np.searchsorted(inv[order], np.arange(inv.max() + 2))
    groups = [order[bnd[i]:bnd[i + 1]] for i in range(inv.max() + 1)]
    rng = np.random.default_rng(2000 + num)
    wide = [wide_of[j] for j in iprim]
    own = [own_of[j] for j in iprim]
    ext_wo = [(wide_of[j], own_of[j]) for j in iext]
    bu = np.zeros((N_BOOT, len(prim)))
    bw = np.zeros((N_BOOT, len(prim)))
    loc = np.zeros((N_BOOT, len(prim)))
    be = np.zeros((N_BOOT, len(ext)))
    for b in range(N_BOOT):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        tb, yb, db = t[idx], y[idx], dy[idx]
        for j, (g, m) in enumerate(zip(wide, own)):
            p = power(tb, yb, g)
            loc[b, j] = 24.0 / g[int(np.argmax(p))]
            bu[b, j] = float(p[m].max())
            bw[b, j] = float(power(tb, yb, g, dy=db)[m].max())
        for j, (g, m) in enumerate(ext_wo):
            be[b, j] = float(power(tb, yb, g)[m].max())
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
    # exact-chi2 subset checks
    stn = np.array([s[:3] for s in sheet])
    grp = np.array([GROUP_OF.get(s, "other") for s in stn])
    subsets = {}
    for gname in list(GROUPS) + ["other"]:
        if np.sum(grp == gname) >= 150:
            subsets[f"group:{gname}"] = grp == gname
    for ds in np.unique(sheet):
        m = sheet == ds
        if m.sum() >= 100:
            subsets[f"dataset:{ds}"] = m
        subsets[f"without:{ds}"] = sheet != ds
    allc = prim + ext
    sub_res = {}
    for name, m in subsets.items():
        tb, yb = t[m], y[m]
        bb = max(float(tb.max() - tb.min()), 1.0)
        vals = [float(power(tb, yb, grid(c["f"], bb, 3.0, spp=20), exact=True).max()) for c in allc]
        o = np.argsort(vals)[::-1]
        sub_res[name] = {"n": int(m.sum()), "baseline_days": bb, "power": vals, "preferred_P": allc[int(o[0])]["P"],
                         "preferred_primary_P": max(prim, key=lambda c: vals[allc.index(c)])["P"],
                         "ratio_second_to_first": float(vals[o[1]] / vals[o[0]]) if len(vals) > 1 and vals[o[0]] > 0 else None}
    for j, c in enumerate(cands):
        ow = wide_of[j][own_of[j]]
        c["own_window_P"] = [float(24.0 / ow.max()), float(24.0 / ow.min())]
    out = {"asteroid": num, "baseline_days": base, "n": int(len(t)), "primary": prim, "extended": ext,
           "subsets": sub_res, "n_boot": N_BOOT, "night_blocks": int(len(groups)),
           "windows": {"own_half_width_res": 1.5, "own_windows_disjoint": True, "position_half_width_res": 20.0,
                       "powers_from": "wide grid, masked"}}
    write_json(os.path.join(RES, str(num), f"{num}_rebootstrap2.json"), out)
    return f"[{num}] {len(prim)} primary, {len(ext)} extended, {len(subsets)} subsets"


def main(argv):
    workers = 16
    if "--workers" in argv:
        k = argv.index("--workers")
        workers = int(argv[k + 1])
        argv = argv[:k] + argv[k + 2:]
    nums = [int(x) for x in argv] or sorted(int(k) for k in json.load(open(os.path.join(ROOT, "adopted.json"))))
    with mp.get_context("spawn").Pool(workers) as pool:
        for msg in pool.imap_unordered(run, nums):
            print(msg, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
