"""
Fine-grid Lomb-Scargle re-analysis of the asteroids with adopted periods longer than one day (P > 24 h).

For every asteroid (the data of the combined analysis, results/<N>/; for asteroids with a significant aspect model
also the geometry-corrected data of results_geo/<N>/):
 1 periodogram (nterms = 2, unweighted and noise-weighted as in rebootstrap2) from 2 to 5000 h:
   - 24-5000 h: exact chi2 power on a grid of 20 points per resolution element 1/T
   - 2-24 h: fastchi2, 10 points per resolution element (only to see whether a short period competes)
   the ten strongest peak families (peaks within 1 % or in a 2:1 / 1:2 ratio grouped) are refined on a local exact
   grid of 200 points per resolution element
 2 spectral window W(f) = |sum exp(2 pi i f t)|^2 / N^2 of the sampling on the long-period grid; for every candidate
   the strongest window peaks and whether the candidate lies within one resolution element of a window peak or of
   a window peak's alias with the day (f +- 1, 2 c/d) - such a candidate can be produced by the sampling alone
   (lunar month 29.53 d = 708.7 h, year, apparition length)
 3 significance: the nightly mean levels of every dataset are permuted among the nights of the same dataset and
   apparition (sampling, noise and night-to-night scatter kept, a coherent long period destroyed); 200 permutations;
   p-value = fraction of permutations whose highest power in 24-5000 h reaches the candidate's power
 4 independent halves: odd vs even apparitions, and ATLAS (T05, T08, M22, W68) vs all other stations; for each half
   the power at the candidate (local exact grid) and the half's own strongest period in 24-5000 h
Output: longperiod_ls.json, review/longperiod_ls_<N>.png
Usage: python ls_longperiod.py [--workers 6] [--nums a,b]
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
RES, RES_GEO = os.path.join(ROOT, "results"), os.path.join(ROOT, "results_geo")
LUNAR_D, YEAR_D = 29.530589, 365.25
ATLAS = ("T05", "T08", "M22", "W68")
N_PERM = 200
GAP_D = 60.0


def ls_power(t, y, f, dy=None, exact=True):
    from astropy.timeseries import LombScargle
    ls = LombScargle(t, y, dy=dy, nterms=2)
    if exact:
        return ls.power(f, method="chi2")
    return ls.power(f, method="fastchi2", assume_regular_frequency=True)


def apparition_index(t):
    o = np.argsort(t)
    lab = np.zeros(len(t), int)
    cuts = np.where(np.diff(t[o]) > GAP_D)[0]
    for k, g in enumerate(np.split(o, cuts + 1)):
        lab[g] = k
    return lab


def peak_families(f, p, n=10):
    from scipy.signal import find_peaks
    k, _ = find_peaks(p)
    k = k[np.argsort(p[k])[::-1]]
    fams = []
    for i in k:
        P = 24.0 / f[i]
        same = False
        for F in fams:
            for Q in F["members"]:
                if any(abs(P - r * Q) <= 0.01 * r * Q for r in (1.0, 2.0, 0.5)):
                    same = True
                    break
            if same:
                F["members"].append(P)
                break
        if not same:
            fams.append({"P": P, "power": float(p[i]), "members": [P]})
        if len(fams) >= n:
            break
    return fams


def window(t, f):
    w = np.empty(len(f))
    for s in range(0, len(f), 20000):
        ph = 2 * np.pi * np.outer(f[s:s + 20000], t)
        w[s:s + 20000] = (np.cos(ph).sum(1) ** 2 + np.sin(ph).sum(1) ** 2) / len(t) ** 2
    return w


def permute_nights(rng, t, y, sheet, app, t0, lon):
    """Permute the nightly mean levels of each dataset among its nights in the same apparition."""
    night = np.floor(t + t0 + lon / 360.0).astype(int)
    out = y.copy()
    for s in np.unique(sheet):
        ms = sheet == s
        for a in np.unique(app[ms]):
            m = np.where(ms & (app == a))[0]
            nights = night[m]
            u = np.unique(nights)
            if len(u) < 2:
                continue
            means = {k: y[m][nights == k].mean() for k in u}
            perm = dict(zip(u, rng.permutation([means[k] for k in u])))
            out[m] = y[m] - np.array([means[k] for k in nights]) + np.array([perm[k] for k in nights])
    return out


def station_lon(sheet):
    obs = json.load(open(os.path.join(ROOT, "data", "mpc_obscodes.json"), encoding="utf-8"))
    lon = {}
    for s in np.unique(sheet):
        code = s[:3]
        v = obs.get(code, {})
        lon[s] = float(v.get("lon", 0.0) or 0.0) if isinstance(v, dict) else 0.0
    return np.array([lon[s] for s in sheet])


def analyse(t, y, sheet, t0, adopted_P, label, rng):
    base = float(t.max() - t.min())
    sd = {ds: np.std(y[sheet == ds], ddof=1) for ds in np.unique(sheet)}
    dy = np.array([sd[s] for s in sheet])
    fL = np.arange(24.0 / 5000.0, 1.0, 1.0 / (20 * base))           # 24-5000 h
    pL = ls_power(t, y, fL)
    pLw = ls_power(t, y, fL, dy=dy)
    fS = np.arange(1.0, 12.0, 1.0 / (10 * base))                      # 2-24 h
    pS = ls_power(t, y, fS, exact=False)
    fams = peak_families(fL, pL, n=10)
    # make sure the adopted period and its harmonics are among the candidates
    if not any(any(abs(adopted_P - r * Q) <= 0.01 * r * Q for r in (1, 2, 0.5)) for F in fams for Q in F["members"]):
        fams.append({"P": adopted_P, "power": np.nan, "members": [adopted_P]})
    W = window(t, fL)
    from scipy.signal import find_peaks
    kw, _ = find_peaks(W)
    kw = kw[np.argsort(W[kw])[::-1]][:15]
    wpk = [{"P_h": float(24.0 / fL[i]), "f": float(fL[i]), "W": float(W[i])} for i in kw]
    app = apparition_index(t)
    lon = station_lon(sheet)
    # permutations: highest fastchi2 power in 24-5000 h
    pmax_perm = []
    for _ in range(N_PERM):
        yp = permute_nights(rng, t, y, sheet, app, t0, lon)
        pmax_perm.append(float(ls_power(t, yp, fL, exact=False).max()))
    pmax_perm = np.array(pmax_perm)
    obs_fast = ls_power(t, y, fL, exact=False)
    halves = {"odd_apparitions": app % 2 == 1, "even_apparitions": app % 2 == 0,
              "atlas": np.isin([s[:3] for s in sheet], ATLAS), "not_atlas": ~np.isin([s[:3] for s in sheet], ATLAS)}
    half_best = {}
    for h, m in halves.items():
        if m.sum() < 200:
            continue
        ph = ls_power(t[m], y[m], fL)
        half_best[h] = {"n": int(m.sum()), "P_best": float(24.0 / fL[int(np.argmax(ph))]), "power_best": float(ph.max())}
    cands = []
    for F in fams:
        P = F["P"]
        f0 = 24.0 / P
        g = f0 + (np.arange(-600, 601) / 200.0) / base                 # +-3 resolution elements, 200 per element
        g = g[g > 0]
        pg = ls_power(t, y, g)
        k = int(np.argmax(pg))
        Pr, fr = 24.0 / g[k], g[k]
        pw = float(ls_power(t, y, np.array([fr]), dy=dy)[0])
        i_fast = int(np.argmin(np.abs(fL - fr)))
        near_window = []
        for wp in wpk:
            for m_day in (0, 1, 2):
                for sgn in (1, -1):
                    fa = wp["f"] + sgn * m_day
                    if fa > 0 and abs(fr - fa) <= 1.0 / base:
                        near_window.append({"window_P_h": wp["P_h"], "day_alias": sgn * m_day, "W": wp["W"]})
        halves_at = {}
        for h, m in halves.items():
            if m.sum() < 200:
                continue
            ph = ls_power(t[m], y[m], g)
            halves_at[h] = {"power_at_P": float(ph.max()), "P_local": float(24.0 / g[int(np.argmax(ph))])}
        cands.append({"P": float(Pr), "P_res_h": float(Pr ** 2 / 24.0 / base), "power": float(pg.max()),
                      "power_weighted": pw, "p_value": float(np.mean(pmax_perm >= obs_fast[i_fast])),
                      "W_at_P": float(W[i_fast]), "near_window_peaks": near_window,
                      "P_over_lunar_month": float(Pr / (LUNAR_D * 24)), "halves": halves_at,
                      "family_members": [float(x) for x in F["members"][:6]],
                      "is_adopted_family": bool(any(abs(adopted_P - r * Pr) <= 0.01 * r * Pr for r in (1, 2, 0.5)))})
    k = int(np.argmax(pS))
    return {"label": label, "n": int(len(t)), "baseline_d": base, "candidates": cands, "window_peaks": wpk,
            "perm_pmax_median": float(np.median(pmax_perm)), "perm_pmax_95": float(np.percentile(pmax_perm, 95)),
            "short_best": {"P": float(24.0 / fS[k]), "power_fastchi2": float(pS[k])}, "halves_best": half_best,
            "curves": {"fL": fL[::4].tolist(), "pL": pL[::4].tolist(), "pLw": pLw[::4].tolist(), "W": W[::4].tolist()}}


def run(num):
    ad = json.load(open(os.path.join(ROOT, "adopted.json"), encoding="utf-8"))[num]
    rng = np.random.default_rng(int(num))
    out = {"num": num, "adopted_P": ad["P"], "quality": ad["quality"], "versions": []}
    for label, path in (("original", os.path.join(RES, num, f"{num}_arrays.npz")),
                        ("geometry_corrected", os.path.join(RES_GEO, num, f"{num}_arrays.npz"))):
        if not os.path.exists(path):
            continue
        a = np.load(path, allow_pickle=True)
        t, y, sheet, t0 = a["t"].astype(float), a["y"].astype(float), a["sheet"].astype(str), float(a["time_zero"])
        out["versions"].append(analyse(t, y, sheet, t0, ad["P"], label, rng))
    plot(out)
    for v in out["versions"]:
        v.pop("curves", None)
    return num, out


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    v = out["versions"][0]
    c = v["curves"]
    P = 24.0 / np.array(c["fL"])
    fig, ax = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    ax[0].plot(P, c["pL"], color="#1a5276", lw=0.7, label="unweighted")
    ax[0].plot(P, c["pLw"], color="#b9770e", lw=0.6, alpha=0.7, label="noise-weighted")
    ax[0].axhline(v["perm_pmax_95"], color="0.4", ls="--", lw=1, label="95 % of permuted maxima")
    ax[0].axvline(out["adopted_P"], color="#c0392b", lw=1, ls=":")
    ax[0].set_ylabel("power")
    ax[0].legend(fontsize=8, frameon=False, loc="upper left")
    ax[0].set_title(f"({out['num']}) adopted P = {out['adopted_P']:g} h, quality {out['quality']}", loc="left")
    ax[1].plot(P, c["W"], color="0.3", lw=0.7)
    for x, lab in ((LUNAR_D * 24, "lunar month"), (YEAR_D * 24 / 2, "half year")):
        ax[1].axvline(x, color="0.6", ls=":", lw=1)
        ax[1].text(x, 0.9 * max(c["W"]), lab, fontsize=8, color="0.4", rotation=90, va="top", ha="right")
    ax[1].set_ylabel("spectral window")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("period (h)")
    ax[1].set_xlim(24, 5000)
    fig.tight_layout()
    os.makedirs(os.path.join(ROOT, "review"), exist_ok=True)
    fig.savefig(os.path.join(ROOT, "review", f"longperiod_ls_{out['num']}.png"), dpi=120)
    plt.close(fig)


def main(argv):
    workers = int(argv[argv.index("--workers") + 1]) if "--workers" in argv else 6
    ad = json.load(open(os.path.join(ROOT, "adopted.json"), encoding="utf-8"))
    nums = sorted([n for n in ad if ad[n]["P"] > 24.0], key=lambda n: -ad[n]["P"])
    if "--nums" in argv:
        nums = argv[argv.index("--nums") + 1].split(",")
    with mp.get_context("spawn").Pool(workers) as pool:
        res = dict(pool.imap_unordered(run, nums))
    path = os.path.join(ROOT, "longperiod_ls.json")
    old = json.load(open(path)) if os.path.exists(path) and "--nums" in argv else {}
    old.update(res)
    json.dump(old, open(path, "w"), indent=1)
    for n in nums:
        r = res[n]
        v = r["versions"][0]
        print(f"({n}) adopted {r['adopted_P']:g} h {r['quality']}; permuted max power: median "
              f"{v['perm_pmax_median']:.3f}, 95% {v['perm_pmax_95']:.3f}; short best {v['short_best']['P']:.3f} h")
        for c in sorted(v["candidates"], key=lambda c: -c["power"])[:6]:
            nw = ",".join(f"{w['window_P_h']:.0f}h{w['day_alias']:+d}" for w in c["near_window_peaks"][:3])
            hv = " ".join(f"{h[:4]}={x['power_at_P']:.2f}" for h, x in c["halves"].items())
            print(f"   P {c['P']:10.3f} h  pow {c['power']:.3f} w {c['power_weighted']:.3f} p {c['p_value']:.3f} "
                  f"W {c['W_at_P']:.3f} {'ADOPTED ' if c['is_adopted_family'] else ''}{'window:' + nw if nw else ''} {hv}")


if __name__ == "__main__":
    main(sys.argv[1:])
