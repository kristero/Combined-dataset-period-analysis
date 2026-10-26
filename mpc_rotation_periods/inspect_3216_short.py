"""
(3216) Harrington: periods below about 400 h, where the spectral window of the sampling is weak.
For the strongest peak families in 24-400 h of the combined periodogram (nterms = 2, exact chi2; original and
geometry-corrected data): power, noise-weighted power, spectral-window power at the period, relation to window peaks
(+- day aliases) and to harmonics of the 632/1267 h signal, permutation significance (nightly levels permuted within each
dataset and apparition, maximum over 24-400 h), independent halves and single apparitions; plus the convex-inversion
chi2 landscape of damit_longrange.py in 120-400 h.
Output: inspect_3216_short.json, review/inspect_3216_short.png
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import ls_longperiod as L  # noqa: E402

num = "3216"
P_LONG = (631.5, 1266.55)
out = {}
rng = np.random.default_rng(3216)
for label, path in (("original", f"results/{num}/{num}_arrays.npz"), ("geometry_corrected", f"results_geo/{num}/{num}_arrays.npz")):
    a = np.load(os.path.join(ROOT, path), allow_pickle=True)
    t, y, sh, t0 = a["t"].astype(float), a["y"].astype(float), a["sheet"].astype(str), float(a["time_zero"])
    T = float(np.ptp(t))
    sd = {s: np.std(y[sh == s], ddof=1) for s in np.unique(sh)}
    dy = np.array([sd[s] for s in sh])
    f = np.arange(24.0 / 400.0, 1.0, 1.0 / (20 * T))
    p = L.ls_power(t, y, f)
    pw = L.ls_power(t, y, f, dy=dy)
    W = L.window(t, f)
    from scipy.signal import find_peaks
    kw, _ = find_peaks(W)
    wpk = sorted([(W[i], f[i]) for i in kw], reverse=True)[:25]
    fams = L.peak_families(f, p, n=10)
    app = L.apparition_index(t)
    lon = L.station_lon(sh)
    pmax_perm = np.array([L.ls_power(t, L.permute_nights(rng, t, y, sh, app, t0, lon), f, exact=False).max()
                          for _ in range(100)])
    pfast = L.ls_power(t, y, f, exact=False)
    halves = {"odd": app % 2 == 1, "even": app % 2 == 0,
              "atlas": np.isin([s[:3] for s in sh], L.ATLAS), "not_atlas": ~np.isin([s[:3] for s in sh], L.ATLAS)}
    half_best = {}
    for h, m in halves.items():
        ph = L.ls_power(t[m], y[m], f)
        half_best[h] = {"n": int(m.sum()), "P_best": float(24 / f[np.argmax(ph)]), "power": float(ph.max())}
    rows = []
    for F in fams:
        P = F["P"]
        f0 = 24.0 / P
        g = f0 + np.arange(-300, 301) / 100.0 / T
        pg = L.ls_power(t, y, g)
        k = int(np.argmax(pg))
        fr = g[k]
        Pr = 24.0 / fr
        iW = int(np.argmin(np.abs(f - fr)))
        near = [(round(24 / fw, 1), d, round(wv, 3)) for wv, fw in wpk for d in (-2, -1, 0, 1, 2)
                if fw + d > 0 and abs(fr - (fw + d)) <= 1.0 / T]
        harm = [(Pl, kk) for Pl in P_LONG for kk in range(2, 9) if abs(Pl / kk - Pr) <= 0.01 * Pr]
        hv = {}
        for h, m in halves.items():
            ph = L.ls_power(t[m], y[m], g)
            hv[h] = round(float(ph.max()), 3)
        apps = []
        for kk in np.unique(app):
            m = app == kk
            if m.sum() < 100 or np.ptp(t[m]) < 30:
                continue
            ga = np.arange(24.0 / min(400.0, 2 * np.ptp(t[m]) * 24), 1.0, 1.0 / (20 * np.ptp(t[m])))
            pa = L.ls_power(t[m], y[m], ga)
            w = np.abs(ga - fr) <= 1.0 / np.ptp(t[m])
            apps.append({"k": int(kk), "best_P": round(float(24 / ga[np.argmax(pa)]), 1), "best": round(float(pa.max()), 3),
                         "at_P": round(float(pa[w].max()), 3) if w.any() else None})
        rows.append({"P": float(Pr), "power": float(pg.max()), "power_weighted": float(L.ls_power(t, y, np.array([fr]), dy=dy)[0]),
                     "W_at_P": float(W[iW]), "near_window_peaks": near, "harmonic_of_long": harm,
                     "p_value": float(np.mean(pmax_perm >= pfast[np.argmin(np.abs(f - fr))])), "halves": hv,
                     "apparitions": apps, "members": [round(x, 2) for x in F["members"][:5]]})
    out[label] = {"n": int(len(t)), "T_d": T, "perm_max_median": float(np.median(pmax_perm)),
                  "perm_max_95": float(np.percentile(pmax_perm, 95)), "half_best_24_400": half_best, "peaks": rows,
                  "curve": {"P": (24 / f[::3]).tolist(), "p": p[::3].tolist(), "pw": pw[::3].tolist(), "W": W[::3].tolist()}}

# convex inversion landscape (damit_longrange.py)
lr = {}
for name in ("all", "odd", "even"):
    d = os.path.join(ROOT, "damit_runs", f"paper_{num}", "longrange", name)
    rows = []
    for k in range(24):
        fpath = os.path.join(d, f"chunk_{k}", "out_periods")
        if os.path.exists(fpath):
            rows += [ln.split() for ln in open(fpath) if len(ln.split()) == 5]
    a = np.array(rows, float)
    a = a[np.isfinite(a[:, 1])]
    a = a[np.argsort(a[:, 0])]
    gbest = a[:, 1].min()
    sel = (a[:, 0] >= 120) & (a[:, 0] <= 400)
    b = a[sel]
    mins = []
    for j in np.argsort(b[:, 1]):
        if all(abs(b[j, 0] - m) > 0.03 * m for m, _ in mins):
            mins.append((float(b[j, 0]), float((b[j, 1] / gbest) ** 2)))
        if len(mins) >= 6:
            break
    lr[name] = {"global_best_P": float(a[np.argmin(a[:, 1]), 0]), "minima_120_400": mins,
                "chi2_ratio_median_120_400": float(np.median((b[:, 1] / gbest) ** 2)),
                "curve": {"P": a[:, 0].tolist(), "chi2_ratio": ((a[:, 1] / gbest) ** 2).tolist()}}
out["damit_longrange"] = lr
json.dump({k: ({kk: vv for kk, vv in v.items() if kk != "curve"} if isinstance(v, dict) else v) for k, v in out.items()},
          open(os.path.join(ROOT, "inspect_3216_short.json"), "w"), indent=1, default=float)

# figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
c = out["original"]["curve"]
fig, ax = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True, gridspec_kw={"height_ratios": [1.6, 0.8, 1.0]})
ax[0].plot(c["P"], c["p"], color="#1a5276", lw=0.7, label="unweighted")
ax[0].plot(c["P"], out["geometry_corrected"]["curve"]["p"], color="#b9770e", lw=0.6, alpha=0.8, label="after aspect correction")
ax[0].axhline(out["original"]["perm_max_95"], color="0.4", ls="--", lw=1, label="95 % of permuted maxima (24-400 h)")
for r in out["original"]["peaks"][:6]:
    ax[0].annotate(f"{r['P']:.1f}", (r["P"], r["power"]), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)
ax[0].set_ylabel("LS power")
ax[0].legend(fontsize=8, frameon=False)
ax[0].set_title("(3216) Harrington: periods 24-400 h", loc="left")
ax[1].plot(c["P"], c["W"], color="0.3", lw=0.7)
ax[1].set_ylabel("spectral window")
cl = lr["all"]["curve"]
for name, col in (("all", "#1a5276"), ("odd", "#7d3c98"), ("even", "#b9770e")):
    cc = lr[name]["curve"]
    ax[2].plot(cc["P"], cc["chi2_ratio"], ".", ms=2, color=col, label=f"convex inversion, {name}")
ax[2].set_ylim(0.98, 1.6)
ax[2].set_ylabel("chi2 / chi2(best)")
ax[2].legend(fontsize=8, frameon=False)
ax[2].set_xscale("log")
ax[2].set_xlim(24, 400)
ax[2].set_xlabel("period (h)")
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "review", "inspect_3216_short.png"), dpi=130)
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("curve", "peaks")} for k, v in out.items() if k != "damit_longrange"}, indent=1, default=float))
for lab in ("original", "geometry_corrected"):
    print("=====", lab)
    for r in out[lab]["peaks"]:
        print(f"P {r['P']:8.2f} pow {r['power']:.3f} w {r['power_weighted']:.3f} W {r['W_at_P']:.3f} p {r['p_value']:.2f} "
              f"harm {r['harmonic_of_long']} win {r['near_window_peaks'][:2]} halves {r['halves']}")
        print("        apparitions:", [(x['best_P'], x['at_P']) for x in r["apparitions"]])
for name, v in lr.items():
    print("DAMIT", name, "global best", round(v["global_best_P"], 2), "minima 120-400 h (P, chi2 ratio):",
          [(round(a_, 1), round(b_, 3)) for a_, b_ in v["minima_120_400"]], "median ratio", round(v["chi2_ratio_median_120_400"], 3))
