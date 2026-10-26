"""
Reliability tests for the long adopted periods (P > 100 h): can the MPC sampling establish them at all?

For every asteroid and candidate period (adopted P, P/2 and the period DAMIT favours):
 1 coverage   per apparition (gap > 60 d): span, nights, rotation cycles covered (span / P), and the fraction of 20
              rotational phase bins that contain at least one night
 2 apparition single-apparition Lomb-Scargle (nterms = 2, exact, 24 h to 2 x span): best period and power at the
              candidate; a single apparition cannot phase-link anything, so a long period seen only in the combined
              data rests entirely on the coherence across the seasonal gaps
 3 injection  synthetic double-peaked light curves (P_inj = 150-1500 h, amplitude = the asteroid's per-apparition
              amplitude, random phase) added to the asteroid's own noise (nightly levels permuted within each dataset
              and apparition, deviations within nights permuted); recovered if the strongest peak in 24-5000 h lies
              within 3 % of P_inj, P_inj/2 or 2 P_inj; 40 trials per P_inj
 4 trend null each dataset-apparition block replaced by a quadratic trend in time fitted to its data plus permuted
              residual noise (no rotation at all); 100 realisations; distribution of the highest power in 300-5000 h
              compared with the observed one
 5 damping    tumbling damping time tau = P^3 / (C^3 D^2) Gyr (Pravec et al. 2005; C = 17, P in h, D in km), D from
              the JPL SBDB; tau far above the age of the Solar System means a non-principal-axis (tumbling) state is
              expected, for which no single period exists
Output: longperiod_tests.json
Usage: python longperiod_tests.py [--workers 9]
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import ls_longperiod as L  # noqa: E402

TARGETS = {"3216": [632.0], "3173": [1191.0], "1944": [922.9, 929.7], "3459": [537.5], "4371": [226.7],
           "4129": [288.8], "4331": [146.8], "4478": [64.86], "4482": [105.36]}
P_INJ = [150.0, 300.0, 450.0, 600.0, 900.0, 1200.0, 1500.0]
N_INJ, N_NULL = 40, 100


def fast_power(t, y, f):
    return L.ls_power(t, y, f, exact=False)


def recovered(P_best, P_true, tol=0.03):
    return any(abs(P_best - r * P_true) <= tol * r * P_true for r in (1.0, 0.5, 2.0))


def diameter(num):
    try:
        O = json.load(open(os.path.join(ROOT, "data", "sbdb_phys_orbit.json"), encoding="utf-8"))[num]
        for p in O.get("phys_par", []):
            if p.get("name") == "diameter" and p.get("value"):
                return float(p["value"])
    except Exception:  # noqa: BLE001
        pass
    return None


def run(num):
    ad = json.load(open(os.path.join(ROOT, "adopted.json"), encoding="utf-8"))[num]
    fig = json.load(open(os.path.join(ROOT, "figure_info.json"), encoding="utf-8"))
    amp = float(fig["lc_amplitudes_all"][num]["amp"])
    a = np.load(os.path.join(ROOT, "results", num, f"{num}_arrays.npz"), allow_pickle=True)
    t, y, sheet, t0 = a["t"].astype(float), a["y"].astype(float), a["sheet"].astype(str), float(a["time_zero"])
    rng = np.random.default_rng(int(num) + 7)
    app = L.apparition_index(t)
    lon = L.station_lon(sheet)
    night = np.floor(t + t0 + lon / 360.0).astype(int)
    base = float(t.max() - t.min())
    fL = np.arange(24.0 / 5000.0, 1.0, 1.0 / (20 * base))
    cands = sorted({ad["P"], ad["P"] / 2.0, *TARGETS[num]})
    out = {"num": num, "adopted_P": ad["P"], "quality": ad["quality"], "amp": amp, "candidates": cands}
    # 1 coverage + 2 single apparitions
    apps = []
    for k in np.unique(app):
        m = app == k
        span_h = float(np.ptp(t[m]) * 24.0)
        nights = np.unique(night[m])
        tn = np.array([t[m][night[m] == n_].mean() for n_ in nights])
        row = {"k": int(k), "n": int(m.sum()), "nights": int(len(nights)), "span_d": span_h / 24.0, "cand": {}}
        for P in cands:
            ph = ((tn * 24.0 / P) % 1.0 * 20).astype(int)
            row["cand"][f"{P:g}"] = {"cycles": span_h / P, "phase_cov": len(np.unique(ph)) / 20.0}
        if m.sum() >= 100 and span_h / 24.0 >= 30:
            fa = np.arange(1.0 / (2 * np.ptp(t[m])), 1.0, 1.0 / (20 * np.ptp(t[m])))
            fa = fa[fa >= 24.0 / 5000.0]
            pa = L.ls_power(t[m], y[m], fa)
            row["best_P"] = float(24.0 / fa[int(np.argmax(pa))])
            row["best_power"] = float(pa.max())
            for P in cands:
                w = np.abs(fa - 24.0 / P) <= 1.0 / np.ptp(t[m])
                row["cand"][f"{P:g}"]["power"] = float(pa[w].max()) if w.any() else None
        apps.append(row)
    out["apparitions"] = apps
    # 3 injection-recovery
    def noise():
        return L.permute_nights(rng, t, y, sheet, app, t0, lon) - np.mean(y)
    inj = {}
    for P in P_INJ + [ad["P"]]:
        hits, Pb = 0, []
        for _ in range(N_INJ):
            ph = (t * 24.0 / P) % 1.0
            th1, th2 = rng.uniform(0, 2 * np.pi, 2)
            s = 0.5 * amp * (0.85 * np.cos(4 * np.pi * ph + th2) + 0.25 * np.cos(2 * np.pi * ph + th1)) / 0.85
            p = fast_power(t, noise() + s, fL)
            best = float(24.0 / fL[int(np.argmax(p))])
            Pb.append(best)
            hits += recovered(best, P)
        inj[f"{P:g}"] = {"recovered_frac": hits / N_INJ, "P_best_median": float(np.median(Pb))}
    out["injection"] = inj
    # 4 trend null (quadratic per dataset-apparition block + permuted residual noise)
    trend = np.empty_like(y)
    for s in np.unique(sheet):
        for k in np.unique(app[sheet == s]):
            m = (sheet == s) & (app == k)
            tt = t[m] - t[m].mean()
            deg = 2 if np.ptp(tt) > 10 and m.sum() > 20 else (1 if m.sum() > 5 else 0)
            c = np.polyfit(tt, y[m], deg) if deg else [y[m].mean()]
            trend[m] = np.polyval(c, tt)
    resid = y - trend
    sel = 24.0 / fL >= 300.0
    obs = fast_power(t, y, fL)
    obs_max = float(obs[sel].max())
    obs_P = float(24.0 / fL[sel][int(np.argmax(obs[sel]))])
    tr_only = fast_power(t, trend, fL)
    null_max, null_P = [], []
    for _ in range(N_NULL):
        yn = trend + L.permute_nights(rng, t, resid, sheet, app, t0, lon)
        p = fast_power(t, yn, fL)
        null_max.append(float(p[sel].max()))
        null_P.append(float(24.0 / fL[sel][int(np.argmax(p[sel]))]))
    out["trend_null"] = {"observed_max_power_300_5000": obs_max, "observed_P": obs_P,
                         "trend_only_max_power": float(tr_only[sel].max()),
                         "trend_only_P": float(24.0 / fL[sel][int(np.argmax(tr_only[sel]))]),
                         "null_max_median": float(np.median(null_max)), "null_max_95": float(np.percentile(null_max, 95)),
                         "frac_null_ge_observed": float(np.mean(np.array(null_max) >= obs_max)),
                         "null_P_samples": [round(x, 1) for x in null_P[:30]]}
    # 5 damping time
    D = diameter(num)
    out["diameter_km"] = D
    out["damping_Gyr"] = {f"{P:g}": (P ** 3 / (17.0 ** 3 * D ** 2) if D else None) for P in cands}
    return num, out


def main(argv):
    workers = int(argv[argv.index("--workers") + 1]) if "--workers" in argv else 9
    with mp.get_context("spawn").Pool(workers) as pool:
        res = dict(pool.imap_unordered(run, list(TARGETS)))
    json.dump(res, open(os.path.join(ROOT, "longperiod_tests.json"), "w"), indent=1)
    for n in TARGETS:
        r = res[n]
        print(f"({n}) adopted {r['adopted_P']:g} h {r['quality']}, amp {r['amp']:.2f}, D {r['diameter_km']} km")
        for P in r["candidates"]:
            cyc = [a["cand"][f"{P:g}"]["cycles"] for a in r["apparitions"]]
            cov = [a["cand"][f"{P:g}"]["phase_cov"] for a in r["apparitions"]]
            print(f"   P {P:8.1f}: cycles per apparition median {np.median(cyc):.1f} max {max(cyc):.1f}; "
                  f"phase coverage median {np.median(cov):.2f}; damping {r['damping_Gyr'][f'{P:g}']}")
        print("   injection recovered:", {k: v["recovered_frac"] for k, v in r["injection"].items()})
        tn = r["trend_null"]
        print(f"   trend null: observed max {tn['observed_max_power_300_5000']:.3f} at {tn['observed_P']:.0f} h; "
              f"trend only {tn['trend_only_max_power']:.3f} at {tn['trend_only_P']:.0f} h; null median "
              f"{tn['null_max_median']:.3f}, 95% {tn['null_max_95']:.3f}; P(null >= obs) {tn['frac_null_ge_observed']:.2f}")
        best = [(a["k"], round(a.get("best_P", 0)), round(a.get("best_power", 0), 2)) for a in r["apparitions"] if "best_P" in a]
        print("   single-apparition best periods:", best)


if __name__ == "__main__":
    main(sys.argv[1:])
