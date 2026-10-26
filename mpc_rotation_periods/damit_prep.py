"""
Input files for the DAMIT convex inversion (convexinv / period_scan, Kaasalainen & Torppa 2001; Kaasalainen et al.
2001; C code v0.2.1 by J. Durech, damit/version_0.2.1) from the same MPC measurements as the amplitude study.

Every measurement keeps the selection of the combined analysis (results/<N>/ or validation/results/). The
brightness is the distance-reduced magnitude H_red = m - 5 log10(r Delta) of the observation table (no phase
correction: convexinv fits its own phase function), converted to intensity. Each dataset (observatory + band) is
one relative lightcurve (flag 0), so per-dataset zero points are free, as in the ellipsoid fit.

The asteroid-centric Sun and Earth vectors are reconstructed from the Horizons quantities of every measurement
(r, Delta, alpha, phase-angle bisector) and the heliocentric position of the Earth (astropy built-in ephemeris,
ecliptic J2000): with u_h, u_g the heliocentric and geocentric unit vectors of the asteroid and b the bisector,
u_h,g = cos(alpha/2) b +/- sin(alpha/2) w, and r u_h - Delta u_g = E_earth fixes w = unit(E - (E.b) b).

Output: damit_runs/<sample>_<N>/lcs, prep.npz (per-point arrays), prep.json (diagnostics, scan interval).
Usage: python damit_prep.py [--only paper|val]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import amp_data  # noqa: E402

OUT = os.path.join(ROOT, "damit_runs")
EPS_J2000 = np.deg2rad(84381.448 / 3600.0)


def earth_helio_ecl(jd_utc):
    from astropy.coordinates import get_body_barycentric
    from astropy.time import Time
    t = Time(jd_utc, format="jd", scale="utc")
    e = get_body_barycentric("earth", t, ephemeris="builtin").xyz.to_value("au")
    s = get_body_barycentric("sun", t, ephemeris="builtin").xyz.to_value("au")
    x, y, z = e - s
    ce, se = np.cos(EPS_J2000), np.sin(EPS_J2000)
    return np.stack([x, ce * y + se * z, -se * y + ce * z], axis=1)


def unit_lonlat(lon, lat):
    lo, la = np.deg2rad(lon), np.deg2rad(lat)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=1)


def reconstruct(r, delta, alpha, pab_lon, pab_lat, E):
    b = unit_lonlat(pab_lon, pab_lat)
    eb = np.sum(E * b, axis=1)
    Ep = E - eb[:, None] * b
    w = Ep / np.linalg.norm(Ep, axis=1)[:, None]
    ha = np.deg2rad(alpha) / 2
    uh = np.cos(ha)[:, None] * b + np.sin(ha)[:, None] * w
    ug = np.cos(ha)[:, None] * b - np.sin(ha)[:, None] * w
    A = r[:, None] * uh                       # heliocentric asteroid
    G = delta[:, None] * ug                   # geocentric asteroid
    resid = np.linalg.norm(A - G - E, axis=1)  # closure error (AU)
    return -A, -G, resid                      # asteroid-centric Sun, Earth


def match_obs(t, sheet, t0, obs):
    idx = np.full(len(t), -1)
    for ds in np.unique(sheet):
        o = obs[obs["dataset"] == ds].sort_values("jd_lt")
        m = np.where(sheet == ds)[0]
        jd = t[m] + t0
        x = o["jd_lt"].to_numpy()
        k = np.clip(np.searchsorted(x, jd), 1, len(o) - 1)
        kk = np.where(np.abs(jd - x[k - 1]) <= np.abs(jd - x[k]), k - 1, k)
        ok = np.abs(x[kk] - jd) < 1e-4
        idx[m[ok]] = o.index.to_numpy()[kk[ok]]
    return idx


def scan_interval(P, span_h, r_mean, sigma_P):
    P_orb_h = r_mean ** 1.5 * 365.25 * 24
    half = 1.5 * P * P / P_orb_h + 3 * sigma_P
    dP = 0.5 * P * P / span_h
    return P - half, P + half, dP


def prepare(sample, num):
    num = str(num)
    d = amp_data.load(sample, num)
    if sample == "paper":
        a = np.load(os.path.join(ROOT, "results", num, f"{num}_arrays.npz"))
        obs = pd.read_csv(os.path.join(ROOT, "data", "obs_tables", f"{num}_obs.csv"))
        P0, sigma_P = amp_data.ADOPTED[num]["P"], amp_data.ADOPTED[num].get("sigma_P") or 0.0
    else:
        a = np.load(os.path.join(amp_data.VAL, "results", f"{num}_arrays.npz"), allow_pickle=True)
        obs = pd.read_csv(os.path.join(amp_data.VAL, "data", "obs_tables", f"{num}_obs.csv"))
        # centre on the period of the MPC data themselves (strongest peak within 0.5 % of the LCDB period): the
        # LCDB summary periods are rounded (e.g. 4.258 h for (3382), sidereal 4.2531 h)
        P0 = float(d["P"])
        sigma_P = 1.5e-5 * P0
    t, sheet, t0 = a["t"].astype(float), a["sheet"].astype(str), float(a["time_zero"])
    assert np.allclose(t, d["t"]) and np.array_equal(sheet, d["sheet"].astype(str))
    idx = match_obs(t, sheet, t0, obs)
    ok = idx >= 0
    o = obs.loc[idx[ok]]
    for c in ("r", "delta", "alpha", "pab_lon", "pab_lat", "H_red", "jd", "jd_lt"):
        assert np.all(np.isfinite(o[c].to_numpy(float))), c
    jd, jd_lt = o["jd"].to_numpy(float), o["jd_lt"].to_numpy(float)
    r, delta, alpha = o["r"].to_numpy(float), o["delta"].to_numpy(float), o["alpha"].to_numpy(float)
    E = earth_helio_ecl(jd)
    sun, earth, resid = reconstruct(r, delta, alpha, o["pab_lon"].to_numpy(float), o["pab_lat"].to_numpy(float), E)
    H = o["H_red"].to_numpy(float)
    y = d["y"][ok]
    sh = sheet[ok]
    # phase angle from the reconstructed vectors must equal Horizons alpha
    ca = np.sum(sun * earth, axis=1) / np.linalg.norm(sun, axis=1) / np.linalg.norm(earth, axis=1)
    dalpha = np.rad2deg(np.arccos(np.clip(ca, -1, 1))) - alpha
    order = np.lexsort((jd_lt, sh))
    run = os.path.join(OUT, f"{sample}_{num}")
    os.makedirs(run, exist_ok=True)
    sets = list(dict.fromkeys(sh[order]))
    lines = [str(len(sets))]
    for s in sets:
        m = order[sh[order] == s]
        lines.append(f"{len(m)} 0")
        for i in m:
            lines.append(f"{jd_lt[i]:.6f} {10 ** (-0.4 * (H[i] - 10)):.6e} "
                         f"{sun[i, 0]:.6e} {sun[i, 1]:.6e} {sun[i, 2]:.6e} "
                         f"{earth[i, 0]:.6e} {earth[i, 1]:.6e} {earth[i, 2]:.6e}")
    with open(os.path.join(run, "lcs"), "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    file_order = np.concatenate([order[sh[order] == s] for s in sets])
    np.savez_compressed(os.path.join(run, "prep.npz"), jd=jd, jd_lt=jd_lt, H_red=H, y=y, sheet=sh, sun=sun,
                        earth=earth, alpha=alpha, file_order=file_order, src_index=np.where(ok)[0])
    span_h = (jd_lt.max() - jd_lt.min()) * 24
    P_lo, P_hi, dP = scan_interval(P0, span_h, float(np.mean(r)), sigma_P)
    info = {"sample": sample, "num": num, "n": int(ok.sum()), "n_unmatched": int((~ok).sum()), "n_lc": len(sets),
            "P_lcdb": float(amp_data.VSAMPLE[num]["period"]) if sample == "val" else None,
            "max_points_per_lc": int(max(np.sum(sh == s) for s in sets)), "P0": P0, "sigma_P": sigma_P,
            "span_days": span_h / 24, "P_lo": P_lo, "P_hi": P_hi, "dP": dP,
            "n_trial_periods": int(np.ceil((P_hi - P_lo) / (0.8 * dP))) + 1,
            "closure_resid_au": {"median": float(np.median(resid)), "max": float(np.max(resid))},
            "alpha_check_deg": {"median_abs": float(np.median(np.abs(dalpha))), "max_abs": float(np.max(np.abs(dalpha)))}}
    json.dump(info, open(os.path.join(run, "prep.json"), "w"), indent=1)
    return info


def main(argv):
    only = argv[argv.index("--only") + 1] if "--only" in argv else None
    jobs = [("paper", n) for n in amp_data.ADOPTED] + [("val", n) for n in amp_data.VSAMPLE]
    jobs = [j for j in jobs if only is None or j[0] == only]
    if "--nums" in argv:
        want = set(argv[argv.index("--nums") + 1].split(","))
        jobs = [j for j in jobs if j[1] in want]
    out = {}
    for s, n in jobs:
        try:
            out[f"{s}:{n}"] = prepare(s, n)
            i = out[f"{s}:{n}"]
            print(f"{s}:{n} n={i['n']} lcs={i['n_lc']} trials={i['n_trial_periods']} "
                  f"closure={i['closure_resid_au']['max']:.2e} dalpha={i['alpha_check_deg']['max_abs']:.3f}", flush=True)
        except Exception as exc:  # noqa: BLE001
            out[f"{s}:{n}"] = {"error": repr(exc)[:300]}
            print(f"{s}:{n} FAILED {exc!r}", flush=True)
    if only is None and "--nums" not in argv:
        json.dump(out, open(os.path.join(OUT, "prep_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
