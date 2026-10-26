"""
Period analysis of the geometry-corrected photometry.

For every asteroid whose aspect model is significant (Delta BIC > 10, geometry_validate.py), the rotation-averaged
brightness change with aspect is removed from every measurement: y_geo = y - [M(xi_i) - <M>], with xi_i from the
measurement's own phase-angle bisector. Then
  paper sample:      periodogram scan 0.8-1500 h; candidates of rebootstrap.py plus the strongest new peaks;
                     rebootstrap2.py (bootstrap, exact-chi2 subsets) on the corrected data -> results_geo/<N>/;
                     quality rule (quality_rule.grade) -> comparison with the adopted solution;
  validation sample: periodogram power at the LCDB period and whether the strongest primary-range peak belongs to
                     its alias family, before and after the correction.
Usage: python geometry_reanalysis.py [prep|boot|grade|validation|all]
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import shutil
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results_geo")
GEO = json.load(open(os.path.join(ROOT, "geometry_results.json"), encoding="utf-8"))
DBIC_MIN = 10.0


def significant(sample):
    return [k.split(":")[1] for k, r in GEO.items() if k.startswith(sample + ":") and r and r.get("geometry")
            and r["geometry"]["delta_bic"] > DBIC_MIN]


def corrected_y(d, par):
    import amp_geometry as G
    _, Mo = G.model_at(par, G.pab_vectors(d["pab_lon"], d["pab_lat"]), d["alpha"])
    return d["y"] - (Mo - np.mean(Mo)), Mo


def scan(t, y, pmin=0.8, pmax=1500.0, spp=5):
    from astropy.timeseries import LombScargle
    from scipy.signal import find_peaks
    base = float(t.max() - t.min())
    f = np.arange(24.0 / pmax, 24.0 / pmin, 1.0 / (spp * base))
    p = LombScargle(t, y, nterms=2).power(f, method="fastchi2", assume_regular_frequency=True)
    k, _ = find_peaks(p)
    order = k[np.argsort(p[k])[::-1]]
    peaks = []
    for i in order:
        P = 24.0 / f[i]
        if all(abs(P - q) > 0.01 * q for q, _ in peaks):
            peaks.append((float(P), float(p[i])))
        if len(peaks) >= 12:
            break
    return f, p, peaks


def prep_one(num):
    import amp_data as D
    d = D.load("paper", num)
    par = GEO[f"paper:{num}"]["geometry"]
    yg, Mo = corrected_y(d, par)
    os.makedirs(os.path.join(OUT, num), exist_ok=True)
    a = dict(np.load(os.path.join(ROOT, "results", num, f"{num}_arrays.npz")))
    a["y"] = yg
    np.savez_compressed(os.path.join(OUT, num, f"{num}_arrays.npz"), **a)
    _, _, pk0 = scan(d["t"], d["y"])
    _, _, pk1 = scan(d["t"], yg)
    rb = json.load(open(os.path.join(ROOT, "results", num, f"{num}_rebootstrap.json"), encoding="utf-8"))
    base = float(d["t"].max() - d["t"].min())
    have = [c["P"] for c in rb["primary"] + rb["extended"]]
    added = []
    for P, _ in pk1[:6]:
        dP = P ** 2 / 24.0 / base
        if all(abs(P - q) > 1.5 * dP for q in have):
            (rb["primary"] if P <= 240 else rb["extended"]).append({"P": P})
            have.append(P)
            added.append(P)
    json.dump(rb, open(os.path.join(OUT, num, f"{num}_rebootstrap.json"), "w"), indent=1)
    info = {"num": num, "correction_ptp": float(np.ptp(Mo)), "correction_std": float(np.std(Mo)),
            "peaks_before": pk0[:8], "peaks_after": pk1[:8], "candidates_added": added}
    json.dump(info, open(os.path.join(OUT, num, f"{num}_geo_info.json"), "w"), indent=1)
    return num, f"correction ptp {np.ptp(Mo):.3f} mag; top peak before {pk0[0][0]:.4f} ({pk0[0][1]:.3f}) after " \
                f"{pk1[0][0]:.4f} ({pk1[0][1]:.3f}); added {['%.3f' % x for x in added]}"


def boot_one(num):
    import rebootstrap2 as r2
    r2.RES = OUT
    return r2.run(int(num))


def repo_threshold(power, distance=200, nsigma=5.0):
    """5-sigma background threshold computed as in AsteroidLSPipeline.find_peaks (repository code): candidate peaks
    masked over 1.5 half-widths, running mean over len/20 points, threshold = median + nsigma * std(residual)."""
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import find_peaks, peak_widths
    y_p = np.asarray(power, float)
    cand, _ = find_peaks(y_p, distance=distance)
    y_bg = y_p.copy()
    if cand.size:
        widths = peak_widths(y_p, cand, rel_height=0.5)[0]
        mask = np.zeros(len(y_bg), bool)
        for p_idx, w in zip(cand, widths):
            hs = max(2, int(np.ceil(1.5 * w)))
            mask[max(0, p_idx - hs):min(len(y_bg), p_idx + hs + 1)] = True
        y_bg[mask] = np.nan
        ok = np.isfinite(y_bg)
        x = np.arange(len(y_bg), dtype=float)
        y_bg[~ok] = np.interp(x[~ok], x[ok], y_bg[ok])
    k = max(51, len(y_p) // 20)
    k += 1 - k % 2
    sm = uniform_filter1d(y_bg, size=int(k), mode="nearest")
    return float(np.nanmedian(sm) + nsigma * np.nanstd(y_bg - sm))


def figarrays_one(num):
    """Periodogram of the corrected data as in mpc_redo.py (0.8-1500 h, 5 samples per peak, fastchi2, unweighted and
    noise-weighted; threshold from the primary range), reduced to 60 000 points; written into results_geo arrays."""
    from astropy.timeseries import LombScargle
    path = os.path.join(OUT, num, f"{num}_arrays.npz")
    a = dict(np.load(path))
    t, y, sheet = a["t"], a["y"], a["sheet"]
    ls = LombScargle(t, y, nterms=2)
    freq = ls.autofrequency(samples_per_peak=5, minimum_frequency=24.0 / 1500.0, maximum_frequency=24.0 / 0.8)
    power = ls.power(freq, method="fastchi2", assume_regular_frequency=True)
    sd = {ds: np.std(y[sheet == ds], ddof=1) for ds in np.unique(sheet)}
    dy = np.array([sd[s] for s in sheet])
    power_w = LombScargle(t, y, dy=dy, nterms=2).power(freq, method="fastchi2", assume_regular_frequency=True)
    periods = 24.0 / freq
    thr = repo_threshold(power[periods <= 240.0])
    order_p = np.argsort(periods)
    ps, pw, pww = periods[order_p], power[order_p], power_w[order_p]
    edges = np.linspace(0, len(ps), 60001).astype(int)
    keep = [s + int(np.argmax(pw[s:e])) for s, e in zip(edges[:-1], edges[1:]) if e > s]
    a.update({"period": ps[keep], "power": pw[keep].astype(np.float32), "power_weighted": pww[keep].astype(np.float32),
              "threshold": thr})
    np.savez_compressed(path, **a)
    return num, thr


def validation_one(num):
    import amp_data as D
    d = D.load("val", num)
    par = GEO[f"val:{num}"]["geometry"]
    yg, Mo = corrected_y(d, par)
    P0 = d["P"]
    out = {"num": num, "correction_ptp": float(np.ptp(Mo))}
    for name, y in (("before", d["y"]), ("after", yg)):
        f, p, pk = scan(d["t"], y, pmin=0.8, pmax=240.0)
        j = np.argmin(np.abs(24.0 / f - P0))
        win = np.abs(24.0 / f - P0) <= 0.003 * P0
        fam = any(abs(pk[0][0] - r * P0) <= 0.01 * pk[0][0] for r in (1.0, 2.0, 0.5))
        out[name] = {"power_at_P": float(p[win].max()), "top_peak": pk[0], "top_in_family": bool(fam),
                     "median_power": float(np.median(p))}
    return num, out


def main(argv):
    stage = argv[0] if argv else "all"
    paper = significant("paper")
    if stage in ("prep", "all"):
        with mp.get_context("spawn").Pool(13) as pool:
            for n, msg in pool.imap_unordered(prep_one, paper):
                print(f"[prep {n}] {msg}", flush=True)
    if stage in ("boot", "all"):
        with mp.get_context("spawn").Pool(13) as pool:
            for msg in pool.imap_unordered(boot_one, paper):
                print("[boot]", msg, flush=True)
    if stage in ("grade", "all"):
        import quality_rule as Q
        A = json.load(open(os.path.join(ROOT, "adopted.json"), encoding="utf-8"))
        QF = json.load(open(os.path.join(ROOT, "quality_final.json"), encoding="utf-8"))
        res = {}
        for n in sorted(paper, key=int):
            v = json.load(open(os.path.join(OUT, n, f"{n}_rebootstrap2.json"), encoding="utf-8"))
            g = Q.grade(n, A[n], v)
            info = json.load(open(os.path.join(OUT, n, f"{n}_geo_info.json"), encoding="utf-8"))
            res[n] = {"before": {k: QF[n][k] for k in ("quality_final", "quality_rule", "fam_u", "fam_w", "groups", "loo", "tess")},
                      "after": {k: g[k] for k in ("quality_final", "quality_rule", "fam_u", "fam_w", "groups", "loo", "tess")},
                      "sigma_P_after": g["sigma_P"], **info}
            b, a_ = res[n]["before"], res[n]["after"]
            print(f"{n}: P={A[n]['P']:.4f} grade {b['quality_final']} -> {a_['quality_final']} (rule {b['quality_rule']} -> "
                  f"{a_['quality_rule']}); famU {b['fam_u']:.2f}->{a_['fam_u']:.2f} famW {b['fam_w']:.2f}->{a_['fam_w']:.2f} "
                  f"groups {b['groups']}->{a_['groups']} loo {b['loo']}->{a_['loo']}; correction {info['correction_ptp']:.2f} mag; "
                  f"top peak {info['peaks_before'][0][0]:.3f} -> {info['peaks_after'][0][0]:.3f}")
        json.dump(res, open(os.path.join(ROOT, "geometry_reanalysis.json"), "w"), indent=1)
    if stage in ("figarrays", "all"):
        with mp.get_context("spawn").Pool(13) as pool:
            for n, thr in pool.imap_unordered(figarrays_one, paper):
                print(f"[figarrays {n}] threshold {thr:.4f}", flush=True)
    if stage in ("validation", "all"):
        val = significant("val")
        with mp.get_context("spawn").Pool(16) as pool:
            out = dict(pool.imap_unordered(validation_one, val))
        json.dump(out, open(os.path.join(ROOT, "geometry_reanalysis_validation.json"), "w"), indent=1)
        pb = np.array([o["before"]["power_at_P"] for o in out.values()])
        pa = np.array([o["after"]["power_at_P"] for o in out.values()])
        fb = sum(o["before"]["top_in_family"] for o in out.values())
        fa = sum(o["after"]["top_in_family"] for o in out.values())
        print(f"validation ({len(out)} asteroids with a significant aspect term): power at the LCDB period "
              f"median {np.median(pb):.3f} -> {np.median(pa):.3f} (higher after correction for {int(np.sum(pa > pb))}); "
              f"strongest primary-range peak in the LCDB alias family: {fb} -> {fa}")


if __name__ == "__main__":
    main(sys.argv[1:])
