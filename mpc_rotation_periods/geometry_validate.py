"""
Validation of the viewing-geometry model (amp_geometry.py) on the paper and validation samples.
1. Fit the model to every asteroid (template windows of amp_methods.template_fit).
2. Spin axes: distance between our pole and the nearest published solution (LCDB spin-axis file), with a random-pole
   baseline (isotropic poles, same equivalences).
3. Literature amplitudes at their epochs: for every dense-photometry LCDB amplitude (lcdb_dense_refs.json) whose
   mid-date lies inside the ephemeris span, the model amplitude at that date's viewing direction (PAB) and phase angle
   (from the cached daily JPL Horizons ephemerides) is compared with the published amplitude, and with the
   geometry-independent per-apparition mean amplitude.
Output: geometry_results.json, geometry_validation.json
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))


def eph_path(sample, num):
    return os.path.join(ROOT, "data" if sample == "paper" else os.path.join("validation", "data"), "horizons",
                        f"{num}_geocentric_daily.csv")


def run(job):
    import amp_data as D
    import amp_methods as M
    import amp_geometry as G
    from astropy.time import Time
    s, n = job
    d = D.load(s, n)
    tp = M.template_fit(d)
    if tp is None:
        return f"{s}:{n}", None
    par = G.fit(tp["rows"])
    out = {"APP_TPL_wmean": tp["wmean"], "windows": [{k: v for k, v in r.items() if k != "idx"} for r in tp["rows"]],
           "geometry": par}
    if par is None:
        return f"{s}:{n}", out
    # model amplitude of every window (fit quality)
    E = np.array([r["pab_vec"] for r in tp["rows"]])
    al = np.array([r["alpha_mean"] for r in tp["rows"]])
    Aw, Mw = G.model_at(par, E, al)
    out["window_model_A"] = Aw.tolist()
    out["window_model_M"] = Mw.tolist()
    # predictions at the epochs of the dense LCDB amplitudes
    dense = json.load(open(os.path.join(ROOT, "data", "lcdb_dense_refs.json"), encoding="utf-8")).get(n)
    preds = []
    if dense and dense.get("amps") and os.path.exists(eph_path(s, n)):
        eph = pd.read_csv(eph_path(s, n))
        jd = eph["jd"].to_numpy()
        lon = np.degrees(np.unwrap(np.radians(eph["pab_lon"].to_numpy())))
        for date, A_lit, ref in zip(dense["dates"], dense["amps"], dense["refs"]):
            try:
                j = Time(date).jd
            except Exception:  # noqa: BLE001
                continue
            if not (jd[0] <= j <= jd[-1]):
                continue
            L = float(np.interp(j, jd, lon)) % 360
            B = float(np.interp(j, jd, eph["pab_lat"].to_numpy()))
            a = float(np.interp(j, jd, eph["alpha"].to_numpy()))
            A_mod, _ = G.model_at(par, G.pab_vectors(np.array([L]), np.array([B])), np.array([a]))
            preds.append({"date": date, "ref": ref, "A_lit": A_lit, "A_model": float(A_mod[0]), "alpha": a,
                          "pab_lon": L, "pab_lat": B})
    out["lcdb_epoch_predictions"] = preds
    return f"{s}:{n}", out


def stats(pairs):
    if not pairs:
        return None
    a = np.array(pairs)
    lr = np.log10(a[:, 0] / a[:, 1])
    return {"n": int(len(a)), "median_ratio": float(10 ** np.median(lr)), "rms_dex": float(np.sqrt(np.mean(lr ** 2))),
            "scatter_dex": float(1.4826 * np.median(np.abs(lr - np.median(lr))))}


def main():
    import amp_data as D
    import amp_geometry as G
    with mp.get_context("spawn").Pool(16) as pool:
        res = dict(pool.imap_unordered(run, D.all_ids()))
    json.dump(res, open(os.path.join(ROOT, "geometry_results.json"), "w"), indent=1)
    val = {}
    # 1. aspect-model significance
    dbic = {k: r["geometry"]["delta_bic"] for k, r in res.items() if r and r.get("geometry")}
    val["n_fitted"] = len(dbic)
    val["n_dbic_gt10"] = int(sum(v > 10 for v in dbic.values()))
    print(f"aspect model fitted for {len(dbic)} asteroids; Delta BIC > 10 for {val['n_dbic_gt10']}")
    # 2. poles
    poles = json.load(open(os.path.join(ROOT, "data", "lcdb_poles.json"), encoding="utf-8"))
    rng = np.random.default_rng(3)
    dist, rand, rows = [], [], []
    for k, r in res.items():
        n = k.split(":")[1]
        if not r or not r.get("geometry") or n not in poles:
            continue
        sols = [p for e in poles[n] for p in e["poles"]]
        if not sols:                     # latitude-only published solutions (lcdb_poles.json "lat_only")
            continue
        g = r["geometry"]
        dd = min(G.pole_distance(g["lambda"], g["beta"], l, b) for l, b in sols)
        dist.append(dd)
        for _ in range(200):
            v = rng.normal(size=3)
            v /= np.linalg.norm(v)
            lr, br = np.degrees(np.arctan2(v[1], v[0])) % 360, np.degrees(np.arcsin(v[2]))
            rand.append(min(G.pole_distance(lr, br, l, b) for l, b in sols))
        rows.append({"asteroid": k, "ours": [g["lambda"], g["beta"]], "published": sols, "distance": dd,
                     "delta_bic": g["delta_bic"]})
    val["poles"] = {"n": len(dist), "median_distance": float(np.median(dist)), "random_median": float(np.median(rand)),
                    "frac_within_30": float(np.mean(np.array(dist) <= 30)), "random_frac_within_30": float(np.mean(np.array(rand) <= 30)),
                    "rows": rows}
    print(f"poles: n={len(dist)}, median distance {np.median(dist):.0f} deg (random poles: {np.median(rand):.0f} deg); "
          f"within 30 deg: {np.mean(np.array(dist) <= 30):.2f} (random {np.mean(np.array(rand) <= 30):.2f})")
    # 3. amplitudes at LCDB epochs
    groups = {"all": lambda r: True, "dBIC>10": lambda r: r["geometry"]["delta_bic"] > 10}
    val["epoch_amplitudes"] = {}
    for gname, cond in groups.items():
        pm, pc, var_lit, var_mod = [], [], [], []
        for k, r in res.items():
            if not r or not r.get("geometry") or not cond(r):
                continue
            pr = r["lcdb_epoch_predictions"]
            for p in pr:
                pm.append((p["A_model"], p["A_lit"]))
                pc.append((r["APP_TPL_wmean"], p["A_lit"]))
            if len(pr) >= 2 and np.ptp([p["A_lit"] for p in pr]) >= 0.05:
                var_lit.append(np.ptp([p["A_lit"] for p in pr]) * np.sign(pr[int(np.argmax([p["A_lit"] for p in pr]))]["A_model"]
                                                                           - pr[int(np.argmin([p["A_lit"] for p in pr]))]["A_model"]))
        val["epoch_amplitudes"][gname] = {"aspect_model": stats(pm), "per_apparition_mean": stats(pc),
                                          "n_objects_with_varying_lit_amp": len(var_lit),
                                          "frac_model_predicts_direction": float(np.mean(np.array(var_lit) > 0)) if var_lit else None}
        e = val["epoch_amplitudes"][gname]
        print(f"[{gname}] LCDB-epoch amplitudes: aspect model {e['aspect_model']}; constant per-apparition mean "
              f"{e['per_apparition_mean']}; objects whose literature amplitude varies by >= 0.05 mag: "
              f"{e['n_objects_with_varying_lit_amp']}, model predicts which epoch is larger in {e['frac_model_predicts_direction']}")
    json.dump(val, open(os.path.join(ROOT, "geometry_validation.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
