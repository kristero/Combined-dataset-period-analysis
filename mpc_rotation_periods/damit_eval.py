"""
Comparison of the DAMIT convex inversion with the triaxial-ellipsoid models, on the same MPC data.

Models
  E1  aspect fit of amp_geometry (geometric-scattering ellipsoid fitted to the per-apparition amplitudes g_j and
      mean levels mu_j of the template fit) - the model used in the paper
  C   convex model (damit_run.py: period scan + 24 initial poles, degree/order 6, 8 rows) fitted to all points
  E2  point-level triaxial ellipsoid (damit_ellipsoid.py): same points, scattering law, phase function and
      relative chi^2 as C; only the shape differs
Metrics
  fit        rms deviation (relative intensity) of C and E2; scan and pole uniqueness of C; dark facet
  period     sidereal period of C (and E2) vs the published sidereal periods (LCDB spin-axis file)
  poles      axis distance to the nearest published pole (as geometry_validate: v, -v and lambda+180 equivalent)
             for E1, C, E2; pole distance including the sense of rotation for C, E2
  epochs     literature amplitudes at their epochs: does the model predict which epoch had the larger amplitude
             (as geometry_validate, E1: 79 % / 88 %)
  loao       leave-one-apparition-out prediction of g_j and mu_j (E1 refitted without window j; C and E2 refits
             of damit_run loao / damit_ellipsoid), and a null model (constant amplitude, linear trend in alpha)
  correction brightness correction before the period search (validation sample): power at P and whether the
             strongest peak below 240 h is in the alias family of P, for y; y - E1; H_red - C; y - C(aspect)
Output: damit_runs/<run>/eval.json, damit_comparison.json
Usage: python damit_eval.py [--workers 16]
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
RUNS = os.path.join(ROOT, "damit_runs")
POLE_SEP = 30.0           # deg: solutions further apart are different pole solutions


def chi2_threshold(nu):
    return 1.0 + 0.5 * np.sqrt(2.0 / nu)


def vec(l, b):
    l, b = np.radians(l), np.radians(b)
    return np.array([np.cos(b) * np.cos(l), np.cos(b) * np.sin(l), np.sin(b)])


def ang(v1, v2):
    return float(np.degrees(np.arccos(np.clip(v1 @ v2, -1, 1))))


def pole_distance_sense(l1, b1, l2, b2):
    """Distance between poles including the sense of rotation; the lambda+180 mirror of pole 2 is equivalent."""
    return min(ang(vec(l1, b1), vec(l, b2)) for l in (l2, l2 + 180))


def norm_pole(lam, bet):
    """convexinv may return beta outside [-90, 90] (colatitude unbounded): map to the equivalent pole."""
    v = vec(lam, bet)
    return float(np.degrees(np.arctan2(v[1], v[0])) % 360), float(np.degrees(np.arcsin(np.clip(v[2], -1, 1))))


def ellipsoid_model(x, jd0, nrows=6):
    import damit_model as M
    nor, dO = M.gaussian_sphere(nrows)
    lam, bet, P, phi0, u, v, pa, pd, pk = x
    b = np.exp(v)
    a = b * np.exp(u)
    A = M.ellipsoid_areas(nor, dO, a, b, 1.0)
    return M.Model(A, nor, {"lambda": lam, "beta": bet, "P": P, "t0": jd0, "phi0": phi0, "a": pa, "d": pd, "k": pk,
                            "c": 0.1})


def convex_model(run, sub, k):
    import damit_model as M
    import damit_wsl as W
    A, N, dark = W.read_areas(os.path.join(run, sub, f"areas_{k}"))
    return M.Model(A, N, W.read_par(os.path.join(run, sub, f"par_{k}"))), dark


def eph_vectors(sample, num, jd):
    """Asteroid-centric Sun and Earth vectors at epochs jd from the cached daily Horizons ephemeris."""
    import pandas as pd
    import damit_prep as DP
    path = os.path.join(ROOT, "data" if sample == "paper" else os.path.join("validation", "data"), "horizons",
                        f"{num}_geocentric_daily.csv")
    eph = pd.read_csv(path)
    t = eph["jd"].to_numpy()
    jd = np.atleast_1d(jd)
    ok = (jd >= t[0]) & (jd <= t[-1])
    lon = np.degrees(np.unwrap(np.radians(eph["pab_lon"].to_numpy())))
    q = {c: np.interp(jd, t, eph[c].to_numpy()) for c in ("r", "delta", "alpha", "pab_lat")}
    q["pab_lon"] = np.interp(jd, t, lon) % 360
    E = DP.earth_helio_ecl(jd)
    sun, earth, _ = DP.reconstruct(q["r"], q["delta"], q["alpha"], q["pab_lon"], q["pab_lat"], E)
    return sun, earth, ok, q


def window_levels(model, prep, rows_of_win, train_mask, nphi=36):
    """Predicted mu_j (y scale) and g_j of each window for a model.
    Dataset zero points Z_s = mean over training points of (H_red - m_model); level of a point on the y scale:
    mean-rotation model magnitude + Z_s - (H_red - y). mu_j: mean over the window's points of each dataset, weighted
    by the number of points (as the template fit)."""
    H, y, sh = prep["H_red"], prep["y"], prep["sheet"]
    sun, earth = prep["sun"], prep["earth"]
    m_inst = -2.5 * np.log10(model.brightness(prep["jd_lt"], sun, earth))
    Z = {s: float(np.mean((H - m_inst)[train_mask & (sh == s)])) for s in np.unique(sh) if np.any(train_mask & (sh == s))}
    out = []
    for rows in rows_of_win:
        mbar = model.mean_mag(sun[rows], earth[rows], nphi=nphi)
        lev, wts = [], []
        for s in np.unique(sh[rows]):
            if s not in Z:
                continue
            m = sh[rows] == s
            lev.append(np.mean(mbar[m] + Z[s] - (H[rows][m] - y[rows][m])))
            wts.append(m.sum())
        mu = float(np.average(lev, weights=wts)) if lev else np.nan
        pick = rows[np.linspace(0, len(rows) - 1, min(7, len(rows))).astype(int)]
        g = float(np.mean(model.amplitude(sun[pick], earth[pick], nphi=180)))
        out.append((g, mu, float(np.sum(wts) / len(rows)) if lev else 0.0))
    return out


def e1_predict(rows_all, j):
    import amp_geometry as G
    tr = [r for i, r in enumerate(rows_all) if i != j]
    par = G.fit(tr)
    r = rows_all[j]
    if par is None:
        return None
    al_tr = np.array([q["alpha_mean"] for q in tr if np.isfinite(q["g"]) and np.isfinite(q["mu"])])
    A, Mj = G.model_at(par, np.array([r["pab_vec"]]), np.array([r["alpha_mean"]]))
    return float(A[0]), float(par["mu0"] + par["c1"] * (r["alpha_mean"] - al_tr.mean()) + Mj[0])


def null_predict(rows_all, j):
    import amp_geometry as G
    tr = [r for i, r in enumerate(rows_all) if i != j and np.isfinite(r["g"]) and np.isfinite(r["mu"])]
    g = np.array([r["g"] for r in tr])
    sg = np.array([r["sigma_g"] for r in tr])
    mu = np.array([r["mu"] for r in tr])
    smu = np.array([r["sigma_mu"] for r in tr])
    al = np.array([r["alpha_mean"] for r in tr])
    wA = 1.0 / (sg ** 2 + G.SIG_A_SYS ** 2 + (G.SIG_A_REL * g) ** 2)
    wM = 1.0 / (smu ** 2 + G.SIG_MU_SYS ** 2)
    x = al - al.mean()
    _, p0, p1 = G.wls_offset_slope(mu[:, None], x, wM)
    return float(np.sum(wA * g) / np.sum(wA)), float(p0[0] + p1[0] * (rows_all[j]["alpha_mean"] - al.mean()))


def scan_unique(run, n_data, n_par):
    import damit_run as R
    sc = R.read_scan(run)
    i = int(np.argmin(sc["dev"]))
    P, dev = sc["P"], sc["dev"]
    info = json.load(open(os.path.join(run, "prep.json")))
    far = np.abs(P - P[i]) > info["dP"]
    thr = chi2_threshold(n_data - n_par)
    ratio = float((dev[far].min() / dev[i]) ** 2) if far.any() else np.nan
    return {"P_best": float(P[i]), "dev_min": float(dev[i]), "dev_median": float(np.median(dev)),
            "chi2_ratio_next": ratio, "threshold": float(thr), "unique": bool(ratio > thr) if far.any() else None,
            "n_trials": int(len(P))}


def sol_index(r):
    import damit_run as R
    return r["idx"] if "idx" in r else R.POLES.index(tuple(r["start"]))


def pole_clusters(sols, n_data, n_par):
    s = sorted(sols, key=lambda r: r["dev"])
    clusters = []
    for r in s:
        l, b = norm_pole(r["lambda"], r["beta"])
        v = vec(l, b)
        for c in clusters:
            if ang(v, c["v"]) <= POLE_SEP:
                break
        else:
            clusters.append({"v": v, "lambda": l, "beta": b, "dev": r["dev"], "P": r["P"], "idx": sol_index(r)})
    thr = chi2_threshold(n_data - n_par)
    best = clusters[0]
    ok = [c for c in clusters if (c["dev"] / best["dev"]) ** 2 <= thr]
    # Durech et al. (2018): at most two poles below the threshold, and a pair must differ by 120-240 deg in lambda
    # and by less than 50 deg in beta (the mirror ambiguity)
    mirror = [c for c in ok[1:] if abs(((c["lambda"] - best["lambda"] + 180) % 360) - 180) >= 120 and
              abs(c["beta"] - best["beta"]) < 50]
    return {"n_clusters": len(clusters), "n_within_threshold": len(ok),
            "within": [{"lambda": c["lambda"], "beta": c["beta"], "dev": c["dev"]} for c in ok],
            "only_mirror_ambiguity": bool(len(ok) == 1 or (len(ok) == 2 and len(mirror) == 1))}


def periodogram_test(t, y, P0):
    import geometry_reanalysis as GR
    # primary range 0.8-240 h as in the paper; for longer adopted periods the range extends to 2.5 P0
    f, p, pk = GR.scan(t, y, pmin=0.8, pmax=240.0 if P0 < 200.0 else min(5000.0, 2.5 * P0))
    win = np.abs(24.0 / f - P0) <= 0.003 * P0
    fam = any(abs(pk[0][0] - r * P0) <= 0.01 * pk[0][0] for r in (1.0, 2.0, 0.5))
    return {"power_at_P": float(p[win].max()), "top_peak": pk[0], "top_in_family": bool(fam)}


def evaluate(run):
    out_path = os.path.join(run, "eval.json")
    try:
        import amp_data as D
        import amp_geometry as G
        import amp_methods as AM
        import damit_model as M
        info = json.load(open(os.path.join(run, "prep.json")))
        sample, num = info["sample"], info["num"]
        key = f"{sample}:{num}"
        GEO = json.load(open(os.path.join(ROOT, "geometry_results.json")))[key]
        prep = dict(np.load(os.path.join(run, "prep.npz"), allow_pickle=True))
        prep["sheet"] = prep["sheet"].astype(str)
        d = D.load(sample, num)
        tp = AM.template_fit(d)
        rows = tp["rows"] if tp else []
        pos_of_src = np.full(len(d["t"]), -1)
        pos_of_src[prep["src_index"]] = np.arange(len(prep["src_index"]))
        win_rows = [pos_of_src[r["idx"]] for r in rows]
        n_data = len(prep["y"])
        res = {"key": key, "n": n_data, "n_windows": len(rows)}
        # ---- C
        polesj = json.load(open(os.path.join(run, "poles", "poles.json")))
        bestC = polesj.get("best") or min(polesj["solutions"], key=lambda r: r["dev"])
        n_par_C = 49 + 3 + 3 - 1
        modC, darkC = convex_model(run, "poles", sol_index(bestC))
        lC, bC = norm_pole(bestC["lambda"], bestC["beta"])
        L = M.read_lcs(os.path.join(run, "lcs"))
        devC_np, _ = M.relative_dev(modC.brightness(L["jd"], L["sun"], L["earth"]), L["br"], L["lc"])
        res["C"] = {"lambda": lC, "beta": bC, "P": bestC["P"], "dev": bestC["dev"], "dev_numpy": devC_np,
                    "dark_pct": bestC["dark_pct"], "phase": [bestC["a"], bestC["d"], bestC["k"]],
                    "scan": scan_unique(run, n_data, n_par_C),
                    "poles": pole_clusters(polesj["solutions"], n_data, n_par_C)}
        # ---- E2
        ell = json.load(open(os.path.join(run, "ellipsoid.json")))
        xb = [ell["best"][k] for k in ell["names"]]
        modE2 = ellipsoid_model(xb, ell["jd0"])
        lE, bE = norm_pole(xb[0], xb[1])
        res["E2"] = {"lambda": lE, "beta": bE, "P": xb[2], "a_b": float(np.exp(xb[4])), "b_c": float(np.exp(xb[5])),
                     "dev": ell["dev"]}
        # ---- E1
        e1 = GEO.get("geometry")
        res["E1"] = None if e1 is None else {"lambda": e1["lambda"], "beta": e1["beta"], "a_b": e1["a_b"],
                                             "b_c": e1["b_c"], "delta_bic": e1["delta_bic"]}
        # ---- windows: in-sample and leave-one-apparition-out
        allmask = np.ones(n_data, bool)
        insC = window_levels(modC, prep, win_rows, allmask) if rows else []
        insE2 = window_levels(modE2, prep, win_rows, allmask) if rows else []
        loao_C = json.load(open(os.path.join(run, "loao", "loao.json")))["windows"]
        wins = []
        for j, r in enumerate(rows):
            train = np.ones(n_data, bool)
            train[win_rows[j]] = False
            w = {"j": j, "g": r["g"], "sigma_g": r["sigma_g"], "mu": r["mu"], "sigma_mu": r["sigma_mu"],
                 "alpha": r["alpha_mean"], "n": r["n"],
                 "C_in": insC[j][:2], "E2_in": insE2[j][:2]}
            if e1 is not None:
                A1, M1 = G.model_at(e1, np.array([r["pab_vec"]]), np.array([r["alpha_mean"]]))
                al = np.array([q["alpha_mean"] for q in rows])
                w["E1_in"] = [float(A1[0]), float(e1["mu0"] + e1["c1"] * (r["alpha_mean"] - al.mean()) + M1[0])]
            w["E1_loao"] = e1_predict(rows, j)
            w["null_loao"] = null_predict(rows, j)
            lc = [q for q in loao_C if q["j"] == j and "failed" not in q]
            if lc:
                mj, _ = convex_model(run, "loao", j)
                g_, mu_, cov = window_levels(mj, prep, [win_rows[j]], train)[0]
                w["C_loao"] = [g_, mu_]
                w["zp_coverage"] = cov
            le = [q for q in ell["loao"] if q["j"] == j]
            if le:
                mj = ellipsoid_model(le[0]["x"], ell["jd0"])
                g_, mu_, _ = window_levels(mj, prep, [win_rows[j]], train)[0]
                w["E2_loao"] = [g_, mu_]
            wins.append(w)
        res["windows"] = wins
        # ---- literature amplitudes at their epochs
        dense = json.load(open(os.path.join(ROOT, "data", "lcdb_dense_refs.json"), encoding="utf-8")).get(num)
        ep = []
        if dense and dense.get("amps"):
            from astropy.time import Time
            jds = []
            for date in dense["dates"]:
                try:
                    jds.append(Time(date).jd)
                except Exception:  # noqa: BLE001
                    jds.append(np.nan)
            jds = np.array(jds)
            good = np.isfinite(jds)
            if good.any():
                sun, earth, ok, q = eph_vectors(sample, num, np.where(good, jds, 2451545.0))
                AC = modC.amplitude(sun, earth)
                AE = modE2.amplitude(sun, earth)
                for i, (A_lit, ref) in enumerate(zip(dense["amps"], dense["refs"])):
                    if not (good[i] and ok[i]):
                        continue
                    A1 = None
                    if e1 is not None:
                        a1, _ = G.model_at(e1, G.pab_vectors(np.array([q["pab_lon"][i]]), np.array([q["pab_lat"][i]])),
                                           np.array([q["alpha"][i]]))
                        A1 = float(a1[0])
                    ep.append({"jd": float(jds[i]), "ref": ref, "A_lit": A_lit, "A_E1": A1, "A_C": float(AC[i]),
                               "A_E2": float(AE[i]), "alpha": float(q["alpha"][i])})
        res["epochs"] = ep
        # ---- brightness correction for the period search
        y, H, sh = prep["y"], prep["H_red"], prep["sheet"]
        t = d["t"][prep["src_index"]]
        mbar = modC.mean_mag(prep["sun"], prep["earth"], nphi=36)
        mbarE = modE2.mean_mag(prep["sun"], prep["earth"], nphi=36)
        yC = H - mbar
        yC = yC - _zp(yC, sh)
        yE = H - mbarE
        yE = yE - _zp(yE, sh)
        # aspect-only part of C (C's dependence on phase angle removed with a cubic in alpha) applied to y
        al = prep["alpha"]
        cf = np.polyfit(al, mbar, 3)
        yCa = y - (mbar - np.polyval(cf, al))
        variants = {"none": y, "C_full": yC, "C_aspect": yCa, "E2_full": yE}
        if e1 is not None:
            _, Mo = G.model_at(e1, G.pab_vectors(d["pab_lon"], d["pab_lat"]), d["alpha"])
            variants["E1"] = y - (Mo - Mo.mean())[prep["src_index"]]
        P0 = d["P"]
        res["periodogram"] = {k: periodogram_test(t, v, P0) for k, v in variants.items()}
        res["correction_ptp"] = {"C_aspect": float(np.ptp(mbar - np.polyval(cf, al)))}
        json.dump(res, open(out_path, "w"), indent=1, default=float)
        return run, "ok"
    except Exception:  # noqa: BLE001
        return run, "FAILED " + traceback.format_exc()[-800:]


def _zp(r, sh):
    z = np.empty_like(r)
    for s in np.unique(sh):
        m = sh == s
        z[m] = np.mean(r[m])
    return z


# =============================================================================================== aggregation
def summarize():
    import amp_geometry as G
    poles = json.load(open(os.path.join(ROOT, "data", "lcdb_poles.json"), encoding="utf-8"))
    E = {}
    for dname in sorted(os.listdir(RUNS)):
        p = os.path.join(RUNS, dname, "eval.json")
        if os.path.exists(p):
            r = json.load(open(p))
            E[r["key"]] = r
    S = {"n": len(E)}
    # fit
    dC = np.array([r["C"]["dev"] for r in E.values()])
    dE = np.array([r["E2"]["dev"] for r in E.values()])
    nn = np.array([r["n"] for r in E.values()])
    dchi = nn * (dE ** 2 - dC ** 2) / dC ** 2          # chi2 improvement in units of the C residual variance
    dk = (49 + 3 + 3 - 1) - 9
    dbic = dchi - dk * np.log(nn)
    S["fit"] = {"median_dev_C": float(np.median(dC)), "median_dev_E2": float(np.median(dE)),
                "median_dev_ratio_E2_over_C": float(np.median(dE / dC)),
                "frac_C_better_by_BIC": float(np.mean(dbic > 10)), "median_delta_BIC": float(np.median(dbic)),
                "scan_unique_frac": float(np.mean([bool(r["C"]["scan"]["unique"]) for r in E.values()])),
                "pole_unique_frac": float(np.mean([r["C"]["poles"]["only_mirror_ambiguity"] for r in E.values()])),
                "dark_lt1_frac": float(np.mean([r["C"]["dark_pct"] < 1 for r in E.values()]))}
    # periods and poles
    rows, dist = [], {"E1": [], "C": [], "E2": [], "C_sense": [], "E2_sense": []}
    rng = np.random.default_rng(3)
    rand = []
    for k, r in E.items():
        n = k.split(":")[1]
        if n not in poles:
            continue
        sols = [p for e in poles[n] for p in e["poles"]]          # full poles only (lcdb_poles.py)
        sid = [e["sid_per"] for e in poles[n] if e.get("sid_per")]
        row = {"asteroid": k, "published": sols, "sid_per": sid}
        for m in ("E1", "C", "E2"):
            if r.get(m) is None or not sols:
                if m in ("C", "E2") and r.get(m) is not None:
                    row[m] = {"P": r[m]["P"]}
                continue
            dd = min(G.pole_distance(r[m]["lambda"], r[m]["beta"], l, b) for l, b in sols)
            dist[m].append(dd)
            row[m] = {"lambda": r[m]["lambda"], "beta": r[m]["beta"], "axis_distance": dd}
            if m in ("C", "E2"):
                ds = min(pole_distance_sense(r[m]["lambda"], r[m]["beta"], l, b) for l, b in sols)
                dist[m + "_sense"].append(ds)
                row[m]["sense_distance"] = ds
                row[m]["P"] = r[m]["P"]
                if sid:
                    row[m]["dP_rel"] = float(min(abs(r[m]["P"] - s) / s for s in sid))
        for _ in range(200 if sols else 0):
            v = rng.normal(size=3)
            v /= np.linalg.norm(v)
            lr, br = np.degrees(np.arctan2(v[1], v[0])) % 360, np.degrees(np.arcsin(v[2]))
            rand.append((min(G.pole_distance(lr, br, l, b) for l, b in sols),
                         min(pole_distance_sense(lr, br, l, b) for l, b in sols)))
        row["C_scan_unique"] = r["C"]["scan"]["unique"]
        rows.append(row)
    rand = np.array(rand)
    S["poles"] = {m: {"n": len(v), "median": float(np.median(v)), "frac_within_30": float(np.mean(np.array(v) <= 30)),
                      "frac_within_20": float(np.mean(np.array(v) <= 20))} for m, v in dist.items() if v}
    S["poles"]["random_axis"] = {"median": float(np.median(rand[:, 0])), "frac_within_30": float(np.mean(rand[:, 0] <= 30))}
    S["poles"]["random_sense"] = {"median": float(np.median(rand[:, 1])), "frac_within_30": float(np.mean(rand[:, 1] <= 30))}
    S["poles"]["rows"] = rows
    # sidereal period of C vs the periodogram period of the MPC data (P_mpc), against published sidereal periods
    prow = []
    for row in rows:
        k = row["asteroid"]
        if not row["sid_per"] or "C" not in row:
            continue
        r = E[k]
        info = json.load(open(os.path.join(RUNS, k.replace(":", "_"), "prep.json")))
        sid = min(row["sid_per"], key=lambda s: abs(s - info["P0"]))
        if abs(sid - info["P0"]) / sid > 0.005:       # published period is a different period (or a typo)
            continue
        prow.append({"asteroid": k, "sid": sid, "P_mpc": info["P0"], "P_C": r["C"]["P"], "P_E2": r["E2"]["P"],
                     "dP_mpc_in_dP": abs(info["P0"] - sid) / info["dP"], "dP_C_in_dP": abs(r["C"]["P"] - sid) / info["dP"],
                     "dP_E2_in_dP": abs(r["E2"]["P"] - sid) / info["dP"]})
    S["sidereal_period"] = {"n": len(prow),
                            "frac_C_within_1dP": float(np.mean([p["dP_C_in_dP"] <= 1 for p in prow])) if prow else None,
                            "frac_mpc_within_1dP": float(np.mean([p["dP_mpc_in_dP"] <= 1 for p in prow])) if prow else None,
                            "frac_E2_within_1dP": float(np.mean([p["dP_E2_in_dP"] <= 1 for p in prow])) if prow else None,
                            "rows": prow}
    # epochs: model amplitude / literature amplitude
    for m in ("A_E1", "A_C", "A_E2"):
        pairs = [(p[m], p["A_lit"]) for r in E.values() for p in r["epochs"] if p.get(m) is not None and p[m] > 0]
        lr = np.log10(np.array([a / b for a, b in pairs])) if pairs else np.array([np.nan])
        S.setdefault("epoch_amplitude_ratio", {})[m] = {
            "n": len(pairs), "median_ratio": float(10 ** np.median(lr)),
            "scatter_dex": float(1.4826 * np.median(np.abs(lr - np.median(lr)))),
            "rms_dex": float(np.sqrt(np.mean(lr ** 2)))}
    # epochs: direction test
    for grp, cond in (("all", lambda r: True), ("E1_dBIC>10", lambda r: r["E1"] and r["E1"]["delta_bic"] > 10)):
        out = {}
        for m in ("A_E1", "A_C", "A_E2"):
            hits, n_obj = [], 0
            for r in E.values():
                if not cond(r):
                    continue
                pr = [p for p in r["epochs"] if p.get(m) is not None]
                if len(pr) >= 2 and np.ptp([p["A_lit"] for p in pr]) >= 0.05:
                    hi = pr[int(np.argmax([p["A_lit"] for p in pr]))]
                    lo = pr[int(np.argmin([p["A_lit"] for p in pr]))]
                    hits.append(hi[m] - lo[m] > 0)
            out[m] = {"n_objects": len(hits), "frac_correct": float(np.mean(hits)) if hits else None}
        S["epochs_" + grp] = out
    # leave-one-apparition-out
    def errs(model, kind):
        eg, em, zg, zm = [], [], [], []
        for r in E.values():
            for w in r["windows"]:
                p = w.get(model)
                if p is None or not np.all(np.isfinite(p)):
                    continue
                sg = np.sqrt(w["sigma_g"] ** 2 + G.SIG_A_SYS ** 2 + (G.SIG_A_REL * w["g"]) ** 2)
                sm = np.sqrt(w["sigma_mu"] ** 2 + G.SIG_MU_SYS ** 2)
                eg.append(w["g"] - p[0])
                em.append(w["mu"] - p[1])
                zg.append((w["g"] - p[0]) / sg)
                zm.append((w["mu"] - p[1]) / sm)
        eg, em, zg, zm = map(np.array, (eg, em, zg, zm))
        return {"n": int(len(eg)), "g_median_abs": float(np.median(np.abs(eg))), "g_rms_norm": float(np.sqrt(np.mean(zg ** 2))),
                "mu_median_abs": float(np.median(np.abs(em))), "mu_rms_norm": float(np.sqrt(np.mean(zm ** 2)))}
    S["windows_in_sample"] = {m: errs(m, "in") for m in ("E1_in", "C_in", "E2_in")}
    S["windows_loao"] = {m: errs(m, "loao") for m in ("null_loao", "E1_loao", "C_loao", "E2_loao")}
    # common windows only (all four predictions present)
    def common(models):
        keep = []
        for r in E.values():
            for w in r["windows"]:
                if all(w.get(m) is not None and np.all(np.isfinite(w[m])) for m in models):
                    keep.append(w)
        out = {}
        for m in models:
            eg = np.array([abs(w["g"] - w[m][0]) for w in keep])
            em = np.array([abs(w["mu"] - w[m][1]) for w in keep])
            out[m] = {"n": len(keep), "g_median_abs": float(np.median(eg)), "mu_median_abs": float(np.median(em))}
        return out
    S["windows_loao_common"] = common(("null_loao", "E1_loao", "C_loao", "E2_loao"))
    # brightness correction (validation sample)
    for grp, cond in (("val_all", lambda k, r: k.startswith("val:")),
                      ("val_E1_dBIC>10", lambda k, r: k.startswith("val:") and r["E1"] and r["E1"]["delta_bic"] > 10)):
        sel = [r for k, r in E.items() if cond(k, r)]
        out = {}
        for v in ("none", "E1", "C_full", "C_aspect", "E2_full"):
            pw = [r["periodogram"][v]["power_at_P"] for r in sel if v in r["periodogram"]]
            fam = [r["periodogram"][v]["top_in_family"] for r in sel if v in r["periodogram"]]
            up = [r["periodogram"][v]["power_at_P"] > r["periodogram"]["none"]["power_at_P"] for r in sel
                  if v in r["periodogram"]]
            out[v] = {"n": len(pw), "median_power_at_P": float(np.median(pw)) if pw else None,
                      "top_in_family": int(np.sum(fam)), "frac_power_increased": float(np.mean(up)) if up else None}
        S["correction_" + grp] = out
    json.dump(S, open(os.path.join(ROOT, "damit_comparison.json"), "w"), indent=1, default=float)
    return S


def main(argv):
    workers = int(argv[argv.index("--workers") + 1]) if "--workers" in argv else 16
    nums = set(argv[argv.index("--nums") + 1].split(",")) if "--nums" in argv else None
    if "--summary" not in argv:
        runs = [os.path.join(RUNS, d) for d in sorted(os.listdir(RUNS))
                if os.path.exists(os.path.join(RUNS, d, "ellipsoid.json"))
                and os.path.exists(os.path.join(RUNS, d, "loao", "loao.json"))
                and (nums is None or d.split("_", 1)[1] in nums)
                and ("--missing" not in argv or not os.path.exists(os.path.join(RUNS, d, "eval.json")))]
        with mp.get_context("spawn").Pool(workers) as pool:
            for k, (r, msg) in enumerate(pool.imap_unordered(evaluate, runs)):
                print(f"[{k + 1}/{len(runs)}] {os.path.basename(r)}: {msg}", flush=True)
    if "--no-summary" in argv:
        return
    S = summarize()
    print(json.dumps({k: v for k, v in S.items() if k != "poles"}, indent=1, default=float))
    print(json.dumps({k: v for k, v in S["poles"].items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
