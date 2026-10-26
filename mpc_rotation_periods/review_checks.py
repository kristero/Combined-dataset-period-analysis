"""
Checks requested by the referee review of 2026-10-19 (run on the LU HPC: hpc/review_checks.sbatch).
  1. harmonic test (A02, A03): for every asteroid with an adopted period, the combined data are fitted with a
     four-term Fourier series at P and with the nested two-term series at P/2 (harmonics 2 and 4 of P only);
     Delta BIC uses the number of observatory nights as the effective number of independent points
  2. convex model at P/2 for (2503) Liaoning (A02): convexinv from 12 poles at 51.398797 h, as for the paper runs
  3. principal-axis check (A05): angle between the spin axis (body z) and the axis of maximum moment of inertia of the
     uniform-density polyhedron, for the 14 models of Table 4 and both (2503) solutions
  4. injection of a 3.043 h signal into the data of (4482) Frerebasile (D03)
  5. phase function positivity for the accepted models (B11)
  6. DAMIT models of our 14 asteroids and of (2503) (C05)
Writes review_checks.json
"""
from __future__ import annotations

import csv
import io
import json
import os
import shutil
import sys

import numpy as np
from scipy.spatial import ConvexHull

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
OUT = {}
AD = json.load(open(os.path.join(ROOT, "adopted.json"), encoding="utf-8"))
OBS = json.load(open(os.path.join(ROOT, "data", "mpc_obscodes.json"), encoding="utf-8"))


def load(num):
    for d in ("results_geo", "results"):
        p = os.path.join(ROOT, d, str(num), f"{num}_arrays.npz")
        if os.path.exists(p):
            return np.load(p, allow_pickle=False)
    raise FileNotFoundError(num)


def nights(a):
    t0 = float(a["time_zero"]) if "time_zero" in a.files else 0.0
    lon = np.array([float((OBS.get(str(s)[:3]) or {}).get("lon") or 0.0) for s in a["sheet"]])
    key = [f"{s}|{int(np.floor(t + t0 + l / 360.0))}" for s, t, l in zip(a["sheet"], a["t"], lon)]
    return len(set(key))


def fourier_rss(t, y, P, harmonics):
    w = 2 * np.pi * 24.0 / P
    cols = [np.ones_like(t)]
    for k in harmonics:
        cols += [np.sin(k * w * t), np.cos(k * w * t)]
    A = np.vstack(cols).T
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ c
    return float(r @ r), c


def harmonic_test():
    res = {}
    for n, ad in AD.items():
        if ad.get("status") == "undetermined":
            continue
        a = load(n)
        t, y = a["t"].astype(float), a["y"].astype(float)
        P = ad["P"]
        rss_f, cf = fourier_rss(t, y, P, (1, 2, 3, 4))
        rss_e, _ = fourier_rss(t, y, P, (2, 4))
        ne = nights(a)
        odd = float(np.hypot(cf[1], cf[2]) + np.hypot(cf[5], cf[6]))     # semi-amplitudes of harmonics 1 and 3
        res[n] = dict(P=P, n=len(t), n_nights=ne, dbic_eff=float(ne * np.log(rss_e / rss_f) - 4 * np.log(ne)),
                      dbic_naive=float(len(t) * np.log(rss_e / rss_f) - 4 * np.log(len(t))), odd_semiamp=odd,
                      rms_gain_pct=float(100 * (1 - np.sqrt(rss_f / rss_e))), quality=ad["quality"])
    return res


def half_period_model():
    import damit_run as R
    src = os.path.join(ROOT, "damit_runs", "paper_2503")
    run = os.path.join(ROOT, "damit_runs", "paper_2503_half")
    os.makedirs(os.path.join(run, "scan"), exist_ok=True)
    for f in ("lcs", "prep.json"):
        shutil.copy(os.path.join(src, f), os.path.join(run, f))
    open(os.path.join(run, "scan", "out_periods"), "w").write("51.398797 0.098243 1.0 50 0.5\n")
    msg = R.do_poles(run)
    pj = json.load(open(os.path.join(run, "poles", "poles.json")))
    return dict(message=str(msg), best=pj["best"])


def minkowski_vertices(areas, tmp):
    import subprocess
    BIN = os.path.join(ROOT, "damit", "build_linux")
    cmd = f"ulimit -s unlimited; cat '{areas}' | timeout 900 '{BIN}/minkowski' | '{BIN}/standardtri' > '{tmp}'"
    subprocess.run(["bash", "-c", cmd], check=True)
    v = open(tmp).read().split()
    nv = int(v[0])
    return np.array(v[2:2 + 3 * nv], float).reshape(nv, 3)


def inertia(verts):
    h = ConvexHull(verts)
    c0 = verts[h.vertices].mean(axis=0)
    tri = verts[h.simplices] - c0
    vol6 = np.abs(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])))
    cen = c0 + np.sum(vol6[:, None] * tri.sum(axis=1), axis=0) / (4 * vol6.sum())
    t2 = verts[h.simplices] - cen
    d2 = np.abs(np.einsum("ij,ij->i", t2[:, 0], np.cross(t2[:, 1], t2[:, 2])))
    ss = t2.sum(axis=1)
    C = (np.einsum("k,kij,kil->jl", d2, t2, t2) + np.einsum("k,kj,kl->jl", d2, ss, ss)) / 120.0
    I = np.trace(C) * np.eye(3) - C
    ev, evec = np.linalg.eigh(I)
    ax = evec[:, 2]                       # axis of maximum moment
    ang = float(np.degrees(np.arccos(min(1.0, abs(ax[2])))))
    return dict(angle_maxI_to_spin_deg=ang, I_ratios=[float(ev[0] / ev[2]), float(ev[1] / ev[2])],
                Izz_over_Imax=float(I[2, 2] / ev[2]))


def principal_axis_checks(half):
    out = {}
    tmpd = os.path.join(ROOT, "review_tmp")
    os.makedirs(tmpd, exist_ok=True)
    T = json.load(open(os.path.join(ROOT, "damit_paper_table.json"), encoding="utf-8"))["models"]
    items = [(str(m["num"]), os.path.join(ROOT, "damit_runs", f"paper_{m['num']}", "poles", m["areas"])) for m in T]
    items.append(("2503_P", os.path.join(ROOT, "damit_runs", "paper_2503", "poles", "areas_best")))
    items.append(("2503_P2", os.path.join(ROOT, "damit_runs", "paper_2503_half", "poles", "areas_best")))
    for key, areas in items:
        try:
            v = minkowski_vertices(areas, os.path.join(tmpd, key + ".shape"))
            out[key] = inertia(v)
        except Exception as e:
            out[key] = dict(error=str(e)[:200])
    return out


def injection_4482():
    from astropy.timeseries import LombScargle
    a = load("4482")
    t, y = a["t"].astype(float), a["y"].astype(float)
    res = json.load(open(os.path.join(ROOT, "results", "4482", "4482_result.json"), encoding="utf-8"))
    thr = float(res["threshold_5sigma"])
    P0 = 3.043
    rng = np.random.default_rng(7)
    f0 = 24.0 / P0
    T = t.max() - t.min()
    fg = np.arange(f0 - 3 / T, f0 + 3 / T, 1 / (50 * T))
    base = float(np.max(LombScargle(t, y, nterms=2).power(fg, method="chi2")))
    out = dict(threshold_5sigma=thr, power_at_3043_without_injection=base, trials={})
    fgrid = np.linspace(24 / 240.0, 24 / 0.8, 400000)
    for amp in (0.12, 0.08, 0.05, 0.03):
        above = strongest = 0
        pw = []
        for _ in range(20):
            ph1, ph2 = rng.uniform(0, 2 * np.pi, 2)
            sig = 0.5 * amp * (0.85 * np.cos(2 * 2 * np.pi * t / (P0 / 24) + ph2) + 0.15 * np.cos(2 * np.pi * t / (P0 / 24) + ph1))
            yy = y + sig
            p = float(np.max(LombScargle(t, yy, nterms=2).power(fg, method="chi2")))
            pw.append(p)
            above += p > thr
            full = LombScargle(t, yy, nterms=2).power(fgrid, method="fastchi2")
            fbest = fgrid[int(np.argmax(full))]
            strongest += abs(24 / fbest - P0) / P0 < 0.01 or abs(24 / fbest - P0 / 2) / (P0 / 2) < 0.01
        out["trials"][str(amp)] = dict(n=20, above_threshold=int(above), strongest_peak=int(strongest),
                                       median_power=float(np.median(pw)))
    return out


def phase_positivity():
    import damit_wsl as W
    out = {}
    T = json.load(open(os.path.join(ROOT, "damit_paper_table.json"), encoding="utf-8"))["models"]
    for m in T:
        n = str(m["num"])
        pdir = os.path.join(ROOT, "damit_runs", f"paper_{n}", "poles")
        par = W.read_par(os.path.join(pdir, "par_best"))
        prep = np.load(os.path.join(ROOT, "damit_runs", f"paper_{n}", "prep.npz"), allow_pickle=False)
        amax = float(np.radians(np.nanmax(prep["alpha"]))) if "alpha" in prep.files else None
        k = float(par["k"])
        f = 1 + 0.5 * np.exp(-amax / 0.1) + k * amax if amax is not None else None
        out[n] = dict(k=k, alpha_max_rad=amax, f_at_alpha_max=f)
    return out


def damit_membership():
    D = os.path.join(ROOT, "damit_ref", "tables")
    rd = lambda n: list(csv.DictReader(io.StringIO(open(os.path.join(D, n + ".csv"), "rb").read().decode("utf-8-sig"))))
    ast = {a["id"]: a for a in rd("asteroids")}
    refs = {r["id"]: r for r in rd("references")}
    mref = {}
    for x in rd("asteroid_models_references"):
        mref.setdefault(x["asteroid_model_id"], []).append(refs.get(x["reference_id"], {}))
    T = json.load(open(os.path.join(ROOT, "damit_paper_table.json"), encoding="utf-8"))["models"]
    want = {str(m["num"]) for m in T} | {"2503"}
    out = {n: [] for n in want}
    for m in rd("asteroid_models"):
        a = ast.get(m["asteroid_id"])
        if a and a["number"] in want:
            out[a["number"]].append(dict(model=m["id"], lam=m["lambda"], beta=m["beta"], P=m["period"],
                                         refs=[f"{r.get('author_short')} {r.get('year')}" for r in mref.get(m["id"], [])]))
    return out


def main():
    steps = [("harmonic_test", harmonic_test), ("half_period_model", half_period_model),
             ("injection_4482", injection_4482), ("phase_positivity", phase_positivity),
             ("damit_membership", damit_membership)]
    for name, fn in steps:
        try:
            OUT[name] = fn()
        except Exception as e:
            OUT[name] = dict(error=repr(e)[:500])
        print(name, "done", flush=True)
    try:
        OUT["principal_axis"] = principal_axis_checks(OUT.get("half_period_model"))
    except Exception as e:
        OUT["principal_axis"] = dict(error=repr(e)[:500])
    json.dump(OUT, open(os.path.join(ROOT, "review_checks.json"), "w", encoding="utf-8"), indent=1, default=float)
    print("written review_checks.json")


if __name__ == "__main__":
    main()
