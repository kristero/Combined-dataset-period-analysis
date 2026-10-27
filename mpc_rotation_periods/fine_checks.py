"""
Finer analyses requested after the first round of the three checks (2026-10-27). Run on the HPC (Linux DAMIT build).

inertia  Convex models that fail the principal-axis check, at their period: (870) at 171.73 h, (3716), (4239), the
         (3716) model at the competing 8.594 h, and the five validation models at their LCDB periods with the spin
         axis 84-90 deg from the axis of maximum inertia. convexinv from N_FINE initial poles on a Fibonacci sphere
         (spacing about 26 deg, against 63 deg for the 12 icosahedron vertices of damit_run.py), the best start polished,
         the best start of every other pole cluster near the threshold refined (damit_mirrors.do_run), and the
         principal-axis angle of every refined solution and of every start within the uniqueness threshold.
synsid   Asteroids flagged by synsid_check.py, (4618) and (4761) of the paper sample and (553), (594), (905), (1608)
         of the validation sample. With P_mpc the periodogram period, P_C the sidereal period of our model, e the sign
         of cos(obliquity) and s = P^2/P_orb:
           periodogram  Lomb-Scargle (nterms 2) on a grid of 200 points per resolution element around P_mpc and the
                        convex-implied mean synodic period P_C + e s; 300 night-block bootstrap samples (local nights
                        as in rebootstrap2.py): share of samples whose peak lies closer to P_C + e s than to P_mpc;
           scan         period_scan with step coefficient 0.2 (0.8 in damit_run.py) and degree 6 over both sidereal
                        candidates, P_C and P_alt = P_mpc - e s, plus 5 dP on each side;
           poles        the fine pole grid at P_C and at P_alt, best start polished, chi2 of the two compared.
Usage: python fine_checks.py [inertia] [synsid] [--workers 30]      Writes fine_checks.json (merged per mode)
"""
import concurrent.futures as cf
import json
import os
import shutil
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
RUNS = os.path.join(ROOT, "damit_runs")
OUT = os.path.join(ROOT, "fine_checks.json")
N_FINE = 60
INERTIA_RUNS = ["paper_870", "paper_3716", "paper_4239", "paper_3716_alt8p5939",
                "val_822", "val_2964", "val_3609", "val_5484", "val_5598"]
SYNSID = ["paper:4618", "paper:4761", "val:553", "val:594", "val:905", "val:1608"]


def fibonacci_poles(n):
    """n nearly uniform directions (lambda, beta) in degrees."""
    k = np.arange(n) + 0.5
    beta = np.degrees(np.arcsin(1 - 2 * k / n))
    lam = (np.degrees(np.pi * (1 + 5 ** 0.5) * k)) % 360
    return [(float(a), float(b)) for a, b in zip(lam, beta)]


def clone(src, dst, scan=True):
    os.makedirs(dst, exist_ok=True)
    for f in ("lcs", "prep.json", "prep.npz"):
        shutil.copy(os.path.join(src, f), os.path.join(dst, f))
    if scan and os.path.isdir(os.path.join(src, "scan")) and not os.path.isdir(os.path.join(dst, "scan")):
        shutil.copytree(os.path.join(src, "scan"), os.path.join(dst, "scan"))


def fine_poles(run, P_start, threads=1):
    """damit_run.do_poles with the fine pole grid and a given start period; writes poles/poles.json in its format."""
    import damit_run as R
    import damit_wsl as W
    pdir = os.path.join(run, "poles")
    summ = os.path.join(pdir, "poles.json")
    if os.path.exists(summ):
        return json.load(open(summ))
    os.makedirs(pdir, exist_ok=True)
    n = json.load(open(os.path.join(run, "prep.json")))["n"]
    tmax = max(900, 0.1 * n)

    def one(k, lam, bet):
        open(os.path.join(pdir, f"in_{k}"), "w", newline="\n").write(
            W.par_text(lam, bet, P_start, lm=R.FINAL_LM, rows=R.FINAL_ROWS, stop=R.START_ITER))
        try:
            so, sec = W.run(run, "convexinv", f"-v -o poles/areas_{k} -p poles/par_{k} poles/in_{k} /dev/null", timeout=tmax)
            dev, dark = R.last_dev(so)
            par = W.read_par(os.path.join(pdir, f"par_{k}"))
            if not (np.isfinite(dev) and np.isfinite(par["P"])):
                raise RuntimeError("non-finite solution")
            return {**par, "idx": k, "start": [lam, bet], "dev": dev, "dark_pct": dark, "sec": sec}
        except Exception as exc:  # noqa: BLE001
            return {"idx": k, "start": [lam, bet], "error": str(exc)[:200]}

    with cf.ThreadPoolExecutor(threads) as ex:
        out = list(ex.map(lambda a: one(*a), [(k, lam, bet) for k, (lam, bet) in enumerate(fibonacci_poles(N_FINE))]))
    res = [r for r in out if "error" not in r]
    failed = [r for r in out if "error" in r]
    b = min(res, key=lambda r: r["dev"])
    open(os.path.join(pdir, "in_best"), "w", newline="\n").write(
        W.par_text(b["lambda"], b["beta"], b["P"], lm=R.FINAL_LM, rows=R.FINAL_ROWS, stop=R.POLISH_ITER,
                   phase=(b["a"], b["d"], b["k"])))
    so, sec = W.run(run, "convexinv", "-v -o poles/areas_best -p poles/par_best poles/in_best /dev/null", timeout=3 * tmax)
    dev, dark = R.last_dev(so)
    best = {**W.read_par(os.path.join(pdir, "par_best")), "idx": "best", "from_start": b["idx"], "dev": dev,
            "dark_pct": dark, "sec": sec}
    if not (np.isfinite(dev) and dev <= b["dev"]):
        best = {**b, "from_start": b["idx"], "polish_dev": dev}
    pj = {"P_start": P_start, "n_starts": N_FINE, "phase_free": W.PHASE_FREE, "start_iter": R.START_ITER,
          "polish_iter": R.POLISH_ITER, "solutions": res, "failed": failed, "best": best}
    json.dump(pj, open(summ, "w"), indent=1)
    return pj


def angle(run, areas, tag):
    from review_checks import inertia
    from review_inertia2 import vertices
    try:
        return float(inertia(vertices(os.path.join(run, "poles", areas), tag))["angle_maxI_to_spin_deg"])
    except Exception as exc:  # noqa: BLE001
        return f"error: {str(exc)[:80]}"


def job_inertia(name):
    import damit_eval as E
    import damit_mirrors as DM
    import damit_run as R
    src = os.path.join(RUNS, name)
    run = os.path.join(RUNS, name + "_fine")
    clone(src, run)
    sc = R.read_scan(run)
    ok = np.isfinite(sc["dev"])
    P0 = float(sc["P"][ok][int(np.argmin(sc["dev"][ok]))])
    pj = fine_poles(run, P0, threads=1)
    DM.do_run(run)
    mj = json.load(open(os.path.join(run, "poles", "mirrors.json")))
    thr = mj["threshold"]
    within = [{**w, "angle": angle(run, w["areas"], f"fine_{name}_{w['areas']}")} for w in mj["within"]]
    dmin = min(s["dev"] for s in pj["solutions"])
    starts = [{"idx": s["idx"], "lambda": s["lambda"], "beta": s["beta"], "ratio": (s["dev"] / dmin) ** 2,
               "dark_pct": s["dark_pct"], "angle": angle(run, f"areas_{s['idx']}", f"fine_{name}_s{s['idx']}")}
              for s in sorted(pj["solutions"], key=lambda s: s["dev"]) if (s["dev"] / dmin) ** 2 <= thr]
    m0 = os.path.join(src, "poles", "mirrors.json")
    dev12 = json.load(open(m0))["within"][0]["dev"] if os.path.exists(m0) else json.load(open(os.path.join(src, "eval.json")))["C"]["dev"]
    return name, {"P": P0, "threshold": thr, "within": within, "only_mirror_ambiguity": mj["only_mirror_ambiguity"],
                  "starts_within": starts, "n_failed": len(pj["failed"]),
                  "chi2_ratio_fine_over_12": (mj["within"][0]["dev"] / dev12) ** 2,
                  "aligned_within": [w for w in within if isinstance(w["angle"], float) and w["angle"] <= 30]}


def synsid_rows():
    S = json.load(open(os.path.join(ROOT, "synsid_check.json")))
    rows = {x["asteroid"]: x for x in S["validation"]}
    for x in S["paper"]:
        rows.setdefault(f"paper:{x['num']}", {"asteroid": f"paper:{x['num']}", "P_mpc": x["P"], "P_C": x["P_sid"],
                                              "P2_over_Porb": x["P2_over_Porb"], "expected_own": x["expected_own"]})
    return {a: rows[a] for a in SYNSID}


def arrays(a):
    kind, n = a.split(":")
    cands = ([os.path.join(ROOT, "vb", "results", n, f"{n}_arrays.npz")] if kind == "val" else
             [os.path.join(ROOT, s, n, f"{n}_arrays.npz") for s in ("results_geo", "results")])
    return np.load(next(p for p in cands if os.path.exists(p)))


def job_periodogram(a, x):
    from rebootstrap2 import OBS, power
    g = arrays(a)
    t, y, sheet = g["t"], g["y"], g["sheet"].astype(str)
    t0 = float(g["time_zero"])
    base = float(t.max() - t.min())
    P_imp = x["P_C"] + x["expected_own"] * x["P2_over_Porb"]
    f1, f2 = sorted([24.0 / x["P_mpc"], 24.0 / P_imp])
    df = 1.0 / (200 * base)
    f = np.arange(f1 - 5 / base, f2 + 5 / base, df)
    p = power(t, y, f, exact=True)
    k = int(np.argmax(p))
    near = lambda P: float(p[int(np.argmin(np.abs(f - 24.0 / P)))])
    lon = np.array([(OBS.get(s[:3], {}).get("lon") or 0.0) for s in sheet], dtype=float)
    night = np.array([f"{s}|{int(np.floor(xx + t0 + L / 360.0))}" for s, xx, L in zip(sheet, t, lon)])
    _, inv = np.unique(night, return_inverse=True)
    blocks = [np.where(inv == i)[0] for i in range(inv.max() + 1)]
    rng = np.random.default_rng(7)
    peaks = []
    for _ in range(300):
        idx = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        pb = power(t[idx], y[idx], f)
        peaks.append(24.0 / f[int(np.argmax(pb))])
    peaks = np.array(peaks)
    closer = np.abs(peaks - P_imp) < np.abs(peaks - x["P_mpc"])
    return {"P_peak_fine": float(24.0 / f[k]), "power_peak": float(p[k]), "power_at_P_mpc": near(x["P_mpc"]),
            "P_implied": P_imp, "power_at_P_implied": near(P_imp),
            "separation_resolution_elements": float(abs(24.0 / x["P_mpc"] - 24.0 / P_imp) * base),
            "boot_peak_median": float(np.median(peaks)), "boot_peak_std": float(np.std(peaks)),
            "boot_frac_closer_to_implied": float(np.mean(closer))}


def job_scan(a, x):
    import damit_run as R
    import damit_wsl as W
    kind, n = a.split(":")
    run = os.path.join(RUNS, f"{kind}_{n}_fsc")
    clone(os.path.join(RUNS, f"{kind}_{n}"), run, scan=False)
    info = json.load(open(os.path.join(run, "prep.json")))
    P_alt = x["P_mpc"] - x["expected_own"] * x["P2_over_Porb"]
    lo, hi = min(x["P_C"], P_alt) - 5 * info["dP"], max(x["P_C"], P_alt) + 5 * info["dP"]
    out = os.path.join(run, "scan", "out_periods")
    if not os.path.exists(out):
        os.makedirs(os.path.join(run, "scan"), exist_ok=True)
        open(os.path.join(run, "scan", "scan_in"), "w", newline="\n").write(
            W.scan_text(lo, hi, coef=0.2, lm=R.FINAL_LM, rows=R.FINAL_ROWS, stop=R.START_ITER))
        W.run(run, "period_scan", "scan/scan_in scan/out_periods", timeout=12 * 3600)
    sc = R.read_scan(run)
    ok = np.isfinite(sc["dev"])
    P, d = sc["P"][ok], sc["dev"][ok]
    near = lambda Pc: float(d[np.abs(P - Pc) <= 1.5 * info["dP"]].min()) if np.any(np.abs(P - Pc) <= 1.5 * info["dP"]) else None
    dC, dA = near(x["P_C"]), near(P_alt)
    return {"window": [lo, hi], "n_trials": int(ok.sum()), "P_best": float(P[np.argmin(d)]), "dev_min": float(d.min()),
            "P_C": x["P_C"], "P_alt": P_alt, "dev_near_P_C": dC, "dev_near_P_alt": dA,
            "chi2_ratio_alt_over_C": (dA / dC) ** 2 if dA and dC else None, "dP": info["dP"]}


def job_poles(a, x, which):
    import damit_eval as E
    kind, n = a.split(":")
    P_start = x["P_C"] if which == "C" else x["P_mpc"] - x["expected_own"] * x["P2_over_Porb"]
    run = os.path.join(RUNS, f"{kind}_{n}_fp{which}")
    clone(os.path.join(RUNS, f"{kind}_{n}"), run, scan=False)
    pj = fine_poles(run, P_start, threads=1)
    b = pj["best"]
    n_data = len(np.load(os.path.join(run, "prep.npz"))["y"])
    return {"P_start": P_start, "P_final": b["P"], "dev": b["dev"], "lambda": b["lambda"], "beta": b["beta"],
            "dark_pct": b.get("dark_pct"), "threshold": E.chi2_threshold(n_data - (49 + 3 + 3 - 1))}


def dispatch(task):
    mode, key, sub = task
    try:
        if mode == "inertia":
            return task, job_inertia(key)[1]
        x = synsid_rows()[key]
        return task, {"periodogram": job_periodogram, "scan": job_scan}[sub](key, x) if sub in ("periodogram", "scan") \
            else job_poles(key, x, sub[-1])
    except Exception as exc:  # noqa: BLE001
        import traceback
        return task, {"error": str(exc)[:300], "trace": traceback.format_exc()[-800:]}


if __name__ == "__main__":
    modes = [m for m in ("inertia", "synsid") if m in sys.argv] or ["inertia", "synsid"]
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 30
    tasks = []
    if "inertia" in modes:
        tasks += [("inertia", r, None) for r in INERTIA_RUNS]
    if "synsid" in modes:
        tasks += [("synsid", a, s) for a in SYNSID for s in ("periodogram", "scan", "polesC", "polesA")]
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    with cf.ProcessPoolExecutor(workers) as ex:
        for (mode, key, sub), r in ex.map(dispatch, tasks):
            res.setdefault(mode, {}).setdefault(key, {})
            if mode == "inertia":
                res[mode][key] = r
            else:
                res[mode][key][sub] = r
            print(mode, key, sub, json.dumps(r, default=float)[:600], flush=True)
            json.dump(res, open(OUT, "w"), indent=1, default=float)
    # summary of the synsid comparison: chi2 of the fine pole searches at the two sidereal candidates
    for a, v in res.get("synsid", {}).items():
        if "polesC" in v and "polesA" in v and "dev" in v["polesC"] and "dev" in v["polesA"]:
            v["chi2_ratio_alt_over_C_fine_poles"] = (v["polesA"]["dev"] / v["polesC"]["dev"]) ** 2
            v["alt_rejected"] = v["chi2_ratio_alt_over_C_fine_poles"] > v["polesC"]["threshold"]
    json.dump(res, open(OUT, "w"), indent=1, default=float)
    print("done")
