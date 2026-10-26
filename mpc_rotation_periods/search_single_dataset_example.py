"""Find a clear example for the separate-dataset figure (Fig. 2): a single ground-based dataset reported to 0.01 mag,
reduced with the separate-dataset method (Eqs. 3-5, alpha > 7 deg), whose periodogram has one distinct peak family
at the adopted (quality A) period. Metric: power of the adopted family (P, P/2, 2P, incl. its comb teeth) divided
by the strongest peak outside it, both from exact-chi2 refinements of the fastchi2 scan peaks.
Output: single_dataset_candidates.json (ranked)."""
import json
import multiprocessing as mp
import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
C_AU_PER_DAY = 173.1446326846693


def reduce_separate(num, ds):
    import pandas as pd
    obs = pd.read_csv(os.path.join(ROOT, "data", "obs_tables", f"{num}_obs.csv"))
    d = obs[(obs["dataset"] == ds) & (obs["alpha"] > 7.0)].sort_values("jd")
    if len(d) < 150:
        return None
    D0, r0, a0 = d["delta"].iloc[0], d["r"].iloc[0], d["alpha"].iloc[0]
    m_r = d["mag_f"] - 5 * np.log10((d["delta"] / D0) * (d["r"] / r0))
    k_lin = np.polyfit(d["alpha"], m_r, 1)[0]
    y = (m_r - (d["alpha"] - a0) * k_lin).to_numpy()
    t_abs = (d["jd"] + (D0 - d["delta"]) / C_AU_PER_DAY).to_numpy()
    # 3-sigma clipping about the median, as a light cleaning of single-dataset outliers
    keep = np.abs(y - np.median(y)) <= 3 * 1.4826 * np.median(np.abs(y - np.median(y)))
    return t_abs[keep] - t_abs[keep][0], y[keep]


def in_family(Pc, P, tol):
    return any(abs(Pc - r * P) <= tol * Pc for r in (1.0, 2.0, 0.5))


def evaluate(job):
    num, ds, P_ad = job
    from astropy.timeseries import LombScargle
    from scipy.signal import find_peaks
    red = reduce_separate(num, ds)
    if red is None:
        return None
    t, y = red
    ls = LombScargle(t, y, nterms=2)
    f = ls.autofrequency(samples_per_peak=10, minimum_frequency=24 / 240.0, maximum_frequency=24 / 0.8)
    p = ls.power(f, method="fastchi2", assume_regular_frequency=True)
    pk, _ = find_peaks(p)
    top = pk[np.argsort(p[pk])[::-1][:40]]
    base = t.max() - t.min()
    res = []
    for i in top:
        g = np.linspace(f[i] - 2 / base, f[i] + 2 / base, 81)
        pe = ls.power(g, method="chi2")
        k = int(np.argmax(pe))
        res.append((24 / g[k], float(pe[k])))
    fam = [pw for P, pw in res if in_family(P, P_ad, 0.02)]
    other = [pw for P, pw in res if not in_family(P, P_ad, 0.02)]
    if not fam:
        return {"asteroid": num, "dataset": ds, "n": len(t), "dominance": 0.0}
    best_P, best_pw = max(res, key=lambda r: r[1])
    fam_best = max(fam)
    P_fam = max(((P, pw) for P, pw in res if in_family(P, P_ad, 0.02)), key=lambda r: r[1])[0]
    return {"asteroid": num, "dataset": ds, "n": len(t), "baseline_yr": base / 365.25,
            "dominance": fam_best / max(other) if other else 99.0, "family_is_global_max": bool(in_family(best_P, P_ad, 0.02)),
            "P_best_family": P_fam, "P_adopted": P_ad, "power_family": fam_best,
            "P_is_full_period": bool(abs(P_fam - P_ad) <= 0.01 * P_ad),
            "top_peaks": sorted(res, key=lambda r: -r[1])[:6]}


def main():
    A = json.load(open(os.path.join(ROOT, "adopted.json"), encoding="utf-8"))
    prec = json.load(open(os.path.join(ROOT, "data", "dataset_precision.json"), encoding="utf-8"))["by_asteroid"]
    jobs = []
    for num, ad in A.items():
        if ad["quality"] != "A" or ad["P"] > 240:
            continue
        for ds, frac01 in prec[num].items():
            if frac01 < 0.5 and not ds.startswith("C57"):   # reported to 0.01 mag
                jobs.append((int(num), ds, ad["P"]))
    print(len(jobs), "datasets to evaluate", flush=True)
    with mp.get_context("spawn").Pool(16) as pool:
        out = [r for r in pool.imap_unordered(evaluate, jobs) if r]
    out.sort(key=lambda r: -r["dominance"])
    json.dump(out, open(os.path.join(ROOT, "single_dataset_candidates.json"), "w"), indent=1)
    for r in out[:15]:
        print(f"({r['asteroid']}) {r['dataset']}: n={r['n']}, dominance {r['dominance']:.2f}, global max in family "
              f"{r.get('family_is_global_max')}, P_family {r.get('P_best_family', 0):.4f} vs adopted {r.get('P_adopted', 0):.4f}, "
              f"full period {r.get('P_is_full_period')}")


if __name__ == "__main__":
    main()
