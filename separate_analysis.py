"""
Separate dataset analysis (S-method, Section 3.3 of the paper), recomputed with the corrected light time.

For every dataset of Table 2 (one observatory in one band, at least 70 measurements):
  * magnitudes reduced to 1 AU, DePhOCUS offsets per measurement (dephocus.py), light-time corrected
    times t = epoch - Delta/c (the JDc2 columns of the workbooks are not used);
  * 3-sigma outlier removal in 5 deg bins of phase angle (5 sigma below 7 deg), as for the reference fit;
  * phase-angle dependence removed with a free HG1G2 fit of the dataset itself;
  * Lomb-Scargle (astropy, nterms = 2, standard normalization) over 0.1-48 cycles/day (0.5-240 h), and
    a refined search of every candidate in an interval of +/- 3 % around its rotation frequency.
For every candidate period, each dataset gives the highest periodogram peak inside the interval (p_i, w_i).
The dataset is used if (1) the n = 2 model folded at p_i has two maxima and two minima, (2) the peak is
at least 5 sigma above the mean power of the dataset's periodogram, and (3) p_i is consistent with the
other datasets: values farther than 3 s from the median are discarded, where s is the larger of the robust
standard deviation (1.4826 x median absolute deviation) and the spacing of the yearly aliases,
P^2 / (24 x 730.5) hours (one pass). R_i^2 is the coefficient of
determination of an n = 4 Fourier fit of the dataset folded at p_i. Then (equations of Section 3.3)
  P = [sum (w_i/w_s) p_i + sum (N_i/N_s) p_i + sum (R_i^2/R_s^2) p_i] / 3,
  sigma_kv = sqrt(sum (p_i - P)^2 / ((M - 1) M)),  xi = W_av P / sigma_kv,  log xi = log10(xi / 50).

Run:  python separate_analysis.py [N ...]      -> paper_figures/separate/<N>.json
      python separate_analysis.py --figure     -> summary figure and table from the json files
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_1 = os.path.join(REPO_DIR, "..", "Asteroid_data", "12asteroidudatiicaruspubl")
DATA_2 = os.path.join(REPO_DIR, "..", "Asteroid_data", "12asteroidudatiicaruspubl2")
OUT = os.path.join(REPO_DIR, "paper_figures", "separate")
F_MIN, F_MAX = 24.0 / 240.0, 24.0 / 0.5          # cycles/day of the rotation frequency
WIDTH = 0.03                                      # +/- 3 % candidate interval
C_DAY = 4.99 / 36 / 24                            # 1 AU / c in days

# Datasets of Table 2 and the candidate periods (h) of Section 4.
S_CONFIG = {
    1951: dict(wb=(DATA_1, "1951-excelBF.xlsx"), cand=[5.302],
               ds="703G 703V C57G H45R I41g I41r M22o T05c T05o T05w T08c T08o W68o"),
    1963: dict(wb=(DATA_1, "1963-excelBF.xlsx"), cand=[18.160],
               ds="689V 703G 703V I41r M22o T05c T05o T05w T08o W68c W68o"),
    2134: dict(wb=(DATA_1, "2134-excelBF.xlsx"), cand=[4.114],
               ds="703G 703V C57G G45r I41g I41r T05c T05o T08o W68o"),
    2150: dict(wb=(DATA_1, "2150-excelBF.xlsx"), cand=[6.125],
               ds="703G 703V C57G G45r I41r M22o T05c T05o T08c T08o W68o"),
    2607: dict(wb=(DATA_1, "2607-excelBF.xlsx"), cand=[2.94],
               ds="703G 703V D29R F51w G45r G96G G96V T05c T05o T08o"),
    2968: dict(wb=(DATA_1, "2968-excelBF.xlsx"), cand=[4.56, 4.16, 3.83],
               ds="703G 703V F51w G45r G45G G96G T05c T05o T08c T08o"),
    2971: dict(wb=(DATA_1, "2971-excelBF.xlsx"), cand=[4.49, 4.80],
               ds="703G 703V C57G F51w G45r G96G I41r M22o T05c T05o T08o W68o"),
    3081: dict(wb=(DATA_1, "3081-excelBD.xlsx"), cand=[8.007],
               ds="703G 703V C57G D29R G96G G96V M22o T05c T05o T08o W68o"),
    3173: dict(wb=(DATA_2, "3173-excelBF.xlsx"), cand=[46.0, 50.0],
               ds="703G 703V C57G D29R G45r G96G I41r M22o T05c T05o T08o W68o"),
    3473: dict(wb=(DATA_2, "3473-excelBF.xlsx"), cand=[9.074],
               ds="691V 703G 703V C57G D29R G45r G96G G96V I41r M22o T05c T05o T08o W68o"),
    3716: dict(wb=(DATA_2, "3716-excelBF.xlsx"), cand=[10.47, 13.40, 18.60, 30.40],
               ds="703G 703V D29R F52w G45r G96G G96V M22o P07G T05c T05o T08o W68o"),
    4303: dict(wb=(DATA_2, "4303-excelBF.xlsx"), cand=[6.136],
               ds="703G 703V C57G D29R F51w G45r G96G G96V M22o P07G T05c T05o T08o W68o"),
}


def _sheet_name(xls: pd.ExcelFile, ds: str):
    """Sheet of a dataset; a sheet without the full columns (e.g. 2971 'T08o', a reduced copy of 'T08o1')
    is skipped in favor of the full one."""
    for s in (ds, ds + "1"):
        if s in xls.sheet_names:
            cols = list(pd.read_excel(xls, sheet_name=s, nrows=0).columns)
            if "Ph" in cols and len(cols) >= 6:
                return s
    return None


def load_dataset(num, wb_path, sheet):
    """Corrected, phase-reduced magnitudes of one dataset (t in JD, y in mag)."""
    from Asteroid_pickle import DatasetGenerator
    import dephocus
    df = pd.read_excel(wb_path, sheet_name=sheet)
    if "Ph" not in df.columns or "magred" not in df.columns:
        return None
    df = df.dropna(subset=["magred", "Ph"])
    if "mag" not in df.columns:          # some sheets have no header on the magnitude column (column 2)
        df = df.rename(columns={df.columns[2]: "mag"})
    if "epoch" not in df.columns:
        df = df.rename(columns={df.columns[1]: "epoch"})
    mag = pd.to_numeric(df["mag"], errors="coerce").to_numpy(float)
    delta = df.iloc[:, 3].to_numpy(float)
    rdist = df.iloc[:, 4].to_numpy(float)
    ph = df["Ph"].to_numpy(float)
    t = df["epoch"].to_numpy(float) - delta * C_DAY
    d, _, _ = dephocus.sheet_offsets(num, sheet, df["epoch"].to_numpy(float), mag)
    H = mag - 5 * np.log10(rdist * delta) + d
    gen = DatasetGenerator(os.path.dirname(wb_path) + os.sep, os.path.basename(wb_path), num, [sheet],
                           base_dir=REPO_DIR)
    ph_k, H_k, rem = gen.removeOutliers(ph, H, 3.0)
    t_k = np.delete(t, rem)
    try:
        res = gen.fit(ph_k, H_k, method="HG1G2")
        model = gen.hg1g2_phase_function(ph_k, res.params["H"].value, res.params["G1"].value, res.params["G2"].value)
        y = H_k - (model - res.params["H"].value)
        phase_model = "HG1G2"
    except Exception:
        c = np.polyfit(ph_k, H_k, 2)
        y = H_k - np.polyval(c, ph_k) + np.polyval(c, 0.0)
        phase_model = "quadratic"
    return {"t": np.asarray(t_k), "y": np.asarray(y), "n_raw": int(len(df)), "n": int(len(y)),
            "phase_model": phase_model, "n_removed": int(len(rem))}


def _n_extrema(ls, f):
    ph = np.linspace(0, 1, 800, endpoint=False)
    m = ls.model(ph / f, f)
    m_c = np.concatenate([m, m[:2]])
    return len(find_peaks(m_c)[0]), len(find_peaks(-m_c)[0])


def r2_fourier(t, y, f, nterms=4):
    ls = LombScargle(t, y, nterms=nterms)
    m = ls.model(t, f)
    return float(1 - np.sum((y - m) ** 2) / np.sum((y - np.mean(y)) ** 2))


def analyse_dataset(args):
    num, wb_path, ds, sheet, cands = args
    d = load_dataset(num, wb_path, sheet)
    if d is None:
        return ds, {"skipped": "no phase-angle column"}
    t, y = d["t"] - d["t"].min(), d["y"]
    base = float(t.max())
    ls = LombScargle(t, y, nterms=2)
    # full search (top peaks, for information)
    f = np.arange(F_MIN, F_MAX, 1.0 / (10 * base))
    p = ls.power(f, method="fastchi2", assume_regular_frequency=True)
    pk, _ = find_peaks(p, distance=5)
    pk = pk[np.argsort(p[pk])[::-1]][:8]
    out = {k: v for k, v in d.items() if k not in ("t", "y")}
    out.update(baseline_days=base, mean_power=float(np.mean(p)), std_power=float(np.std(p)),
               top_peaks=[{"P": float(24 / f[i]), "w": float(p[i])} for i in pk], candidates={})
    for P_c in cands:
        fc = 24.0 / P_c
        fg = np.arange(fc * (1 - WIDTH), fc * (1 + WIDTH), 1.0 / (50 * base))
        pg = ls.power(fg, method="fastchi2", assume_regular_frequency=True)
        loc, _ = find_peaks(pg)
        if len(loc) == 0:
            out["candidates"][str(P_c)] = {"found": False}
            continue
        i = int(loc[np.argmax(pg[loc])])
        f_best = float(fg[i])
        n_max, n_min = _n_extrema(ls, f_best)
        out["candidates"][str(P_c)] = {
            "found": True, "p": 24.0 / f_best, "w": float(pg[i]), "N": d["n"],
            "R2": r2_fourier(t, y, f_best, 4), "n_max": n_max, "n_min": n_min,
            "sigma_above_mean": float((pg[i] - np.mean(p)) / np.std(p)),
        }
    return ds, out


def weighted_period(rows):
    p = np.array([r["p"] for r in rows])
    w = np.array([r["w"] for r in rows])
    N = np.array([r["N"] for r in rows], float)
    R2 = np.array([max(r["R2"], 0.0) for r in rows])
    P = (np.sum(w / w.sum() * p) + np.sum(N / N.sum() * p) + np.sum(R2 / R2.sum() * p)) / 3.0
    M = len(p)
    s_kv = float(np.sqrt(np.sum((p - P) ** 2) / ((M - 1) * M))) if M > 1 else np.nan
    W_av = float(np.mean(w))
    xi = W_av * P / s_kv if s_kv > 0 else np.nan
    return {"P": float(P), "sigma_kv": s_kv, "M": M, "W_av": W_av, "W_sum": float(w.sum()),
            "xi": float(xi), "log_xi": float(np.log10(xi / 50.0)), "N_sum": int(N.sum())}


def select(rows):
    """Criteria of Section 3.3 and the weighted period of the datasets that pass them.
    (1) two maxima and two minima of the n = 2 model, (2) peak at least 5 sigma above the mean power,
    (3) consistency: |p_i - median| <= 3 s, with s = max(1.4826 MAD, dP_year), where dP_year = P^2 / (24 * 730.5)
        is the spacing of the yearly aliases of a double-peaked light curve (one pass)."""
    failed = {}
    for r in rows:
        if r["n_max"] < 2 or r["n_min"] < 2:
            failed[r["ds"]] = "single-peaked"
        elif r["sigma_above_mean"] < 5.0:
            failed[r["ds"]] = "below 5 sigma"
    cand = [r for r in rows if r["ds"] not in failed]
    p = np.array([r["p"] for r in cand])
    if len(p) > 2:
        med = float(np.median(p))
        s = max(1.4826 * float(np.median(np.abs(p - med))), med ** 2 / (24.0 * 730.5))
        keep = np.abs(p - med) <= 3 * s
    else:
        keep = np.ones(len(p), bool)
    for r, k in zip(cand, keep):
        if not k:
            failed[r["ds"]] = "inconsistent"
    used = [r for r, k in zip(cand, keep) if k]
    res = weighted_period(used) if len(used) > 1 else {}
    res["used"] = [r["ds"] for r in used]
    res["rejected"] = failed
    res["per_dataset"] = rows
    return res


def reselect(nums):
    """Re-apply the selection to existing json files (no new periodograms)."""
    for n in nums:
        path = os.path.join(OUT, f"{n}.json")
        s = json.load(open(path))
        for P_c, res in s["candidates"].items():
            s["candidates"][P_c] = select(res["per_dataset"])
            r = s["candidates"][P_c]
            print(n, P_c, {k: r.get(k) for k in ("P", "sigma_kv", "M", "W_av", "log_xi")}, "rejected", r["rejected"])
        json.dump(s, open(path, "w"), indent=1)


def run(num):
    cfg = S_CONFIG[num]
    wb_path = os.path.join(*cfg["wb"])
    xls = pd.ExcelFile(wb_path)
    jobs = []
    missing = []
    for ds in cfg["ds"].split():
        s = _sheet_name(xls, ds)
        if s is None:
            missing.append(ds)
        else:
            jobs.append((num, wb_path, ds, s, cfg["cand"]))
    with ProcessPoolExecutor(max_workers=int(os.environ.get("NPROC", "4"))) as ex:
        results = dict(ex.map(analyse_dataset, jobs))
    summary = {"asteroid": num, "workbook": wb_path, "missing_sheets": missing, "datasets": results,
               "candidates": {}}
    for P_c in cfg["cand"]:
        rows = []
        for ds, r in results.items():
            c = r.get("candidates", {}).get(str(P_c))
            if c and c["found"]:
                rows.append(dict(c, ds=ds))
        summary["candidates"][str(P_c)] = select(rows)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{num}.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    for P_c, res in summary["candidates"].items():
        print(num, P_c, {k: res.get(k) for k in ("P", "sigma_kv", "M", "W_av", "log_xi")}, "rejected", res["rejected"])
    return summary


def adopted_candidate(summary):
    """Candidate with the highest log xi (Section 3.3)."""
    c = {k: v for k, v in summary["candidates"].items() if "log_xi" in v}
    return max(c, key=lambda k: c[k]["log_xi"])


SHOWN = {2968: "4.56", 2971: "4.49", 3173: "46.0", 3716: "10.47"}   # candidates discussed in Section 4


def figure(nums=(2607, 2968, 2971, 3081, 3173, 3473, 3716, 4303), out_name="S_method_summary"):
    """Summary of the S-method (paper figure): one row per asteroid, one column per dataset. The number in a
    cell is the period of that dataset (hours), the color its peak power w_i. Datasets that failed the
    criteria are gray. The last column is the weighted period P. For asteroids with several candidates,
    the candidate of the period adopted in Section 4 is shown (SHOWN)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
    S = {n: json.load(open(os.path.join(OUT, f"{n}.json"))) for n in nums}
    order = ["703G", "703V", "691V", "689V", "D29R", "F51w", "F52w", "G45G", "G45r", "G96G", "G96V", "T05c", "T05o",
             "T08c", "T08o", "C57G", "I41r", "M22o", "W68o", "P07G", "H45R"]
    used_cols = [c for c in order if any(c in S[n]["datasets"] for n in nums)]
    cols = used_cols + ["Final"]
    cmap = plt.get_cmap("YlOrBr")
    norm = mcolors.Normalize(vmin=0.0, vmax=0.9)
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(nums) + 1.0), constrained_layout=True)
    for i, n in enumerate(nums):
        key = SHOWN.get(n, adopted_candidate(S[n]))
        cand = S[n]["candidates"][key]
        per = {r["ds"]: r for r in cand["per_dataset"]}
        for j, c in enumerate(cols):
            if c == "Final":
                ax.add_patch(plt.Rectangle((j, i), 1, 1, color="#e9967a"))
                ax.text(j + 0.5, i + 0.5, f"{cand['P']:.2f}", ha="center", va="center", fontsize=6.5)
                continue
            r = per.get(c)
            if r is None:
                continue
            ok = c in cand["used"]
            fc = cmap(norm(r["w"])) if ok else "#d9d9d9"
            ax.add_patch(plt.Rectangle((j, i), 1, 1, color=fc))
            tc = "white" if (ok and r["w"] > 0.55) else "black"
            ax.text(j + 0.5, i + 0.5, f"{r['p']:.2f}", ha="center", va="center", fontsize=6, color=tc)
    ax.set_xlim(0, len(cols))
    ax.set_ylim(len(nums), 0)
    ax.set_xticks(np.arange(len(cols)) + 0.5)
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(nums)) + 0.5)
    ax.set_yticklabels([str(n) for n in nums])
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Asteroid")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, label="Peak power $w_i$", pad=0.01)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{out_name}.{ext}"), dpi=300)
    rows = []
    for n in list(nums) + [1951, 1963, 2134, 2150]:
        if not os.path.exists(os.path.join(OUT, f"{n}.json")):
            continue
        s = json.load(open(os.path.join(OUT, f"{n}.json")))
        for k, v in s["candidates"].items():
            rows.append({"asteroid": n, "candidate": k, "adopted": k == adopted_candidate(s),
                         **{x: v.get(x) for x in ("P", "sigma_kv", "M", "W_av", "xi", "log_xi", "N_sum")},
                         "used": " ".join(v.get("used", [])),
                         "rejected": "; ".join(f"{a}:{b}" for a, b in v.get("rejected", {}).items())})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "S_method_table.csv"), index=False)


if __name__ == "__main__":
    if "--figure" in sys.argv:
        figure()
        sys.exit()
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    nums = [int(a) for a in args] if args else list(S_CONFIG)
    if "--reselect" in sys.argv:
        reselect(nums)
        sys.exit()
    for n in nums:
        run(n)
