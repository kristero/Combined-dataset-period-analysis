"""
LaTeX rows of Tables 1-5 of the manuscript (second revision) and a digest of the numbers quoted in the text,
generated from the result files in paper_figures/rev3_results and from the rebuilt workbooks
(../Asteroid_data/rev3, build_workbooks.py), so that nothing is transcribed by hand.

python make_tables_rev3.py  -> paper_figures/rev3_results/tables_rev3.tex and numbers_rev3.json
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from astropy.time import Time

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "paper_figures", "rev3_results")
WB = os.path.join(HERE, "..", "Asteroid_data", "rev3")
TARGETS = [2607, 2968, 2971, 3081, 3173, 3473, 3716, 4303]
VALID = [1951, 1963, 2134, 2150]
ALCDEF = {1951: "5.302", 1963: "18.160", 2134: "4.114", 2150: "6.125", 2607: "2.81$^{a}$"}
ADOPT_S = {2968: "4.56", 2971: "4.49", 3173: "46.0", 3716: "10.47"}   # S-method candidate closest to the C-method period
NAMES = {1951: ("Lick", "1949 OA", "0.30579"), 1963: ("Bezovec", "1975 CB", "0.93243"),
         2134: ("Dennispalm", "1976 YB", "1.0979"), 2150: ("Nyctimene", "1977 TA", "0.84457"),
         2607: ("Yakutia", "1977 NR", "0.82911"), 2968: ("Iliya", "1978 QJ", "0.62344"),
         2971: ("Mohr", "1980 YL", "0.99639"), 3081: ("Martinuboh", "1971 UP", "0.96276"),
         3173: ("McNaught", "1981 WY", "0.73088"), 3473: ("Sapporo", "A924 EG", "0.99465"),
         3716: ("Petzval", "1980 TG", "0.86715"), 4303: ("Savitskij", "1973 SZ3", "0.89591")}
OBS = {"689": "U.S. Naval Observatory, Flagstaff", "691": "Steward Observatory, Kitt Peak-Spacewatch",
       "703": "Catalina Sky Survey", "704": "Lincoln Laboratory ETS, New Mexico (LINEAR)",
       "C57": "Transiting Exoplanet Survey Satellite", "D29": "Purple Mountain Observatory, XuYi Station",
       "F51": "Pan-STARRS 1, Haleakala", "F52": "Pan-STARRS 2, Haleakala",
       "G45": "Space Surveillance Telescope, Atom Site", "G96": "Mt. Lemmon Survey",
       "H45": "Arkansas Sky Obs., Petit Jean Mountain South", "I41": "Zwicky Transient Facility",
       "M22": "ATLAS South Africa, Sutherland", "P07": "Space Surveillance Telescope, HEH Station",
       "T05": "ATLAS Haleakala", "T08": "ATLAS Mauna Loa", "W68": "ATLAS Chile, Rio Hurtado",
       "Y00": "SONEAR Observatory, Oliveira"}


def jl(p):
    return json.load(open(p, encoding="utf-8"))


def summ(n):
    return jl(os.path.join(RES, "runs", "rev3", "summaries", f"{n}.json"))


def sep(n):
    return jl(os.path.join(RES, "separate", f"{n}.json"))


def lc(s):
    return next(v for k, v in s.items() if k.startswith("light_curve_P="))


def s_cand(n):
    s = sep(n)
    key = ADOPT_S.get(n) or next(iter(s["candidates"]))
    return key, s["candidates"][key]


def sig(x):
    """sigma_kv with two significant digits below 0.001 h, otherwise four decimals."""
    return f"{x:.5f}" if x < 0.001 else f"{x:.4f}"


def workbook_datasets(n):
    xls = pd.ExcelFile(os.path.join(WB, f"{n}-excelRB.xlsx"))
    out = {}
    for s in xls.sheet_names:
        df = xls.parse(s).dropna(subset=["epoch", "mag", "magred", "Ph"])   # the rows the pipeline can use
        out[s.replace("T08o1", "T08o")] = df
    return out


def table1_2():
    rows2, per_code = [], defaultdict(lambda: [set(), 0])
    for n in VALID + TARGETS:
        ds = workbook_datasets(n)
        ep = np.concatenate([d["epoch"].to_numpy(float) for d in ds.values()])
        ph = np.concatenate([pd.to_numeric(d["Ph"], errors="coerce").to_numpy(float) for d in ds.values()])
        t1, t2 = Time(ep.min(), format="jd").iso[:7], Time(ep.max(), format="jd").iso[:7]
        lst = ", ".join(f"{k}({len(v)})" for k, v in sorted(ds.items()))
        name, desig, moid = NAMES[n]
        rows2.append(f"{n} & {name} & {desig} & {moid} & {lst} & {sum(len(v) for v in ds.values())} & "
                     f"{t1.replace('-', '.')}-{t2.replace('-', '.')} & {np.nanmin(ph):.1f}-{np.nanmax(ph):.1f} \\\\ \\hline")
        for k, v in ds.items():
            per_code[k[:3]][0].add(k[3:])
            per_code[k[:3]][1] += len(v)
    order = sorted(per_code, key=lambda c: (not c[0].isdigit(), c))
    rows1 = [f"{c} & {OBS.get(c, '?')} & {', '.join(sorted(per_code[c][0], key=lambda b: (b.lower(), b)))} & "
             f"{per_code[c][1]} \\\\ \\hline" for c in order]
    return "\n".join(rows1), "\n".join(rows2)


def table3():
    rows = []
    for n in VALID + TARGETS:
        s = summ(n)
        L = lc(s)
        key, c = s_cand(n)
        g = s.get("combined_peak_gaussian", {})
        hw = np.ceil(g.get("period_resolution_hours", np.nan) * 1e4) / 1e4   # P^2 / Delta T, rounded up
        PC = L["period_hours"]
        note_c = {2968: "$^{b}$", 3173: "$^{c}$"}.get(n, "")
        note_s = {3173: "$^{c}$", 3716: "$^{d}$"}.get(n, "")
        rows.append(f"{n} & {ALCDEF.get(n, '-')} & {c['P']:.4f}{note_s} & {sig(c['sigma_kv'])} & {c['log_xi']:.2f} & "
                    f"{PC:.4f}{note_c} & {hw:.4f} & {L['template_amp']:.2f} & {L['template_err']:.2f} \\\\ \\hline")
    return "\n".join(rows)


def at_limit(r):
    return (r["H_err"] > 1.0 or r["G1_err"] > abs(r["G1"]) + 0.3 or r["G1"] < 0.005 or r["G2"] < 0.005
            or r["G1"] + r["G2"] > 0.995)


def table4():
    rows = []
    for n in VALID + TARGETS:
        s = summ(n)
        cnt = s["n_per_observatory"]
        ds = ", ".join(f"{k.replace('T08o1', 'T08o')}({v})" for k, v in sorted(cnt.items()))
        r = s["reference_fit"]
        mark = "$^{a}$" if (at_limit(r) or n == 1951) else ""
        if mark:
            Hs, G1s, G2s = f"{r['H']:.2f}{mark}", f"{r['G1']:.2f}{mark}", f"{r['G2']:.2f}{mark}"
        else:
            Hs = f"{r['H']:.2f} $\\pm$ {r['H_err']:.2f}"
            G1s = f"{r['G1']:.2f} $\\pm$ {r['G1_err']:.2f}"
            G2s = f"{r['G2']:.2f} $\\pm$ {r['G2_err']:.2f}"
        y1, y2 = [x.split(" ")[0] for x in s["date_range_utc"]]
        span = f"{y1[:7].replace('-', '.')}-{y2[:7].replace('-', '.')}"
        rows.append(f"{n} & T08o & {ds} & {sum(cnt.values())} & {Hs} & {G1s} & {G2s} & "
                    f"{s['time_zero_jd']:.2f} & {span} \\\\ \\hline")
    return "\n".join(rows)


def table5():
    a = jl(os.path.join(RES, "runs", "rev3", "absolute_magnitudes.json"))
    rows = []
    for n in VALID + TARGETS:
        d, b = a[str(n)]["dephocus"], a[str(n)]["band"]
        rows.append(f"{n} & {d['H']:.2f} $\\pm$ {d['H_err']:.2f} & {d['G1']:.2f} $\\pm$ {d['G1_err']:.2f} & "
                    f"{d['G2']:.2f} $\\pm$ {d['G2_err']:.2f} & {d['phase_min']:.1f}-{d['phase_max']:.1f} & "
                    f"{d['offset_rms']:.2f} & {b['offset_rms']:.2f} \\\\ \\hline")
    return "\n".join(rows)


def digest():
    out = {}
    pn_path = os.path.join(RES, "runs", "rev3", "paper_numbers.json")
    pn = jl(pn_path) if os.path.exists(pn_path) else {}
    for n in VALID + TARGETS:
        try:
            s = summ(n)
            sj = sep(n)
        except FileNotFoundError as exc:
            print("missing results:", n, exc.filename)
            continue
        L = lc(s)
        d = {"P_C": L["period_hours"], "t0": s["time_zero_jd"], "N_C": s["n_total"], "N_after": s["n_after_outlier_removal"],
             "datasets": s["n_per_observatory"],
             "A_template": [L.get("template_amp"), L.get("template_err"), L.get("template_n_app")],
             "A_fourier4": [L.get("fourier_amp"), L.get("fourier_err")],
             "ref": s["reference_fit"], "H_levels": s.get("H_levels"),
             "redchi_ratio": {k: round(v["redchi"] / s["delta_H"]["T08o1" if n == 2971 else "T08o"]["redchi"], 2)
                              for k, v in s.get("delta_H", {}).items()},
             "dH": {k: round(v["dH"], 3) for k, v in s.get("delta_H", {}).items()},
             "peaks": [(round(p["period_hours"], 4), round(p["power"], 3)) for p in s.get("periodogram_n=2", {}).get("peaks", [])],
             "gauss": {k: s.get("combined_peak_gaussian", {}).get(k) for k in ("gauss_mu_hours", "gauss_fwhm_hours", "period_resolution_hours")},
             "baseline_days": s.get("baseline_days"), "date_range": s.get("date_range_utc"),
             "near": pn.get(str(n), {}).get("near"), "exact": pn.get(str(n), {}).get("exact")}
        d["S"] = {k: {x: v.get(x) for x in ("P", "sigma_kv", "M", "W_av", "xi", "log_xi", "used", "rejected")}
                  for k, v in sj["candidates"].items()}
        d["S_datasets"] = {ds: {"n_raw": r.get("n_raw"), "n": r.get("n"), "top": [(round(p["P"], 3), round(p["w"], 2)) for p in r.get("top_peaks", [])[:3]]}
                           for ds, r in sj["datasets"].items() if "n_raw" in r}
        for k in ("oppositions", "single_observatory_T08o", "spectral_window"):
            if k in s:
                d[k] = s[k]
        out[n] = d
    for k, f in (("appendix_3173", "appendix_3173.json"), ("abs_mag", "absolute_magnitudes.json")):
        p = os.path.join(RES, "runs", "rev3", f)
        if os.path.exists(p):
            out[k] = jl(p)
    # Variant runs: band-dependent G1, G2 (rev3bp) and band offsets instead of DePhOCUS (rev3band).
    for tag, key in (("rev3bp", "periods_band_phase"), ("rev3band", "periods_band")):
        p = os.path.join(RES, "runs", tag, "summaries")
        if not os.path.isdir(p):
            continue
        out[key] = {}
        for n in VALID + TARGETS:
            f = os.path.join(p, f"{n}.json")
            if os.path.exists(f):
                s = jl(f)
                if "error" in s:
                    out[key][n] = s["error"]
                    continue
                P = lc(s)["period_hours"]
                out[key][n] = {"P": round(P, 5), "dP": round(P - out[n]["P_C"], 5) if n in out else None,
                               "band_slopes": {b: (v.get("fitted"), round(v["G1"], 2), round(v["G2"], 2))
                                               for b, v in s.get("band_slopes", {}).items()}}
    return out


if __name__ == "__main__":
    if "--digest" in sys.argv:      # numbers only (works with partial results)
        json.dump(digest(), open(os.path.join(RES, "numbers_rev3.json"), "w", encoding="utf-8"), indent=1, default=str)
        sys.exit()
    t1, t2 = table1_2()
    tex = ["% Table 1 rows", t1, "", "% Table 2 rows", t2, "", "% Table 3 rows", table3(), "",
           "% Table 4 rows", table4(), "", "% Table 5 rows", table5()]
    open(os.path.join(RES, "tables_rev3.tex"), "w", encoding="utf-8").write("\n".join(tex))
    json.dump(digest(), open(os.path.join(RES, "numbers_rev3.json"), "w", encoding="utf-8"), indent=1, default=str)
    print("\n".join(tex))
