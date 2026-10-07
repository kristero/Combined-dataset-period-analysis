"""
LaTeX rows of Tables 1-5 of the revised manuscript and a digest of the numbers quoted in the text, generated
from the result files (paper_figures/rev2_results) so that nothing is transcribed by hand.
python make_tables_rev2.py  -> paper_figures/rev2_results/tables_rev2.tex and numbers_rev2.json
"""
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "paper_figures", "rev2_results")
SCR = os.environ.get("TABLE2_JSON", "")
TARGETS = [2607, 2968, 2971, 3081, 3173, 3473, 3716, 4303]
VALID = [1951, 1963, 2134, 2150]
ALCDEF = {1951: "5.302", 1963: "18.160", 2134: "4.114", 2150: "6.125", 2607: "2.81$^{a}$"}
ADOPT_S = {2968: "4.56", 2971: "4.49", 3173: "46.0", 3716: "10.47"}   # candidate shown in Table 3 (C-method period)


def jl(p):
    return json.load(open(p, encoding="utf-8"))


def summ(n):
    return jl(os.path.join(RES, "runs", "deph", "summaries", f"{n}.json"))


def sep(n):
    return jl(os.path.join(RES, "separate", f"{n}.json"))


def lc(s):
    return next(v for k, v in s.items() if k.startswith("light_curve_P="))


def s_cand(n):
    s = sep(n)
    key = ADOPT_S.get(n) or next(iter(s["candidates"]))
    return key, s["candidates"][key]


def fmt_err(x, digits=4):
    return f"{x:.{digits}f}"


def table3():
    rows = []
    for n in VALID + TARGETS:
        s = summ(n)
        L = lc(s)
        key, c = s_cand(n)
        g = s.get("combined_peak_gaussian", {})
        hw = np.ceil(g.get("period_resolution_hours", np.nan) * 1e4) / 1e4   # P^2 / Delta T, rounded up
        PC = L["period_hours"]
        note = {2968: "$^{b}$", 3173: "$^{c}$"}.get(n, "")
        rows.append(f"{n} & {ALCDEF.get(n, '-')} & {c['P']:.4f} & {c['sigma_kv']:.4f} & {c['log_xi']:.2f} & "
                    f"{PC:.4f}{note} & {hw:.4f} & {L['template_amp']:.2f} & {L['template_err']:.2f} \\\\ \\hline")
    return "\n".join(rows)


def table4():
    rows = []
    for n in VALID + TARGETS:
        s = summ(n)
        cnt = s["n_per_observatory"]
        ds = ", ".join(f"{k.replace('T08o1', 'T08o')}({v})" for k, v in sorted(cnt.items()))
        r = s["reference_fit"]
        mark = "$^{a}$" if (r["H_err"] > 1.0 or n in (1951, 3081)) else ""
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
    a = jl(os.path.join(RES, "runs", "final", "absolute_magnitudes.json"))
    rows = []
    for n in VALID + TARGETS:
        d, b = a[str(n)]["dephocus"], a[str(n)]["band"]
        rows.append(f"{n} & {d['H']:.2f} $\\pm$ {d['H_err']:.2f} & {d['G1']:.2f} $\\pm$ {d['G1_err']:.2f} & "
                    f"{d['G2']:.2f} $\\pm$ {d['G2_err']:.2f} & {d['phase_min']:.1f}-{d['phase_max']:.1f} & "
                    f"{d['offset_rms']:.2f} & {b['offset_rms']:.2f} \\\\ \\hline")
    return "\n".join(rows)


def digest():
    out = {}
    pn = jl(os.path.join(RES, "runs", "deph", "paper_numbers.json"))
    for n in VALID + TARGETS:
        s = summ(n)
        L = lc(s)
        d = {"P_C": L["period_hours"], "t0": s["time_zero_jd"], "N_C": s["n_total"], "N_after": s["n_after_outlier_removal"],
             "A_template": [L.get("template_amp"), L.get("template_err"), L.get("template_n_app")],
             "A_fourier4": [L.get("fourier_amp"), L.get("fourier_err")], "A_bins": L.get("amplitude_mag"),
             "ref": s["reference_fit"], "H_levels": s.get("H_levels"),
             "dH": {k: round(v["dH"], 3) for k, v in s.get("delta_H", {}).items()},
             "peaks": [(round(p["period_hours"], 4), round(p["power"], 3)) for p in s.get("periodogram_n=2", {}).get("peaks", [])],
             "gauss": {k: s.get("combined_peak_gaussian", {}).get(k) for k in ("gauss_mu_hours", "gauss_fwhm_hours", "period_resolution_hours")},
             "near": pn.get(str(n), {}).get("near"), "exact": pn.get(str(n), {}).get("exact")}
        sj = sep(n)
        d["S"] = {k: {x: v.get(x) for x in ("P", "sigma_kv", "M", "W_av", "xi", "log_xi", "used", "rejected")}
                  for k, v in sj["candidates"].items()}
        d["S_datasets"] = {ds: {"n_raw": r.get("n_raw"), "n": r.get("n"), "top": [(round(p["P"], 3), round(p["w"], 2)) for p in r.get("top_peaks", [])[:3]]}
                           for ds, r in sj["datasets"].items() if "n_raw" in r}
        if "oppositions" in s:
            d["oppositions"] = s["oppositions"]
        out[n] = d
    out["appendix_3173"] = jl(os.path.join(RES, "runs", "deph", "appendix_3173.json"))
    out["abs_mag"] = jl(os.path.join(RES, "runs", "final", "absolute_magnitudes.json"))
    for tag in ("band", "final"):
        out[f"periods_{tag}"] = {}
        for n in VALID + TARGETS:
            p = os.path.join(RES, "runs", tag, "summaries", f"{n}.json")
            if os.path.exists(p):
                s = jl(p)
                L = lc(s)
                out[f"periods_{tag}"][n] = [round(L["period_hours"], 5), s.get("periodogram_n=2", {}).get("peaks", [{}])[0].get("power")]
    return out


if __name__ == "__main__":
    tex = ["% Table 3 rows", table3(), "", "% Table 4 rows", table4(), "", "% Table 5 rows", table5()]
    open(os.path.join(RES, "tables_rev2.tex"), "w", encoding="utf-8").write("\n".join(tex))
    json.dump(digest(), open(os.path.join(RES, "numbers_rev2.json"), "w", encoding="utf-8"), indent=1, default=str)
    print("\n".join(tex))
