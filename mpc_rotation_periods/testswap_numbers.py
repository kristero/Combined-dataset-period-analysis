"""
Numbers of the manuscript that depend on the sample, recomputed for the revised test sample (retired.json; six new
test asteroids). Every value is derived from the pipeline outputs; nothing is typed in. Sections whose inputs are
missing are skipped with a note. Writes testswap_numbers.json and prints a readable summary.
"""
import json
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
J = lambda *p: json.load(open(os.path.join(ROOT, *p), encoding="utf-8"))
EX = lambda *p: os.path.exists(os.path.join(ROOT, *p))
A = J("adopted.json")
NUMS = sorted(A, key=int)
TEST = [n for n in NUMS if A[n].get("table") == 2]
MAIN = [n for n in NUMS if A[n].get("table") in (3, 4)]
WITH_P = [n for n in NUMS if A[n].get("status") != "undetermined"]
SB = J("data", "sbdb_phys_orbit.json")
OUT = {}


def res(n):
    return J("results", n, f"{n}_result.json")


def info(n, d="data"):
    return J(d, "obs_tables", f"{n}_info.json")


def section(name):
    def deco(fn):
        try:
            OUT[name] = fn()
        except FileNotFoundError as e:
            OUT[name] = {"skipped": f"missing input: {e.filename}"}
        except Exception as e:  # noqa: BLE001
            OUT[name] = {"error": repr(e)[:300]}
        return fn
    return deco


@section("sample")
def _():
    moid = {n: float(SB[n]["orbit"]["moid"]) for n in NUMS}
    cls = Counter(SB[n]["object"]["orbit_class"]["code"] for n in NUMS)
    mca = [n for n in NUMS if SB[n]["object"]["orbit_class"]["code"] == "MCA"]
    porb = {n: float(next(e["value"] for e in SB[n]["orbit"]["elements"] if e["name"] == "per")) / 365.25 for n in NUMS}
    psyn = {n: 1 / abs(1 - 1 / porb[n]) for n in NUMS}
    return {"n": len(NUMS), "n_test": len(TEST), "n_main": len(MAIN), "test": TEST,
            "moid_range": [min(moid.values()), max(moid.values())],
            "moid_outside_0.6_1.0": {n: m for n, m in moid.items() if not 0.6 <= m <= 1.0},
            "orbit_classes": dict(cls), "mars_crossers": mca,
            "synodic_orbital_period_yr": [min(psyn.values()), max(psyn.values()), float(np.median(list(psyn.values())))]}


@section("data_volume")
def _():
    rec = sum(info(n)["n_records"] for n in NUMS)
    mag = sum(info(n)["n_with_mag"] for n in NUMS)
    kept70 = sum(info(n, "data_unsplit")["n_kept"] for n in NUMS)
    # retrieval: data/fetch_summary.json holds the last fetch (the six new test asteroids); the other 52 asteroids
    # were retrieved on 2026-09-28 (manuscript, unchanged data)
    fs = J("data", "fetch_summary.json")
    dates = {fs["retrieved_utc"][:10]: sorted(fs["results"], key=int), "2026-09-28": [n for n in NUMS if n not in fs["results"]]}
    nores = ntot = 0
    first, last = [], []
    for n in NUMS:
        o = pd.read_csv(os.path.join(ROOT, "data", "obs_tables", f"{n}_obs.csv"), usecols=["jd", "rmsmag_f"])
        nores += int(o["rmsmag_f"].isna().sum())
        ntot += len(o)
        first.append(o["jd"].min())
        last.append(o["jd"].max())
    from astropy.time import Time
    yr = lambda jd: Time(jd, format="jd").datetime.year
    ZS = J("zp", "zp_split.json")
    split = drop = 0
    for n in NUMS:
        for ds, L in ZS.get(n, {}).items():
            if L["pre"] >= 30 and L["post"] >= 30:
                split += 1
            for part in ("pre", "post"):
                if 0 < L[part] < 30:
                    drop += L[part]
    return {"records": rec, "with_mag": mag, "retrieval_dates": dict(dates), "kept_ge70_before_split": kept70,
            "frac_no_uncertainty": nores / ntot, "split_datasets": split, "dropped_split_measurements": drop,
            "first_obs_year_range": [yr(min(first)), yr(max(first))], "last_obs_year_range": [yr(min(last)), yr(max(last))]}


@section("analysis")
def _():
    ATLAS = {"T05", "T08", "M22", "W68", "R17"}
    groups = {"ATLAS": ATLAS, "TESS": {"C57"}, "CSS+MLS": {"703", "G96"}, "SST": {"G45", "P07"},
              "PanSTARRS": {"F51", "F52"}, "PMO": {"D29"}, "ZTF": {"I41"}}
    nds, nmeas, base, stns, cnt = [], [], [], set(), Counter()
    scr_all = scr_rm = scr_chi = scr_bnd = 0
    tess_out, tess_av, tess_used, linear = [], [], [], [0, 0]
    alpha_max = {}
    nast = defaultdict(set)
    for n in NUMS:
        r = res(n)
        u = r["datasets_used"]
        nds.append(len(u))
        nmeas.append(r["n_after_outlier_removal"])
        base.append(r["baseline_days"] / 365.25)
        stns |= {d[:3] for d in u}
        for d in u:
            nast[d[:3]].add(n)
        for d, k in r["n_per_dataset"].items():
            cnt[d[:3]] += k
        for s in r["screening"]:
            scr_all += 1
            if s["dataset"].startswith("704"):
                linear[1] += 1
            if not s["keep"]:
                scr_rm += 1
                reason = s.get("reason", "")
                scr_chi += "chi2" in reason
                scr_bnd += "boundary" in reason
                if s["dataset"].startswith("C57"):
                    tess_out.append(n)
                if s["dataset"].startswith("704"):
                    linear[0] += 1
        alpha_max[n] = max(s["alpha_max"] for s in r["screening"] if s["keep"])
        if any(d.startswith("C57") for d in r["datasets_available"]):
            tess_av.append(n)
        if any(d.startswith("C57") for d in u):
            tess_used.append(n)
    tot = sum(nmeas)
    return {"datasets_screened": scr_all, "removed": scr_rm, "removed_chi2": scr_chi, "removed_boundary": scr_bnd,
            "tess_screened_out": tess_out, "linear_removed_of": linear, "datasets_used": sum(nds),
            "observatories": len(stns), "datasets_per_asteroid": [min(nds), max(nds)],
            "N_range": [min(nmeas), max(nmeas)], "N_median": float(np.median(nmeas)), "N_total": tot,
            "baseline_yr": [min(base), max(base), float(np.median(base))],
            "shares_pct": {g: 100 * sum(cnt[s] for s in S) / tot for g, S in groups.items()},
            "tess_available": len(tess_av), "tess_used": len(tess_used),
            "alpha_max_range": [min(alpha_max.values()), max(alpha_max.values())],
            "alpha_max_mca": {n: alpha_max[n] for n in NUMS if SB[n]["object"]["orbit_class"]["code"] == "MCA"},
            "table1_N_ast": {s: len(v) for s, v in sorted(nast.items())}}


def fmt_p(P, s):
    exp = int(np.floor(np.log10(s)))
    dec = min(max(0, -exp + 1), 6)
    s_r = round(s, dec) or 10 ** (-dec)
    return f"{P:.{dec}f}", f"{s_r:.{dec}f}"


@section("table_known")
def _():
    FI = J("figure_info.json")["lc_amplitudes_all"]
    RR = J("review_reliability.json")
    LC = J("testswap_litcompare.json")
    rows = []
    for n in TEST:
        ad, r = A[n], res(n)
        full = SB[n]["object"]["fullname"]
        name, des = full.split(" ", 1)[1].split(" (")
        des = des.rstrip(")")
        mark = "$^\\dagger$" if SB[n]["object"]["orbit_class"]["code"] == "MCA" else ""
        Ps, ss = fmt_p(ad["P"], ad["sigma_P"])
        f = RR[n]
        plit = LC[n]["P_lit"]
        d = 100 * (ad["P"] - plit) / plit
        dcell = f"${d:+.3f}$" if abs(d) < 5 else "\\nodata"
        ft = "\\nodata" if f["f_tooth"] is None else f"{f['f_tooth']:.2f}"
        rows.append(f"({n}) {name}{mark} & {des} & {float(SB[n]['orbit']['moid']):.3f} & {len(r['datasets_used'])} & "
                    f"{r['n_after_outlier_removal']} & ${Ps}\\pm{ss}$ & {FI[n]['amp']:.2f} & {ad['quality']} & "
                    f"{f['f_u']:.2f} & {f['f_w']:.2f} & {ft} & {plit:g} & {dcell} \\\\")
    return {"rows": rows}


@section("test_literature")
def _():
    LC = J("testswap_litcompare.json")
    out = {}
    for n in TEST:
        x = LC[n]
        out[n] = {k: x.get(k) for k in ("P", "quality", "P_lit", "dP_lit", "dP_lit_pct", "dP_lit_in_P2Porb", "P2_over_Porb",
                                         "P_sid_lit", "dP_sid_in_P2Porb", "P_sid_own", "dP_sid_own_in_P2Porb")}
    close = [n for n in TEST if abs(out[n]["dP_lit_pct"]) < 0.05]
    return {"per_asteroid": out, "within_0.05pct": close,
            "dP_h_range_close": [min(abs(out[n]["dP_lit"]) for n in close), max(abs(out[n]["dP_lit"]) for n in close)],
            "P2Porb_range_close": [min(out[n]["P2_over_Porb"] for n in close), max(out[n]["P2_over_Porb"] for n in close)],
            "quality_counts": dict(Counter(A[n]["quality"] for n in TEST))}


@section("strongest_at_half")
def _():
    out = []
    for n in WITH_P:
        r = res(n)
        top = r["families_primary"][0]["P"]
        if abs(top - A[n]["P"] / 2) <= 0.01 * A[n]["P"] / 2:
            out.append(n)
    return {"all": out, "test": [n for n in out if n in TEST]}


@section("quantization")
def _():
    Q = J("no_quantized_check.json")
    by = defaultdict(lambda: [0, 0, []])
    for n in WITH_P:
        q = A[n]["quality"]
        by[q][1] += 1
        if Q[n]["agrees_with_adopted_family"]:
            by[q][0] += 1
        else:
            by[q][2].append(n)
    return {q: {"agree": v[0], "of": v[1], "different": v[2]} for q, v in sorted(by.items())}


@section("taxonomy")
def _():
    out = {}
    for n in NUMS:
        p = {x["name"]: x["value"] for x in SB[n].get("phys_par", [])}
        t = p.get("spec_T") or p.get("spec_B")
        if t:
            out[n] = {"spec_T": p.get("spec_T"), "spec_B": p.get("spec_B")}
    return {"n": len(out), "classes": out}


@section("amplitudes")
def _():
    FI = J("figure_info.json")["lc_amplitudes_all"]
    ratio = np.array([FI[n]["amp"] / FI[n]["amp_combined_fold"] for n in WITH_P])
    napp = [FI[n]["n_apparitions"] for n in WITH_P]
    low = {n: round(FI[n]["amp"], 3) for n in WITH_P if FI[n]["amp"] <= 0.15 + 1e-9}
    lam = J("data", "lcdb_amplitudes.json")
    test_vs = {}
    for n in TEST:
        l = lam.get(n) or lam.get(int(n))
        if not l or l["summary"]["amax"] is None:
            continue
        lo = l["summary"]["amin"] if l["summary"]["amin"] is not None else l["summary"]["amax"]
        a = round(FI[n]["amp"], 2)
        test_vs[n] = {"amp": a, "lcdb": [lo, l["summary"]["amax"]], "inside": lo - 1e-9 <= a <= l["summary"]["amax"] + 1e-9}
    amps_main = [FI[n]["amp"] for n in MAIN if n in WITH_P]
    return {"n_with_period": len(WITH_P), "app_over_comb_median": float(np.median(ratio)),
            "app_over_comb_16_84": [float(x) for x in np.percentile(ratio, [16, 84])],
            "n_apparitions": [min(napp), max(napp), float(np.median(napp))],
            "le_0.15": {n: (v, A[n]["quality"]) for n, v in low.items()}, "test_vs_lcdb": test_vs,
            "main_amp_range": [min(amps_main), max(amps_main)]}


@section("injection")
def _():
    S = J("amp_injection2_summary.json")
    return {k: {a: round(S[k][a][1], 3) for a in S[k]} for k in ("COMB_F4", "APP_TPL_wmean", "APP_FS_med", "APP_VAR_med",
                                                                 "APP_VARs_med", "APP_FS_max")}


def grades_unc_geo(sig):
    """Quality rule on the bootstrap of the uncorrected (results/) and the aspect-corrected (results_geo/) data
    (as audit_tmp/r409_geo_grades.py; the before/after blocks of geometry_reanalysis.json are not uncorrected grades)."""
    import sys
    sys.path.insert(0, ROOT)
    import quality_rule as Q
    out = {}
    for n in sig:
        g = []
        for sub in ("results", "results_geo"):
            v = J(sub, n, f"{n}_rebootstrap2.json")
            g.append(Q.grade(n, A[n], v, os.path.join(ROOT, sub, n, f"{n}_arrays.npz"))["quality_final"])
        out[n] = g
    return out


@section("aspect")
def _():
    GR = J("geometry_results.json")
    GV = J("geometry_validation.json")
    sig = sorted([k.split(":")[1] for k, r in GR.items() if k.startswith("paper:") and r and r.get("geometry")
                  and r["geometry"]["delta_bic"] > 10], key=int)
    GRA = J("geometry_reanalysis.json")
    up = []
    for n in sig:
        a = np.load(os.path.join(ROOT, "results", n, f"{n}_arrays.npz"))
        g = np.load(os.path.join(ROOT, "results_geo", n, f"{n}_arrays.npz"))
        from astropy.timeseries import LombScargle
        f0 = 24.0 / A[n]["P"]
        T = float(a["t"].max() - a["t"].min())
        fg = np.linspace(f0 - 0.5 / T, f0 + 0.5 / T, 201)
        pa = float(np.max(LombScargle(a["t"], a["y"], nterms=2).power(fg, method="chi2")))
        pg = float(np.max(LombScargle(g["t"], g["y"], nterms=2).power(fg, method="chi2")))
        up.append({"num": n, "power_before": pa, "power_after": pg, "higher": pg > pa,
                   "has_period": A[n].get("status") != "undetermined"})
    return {"n_fitted": GV["n_fitted"], "n_dbic_gt10": GV["n_dbic_gt10"],
            "n_fitted_paper": sum(1 for k, r in GR.items() if k.startswith("paper:") and r and r.get("geometry")),
            "significant_paper": sig, "poles": {k: GV["poles"][k] for k in GV["poles"] if not isinstance(GV["poles"][k], (list, dict))},
            "epoch_amplitudes": GV.get("epoch_amplitudes"),
            "correction_ptp_range": [min(GRA[n]["correction_ptp"] for n in GRA), max(GRA[n]["correction_ptp"] for n in GRA)],
            "grades_uncorrected_vs_corrected": grades_unc_geo(sig),
            "power_at_P": up}


@section("geometry_terms")
def _():
    G = J("geometry_terms.json")
    return {"counts": G["counts"], "preferred": G["preferred"]}


@section("damit")
def _():
    S = J("damit_comparison.json")
    T = J("damit_paper_table.json")
    out = {"n_runs": S["n"], "poles": {k: v for k, v in S["poles"].items() if k != "rows"},
           "sidereal": {k: v for k, v in S["sidereal_period"].items() if k != "rows"},
           "sidereal_rows_paper": [r["asteroid"] for r in S["sidereal_period"]["rows"] if r["asteroid"].startswith("paper:")],
           "pole_rows_paper": [r["asteroid"] for r in S["poles"]["rows"] if r["asteroid"].startswith("paper:")],
           "period_mismatch": S.get("period_mismatch"), "epoch_amplitude_ratio": S.get("epoch_amplitude_ratio"),
           "fit": S["fit"], "accuracy": T["accuracy"], "k_unc": T["k_unc"],
           "models": [(m["num"], m["quality"], m["P_adopted"], m["P_sid"], m["sigma_sid"], m["poles"], m["dark_pct"],
                       m["rel_diff"], m.get("published_poles")) for m in T["models"]],
           "rejected": [(m["num"], m["reason"]) for m in T.get("rejected", [])]}
    # Delta BIC of the convex model against the ellipsoid E2 (same formula and run selection as damit_eval.summarize)
    keep = set(A) | {n for n, a in J("adopted_retired.json").items() if a.get("convex_only")}
    db = []
    for d in sorted(os.listdir(os.path.join(ROOT, "damit_runs"))):
        parts = d.split("_")
        p = os.path.join(ROOT, "damit_runs", d, "eval.json")
        if len(parts) != 2 or (parts[0] == "paper" and parts[1] not in keep) or not os.path.exists(p):
            continue
        e = J("damit_runs", d, "eval.json")
        dC, dE, nn = e["C"]["dev"], e["E2"]["dev"], e["n"]
        db.append(nn * (dE ** 2 - dC ** 2) / dC ** 2 - ((49 + 3 + 3 - 1) - 9) * np.log(nn))
    db = np.array(db)
    out["bic_C_vs_E2"] = {"n": len(db), "frac_ellipsoid_preferred_lt_-10": float(np.mean(db < -10)),
                          "frac_convex_preferred_gt_10": float(np.mean(db > 10))}
    if EX("testswap_checks.json"):
        C = J("testswap_checks.json")
        out["principal_axis"] = {k: (v.get("angle_maxI_to_spin_deg"), v.get("error")) for k, v in C["principal_axis"].items()}
        out["phase_positivity"] = C.get("phase_positivity")
        out["damit_membership"] = C.get("damit_membership")
        H = C.get("harmonic_test", {})
        out["harmonic"] = {"n": len(H), "sig_eff": sorted([n for n, h in H.items() if h["dbic_eff"] > 10], key=int),
                           "pos_eff": sorted([n for n, h in H.items() if h["dbic_eff"] > 0], key=int),
                           "naive_gt10": len([n for n, h in H.items() if h["dbic_naive"] > 10])}
    return out


@section("fig7")
def _():
    F = J("fig_dataset_matrix_info.json")
    return {k: v for k, v in F.items() if k not in ("cells", "rows")}


@section("single_dataset")
def _():
    # statistic of the comparison of methods: quality A, at least 150 points, ground-based (audit_tmp/audit_numbers3.py)
    S = J("single_dataset_candidates.json")
    sel = [r for r in S if A[str(r["asteroid"])]["quality"] == "A" and r["n"] >= 150 and not r["dataset"].startswith("C57")]
    return {"n_all": len(S), "n": len(sel), "max_dominance": max(r["dominance"] for r in sel)}


@section("template_phase_angles")
def _():
    # mean phase angles of the apparition windows of the amplitude templates (geometry_results.json, all fitted asteroids)
    GR = J("geometry_results.json")
    al = [w["alpha_mean"] for k, r in GR.items() if r for w in r.get("windows", [])]
    alp = [w["alpha_mean"] for k, r in GR.items() if r and k.startswith("paper:") for w in r.get("windows", [])]
    return {"all_16_84": [float(x) for x in np.percentile(al, [16, 84])], "paper_16_84": [float(x) for x in np.percentile(alp, [16, 84])]}


@section("tess_geometry")
def _():
    dmin = (9.0, None)
    for n in NUMS:
        p = os.path.join(ROOT, "damit_runs", f"paper_{n}", "prep.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p, allow_pickle=True)
        m = np.array([str(s).startswith("C57") for s in z["sheet"]])
        if m.any():
            d = float(np.min(np.linalg.norm(z["earth"][m], axis=1)))
            if d < dmin[0]:
                dmin = (d, n)
    return {"min_delta_tess_au": dmin[0], "asteroid": dmin[1], "bound_deg": float(np.degrees(0.0025 / dmin[0])) if dmin[1] else None}


json.dump(OUT, open(os.path.join(ROOT, "testswap_numbers.json"), "w", encoding="utf-8"), indent=1, default=str)
for k, v in OUT.items():
    s = json.dumps(v, default=str)
    print(f"== {k}: {s[:1500]}")
