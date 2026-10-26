"""
Writes pav_recompute.md from pav_recompute.json and pav_recompute_recount.json (run pav_recompute.py and
pav_recompute_recount.py first).  Also runs the nterms = 1 / nterms = 2 check on the original workbook datasets.
"""
import json
import os

import numpy as np
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks

import pav_recompute as pr

ROOT = os.path.dirname(os.path.abspath(__file__))
J = json.load(open(os.path.join(ROOT, "pav_recompute.json"), encoding="utf-8"))["asteroids"]
RC = json.load(open(os.path.join(ROOT, "pav_recompute_recount.json"), encoding="utf-8"))
AD = json.load(open(os.path.join(ROOT, "adopted.json"), encoding="utf-8"))
PUB = json.load(open(os.path.join(ROOT, "published.json"), encoding="utf-8"))
MAIN = [k for k in J if J[k]["table"] != 2]
TEST = [k for k in J if J[k]["table"] == 2 and J[k].get("P_av_new") is not None]
WB = [k for k in J if "S" in J[k]]


def nterms_check():
    """nearest local maximum to the original p_i with nterms = 1 and nterms = 2 (wrong-sign JDc2, as originally)."""
    res = []
    for k in WB:
        for r in pr.block_sources(int(k)):
            a = r.get("_a")
            if a is None:
                continue
            m = np.isfinite(a["D"])
            t, y = a["jdc2"][m], a["mph"][m]
            T = t.max() - t.min()
            f0 = 24 / r["p"]
            f = np.arange(f0 * 0.98, f0 * 1.02, 1 / (20 * T))
            out = dict(ast=k, ds=r["ds"], p=r["p"], w=r["w"])
            for nt in (1, 2):
                p = LombScargle(t - t.min(), y, nterms=nt).power(f, method="fast" if nt == 1 else "fastchi2")
                pk = find_peaks(p)[0]
                j = pk[np.argmin(np.abs(f[pk] - f0))] if len(pk) else int(np.argmax(p))
                out[f"dp{nt}"] = 24 / f[j] - r["p"]
                out[f"w{nt}"] = float(p[j])
            res.append(out)
    return res


def fmt(x, n=3):
    return "--" if x is None else f"{x:.{n}f}"


def main():
    L = []
    w = L.append
    nt = nterms_check()
    dp1 = np.array([abs(x["dp1"]) for x in nt]); dp2 = np.array([abs(x["dp2"]) for x in nt])
    dw1 = np.array([x["w1"] - x["w"] for x in nt]); dw2 = np.array([x["w2"] - x["w"] for x in nt])
    # reproduction statistics (nterms = 2, refined)
    rep = [(k, d) for k in WB for d in J[k]["S"]["reproduction"]]
    dps = np.array([abs(d["dp"]) for _, d in rep]); dws = np.array([d["dw"] for _, d in rep])
    rel_main = np.array([J[k]["rel_change"] for k in MAIN])
    emul_main = np.array([J[k]["rel_change_emul"] for k in MAIN])
    worst_main = np.array([J[k]["rel_change_max_abs_all_variants"] for k in MAIN])
    kmax = MAIN[int(np.argmax(np.abs(rel_main)))]
    kworst = MAIN[int(np.argmax(worst_main))]
    nds = {k: len(J[k]["S"]["datasets"]) for k in WB}
    spread = [d["timing_error_spread_min"] for k in WB for d in J[k]["S"]["datasets"] if "timing_error_spread_min" in d]
    spread += [d["timing_error_spread_min"] for k in MAIN for v in ("R_rule3", "R_all") for d in J[k][v]["datasets"]]
    spread = np.array(spread)
    unc = {k: J[k]["rel_change_diapmax_unconstrained"] for k in WB}
    ku_max, ku_min = max(unc, key=unc.get), min(unc, key=unc.get)
    cal = [(abs(J[k]["S"]["summary"]["delta_sign_rel"] - J[k]["R_rule3"]["summary"]["delta_sign_rel"]),
            abs(J[k]["S"]["summary"]["delta_sign_rel"] - J[k]["R_all"]["summary"]["delta_sign_rel"]))
           for k in WB if J[k]["table"] != 2]
    cal_max = max(max(a, b) for a, b in cal)

    w("# Separate-dataset periods P_av recomputed with the correct light-time sign")
    w("")
    w("Scripts: `pav_recompute.py` (computation, writes `pav_recompute.json`), `pav_recompute_recount.py` (recounts, "
      "`pav_recompute_recount.json`), `pav_recompute_report.py` (this file). Workbook caches: `pav_recompute_cache/`. "
      "Nothing outside these new files was modified.")
    w("")
    w("## Summary")
    w("")
    w(f"* The original analysis used the `JDc2` column of the `*-excelB?.xlsx` workbooks, `=B2+(D2-$D$2)*4.99/36/24`, i.e. "
      f"epoch + (Delta_i - Delta_0)/c (wrong sign). Within one dataset the resulting timing error spans "
      f"2(Delta_max - Delta_min)/c = {np.percentile(spread, 10):.0f}-{np.percentile(spread, 90):.0f} min "
      f"(10-90 %, median {np.median(spread):.0f} min, max {spread.max():.0f} min).")
    w(f"* The per-dataset periods of the workbooks are reproduced only with a **two-term** Lomb-Scargle periodogram "
      f"(nterms = 2, standard normalization) at the rotation frequency: nearest-peak |Delta p| <= 1e-4 h for "
      f"{np.sum(dp2 <= 1e-4)}/{len(dp2)} datasets with nterms = 2 against {np.sum(dp1 <= 1e-4)}/{len(dp1)} with nterms = 1, "
      f"and median power difference to the tabulated W of {np.median(dw2):+.3f} (nterms = 2) against {np.median(dw1):+.3f} "
      f"(nterms = 1). All recomputations therefore use nterms = 2 (the task's default of nterms = 1 is contradicted by the workbooks).")
    w(f"* Validation on the 12 asteroids with workbooks ({len(rep)} datasets): the refined peaks reproduce the tabulated p_i with "
      f"median |Delta p| = {np.median(dps):.1e} h ({np.sum(dps <= 1e-4)}/{len(dps)} within 1e-4 h, {np.sum(dps <= 5e-4)}/{len(dps)} "
      f"within 5e-4 h, max {dps.max():.4f} h); the tabulated p_i have 4 decimals. With the reproduced p_i and the tabulated "
      f"w_i, N_i, R_i^2, Eq. (eq:pav) reproduces the workbook P_av to within "
      f"{100*max(abs(J[k]['S']['P_av_repro_pi_only']/J[k]['S']['P_av_sheet_eq']-1) for k in WB):.4f} %; with recomputed w_i "
      f"(the tabulated W were read off plots, median |Delta w| = {np.median(np.abs(dws)):.3f}) to within "
      f"{100*max(abs(J[k]['S']['summary']['P_av_wrong']/J[k]['S']['P_av_sheet_eq']-1) for k in WB if k != '2607'):.3f} % "
      f"(all except 2607, {100*(J['2607']['S']['summary']['P_av_wrong']/J['2607']['S']['P_av_sheet_eq']-1):+.2f} %).")
    w("* P_av is published (published.json, Table tab:new) for the **44 main-sample asteroids only**, not for the 13 test "
      "asteroids (Table tab:known has no P_av column). Workbooks exist for 8 main-sample and 4 test asteroids.")
    w(f"* Published P_av reproduced by a workbook block: 6 of the 8 main-sample asteroids with workbooks exactly (2968, 2971, "
      f"3081, 3173, 3473, 4303). **(3716): the published 10.743 h is a typo** for the workbook value 10.473 h "
      f"(10.473132 +- 0.002521 h, 13 datasets). **(2607): the published 2.943 h is not reproduced** by the only block in its "
      f"workbook (Eq. pav = 2.97089 h from 10 datasets); no unique subset of these datasets gives 2.943.")
    w(f"* Effect of the sign, keeping the original peak of every dataset (peak followed from the wrong-sign to the corrected "
      f"times): |Delta P_av / P_av| <= {100*np.abs(rel_main).max():.4f} % for the 44 main-sample asteroids (largest: ({kmax})); "
      f"median {100*np.median(np.abs(rel_main)):.4f} %. **No P_av changes by more than 0.1 % (0 of 44) or 1 % (0 of 44)**; "
      f"{int(np.sum(np.abs(rel_main) > 1e-4))} of 44 change by more than 0.01 %.")
    w(f"* Sensitivity (an analyst re-choosing the highest neighbouring peak after the correction): at most "
      f"{100*worst_main.max():.3f} % (({kworst})); {int(np.sum(worst_main > 1e-3))} of 44 exceed 0.1 % in the most "
      f"unfavourable variant, none exceeds 1 %.")
    w("* Recounts: the sign correction changes **none** of the statements of Section sec:compare or Table tab:slow "
      "(27 agree within 0.5 %, median 0.06 %; the same 11 daily aliases; the same 6 other cases). The (3716) typo, "
      "independently of the sign, changes 27 -> 28 (median 0.05 %) and 'remaining six' -> 'remaining five'.")
    w("")

    # ------------------------------------------------------------------------------------------------------------
    w("## 1. The separate-dataset method as implemented (from the workbooks)")
    w("")
    w("Workbooks: `Asteroid_data/12asteroidudatiicaruspubl{,2}/N-excelB?.xlsx` (20 files, 12 asteroids: test asteroids 1951, "
      "1963, 2134, 2150; main sample 2607, 2968, 2971, 3081, 3173, 3473, 3716, 4303). For every dataset (station + band) "
      "a sheet holds:")
    w("")
    w("* `Delta`, `r`, `Ph` by `VLOOKUP(INT(epoch), mpj, ...)` from a daily ephemeris sheet `mpj`;")
    w("* `magred = mag - 5 log10(r Delta)`; a linear fit of `magred` against `Ph` (observations at alpha > 7 deg; "
      "the slope is typed into the formula) gives `magph7 = magred - (Ph - Ph_0) k` (Eq. eq:linphase). The tabulated "
      "R^2 is the coefficient of determination of this linear fit (reproduced exactly, e.g. 3173/703G 0.3122, 4303/703G 0.3306);")
    w("* `JDc2 = epoch + (Delta - Delta_0) * 4.99/36/24` (formula `=B2+(D2-$D$2)*4.99/36/24`): the **wrong sign**. In some "
      "sheets rows were deleted after the formula was written, so Delta_0 refers to a removed row (constant offsets of up "
      "to 1.6 d, e.g. 3473/M22o, 4303/703V); a constant offset does not affect a single-dataset periodogram;")
    w("* an `a` sheet (value copy of `JDc2`, `magph7`) that was analysed with a Lomb-Scargle periodogram in a zoomed "
      "frequency interval (`diap`, cycles/day of the rotation frequency, e.g. 0.5-0.54 for 46 h).")
    w("")
    w("The per-dataset results (p_i, w_i, N_i, R_i^2, quality remark) are collected in `Periods`/`PeriodsD`/`Periodi` sheets "
      "and combined with Eq. (eq:pav) (`=AVERAGE(sum w_i p_i/w_s, sum N_i p_i/N_s, sum R_i^2 p_i/R_s^2)`). Blocks used here:")
    w("")
    w("| Asteroid | Workbook / sheet | datasets | Eq. pav of block (h) | published P_av (h) | remark |")
    w("|---|---|---|---|---|---|")
    for k in sorted(WB, key=lambda x: (J[x]["table"] == 2, int(x))):
        s = J[k]["S"]
        rem = ""
        if k == "3716":
            rem = "published value has transposed digits"
        elif k == "2607":
            rem = "published value not reproduced"
        elif J[k]["table"] == 2:
            rem = "test asteroid: P_av not in this manuscript"
        elif k == "3473":
            rem = "reproduced (the Periods block, 9.07395 h, also rounds to 9.074)"
        else:
            rem = "reproduced"
        if k == "1963":
            rem += "; rows 16-26 (18.1 h block) of a mixed sheet"
        w(f"| {k} | {os.path.basename(s['workbook'])} / {s['sheet']} | {nds[k]} | {s['P_av_sheet_eq']:.5f} | "
          f"{fmt(J[k]['P_av_old_published'])} | {rem} |")
    w("")
    w("Datasets actually combined: **8-14 per asteroid** (all datasets with a usable peak, down to N = 43), not two or three. "
      "The manuscript rule 'two or three observatories with >= 150 observations in a single band' describes only the initial "
      "period search (Section sec:separate). For the 36 main-sample asteroids without workbooks the original datasets are not "
      "recoverable (no workbooks; `Asteroid_data/Plots_and_data` holds only data-reduction plots, phase-curve fits, combined power spectra and amplitude plots for 8 of the workbook asteroids, no per-dataset periods).")
    w("")
    w(f"Periodogram: two-term model at the rotation frequency. Nearest local maximum to the tabulated p_i (wrong-sign JDc2, "
      f"astropy, grid 1/(20 T)), {len(nt)} workbook datasets:")
    w("")
    w("| | nterms = 1 | nterms = 2 |")
    w("|---|---|---|")
    w(f"| \\|Delta p\\| <= 1e-4 h | {np.sum(dp1 <= 1e-4)} | {np.sum(dp2 <= 1e-4)} |")
    w(f"| \\|Delta p\\| <= 1e-3 h | {np.sum(dp1 <= 1e-3)} | {np.sum(dp2 <= 1e-3)} |")
    w(f"| median (w - W) | {np.median(dw1):+.3f} | {np.median(dw2):+.3f} |")
    w(f"| median \\|w - W\\| | {np.median(np.abs(dw1)):.3f} | {np.median(np.abs(dw2)):.3f} |")
    w("")
    w("With nterms = 1 a double-peaked light curve has almost no power at the rotation frequency (e.g. 3173/703G: "
      "w = 0.04 against W = 0.45); with nterms = 2 the tabulated W are reproduced to a few hundredths for most datasets "
      "(some W, e.g. 2607/D29R 0.38 and F51w 0.40, are not reproduced by either model; they were probably read from a "
      "different plot).")
    w("")

    # ------------------------------------------------------------------------------------------------------------
    w("## 2. Recomputation")
    w("")
    w("Exact two-term Lomb-Scargle power (vectorized normal equations; identical to astropy `method='chi2'` to 3e-10), "
      "standard normalization, no weights, as in the original analysis.")
    w("")
    w("**S (12 asteroids with workbooks).** The same rows and `magph7` values as the original block (the sheet whose row count "
      "equals N_i; 'a' sheets whose rows are not in a dataset sheet take Delta from `mpj`). Wrong-sign times t_w = JDc2; "
      "corrected times t_c = JDc2 - 2 Delta/c (= epoch - (Delta - Delta_0)/c up to a constant, Eq. eq:lighttime_rel). "
      "For every dataset: local maximum nearest to the tabulated p_i in the t_w periodogram (refined on a fine grid), then the "
      "local maximum nearest to that peak in the t_c periodogram (the same peak followed). w_i from the recomputed "
      "periodograms, N_i and R_i^2 from the data (identical to the tabulated values except where noted in the json).")
    w("")
    w("**R (all 44 main-sample asteroids, our MPC data `data/obs_tables/<N>_obs.csv`).** Each dataset alone, alpha > 7 deg, "
      "relative reduction (Eqs. eq:reduced_rel, eq:linphase, R^2 of the linear magnitude-phase fit), t_w = jd + (Delta - "
      "Delta_first)/c (emulation of the workbooks) and t_c = jd_lt. Peak nearest to the published P_av (local maxima with "
      "prominence >= 0.1 of the window maximum, to skip noise ripples) with t_w, then followed with t_c. Dataset choices: "
      "**rule3** = largest dataset of each station with >= 150 measurements at alpha > 7 deg, the 2-3 largest stations "
      "(manuscript rule; used where no workbook exists); **all** = every dataset with >= 50 such measurements (the "
      "practice shown by the workbooks). Datasets must span >= 3 reference periods (a TESS visit cannot measure 447 h).")
    w("")
    w("**Adopted new value:** P_av_new = P_av_old + Delta_sign, with Delta_sign = P_av(t_c) - P_av(t_w) computed on identical "
      "data and peaks: from S where a workbook exists, otherwise from R/rule3. P_av_old is the published value (3716: "
      "workbook value 10.473132 h; test asteroids: workbook value). This isolates the sign; the absolute values of a "
      "recomputation from our data differ from the published ones for reasons unrelated to the sign (different time span, "
      "datasets, reduction), see the calibration below.")
    w("")
    w("**Sensitivity variants** (all in the json): (a) *family maximum*: where the original peak was the highest local "
      "maximum within +-1 % of itself, the highest local maximum within +-1 % is taken also with t_c; (b) *zoom-interval "
      "maximum* (workbooks only): where the original p_i was the highest peak of the tabulated `diap` interval, the highest "
      "peak of that interval with t_c, accepted only if it lies within the range of the original p_i of the block "
      "(unconstrained, it jumps to inconsistent daily aliases, e.g. 3081/T05o 8.007 -> 7.516 h, 2607/703V 2.933 -> 3.168 h, "
      "which an analyst requiring 'consistent periods in several datasets' would not accept); (c) R/all instead of R/rule3.")
    w("")

    # ------------------------------------------------------------------------------------------------------------
    w("## 3. Validation")
    w("")
    w("Workbook asteroids (S): reproduction of the original P_av from the wrong-sign times, and the effect of the sign.")
    w("")
    w("| Ast. | N_ds | P_av block (h) | repro, tabulated w (h) | repro, recomputed w (h) | max \\|dp_i\\| (h) | "
      "Delta_sign (h) | rel. (%) | family-max (%) | zoom-max (%) |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for k in sorted(WB, key=lambda x: (J[x]["table"] == 2, int(x))):
        s = J[k]["S"]; sm = s["summary"]
        mdp = max(abs(d["dp"]) for d in s["reproduction"])
        w(f"| {k} | {sm['n_datasets']} | {s['P_av_sheet_eq']:.5f} | {s['P_av_repro_pi_only']:.5f} | {sm['P_av_wrong']:.5f} | "
          f"{mdp:.4f} | {sm['delta_sign']:+.5f} | {100*sm['delta_sign_rel']:+.4f} | "
          f"{100*sm['delta_sign_emul']/sm['P_av_wrong']:+.4f} | "
          f"{100*sm.get('delta_sign_diapmax_consistent', float('nan'))/sm['P_av_wrong']:+.4f} |")
    w("")
    w("The largest per-dataset deviations of the reproduced p_i are TESS (C57G: 4303 -0.0028 h, 3473 +0.0024 h; a single "
      "visit of 14-72 d gives a broad peak) and a few datasets whose tabulated p_i lies on a flank (3173/T05c +0.0044 h, "
      "3173/T08o -0.0039 h, 2971/G45r -0.0027 h).")
    w("")
    w("Check of the claim in NOTES_FOR_AUTHORS.md (section 4, item 1) on the workbook data of (2607)/703V (198 points): "
      "between 2.925 and 2.945 h the highest peak is 2.93318 h (w = 0.187) with the wrong sign and 2.93564 h (0.159; "
      "2.93318 h drops to 0.152) with the corrected times, so the claim is confirmed; over the whole zoom interval "
      "7.5-8.5 c/d the highest peak with the corrected times is 3.168 h (0.183), an alias.")
    w("")
    w("Calibration of R on the 8 main-sample workbook asteroids (R/rule3 from our data vs S):")
    w("")
    w("| Ast. | P_av old (h) | R/rule3 wrong-sign P_av (h) | diff. (%) | Delta_sign S (%) | Delta_sign R/rule3 (%) | "
      "Delta_sign R/all (%) |")
    w("|---|---|---|---|---|---|---|")
    for k in sorted([x for x in WB if J[x]["table"] != 2], key=int):
        r = J[k]; r3 = r["R_rule3"]["summary"]; ra = r["R_all"]["summary"]
        w(f"| {k} | {r['P_av_old']:.5f} | {r3['P_av_wrong']:.5f} | {100*(r3['P_av_wrong']/r['P_av_old']-1):+.3f} | "
          f"{100*r['S']['summary']['delta_sign_rel']:+.4f} | {100*r3['delta_sign_rel']:+.4f} | {100*ra['delta_sign_rel']:+.4f} |")
    w("")
    r3d = np.array([J[k]["R_rule3"]["summary"]["P_av_wrong"] / J[k]["P_av_old"] - 1 for k in MAIN])
    w(f"For all 44 main-sample asteroids the R/rule3 wrong-sign P_av differs from the published P_av by a median "
      f"{100*np.median(np.abs(r3d)):.3f} % (max {100*np.abs(r3d).max():.2f} %, ({MAIN[int(np.argmax(np.abs(r3d)))]})); these "
      f"differences come from the different data and dataset choice, not from the sign, and are "
      f"larger than Delta_sign (typically by 1-2 orders of magnitude). Delta_sign itself agrees between S, R/rule3 and R/all to within {cal_max:.1e} (relative) "
      f"for these 8 asteroids.")
    w("")

    # ------------------------------------------------------------------------------------------------------------
    w("## 4. Old and new P_av")
    w("")
    w("P = adopted combined-dataset period (candidate period in parentheses for status 'undetermined'). Delta = P_av_new - "
      "P_av_old. 'src': S = workbook, R3 = our data/rule3 (datasets listed). 'max var.' = largest |relative change| over all "
      "variants (S, R/rule3, R/all; followed peak, family maximum, consistent zoom maximum). 'class' = recount class with "
      "P_av_new (agree <= 0.5 %, daily alias, other).")
    w("")
    w("| Ast. | Q | P (h) | P_av old (h) | P_av new (h) | Delta (h) | rel. (%) | max var. (%) | src | datasets (src) | class |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    cls_new = RC["new"]["per_asteroid"]
    for k in sorted(MAIN, key=int):
        r = J[k]
        und = r["status"] == "undetermined"
        Ps = f"({r['P_adopted']:g})" if und else f"{r['P_adopted']:g}"
        src = "S" if r["delta_sign_source"] == "S" else "R3"
        dsl = ", ".join(d["ds"] for d in r["primary"]["datasets"])
        old = f"{r['P_av_old_published']:.3f}" + (" (typo; 10.473132)" if k == "3716" else "")
        c = cls_new[k]
        cl = c["cls"] + (f" (k={c['k']}{', sum' if c.get('kind') == 'sum' else ''})" if c["cls"] == "alias" else "")
        w(f"| {k} | {r['quality'] or '--'} | {Ps} | {old} | {r['P_av_new']:.5f} | {r['delta_sign']:+.5f} | "
          f"{100*r['rel_change']:+.4f} | {100*r['rel_change_max_abs_all_variants']:.4f} | {src} | {dsl} | {cl} |")
    w("")
    w("Test asteroids (P_av is not given in this manuscript; values from the workbooks):")
    w("")
    w("| Ast. | P (h) | P_av old, workbook (h) | P_av new (h) | rel. (%) | max var. (%) |")
    w("|---|---|---|---|---|---|")
    for k in sorted(TEST, key=int):
        r = J[k]
        w(f"| {k} | {r['P_adopted']:g} | {r['P_av_old']:.5f} | {r['P_av_new']:.5f} | {100*r['rel_change']:+.4f} | "
          f"{100*r['rel_change_max_abs_all_variants']:.4f} |")
    w("")
    w(f"The other nine test asteroids have no separate-dataset value anywhere (no workbook, nothing published) and were not "
      f"computed.")
    w("")
    ch = RC["table3_last_digit_changes"]
    chd = [abs(J[k]["delta_sign"]) for k in ch if k != "3716"]
    chr_ = [abs(J[k]["rel_change"]) for k in ch if k != "3716"]
    w("Values of Table tab:new that change in the third decimal: " +
      "; ".join(f"({k}) {v['published']:.3f} -> {v['new_rounded']:.3f}" for k, v in sorted(ch.items(), key=lambda x: int(x[0]))) +
      f". Apart from (3716) (typo), these come from R/rule3 estimates of {min(chd):.4f}-{max(chd):.4f} h "
      f"(<= {100*max(chr_):.4f} %), of the same order as the differences between the R variants for the same asteroids; "
      "none of the eight workbook-based values changes at three decimals.")
    w("")

    # ------------------------------------------------------------------------------------------------------------
    w("## 5. Recounts of the manuscript statements and proposed text")
    w("")
    w("Classification as in Section sec:compare: agree if |P_av/P - 1| <= 0.5 %; daily alias if the second-harmonic "
      "frequencies 2/P_av and 2/P differ by an integer number of cycles/day within 0.02 c/d (approximate if the deviation "
      "exceeds 0.01 c/d) or, for (4129), add up to 1 c/d; P from adopted.json (candidate P for the six asteroids without "
      "a determined period). With the published values this reproduces the manuscript exactly.")
    w("")
    w("| P_av set | agree <= 0.5 % | median of agreeing | daily aliases | other |")
    w("|---|---|---|---|---|")
    labels = dict(published="published (as in manuscript)", sign_only="published + sign correction only",
                  published_typo_fixed="published, 3716 typo fixed", new="new (sign corrected, 3716 fixed)",
                  new_worst_variant="new, most unfavourable variant")
    for n in ("published", "sign_only", "published_typo_fixed", "new", "new_worst_variant"):
        r = RC[n]
        w(f"| {labels[n]} | {r['n_agree']} | {100*r['median_abs_rel_agree']:.3f} % | {r['n_alias']} "
          f"({', '.join(r['alias'])}; approx. {', '.join(r['approximate_alias'])}) | {r['n_other']} "
          f"({', '.join(f'{k} {100*v:+.1f} %' for k, v in r['other_rel'])}) |")
    w("")
    al = RC["new"]["per_asteroid"]
    w("Daily-alias relations with P_av_new (second-harmonic frequency difference minus k, c/d): " +
      "; ".join(f"({k}) k={al[k]['k']}{' sum' if al[k]['kind'] == 'sum' else ''} {al[k]['dev']:+.4f}"
                for k in RC["new"]["alias"]) + ". Unchanged from the published values to 1e-4 c/d. (4320) remains "
      f"outside the alias tolerance ({RC['new']['per_asteroid']['4320']['d2f']:+.3f} c/d), as in the manuscript.")
    w("")
    w("**Section sec:compare, first paragraph.** The sign correction changes nothing. Only the (3716) typo matters:")
    w("")
    w("> current: *agree with the combined-dataset periods to within 0.5\\% (median 0.06\\%) for 27 of the 44 asteroids.*")
    w(">")
    w("> proposed: *agree with the combined-dataset periods to within 0.5\\% (median 0.05\\%) for 28 of the 44 asteroids.*")
    w("")
    w("> current: *For the remaining six asteroids the two approaches disagree in other ways: ... for (3716), (3954), (4320), "
      "and (4682) the differences are 1--6\\%.*")
    w(">")
    w("> proposed: *For the remaining five asteroids the two approaches disagree in other ways: ... for (3954), (4320), and "
      "(4682) the differences are 1--6\\%.*")
    w("")
    w("The list of eleven daily aliases, the parenthetical '(two for (3172); approximately for (3172) and (3891))', and the "
      "(4129) sum relation are unchanged. Optional sentence after the first one: *The separate-dataset periods were "
      "recomputed with the light-time correction of Equation~(\\ref{eq:lighttime_rel}); the correction changes them by at most "
      "0.01\\%.* (If quoted, note that for 36 asteroids this is an estimate from our data, see Limitations.)")
    w("")
    w("**(4239) example.** *compared with 0.292 mag at the alias 15.61 h*: 15.61 h is the combined-data alias peak "
      "(15.61252 h, `alias_example.py`), not P_av; P_av_new = "
      f"{J['4239']['P_av_new']:.3f} h is unchanged. No change.")
    w("")
    w("**Table tab:new, P_av column.** (3716): 10.743 -> **10.473**. All other entries can stay (sign effect <= 0.010 %, "
      "below the reproducibility of P_av); if the recomputed values are preferred, the third-decimal changes are listed in "
      "Section 4.")
    w("")
    sl = RC["slow_table"]
    w("**Table tab:slow, P_av column.** New values (3 decimals): " +
      "; ".join(f"({k}) {v['P_av_published']:.3f} -> {v['rounded_new']:.3f}" for k, v in sl.items()) +
      ". Changes: " + ", ".join(f"({k}) {J[k]['delta_sign']:+.4f} h ({100*J[k]['rel_change']:+.4f} %)" for k in sl
                                if abs(round(J[k]['P_av_new'], 3) - PUB[k]['P_av']) > 1e-9) +
      ", all R/rule3 estimates; recommended: keep the published values. The statements of the second paragraph of "
      "sec:compare (*recovers the long period for five of these "
      "eight asteroids*; *daily aliases at 31--57\\,h* for (3891), (4129), (4478); *daily aliases near 46\\,h* for (1944), "
      "(3173), (3216)) are unchanged.")
    w("")
    w("**Section sec:separate (inconsistency found, independent of the sign).** *With only two or three datasets per asteroid, "
      "the scatter of the individual periods does not give a reliable uncertainty*: the workbooks combine 8-14 datasets per "
      "asteroid in Eq. (eq:pav); 'two or three observatories with >= 150 observations' applies to the initial search only. "
      "Proposed: *The periods of all datasets that show the peak (8--14 per asteroid) were combined as ...* and *Because the "
      "individual periods often fall on different comb teeth or aliases, their scatter does not give a reliable "
      "uncertainty, and $P_{\\mathrm{av}}$ is listed without one*. Authors to confirm for the 36 asteroids without workbooks.")
    w("")
    w("**Discussion / Conclusions** (*This explains most of the discrepancies between the separate-dataset and the "
      "combined-dataset periods*; *remains a useful consistency check*): unchanged.")
    w("")
    w("**NOTES_FOR_AUTHORS.md** section 2 (*agree with the new periods for 27 of 44 objects; for 11 they are daily aliases*) "
      "and section 4 item 1 / section 5 (P_av should be recomputed): can be updated to 28 of 44 (3716 typo) and 'recomputed, "
      "changes < 0.01 %'.")
    w("")

    # ------------------------------------------------------------------------------------------------------------
    w("## 6. Limitations")
    w("")
    w("* Only the 8 main-sample (and 4 test) asteroids with workbooks are recomputed on the original data; for the other 36 the "
      "sign effect is estimated on our MPC data (longer time span to 2026, possibly different datasets and reduction). The "
      f"calibration on the 8 shows that this estimate of Delta_sign agrees with the workbook value to within {cal_max:.1e} "
      "(relative), but the "
      "absolute P_av from our data differ from the published ones by up to "
      f"{100*np.abs(r3d).max():.1f} %, so P_av_new = P_av_old + Delta_sign is used rather than the absolute recomputation.")
    w("* The primary result follows the same periodogram peak (the task's 'keep the original peak family choice'). An analyst "
      "who re-selects the highest neighbouring peak could obtain changes up to "
      f"{100*worst_main.max():.2f} % (({kworst})); larger jumps to other daily aliases are possible in single datasets "
      f"(unconstrained zoom-interval maximum: {100*unc[ku_max]:+.2f} % for ({ku_max}) and {100*unc[ku_min]:+.2f} % for "
      f"({ku_min})) but would violate the consistency requirement of the method.")
    w("* The tabulated W of the workbooks were read off plots (two decimals) and some do not match any periodogram of the "
      "tabulated data (2607); recomputed w_i are used for both signs, so Delta_sign is not affected.")
    w("* (2607): the published 2.943 h cannot be traced to the workbook; its Delta_sign is taken from the workbook block "
      "(Eq. pav 2.97089 h).")
    w("* The light time uses the daily ephemeris of the workbooks (S) or the per-observation Horizons distances (R); "
      "the difference is < 0.1 min.")
    w("")
    w("## Files")
    w("")
    w("* `pav_recompute.py`, `pav_recompute.json` (per asteroid: P_av_old/new, delta_sign, rel_change, variants, `primary` "
      "= datasets with N_i, R2_i, p_i/w_i wrong and new; `S` = workbook block with original values, reproduction and "
      "per-dataset peaks; `R_rule3`, `R_all`; flags)")
    w("* `pav_recompute_recount.py`, `pav_recompute_recount.json`")
    w("* `pav_recompute_report.py`, `pav_recompute.md`")
    w("* `pav_recompute_cache/` (pickled workbook sheets)")
    open(os.path.join(ROOT, "pav_recompute.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("written pav_recompute.md;", len(L), "lines")


if __name__ == "__main__":
    main()
