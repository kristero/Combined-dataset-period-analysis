"""
Algorithmic proposal for the adopted period of every asteroid, plus a compact evidence summary
(evidence/<N>.md) for the independent review of the alias choice.

Inputs: results/<N>/<N>_result.json (mpc_redo.py) and results/<N>/<N>_rebootstrap.json
(rebootstrap.py: night-block bootstrap restricted to the primary range 0.8-240 h, unweighted and
noise-weighted periodogram).

Rules
  1. Candidates in the primary range are grouped into harmonic families (P, 2P, P/2). The family
     that gives the highest power in most bootstrap resamplings wins (unweighted periodogram, as in
     the published code). If the noise-weighted periodogram prefers another family, the weighted
     family is adopted when the unweighted winner is day-commensurate (2f within 0.01 c/d of an
     integer: nightly zero-point systematics) and the weighted one is not, or when TESS alone
     supports the weighted family but not the unweighted one.
  2. Within the adopted family the double-peaked solution is chosen (two maxima and two minima of
     the four-term Fourier model); if the strongest member is single-peaked its double is adopted.
  3. Quality: A = both bootstrap family fractions >= 0.95, all leave-one-dataset-out solutions and
     >= 2/3 of the longitude groups in the family, TESS not contradicting, not day-commensurate;
     B = adopted-family fraction >= 0.80 (in the periodogram used for the decision) and a majority
     of the independent checks agree; C = otherwise (competing solutions remain).
  4. Uncertainty: max(bootstrap scatter within the comb tooth, P^2/(2 dT)); the scatter over all
     comb teeth is used when fewer than 90 % of the resamplings stay in the same tooth.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
EVD = os.path.join(ROOT, "evidence")
os.makedirs(EVD, exist_ok=True)
PUB = json.load(open(os.path.join(ROOT, "published.json"), encoding="utf-8"))
SBDB = json.load(open(os.path.join(ROOT, "data", "sbdb_phys_orbit.json"), encoding="utf-8"))


def same_family(Pa, Pb, tol=0.004):
    return any(abs(Pa - r * Pb) <= tol * Pa for r in (1.0, 2.0, 0.5))


def day_commensurate(P, tol=0.01):
    f2 = 2 * 24.0 / P
    k = round(f2)
    return bool(k >= 1 and abs(f2 - k) < tol)


def fold_extrema(t, y, P, n=4, prom=0.02):
    from scipy.signal import find_peaks
    w = 2 * np.pi * 24.0 / P
    A = np.vstack([np.ones_like(t)] + [f(k * w * t) for k in range(1, n + 1) for f in (np.sin, np.cos)]).T
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    ph = np.linspace(0, 1, 600, endpoint=False)
    G = np.vstack([np.ones_like(ph)] + [f(2 * np.pi * k * ph) for k in range(1, n + 1) for f in (np.sin, np.cos)]).T
    mg = G @ c
    e = np.r_[mg, mg, mg]
    pk, _ = find_peaks(e, prominence=prom)
    tr, _ = find_peaks(-e, prominence=prom)
    m = len(mg)
    return int(np.sum((pk >= m) & (pk < 2 * m))), int(np.sum((tr >= m) & (tr < 2 * m))), float(mg.max() - mg.min()), \
        float(np.std(y - A @ c))


def propose(num):
    r = json.load(open(os.path.join(RES, str(num), f"{num}_result.json"), encoding="utf-8"))
    if "error" in r:
        return {"asteroid": num, "error": r["error"]}
    rb_path = os.path.join(RES, str(num), f"{num}_rebootstrap.json")
    if not os.path.exists(rb_path):
        return {"asteroid": num, "error": "no rebootstrap"}
    rb = json.load(open(rb_path, encoding="utf-8"))
    a = np.load(os.path.join(RES, str(num), f"{num}_arrays.npz"))
    t, y = a["t"], a["y"]
    prim = rb["primary"]
    for c in prim:
        c["n4_max"], c["n4_min"], c["amp4"], c["rms4"] = fold_extrema(t, y, c["P"])
    fams = sorted(set(c["family"] for c in prim))
    fw_u = {f: max(c["famwin_u"] for c in prim if c["family"] == f) for f in fams}
    fw_w = {f: max(c["famwin_w"] for c in prim if c["family"] == f) for f in fams}
    best_u = max(fams, key=lambda f: fw_u[f])
    best_w = max(fams, key=lambda f: fw_w[f])
    tess = r.get("tess_only")

    def rep(f):
        mem = sorted([c for c in prim if c["family"] == f], key=lambda c: -c["power"])
        top = mem[0]
        if top["n4_max"] >= 2 and top["n4_min"] >= 2:
            return top, "strongest member (double-peaked)"
        dbl = [c for c in mem if abs(c["P"] - 2 * top["P"]) <= 0.004 * c["P"]]
        if dbl and dbl[0]["power"] >= 0.6 * top["power"]:
            return dbl[0], "double of the strongest (single-peaked) member"
        dp = [c for c in mem if c["n4_max"] >= 2 and c["n4_min"] >= 2]
        if dp and dp[0]["power"] >= 0.6 * top["power"]:
            return dp[0], "strongest double-peaked member"
        return top, "strongest member (no double-peaked alternative with comparable power)"

    def tess_supports(f):
        if not (tess and tess["peaks"]):
            return None
        P_t = tess["peaks"][0]["P"]
        if P_t > 240:
            return None      # TESS dominated by a trend: no information on the primary range
        return any(same_family(P_t, c["P"], tol=0.01) for c in prim if c["family"] == f)

    best_f, decision, frac = best_u, "unweighted bootstrap winner", fw_u[best_u]
    if best_w != best_u:
        du, dw = day_commensurate(rep(best_u)[0]["P"]), day_commensurate(rep(best_w)[0]["P"])
        if du and not dw:
            best_f, decision, frac = best_w, "weighted winner (unweighted winner day-commensurate)", fw_w[best_w]
        elif tess_supports(best_w) and not tess_supports(best_u):
            best_f, decision, frac = best_w, "weighted winner supported by TESS", fw_w[best_w]
        else:
            decision = "unweighted winner; weighted periodogram prefers another family"
    chosen, rule = rep(best_f)
    P = chosen["P"]

    subs = r.get("subsets", {})
    groups = {k: v for k, v in subs.items() if k.startswith("group:") and k != "group:space"}
    loo = {k: v for k, v in subs.items() if k.startswith("without:")}
    single = {k: v for k, v in subs.items() if k.startswith("dataset:") and not k.startswith("dataset:C57")}
    ok = lambda v: same_family(v["preferred_P"], P)
    g_ok, loo_ok, s_ok = sum(map(ok, groups.values())), sum(map(ok, loo.values())), sum(map(ok, single.values()))
    ts = tess_supports(best_f)
    dc = day_commensurate(P)
    fg = g_ok / len(groups) if groups else 1.0
    if (fw_u[best_f] >= 0.95 and fw_w[best_f] >= 0.95 and loo_ok == len(loo) and fg >= 2 / 3 and ts is not False
            and not dc):
        q = "A"
    elif frac >= 0.80 and (g_ok + loo_ok + s_ok) >= 0.5 * (len(groups) + len(loo) + len(single)):
        q = "B"
    else:
        q = "C"
    if best_f != best_u and q == "A":
        q = "B"

    base_h = rb["baseline_days"] * 24.0
    dPres = P ** 2 / base_h
    if chosen["same_tooth"] >= 0.9 and chosen["sigma_tooth"]:
        s_boot, teeth_note = chosen["sigma_tooth"], ""
    else:
        s_boot, teeth_note = chosen["sigma_all"], "bootstrap jumps between comb teeth (cycle-count ambiguity)"
    sigma = max(s_boot, 0.5 * dPres)
    per_days = next(float(e["value"]) for e in SBDB[str(num)]["orbit"]["elements"] if e["name"] == "per")

    alts = []
    for f in sorted(fams, key=lambda f: -max(fw_u[f], fw_w[f])):
        if f == best_f:
            continue
        c, _ = rep(f)
        alts.append({"P": c["P"], "power": c["power"], "power_weighted": c["power_weighted"],
                     "famwin_u": fw_u[f], "famwin_w": fw_w[f], "n4_extrema": [c["n4_max"], c["n4_min"]],
                     "day_commensurate": day_commensurate(c["P"])})
    ext = [{"P": c["P"], "power": c["power"], "frac_above_best_primary": c["frac_above_best_primary"]}
           for c in rb["extended"]]
    pub = PUB.get(str(num), {})
    cmp_pub = {k: {"value": pub[k], "same_family": same_family(pub[k], P), "rel_diff": (P - pub[k]) / pub[k]}
               for k in ("P_c", "P_av", "P_fig", "P_lit", "P_text") if k in pub}
    return {
        "asteroid": num, "table": pub.get("table"), "P": P, "sigma_P": sigma, "sigma_boot": s_boot,
        "teeth_note": teeth_note, "dP_resolution": dPres, "syn_sid_scale": P ** 2 / (per_days * 24.0),
        "orbital_period_days": per_days, "power": chosen["power"], "power_weighted": chosen["power_weighted"],
        "threshold": r["threshold_5sigma"], "famwin_u": fw_u[best_f], "famwin_w": fw_w[best_f],
        "decision": decision, "rule": rule, "day_commensurate": dc, "tess_supports": ts, "quality_auto": q,
        "fold_n4": {"maxima": chosen["n4_max"], "minima": chosen["n4_min"], "amp": chosen["amp4"], "rms": chosen["rms4"]},
        "checks": {"groups_agree": f"{g_ok}/{len(groups)}", "leave_one_out_agree": f"{loo_ok}/{len(loo)}",
                   "single_datasets_agree": f"{s_ok}/{len(single)}"},
        "alternatives_auto": alts[:4], "extended_candidates": ext, "published": cmp_pub,
        "n": r["n_after_outlier_removal"], "baseline_days": rb["baseline_days"], "datasets_used": r["datasets_used"],
        "screened_out": [s["dataset"] + " (" + s["reason"] + ")" for s in r["screening"] if not s["keep"]],
        "_prim": prim,
    }


def evidence_md(num, prop):
    r = json.load(open(os.path.join(RES, str(num), f"{num}_result.json"), encoding="utf-8"))
    rb = json.load(open(os.path.join(RES, str(num), f"{num}_rebootstrap.json"), encoding="utf-8"))
    full = SBDB[str(num)]["object"]["fullname"]
    cls = SBDB[str(num)]["object"]["orbit_class"]["code"]
    L = [f"# ({num}) {full}  [orbit class {cls}]", ""]
    pub = PUB.get(str(num), {})
    blind = pub.get("table") == 2
    if blind:
        L.append("VALIDATION OBJECT (asteroid with a published period): literature values are hidden; decide blind.")
    else:
        L.append("Values in the submitted manuscript: " + ", ".join(f"{k}={v}" for k, v in pub.items()))
    L.append(f"N = {r['n_after_outlier_removal']} points after outlier removal, baseline {rb['baseline_days'] / 365.25:.1f} yr, "
             f"reference {r['reference']}; datasets used (N): " + ", ".join(f"{k}({v})" for k, v in r["n_per_dataset"].items()))
    L.append("Screened out: " + (", ".join(prop["screened_out"]) or "none"))
    L.append(f"5-sigma threshold: {r['threshold_5sigma']:.4f}; strongest power in 0.8-240 h: {r['max_power_primary']:.4f}")
    L.append("")
    L.append("## Peak families, 0.8-240 h (strongest tooth; number of comb teeth within 1 %; relation to strongest)")
    for f in r["families_primary"][:10]:
        rel = "; ".join(f["relation_to_best"])
        L.append(f"- {f['P']:.4f} h, power {f['power']:.4f}, teeth {f['n_teeth']} (2nd tooth {f['second_tooth_ratio']:.2f})" + (f"; {rel}" if rel else ""))
    L.append("## Noise-weighted periodogram, strongest families: " +
             ", ".join(f"{f['P']:.4f} h ({f['power']:.3f})" for f in r["families_weighted"]))
    L.append("## Extended range 240-1500 h (fraction of bootstraps in which it beats the best primary candidate)")
    for c in prop["extended_candidates"]:
        L.append(f"- {c['P']:.1f} h, power {c['power']:.4f}, beats best primary in {c['frac_above_best_primary']:.2f}")
    L.append("")
    L.append("## Primary-range candidates, bootstrap (300 night-block resamplings)")
    L.append("| P (h) | family | power | weighted power | win u | family win u | win w | family win w | same tooth | sigma tooth (h) | n4 max/min | n4 amp | day-comm. |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in sorted(prop["_prim"], key=lambda c: (-c["famwin_u"], -c["power"])):
        L.append(f"| {c['P']:.5f} | {c['family']} | {c['power']:.4f} | {c['power_weighted']:.4f} | {c['win_u']:.2f} | {c['famwin_u']:.2f} | "
                 f"{c['win_w']:.2f} | {c['famwin_w']:.2f} | {c['same_tooth']:.2f} | "
                 f"{(c['sigma_tooth'] if c['sigma_tooth'] is not None else float('nan')):.2g} | "
                 f"{c.get('n4_max', '?')}/{c.get('n4_min', '?')} | {c.get('amp4', float('nan')):.3f} | {day_commensurate(c['P'])} |")
    L.append("")
    L.append("## Independent subsets (candidate with the highest power; ratio 2nd/1st)")
    for k, v in r["subsets"].items():
        ratio = v.get("ratio_second_to_first")
        L.append(f"- {k} (n={v['n']}, {v['baseline_days'] / 365.25:.1f} yr): {v['preferred_P']:.4f} h" +
                 (f" (2nd/1st {ratio:.2f})" if ratio is not None else ""))
    if r.get("tess_only"):
        tt = r["tess_only"]
        L.append(f"- TESS alone: n={tt['n']}, visits (JD start, end, n): {[(round(x, 1), round(y_, 1), n) for x, y_, n in tt['visits']]}; "
                 "strongest peaks: " + ", ".join(f"{p['P']:.3f} h ({p['power']:.3f})" for p in tt["peaks"][:5]))
    if r.get("spectral_window"):
        L.append("## Spectral window peaks (c/d, power): " + ", ".join(f"{x:.4f} ({w:.2f})" for x, w in r["spectral_window"]["window_peaks"]))
    L.append("")
    L.append("## Algorithmic proposal")
    hide = ("datasets_used", "screened_out") + (("published",) if blind else ())
    L.append(json.dumps({k: v for k, v in prop.items() if k not in hide and not k.startswith("_")}, indent=1))
    L.append("")
    L.append(f"## Figures in results/{num}/ (PNG)")
    L += [f"- {f}" for f in sorted(os.listdir(os.path.join(RES, str(num)))) if f.endswith(".png")]
    open(os.path.join(EVD, f"{num}.md"), "w", encoding="utf-8").write("\n".join(L))


def main(argv):
    nums = [int(x) for x in argv] or sorted(int(k) for k in PUB if not k.startswith("_"))
    props = {}
    for n in nums:
        if not os.path.exists(os.path.join(RES, str(n), f"{n}_rebootstrap.json")):
            print(n, "not ready")
            continue
        p = propose(n)
        props[str(n)] = p
        if "error" in p:
            print(n, "ERROR", p["error"])
            continue
        evidence_md(n, p)
        pub = PUB.get(str(n), {})
        ref = pub.get("P_c") or pub.get("P_lit")
        flag = "" if (ref and same_family(ref, p["P"]) and abs(p["P"] - ref) / ref < 0.003 and abs(p["P"] / ref - 1) < 0.003) else "  <-- differs"
        print(f"{n}: P={p['P']:.4f} +/- {p['sigma_P']:.2g} h Q={p['quality_auto']} fam_u={p['famwin_u']:.2f} fam_w={p['famwin_w']:.2f} "
              f"[{p['decision']}; {p['rule']}] groups {p['checks']['groups_agree']} loo {p['checks']['leave_one_out_agree']} "
              f"tess={p['tess_supports']} dc={p['day_commensurate']} | P_c={pub.get('P_c')} P_av={pub.get('P_av')} P_fig={pub.get('P_fig')}{flag}")
    old = {}
    if os.path.exists(os.path.join(ROOT, "proposals.json")):
        old = json.load(open(os.path.join(ROOT, "proposals.json"), encoding="utf-8"))
    old.update({k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in props.items()})
    json.dump(old, open(os.path.join(ROOT, "proposals.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
