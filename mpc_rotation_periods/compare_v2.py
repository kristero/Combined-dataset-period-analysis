"""Compare adopted.json with the corrected bootstrap / exact subset checks (rebootstrap2): does the adopted
family still win, what does the stated quality rule give, and is the uncertainty rule satisfied?"""
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
A = json.load(open(os.path.join(ROOT, "adopted.json"), encoding="utf-8"))


def same_family(Pa, Pb, tol=0.004):
    return any(abs(Pa - r * Pb) <= tol * Pa for r in (1.0, 2.0, 0.5))


def day_comm(P, tol=0.01):
    f2 = 48.0 / P
    return round(f2) >= 1 and abs(f2 - round(f2)) < tol


rows = {}
for num, ad in sorted(A.items(), key=lambda kv: int(kv[0])):
    v = json.load(open(os.path.join(ROOT, "results", num, f"{num}_rebootstrap2.json"), encoding="utf-8"))
    P = ad["P"]
    prim = v["primary"]
    match = [c for c in prim if abs(c["P"] - P) <= 1.5 * c["dP"]]
    fam_members = [c for c in prim if same_family(c["P"], P, 0.004)]
    famwin_u = max((c["famwin_u"] for c in fam_members), default=0.0)
    famwin_w = max((c["famwin_w"] for c in fam_members), default=0.0)
    c0 = match[0] if match else None
    subs = v["subsets"]
    key = "preferred_primary_P" if P <= 240 else "preferred_P"
    groups = {k: s for k, s in subs.items() if k.startswith("group:") and k != "group:space"}
    loo = {k: s for k, s in subs.items() if k.startswith("without:")}
    single = {k: s for k, s in subs.items() if k.startswith("dataset:") and not k.startswith("dataset:C57")}
    tess = subs.get("group:space") or subs.get("dataset:C57G")
    ok = lambda s: same_family(s[key], P, 0.006)
    g_ok, loo_ok, s_ok = sum(map(ok, groups.values())), sum(map(ok, loo.values())), sum(map(ok, single.values()))
    t_ok = None if tess is None else ok(tess)
    dc = day_comm(P)
    fg = g_ok / len(groups) if groups else 1.0
    if P <= 240 and famwin_u >= 0.95 and famwin_w >= 0.95 and loo_ok == len(loo) and fg >= 2 / 3 and t_ok is not False and not dc:
        q_rule = "A"
    elif max(famwin_u, famwin_w) >= 0.80 and (g_ok + loo_ok) >= 0.5 * (len(groups) + len(loo)):
        q_rule = "B"
    else:
        q_rule = "C"
    if c0:
        s_boot = c0["sigma_tooth"] if (c0["same_tooth"] >= 0.9 and c0["sigma_tooth"]) else c0["sigma_all"]
        sig_rule = max(s_boot, 0.5 * c0["dP"])
    else:
        s_boot, sig_rule = None, None
    ext_beat = [(round(c["P"], 1), c["frac_above_best_primary"]) for c in v["extended"] if c["frac_above_best_primary"] > 0.5]
    rows[num] = {"P": P, "Q_adopted": ad["quality"], "Q_rule": q_rule, "famwin_u": famwin_u, "famwin_w": famwin_w,
                 "tooth_win_u": c0["win_u"] if c0 else None, "same_tooth": c0["same_tooth"] if c0 else None,
                 "groups": f"{g_ok}/{len(groups)}", "loo": f"{loo_ok}/{len(loo)}", "single": f"{s_ok}/{len(single)}",
                 "tess": t_ok, "day_comm": dc, "sigma_adopted": ad["sigma_P"], "sigma_rule": sig_rule,
                 "extended_beats_primary": ext_beat}
json.dump(rows, open(os.path.join(ROOT, "compare_v2.json"), "w"), indent=1)
print(f"{'num':>5} {'P':>10} Qa Qr famU  famW  tooth  groups loo    single tess  dc  sig_ad   sig_rule  ext>prim")
for num, r in rows.items():
    flag = "" if (r["Q_adopted"] == r["Q_rule"] and r["tooth_win_u"] is not None) else "  <--"
    if r["P"] > 240:
        flag += " (P>240)"
    su = f"{r['sigma_rule']:.2g}" if r["sigma_rule"] else "-"
    print(f"{num:>5} {r['P']:>10.5f} {r['Q_adopted']}  {r['Q_rule']}  {r['famwin_u']:.2f}  {r['famwin_w']:.2f}  "
          f"{(r['same_tooth'] if r['same_tooth'] is not None else float('nan')):.2f}  {r['groups']:>5}  {r['loo']:>5}  "
          f"{r['single']:>5}  {str(r['tess'])[:5]:5} {str(r['day_comm'])[0]}  {r['sigma_adopted']:.2g}  {su:>8}  {r['extended_beats_primary']}{flag}")
