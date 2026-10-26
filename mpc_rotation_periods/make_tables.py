"""
LaTeX tables of the revised manuscript, generated from the analysis results, the adopted periods
(adopted.json), the amplitudes measured for the figures (figure_info.json), the JPL SBDB records and
the MPC observatory list. The snippets are inserted into the manuscript at the %%%TABLEn%%% markers.
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
TEX = os.path.join(ROOT, "manuscript_revised", "mpc_rotation_periods.tex")

ADOPTED = json.load(open(os.path.join(ROOT, "adopted.json"), encoding="utf-8"))
FIGINFO = json.load(open(os.path.join(ROOT, "figure_info.json"), encoding="utf-8"))
SBDB = json.load(open(os.path.join(ROOT, "data", "sbdb_phys_orbit.json"), encoding="utf-8"))
OBS = json.load(open(os.path.join(ROOT, "data", "mpc_obscodes.json"), encoding="utf-8"))
PUB = json.load(open(os.path.join(ROOT, "published.json"), encoding="utf-8"))


def result(num):
    return json.load(open(os.path.join(RES, str(num), f"{num}_result.json"), encoding="utf-8"))


def name_des(num):
    full = SBDB[str(num)]["object"]["fullname"]           # "1820 Lohmann (1949 PO)"
    rest = full.split(" ", 1)[1]
    name, des = rest.split(" (")
    return name, des.rstrip(")")


def tex_name(num):
    name, des = name_des(num)
    cls = SBDB[str(num)]["object"]["orbit_class"]["code"]
    mark = "$^\\dagger$" if cls == "MCA" else ""
    return f"({num}) {name}{mark}", des


def moid(num):
    return float(SBDB[str(num)]["orbit"]["moid"])


def dataset_list(used):
    by = defaultdict(list)
    for ds in used:
        stn, band = ds[:3], (ds[-1] if len(ds) > 3 else "")   # 703bG: post-2018 part of 703G (zp_split.py)
        if band not in by[stn]:
            by[stn].append(band)
    parts = []
    for stn in sorted(by, key=lambda s: (not s[0].isdigit(), s)):
        bands = ",".join(sorted(b for b in by[stn] if b)) or ""
        parts.append(f"{stn}{bands}")
    return ", ".join(parts)


def fmt_p(P, s):
    if s <= 0 or not np.isfinite(s):
        return f"{P:.4f}", ""
    exp = int(np.floor(np.log10(s)))
    dec = max(0, -exp + 1)
    dec = min(dec, 6)
    s_r = round(s, dec)
    if s_r == 0:
        s_r = 10 ** (-dec)
    return f"{P:.{dec}f}", f"{s_r:.{dec}f}"


def table1():
    use = defaultdict(lambda: {"bands": set(), "ast": set()})
    for num in ADOPTED:
        r = result(num)
        for ds in r["datasets_used"]:
            use[ds[:3]]["bands"].add(ds[-1] if len(ds) > 3 else "--")
            use[ds[:3]]["ast"].add(num)
    rows = []
    for code in sorted(use, key=lambda c: (not c[0].isdigit(), c)):
        nm = OBS.get(code, {}).get("name", "?").replace("&", "\\&")
        bands = ", ".join(sorted(use[code]["bands"]))
        rows.append(f"{code} & {nm} & {bands} & {len(use[code]['ast'])} \\\\")
    body = "\n".join(rows)
    return (r"""\begin{deluxetable}{llcc}
\tablecaption{Observatories and photometric bands of the datasets used in the combined analysis\label{tab:obs}}
\tablehead{\colhead{Code} & \colhead{Observatory (MPC name)} & \colhead{Bands} & \colhead{$N_{\mathrm{ast}}$}}
\startdata
""" + body + r"""
\enddata
\tablecomments{Code: IAU/MPC observatory code. Bands: photometric band as reported to the MPC ($c$ and $o$: ATLAS cyan and orange; $w$: Pan-STARRS wide band; $G$: Gaia $G$ calibration). $N_{\mathrm{ast}}$: number of asteroids for which a dataset of this observatory entered the final analysis.}
\end{deluxetable}
""", len(use))


def lcdb_period(num):
    """Rotation period of the LCDB (Warner et al. 2009, revision 2023 October) as given in the JPL SBDB record."""
    for p in SBDB[str(num)].get("phys_par", []) or []:
        if p.get("name") == "rot_per":
            return float(p["value"])
    return None


def table2():
    rows = []
    stats = []
    for num in sorted(ADOPTED, key=int):
        ad = ADOPTED[num]
        if ad.get("table") != 2:
            continue
        r = result(num)
        nm, des = tex_name(int(num))
        Ps, ss = fmt_p(ad["P"], ad["sigma_P"])
        amp = FIGINFO["lc_amplitudes_all"][num]
        plit = lcdb_period(num)
        src = "LCDB"
        if plit is None:
            plit, src = PUB[num]["P_lit"], "ALCDEF"
        diff = 100 * (ad["P"] - plit) / plit
        same = abs(diff) < 5.0
        stats.append({"num": num, "P": ad["P"], "P_lit": plit, "P_lit_source": src, "diff_pct": diff if same else None,
                      "diff_h": ad["P"] - plit if same else None, "syn_sid": ad.get("syn_sid_scale"),
                      "quality": ad["quality"]})
        diff_s = f"${diff:+.3f}$" if same else "\\nodata"
        mark = "$^{a}$" if src == "ALCDEF" else ""
        rows.append(f"{nm} & {des} & {moid(int(num)):.3f} & {len(r['datasets_used'])} & {r['n_after_outlier_removal']} & "
                    f"${Ps}\\pm{ss}$ & {amp['amp']:.2f} & {ad['quality']} & {plit:g}{mark} & {diff_s} \\\\")
    body = "\n".join(rows)
    tab = (r"""\begin{deluxetable*}{llcrrccccr}
\tablecaption{Asteroids with previously published rotation periods: combined dataset periods and literature values\label{tab:known}}
\tablehead{\colhead{Asteroid} & \colhead{Designation} & \colhead{MOID} & \colhead{$N_{\mathrm{ds}}$} & \colhead{$N$} & \colhead{$P$} & \colhead{$A$} & \colhead{$Q$} & \colhead{$P_{\mathrm{lit}}$} & \colhead{$\Delta P/P_{\mathrm{lit}}$} \\
\colhead{} & \colhead{} & \colhead{(AU)} & \colhead{} & \colhead{} & \colhead{(h)} & \colhead{(mag)} & \colhead{} & \colhead{(h)} & \colhead{(\%)}}
\startdata
""" + body + r"""
\enddata
\tablecomments{MOID: Earth MOID from the JPL Small-Body Database \citep{JPLSBDB}. $^\dagger$ Mars-crossing asteroid. $N_{\mathrm{ds}}$: number of datasets retained. $N$: number of measurements after outlier removal. $P$: combined dataset period with its uncertainty, determined without reference to the literature values (Sections~\ref{sec:combined} and \ref{sec:choice}). $A$: light curve amplitude, mean of the amplitudes per apparition, weighted by the number of measurements (Section~\ref{sec:geometry}). $Q$: quality code (Section~\ref{sec:choice}). $P_{\mathrm{lit}}$: period in the LCDB \citep[revision of 2023 October;][]{Warner2009}, as given in the JPL Small-Body Database. $^{a}$ No LCDB period is available, and the value is taken from ALCDEF \citep{Warner2024}. For (1717) Arlon the data contain a second, independent periodicity at $5.14753\pm0.00021$\,h (amplitude 0.09\,mag), which agrees with $P_{\mathrm{lit}}$. The two periods match the two light curve components of this binary system \citep{Cooney2006a,Pravec2016} (Section~\ref{sec:known}).}
\end{deluxetable*}
""")
    return tab, stats


# transcription error in the submitted table: 10.743 h for the workbook value 10.473132 h (pav_recompute.md)
PAV_FIX = {"3716": 10.473132}


def table3():
    rows = []
    for num in sorted(ADOPTED, key=int):
        ad = ADOPTED[num]
        if ad.get("table") not in (3, 4):
            continue
        r = result(num)
        nm, des = tex_name(int(num))
        Ps, ss = fmt_p(ad["P"], ad["sigma_P"])
        amp = FIGINFO["lc_amplitudes_all"][num]
        pav = PAV_FIX.get(num, PUB[num].get("P_av"))
        pav_s = f"{pav:.3f}" if pav is not None else "\\nodata"
        if ad.get("status") == "undetermined":          # no rotation period determined (Section 3.3)
            p_cells = "\\nodata & \\nodata & \\nodata"
        else:
            p_cells = f"${Ps}\\pm{ss}$ & {amp['amp']:.2f} & {ad['quality']}"
        rows.append(f"{nm} & {des} & {moid(int(num)):.3f} & {dataset_list(r['datasets_used'])} & {r['n_after_outlier_removal']} & "
                    f"{p_cells} & {pav_s} \\\\")
    body = "\n".join(rows)
    return (r"""\startlongtable
\begin{deluxetable*}{llcp{5.0cm}rcccc}
\tabletypesize{\scriptsize}
\tablecaption{Rotation periods of the asteroids of the main sample\label{tab:new}}
\tablehead{\colhead{Asteroid} & \colhead{Designation} & \colhead{MOID} & \colhead{Datasets} & \colhead{$N$} & \colhead{$P$} & \colhead{$A$} & \colhead{$Q$} & \colhead{$P_{\mathrm{av}}$} \\
\colhead{} & \colhead{} & \colhead{(AU)} & \colhead{} & \colhead{} & \colhead{(h)} & \colhead{(mag)} & \colhead{} & \colhead{(h)}}
\startdata
""" + body + r"""
\enddata
\tablecomments{MOID: Earth MOID \citep{JPLSBDB}. $^\dagger$ Mars-crossing asteroid. Datasets: observatory code followed by the band(s) of the datasets retained in the combined analysis (Table~\ref{tab:obs}). $N$: number of measurements after outlier removal. $P$: combined dataset period and uncertainty. $A$: light curve amplitude, mean of the amplitudes per apparition, weighted by the number of measurements (Section~\ref{sec:geometry}). $Q$: reliability of the period (A: secure, B: probable, C: ambiguous; Section~\ref{sec:choice}). $P_{\mathrm{av}}$: weighted mean period of the separate dataset approach (Section~\ref{sec:separate}). Literature periods of (3459) and (4482) are discussed in Section~\ref{sec:notable}. No rotation period was determined for the six asteroids without entries in $P$, $A$, and $Q$ (Section~\ref{sec:notable}).}
\end{deluxetable*}
""")


def main():
    t1, n_codes = table1()
    t2, stats2 = table2()
    t3 = table3()
    for fn, t in (("table1.tex", t1), ("table2.tex", t2), ("table3.tex", t3)):
        open(os.path.join(ROOT, "manuscript_revised", fn), "w", encoding="utf-8").write(t)
    json.dump({"n_observatory_codes": n_codes, "table2_stats": stats2},
              open(os.path.join(ROOT, "table_info.json"), "w", encoding="utf-8"), indent=1)
    tex = open(TEX, encoding="utf-8").read()
    for marker, label, t in (("%%%TABLE1%%%", "tab:obs", t1), ("%%%TABLE2%%%", "tab:known", t2), ("%%%TABLE3%%%", "tab:new", t3)):
        if marker in tex:
            tex = tex.replace(marker, t.strip())
            continue
        # replace an already inserted table (identified by its label) so that the script can be re-run
        pat = re.compile(r"(\\startlongtable\s*)?\\begin\{deluxetable\*?\}[^\n]*\n(?:(?!\\end\{deluxetable).)*?\\label\{"
                         + re.escape(label) + r"\}.*?\\end\{deluxetable\*?\}", re.S)
        tex, n = pat.subn(lambda m: t.strip(), tex, count=1)
        print(label, "replaced" if n else "NOT FOUND")
    open(TEX, "w", encoding="utf-8", newline="\n").write(tex)
    print("tables written;", n_codes, "observatory codes")


if __name__ == "__main__":
    main()
