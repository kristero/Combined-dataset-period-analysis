"""
Candidates for new test asteroids (user request 2026-10-26): numbered main-belt asteroids and Mars crossers with an
Earth MOID of 0.6-1.0 AU and a secure LCDB period, not analysed in Eglitis & Svincicka (2025), Nagainis & Eglitis
(2026), this paper or its validation sample. Sources: JPL SBDB query API (MOID, H, observation count) and the local
LCDB (revision of 2023 October: summary, details, spin-axis tables). Writes review_candidates.json.
"""
import json
import os
import re
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
LCDB = os.path.join(ROOT, "data", "lcdb")
E25 = {1779, 1818, 1951, 1963, 2128, 2134, 2150, 2174, 2318, 2497, 2503, 2538, 2539, 2583}
N26 = {1951, 1963, 2134, 2150, 2607, 2968, 2971, 3081, 3173, 3473, 3716, 4303}
OURS = {int(k) for k in json.load(open(os.path.join(ROOT, "adopted.json"), encoding="utf-8"))}
VAL = {int(k.split(":")[1]) for k in json.load(open(os.path.join(ROOT, "amp_eval_results.json"), encoding="utf-8"))
       if k.startswith("val:")}
EXCLUDE = E25 | N26 | OURS | VAL
# references that are sparse or wide-field surveys (not dense light curves)
SURVEY = re.compile(r"^(Pal |Durech|Hanus|Waszczak|Chang |Polishook 2012|McNeill|Erasmus|Martikainen|Mahlke|Cibulkova|"
                    r"Ferrone|Masiero 2009|Szabo|Kim 2014|Pan-STARRS|Marciniak 2021|Muinonen|Ryan|Kiss|Molnar|Szakats|"
                    r"Berthier|Kalup|Barucci|Vaduvescu|Dziadura)")


def sbdb():
    q = {"fields": "pdes,full_name,moid,class,H,diameter,n_obs_used,data_arc", "sb-ns": "n", "sb-kind": "a",
         "sb-class": "MBA,IMB,MCA", "sb-cdata": json.dumps({"AND": ["moid|RG|0.6|1.0", "H|LT|14.5"]})}
    url = "https://ssd-api.jpl.nasa.gov/sbdb_query.api?" + urllib.parse.urlencode(q)
    with urllib.request.urlopen(url, timeout=120) as r:
        d = json.load(r)
    f = d["fields"]
    return {int(row[f.index("pdes")]): dict(zip(f, row)) for row in d["data"]}


def lcdb_summary():
    lines = open(os.path.join(LCDB, "lc_summary_pub.txt"), encoding="latin-1").read().split("\n")
    hdr = next(i for i, l in enumerate(lines) if l.startswith("NUMBER"))
    h = lines[hdr]
    cols = {k: h.index(k) for k in ("PERIOD", "P DESC", "AMIN", "AMAX", "U ", "NOTES", "BIN", "SAM")}
    out = {}
    for l in lines[hdr + 2:]:
        m = re.match(r"\s*(\d+)\s", l)
        if not m:
            continue
        g = lambda a, b: l[cols[a]:cols[b]].strip() if b else l[cols[a]:].strip()
        out[int(m.group(1))] = {"period": g("PERIOD", "P DESC"), "pdesc": g("P DESC", "AMIN"), "amax": g("AMAX", "U "),
                                "U": g("U ", "NOTES"), "notes": g("NOTES", "BIN"), "bin": g("BIN", "SAM"),
                                "name": l[10:41].strip(), "class": l[86:92].strip() if len(l) > 92 else ""}
    return out


def lcdb_details():
    refs, cur = {}, None
    for l in open(os.path.join(LCDB, "lc_details_pub.txt"), encoding="latin-1"):
        m = re.match(r"\s*(\d+)\s", l)
        if m:
            cur = int(m.group(1))
            refs[cur] = []
        elif cur is not None and l.startswith("          ") and l[10:11].strip():
            name = l[10:40].strip()
            per = l[148:162].strip() if len(l) > 150 else ""
            refs[cur].append((name, per))
    return refs


def spin_models():
    s = set()
    for l in open(os.path.join(LCDB, "lc_spinaxis_pub.txt"), encoding="latin-1"):
        m = re.match(r"\s*(\d+)\s", l)
        if m:
            s.add(int(m.group(1)))
    return s


def main():
    sb, lc, det, spin = sbdb(), lcdb_summary(), lcdb_details(), spin_models()
    rows = []
    for n, a in sb.items():
        if n in EXCLUDE or n not in lc:
            continue
        L = lc[n]
        try:
            P, amax = float(L["period"].split()[0]), float(L["amax"].split()[-1])
        except (ValueError, IndexError):
            continue
        dense = [r for r in det.get(n, []) if r[1] and not SURVEY.match(r[0])]
        rows.append({"num": n, "name": L["name"], "P": P, "U": L["U"], "amax": amax, "bin": L["bin"], "notes": L["notes"],
                     "moid": float(a["moid"]), "class": a["class"], "H": float(a["H"]), "D": a["diameter"],
                     "n_obs": int(a["n_obs_used"]), "dense_refs": [r[0] for r in dense],
                     "n_refs": len(det.get(n, [])), "spin_model": n in spin})
    sel = [r for r in rows if r["U"] in ("3", "3-") and r["amax"] >= 0.2 and not r["bin"] and "T" not in r["notes"]
           and len(r["dense_refs"]) >= 1]
    sel.sort(key=lambda r: (-len(r["dense_refs"]), -r["n_obs"]))
    json.dump({"n_sbdb": len(sb), "n_with_lcdb": len(rows), "candidates": sel}, open(os.path.join(ROOT, "review_candidates.json"), "w"), indent=1)
    print(f"SBDB: {len(sb)} numbered asteroids (MOID 0.6-1.0 AU, H<14.5); with an LCDB entry and not excluded: {len(rows)}; "
          f"secure dense-photometry candidates: {len(sel)}")
    for r in sel[:60]:
        print(f"({r['num']}) {r['name']:<16} P={r['P']:<9g} U={r['U']:<3} Amax={r['amax']:.2f} MOID={r['moid']:.3f} "
              f"{r['class']} H={r['H']:.1f} nobs={r['n_obs']} dense={len(r['dense_refs'])} spin={r['spin_model']} "
              f"[{'; '.join(r['dense_refs'][:4])}]")


if __name__ == "__main__":
    main()
