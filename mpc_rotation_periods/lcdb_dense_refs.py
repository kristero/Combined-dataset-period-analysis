"""Dense-photometry reference amplitudes from the LCDB details file: for every asteroid of the paper and validation
samples, the publication lines with a period within 2 % of the summary period, period quality U >= 2-, and no survey
flag (or LCI-D, dense data used for lightcurve inversion). The summary Amin/Amax also include sparse-survey entries
(SWF, DWF, LCI-W), which are not comparable single-apparition dense light curves.
Output: data/lcdb_dense_refs.json  {num: {"amps": [...], "dates": [...], "min", "max", "median"}}"""
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
LCDB = os.path.join(ROOT, "data", "lcdb")
nums = set(json.load(open(os.path.join(ROOT, "adopted.json"), encoding="utf-8")))
nums |= {s["num"] for s in json.load(open(os.path.join(ROOT, "validation", "sample.json"), encoding="utf-8"))["sample"]}
# retired test asteroids whose convex models stay in the paper (adopted_retired.json, convex_only)
_ra = os.path.join(ROOT, "adopted_retired.json")
nums |= {n for n, a in (json.load(open(_ra, encoding="utf-8")) if os.path.exists(_ra) else {}).items() if a.get("convex_only")}
UOK = {"2-", "2", "2+", "3-", "3"}


def fnum(s):
    s = s.strip().lstrip("<>")
    try:
        return float(s)
    except ValueError:
        return None


out, cur, summ_p = {}, None, None
for line in open(os.path.join(LCDB, "lc_details_pub.txt"), encoding="latin-1").read().splitlines()[5:]:
    num = line[0:10].strip()
    if num.isdigit():
        cur = num if num in nums else None
        summ_p = fnum(line[145:159]) if cur else None
        if cur:
            out[cur] = {"amps": [], "dates": [], "refs": [], "flags": []}
        continue
    if not cur or not line.strip() or summ_p is None:
        continue
    P, A, U = fnum(line[145:159]), fnum(line[182:187]), line[187:190].strip()
    flag = line[204:210].strip()
    if P is None or A is None or abs(P - summ_p) > 0.02 * summ_p or U not in UOK or flag not in ("", "LCI-D"):
        continue
    out[cur]["amps"].append(A)
    out[cur]["dates"].append(line[41:62].strip())
    out[cur]["refs"].append(line[10:41].strip())
    out[cur]["flags"].append(flag)
for n, r in out.items():
    if r["amps"]:
        a = sorted(r["amps"])
        r.update({"min": a[0], "max": a[-1], "median": a[len(a) // 2] if len(a) % 2 else 0.5 * (a[len(a) // 2 - 1] + a[len(a) // 2])})
json.dump(out, open(os.path.join(ROOT, "data", "lcdb_dense_refs.json"), "w"), indent=1)
have = [n for n, r in out.items() if r["amps"]]
print(f"{len(have)} of {len(nums)} asteroids have dense-photometry LCDB amplitudes; refs per object median "
      f"{sorted(len(out[n]['amps']) for n in have)[len(have) // 2]}")
