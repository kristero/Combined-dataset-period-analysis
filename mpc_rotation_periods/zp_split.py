"""
Correction of the zero-point jump of stations 703 (Catalina) and G96 (Mt. Lemmon) in summer 2018, when their
reference catalogue changed from Gaia DR1 to DR2 (Hoffmann et al. 2025; measured here: +0.071 +- 0.014 and
+0.086 +- 0.033 mag, station_zeropoint_check.py). Every 703 and G96 dataset is split at JD 2458300.5
(2018 July 1) into two datasets, so that each part receives its own shift onto the reference phase curve.
The post-jump part is named <stn>b<band> (e.g. 703bG): the station code stays in the first three characters
and the band in the last one, as the repository code and the analysis scripts expect. Parts with fewer than
MIN_N = 30 measurements are dropped (a 30-point part still fixes its own shift to about 0.04 mag; the 70-point
minimum of mpc_fetch.py would discard about 1000 measurements of the pre-2018 703G parts).
Input: data/workbooks/<N>-mpc.xlsx, data/obs_tables/<N>_info.json (and _obs.csv)
Output: zp/data/workbooks/<N>-mpc.xlsx, zp/data/obs_tables/<N>_info.json, _obs.csv; zp/zp_split.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
ZP = os.path.join(ROOT, "zp")
JD_SPLIT = 2458300.5
STATIONS = ("703", "G96")
MIN_N = 30


def split_one(num):
    wb = os.path.join(ROOT, "data", "workbooks", f"{num}-mpc.xlsx")
    info = json.load(open(os.path.join(ROOT, "data", "obs_tables", f"{num}_info.json"), encoding="utf-8"))
    sheets = pd.read_excel(wb, sheet_name=None)
    out, log = {"mpj": sheets["mpj"]}, {}
    for ds in info["datasets"]:
        d = sheets[ds[:31]]
        if ds[:3] in STATIONS:
            pre, post = d[d["epoch"] < JD_SPLIT], d[d["epoch"] >= JD_SPLIT]
            name_post = ds[:3] + "b" + ds[3:]
            log[ds] = {"n": len(d), "pre": len(pre), "post": len(post),
                       "kept": [n for n, p in ((ds, pre), (name_post, post)) if len(p) >= MIN_N]}
            if len(pre) >= MIN_N:
                out[ds] = pre
            if len(post) >= MIN_N:
                out[name_post] = post
        else:
            out[ds] = d
    order = sorted([k for k in out if k != "mpj"], key=lambda k: -len(out[k]))
    os.makedirs(os.path.join(ZP, "data", "workbooks"), exist_ok=True)
    os.makedirs(os.path.join(ZP, "data", "obs_tables"), exist_ok=True)
    with pd.ExcelWriter(os.path.join(ZP, "data", "workbooks", f"{num}-mpc.xlsx"), engine="openpyxl") as xw:
        out["mpj"].to_excel(xw, sheet_name="mpj", index=False)
        for k in order:
            out[k].to_excel(xw, sheet_name=k[:31], index=False)
    info2 = dict(info)
    info2["datasets"] = {k: int(len(out[k])) for k in order}
    info2["n_kept"] = int(sum(info2["datasets"].values()))
    info2["workbook"] = os.path.join(ZP, "data", "workbooks", f"{num}-mpc.xlsx")
    info2["zp_split"] = log
    json.dump(info2, open(os.path.join(ZP, "data", "obs_tables", f"{num}_info.json"), "w", encoding="utf-8"), indent=2)
    csv = os.path.join(ROOT, "data", "obs_tables", f"{num}_obs.csv")
    if os.path.exists(csv):
        o = pd.read_csv(csv)
        post = o["stn"].astype(str).isin(STATIONS) & (o["jd"] >= JD_SPLIT)
        o.loc[post, "dataset"] = o.loc[post, "stn"].astype(str) + "b" + o.loc[post, "band1"].astype(str)
        o = o[o["dataset"].isin(order)]
        o.to_csv(os.path.join(ZP, "data", "obs_tables", f"{num}_obs.csv"), index=False)
    return log


def main(nums):
    allog = {}
    for n in nums:
        allog[str(n)] = split_one(n)
        print(n, allog[str(n)])
    json.dump(allog, open(os.path.join(ZP, "zp_split.json"), "w"), indent=1)


if __name__ == "__main__":
    pub = json.load(open(os.path.join(ROOT, "published.json"), encoding="utf-8"))
    nums = sys.argv[1:] or sorted([k for k in pub if k.isdigit()], key=int)
    main(nums)
