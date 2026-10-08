"""
Rebuilt workbooks for the second revision (values only, one sheet per dataset).

For every asteroid the workbook of the paper is copied sheet by sheet (the datasets of Table 2), and the
datasets that are missing from it are added from the MPC observation records (mpc_catalog/raw/<N>_ades.json):
every observatory code and photometric band with at least MIN_N magnitudes up to the end of the original
data retrieval of that asteroid. ADES band codes are mapped to the MPC ones (Ao -> o, Ac -> c, Pw -> w, ...).
The geocentric and heliocentric distances and the phase angle of the added rows are taken from the daily
ephemeris of the workbook (sheet 'mpj': year, month, day, JD, Delta, r, phase angle) at the integer JD of
the observation, in the same way as in the original workbooks.
For 3081 only a reduced workbook (phase angles above 7 deg, no TESS) exists, so all its datasets are built
from the MPC records.

Output: ../Asteroid_data/rev3/<N>-excelRB.xlsx  (+ build_log.json)
Columns of a data sheet: INT, epoch (JD), mag, Delta (AU), r (AU), Ph (deg), magred, magph7, JDc2,
in the same positions as in the original workbooks (Asteroid_pickle reads Delta and r by position).

Run:  python build_workbooks.py [N ...]
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from astropy.time import Time

import dephocus

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_1 = os.path.join(REPO_DIR, "..", "Asteroid_data", "12asteroidudatiicaruspubl")
DATA_2 = os.path.join(REPO_DIR, "..", "Asteroid_data", "12asteroidudatiicaruspubl2")
OUT_DIR = os.path.join(REPO_DIR, "..", "Asteroid_data", "rev3")
RAW_DIR = os.path.join(REPO_DIR, "mpc_catalog", "raw")
MIN_N = 70

# Workbook of the paper and its datasets (Table 2). Sheet names that differ from the dataset name are mapped.
SOURCES = {
    1951: (DATA_1, "1951-excelBF.xlsx", "703G 703V C57G H45R I41g I41r M22o T05c T05o T05w T08c T08o W68o Y00R"),
    1963: (DATA_1, "1963-excelBF.xlsx", "689V 703G 703V I41r M22o T05c T05o T05w T08o W68c W68o C57G2"),
    2134: (DATA_1, "2134-excelBF.xlsx", "703G 703V C57G G45r I41g I41r T05c T05o T08o W68o"),
    2150: (DATA_1, "2150-excelBF.xlsx", "703G 703V C57G G45r I41r M22o T05c T05o T08c T08o W68o"),
    2607: (DATA_1, "2607-excelBF.xlsx", "703G 703V D29R F51w G45r G96G G96V T05c T05o T08o"),
    2968: (DATA_1, "2968-excelBF.xlsx", "703G 703V F51w G45r G45G G96G T05c T05o T08c T08o"),
    2971: (DATA_1, "2971-excelBF.xlsx", "703G 703V C57G F51w G45r G96G I41r M22o T05c T05o T08o1 W68o"),
    3081: (DATA_1, "3081-excelBD.xlsx", ""),
    3173: (DATA_2, "3173-excelBF.xlsx", "703G 703V C57G D29R G45r G96G I41r M22o T05c T05o T08o W68o"),
    3473: (DATA_2, "3473-excelBF.xlsx", "691V 703G 703V C57G D29R G45r G96G G96V I41r M22o T05c T05o T08o W68o"),
    3716: (DATA_2, "3716-excelBF.xlsx", "703G 703V D29R F52w G45r G96G G96V M22o P07G T05c T05o T08o W68o"),
    4303: (DATA_2, "4303-excelBF.xlsx", "703G 703V C57G D29R F51w G45r G96G G96V M22o P07G T05c T05o T08o W68o"),
}
# Sheets copied under another name: the 1963 TESS sheet has its phase angle in an unnamed column.
RENAME = {(1963, "C57G2"): "C57G"}
COLUMNS = ["INT", "epoch", "mag", "Delta", "r", "Ph", "magred", "magph7", "JDc2"]


def dataset_key(stn: str, band: str) -> str:
    return f"{stn}{dephocus.normalise_band(band)}"


def mpc_records(num: int) -> pd.DataFrame:
    recs = json.load(open(os.path.join(RAW_DIR, f"{num}_ades.json")))
    rows = []
    for r in recs:
        if r.get("mag") in (None, "") or not r.get("obstime") or not r.get("band"):
            continue
        rows.append((r["stn"], dephocus.normalise_band(r["band"]), r["obstime"].replace("Z", ""), float(r["mag"])))
    df = pd.DataFrame(rows, columns=["stn", "band", "obstime", "mag"])
    df["epoch"] = Time(list(df["obstime"]), format="isot", scale="utc").jd
    df["key"] = df["stn"] + df["band"]
    return df.drop_duplicates(subset=["key", "epoch", "mag"]).reset_index(drop=True)


def ephemeris(xls: pd.ExcelFile) -> pd.DataFrame:
    e = pd.read_excel(xls, sheet_name="mpj", header=None).iloc[:, :7]
    e.columns = ["year", "month", "day", "jd", "Delta", "r", "phase"]
    e = e.apply(pd.to_numeric, errors="coerce").dropna()
    e = e.drop_duplicates(subset=["jd"], keep="first").sort_values("jd", kind="stable")
    return e.reset_index(drop=True)


def geometry(eph: pd.DataFrame, jd) -> tuple:
    """Delta, r and phase angle of the ephemeris row at the integer JD of each epoch, as in the original
    workbooks (the 'mpj' sheet has duplicated and missing days, so it is not interpolated). If that day is
    missing, the nearest row is used."""
    jd = np.floor(np.asarray(jd, float))
    e_jd = eph["jd"].to_numpy()
    i = np.clip(np.searchsorted(e_jd, jd, side="left"), 0, len(e_jd) - 1)
    prev = np.clip(i - 1, 0, len(e_jd) - 1)
    use_prev = (e_jd[i] != jd) & (np.abs(e_jd[prev] - jd) < np.abs(e_jd[i] - jd))
    i = np.where(use_prev, prev, i)
    return tuple(eph[c].to_numpy()[i] for c in ("Delta", "r", "phase"))


def sheet_from_records(sub: pd.DataFrame, eph: pd.DataFrame) -> pd.DataFrame:
    sub = sub.sort_values("epoch")
    delta, r, ph = geometry(eph, sub["epoch"])
    magred = sub["mag"].to_numpy() - 5 * np.log10(r * delta)
    return pd.DataFrame({"INT": np.floor(sub["epoch"].to_numpy()), "epoch": sub["epoch"].to_numpy(),
                         "mag": sub["mag"].to_numpy(), "Delta": np.round(delta, 4), "r": np.round(r, 4),
                         "Ph": np.round(ph, 2), "magred": magred, "magph7": magred, "JDc2": sub["epoch"].to_numpy()})


def copy_sheet(xls: pd.ExcelFile, sheet: str) -> pd.DataFrame:
    df = pd.read_excel(xls, sheet_name=sheet)
    df = df.iloc[:, :9].copy()
    df.columns = COLUMNS[:df.shape[1]]
    return df.dropna(subset=["epoch", "mag"])


def build(num: int) -> dict:
    d, name, ds = SOURCES[num]
    xls = pd.ExcelFile(os.path.join(d, name))
    eph = ephemeris(xls)
    rec = mpc_records(num)
    sheets, log = {}, {"source": name, "copied": {}, "added": {}, "geometry_check": {}}

    for s in ds.split():
        out = RENAME.get((num, s), s)
        df = copy_sheet(xls, s)
        sheets[out] = df
        log["copied"][out] = int(len(df))
    # End of the original data retrieval: last epoch of the copied datasets (3081: of its reduced workbook).
    if sheets:
        t_end = max(float(df["epoch"].max()) for df in sheets.values())
    else:
        ends = []
        for s in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=s)
            if {"epoch", "mag", "Ph"} <= set(df.columns):
                ends.append(float(pd.to_numeric(df["epoch"], errors="coerce").max()))
        t_end = max(ends)
    log["t_end"] = Time(t_end, format="jd").iso[:10]
    rec = rec[rec["epoch"] <= t_end + 0.5]
    counts = Counter(rec["key"])
    have = {k[:4] for k in sheets}
    for key, n in sorted(counts.items(), key=lambda x: -x[1]):
        if n < MIN_N or key in have or len(key) != 4:
            continue
        sheets[key] = sheet_from_records(rec[rec["key"] == key], eph)
        log["added"][key] = int(n)

    # Check of the interpolated geometry against the copied sheets.
    for key, df in sheets.items():
        if key in log["added"]:
            continue
        dl, rr, pp = geometry(eph, df["epoch"].astype(float))
        log["geometry_check"][key] = {"dDelta_max": float(np.nanmax(np.abs(dl - df["Delta"].astype(float)))),
                                      "dr_max": float(np.nanmax(np.abs(rr - df["r"].astype(float)))),
                                      "dPh_max": float(np.nanmax(np.abs(pp - df["Ph"].astype(float))))}
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{num}-excelRB.xlsx")
    with pd.ExcelWriter(path) as w:
        for key in sorted(sheets):
            sheets[key].to_excel(w, sheet_name=key, index=False)
    log["path"] = path
    log["n_per_sheet"] = {k: int(len(v)) for k, v in sheets.items()}
    return log


if __name__ == "__main__":
    nums = [int(a) for a in sys.argv[1:]] or list(SOURCES)
    logs = {}
    log_path = os.path.join(OUT_DIR, "build_log.json")
    if os.path.exists(log_path):
        logs = {int(k): v for k, v in json.load(open(log_path)).items()}
    for n in nums:
        logs[n] = build(n)
        print(n, "added:", logs[n]["added"], "end:", logs[n]["t_end"])
    json.dump(logs, open(log_path, "w"), indent=1)
