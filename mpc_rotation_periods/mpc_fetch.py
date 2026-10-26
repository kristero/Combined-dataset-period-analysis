"""
Build pipeline-ready inputs from the MPC observation database.

For every asteroid:
  1. download all observations from the MPC observations API (ADES data frame, cached as JSON);
  2. keep CCD/CMOS photometry (no photographic, no deprecated records, rms_mag <= 0.2 mag when given);
  3. get the geocentric r, Delta and solar phase angle from JPL Horizons (daily steps, cubic-spline
     interpolated to each observation);
  4. split the photometry into observatory+band datasets and keep datasets with >= 70 measurements;
  5. write an Excel workbook in the format read by Asteroid_pickle.DatasetGenerator
     (one sheet per dataset: day, epoch, mag, delta, r, Ph, magred, magph7, Jdc2) and a flat CSV.

Light-time correction: Jdc2 = JD_UTC - Delta / c (emission time). DatasetGenerator uses the "Jdc2"
column when present, so every dataset is on the same absolute, light-time-corrected time scale.

Usage:  python mpc_fetch.py 4239 2539 ...      (no arguments: all asteroids of the paper)
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import requests
from astropy.time import Time
from scipy.interpolate import CubicSpline

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(ROOT, "data", "mpc_raw")
EPH_DIR = os.path.join(ROOT, "data", "horizons")
WB_DIR = os.path.join(ROOT, "data", "workbooks")
OBS_DIR = os.path.join(ROOT, "data", "obs_tables")
for _d in (RAW_DIR, EPH_DIR, WB_DIR, OBS_DIR):
    os.makedirs(_d, exist_ok=True)

C_AU_PER_DAY = 173.1446326846693
MIN_N = 70
MAX_RMSMAG = 0.2
EXCLUDED_MODES = {"PHO", "ENC", "MIC", "MER", "PMT", "VIS", "FBK", "OCC"}

# test sample after the revision (retired.json: 1951, 1963, 2134, 2150 and 2318 replaced by six new test asteroids,
# whose records were retrieved on 2026-10-26)
TABLE2 = [249, 486, 870, 1600, 1700, 1706, 1717, 1798, 1820, 1920, 1942, 1949, 3567, 7750]
TABLE34 = [1818, 1944, 2497, 2503, 2538, 2539, 2583, 2607, 2614, 2817, 2968, 2971, 3081, 3172, 3173,
           3216, 3459, 3473, 3716, 3777, 3858, 3891, 3954, 4048, 4129, 4239, 4245, 4267, 4270, 4273,
           4303, 4320, 4322, 4331, 4371, 4396, 4456, 4478, 4482, 4556, 4618, 4682, 4761, 4777]
ALL = TABLE2 + TABLE34


def normalise_band(band) -> str | None:
    """Map MPC/ADES band codes to the single-letter codes used for the dataset names."""
    if band is None:
        return ""
    b = str(band).strip()
    if b == "":
        return ""
    if b.upper() in ("UNK", "NAN", "NONE"):   # unknown band: not usable
        return None
    if b in ("Vj", "Rc", "Ic", "Bj", "Uj"):
        return b[0]
    if len(b) == 2 and b[0] in "PSALM":        # Pan-STARRS, SDSS, ATLAS, LSST, ... e.g. Pw, Sg, Ao, Ac
        return b[1]
    if len(b) == 1:
        return b
    if b.upper().startswith("W") or b in ("J", "H", "K", "Ks"):  # infrared: not used
        return None
    return b[-1]


def fetch_mpc(num: int, force: bool = False) -> list:
    path = os.path.join(RAW_DIR, f"{num}_ades.json")
    if os.path.exists(path) and not force:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    for attempt in range(5):
        try:
            r = requests.get("https://data.minorplanetcenter.net/api/get-obs",
                             json={"desigs": [str(num)], "output_format": ["ADES_DF"]}, timeout=300)
            r.raise_for_status()
            ades = r.json()[0]["ADES_DF"]
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(ades, fh)
            return ades
        except Exception as exc:  # network hiccups: retry with back-off
            print(f"[{num}] MPC attempt {attempt + 1} failed: {exc}")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"MPC download failed for {num}")


def fetch_horizons(num: int, jd_start: float, jd_stop: float) -> pd.DataFrame:
    path = os.path.join(EPH_DIR, f"{num}_geocentric_daily.csv")
    if os.path.exists(path):
        eph = pd.read_csv(path)
        if eph["jd"].min() <= jd_start and eph["jd"].max() >= jd_stop:
            return eph
    start = Time(np.floor(jd_start) - 2, format="jd").utc.strftime("%Y-%m-%d")
    stop = Time(np.ceil(jd_stop) + 2, format="jd").utc.strftime("%Y-%m-%d")
    params = {"format": "json", "COMMAND": f"'{num};'", "EPHEM_TYPE": "OBSERVER", "CENTER": "'500@399'",
              "START_TIME": f"'{start}'", "STOP_TIME": f"'{stop}'", "STEP_SIZE": "'1 d'",
              "QUANTITIES": "'19,20,43'", "OBJ_DATA": "NO", "CSV_FORMAT": "YES", "CAL_FORMAT": "JD"}
    for attempt in range(5):
        try:
            h = requests.get("https://ssd.jpl.nasa.gov/api/horizons.api", params=params, timeout=300)
            h.raise_for_status()
            res = h.json()["result"]
            body = res[res.index("$$SOE") + 5: res.index("$$EOE")]
            rows = []
            for line in body.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                # JD, (solar flag), (lunar flag), r, rdot, delta, deldot, phi, PAB-LON, PAB-LAT
                rows.append([float(parts[0]), float(parts[3]), float(parts[5]), float(parts[7]),
                             float(parts[8]), float(parts[9])])
            eph = pd.DataFrame(rows, columns=["jd", "r", "delta", "alpha", "pab_lon", "pab_lat"])
            eph.to_csv(path, index=False)
            return eph
        except Exception as exc:
            print(f"[{num}] Horizons attempt {attempt + 1} failed: {exc}")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Horizons query failed for {num}")


def build(num: int, force: bool = False) -> dict:
    ades = fetch_mpc(num, force=force)
    df = pd.DataFrame(ades)
    n_all = len(df)
    df["mag_f"] = pd.to_numeric(df.get("mag"), errors="coerce")
    df["rmsmag_f"] = pd.to_numeric(df.get("rmsmag"), errors="coerce")
    df = df[np.isfinite(df["mag_f"])].copy()
    n_mag = len(df)
    mode = df.get("mode", pd.Series([""] * len(df), index=df.index)).fillna("").astype(str).str.upper()
    dep = df.get("deprecated", pd.Series([None] * len(df), index=df.index)).fillna("").astype(str).str.strip()
    keep = (~mode.isin(EXCLUDED_MODES)) & (dep == "")
    keep &= ~(np.isfinite(df["rmsmag_f"]) & (df["rmsmag_f"] > MAX_RMSMAG))
    df = df[keep].copy()
    df["band1"] = df["band"].map(normalise_band)
    df = df[df["band1"].notna()].copy()
    df["dataset"] = df["stn"].astype(str) + df["band1"].astype(str)
    df["jd"] = Time(list(df["obstime"].str.replace("Z", "", regex=False)), format="isot", scale="utc").jd

    counts = df["dataset"].value_counts()
    good = counts[counts >= MIN_N].index.tolist()
    df = df[df["dataset"].isin(good)].copy()
    if df.empty:
        raise RuntimeError(f"No dataset with >= {MIN_N} measurements for {num}")

    eph = fetch_horizons(num, float(df["jd"].min()), float(df["jd"].max()))
    for col in ("r", "delta", "alpha", "pab_lon", "pab_lat"):
        y = eph[col].to_numpy()
        if col == "pab_lon":
            y = np.degrees(np.unwrap(np.radians(y)))
        df[col] = CubicSpline(eph["jd"].to_numpy(), y)(df["jd"].to_numpy())
    df["pab_lon"] = np.mod(df["pab_lon"], 360.0)
    df["jd_lt"] = df["jd"] - df["delta"] / C_AU_PER_DAY
    df["H_red"] = df["mag_f"] - 5.0 * np.log10(df["r"] * df["delta"])
    df = df.sort_values(["dataset", "jd"]).reset_index(drop=True)

    cols = ["dataset", "stn", "band", "band1", "mode", "obstime", "jd", "jd_lt", "mag_f", "rmsmag_f",
            "r", "delta", "alpha", "pab_lon", "pab_lat", "H_red", "photcat", "trkid", "obsid"]
    df[[c for c in cols if c in df.columns]].to_csv(os.path.join(OBS_DIR, f"{num}_obs.csv"), index=False)

    # Workbook in the DatasetGenerator format.  Columns 3 and 4 must be the two distances.
    order = counts.loc[good].sort_values(ascending=False).index.tolist()
    wb_path = os.path.join(WB_DIR, f"{num}-mpc.xlsx")
    with pd.ExcelWriter(wb_path, engine="openpyxl") as xw:
        eph_out = eph[["jd", "delta", "r", "alpha"]].copy()
        eph_out.to_excel(xw, sheet_name="mpj", index=False)
        for ds in order:
            d = df[df["dataset"] == ds]
            sheet = pd.DataFrame({
                "day": np.floor(d["jd"].to_numpy()),
                "epoch": d["jd"].to_numpy(),
                "mag": d["mag_f"].to_numpy(),
                "delta": d["delta"].to_numpy(),
                "r": d["r"].to_numpy(),
                "Ph": d["alpha"].to_numpy(),
                "magred": d["H_red"].to_numpy(),
                "magph7": d["H_red"].to_numpy(),
                "Jdc2": d["jd_lt"].to_numpy(),
            })
            sheet.to_excel(xw, sheet_name=ds[:31], index=False)

    info = {
        "asteroid": num, "n_records": n_all, "n_with_mag": n_mag, "n_kept": int(len(df)),
        "datasets": {ds: int(counts[ds]) for ds in order},
        "jd_first": float(df["jd"].min()), "jd_last": float(df["jd"].max()),
        "utc_first": Time(df["jd"].min(), format="jd").utc.iso[:10],
        "utc_last": Time(df["jd"].max(), format="jd").utc.iso[:10],
        "alpha_range": [float(df["alpha"].min()), float(df["alpha"].max())],
        "bands_raw": sorted(set(map(str, pd.DataFrame(ades).get("band", pd.Series(dtype=str)).dropna()))),
        "workbook": wb_path,
    }
    with open(os.path.join(OBS_DIR, f"{num}_info.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)
    return info


def main(argv):
    force = "--force" in argv
    nums = [int(a) for a in argv if not a.startswith("--")] or ALL
    results, failures = {}, {}
    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(build, n, force): n for n in nums}
        for fut in cf.as_completed(futs):
            n = futs[fut]
            try:
                info = fut.result()
                results[n] = info
                print(f"[{n}] kept {info['n_kept']} of {info['n_with_mag']} mags; {len(info['datasets'])} datasets; "
                      f"{info['utc_first']} .. {info['utc_last']}")
            except Exception as exc:
                failures[n] = f"{type(exc).__name__}: {exc}"
                print(f"[{n}] FAILED: {failures[n]}")
    with open(os.path.join(ROOT, "data", "fetch_summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"results": {str(k): v for k, v in sorted(results.items())}, "failures": failures,
                   "retrieved_utc": Time.now().utc.iso[:19]}, fh, indent=2)
    print(f"done: {len(results)} ok, {len(failures)} failed")


if __name__ == "__main__":
    main(sys.argv[1:])
