"""
DePhOCUS photometric corrections (Hoffmann et al. 2025, Icarus 426, 116366) for MPC magnitudes.

Every MPC magnitude is shifted by an offset d that depends on the observatory code, the photometric
band and the star catalog used in the reduction (the MPC one-character catalog code):
    m_corrected = m_reported + d
d converts the reported magnitude to the Johnson V band. It is a total offset: the rules of a correction
file (correction_files/corr_DePhOCUS_<level>.txt, supplementary data of the paper) are applied in file
order, band -> band+catalog -> band+catalog+station, and the most specific matching rule replaces the
others. The paper recommends the file of confidence level 0.90 (g90), which is the default here.

The workbooks of this repository do not contain the catalog code, so it is attached to every
measurement by matching the workbook rows (station, band, epoch, magnitude) with the MPC observation
records downloaded by mpc_catalog/fetch_mpc_obs.py.
"""
from __future__ import annotations

import ast
import json
import os
from functools import lru_cache

import numpy as np
import pandas as pd
from astropy.time import Time

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
CORR_DIR = os.path.join(REPO_DIR, "correction_files")
RAW_DIR = os.path.join(REPO_DIR, "mpc_catalog", "raw")

# ADES catalog names -> MPC one-character catalog codes (MPC "Astrometric catalogue codes").
ADES_TO_MPC = {
    "USNOA1": "a", "USNOSA1": "b", "USNOA2": "c", "USNOSA2": "d", "UCAC1": "e", "Tyc1": "f", "Tyc2": "g",
    "GSC1.0": "h", "GSC1.1": "i", "GSC1.2": "j", "GSC2.2": "k", "ACT": "l", "GSCACT": "m", "SDSS8": "n",
    "USNOB1": "o", "PPM": "p", "UCAC4": "q", "UCAC2": "r", "USNOB2": "s", "PPMXL": "t", "UCAC3": "u",
    "NOMAD": "v", "CMC14": "w", "Hip2": "x", "Hip1": "y", "GSC": "z", "AC": "A", "SAO1984": "B", "SAO": "C",
    "AGK3": "D", "FK4": "E", "ACRS": "F", "LickGas": "G", "Ida93": "H", "Perth70": "I", "COSUKST": "J",
    "Yale": "K", "2MASS": "L", "GSC2.3": "M", "SDSS7": "N", "SSTRC1": "O", "MPOSC3": "P", "CMC15": "Q",
    "SSTRC4": "R", "URAT1": "S", "URAT2": "T", "Gaia1": "U", "Gaia2": "V", "Gaia3": "W", "Gaia3E": "X",
    "UCAC5": "Y", "ATLAS2": "Z", "IHW": "0", "PS1_DR1": "1", "PS1_DR2": "2", "Gaia_Int": "3", "GZ": "4",
    "UBSC": "5", "Gaia_2016": "6",
}


def normalise_band(band) -> str:
    """ADES band (e.g. 'Ao', 'Pw', 'Vj') -> the one-letter band used in the sheet names."""
    if band is None:
        return ""
    b = str(band).strip()
    if b in ("Vj", "Rc", "Ic", "Bj", "Uj"):
        return b[0]
    if len(b) == 2 and b[0] in "PSALM":
        return b[1]
    return b


DEFAULT_LEVEL = "g90"


@lru_cache(maxsize=None)
def load_table(level: str = DEFAULT_LEVEL):
    """Rules of one DePhOCUS correction file: band+catalog+station, band+catalog and band-only offsets.
    Later rules of the same kind overwrite earlier ones (file order, as in the paper)."""
    rows = [ast.literal_eval(line) for line in open(os.path.join(CORR_DIR, f"corr_DePhOCUS_{level}.txt"))
            if line.strip()]
    station, band_cat, band_only = {}, {}, {}
    for r in rows:
        if r.get("code") is None and r.get("catalog") is None:
            band_only[r["band"]] = float(r["d"])
        elif r.get("code") is None:
            band_cat[(r["band"], r["catalog"])] = float(r["d"])
        else:
            station[(r["code"], r["band"], r.get("catalog") or "")] = float(r["d"])
    return station, band_cat, band_only


def offsets(code: str, band: str, catalogs, level: str = DEFAULT_LEVEL):
    """Offset d and the rule that gave it ('station', 'catalog' or 'band') for every catalog code."""
    station, band_cat, band_only = load_table(level)
    d = np.empty(len(catalogs), dtype=float)
    src = np.empty(len(catalogs), dtype=object)
    for i, cat in enumerate(catalogs):
        cat = cat if isinstance(cat, str) else ""
        if (code, band, cat) in station:
            d[i], src[i] = station[(code, band, cat)], "station"
        elif (band, cat) in band_cat:
            d[i], src[i] = band_cat[(band, cat)], "catalog"
        else:
            d[i], src[i] = band_only.get(band, 0.0), "band"
    return d, src


@lru_cache(maxsize=None)
def _mpc_records(num: int) -> pd.DataFrame:
    path = os.path.join(RAW_DIR, f"{num}_ades.json")
    df = pd.DataFrame(json.load(open(path, encoding="utf-8")))
    df["mag_f"] = pd.to_numeric(df.get("mag"), errors="coerce")
    df = df[np.isfinite(df["mag_f"])].copy()
    df["band1"] = df["band"].map(normalise_band)
    df["jd"] = Time(list(df["obstime"].str.replace("Z", "", regex=False)), format="isot", scale="utc").jd
    df["cat"] = df["astcat"].map(lambda c: ADES_TO_MPC.get(str(c), "") if c not in (None, "UNK") else "")
    return df[["stn", "band1", "jd", "mag_f", "cat", "astcat"]].reset_index(drop=True)


def match_catalogs(num: int, sheet: str, epoch, mag, tol_days: float = 5e-5, tol_mag: float = 0.011):
    """MPC catalog code of every workbook row (station = sheet[:3], band = sheet[3]).

    Rows are matched to the MPC records of the same station and band by time (within tol_days)
    and magnitude. Unmatched rows get catalog None (-> band-only offset)."""
    code, band = sheet[:3], sheet[3] if len(sheet) > 3 else ""
    rec = _mpc_records(num)
    rec = rec[(rec["stn"] == code) & (rec["band1"] == band)]
    epoch = np.asarray(epoch, dtype=float)
    mag = np.asarray(mag, dtype=float)
    out = np.full(len(epoch), None, dtype=object)
    if rec.empty:
        return out
    order = np.argsort(rec["jd"].to_numpy())
    jd = rec["jd"].to_numpy()[order]
    mg = rec["mag_f"].to_numpy()[order]
    cat = rec["cat"].to_numpy()[order]
    pos = np.searchsorted(jd, epoch)
    for i, (t, m, p) in enumerate(zip(epoch, mag, pos)):
        best, best_dt = None, tol_days
        for k in range(max(p - 3, 0), min(p + 3, len(jd))):
            dt = abs(jd[k] - t)
            if dt <= best_dt and abs(mg[k] - m) <= tol_mag:
                best, best_dt = k, dt
        if best is not None:
            out[i] = cat[best]
    return out


def sheet_offsets(num: int, sheet: str, epoch, mag, level: str = DEFAULT_LEVEL):
    """DePhOCUS offsets for the rows of one workbook sheet, with the matched catalogs and sources."""
    code, band = sheet[:3], sheet[3] if len(sheet) > 3 else ""
    cats = match_catalogs(num, sheet, epoch, mag)
    d, src = offsets(code, band, [c if c is not None else "" for c in cats], level=level)
    unmatched = np.array([c is None for c in cats])
    band_only = load_table(level)[2]
    d[unmatched] = band_only.get(band, 0.0)
    src[unmatched] = "band (unmatched)"
    return d, cats, src
