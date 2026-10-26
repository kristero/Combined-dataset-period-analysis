"""
Checks that the DAMIT input files satisfy what convexinv/period_scan expect (manual v0.2.1 and the C source:
convexinv.c reads JD, brightness, Sun vector, Earth vector; bright.c uses only the directions of the vectors):
  1 epochs are light-time corrected: JD_lt = JD_obs - Delta * tau_AU  (tau_AU = 499.004784 s)
  2 brightness is reduced to unit distances and NOT phase corrected: H_red = m - 5 log10(r Delta)
  3 the lcs file holds JD_lt and 10^(-0.4 (H_red - 10)) for exactly the points of the analysis
  4 the vectors are asteroid-centric (Sun first, then Earth), in AU, ecliptic J2000: |Sun| = r, |Earth| = Delta,
    angle(Sun, Earth) = phase angle, Earth - Sun = heliocentric Earth position (astropy, ecliptic J2000)
  5 each dataset (station + band) is one relative lightcurve (flag 0) with a single band
Output: damit_runs/input_check.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import amp_data as D  # noqa: E402
import damit_model as M  # noqa: E402
import damit_prep as DP  # noqa: E402

TAU_AU_DAYS = 499.004784 / 86400.0
RUNS = os.path.join(ROOT, "damit_runs")


def check(sample, num):
    run = os.path.join(RUNS, f"{sample}_{num}")
    prep = np.load(os.path.join(run, "prep.npz"), allow_pickle=True)
    obs = pd.read_csv(os.path.join(ROOT, "data", "obs_tables", f"{num}_obs.csv") if sample == "paper" else
                      os.path.join(D.VAL, "data", "obs_tables", f"{num}_obs.csv"))
    d = D.load(sample, num)
    a = np.load(os.path.join(ROOT, "results", num, f"{num}_arrays.npz")) if sample == "paper" else \
        np.load(os.path.join(D.VAL, "results", f"{num}_arrays.npz"), allow_pickle=True)
    t, sheet, t0 = a["t"].astype(float), a["sheet"].astype(str), float(a["time_zero"])
    idx = DP.match_obs(t, sheet, t0, obs)
    o = obs.loc[idx[prep["src_index"]]]
    jd, jd_lt, r, de, al = (o[c].to_numpy(float) for c in ("jd", "jd_lt", "r", "delta", "alpha"))
    m, H = o["mag_f"].to_numpy(float), o["H_red"].to_numpy(float)
    out = {}
    out["lt_resid_s"] = float(np.max(np.abs((jd - jd_lt) - de * TAU_AU_DAYS)) * 86400)
    out["lt_sign_ok"] = bool(np.all(jd_lt < jd))
    out["Hred_resid_mag"] = float(np.max(np.abs(H - (m - 5 * np.log10(r * de)))))
    out["prep_matches_table"] = bool(np.allclose(prep["jd_lt"], jd_lt) and np.allclose(prep["H_red"], H))
    out["analysis_times_match"] = float(np.max(np.abs(prep["jd_lt"] - (t[prep["src_index"]] + t0))) * 86400)
    L = M.read_lcs(os.path.join(run, "lcs"))
    fo = prep["file_order"]
    out["lcs_time_resid_s"] = float(np.max(np.abs(L["jd"] - prep["jd_lt"][fo])) * 86400)
    out["lcs_brightness_rel_resid"] = float(np.max(np.abs(L["br"] / 10 ** (-0.4 * (prep["H_red"][fo] - 10)) - 1)))
    sun, ear = L["sun"], L["earth"]
    out["sun_len_resid_au"] = float(np.max(np.abs(np.linalg.norm(sun, axis=1) - r[fo])))
    out["earth_len_resid_au"] = float(np.max(np.abs(np.linalg.norm(ear, axis=1) - de[fo])))
    ca = np.sum(sun * ear, axis=1) / np.linalg.norm(sun, axis=1) / np.linalg.norm(ear, axis=1)
    out["phase_angle_resid_deg"] = float(np.max(np.abs(np.degrees(np.arccos(np.clip(ca, -1, 1))) - al[fo])))
    E = DP.earth_helio_ecl(jd[fo])
    out["earth_position_resid_au"] = float(np.max(np.linalg.norm((ear - sun) - E, axis=1)))
    # lightcurves: one per dataset, flag 0, one band each
    v = open(os.path.join(run, "lcs")).read().split("\n")
    nlc, pos, flags, sizes = int(v[0]), 1, [], []
    for _ in range(nlc):
        n_, f_ = map(int, v[pos].split())
        flags.append(f_)
        sizes.append(n_)
        pos += n_ + 1
    sets = list(dict.fromkeys(prep["sheet"].astype(str)[fo]))
    # band1: normalized band; MPC labels such as o/Ao (ATLAS) or w/Pw (Pan-STARRS) are the same filter
    bands = {s: sorted(set(o["band1"].astype(str).to_numpy()[prep["sheet"].astype(str) == s])) for s in sets}
    out["n_lc"] = nlc
    out["all_relative"] = bool(all(f == 0 for f in flags))
    out["one_lc_per_dataset"] = bool(nlc == len(sets) and sizes == [int(np.sum(prep["sheet"].astype(str) == s)) for s in sets])
    out["single_band_per_lc"] = bool(all(len(b) == 1 for b in bands.values()))
    return out


def main():
    res = {}
    for s, n in D.all_ids():
        try:
            res[f"{s}:{n}"] = check(s, n)
        except Exception as exc:  # noqa: BLE001
            res[f"{s}:{n}"] = {"error": repr(exc)[:300]}
    ok = {k: v for k, v in res.items() if "error" not in v}
    summ = {"n": len(res), "errors": [k for k in res if k not in ok]}
    for key in ("lt_resid_s", "Hred_resid_mag", "analysis_times_match", "lcs_time_resid_s", "lcs_brightness_rel_resid",
                "sun_len_resid_au", "earth_len_resid_au", "phase_angle_resid_deg", "earth_position_resid_au"):
        summ[key + "_max"] = float(max(v[key] for v in ok.values()))
    for key in ("lt_sign_ok", "prep_matches_table", "all_relative", "one_lc_per_dataset", "single_band_per_lc"):
        summ[key + "_all"] = bool(all(v[key] for v in ok.values()))
        summ[key + "_failures"] = [k for k, v in ok.items() if not v[key]][:10]
    json.dump({"summary": summ, "per_asteroid": res}, open(os.path.join(RUNS, "input_check.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
