"""
Absolute magnitudes in the V band from the DePhOCUS-corrected MPC magnitudes (Hoffmann et al. 2025).

For every asteroid, the datasets of the combined analysis (paper_figures.CONFIG) are reduced to 1 AU and
corrected per measurement with the DePhOCUS offsets (band, star catalog and observatory), which bring them
to the Johnson V band. Outliers are removed in 5 deg bins of phase angle (3 sigma; 5 sigma below 7 deg).
The H, G1, G2 model is then fitted to all datasets together (no magnitude shifts between the datasets).
The uncertainties come from a bootstrap over observing nights (one night of one dataset is one block).
The offsets of the individual datasets from the common fit (median residuals) measure the remaining
systematic differences; they are given for the DePhOCUS corrections and for band offsets only.

Run: python absolute_magnitudes.py [N ...] [--tag=final]  -> paper_figures/runs/<tag>/absolute_magnitudes.json
"""
from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

import paper_figures as pf
from Asteroid_pickle import DatasetGenerator

warnings.filterwarnings("ignore")


def load(num, cfg, bias_mode):
    wdir, wname = os.path.split(cfg["workbook"])
    gen = DatasetGenerator(wdir + os.sep, wname, num, cfg["sheets"], base_dir=pf.REPO_DIR, bias_mode=bias_mode)
    ph, H, t, ds = [], [], [], []
    for s in cfg["sheets"]:
        df = pd.read_excel(cfg["workbook"], sheet_name=s).dropna(subset=["magred"])
        h = (df["mag"].to_numpy(float) - 5 * np.log10(df.iloc[:, 3].to_numpy(float) * df.iloc[:, 4].to_numpy(float))
             + gen._bias(s, df))
        p_k, h_k, rem = gen.removeOutliers(df["Ph"].to_numpy(float), h, 3.0)
        ph.append(p_k); H.append(h_k); t.append(np.delete(gen._light_time(df), rem)); ds += [s] * len(h_k)
    return gen, np.concatenate(ph), np.concatenate(H), np.concatenate(t), np.array(ds)


def fit_all(gen, ph, H):
    r = gen.fit(ph, H, method="HG1G2")
    return float(r.params["H"].value), float(r.params["G1"].value), float(r.params["G2"].value), r


def offsets(gen, ph, H, ds, Hf, G1, G2):
    res = H - gen.hg1g2_phase_function(ph, Hf, G1, G2)
    return {s: float(np.median(res[ds == s])) for s in np.unique(ds)}


def run(num, n_boot=200, seed=3):
    cfg = pf.CONFIG[num]
    out = {"asteroid": num, "sheets": cfg["sheets"]}
    for mode in ("dephocus", "band"):
        gen, ph, H, t, ds = load(num, cfg, mode)
        Hf, G1, G2, r = fit_all(gen, ph, H)
        off = offsets(gen, ph, H, ds, Hf, G1, G2)
        o = np.array(list(off.values()))
        rec = {"H": Hf, "G1": G1, "G2": G2, "H_err_formal": float(r.params["H"].stderr or np.nan),
               "n": int(len(H)), "phase_min": float(ph.min()), "phase_max": float(ph.max()),
               "dataset_offsets": off, "offset_rms": float(np.sqrt(np.mean(o ** 2))),
               "offset_std": float(np.std(o, ddof=1))}
        if mode == "dephocus":
            blocks = np.array([f"{s}|{int(np.floor(x))}" for s, x in zip(ds, t)])
            ub, inv = np.unique(blocks, return_inverse=True)
            idx = [np.where(inv == b)[0] for b in range(len(ub))]
            rng = np.random.default_rng(seed)
            bs = []
            for _ in range(n_boot):
                sel = np.concatenate([idx[b] for b in rng.integers(0, len(ub), len(ub))])
                bs.append(fit_all(gen, ph[sel], H[sel])[:3])
            bs = np.array(bs)
            rec.update(H_err=float(np.std(bs[:, 0], ddof=1)), G1_err=float(np.std(bs[:, 1], ddof=1)),
                       G2_err=float(np.std(bs[:, 2], ddof=1)))
        out[mode] = rec
    print(num, "H_V = %.2f +- %.2f, G1 = %.2f, G2 = %.2f, offset rms %.3f (band only %.3f)" % (
        out["dephocus"]["H"], out["dephocus"]["H_err"], out["dephocus"]["G1"], out["dephocus"]["G2"],
        out["dephocus"]["offset_rms"], out["band"]["offset_rms"]))
    return out


if __name__ == "__main__":
    pf._set_options([a for a in sys.argv[1:] if a.startswith("--")])
    nums = [int(a) for a in sys.argv[1:] if not a.startswith("--")] or list(pf.CONFIG)
    path = os.path.join(pf.OUT_DIR, "absolute_magnitudes.json")
    allres = json.load(open(path)) if os.path.exists(path) else {}
    for n in nums:
        allres[str(n)] = run(n)
        json.dump(allres, open(path, "w"), indent=1)
