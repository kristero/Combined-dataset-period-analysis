"""Significance of the two geometry terms separately (per asteroid, from the template windows):
  M0: g = A0, mu = mu0 + c1 x               (no geometry terms)
  Mp: g = A0 (1 + m alpha)                  (phase-angle term only)
  Ma: aspect model with m = 0               (aspect term only)
  Map: aspect model with free m             (both)
BIC = chi2 + k ln(2 n_windows). Also compares the equatorial zero-phase amplitude 2.5 log(a/b) with the largest
dense-photometry LCDB amplitude. Output: geometry_terms.json"""
import json
import os

import numpy as np

import amp_geometry as G

ROOT = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(ROOT, "geometry_results.json"), encoding="utf-8"))
DENSE = json.load(open(os.path.join(ROOT, "data", "lcdb_dense_refs.json"), encoding="utf-8"))
MS_SAVE = G.MS.copy()
out, cnt = {}, {"phase_term_sig": 0, "aspect_term_sig": 0, "both_better_than_aspect": 0, "n": 0}
pairs = []
for k, r in R.items():
    if not r or not r.get("geometry"):
        continue
    rows = r["windows"]
    n = len(rows)
    g = np.array([w["g"] for w in rows])
    sg = np.array([w["sigma_g"] for w in rows])
    al = np.array([w["alpha_mean"] for w in rows])
    wA = 1.0 / (sg ** 2 + G.SIG_A_SYS ** 2 + (G.SIG_A_REL * g) ** 2)
    # amplitude part only (the mean-brightness part is the same for M0/Mp and for Ma/Map)
    c0 = float(np.sum(wA * (g - np.sum(wA * g) / np.sum(wA)) ** 2))
    cp = min(float(np.sum(wA * (g - A0 * (1 + m * al)) ** 2)) for m in G.MS
             for A0 in [np.sum(wA * g * (1 + m * al)) / np.sum(wA * (1 + m * al) ** 2)])
    G.MS = np.array([0.0])
    pa = G.fit(rows)
    G.MS = MS_SAVE
    pap = r["geometry"]
    nobs = 2 * n
    ln = np.log(nobs)
    # full BIC (both parts): M0 4 params (A0, mu0, c1 + nothing), Mp 4 (A0, m, mu0, c1), Ma 6, Map 7
    chiM_null = pap["chi2_null"] - cp          # mean-brightness chi2 of the null model
    bic = {"M0": c0 + chiM_null + 3 * ln, "Mp": cp + chiM_null + 4 * ln,
           "Ma": pa["chi2"] + 6 * ln, "Map": pap["chi2"] + 7 * ln}
    best = min(bic, key=bic.get)
    out[k] = {"bic": bic, "best": best, "m": pap["m"], "a_b": pap["a_b"], "A_eq0": pap["A_eq0"]}
    cnt["n"] += 1
    cnt["phase_term_sig"] += int(bic["M0"] - bic["Mp"] > 10)
    cnt["aspect_term_sig"] += int(bic["Mp"] - bic["Map"] > 10)
    cnt["both_better_than_aspect"] += int(bic["Ma"] - bic["Map"] > 10)
    num = k.split(":")[1]
    if DENSE.get(num, {}).get("amps") and pap["delta_bic"] > 10:
        pairs.append((pap["A_eq0"], DENSE[num]["max"], r["APP_TPL_wmean"]))
print(f"{cnt['n']} asteroids: phase-angle term significant (Delta BIC > 10, no aspect) in {cnt['phase_term_sig']}; "
      f"aspect term significant beyond the phase term in {cnt['aspect_term_sig']}; "
      f"phase term significant beyond the aspect term in {cnt['both_better_than_aspect']}")
best = {}
for v in out.values():
    best[v["best"]] = best.get(v["best"], 0) + 1
print("preferred model by BIC:", best)
ms = np.array([v["m"] for v in out.values() if v["best"] in ("Map", "Mp")])
print(f"fitted m where a phase term is preferred: median {np.median(ms):.3f}, 16-84% {np.percentile(ms, 16):.3f}-{np.percentile(ms, 84):.3f} (n={len(ms)})")
p = np.array(pairs)
if len(p):
    print(f"equatorial zero-phase amplitude 2.5 log(a/b) vs largest dense LCDB amplitude (aspect model significant, n={len(p)}): "
          f"median ratio {np.median(p[:, 0] / p[:, 1]):.2f}, scatter {1.4826 * np.median(np.abs(np.log10(p[:, 0] / p[:, 1]) - np.median(np.log10(p[:, 0] / p[:, 1])))):.3f} dex; "
          f"per-apparition mean / largest LCDB: {np.median(p[:, 2] / p[:, 1]):.2f}")
json.dump({"per_object": out, "counts": cnt, "preferred": best}, open(os.path.join(ROOT, "geometry_terms.json"), "w"), indent=1)
