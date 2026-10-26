"""
Point-level triaxial-ellipsoid inversion of the same lightcurve files as the convex inversion (damit_runs/*/lcs),
with the same scattering law, phase function and relative-lightcurve chi^2 as convexinv, so that the only
difference to the convex model is the shape (3 shape parameters a/b, b/c, phi0 instead of 48 coefficients).

Parameters: lambda, beta, P, phi0, ln(a/b) >= 0, ln(b/c) >= 0, phase slope k (a = 0.5, d = 0.1 and Lambert c = 0.1
fixed, as in the convex runs).
Starts: the 12 initial poles of the convex runs x phi0 = 0, 90 deg; P starts at the best scan period of the
convex run. Facets: convexinv's Gaussian-sphere triangulation (6 rows), areas A(n) = (abc)^2 h(n)^-4 dOmega.
Also refits the best solution without each template window (leave-one-apparition-out).
Output: damit_runs/<run>/ellipsoid.json
Usage: python damit_ellipsoid.py [--workers 18] [--only val|paper] [--nums a,b]
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
RUNS = os.path.join(ROOT, "damit_runs")
POLES = [(0.0, 85.0), (180.0, -85.0)] + [(72.0 * k, 26.57) for k in range(5)] + \
        [(36.0 + 72.0 * k, -26.57) for k in range(5)]          # same 12 starts as damit_run.POLES
PHI0 = (0.0, 90.0)
C_LAMBERT = 0.1


class EllFit:
    def __init__(self, L, nrows=6):
        import damit_model as M
        self.M = M
        self.jd, self.br, self.lc = L["jd"], L["br"], L["lc"]
        self.e = M.unit(L["earth"])
        self.e0 = M.unit(L["sun"])
        self.alpha = np.arccos(np.clip(np.sum(self.e * self.e0, axis=1), -1, 1))
        self.nor, self.dO = M.gaussian_sphere(nrows)
        self.jd0 = float(int(self.jd.min()))
        self.lcs = [np.where(self.lc == k)[0] for k in np.unique(self.lc)]
        self.ave_o = np.empty(len(self.br))
        for m in self.lcs:
            self.ave_o[m] = self.br[m].mean()

    def model(self, x, idx=None):
        lam, bet, P, phi0, u, v, pa, pd, pk = x
        idx = slice(None) if idx is None else idx
        T = self.M.rotation(lam, bet, P, self.jd0, phi0, self.jd[idx])
        e = np.einsum("nij,nj->ni", T, self.e[idx])
        e0 = np.einsum("nij,nj->ni", T, self.e0[idx])
        c = 1.0
        b = c * np.exp(v)
        a = b * np.exp(u)
        A = self.M.ellipsoid_areas(self.nor, self.dO, a, b, c)
        br = self.M.facet_sum(A, self.nor, e, e0, C_LAMBERT)
        return br * self.M.phase_function(self.alpha[idx], pa, pd, pk)

    def residuals(self, x, keep=None):
        m = self.model(x)
        r = np.zeros_like(m)
        for s in self.lcs:
            if keep is not None:
                s = s[keep[s]]
                if len(s) < 3:
                    continue
            r[s] = (self.br[s] - m[s] * self.br[s].mean() / m[s].mean()) / self.br[s].mean()
        if keep is not None:
            return r[keep]
        return r

    # phase function as in the convex runs: a = 0.5, d = 0.1 fixed, slope k fitted
    FREE = [0, 1, 2, 3, 4, 5, 8]

    def fit(self, x0, P_scale, keep=None, max_nfev=60):
        from scipy.optimize import least_squares
        x0 = np.array(x0, float)
        x0[6], x0[7] = 0.5, 0.1
        lo = np.array([-np.inf, -90, x0[2] - P_scale, -np.inf, 0.0, 0.0, 0.0, 0.02, -2.0])[self.FREE]
        hi = np.array([np.inf, 90, x0[2] + P_scale, np.inf, 1.6, 1.2, 5.0, 1.0, 2.0])[self.FREE]
        z0 = np.clip(x0[self.FREE], lo + 1e-9, hi - 1e-9)

        def f(z):
            x = x0.copy()
            x[self.FREE] = z
            return self.residuals(x, keep)

        res = least_squares(f, z0, bounds=(lo, hi), x_scale="jac", diff_step=1e-7, max_nfev=max_nfev)
        x = x0.copy()
        x[self.FREE] = res.x
        dev = float(np.sqrt(np.mean(res.fun ** 2)))
        return x, dev


def run_one(run):
    out_path = os.path.join(run, "ellipsoid.json")
    if os.path.exists(out_path):
        return run, "cached"
    try:
        import damit_model as M
        import amp_data as D
        import amp_methods as AM
        info = json.load(open(os.path.join(run, "prep.json")))
        L = M.read_lcs(os.path.join(run, "lcs"))
        F = EllFit(L)
        poles = json.load(open(os.path.join(run, "poles", "poles.json")))
        P0 = poles["P_scan_best"]
        span_h = (F.jd.max() - F.jd.min()) * 24
        P_scale = 0.5 * P0 * P0 / span_h                     # half the separation of chi2 minima in period
        sols = []
        for lam, bet in POLES:
            for phi in PHI0:
                x, dev = F.fit(np.array([lam, bet, P0, phi, 0.3, 0.1, 0.5, 0.1, -0.5]), P_scale, max_nfev=25)
                sols.append({"start": [lam, bet, phi], "x": x.tolist(), "dev": dev})
        best = min(sols, key=lambda s: s["dev"])
        xb = np.array(best["x"])
        xb, devb = F.fit(xb, P_scale, max_nfev=200)           # polish
        # leave-one-apparition-out, same windows as the convex runs
        d = D.load(info["sample"], info["num"])
        tp = AM.template_fit(d)
        prep = np.load(os.path.join(run, "prep.npz"), allow_pickle=True)
        order = prep["file_order"]                          # lcs line k -> prep row
        pos_of_src = np.full(len(d["t"]), -1)
        pos_of_src[prep["src_index"]] = np.arange(len(prep["src_index"]))
        line_of_row = np.empty(len(order), int)
        line_of_row[order] = np.arange(len(order))
        loao = []
        for j, r in enumerate(tp["rows"] if tp else []):
            keep = np.ones(len(F.br), bool)
            keep[line_of_row[pos_of_src[r["idx"]]]] = False
            xj, devj = F.fit(xb.copy(), P_scale, keep=keep, max_nfev=60)
            loao.append({"j": j, "x": xj.tolist(), "dev": devj})
        names = ["lambda", "beta", "P", "phi0", "ln_ab", "ln_bc", "a", "d", "k"]
        json.dump({"jd0": F.jd0, "names": names, "best": dict(zip(names, map(float, xb))), "dev": devb,
                   "solutions": sols, "loao": loao}, open(out_path, "w"), indent=1)
        return run, f"dev {devb:.5f} (a/b {np.exp(xb[4]):.2f}, b/c {np.exp(xb[5]):.2f})"
    except Exception:  # noqa: BLE001
        return run, "FAILED " + traceback.format_exc()[-500:]


def main(argv):
    workers = int(argv[argv.index("--workers") + 1]) if "--workers" in argv else 18
    only = argv[argv.index("--only") + 1] if "--only" in argv else None
    nums = set(argv[argv.index("--nums") + 1].split(",")) if "--nums" in argv else None
    runs = []
    for d in sorted(os.listdir(RUNS)):
        p = os.path.join(RUNS, d)
        if not os.path.isfile(os.path.join(p, "poles", "poles.json")):
            continue
        s, n = d.split("_", 1)
        if (only and s != only) or (nums and n not in nums):
            continue
        runs.append(p)
    with mp.get_context("spawn").Pool(workers) as pool:
        for k, (r, msg) in enumerate(pool.imap_unordered(run_one, runs)):
            print(f"[{k + 1}/{len(runs)}] {os.path.basename(r)}: {msg}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
