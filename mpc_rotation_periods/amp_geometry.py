"""
Viewing-geometry model for multi-apparition sparse photometry (per asteroid).

Body: triaxial ellipsoid a >= b >= c (c = 1) rotating about c, geometrically scattering (brightness proportional to the
projected area, as in Cibulkova et al. 2016 and the sHG1G2 aspect term of Carry et al. 2024). For a viewing direction
at aspect angle xi from the spin axis and rotation angle phi:
    L(phi, xi) = sqrt( sin^2 xi cos^2 phi / a^2 + sin^2 xi sin^2 phi / b^2 + cos^2 xi / c^2 )      (x abc, constant)
  mean brightness (rotation average):   M(xi) = -2.5 log10 <L>_phi
  amplitude (zero phase angle):         A(xi) = 1.25 log10[ (sin^2 xi / b^2 + cos^2 xi / c^2) / (sin^2 xi / a^2 + cos^2 xi / c^2) ]
                                        (identical to Thirouin et al. 2016, Eq. 4: 2.5 log(a/b) at xi = 90 deg, 0 pole-on)
  amplitude-phase relation:             A(xi, alpha) = (1 + m alpha) A(xi)               (Zappala et al. 1990)
The viewing direction is the phase-angle bisector (PAB) from JPL Horizons.

Per apparition j (template fit of amp_methods: amplitude g_j, mean level mu_j, mean PAB vector e_j, mean phase angle):
    g_j  = (1 + m alpha_j) A(xi_j)
    mu_j = mu_0 + c_1 (alpha_j - <alpha>) + M(xi_j)          (c_1 absorbs residual phase-curve misfit)
    cos xi_j = |v . e_j|,  v = (cos beta cos lambda, cos beta sin lambda, sin beta)  (ecliptic spin axis)
Grid fit (chi^2 of both data sets): lambda 0-355 deg and beta 0-90 deg in 5-deg steps (the observables depend on
|cos xi|, so v and -v are equivalent), a/b 1.0-3.0, b/c 1.0-2.0, m 0-0.05 per degree.
Null model (no aspect dependence): g_j = A_0 (1 + m alpha_j), mu_j = mu_0 + c_1 (alpha_j - <alpha>).
"""
from __future__ import annotations

import numpy as np

AB = np.round(np.arange(1.0, 3.001, 0.05), 3)
BC = np.round(np.arange(1.0, 2.001, 0.1), 3)
MS = np.round(np.arange(0.0, 0.0501, 0.005), 4)
XI_DEG = np.linspace(0.0, 90.0, 181)
PHI = np.linspace(0.0, np.pi, 361)
LAM = np.arange(0, 360, 5.0)
BET = np.arange(0, 91, 5.0)
SIG_A_SYS, SIG_A_REL, SIG_MU_SYS = 0.03, 0.05, 0.03


def ellipsoid_tables():
    """A(xi) and M(xi) for every shape on the XI_DEG grid. Returns arrays [n_ab, n_bc, n_xi]."""
    xi = np.radians(XI_DEG)
    s2, c2 = np.sin(xi) ** 2, np.cos(xi) ** 2
    A = np.zeros((len(AB), len(BC), len(xi)))
    Mm = np.zeros_like(A)
    for i, ab in enumerate(AB):
        for k, bc in enumerate(BC):
            c, b = 1.0, bc
            a = ab * b
            A[i, k] = 1.25 * np.log10((s2 / b ** 2 + c2 / c ** 2) / (s2 / a ** 2 + c2 / c ** 2))
            L = np.sqrt(np.outer(s2, np.cos(PHI) ** 2) / a ** 2 + np.outer(s2, np.sin(PHI) ** 2) / b ** 2
                        + c2[:, None] / c ** 2)
            Mm[i, k] = -2.5 * np.log10(L.mean(axis=1))
    return A, Mm


A_TAB, M_TAB = ellipsoid_tables()


def pole_vectors():
    L, B = np.meshgrid(np.radians(LAM), np.radians(BET), indexing="ij")
    V = np.stack([np.cos(B) * np.cos(L), np.cos(B) * np.sin(L), np.sin(B)], axis=-1)
    return V.reshape(-1, 3), L.ravel(), B.ravel()


def xi_index(cosxi):
    xi = np.degrees(np.arccos(np.clip(cosxi, 0.0, 1.0)))
    return np.clip(np.rint(xi / (XI_DEG[1] - XI_DEG[0])).astype(int), 0, len(XI_DEG) - 1)


def wls_offset_slope(y, x, w):
    """Weighted least squares y = p0 + p1 x for many columns at once (y: [n_win, n_col]). Returns chi2 per column."""
    W = w[:, None]
    Sw, Sx, Sxx = W.sum(0), (W * x[:, None]).sum(0), (W * x[:, None] ** 2).sum(0)
    Sy, Sxy = (W * y).sum(0), (W * x[:, None] * y).sum(0)
    det = Sw * Sxx - Sx ** 2
    p0 = (Sxx * Sy - Sx * Sxy) / det
    p1 = (Sw * Sxy - Sx * Sy) / det
    r = y - p0 - p1 * x[:, None]
    return (W * r ** 2).sum(0), p0, p1


def fit(rows, min_windows=4):
    """rows: template windows (g, sigma_g, mu, sigma_mu, alpha_mean, pab_vec). Returns the best model, the chi2 of the
    null model, Delta BIC, and the chi2 map over poles (profiled over shape and m)."""
    rows = [r for r in rows if np.isfinite(r["g"]) and np.isfinite(r["mu"])]
    n = len(rows)
    if n < min_windows:
        return None
    g = np.array([r["g"] for r in rows])
    sg = np.array([r["sigma_g"] for r in rows])
    mu = np.array([r["mu"] for r in rows])
    smu = np.array([r["sigma_mu"] for r in rows])
    al = np.array([r["alpha_mean"] for r in rows])
    E = np.array([r["pab_vec"] for r in rows])
    wA = 1.0 / (sg ** 2 + SIG_A_SYS ** 2 + (SIG_A_REL * g) ** 2)
    wM = 1.0 / (smu ** 2 + SIG_MU_SYS ** 2)
    x = al - al.mean()
    V, LAMr, BETr = pole_vectors()
    xi_idx = xi_index(np.abs(V @ E.T))                    # [n_pole, n_win]
    # null model
    chi_null_A = min(float(np.sum(wA * (g - A0 * (1 + m * al)) ** 2)) for m in MS
                     for A0 in [np.sum(wA * g * (1 + m * al)) / np.sum(wA * (1 + m * al) ** 2)])
    chi_null_M, _, _ = wls_offset_slope(mu[:, None], x, wM)
    chi_null = chi_null_A + float(chi_null_M[0])
    best = (np.inf, None)
    chi_pole = np.full(len(V), np.inf)
    for i in range(len(AB)):
        for k in range(len(BC)):
            Aw = A_TAB[i, k][xi_idx]                      # [n_pole, n_win]
            Mw = M_TAB[i, k][xi_idx]
            cM, p0, p1 = wls_offset_slope((mu[None, :] - Mw).T, x, wM)   # per pole
            fac = 1 + np.outer(MS, al)                    # [n_m, n_win]
            resA = g[None, None, :] - fac[None, :, :] * Aw[:, None, :]
            cA = np.einsum("pmw,w->pm", resA ** 2, wA)    # [n_pole, n_m]
            im = np.argmin(cA, axis=1)
            tot = cA[np.arange(len(V)), im] + cM
            chi_pole = np.minimum(chi_pole, tot)
            j = int(np.argmin(tot))
            if tot[j] < best[0]:
                best = (float(tot[j]), {"lambda": float(np.degrees(LAMr[j])), "beta": float(np.degrees(BETr[j])),
                                        "a_b": float(AB[i]), "b_c": float(BC[k]), "m": float(MS[im[j]]),
                                        "mu0": float(p0[j]), "c1": float(p1[j]), "chi2_A": float(cA[j, im[j]]),
                                        "chi2_M": float(cM[j]), "xi_deg": XI_DEG[xi_idx[j]].tolist()})
    chi_best, par = best
    nobs = 2 * n
    dbic = (chi_null - chi_best) - 3 * np.log(nobs)       # aspect model: 7 parameters, null model: 4
    red = chi_best / max(nobs - 7, 1)
    ok = chi_pole <= chi_best + 4.6 * max(red, 1.0)        # ~90 % joint region for 2 parameters, scaled
    par.update({"chi2": chi_best, "chi2_null": chi_null, "delta_bic": float(dbic), "n_windows": n,
                "reduced_chi2": float(red), "A_eq0": float(2.5 * np.log10(par["a_b"])),
                "pole_region_fraction": float(np.mean(ok)),
                "pole_region": [[float(np.degrees(a_)), float(np.degrees(b_))] for a_, b_ in zip(LAMr[ok], BETr[ok])][:400]})
    return par


def model_at(par, pab_vec, alpha):
    """Amplitude and mean-brightness term of the fitted model for viewing directions pab_vec [n,3] and phase angles."""
    i = int(np.argmin(np.abs(AB - par["a_b"])))
    k = int(np.argmin(np.abs(BC - par["b_c"])))
    v = np.array([np.cos(np.radians(par["beta"])) * np.cos(np.radians(par["lambda"])),
                  np.cos(np.radians(par["beta"])) * np.sin(np.radians(par["lambda"])), np.sin(np.radians(par["beta"]))])
    xi = xi_index(np.abs(np.asarray(pab_vec) @ v))
    return (1 + par["m"] * np.asarray(alpha)) * A_TAB[i, k][xi], M_TAB[i, k][xi]


def pab_vectors(lon_deg, lat_deg):
    lam, bet = np.radians(lon_deg), np.radians(lat_deg)
    return np.vstack([np.cos(bet) * np.cos(lam), np.cos(bet) * np.sin(lam), np.sin(bet)]).T


def pole_distance(l1, b1, l2, b2):
    """Smallest angular distance between pole 1 and the observationally equivalent solutions of pole 2:
    v, -v (the observables depend on |cos xi|) and the lambda + 180 deg mirror."""
    def vec(l, b):
        l, b = np.radians(l), np.radians(b)
        return np.array([np.cos(b) * np.cos(l), np.cos(b) * np.sin(l), np.sin(b)])
    v1 = vec(l1, b1)
    d = []
    for l, b in ((l2, b2), (l2 + 180, b2)):
        v2 = vec(l, b)
        d.append(np.degrees(np.arccos(np.clip(abs(v1 @ v2), -1, 1))))
    return float(min(d))
