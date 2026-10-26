"""
NumPy forward model of DAMIT convexinv solutions (same conventions as bright.c, matrix.c, blmatrix.c, phasec.c):

  e_ast = Fz(phi) B(colat, lambda) e_ecl,  phi = phi0 + (2 pi 24 / P) (jd - jd0)
  brightness = f(alpha) * sum_k A_k mu mu0 (c + 1 / (mu + mu0))   over facets with mu, mu0 > 0
  f(alpha) = 1 + a exp(-alpha / d) + k alpha                         (alpha in radians)

and of a triaxial ellipsoid with the same scattering law (Gaussian-image facet areas A(n) = (abc)^2 / h(n)^4 dOmega,
h = support function), used for the point-level ellipsoid comparison.
"""
from __future__ import annotations

import numpy as np

TINY = 1e-8


def rotation(lam_deg, beta_deg, P_h, jd0, phi0_deg, jd):
    th, la = np.deg2rad(90.0 - beta_deg), np.deg2rad(lam_deg)
    cb, sb, cl, sl = np.cos(th), np.sin(th), np.cos(la), np.sin(la)
    B = np.array([[cb * cl, cb * sl, -sb], [-sl, cl, 0.0], [sb * cl, sb * sl, cb]])
    f = np.deg2rad(phi0_deg) + 2 * np.pi * 24.0 / P_h * (np.asarray(jd) - jd0)
    cf, sf = np.cos(f), np.sin(f)
    F = np.zeros((len(f), 3, 3))
    F[:, 0, 0], F[:, 0, 1], F[:, 1, 0], F[:, 1, 1], F[:, 2, 2] = cf, sf, -sf, cf, 1.0
    return F @ B                                   # (n, 3, 3): e_ast = T @ e_ecl


def phase_function(alpha_rad, a, d, k):
    return 1.0 + a * np.exp(-alpha_rad / d) + k * alpha_rad


def facet_sum(areas, normals, e_ast, e0_ast, c_lambert, chunk=4096):
    out = np.empty(len(e_ast))
    for s in range(0, len(e_ast), chunk):
        mu = e_ast[s:s + chunk] @ normals.T
        mu0 = e0_ast[s:s + chunk] @ normals.T
        ok = (mu > TINY) & (mu0 > TINY)
        sc = np.where(ok, mu * mu0 * (c_lambert + 1.0 / np.where(ok, mu + mu0, 1.0)), 0.0)
        out[s:s + chunk] = sc @ areas
    return out


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


class Model:
    """A convex (or any facet) model with spin and scattering parameters."""

    def __init__(self, areas, normals, par):
        self.areas, self.normals, self.par = np.asarray(areas), np.asarray(normals), dict(par)

    def brightness(self, jd, sun, earth, with_phase=True):
        p = self.par
        T = rotation(p["lambda"], p["beta"], p["P"], p["t0"], p["phi0"], jd)
        e = np.einsum("nij,nj->ni", T, unit(earth))
        e0 = np.einsum("nij,nj->ni", T, unit(sun))
        br = facet_sum(self.areas, self.normals, e, e0, p["c"])
        if with_phase:
            alpha = np.arccos(np.clip(np.sum(unit(sun) * unit(earth), axis=1), -1, 1))
            br = br * phase_function(alpha, p["a"], p["d"], p["k"])
        return br

    def over_rotation(self, sun, earth, nphi=72, with_phase=True):
        """Brightness at fixed geometry for nphi rotation phases: array (len(sun), nphi)."""
        p = self.par
        sun, earth = np.atleast_2d(sun), np.atleast_2d(earth)
        jd = p["t0"] + (np.arange(nphi) / nphi) * p["P"] / 24.0
        T = rotation(p["lambda"], p["beta"], p["P"], p["t0"], p["phi0"], jd)          # (nphi,3,3)
        e = np.einsum("pij,nj->npi", T, unit(earth)).reshape(-1, 3)
        e0 = np.einsum("pij,nj->npi", T, unit(sun)).reshape(-1, 3)
        br = facet_sum(self.areas, self.normals, e, e0, p["c"]).reshape(len(sun), nphi)
        if with_phase:
            alpha = np.arccos(np.clip(np.sum(unit(sun) * unit(earth), axis=1), -1, 1))
            br = br * phase_function(alpha, p["a"], p["d"], p["k"])[:, None]
        return br

    def mean_mag(self, sun, earth, nphi=72, with_phase=True):
        return -2.5 * np.log10(self.over_rotation(sun, earth, nphi, with_phase).mean(axis=1))

    def amplitude(self, sun, earth, nphi=180):
        m = -2.5 * np.log10(self.over_rotation(sun, earth, nphi, with_phase=False))
        return m.max(axis=1) - m.min(axis=1)


def relative_dev(model_br, obs_br, lc_index):
    """convexinv's rms deviation: each lightcurve scaled to the observed mean, residuals relative to that mean."""
    res = np.empty_like(obs_br)
    for k in np.unique(lc_index):
        m = lc_index == k
        ave_o, ave_m = obs_br[m].mean(), model_br[m].mean()
        res[m] = (obs_br[m] - model_br[m] * ave_o / ave_m) / ave_o
    return float(np.sqrt(np.mean(res ** 2))), res


def read_lcs(path):
    v = open(path).read().split()
    nlc, pos = int(v[0]), 1
    rows, lc = [], []
    for k in range(nlc):
        n, _flag = int(v[pos]), int(v[pos + 1])
        pos += 2
        a = np.array(v[pos:pos + 8 * n], float).reshape(n, 8)
        pos += 8 * n
        rows.append(a)
        lc.append(np.full(n, k))
    a = np.vstack(rows)
    return {"jd": a[:, 0], "br": a[:, 1], "sun": a[:, 2:5], "earth": a[:, 5:8], "lc": np.concatenate(lc)}


# ---------------------------------------------------------------- Gaussian sphere + ellipsoid (same grid as convexinv)
def gaussian_sphere(nrows=8):
    """Facet normals and unit-sphere areas of convexinv's triangulation (trifac/areanorm)."""
    dth = np.pi / (2 * nrows)
    t, f = [0.0], [0.0]
    for i in range(1, nrows + 1):
        dph = np.pi / (2 * i)
        for j in range(4 * i):
            t.append(i * dth)
            f.append(j * dph)
    for i in range(nrows - 1, 0, -1):
        dph = np.pi / (2 * i)
        for j in range(4 * i):
            t.append(np.pi - i * dth)
            f.append(j * dph)
    t.append(np.pi)
    f.append(0.0)
    t, f = np.array(t), np.array(f)
    xyz = np.stack([np.sin(t) * np.cos(f), np.sin(t) * np.sin(f), np.cos(t)], axis=1)
    # facets: build via convex hull of the vertices (equivalent triangulation of the sphere grid)
    from scipy.spatial import ConvexHull
    hull = ConvexHull(xyz)
    tri = hull.simplices
    v1, v2, v3 = xyz[tri[:, 0]], xyz[tri[:, 1]], xyz[tri[:, 2]]
    c = np.cross(v2 - v1, v3 - v1)
    cen = (v1 + v2 + v3) / 3
    c = np.where((np.sum(c * cen, axis=1) < 0)[:, None], -c, c)
    area = 0.5 * np.linalg.norm(c, axis=1)
    return c / (2 * area)[:, None], area


def ellipsoid_areas(normals, dOmega, a, b, c):
    h2 = (a * normals[:, 0]) ** 2 + (b * normals[:, 1]) ** 2 + (c * normals[:, 2]) ** 2
    return (a * b * c) ** 2 / h2 ** 2 * dOmega
