"""
Literature-based amplitude estimators for sparse multi-survey photometry with a known rotation period.
Input: a dict from amp_data.load (t [d], y [reduced mag], sheet, alpha, P [h], ...).

Apparitions: the data are split where the time gap exceeds GAP days; an apparition is used if it has at least
MIN_N measurements.

COMB_F4   (baseline) peak-to-peak of an order-4 Fourier series fitted to all data folded at P (current practice).

APP_FS    Per-apparition Fourier fit at the fixed period (Waszczak et al. 2015; Chang et al. 2015, with one zero
          point per dataset, their Z_i; order chosen by BIC among 2..4, after Vavilov & Carry 2025 who select the
          lowest adequate order). Amplitude = peak-to-peak of the Fourier part.

APP_VAR   Per-apparition dispersion (Cibulkova et al. 2016; Mommert et al. 2018): variance of the magnitudes about
          their per-dataset means in the apparition, minus the white-noise variance and the night-to-night
          zero-point variance; A = kappa * sqrt(var_rot) with kappa = 2*sqrt(2) for a sinusoid (APP_VAR) or kappa =
          peak-to-peak/rms of the order-4 shape of the combined data (APP_VARs).

APP_SF    Per-apparition structure function of magnitude changes between consecutive measurements of the same
          dataset in the same night (McNeill et al. 2016; Szabo & Kiss 2008 use two-epoch magnitude differences):
          E[dm^2] = 2 sigma^2 + a^2 (1 - cos(4 pi dt / P)) for a double-peaked sinusoidal light curve of
          semi-amplitude a; fitted by least squares over the pairs; A = 2 a.

Noise: white noise sigma^2 from the same-night pairs (intercept of the structure function, per dataset);
night-to-night zero-point variance from the night means of the residuals of the APP_FS fit.
Each per-apparition estimator is summarised over the apparitions by its median and its maximum, and the mean
phase angle of every apparition is kept, for the amplitude-phase relation A(0) = A(alpha) / (1 + m alpha)
(Zappala et al. 1990).
"""
from __future__ import annotations

import numpy as np

GAP = 60.0          # days
MIN_N = 60
PAIR_DT = 0.15      # days: same-night pairs
MIN_PAIRS = 15


def fourier_design(t, P, N):
    w = 2 * np.pi * 24.0 / P
    return np.vstack([f(k * w * t) for k in range(1, N + 1) for f in (np.sin, np.cos)]).T


def ptp_coef(c, N):
    ph = np.linspace(0, 1, 1000, endpoint=False)
    X = np.vstack([f(2 * np.pi * k * ph) for k in range(1, N + 1) for f in (np.sin, np.cos)]).T
    m = X @ c
    return float(m.max() - m.min()), m


def fit_clipped(X, y, clip=3.0, iters=6):
    keep = np.ones(len(y), bool)
    for _ in range(iters):
        c, *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
        r = y - X @ c
        s = np.std(r[keep], ddof=X.shape[1])
        new = np.abs(r) <= clip * s
        if np.array_equal(new, keep):
            break
        keep = new
    return c, keep, r, s


def apparitions(t):
    o = np.argsort(t)
    cuts = np.where(np.diff(t[o]) > GAP)[0]
    return [g for g in np.split(o, cuts + 1) if len(g) >= MIN_N]


def comb_f4(d):
    X = np.hstack([np.ones((len(d["t"]), 1)), fourier_design(d["t"], d["P"], 4)])
    c, keep, _, _ = fit_clipped(X, d["y"])
    a, shape = ptp_coef(c[1:], 4)
    return a, shape


def pair_noise(t, y, sheet, P):
    """Same-night consecutive pairs per dataset -> per-dataset white-noise variance (structure-function intercept)
    and the global pairs for the rotational term."""
    out, pairs = {}, []
    for ds in np.unique(sheet):
        m = np.where(sheet == ds)[0]
        o = m[np.argsort(t[m])]
        dt = np.diff(t[o])
        dm = np.diff(y[o])
        ok = (dt > 0) & (dt < min(PAIR_DT, P / 24.0 / 4))
        if ok.sum() >= MIN_PAIRS:
            x = 1 - np.cos(4 * np.pi * dt[ok] * 24.0 / P)
            # robust: drop the most discrepant 5 % of pairs (outliers, cosmic rays)
            lim = np.percentile(np.abs(dm[ok]), 95)
            k = np.abs(dm[ok]) <= lim
            A_ = np.vstack([np.full(k.sum(), 2.0), x[k]]).T
            c, *_ = np.linalg.lstsq(A_, dm[ok][k] ** 2, rcond=None)
            out[ds] = max(float(c[0]), 1e-6)
            pairs.append((dt[ok], dm[ok], np.full(ok.sum(), ds)))
    return out, pairs


def per_apparition(d, shape_kappa):
    t, y, sheet, P = d["t"], d["y"], d["sheet"], d["P"]
    noise_all, _ = pair_noise(t, y, sheet, P)
    rows = []
    for g in apparitions(t):
        tg, yg, sg = t[g], y[g], sheet[g]
        ph = (tg * 24.0 / P) % 1.0
        if len(np.unique(np.minimum((ph * 10).astype(int), 9))) < 8:
            continue
        sets = list(np.unique(sg))
        Z = np.vstack([(sg == s).astype(float) for s in sets]).T
        # --- APP_FS: Fourier with per-dataset zero points, order by BIC
        best = None
        for N in (2, 3, 4):
            X = np.hstack([Z, fourier_design(tg, P, N)])
            c, keep, r, s = fit_clipped(X, yg)
            n = keep.sum()
            bic = n * np.log(np.mean(r[keep] ** 2)) + X.shape[1] * np.log(n)
            if best is None or bic < best[0]:
                best = (bic, N, c, keep, r)
        _, N, c, keep, r = best
        a_fs, _ = ptp_coef(c[len(sets):], N)
        # --- noise: white (pairs) and night zero points (night means of the APP_FS residuals)
        sig2 = np.array([noise_all.get(s, np.nan) for s in sg])
        if np.all(~np.isfinite(sig2)):
            sig2 = np.full(len(sg), np.var(r[keep]))
        sig2[~np.isfinite(sig2)] = np.nanmedian(sig2)
        night = np.array([f"{s}|{int(np.floor(x))}" for s, x in zip(sg, tg)])
        nm, nn, nw = [], [], []
        for k in np.unique(night[keep]):
            mm = keep & (night == k)
            if mm.sum() >= 2:
                nm.append(np.mean(r[mm]))
                nn.append(mm.sum())
                nw.append(np.mean(sig2[mm]) / mm.sum())
        var_night = max(float(np.var(nm) - np.mean(nw)), 0.0) if len(nm) >= 5 else 0.0
        # --- APP_VAR: dispersion about per-dataset means
        yc = yg.copy()
        for s in sets:
            mm = sg == s
            yc[mm] -= np.median(yg[mm])
        # 3-sigma clipping of the dispersion sample
        kk = np.abs(yc - np.median(yc)) <= 3 * 1.4826 * np.median(np.abs(yc - np.median(yc))) * 1.5
        var_tot = float(np.var(yc[kk], ddof=len(sets)))
        var_rot = max(var_tot - float(np.mean(sig2[kk])) - var_night, 0.0)
        # --- APP_SF: structure function of same-night pairs within the apparition
        o = np.argsort(tg)
        xs, ys2 = [], []
        for s in sets:
            m = o[sg[o] == s]
            dt, dm = np.diff(tg[m]), np.diff(yg[m])
            ok = (dt > 0) & (dt < min(PAIR_DT, P / 24.0 / 4))
            xs.append(1 - np.cos(4 * np.pi * dt[ok] * 24.0 / P))
            ys2.append(dm[ok])
        xs, dms = np.concatenate(xs), np.concatenate(ys2)
        a_sf = np.nan
        if len(xs) >= MIN_PAIRS and np.ptp(xs) > 0.05:
            lim = np.percentile(np.abs(dms), 95)
            k = np.abs(dms) <= lim
            Asf = np.vstack([np.full(k.sum(), 2.0), xs[k]]).T
            csf, *_ = np.linalg.lstsq(Asf, dms[k] ** 2, rcond=None)
            a_sf = 2 * np.sqrt(max(float(csf[1]), 0.0))
        rows.append({"t_mid": float(np.median(tg)), "n": int(len(g)), "n_sets": len(sets),
                     "alpha_mean": float(np.nanmean(d["alpha"][g])), "N_fs": int(N), "APP_FS": a_fs,
                     "APP_VAR": float(2 * np.sqrt(2) * np.sqrt(var_rot)),
                     "APP_VARs": float(shape_kappa * np.sqrt(var_rot)), "APP_SF": float(a_sf),
                     "var_tot": var_tot, "var_noise": float(np.mean(sig2[kk])), "var_night": var_night,
                     "n_pairs": int(len(xs))})
    return rows


def template_fit(d, n_iter=3, n_grid=200, K=4):
    """APP_TPL: per-apparition scale g_j and phase shift dphi_j of a common light-curve template S (order-K Fourier,
    unit peak-to-peak), with one zero point per dataset and apparition; S is re-estimated from the phase-aligned
    data (Waszczak et al. 2015, F(tau); template formulation of the literature synthesis). Returns per-window
    amplitudes g_j with their least-squares errors, the shifts, and the coherence factors
    C_obs = A_fold / <g>_w and C_pred = |sum w g exp(i 4 pi dphi)| / sum w g."""
    t, y, sheet, P = d["t"], d["y"], d["sheet"], d["P"]
    ph0 = (t * 24.0 / P) % 1.0
    wins = []
    for g in apparitions(t):
        if len(np.unique(np.minimum((ph0[g] * 10).astype(int), 9))) >= 8:
            wins.append(g)
    if not wins:
        return None
    X = np.hstack([np.ones((len(t), 1)), fourier_design(t, P, K)])
    c, _, _, _ = fit_clipped(X, y)
    a0, _ = ptp_coef(c[1:], K)
    coef = c[1:] / a0
    grid = np.arange(n_grid) / n_grid
    shifts = np.zeros(len(wins))
    for it in range(n_iter + 1):
        rows = []
        for j, g in enumerate(wins):
            sg = d["sheet"][g]
            sets = list(np.unique(sg))
            Z = np.vstack([(sg == s).astype(float) for s in sets]).T
            # all shifts at once: S(phi - d) = B(phi) @ R(d) coef, with B = [sin 2pi k phi, cos 2pi k phi];
            # zero points are removed by subtracting the per-dataset means (projection on Z)
            B = np.vstack([f(2 * np.pi * k * ph0[g]) for k in range(1, K + 1) for f in (np.sin, np.cos)]).T
            Cm = np.zeros((2 * K, n_grid))
            for k in range(1, K + 1):
                a_k, b_k = coef[2 * (k - 1)], coef[2 * (k - 1) + 1]
                cd, sd = np.cos(2 * np.pi * k * grid), np.sin(2 * np.pi * k * grid)
                Cm[2 * (k - 1)] = a_k * cd + b_k * sd          # coefficient of sin(2 pi k phi)
                Cm[2 * (k - 1) + 1] = b_k * cd - a_k * sd      # coefficient of cos(2 pi k phi)
            Fall = B @ Cm
            yr = y[g].copy()
            Fr = Fall.copy()
            for s_ in sets:
                m_ = sg == s_
                yr[m_] -= yr[m_].mean()
                Fr[m_] -= Fr[m_].mean(axis=0)
            num_ = Fr.T @ yr
            den_ = np.einsum("ij,ij->j", Fr, Fr)
            gg = num_ / den_
            ssv = yr @ yr - gg * num_
            ssv[gg <= 0] = np.inf
            jbest = int(np.argmin(ssv))
            dphi = float(grid[jbest])
            Xg = np.hstack([Z, Fall[:, [jbest]]])
            cg, *_ = np.linalg.lstsq(Xg, y[g], rcond=None)
            r = y[g] - Xg @ cg
            ss = float(r @ r)
            # one clipping pass at 3 sigma, refit at the chosen shift
            s = np.sqrt(ss / max(len(g) - Xg.shape[1], 1))
            k = np.abs(r) <= 3 * s
            cg, *_ = np.linalg.lstsq(Xg[k], y[g][k], rcond=None)
            r = y[g][k] - Xg[k] @ cg
            s2 = float(r @ r) / max(k.sum() - Xg.shape[1], 1)
            cov = s2 * np.linalg.pinv(Xg[k].T @ Xg[k])
            rows.append({"n": int(len(g)), "g": float(cg[-1]), "sigma_g": float(np.sqrt(cov[-1, -1])),
                         "dphi": float(dphi), "alpha_mean": float(np.nanmean(d["alpha"][g])), "zp": cg[:-1],
                         "zp_var": np.diag(cov)[:-1], "sets": sets})
            shifts[j] = dphi
        if it == n_iter:
            break
        # re-estimate the template from the aligned data (offsets removed, scaled to the window amplitude)
        ta, ya = [], []
        for j, g in enumerate(wins):
            sg = d["sheet"][g]
            zp = dict(zip(rows[j]["sets"], rows[j]["zp"]))
            ta.append((ph0[g] - shifts[j]) % 1.0)
            ya.append((y[g] - np.array([zp[s] for s in sg])) / rows[j]["g"])
        ta, ya = np.concatenate(ta), np.concatenate(ya)
        Xa = np.hstack([np.ones((len(ta), 1)), np.vstack([f(2 * np.pi * k * ta) for k in range(1, K + 1)
                                                           for f in (np.sin, np.cos)]).T])
        ca, _, _, _ = fit_clipped(Xa, ya)
        a1, _ = ptp_coef(ca[1:], K)
        coef = ca[1:] / a1
    gs = np.array([r["g"] for r in rows])
    w = np.array([r["n"] for r in rows], float)
    snr = gs / np.array([r["sigma_g"] for r in rows])
    c_pred = float(abs(np.sum(w * gs * np.exp(1j * 4 * np.pi * shifts))) / np.sum(w * gs))
    lam, bet = np.radians(d.get("pab_lon", np.zeros(len(t)))), np.radians(d.get("pab_lat", np.zeros(len(t))))
    for j, (r, g) in enumerate(zip(rows, wins)):
        # mean brightness level of the window: zero points weighted by the number of points per dataset (the
        # template has zero mean, so the zero point is the rotation-averaged level on the common magnitude scale)
        sg = d["sheet"][g]
        nd = np.array([np.sum(sg == s) for s in r["sets"]], float)
        r["mu"] = float(np.sum(nd * r["zp"]) / np.sum(nd))
        r["sigma_mu"] = float(np.sqrt(np.sum(nd ** 2 * np.asarray(r.get("zp_var", np.zeros(len(nd))))) / np.sum(nd) ** 2))
        v = np.vstack([np.cos(bet[g]) * np.cos(lam[g]), np.cos(bet[g]) * np.sin(lam[g]), np.sin(bet[g])]).T
        vm = np.nanmean(v, axis=0)
        r["pab_vec"] = (vm / np.linalg.norm(vm)).tolist()
        r["t_mid"] = float(np.median(t[g]))
        r["idx"] = g
        r.pop("zp")
        r.pop("sets")
        r.pop("zp_var", None)
    return {"rows": rows, "med": float(np.median(gs)), "wmean": float(np.sum(w * gs) / np.sum(w)),
            "max10": float(np.max(gs[snr >= 10])) if np.any(snr >= 10) else np.nan, "C_pred": c_pred}


def estimate(d):
    a_comb, shape = comb_f4(d)
    kappa = float(np.ptp(shape) / np.std(shape))
    rows = per_apparition(d, kappa)
    res = {"COMB_F4": a_comb, "kappa": kappa, "n_app": len(rows), "apparitions": rows}
    tp = template_fit(d)
    if tp:
        res.update({"APP_TPL_med": tp["med"], "APP_TPL_wmean": tp["wmean"], "APP_TPL_max10": tp["max10"],
                    "C_pred": tp["C_pred"], "C_obs": a_comb / tp["wmean"],
                    "template_windows": [{k: v for k, v in r.items() if k != "idx"} for r in tp["rows"]]})
    else:
        res.update({"APP_TPL_med": np.nan, "APP_TPL_wmean": np.nan, "APP_TPL_max10": np.nan})
    for k in ("APP_FS", "APP_VAR", "APP_VARs", "APP_SF"):
        v = np.array([r[k] for r in rows if np.isfinite(r[k])])
        res[k + "_med"] = float(np.median(v)) if len(v) else np.nan
        res[k + "_max"] = float(np.max(v)) if len(v) else np.nan
        # amplitude-phase relation: reduce each apparition to the mean phase angle of 15 deg with m = 0.02/deg
        al = np.array([r["alpha_mean"] for r in rows if np.isfinite(r[k])])
        if len(v):
            v15 = v * (1 + 0.02 * 15.0) / (1 + 0.02 * al)
            res[k + "_med_apr"] = float(np.median(v15))
    return res
