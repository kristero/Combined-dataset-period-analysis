#!/usr/bin/env python
"""(1717) Arlon -- a-posteriori diagnostic of the ground-based mutual-event search, per apparition.

binary_1717.py stacks the folded night means of the double-prewhitened ground data over apparitions. Panel (j2) of
review/binary_1717.png suggests that a few faint 2017 nights line up near half the literature orbital period. This
script repeats the folded-binned dip statistic for each event-geometry apparition separately (2017 = apparition 8,
2018-19 = apparition 9), with 2000 night permutations within dataset, and lists the faintest nights with their
separations in units of P_orb/2. Because the apparition was chosen after looking at the figure, the p-values are
a-posteriori (look-elsewhere: 2 apparitions x 2 subsets) and are reported only as a diagnostic.

Inputs: results/1717/1717_arrays.npz, data/horizons/1717_geocentric_daily.csv (via binary_1717.load)
Output: binary_1717_app8.json
"""
import os
import json
import numpy as np
import binary_1717 as B

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "binary_1717_app8.json")
N_PERM = 2000


def main():
    rng = np.random.default_rng(8)
    D = B.load()
    t, y, sheet, app, blk, grp = D["t"], D["y"], D["sheet"], D["app"], D["blk"], D["grp"]
    tess = sheet == "C57G"
    ground = ~tess
    _, ev_vecs = B.event_season(D["hor"])
    evdist = B.event_distance(D["pab"], D["pablat"], ev_vecs)
    # frequencies as in binary_1717.json (ground-only joint fit, order 2)
    J = json.load(open(os.path.join(ROOT, "binary_1717.json")))
    f2g, f1g = J["frequencies"]["ground"]["f2_cpd"], J["frequencies"]["ground"]["f1_cpd"]
    m = ground
    tt, yy = t[m], y[m]
    napp = {a: np.sum(app[m] == a) for a in np.unique(app[m])}
    lab = np.array([a if napp[a] >= 60 else -1 for a in app[m]])
    A = B.design(tt, [(f2g, 2, lab), (f1g, 2, lab)], offset_lab=grp[m])
    rfull = np.zeros(len(t))
    rfull[m] = yy - B.lsfit(A, yy)[0]
    out = {"note": __doc__.split("Inputs")[0].strip(), "n_perm": N_PERM, "results": {}}
    Porb = B.PORB
    for apn in (8, 9):
        for lim in (5.0, 10.0):
            sel = ground & (app == apn) & (evdist <= lim)
            nt = B.night_table(t, rfull, blk, sel)
            bl = np.unique(blk[sel])
            grp_n = np.array([grp[blk == b][0] for b in bl])
            app_n = np.full(len(bl), apn)
            span = nt["t"].max() - nt["t"].min()
            Pgrid = B.period_grid_dip(50.0, 150.0, span, 0.05)
            ds = B.DipStack(nt, app_n, Pgrid, wbin=0.05, min_nights=5)
            S, _ = ds.stack(nt["m"], nt["w"])
            wt = ((Pgrid >= Porb - 1.5) & (Pgrid <= Porb + 1.5)) | ((Pgrid >= (Porb - 1.5) / 2) & (Pgrid <= (Porb + 1.5) / 2))
            obs_t, obs_g = float(S[wt].max()), float(S.max())
            nul_t, nul_g = [], []
            for i in range(N_PERM):
                mp, wp = B.permute_nights(nt["m"], nt["w"], grp_n, rng)
                Sp, _ = ds.stack(mp, wp)
                nul_t.append(Sp[wt].max())
                nul_g.append(Sp.max())
            nul_t, nul_g = np.array(nul_t), np.array(nul_g)
            z = nt["m"] * np.sqrt(nt["w"])
            o = np.argsort(-z)[:5]
            from astropy.time import Time
            faint = []
            for i in o:
                faint.append({"date_utc": Time(nt["t"][i] + D["tz"], format="jd").iso[:16],
                              "dataset": str(sheet[blk == bl[i]][0]), "n": int(nt["n"][i]),
                              "mean_resid_mag": float(nt["m"][i]), "z": float(z[i])})
            # separations of the faintest nights in units of Porb/2 (117.0 h literature value)
            tf = nt["t"][o[:3]] * 24.0
            seps = []
            for a_ in range(3):
                for b_ in range(a_ + 1, 3):
                    dt = abs(tf[b_] - tf[a_])
                    nh = dt / (Porb / 2)
                    seps.append({"pair": [faint[a_]["date_utc"], faint[b_]["date_utc"]], "dt_h": float(dt),
                                 "dt_over_halfPorb": float(nh),
                                 "Porb_if_integer_half_cycles_h": float(2 * dt / round(nh)) if round(nh) > 0 else None,
                                 "offset_h_from_integer_half_cycles_at_117h": float(dt - round(nh) * Porb / 2)})
            out["results"][f"app{apn}_{int(lim)}deg"] = {
                "n_nights": int(len(bl)), "target_max": obs_t, "target_P_h": float(Pgrid[wt][np.argmax(S[wt])]),
                "global_max_50_150h": obs_g, "global_P_h": float(Pgrid[np.argmax(S)]),
                "p_target_aposteriori": float((1 + np.sum(nul_t >= obs_t)) / (1 + N_PERM)),
                "p_global_aposteriori": float((1 + np.sum(nul_g >= obs_g)) / (1 + N_PERM)),
                "null_target_q50_q95_q99": np.percentile(nul_t, [50, 95, 99]).tolist(),
                "faintest_nights": faint, "separations_faintest3": seps}
            print(apn, lim, out["results"][f"app{apn}_{int(lim)}deg"]["target_P_h"], obs_t,
                  out["results"][f"app{apn}_{int(lim)}deg"]["p_target_aposteriori"],
                  out["results"][f"app{apn}_{int(lim)}deg"]["p_global_aposteriori"])
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(B._jsonable(out), fh, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
