"""
Second step of the DAMIT shape test (after damit_shape_compare.py, same metrics and cached polyhedra):
  * poles: angle to the closest DAMIT solution of the asteroid (the DAMIT models of every validation asteroid are the
    two mirror solutions of one publication), with and without the mirror ambiguity;
  * rank test: the shape difference (aligned rms of the radius) and the elongation difference |ln (a/b)| between our
    model of an asteroid and its own DAMIT model(s), ranked among the differences to the DAMIT models of all other
    asteroids (percentile 0 = own model closest);
  * triaxial ellipsoid of the same data (damit_runs/val_<N>/ellipsoid.json): its a/b and b/c against DAMIT;
  * the three models whose polyhedron failed in damit_shape_compare.py are retried with facet areas floored at 1e-6.
Run on the LU HPC (hpc/damit_shapes2.sbatch); writes damit_shape_rank.json
"""
from __future__ import annotations

import json
import os

import numpy as np

import damit_shape_compare as D

ROOT = D.ROOT


def our_shape_floor(num, frac):
    areas = os.path.join(D.RUNS, f"val_{num}", "poles", "areas_best")
    out = os.path.join(D.OURS, f"{num}.shape")
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return D.our_shape(num)
    ln = open(areas).read().split("\n")
    n = int(ln[0])
    ar = np.array([float(ln[1 + 2 * i]) for i in range(n)])
    floor = frac * ar.sum()
    reg = out + f".areas_floor{frac:g}"
    rows = [str(n)]
    for i in range(n):
        rows += [f"{max(ar[i], floor):.12e}", ln[2 + 2 * i]]
    open(reg, "w", newline="\n").write("\n".join(rows) + "\n")
    D.minkowski(reg, out)
    return D.our_shape(num)


def pct(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if not len(x):
        return None
    q = np.percentile(x, [16, 50, 84])
    return dict(n=int(len(x)), median=float(q[1]), p16=float(q[0]), p84=float(q[2]))


def main():
    J = json.load(open(os.path.join(ROOT, "damit_shape_compare.json"), encoding="utf-8"))
    M = json.load(open(os.path.join(D.REF, "models.json"), encoding="utf-8"))["models"]
    by = {}
    for m in M.values():
        if m.get("files_ok"):
            by.setdefault(m["number"], []).append(m)
    ours, retried = {}, {}
    for r in J["asteroids"]:
        num = r["number"]
        try:
            if r.get("error"):
                ours[num] = D.body(our_shape_floor(num, 1e-6))
                retried[num] = "ok"
            else:
                ours[num] = D.body(D.our_shape(num))
        except Exception as e:
            retried[num] = f"failed: {e}"[:200]
    damit = {}
    for num, ms in by.items():
        for m in ms:
            damit[m["model_id"]] = (num, m, D.body(D.damit_shape(m["model_id"])))
    nums = sorted(ours)
    mids = sorted(damit)
    rms = np.full((len(nums), len(mids)), np.nan)
    lab = np.full((len(nums), len(mids)), np.nan)
    for i, n in enumerate(nums):
        for j, mid in enumerate(mids):
            rms[i, j] = D.aligned_rms(ours[n]["r"], damit[mid][2]["r"])[0]
            lab[i, j] = abs(np.log(ours[n]["ab"] / damit[mid][2]["ab"]))
    rows, rank_rms, rank_ab = [], [], []
    for i, n in enumerate(nums):
        own = [j for j, mid in enumerate(mids) if damit[mid][0] == n]
        oth = [j for j, mid in enumerate(mids) if damit[mid][0] != n]
        if not own:
            continue
        o_rms, o_ab = float(np.median(rms[i, own])), float(np.median(lab[i, own]))
        rr = float(np.mean(rms[i, oth] < o_rms))
        ra = float(np.mean(lab[i, oth] < o_ab))
        rank_rms.append(rr)
        rank_ab.append(ra)
        par = D.read_par(n)
        dists = [D.pole_dist(par["lam"], par["bet"], damit[mids[j]][1]["lambda_"], damit[mids[j]][1]["beta"]) for j in own]
        ell = json.load(open(os.path.join(D.RUNS, f"val_{n}", "ellipsoid.json")))["best"]
        dab = [damit[mids[j]][2]["ab"] for j in own]
        dbc = [damit[mids[j]][2]["bc"] for j in own]
        src = D.source(damit[mids[own[0]]][1])
        rows.append(dict(number=n, source=src, rms_own=o_rms, rank_rms=rr, rank_ab=ra,
                         pole_min=float(min(d[0] for d in dists)), pole_min_mirror=float(min(d[1] for d in dists)),
                         ab_ours=ours[n]["ab"], bc_ours=ours[n]["bc"], ab_damit=float(np.median(dab)),
                         bc_damit=float(np.median(dbc)), ab_ell=float(np.exp(ell["ln_ab"])),
                         bc_ell=float(np.exp(ell["ln_bc"]))))

    def summ(sel):
        R = [r for r in rows if sel(r)]
        return dict(
            n=len(R),
            pole_min=pct([r["pole_min"] for r in R]), pole_min_mirror=pct([r["pole_min_mirror"] for r in R]),
            pole_within30=float(np.mean([r["pole_min"] <= 30 for r in R])) if R else None,
            rms_own=pct([r["rms_own"] for r in R]),
            rank_rms=pct([r["rank_rms"] for r in R]), rank_ab=pct([r["rank_ab"] for r in R]),
            frac_rank_rms_lt_0p25=float(np.mean([r["rank_rms"] < 0.25 for r in R])) if R else None,
            ab_ratio_convex=pct([r["ab_ours"] / r["ab_damit"] for r in R]),
            bc_ratio_convex=pct([r["bc_ours"] / r["bc_damit"] for r in R]),
            ab_ratio_ellipsoid=pct([r["ab_ell"] / r["ab_damit"] for r in R]),
            bc_ratio_ellipsoid=pct([r["bc_ell"] / r["bc_damit"] for r in R]),
            ab_scatter_dex_convex=float(np.std([np.log10(r["ab_ours"] / r["ab_damit"]) for r in R])) if R else None,
            ab_scatter_dex_ellipsoid=float(np.std([np.log10(r["ab_ell"] / r["ab_damit"]) for r in R])) if R else None,
            corr_ab_convex=float(np.corrcoef([r["ab_ours"] for r in R], [r["ab_damit"] for r in R])[0, 1]) if len(R) > 2 else None,
            corr_ab_ellipsoid=float(np.corrcoef([r["ab_ell"] for r in R], [r["ab_damit"] for r in R])[0, 1]) if len(R) > 2 else None,
        )

    # random-pair baseline from the full matrix: our model of one asteroid against DAMIT models of the others
    oth_rms = [rms[i, j] for i, n in enumerate(nums) for j, mid in enumerate(mids) if damit[mid][0] != n]
    oth_ab = [np.exp(lab[i, j]) for i, n in enumerate(nums) for j, mid in enumerate(mids) if damit[mid][0] != n]
    S = dict(all=summ(lambda r: True), gaia=summ(lambda r: r["source"] == "Gaia DR3"),
             not_gaia=summ(lambda r: r["source"] != "Gaia DR3"),
             sources={s: sum(1 for r in rows if r["source"] == s) for s in ("Gaia DR3", "Lowell", "ATLAS", "other")},
             other_asteroids=dict(rms=pct(oth_rms), ab_abs_ratio=pct(oth_ab)), retried=retried)
    json.dump(dict(summary=S, asteroids=rows), open(os.path.join(ROOT, "damit_shape_rank.json"), "w",
                                                    encoding="utf-8"), indent=1)
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
