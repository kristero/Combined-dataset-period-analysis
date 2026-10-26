"""
Shape test of the convex inversion of the MPC data against the published DAMIT models of the validation asteroids
(damit_ref/, fetched with damit_ref_fetch.py). For every validation asteroid with a DAMIT model:
  * our model: the best solution of damit_runs/val_<N>/poles (areas_best), turned into a polyhedron with DAMIT
    minkowski | standardtri (if minkowski fails, facet areas below 1e-8 of the total are raised to that value, as in
    damit_shapes_page.py);
  * the DAMIT models: shape.obj (convex hull of its vertices, so the few non-convex models enter through their hull)
    and the pole and period of the models table.
Metrics, for bodies scaled to unit volume and centred on their centroid:
  * axis ratios a/b and b/c of the dynamically equivalent ellipsoid (uniform density, from the second-moment tensor);
  * rms difference of the radius over a 2-degree grid of directions (area weighted) after the best alignment, in units
    of the radius of the unit-volume sphere. Both bodies have the spin axis along z; the rotation about z is free
    (2-degree steps), and the body may be flipped (spin axis reversed) and mirrored, so that the sense of rotation and
    the mirror ambiguity of the pole do not enter;
  * angle between our pole and the DAMIT pole, also allowing the mirror pole (lambda + 180 deg).
Baselines with the same metrics: two DAMIT models of the same asteroid from different publications, and random pairs
(our model of one asteroid against a DAMIT model of another).
Run on the LU HPC (hpc/damit_shapes.sbatch): python3 damit_shape_compare.py [workers]; writes damit_shape_compare.json
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.spatial import ConvexHull

ROOT = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(ROOT, "damit_ref")
RUNS = os.path.join(ROOT, "damit_runs")
BIN = os.environ.get("DAMIT_BIN", os.path.join(ROOT, "damit", "build_linux"))
OURS = os.path.join(REF, "ours")
NT, NP = 90, 180                                   # 2-degree grid in colatitude and longitude
R0 = (3.0 / (4.0 * np.pi)) ** (1.0 / 3.0)          # radius of the unit-volume sphere
TH = (np.arange(NT) + 0.5) * np.pi / NT
PH = np.arange(NP) * 2.0 * np.pi / NP
WT = np.sin(TH)
U = np.stack([np.outer(np.sin(TH), np.cos(PH)), np.outer(np.sin(TH), np.sin(PH)),
              np.outer(np.cos(TH), np.ones(NP))], axis=-1).reshape(-1, 3)


# ------------------------------------------------------------------------------------------------------------------
def minkowski(areas, out):
    tmp = out + ".part"
    cmd = f"ulimit -s unlimited; cat '{areas}' | timeout 900 '{BIN}/minkowski' | '{BIN}/standardtri' > '{tmp}'"
    p = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    if p.returncode != 0 or not os.path.exists(tmp) or os.path.getsize(tmp) == 0:
        raise RuntimeError(f"minkowski failed: {p.stderr[-200:]}")
    os.replace(tmp, out)
    return out


def our_shape(num):
    areas = os.path.join(RUNS, f"val_{num}", "poles", "areas_best")
    out = os.path.join(OURS, f"{num}.shape")
    if not (os.path.exists(out) and os.path.getsize(out) > 0):
        try:
            minkowski(areas, out)
        except RuntimeError:
            ln = open(areas).read().split("\n")
            n = int(ln[0])
            ar = np.array([float(ln[1 + 2 * i]) for i in range(n)])
            floor = 1e-8 * ar.sum()
            reg = out + ".areas_floor"
            rows = [str(n)]
            for i in range(n):
                rows += [f"{max(ar[i], floor):.12e}", ln[2 + 2 * i]]
            open(reg, "w", newline="\n").write("\n".join(rows) + "\n")
            minkowski(reg, out)
    v = open(out).read().split()
    nv = int(v[0])
    return np.array(v[2:2 + 3 * nv], float).reshape(nv, 3)


def damit_shape(mid):
    verts = []
    for line in open(os.path.join(REF, "models", str(mid), "shape.obj")):
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
    return np.array(verts)


def read_par(num):
    t = open(os.path.join(RUNS, f"val_{num}", "poles", "par_best")).read().split()
    return dict(lam=float(t[0]), bet=float(t[1]), P=float(t[2]))


# ------------------------------------------------------------------------------------------------------------------
def body(verts):
    """unit-volume convex body centred on its centroid: radial function on the grid and axis ratios"""
    h = ConvexHull(verts)
    c0 = verts[h.vertices].mean(axis=0)
    tri = verts[h.simplices] - c0
    det = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2]))
    vol6 = np.abs(det)
    V = vol6.sum() / 6.0
    cen = c0 + np.sum(vol6[:, None] * tri.sum(axis=1), axis=0) / (4.0 * vol6.sum())
    s = V ** (-1.0 / 3.0)
    # second moments about the centroid (tetrahedra centroid-facet), unit volume
    t2 = (verts[h.simplices] - cen) * s
    d2 = np.abs(np.einsum("ij,ij->i", t2[:, 0], np.cross(t2[:, 1], t2[:, 2])))
    ssum = t2.sum(axis=1)
    C = (np.einsum("k,kij,kil->jl", d2, t2, t2) + np.einsum("k,kj,kl->jl", d2, ssum, ssum)) / 120.0
    ev = np.sort(np.linalg.eigvalsh(C))[::-1]
    a, b, c = np.sqrt(5.0 * np.maximum(ev, 1e-30))       # unit volume: mass 1
    # radial function: hull planes n.x + d = 0 after the shift and scaling
    n = h.equations[:, :3]
    d = (h.equations[:, 3] + n @ cen) * s
    r = np.empty(len(U))
    for k in range(0, len(U), 3000):
        nu = U[k:k + 3000] @ n.T
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(nu > 1e-12, -d[None, :] / nu, np.inf)
        r[k:k + 3000] = t.min(axis=1)
    return dict(r=r.reshape(NT, NP), ab=float(a / b), bc=float(b / c), axes=[float(a), float(b), float(c)])


def variants(r):
    j = np.arange(NP)
    flip = r[::-1][:, (-j) % NP]                         # 180 deg about x: theta -> pi - theta, phi -> -phi
    return {"same": r, "flip": flip, "mirror": r[:, (NP // 2 - j) % NP], "flip+mirror": flip[:, (NP // 2 - j) % NP]}


def aligned_rms(rA, rB):
    """minimum over rotations about z (and flips/mirroring of A) of the area-weighted rms radius difference"""
    FB = np.fft.rfft(rB, axis=1)
    sB = (rB ** 2).sum(axis=1)
    best = (np.inf, None, None)
    for name, A in variants(rA).items():
        corr = np.fft.irfft(np.fft.rfft(A, axis=1) * np.conj(FB), n=NP, axis=1)       # sum_j A[j+k] B[j]
        S = ((A ** 2).sum(axis=1)[:, None] + sB[:, None] - 2.0 * corr) * WT[:, None]
        rms = np.sqrt(np.maximum(S.sum(axis=0), 0) / (NP * WT.sum())) / R0
        k = int(np.argmin(rms))
        if rms[k] < best[0]:
            best = (float(rms[k]), name, 2.0 * k)
    return best


def pole_vec(lam, bet):
    l, b = np.radians(lam), np.radians(bet)
    return np.array([np.cos(b) * np.cos(l), np.cos(b) * np.sin(l), np.sin(b)])


def pole_dist(l1, b1, l2, b2):
    a = np.degrees(np.arccos(np.clip(pole_vec(l1, b1) @ pole_vec(l2, b2), -1, 1)))
    m = np.degrees(np.arccos(np.clip(pole_vec(l1 + 180.0, b1) @ pole_vec(l2, b2), -1, 1)))
    return float(a), float(min(a, m))


def compare(bA, bB):
    rms, var, psi = aligned_rms(bA["r"], bB["r"])
    return dict(rms=rms, variant=var, psi_deg=psi, ab_ratio=bA["ab"] / bB["ab"], bc_ratio=bA["bc"] / bB["bc"])


def source(m):
    t = " ".join(f"{r.get('title') or ''} {r.get('author_short') or ''}" for r in m["references"]).lower()
    if "gaia" in t:
        return "Gaia DR3"
    if "lowell" in t:
        return "Lowell"
    if "atlas" in t:
        return "ATLAS"
    return "other"


# ------------------------------------------------------------------------------------------------------------------
def one(args):
    num, mods = args
    out = dict(number=num, models=[])
    try:
        par = read_par(num)
        bO = body(our_shape(num))
    except Exception as e:
        out["error"] = f"our model: {e}"
        return out, None, {}
    out.update(pole=[par["lam"], par["bet"]], P=par["P"], ab=bO["ab"], bc=bO["bc"])
    bodies = {}
    for m in mods:
        try:
            bD = body(damit_shape(m["model_id"]))
        except Exception as e:
            out["models"].append(dict(model_id=m["model_id"], error=str(e)))
            continue
        bodies[m["model_id"]] = bD
        c = compare(bO, bD)
        dp, dpm = pole_dist(par["lam"], par["bet"], m["lambda_"], m["beta"])
        out["models"].append(dict(model_id=m["model_id"], source=source(m), nonconvex=m.get("nonconvex"),
                                  refs=[f"{r.get('author_short')} {r.get('year')}" for r in m["references"]],
                                  damit_pole=[m["lambda_"], m["beta"]], damit_P=m["period"], ab=bD["ab"], bc=bD["bc"],
                                  pole_dist=dp, pole_dist_mirror=dpm, dP_rel=(par["P"] - m["period"]) / m["period"], **c))
    # DAMIT against DAMIT: pairs of models of this asteroid without a common reference
    pairs = []
    ids = [m["model_id"] for m in mods if m["model_id"] in bodies]
    refs = {m["model_id"]: {r.get("bibcode") for r in m["references"]} for m in mods}
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if refs[ids[i]] & refs[ids[j]]:
                continue
            c = compare(bodies[ids[i]], bodies[ids[j]])
            pairs.append(dict(models=[ids[i], ids[j]], **c))
    out["damit_pairs"] = pairs
    print(num, len(out["models"]), "models", len(pairs), "pairs", flush=True)
    return out, bO, bodies


def stats(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if len(x) == 0:
        return None
    q = np.percentile(x, [16, 50, 84])
    return dict(n=int(len(x)), median=float(q[1]), p16=float(q[0]), p84=float(q[2]))


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    os.makedirs(OURS, exist_ok=True)
    M = json.load(open(os.path.join(REF, "models.json"), encoding="utf-8"))["models"]
    by = {}
    for m in M.values():
        if m.get("files_ok"):
            by.setdefault(m["number"], []).append(m)
    todo = sorted(by.items())
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(one, todo))
    rows = [r for r, _, _ in res]
    # random pairs: our model of asteroid X against a DAMIT model of asteroid Y != X
    rng = np.random.default_rng(3)
    ours = {r["number"]: bO for r, bO, _ in res if bO is not None}
    damit = [(r["number"], mid, b) for r, _, bs in res for mid, b in bs.items()]
    rand = []
    for _ in range(400):
        x = int(rng.choice(list(ours)))
        y = damit[int(rng.integers(len(damit)))]
        if y[0] == x:
            continue
        rand.append(compare(ours[x], y[2]))

    def per_ast(sel):
        """per-asteroid medians over its DAMIT models (sel filters the models)"""
        out = {k: [] for k in ("rms", "ab_ratio", "bc_ratio", "pole_dist", "pole_dist_mirror", "dP_rel")}
        for r in rows:
            ms = [m for m in r.get("models", []) if "rms" in m and sel(m)]
            if not ms:
                continue
            for k in out:
                out[k].append(float(np.median([m[k] for m in ms])))
        return {k: stats(v) for k, v in out.items()}

    summary = dict(
        n_asteroids=sum(1 for r in rows if any("rms" in m for m in r.get("models", []))),
        n_models=sum(1 for r in rows for m in r.get("models", []) if "rms" in m),
        sources={s: sum(1 for r in rows for m in r.get("models", []) if m.get("source") == s)
                 for s in ("Gaia DR3", "Lowell", "ATLAS", "other")},
        all=per_ast(lambda m: True), gaia=per_ast(lambda m: m["source"] == "Gaia DR3"),
        not_gaia=per_ast(lambda m: m["source"] != "Gaia DR3"),
        damit_pairs=dict(rms=stats([p["rms"] for r in rows for p in r.get("damit_pairs", [])]),
                         ab_ratio=stats([p["ab_ratio"] for r in rows for p in r.get("damit_pairs", [])]),
                         n_asteroids=sum(1 for r in rows if r.get("damit_pairs"))),
        random_pairs=dict(rms=stats([p["rms"] for p in rand]), ab_ratio=stats([p["ab_ratio"] for p in rand])),
        ab_ours_vs_damit_corr=float(np.corrcoef(
            [r["ab"] for r in rows if any("rms" in m for m in r.get("models", []))],
            [np.median([m["ab"] for m in r["models"] if "rms" in m]) for r in rows
             if any("rms" in m for m in r.get("models", []))])[0, 1]),
    )
    # fraction of asteroids whose shape difference is below the median DAMIT-DAMIT difference
    dd = summary["damit_pairs"]["rms"]["median"] if summary["damit_pairs"]["rms"] else None
    if dd:
        per = [np.median([m["rms"] for m in r["models"] if "rms" in m]) for r in rows
               if any("rms" in m for m in r.get("models", []))]
        summary["frac_below_damit_pair_median"] = float(np.mean(np.array(per) <= dd))
    json.dump(dict(grid_deg=2, summary=summary, asteroids=rows, random_pairs=rand),
              open(os.path.join(ROOT, "damit_shape_compare.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
