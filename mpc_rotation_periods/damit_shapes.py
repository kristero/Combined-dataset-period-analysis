"""
Shape plots of the DAMIT convex models: the Gaussian image of the best solution (poles/areas_best) is turned into a
convex polyhedron with DAMIT's minkowski + standardtri (Fortran, WSL), and drawn from two equatorial directions
(along the body x and y axes) and pole-on (along +z, the spin axis), shaded with a fixed light. The point-level
ellipsoid of the same data (damit_ellipsoid.py) is drawn below it in the same views.
Usage: python damit_shapes.py RUN [RUN ...] [--out review/damit_shapes.png]
       (RUN = damit_runs sub-directory name, e.g. val_2984)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import damit_wsl as W  # noqa: E402

RUNS = os.path.join(ROOT, "damit_runs")
plt.rcParams.update({"font.size": 11})


def polyhedron(run):
    """Vertices and triangular facets of the convex model of run (cached as poles/shape_best.txt)."""
    pdir = os.path.join(run, "poles")
    out = os.path.join(pdir, "shape_best.txt")
    if not os.path.exists(out):
        areas = "areas_best" if os.path.exists(os.path.join(pdir, "areas_best")) else None
        if areas is None:
            pj = json.load(open(os.path.join(pdir, "poles.json")))
            import damit_eval as V
            areas = f"areas_{V.sol_index(pj.get('best') or min(pj['solutions'], key=lambda r: r['dev']))}"
        cmd = (f"ulimit -s unlimited; cd '{W.wsl_path(pdir)}' && cat {areas} | timeout 1800 {W.BIN}/minkowski "
               f"| {W.BIN}/standardtri > shape_best.txt")
        p = subprocess.run(["wsl", "-d", W.DISTRO, "-e", "bash", "-c", cmd], capture_output=True, text=True)
        if p.returncode != 0 or os.path.getsize(out) == 0:
            if os.path.exists(out):
                os.remove(out)
            raise RuntimeError(f"minkowski failed for {run}: {p.stderr[-300:]}")
    v = open(out).read().split()
    nv, nf = int(v[0]), int(v[1])
    verts = np.array(v[2:2 + 3 * nv], float).reshape(nv, 3)
    faces = np.array(v[2 + 3 * nv:2 + 3 * nv + 3 * nf], int).reshape(nf, 3) - 1
    return verts, faces


def ellipsoid_mesh(a, b, c, n=40):
    u, w = np.meshgrid(np.linspace(0, 2 * np.pi, 2 * n), np.linspace(0, np.pi, n))
    x, y, z = a * np.cos(u) * np.sin(w), b * np.sin(u) * np.sin(w), c * np.cos(w)
    P = np.stack([x, y, z], -1)
    verts = P.reshape(-1, 3)
    idx = np.arange(verts.shape[0]).reshape(n, 2 * n)
    faces = []
    for i in range(n - 1):
        for j in range(2 * n - 1):
            faces += [[idx[i, j], idx[i + 1, j], idx[i + 1, j + 1]], [idx[i, j], idx[i + 1, j + 1], idx[i, j + 1]]]
    return verts, np.array(faces)


def draw(ax, verts, faces, view, color):
    verts = verts - verts.mean(axis=0)
    verts = verts / np.abs(verts).max()
    tri = verts[faces]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    cen = tri.mean(axis=1)
    ln = np.linalg.norm(nrm, axis=1)
    nrm = np.where((ln < 1e-9)[:, None], cen, nrm)              # degenerate triangles: radial direction
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    nrm *= np.sign(np.sum(nrm * cen, axis=1) + 1e-15)[:, None]  # outward (convex bodies centred on 0)
    elev, azim = view
    # light from the viewer's upper left, so every view is lit the same way
    e, a = np.radians(elev), np.radians(azim)
    cam = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    up = np.array([0.0, 0.0, 1.0]) if abs(elev) < 80 else np.array([-np.sin(a), np.cos(a), 0.0]) * -1
    left = np.cross(up, cam)
    light = cam + 0.6 * up + 0.5 * left
    light /= np.linalg.norm(light)
    shade = np.clip(nrm @ light, 0.0, 1.0)
    base = np.array(matplotlib.colors.to_rgb(color))
    cols = np.clip(base[None, :] * (0.30 + 0.85 * shade[:, None]), 0, 1)
    pc = Poly3DCollection(tri, facecolors=cols, edgecolors=cols, linewidths=0.3)
    ax.add_collection3d(pc)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_box_aspect((1, 1, 1), zoom=1.45)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()


def main(argv):
    if "--paper" in argv:
        paper_figure([a for a in argv if not a.startswith("--") and not a.endswith(".pdf")], argv[argv.index("--out") + 1])
        return
    out = argv[argv.index("--out") + 1] if "--out" in argv else os.path.join(ROOT, "review", "damit_shapes.png")
    runs = [a for a in argv if not a.startswith("--") and a != out]
    views = [((0, 0), "equator, along x"), ((0, 90), "equator, along y"), ((90, -90), "pole-on")]
    fig = plt.figure(figsize=(2.6 * len(views), 2.5 * 2 * len(runs)))
    for i, name in enumerate(runs):
        run = os.path.join(RUNS, name) if not os.path.isabs(name) and not os.path.exists(name) else name
        verts, faces = polyhedron(run)
        ell = json.load(open(os.path.join(run, "ellipsoid.json")))["best"]
        bb = np.exp(ell["ln_bc"])
        ev, ef = ellipsoid_mesh(bb * np.exp(ell["ln_ab"]), bb, 1.0)
        pj = json.load(open(os.path.join(run, "poles", "poles.json")))
        bc = pj.get("best") or min(pj["solutions"], key=lambda r: r["dev"])
        info = json.load(open(os.path.join(run, "prep.json")))
        for j, (view, label) in enumerate(views):
            ax = fig.add_subplot(2 * len(runs), len(views), 2 * i * len(views) + j + 1, projection="3d")
            draw(ax, verts, faces, view, "#6b9ac4")
            if j == 0:
                ax.set_title(f"({info['num']}) convex model, P = {bc['P']:.5f} h", loc="left", fontsize=11)
            ax.text2D(0.5, -0.02, label, transform=ax.transAxes, ha="center", fontsize=9, color="0.35")
            ax = fig.add_subplot(2 * len(runs), len(views), (2 * i + 1) * len(views) + j + 1, projection="3d")
            draw(ax, ev, ef, view, "#e0a96d")
            if j == 0:
                ax.set_title(f"ellipsoid, a/b = {np.exp(ell['ln_ab']):.2f}, b/c = {bb:.2f}", loc="left", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(out)


def paper_figure(runs, out):
    """Convex models only: two asteroids per row, three views each (equator along x, along y, pole-on)."""
    names = {"1963": "Bezovec", "2539": "Ningxia", "4270": "Juanvictoria", "4618": "Shakhovskoj", "1942": "Jablunka",
             "1820": "Lohmann", "2817": "Perec", "4761": "Urrutia", "3716": "Petzval", "2134": "Dennispalm"}
    views = [((0, 0), "equator, x"), ((0, 90), "equator, y"), ((90, -90), "pole-on")]
    nrow = (len(runs) + 1) // 2
    fig = plt.figure(figsize=(7.1, 1.75 * nrow + 0.3))
    for i, name in enumerate(runs):
        run = os.path.join(RUNS, name)
        verts, faces = polyhedron(run)
        pj = json.load(open(os.path.join(run, "poles", "poles.json")))
        bc = pj.get("best") or min(pj["solutions"], key=lambda r: r["dev"])
        num = name.split("_")[1]
        r, c0 = i // 2, 3 * (i % 2)
        for j, (view, lab) in enumerate(views):
            ax = fig.add_subplot(nrow, 6, 6 * r + c0 + j + 1, projection="3d")
            draw(ax, verts, faces, view, "#6b9ac4")
            if j == 0:
                ax.text2D(0.0, 1.02, f"({num}) {names.get(num, '')}, P = {bc['P']:.5f} h", transform=ax.transAxes,
                          fontsize=8.5, ha="left", va="bottom")
            if r == nrow - 1:
                ax.text2D(0.5, -0.05, lab, transform=ax.transAxes, ha="center", fontsize=7.5, color="0.35")
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.04, top=0.93, wspace=0.0, hspace=0.12)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main(sys.argv[1:])
