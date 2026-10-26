"""
One-page figure of the convex models of Table tab:damit (manuscript_revised/fig_damit_shapes.pdf and .png): for every
asteroid of damit_paper_table.json ("models") one cell with
  - the convex shape of the best solution, rendered in grey with Lambert shading (light from the upper left), seen
    from a direction in the ecliptic plane perpendicular to the spin axis (from ecliptic longitude lambda - 90 deg,
    ecliptic north up), so that the spin axis lies in the plane of the sky and makes the angle beta with the ecliptic
    (dashed line). The red arrow is the spin axis and points to the pole; the arc and the signed number give beta. The
    rotational phase is the one with the largest projected area. Under the shape: P_sid and the pole(s) of
    Table tab:damit (a second pole is the mirror solution);
  - the Lomb-Scargle periodogram (nterms = 2) of the combined data, and the light curve folded at the adopted period,
    drawn with the functions of make_figures.py (same data, colours and conventions as Figures lc01-lc03: yellow-brown
    points = datasets reported to 0.01 mag, black squares = means in 18 deg bins incl. the 0.1-mag datasets, blue curve
    = two-term Fourier model, red dot = adopted period, dashed line = 5 sigma threshold, open circles = competing peaks).
The last cell holds a compact key.

The polyhedron is built from the Gaussian image of the best solution (areas file) with DAMIT minkowski | standardtri
(WSL, as in damit_shapes.polyhedron) but cached in shapes_page_cache/ under the md5 of the areas file, so the
read-only damit_runs/ tree is never written and a re-fitted model is rebuilt automatically. The best solution is
poles.json["best"]; its areas file is areas_best when par_best holds that solution, otherwise areas_<idx>.

Asteroids without a determined period (adopted.json status "undetermined") are skipped.

Usage:  python damit_shapes_page.py [--out manuscript_revised/fig_damit_shapes.pdf] [--pole-source table|model]
                                    [--only 1820,1942] [--recompute] [--workers 4]
  --pole-source table  (default) orient the body with the first pole of damit_paper_table.json (= Table tab:damit);
                model  orient it with the pole of the best solution itself (eval.json C), from which the shape comes.
Writes also damit_shapes_page_info.json (files used, poles, projected-axis check).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

sys.dont_write_bytecode = True          # never write __pycache__ (make_figures puts a read-only repo on sys.path)

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.patches import Arc, FancyArrowPatch  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import make_figures as MF  # noqa: E402  (style, data loading, periodogram and fold panels)
import damit_wsl as W  # noqa: E402

RUNS = os.path.join(ROOT, "damit_runs")
CACHE = os.path.join(ROOT, "shapes_page_cache")
TABLE = os.path.join(ROOT, "damit_paper_table.json")
INFO = os.path.join(ROOT, "damit_shapes_page_info.json")
OUT = os.path.join(ROOT, "manuscript_revised", "fig_damit_shapes.pdf")

# ---- appearance
GREY = np.array([0.80, 0.79, 0.77])      # albedo colour of the surface (neutral, very slightly warm grey)
AMBIENT = 0.12                            # fraction of the albedo seen on unlit facets
LIGHT = np.array([-0.55, 0.60, 0.75])     # light direction in screen coordinates (right, up, towards the viewer)
AXIS_COLOR = "#c0392b"
ECL_COLOR = "0.55"
FONT = 7.0                                # cell header; smallest text on the page is 6.5 pt

# ---- page layout (inches)
FIG_W = 7.1
NCOLS = 3
CELL_H = 1.45
M_LEFT, M_RIGHT, M_TOP, M_BOTTOM = 0.02, 0.02, 0.03, 0.17
SHAPE = 0.93                              # side of the (square) shape panel
HEAD = 0.14                               # header band above the panels
GAP_Y = 0.33                              # room for the y tick labels and y label of the light-curve panels
PAD_R = 0.05
PG_H, PG_GAP, LC_H = 0.34, 0.13, 0.58
LINE = 6.5 * 1.32 / 72                    # height of one 6.5-pt text line (inches)
TXT_X = (0.0, 0.04)                       # left edge of the text under the shape from the cell edge (inches):
                                          # first column (next to the y labels), other columns
LIM_X, LIM_Y = (-1.30, 1.62), (-1.46, 1.46)   # shape panel limits (body radius 1)


def vec(lam, beta):
    lam, beta = np.radians(lam), np.radians(beta)
    return np.array([np.cos(beta) * np.cos(lam), np.cos(beta) * np.sin(lam), np.sin(beta)])


def ang(u, v):
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))))


def norm_pole(lam, beta):
    v = vec(lam, beta)
    return float(np.degrees(np.arctan2(v[1], v[0])) % 360.0), float(np.degrees(np.arcsin(np.clip(v[2], -1, 1))))


# ------------------------------------------------------------------ convex model files and polyhedron
def best_solution(num):
    """(areas file, best solution dict, eval C dict) of the convex model of num."""
    pdir = os.path.join(RUNS, f"paper_{num}", "poles")
    pj = json.load(open(os.path.join(pdir, "poles.json")))
    best = pj.get("best") or min(pj["solutions"], key=lambda r: r["dev"])
    C = json.load(open(os.path.join(RUNS, f"paper_{num}", "eval.json")))["C"]
    vb = vec(best["lambda"], best["beta"])
    for name in ("best", str(best.get("idx", "")) if "idx" in best else None):
        if name is None:
            continue
        par = os.path.join(pdir, f"par_{name}")
        if not os.path.exists(par):
            continue
        p = W.read_par(par)
        if ang(vec(p["lambda"], p["beta"]), vb) < 0.15 and abs(p["P"] - best["P"]) < 1e-7:
            return os.path.join(pdir, f"areas_{name}"), best, C
    raise RuntimeError(f"({num}): no areas file matches poles.json['best']")


def shape_file(num, areas):
    h = hashlib.md5(open(areas, "rb").read()).hexdigest()[:10]
    return os.path.join(CACHE, f"paper_{num}_{os.path.basename(areas)}_{h}.txt")


def build_shape(areas, out):
    """DAMIT minkowski | standardtri (WSL) on an areas file; output written to the cache only."""
    tmp = out + ".part"
    cmd = (f"ulimit -s unlimited; cd '{W.wsl_path(CACHE)}' && cat '{W.wsl_path(areas)}' | timeout 1800 {W.BIN}/minkowski "
           f"| {W.BIN}/standardtri > '{W.wsl_path(tmp)}'")
    p = subprocess.run(W._bash(cmd), capture_output=True, text=True)
    if p.returncode != 0 or not os.path.exists(tmp) or os.path.getsize(tmp) == 0:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(f"minkowski failed for {areas}: {p.stderr[-300:]}")
    read_shape(tmp)                      # validates
    os.replace(tmp, out)
    return out


def read_shape(path):
    v = open(path).read().split()
    nv, nf = int(v[0]), int(v[1])
    if len(v) != 2 + 3 * nv + 3 * nf or nf != 2 * nv - 4:
        raise RuntimeError(f"{path}: not a closed triangulated polyhedron ({nv} vertices, {nf} facets, {len(v)} numbers)")
    verts = np.array(v[2:2 + 3 * nv], float).reshape(nv, 3)
    faces = np.array(v[2 + 3 * nv:], int).reshape(nf, 3) - 1
    return verts, faces


def prepare_body(verts, faces):
    """Outward-oriented facets, centred on the volume centroid, scaled to a largest radius of 1."""
    tri = verts[faces]
    cen = tri.mean(axis=1)
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    flip = np.sum(nrm * (cen - verts.mean(axis=0)), axis=1) < 0      # convex body: outward = away from the interior
    faces = np.where(flip[:, None], faces[:, ::-1], faces)
    tri = verts[faces]
    vol6 = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2]))   # signed tetrahedra (origin, facet)
    centroid = np.sum(vol6[:, None] * tri.sum(axis=1), axis=0) / (4.0 * vol6.sum())
    v = verts - centroid
    v = v / np.linalg.norm(v, axis=1).max()
    tri = v[faces]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area2 = np.linalg.norm(nrm, axis=1)
    nrm = np.where((area2 < 1e-12)[:, None], tri.mean(axis=1), nrm)    # degenerate facets: radial direction
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    return {"verts": v, "faces": faces, "normals": nrm, "area": 0.5 * area2, "volume": vol6.sum() / 6.0}


def body_to_ecliptic(lam, beta, phi):
    """Rotation matrix (columns = body x, y, z in ecliptic coordinates): z along the pole, x at rotation angle phi."""
    s = vec(lam, beta)
    a = np.cross([0.0, 0.0, 1.0], s)
    a = a / np.linalg.norm(a) if np.linalg.norm(a) > 1e-8 else np.array([1.0, 0.0, 0.0])
    b = np.cross(s, a)
    ex = np.cos(phi) * a + np.sin(phi) * b
    ey = -np.sin(phi) * a + np.cos(phi) * b
    return np.column_stack([ex, ey, s])


def view_frame(lam):
    """Screen axes for a viewer in the ecliptic plane at longitude lam - 90 deg: right = pole-longitude direction,
    up = ecliptic north, d = towards the viewer (right x up = d)."""
    lr = np.radians(lam)
    right = np.array([np.cos(lr), np.sin(lr), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    return right, up, np.cross(right, up)


def best_phase(body, lam, beta):
    """Rotation angle of the body about its spin axis giving the largest projected area."""
    _, _, d = view_frame(lam)
    phis = np.radians(np.arange(0.0, 180.0, 2.0))
    proj = [np.sum(body["area"] * np.clip(body["normals"] @ (body_to_ecliptic(lam, beta, p).T @ d), 0, None)) for p in phis]
    return float(phis[int(np.argmax(proj))])


def draw_shape(ax, body, lam, beta):
    """Grey Lambert-shaded body, ecliptic line, spin axis arrow and latitude arc. Returns the check values."""
    right, up, d = view_frame(lam)
    phi = best_phase(body, lam, beta)
    R = body_to_ecliptic(lam, beta, phi)
    S = np.column_stack([right, up, d])                  # ecliptic -> screen
    v = body["verts"] @ R.T @ S                          # screen coordinates (X, Y, depth)
    n = body["normals"] @ R.T @ S
    tri = v[body["faces"]]
    vis = n[:, 2] > 1e-9
    order = np.argsort(tri[vis][:, :, 2].mean(axis=1))   # back to front (not needed for a convex body; harmless)
    L = LIGHT / np.linalg.norm(LIGHT)
    shade = np.clip(n[vis] @ L, 0.0, 1.0)
    cols = np.clip(GREY[None, :] * (AMBIENT + (1 - AMBIENT) * shade[:, None]), 0, 1)[order]
    polys = tri[vis][order][:, :, :2]
    pc = PolyCollection(polys, facecolors=cols, edgecolors=cols, linewidths=0.25, zorder=3, rasterized=True)
    ax.add_collection(pc)
    # spin axis on the screen: (cos beta, sin beta) by construction; checked from the projected pole vector
    s_scr = vec(lam, beta) @ S
    u = s_scr[:2] / np.linalg.norm(s_scr[:2])
    xy = v[:, :2]
    h_pos, h_neg = float(np.max(xy @ u)), float(np.max(xy @ -u))
    # ecliptic plane (edge-on), behind the body
    ax.plot([-1.5, 1.5], [0, 0], ls=(0, (3, 2)), lw=0.5, color=ECL_COLOR, zorder=1)
    # latitude arc between the ecliptic and the axis, just outside the silhouette
    th = np.radians(np.linspace(0.0, beta, 30))
    h_arc = max(float(np.max(xy @ np.array([np.cos(t), np.sin(t)]))) for t in th)
    r_arc = h_arc + 0.10
    tip = max(h_pos + 0.36, r_arc + 0.22)
    arr = FancyArrowPatch(tuple(-(h_neg + 0.20) * u), tuple(tip * u), arrowstyle="-|>,head_length=4.0,head_width=2.0",
                          lw=0.9, color=AXIS_COLOR, zorder=2, shrinkA=0, shrinkB=0, mutation_scale=1.0)
    ax.add_patch(arr)
    lo, hi = sorted((0.0, beta))
    ax.add_patch(Arc((0, 0), 2 * r_arc, 2 * r_arc, theta1=lo, theta2=hi, lw=0.6, color=AXIS_COLOR, zorder=2))
    sgn = 1.0 if beta >= 0 else -1.0
    if abs(beta) >= 30:                  # number inside the wedge, beyond the arc
        tm, va = np.radians(beta / 2.0), "center"
    else:                                # small wedge: number on the far side of the arrow
        tm, va = np.radians(beta + 8.0 * sgn), ("bottom" if beta >= 0 else "top")
    r_lab = r_arc + 0.09
    ax.text(r_lab * np.cos(tm), r_lab * np.sin(tm), f"{beta:+.0f}°".replace("-", "−"), fontsize=6.5, color=AXIS_COLOR,
            ha="left", va=va, zorder=5)
    ax.set_xlim(*LIM_X)
    ax.set_ylim(*LIM_Y)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return {"phi_deg": float(np.degrees(phi)), "axis_angle_on_screen_deg": float(np.degrees(np.arctan2(u[1], u[0]))),
            "axis_depth_component": float(s_scr[2]), "n_visible_facets": int(vis.sum())}


# ------------------------------------------------------------------ text helpers
def fmt_sig(P, s):
    """Adopted period with the decimals of the tables (uncertainty to two significant digits)."""
    dec = min(max(0, -int(np.floor(np.log10(s))) + 1), 6)
    return f"{P:.{dec}f}"


def fmt_psid(P, s):
    """Sidereal period with the decimals of Table tab:damit (damit_paper_table.fmt_P)."""
    nd = max(0, int(np.ceil(-np.log10(s))) + 1) if s > 0 else 6
    return f"{P:.{nd}f}"


def fmt_pole(lam, beta):
    b = f"{beta:+.0f}".replace("-", "−")
    return f"({lam:.0f}°, {b}°)"


# ------------------------------------------------------------------ page
def key_cell(fig, ax, any_competing):
    """Compact key: a small diagram of the axis drawing and the symbols of the periodogram and light-curve panels."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    bb = ax.get_position()
    aspect = (bb.height * fig.get_figheight()) / (bb.width * fig.get_figwidth())   # data units: y per x
    ax.text(0.0, 1.0, "Key", fontsize=FONT, fontweight="bold", va="top", ha="left")
    # diagram: ecliptic (dashed), spin axis at +50 deg with its latitude arc
    x_c, y_c, r = 0.045, 0.745, 0.11
    t = np.radians(50.0)
    ax.plot([x_c - 0.035, x_c + 0.13], [y_c, y_c], ls=(0, (3, 2)), lw=0.5, color=ECL_COLOR)
    ax.add_patch(FancyArrowPatch((x_c - 0.02 * np.cos(t), y_c - 0.02 * np.sin(t) / aspect),
                                 (x_c + 1.25 * r * np.cos(t), y_c + 1.25 * r * np.sin(t) / aspect), lw=0.9,
                                 color=AXIS_COLOR, arrowstyle="-|>,head_length=4.0,head_width=2.0", mutation_scale=1.0,
                                 shrinkA=0, shrinkB=0))
    ax.add_patch(Arc((x_c, y_c), 2 * 0.75 * r, 2 * 0.75 * r / aspect, theta1=0, theta2=50, lw=0.6, color=AXIS_COLOR))
    ax.text(x_c + 0.9 * r * np.cos(0.3 * t), y_c + 0.9 * r * np.sin(0.3 * t) / aspect, "β", fontsize=6.5,
            color=AXIS_COLOR, ha="left", va="bottom")
    x_sym, x_txt = 0.07, 0.19
    ax.text(x_txt, y_c + 0.035, "spin axis, pointing to the pole (λ, β);\n"
            "β = angle to the ecliptic (dashed)", fontsize=6.5, va="center", ha="left", linespacing=1.2)
    rows = [("dot", "adopted period"), ("thr", "5σ threshold")]
    if any_competing:
        rows.append(("open", "competing peak"))
    rows += [("pts", "measurements reported to 0.01 mag"),
             ("sq", "means in 18° phase bins"),
             ("fourier", "two-term Fourier model")]
    y0, dy = 0.585, 0.083
    for i, (kind, text) in enumerate(rows):
        y = y0 - i * dy
        if kind == "dot":
            ax.scatter([x_sym], [y], s=18, color="#c0392b", lw=0)
        elif kind == "thr":
            ax.plot([x_sym - 0.06, x_sym + 0.06], [y, y], ls="--", lw=0.5, color="0.5")
        elif kind == "open":
            ax.scatter([x_sym], [y], s=18, facecolors="none", edgecolors="#2471a3", lw=0.9)
        elif kind == "pts":
            pal = plt.get_cmap("YlOrBr")(np.linspace(0.45, 0.95, 5))
            ax.scatter(x_sym + np.linspace(-0.05, 0.05, 5), y + np.array([0.01, -0.012, 0.004, -0.008, 0.012]), s=2.5,
                       color=pal, lw=0)
        elif kind == "sq":
            ax.errorbar([x_sym], [y], yerr=[0.025], fmt="s", ms=2.3, color="black", elinewidth=0.6, capsize=0)
        elif kind == "fourier":
            xx = np.linspace(x_sym - 0.06, x_sym + 0.06, 30)
            ax.plot(xx, y + 0.02 * np.sin(2 * np.pi * (xx - xx[0]) / 0.12 * 1.0), color=MF.MODEL_COLOR, lw=1.1)
        ax.text(x_txt, y, text, fontsize=6.5, va="center", ha="left")
    ax.text(0.0, y0 - len(rows) * dy + 0.01,
            "Shapes seen from the ecliptic, ⊥ to the axis,\nlit from the upper left; sizes arbitrary.",
            fontsize=6.5, va="top", ha="left", linespacing=1.2, color="0.25")


def text_overlaps(fig, tol_pt=1.5):
    """Pairs of visible, non-empty texts (figure texts, axes texts, tick and axis labels) whose boxes intersect by more
    than tol_pt in both directions (mathtext boxes include some empty descent), and texts that leave the figure."""
    pad = tol_pt * fig.dpi / 72.0
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    items = [t for t in fig.texts]
    for ax in fig.axes:
        items += list(ax.texts) + [ax.xaxis.label, ax.yaxis.label]
        if ax.axison:
            for axis in (ax.xaxis, ax.yaxis):
                lo, hi = sorted(axis.get_view_interval())
                for tick in axis.get_major_ticks():         # only ticks inside the view limits are drawn
                    if lo - 1e-9 * abs(hi - lo) <= tick.get_loc() <= hi + 1e-9 * abs(hi - lo) and tick.label1.get_visible():
                        items.append(tick.label1)
    boxes = []
    for t in items:
        if t.get_visible() and t.get_text().strip():
            boxes.append((t.get_text(), t.get_window_extent(rend).expanded(1.0, 1.0)))
    W_, H_ = fig.bbox.width, fig.bbox.height
    out = [(s, "<figure edge>") for s, b in boxes if b.x0 < -pad or b.y0 < -pad or b.x1 > W_ + pad or b.y1 > H_ + pad]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            if a.x0 < b.x1 - pad and b.x0 < a.x1 - pad and a.y0 < b.y1 - pad and b.y0 < a.y1 - pad:
                out.append((boxes[i][0], boxes[j][0]))
    return out


def make_page(models, shapes, pole_source, out):
    n = len(models)
    nrows = int(np.ceil((n + 1) / NCOLS))
    fig_h = M_TOP + nrows * CELL_H + M_BOTTOM
    cell_w = (FIG_W - M_LEFT - M_RIGHT) / NCOLS
    info = {}
    rc = {"font.size": FONT, "axes.labelsize": 6.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
          "axes.titlesize": FONT, "xtick.major.size": 2.2, "ytick.major.size": 2.2, "xtick.major.pad": 1.5,
          "ytick.major.pad": 1.5, "axes.linewidth": 0.5, "mathtext.fontset": "dejavuserif",
          "savefig.bbox": "standard"}           # keep the page size exactly (make_figures sets "tight")
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(FIG_W, fig_h))

        def axes_in(x, y, w, h):
            return fig.add_axes([x / FIG_W, y / fig_h, w / FIG_W, h / fig_h])

        any_competing = False
        for i, m in enumerate(models):
            num = m["num"]
            row, col = divmod(i, NCOLS)
            x0 = M_LEFT + col * cell_w
            top = fig_h - M_TOP - row * CELL_H
            last_row = (i + NCOLS >= n)          # no asteroid cell below this one
            ad = MF.ADOPTED[num]
            body, src = shapes[num]
            if pole_source == "table":
                lam, beta = m["poles"][0]
            else:
                lam, beta = norm_pole(src["C"]["lambda"], src["C"]["beta"])
            # header: name (left) and adopted period + quality (right)
            fig.text(x0 / FIG_W, (top - 0.005) / fig_h, MF.label(int(num)), fontsize=FONT, ha="left", va="top")
            fig.text((x0 + SHAPE + GAP_Y) / FIG_W, (top - 0.005) / fig_h,     # left edge of the periodogram panel
                     f"$P$ = {fmt_sig(ad['P'], ad['sigma_P'])} h, {ad['quality']}", fontsize=FONT, ha="left", va="top")
            # shape
            axs = axes_in(x0, top - HEAD - SHAPE, SHAPE, SHAPE)
            chk = draw_shape(axs, body, lam, beta)
            # convex-model results under the shape: P_sid, pole, mirror pole (one 6.5-pt line each)
            y_txt = top - HEAD - SHAPE + 0.035
            lines = [(f"$P_{{\\rm sid}}$ = {fmt_psid(m['P_sid'], m['sigma_sid'])} h", "black"),
                     (f"pole {fmt_pole(lam, beta)}", "black")]      # the drawn pole (= table pole by default)
            if len(m["poles"]) > 1:
                lines.append(("mirror " + "; ".join(fmt_pole(*p) for p in m["poles"][1:]), "0.35"))
            for j, (txt, colr) in enumerate(lines):
                fig.text((x0 + TXT_X[min(col, 1)]) / FIG_W, (y_txt - j * LINE) / fig_h, txt, fontsize=6.5, ha="left",
                         va="top", color=colr)
            # periodogram and folded light curve (make_figures conventions)
            px = x0 + SHAPE + GAP_Y
            pw = cell_w - SHAPE - GAP_Y - PAD_R
            axp = axes_in(px, top - HEAD - 0.02 - PG_H, pw, PG_H)
            axf = axes_in(px, top - HEAD - 0.02 - PG_H - PG_GAP - LC_H, pw, LC_H)
            a, _ = MF.load(int(num))
            shown, left_out = MF.periodogram_axis(axp, a, ad, int(num), show_ylabel=(col == 0))
            any_competing |= bool(shown)
            ticks = list(axp.get_xticks())
            axp.xaxis.set_major_formatter(FuncFormatter(lambda val, _, t=max(ticks): f"{val:g}" + (" h" if val == t else "")))
            amp, amp_err, n_all, n_pts = MF.fold_axis(axf, a, ad, int(num), show_xlabels=True, show_ylabel=(col == 0))
            axf.yaxis.set_major_locator(MaxNLocator(3))
            if not last_row:
                axf.set_xlabel("")
            if col == 0:
                axp.set_ylabel("Power", labelpad=1)
                axf.set_ylabel("Reduced mag", labelpad=1)
            info[num] = {"areas_file": os.path.relpath(src["areas"], ROOT), "shape_file": os.path.relpath(src["shape"], ROOT),
                         "best_solution": {"lambda": src["best"]["lambda"], "beta": src["best"]["beta"], "P": src["best"]["P"],
                                           "dev": src["best"]["dev"], "idx": src["best"].get("idx")},
                         "pole_model_C": norm_pole(src["C"]["lambda"], src["C"]["beta"]), "poles_table": m["poles"],
                         "pole_drawn": [lam, beta], "pole_source": pole_source,
                         "sep_model_vs_table_deg": ang(vec(*norm_pole(src["C"]["lambda"], src["C"]["beta"])), vec(*m["poles"][0])),
                         "P_adopted": ad["P"], "quality": ad["quality"], "P_sid": m["P_sid"],
                         "n_vertices": int(len(body["verts"])), "n_facets": int(len(body["faces"])),
                         "competing_shown": [p for p, _ in shown], "n_fold": n_all, "n_points_drawn": n_pts,
                         "fold_amplitude": amp, **chk}
            assert abs(chk["axis_angle_on_screen_deg"] - beta) < 0.01 and abs(chk["axis_depth_component"]) < 1e-9, num
        # key in the next free cell
        k = n
        row, col = divmod(k, NCOLS)
        x0 = M_LEFT + col * cell_w
        top = fig_h - M_TOP - row * CELL_H
        key_cell(fig, axes_in(x0 + 0.10, top - CELL_H + 0.06, cell_w - 0.12, CELL_H - 0.08), any_competing)
        overlaps = text_overlaps(fig)
        for a_, b_ in overlaps:
            print("  text overlap:", repr(a_), "<->", repr(b_))
        info["_text_overlaps"] = [[a_, b_] for a_, b_ in overlaps]
        fig.savefig(out, bbox_inches=None, dpi=400)          # dpi of the rasterized parts (shapes, data points)
        fig.savefig(out[:-4] + ".png", dpi=200, bbox_inches=None)
        plt.close(fig)
    print("wrote", out, f"({FIG_W} x {fig_h:.2f} in)")
    return info


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--pole-source", choices=("table", "model"), default="table")
    ap.add_argument("--only", default="")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)
    models = json.load(open(TABLE))["models"]
    if args.only:
        keep = set(args.only.split(","))
        models = [m for m in models if m["num"] in keep]
    skipped = [m["num"] for m in models if MF.ADOPTED[m["num"]].get("status") == "undetermined"]
    models = [m for m in models if m["num"] not in skipped]
    if skipped:
        print("skipped (no period determined):", skipped)
    os.makedirs(CACHE, exist_ok=True)
    src = {}
    todo = []
    for m in models:
        areas, best, C = best_solution(m["num"])
        shp = shape_file(m["num"], areas)
        src[m["num"]] = {"areas": areas, "best": best, "C": C, "shape": shp}
        if args.recompute or not os.path.exists(shp):
            todo.append((areas, shp))
    if todo:
        print(f"building {len(todo)} polyhedra with minkowski ({min(args.workers, 4)} at a time) ...", flush=True)
        with ThreadPoolExecutor(max_workers=min(args.workers, 4)) as ex:
            for f in ex.map(lambda t: build_shape(*t), todo):
                print("  ", os.path.basename(f), flush=True)
    shapes = {num: (prepare_body(*read_shape(s["shape"])), s) for num, s in src.items()}
    info = make_page(models, shapes, args.pole_source, args.out)
    info_path = INFO if os.path.abspath(args.out) == os.path.abspath(OUT) else os.path.splitext(args.out)[0] + "_info.json"
    with open(info_path, "w", encoding="utf-8") as fh:
        json.dump({"out": os.path.relpath(args.out, ROOT), "pole_source": args.pole_source, "skipped": skipped,
                   "asteroids": info}, fh, indent=1)
    print("wrote", info_path)


if __name__ == "__main__":
    main(sys.argv[1:])
