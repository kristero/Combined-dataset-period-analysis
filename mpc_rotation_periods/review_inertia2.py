"""
Review A05, second step: principal-axis check of every refined solution (areas_*) of (3716) and (4239), whose best
models rotate about an axis far from the maximum moment of inertia, and of (4761) and (2503), whose polyhedra failed
in review_checks.py (retried with facet areas floored at 1e-8 of the total, as in damit_shape_compare.py).
Writes review_inertia2.json
"""
import glob
import json
import os
import subprocess

import numpy as np

from review_checks import ROOT, inertia

BIN = os.path.join(ROOT, "damit", "build_linux")
TMP = os.path.join(ROOT, "review_tmp")


def vertices(areas, key):
    out = os.path.join(TMP, key + ".shape")
    tmax = int(os.environ.get("MINK_TIMEOUT", "900"))      # seconds per minkowski attempt

    def run(src):
        cmd = f"ulimit -s unlimited; cat '{src}' | timeout {tmax} '{BIN}/minkowski' | '{BIN}/standardtri' > '{out}'"
        p = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
        return p.returncode == 0 and os.path.getsize(out) > 0

    if not run(areas):
        ln = open(areas).read().split("\n")
        n = int(ln[0])
        ar = np.array([float(ln[1 + 2 * i]) for i in range(n)])
        reg = out + ".areas_floor"
        rows = [str(n)]
        for i in range(n):
            rows += [f"{max(ar[i], 1e-8 * ar.sum()):.12e}", ln[2 + 2 * i]]
        open(reg, "w", newline="\n").write("\n".join(rows) + "\n")
        if not run(reg):
            raise RuntimeError("minkowski failed")
    v = open(out).read().split()
    nv = int(v[0])
    return np.array(v[2:2 + 3 * nv], float).reshape(nv, 3)


def pole_of(areas):
    par = areas.replace("areas_", "par_")
    if os.path.exists(par):
        x = open(par).read().split()
        return [float(x[0]), float(x[1]), float(x[2])]
    return None


os.makedirs(TMP, exist_ok=True)
if __name__ == "__main__":
    out = {}
    for run_dir in ("paper_3716", "paper_4239", "paper_4761", "paper_2503", "paper_2503_half"):
        for areas in sorted(glob.glob(os.path.join(ROOT, "damit_runs", run_dir, "poles", "areas_*"))):
            key = f"{run_dir}:{os.path.basename(areas)}"
            try:
                r = inertia(vertices(areas, key.replace(":", "_")))
                r["pole_P"] = pole_of(areas)
            except Exception as e:
                r = {"error": str(e)[:200]}
            out[key] = r
            print(key, r, flush=True)
    json.dump(out, open(os.path.join(ROOT, "review_inertia2.json"), "w"), indent=1)
