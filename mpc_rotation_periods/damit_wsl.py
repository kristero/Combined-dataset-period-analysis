"""Thin wrapper to run the DAMIT C programs (compiled in WSL, distro myrootfs) on a run directory."""
from __future__ import annotations

import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
DISTRO = "myrootfs"
# on Linux (the LU HPC) the programs are called directly; on Windows through WSL
NATIVE = sys.platform.startswith("linux")
BIN = os.path.join(ROOT, "damit", "build_linux") if NATIVE else "/mnt/c/Users/kn18001/Documents/Asteroids/MPC_redo/damit/build"


def wsl_path(p):
    if NATIVE:
        return os.path.abspath(p)
    p = os.path.abspath(p).replace("\\", "/")
    return "/mnt/" + p[0].lower() + p[2:]


def _bash(cmd):
    return ["bash", "-c", cmd] if NATIVE else ["wsl", "-d", DISTRO, "-e", "bash", "-c", cmd]


# phase function f(alpha) = 1 + a exp(-alpha/d) + k alpha: with relative lightcurves only, a and d are fixed and only
# the slope k is fitted (Durech & Hanus 2018, 2023); fitting all three diverged for some asteroids
PHASE0 = (0.5, 0.1, -0.5)
PHASE_FREE = (0, 0, 1)


def _mask(free_phase):
    if isinstance(free_phase, (tuple, list)):
        return tuple(int(bool(x)) for x in free_phase)
    return (1, 1, 1) if free_phase else (0, 0, 0)


def par_text(lam, beta, P, lm=6, rows=8, conw=0.1, phase=PHASE0, free_phase=PHASE_FREE, c_lambert=0.1,
             stop=50, free_spin=True, t0=0.0, phi0=0.0):
    f = 1 if free_spin else 0
    fa, fd, fk = _mask(free_phase)
    return (f"{lam:.4f}\t{f}\tlambda\n{beta:.4f}\t{f}\tbeta\n{P:.9f}\t{f}\tperiod\n{t0}\tt0\n{phi0}\tphi0\n"
            f"{conw}\tconvexity weight\n{lm} {lm}\tdegree order\n{rows}\trows\n"
            f"{phase[0]}\t{fa}\ta\n{phase[1]}\t{fd}\td\n{phase[2]}\t{fk}\tk\n{c_lambert}\t0\tc\n{stop}\tstop\n")


def scan_text(P_lo, P_hi, coef=0.8, lm=6, rows=8, conw=0.1, phase=PHASE0, free_phase=PHASE_FREE,
              c_lambert=0.1, stop=50, min_iter=10):
    fa, fd, fk = _mask(free_phase)
    return (f"{P_lo:.9f} {P_hi:.9f} {coef}\tperiod start end coef\n{conw}\tconvexity weight\n{lm} {lm}\tdegree order\n"
            f"{rows}\trows\n{phase[0]}\t{fa}\n{phase[1]}\t{fd}\n{phase[2]}\t{fk}\n{c_lambert}\t0\n{stop}\tstop\n"
            f"{min_iter}\tmin iter\n")


def run(rundir, program, args, lcs="lcs", timeout=None):
    """cat <lcs> | <program> <args> inside WSL; returns (stdout, seconds). timeout (s) is enforced inside WSL with
    coreutils timeout, so a stuck program is killed there; stderr is kept to its last 2 kB."""
    tcmd = f"timeout {int(timeout)} " if timeout else ""
    cmd = (f"ulimit -s unlimited; cd '{wsl_path(rundir)}' && cat {lcs} | {tcmd}{BIN}/{program} {args} "
           f"2> >(tail -c 2000 >&2)")
    t = time.time()
    p = subprocess.run(_bash(cmd), capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"{program} failed (rc {p.returncode}) in {rundir}: {p.stderr[-300:]}")
    return p.stdout, time.time() - t


def running_in(rundir):
    """True if a period_scan/convexinv process is running in WSL with rundir (or a subdirectory) as working dir."""
    cmd = "for p in $(pgrep -f 'build[^/]*/(period_scan|convexinv)'); do readlink /proc/$p/cwd; done"
    p = subprocess.run(_bash(cmd), capture_output=True, text=True)
    w = wsl_path(rundir).rstrip("/")
    return any(ln.strip() == w or ln.strip().startswith(w + "/") for ln in p.stdout.splitlines())


def read_par(path):
    with open(path) as f:
        lines = f.read().split("\n")
    lam, beta, P = map(float, lines[0].split())
    t0, phi0 = map(float, lines[1].split())
    a, d, k = map(float, lines[2].split())
    c = float(lines[3])
    return {"lambda": lam, "beta": beta, "P": P, "t0": t0, "phi0": phi0, "a": a, "d": d, "k": k, "c": c}


def read_areas(path):
    import numpy as np
    v = open(path).read().split()
    n = int(v[0])
    arr = np.array(v[1:1 + 4 * n], float).reshape(n, 4)
    return arr[:-1, 0], arr[:-1, 1:], float(arr[-1, 0])     # areas, normals, dark facet area
