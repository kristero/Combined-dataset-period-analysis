"""
Quick-look figure for the alias review: periodogram around a set of periods and folded light
curves at each of them, from the arrays of the combined-dataset run.

Usage: python fold_plot.py <asteroid> <P1> [<P2> ...] [--nterms 4]
Writes review/<asteroid>_folds_<P1>_<P2>...png and prints the power, residual rms and number of
maxima/minima of the Fourier model at every period.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from astropy.timeseries import LombScargle  # noqa: E402
from scipy.signal import find_peaks  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "review")
os.makedirs(OUT, exist_ok=True)


def fourier(t, y, P, n):
    w = 2 * np.pi * 24.0 / P
    A = np.vstack([np.ones_like(t)] + [f(k * w * t) for k in range(1, n + 1) for f in (np.sin, np.cos)]).T
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    ph = np.linspace(0, 1, 500, endpoint=False)
    G = np.vstack([np.ones_like(ph)] + [f(2 * np.pi * k * ph) for k in range(1, n + 1) for f in (np.sin, np.cos)]).T
    return A @ c, ph, G @ c


def main(argv):
    nterms = 2
    if "--nterms" in argv:
        k = argv.index("--nterms")
        nterms = int(argv[k + 1])
        argv = argv[:k] + argv[k + 2:]
    num = int(argv[0])
    Ps = [float(x) for x in argv[1:]]
    a = np.load(os.path.join(ROOT, "results", str(num), f"{num}_arrays.npz"))
    t, y = a["t"], a["y"]
    base = t.max() - t.min()
    fig, axs = plt.subplots(len(Ps), 2, figsize=(9, 2.6 * len(Ps)), squeeze=False, constrained_layout=True)
    for i, P in enumerate(Ps):
        f0 = 24.0 / P
        g = np.arange(f0 - 30 / base, f0 + 30 / base, 1 / (30 * base))
        g = g[g > 0]
        p = LombScargle(t, y, nterms=2).power(g, method="fastchi2", assume_regular_frequency=True)
        k = int(np.argmax(p))
        Pk = 24.0 / g[k]
        axs[i, 0].plot(24.0 / g, p, lw=0.6, color="k")
        axs[i, 0].axvline(Pk, color="r", lw=0.6)
        axs[i, 0].set_xlabel("Period (h)")
        axs[i, 0].set_ylabel("power")
        axs[i, 0].set_title(f"({num}) around {P} h: peak {Pk:.5f} h, power {p[k]:.4f}", fontsize=8)
        m, ph, mg = fourier(t, y, Pk, nterms)
        phase = (t * 24.0 / Pk) % 1.0
        pk, _ = find_peaks(np.r_[mg, mg, mg], prominence=0.02)
        tr, _ = find_peaks(-np.r_[mg, mg, mg], prominence=0.02)
        n_max = int(np.sum((pk >= 500) & (pk < 1000)))
        n_min = int(np.sum((tr >= 500) & (tr < 1000)))
        axs[i, 1].scatter(phase, y, s=1.5, color="0.5", alpha=0.5, lw=0)
        order = np.argsort(phase)
        bins = np.linspace(0, 1, 21)
        idx = np.digitize(phase, bins) - 1
        bm = [np.median(y[idx == j]) if np.sum(idx == j) > 4 else np.nan for j in range(20)]
        axs[i, 1].plot(0.5 * (bins[1:] + bins[:-1]), bm, "ks", ms=3)
        axs[i, 1].plot(ph, mg, "r-", lw=1)
        lo, hi = np.percentile(y, [1, 99])
        axs[i, 1].set_ylim(hi + 0.1, lo - 0.1)
        axs[i, 1].set_xlabel("Rotational phase")
        axs[i, 1].set_title(f"fold at {Pk:.5f} h (n={nterms}): rms {np.std(y - m):.4f}, "
                            f"model max/min {n_max}/{n_min}, amp {mg.max() - mg.min():.2f}", fontsize=8)
        print(f"P={Pk:.5f} h power={p[k]:.4f} rms={np.std(y - m):.4f} extrema={n_max}/{n_min} amp={mg.max() - mg.min():.3f}")
    out = os.path.join(OUT, f"{num}_folds_" + "_".join(f"{P:g}" for P in Ps) + ".png")
    fig.savefig(out, dpi=110)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1:])
