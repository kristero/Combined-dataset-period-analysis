"""
Numbers quoted in Section 4: power of the strongest periodogram point within +/- 0.5 % of the listed periods
(local value at exactly the period for the 'exact' list), from the cached n = 2 periodograms of a run.
python paper_numbers.py --tag=deph  -> paper_figures/runs/<tag>/paper_numbers.json
"""
import json
import os
import sys

import numpy as np

import paper_figures as pf

NEAR = {
    1951: [5.300, 2.650], 1963: [18.164, 9.082, 23.97, 47.8], 2134: [4.115, 2.057, 24.05],
    2150: [6.124, 3.062, 5.431, 2.716], 2607: [2.936, 1.468, 2.77, 3.13, 1.38, 1.57],
    2968: [4.560, 3.831, 4.163, 2.280, 24.0, 48.0, 51.5], 2971: [4.491, 2.245, 4.954, 2.477, 4.11],
    3081: [8.007, 4.004, 24.0, 12.0, 6.860, 3.430], 3173: [45.98, 22.99, 49.9, 24.93, 12.2, 236.0],
    3473: [9.074, 4.537, 7.63, 3.81], 3716: [10.474, 5.237, 13.405, 6.703, 8.59, 4.30, 18.6, 30.4],
    4303: [6.136, 3.068, 7.04, 5.44, 3.52, 2.72],
}
EXACT = {3081: [8.000]}


def main(argv):
    pf._set_options(argv)
    out = {}
    for num, plist in NEAR.items():
        path = os.path.join(pf.LS_CACHE_DIR, f"{num}_LS_n=2.npz")
        if not os.path.exists(path):
            continue
        d = np.load(path)
        f, p = np.asarray(d["frequency"], float), np.asarray(d["power"], float)
        P = 24.0 / f
        r = {"near": {}, "exact": {}, "mean_power": float(np.mean(p)), "std_power": float(np.std(p))}
        for x in plist:
            m = np.abs(P - x) <= 0.005 * x
            k = np.where(m)[0][np.argmax(p[m])]
            r["near"][str(x)] = [float(P[k]), float(p[k])]
        for x in EXACT.get(num, []):
            k = int(np.argmin(np.abs(P - x)))
            r["exact"][str(x)] = [float(P[k]), float(p[k])]
        out[str(num)] = r
        print(num, {k: (round(v[0], 3), round(v[1], 3)) for k, v in r["near"].items()}, r["exact"])
    json.dump(out, open(os.path.join(pf.OUT_DIR, "paper_numbers.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
