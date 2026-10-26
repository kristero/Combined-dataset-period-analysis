"""Compare a re-run tree (default zp_base/) with the original results/ for the asteroids it contains."""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
T = os.path.join(ROOT, sys.argv[1] if len(sys.argv) > 1 else "zp_base")
for num in sorted(os.listdir(os.path.join(T, "results"))):
    print("=====", num)
    r0 = json.load(open(os.path.join(ROOT, "results", num, f"{num}_result.json")))
    r1 = json.load(open(os.path.join(T, "results", num, f"{num}_result.json")))
    for k in ("reference", "datasets_used", "n_combined", "n_after_outlier_removal", "baseline_days", "suggested_P"):
        print(f"  {k}: {'SAME' if r0.get(k) == r1.get(k) else 'DIFF'}  {str(r0.get(k))[:80]} | {str(r1.get(k))[:80]}")
    for k in ("H", "G1", "G2"):
        print(f"  ref fit {k}: {r0['reference_fit'][k]:.6f} | {r1['reference_fit'][k]:.6f}")
    a0 = np.load(os.path.join(ROOT, "results", num, f"{num}_arrays.npz"))
    a1 = np.load(os.path.join(T, "results", num, f"{num}_arrays.npz"))
    same_t = a0["t"].shape == a1["t"].shape and np.allclose(a0["t"], a1["t"], atol=1e-9)
    same_y = a0["y"].shape == a1["y"].shape and np.allclose(a0["y"], a1["y"], atol=1e-9)
    print(f"  arrays t identical: {same_t}  y identical: {same_y}  (n {len(a0['t'])} | {len(a1['t'])})")
    if not same_y and a0["y"].shape == a1["y"].shape:
        print(f"  max |dy| = {np.max(np.abs(a0['y'] - a1['y'])):.3g}")
    b0 = json.load(open(os.path.join(ROOT, "results", num, f"{num}_rebootstrap2.json")))
    b1 = json.load(open(os.path.join(T, "results", num, f"{num}_rebootstrap2.json")))
    for c0, c1 in zip(b0["primary"], b1["primary"]):
        print(f"  cand P {c0['P']:.6f} | {c1['P']:.6f}  win_u {c0['win_u']:.3f} | {c1['win_u']:.3f}  "
              f"win_w {c0['win_w']:.3f} | {c1['win_w']:.3f}")
