"""Checks of the corrected search windows in rebootstrap.py.

1. Candidate periods: the corrected script must refine exactly the same candidates as the legacy run
   (rebootstrap2.py uses them as seeds, so the final results depend on them).
2. Own windows: disjoint, contiguous, containing the candidate frequency, at most +/-TOOTH_RES wide.
3. Legacy overlap: how many candidate pairs had overlapping +/-20 resolution-element windows.
4. After a re-run (--compare): legacy vs corrected win fractions and preliminary proposals.
"""
import json
import os
import sys

import numpy as np

import rebootstrap as rb

ROOT = rb.ROOT
RES = rb.RES


def load_arrays(num):
    a = np.load(os.path.join(RES, str(num), f"{num}_arrays.npz"))
    t, y = a["t"], a["y"]
    return t, y, float(t.max() - t.min())


def check_candidates(nums):
    bad, n_pairs, n_overlap = [], 0, {}
    for num in nums:
        leg = json.load(open(os.path.join(RES, str(num), f"{num}_rebootstrap_legacy.json"), encoding="utf-8"))
        t, y, base = load_arrays(num)
        cands = rb.candidates(num, [], t, y, base)
        new_P = [c["P"] for c in cands]
        old_P = [c["P"] for c in leg["primary"]] + [c["P"] for c in leg["extended"]]
        if sorted(np.round(new_P, 9)) != sorted(np.round(old_P, 9)):
            bad.append((num, old_P, new_P))
        freqs = np.array([c["f"] for c in cands])
        # own windows
        for j, c in enumerate(cands):
            g = rb.grid(c["f"], base)
            m = rb.own_window(g, j, list(freqs), base)
            idx = np.flatnonzero(m)
            assert np.all(np.diff(idx) == 1), (num, j, "not contiguous")
            assert np.abs(g[m] - c["f"]).max() <= rb.TOOTH_RES / base * (1 + 1e-6), (num, j, "too wide")
            assert np.abs(g[m] - c["f"]).min() <= 1.0 / (rb.SPP * base), (num, j, "does not contain f")
            for k, f in enumerate(freqs):
                if k != j:
                    assert np.all(np.abs(g[m] - c["f"]) < np.abs(g[m] - f)), (num, j, k, "shared point")
        # legacy overlap of the +/-20 windows
        ov = []
        for i in range(len(freqs)):
            for k in range(i):
                n_pairs += 1
                if abs(freqs[i] - freqs[k]) < 2 * rb.WIDE_RES / base:
                    ov.append((round(24 / freqs[k], 5), round(24 / freqs[i], 5),
                               round(abs(freqs[i] - freqs[k]) * base, 1)))
        if ov:
            n_overlap[num] = ov
    return bad, n_pairs, n_overlap


def compare(nums):
    rows = {}
    for num in nums:
        leg = json.load(open(os.path.join(RES, str(num), f"{num}_rebootstrap_legacy.json"), encoding="utf-8"))
        new = json.load(open(os.path.join(RES, str(num), f"{num}_rebootstrap.json"), encoding="utf-8"))
        if new.get("windows", {}).get("powers_from") != "wide grid, masked":
            raise SystemExit(f"{num}: {num}_rebootstrap.json is not from the current corrected rebootstrap.py")
        lp = {round(c["P"], 7): c for c in leg["primary"]}
        d = []
        for c in new["primary"]:
            o = lp[round(c["P"], 7)]
            d.append({"P": c["P"], "win_u": [o["win_u"], c["win_u"]], "win_w": [o["win_w"], c["win_w"]],
                      "famwin_u": [o["famwin_u"], c["famwin_u"]], "famwin_w": [o["famwin_w"], c["famwin_w"]],
                      "same_tooth": [o["same_tooth"], c["same_tooth"]], "sigma_all": [o["sigma_all"], c["sigma_all"]]})
        le = {round(c["P"], 7): c for c in leg["extended"]}
        e = [{"P": c["P"], "frac_above_best_primary": [le[round(c["P"], 7)]["frac_above_best_primary"],
                                                        c["frac_above_best_primary"]]} for c in new["extended"]]
        rows[str(num)] = {"primary": d, "extended": e, "near_ties": new.get("near_ties")}
    json.dump(rows, open(os.path.join(ROOT, "rebootstrap_fix_compare.json"), "w"), indent=1)
    n_changed = 0
    for num, r in rows.items():
        ch = [x for x in r["primary"] if abs(x["win_u"][0] - x["win_u"][1]) > 0.02 or abs(x["win_w"][0] - x["win_w"][1]) > 0.02]
        che = [x for x in r["extended"] if abs(x["frac_above_best_primary"][0] - x["frac_above_best_primary"][1]) > 0.02]
        same = all(x["same_tooth"][0] == x["same_tooth"][1] and x["sigma_all"][0] == x["sigma_all"][1] for x in r["primary"])
        if ch or che:
            n_changed += 1
            nt = r["near_ties"]
            print(f"{num}: near ties (<{nt['relative']:g}) u={nt['unweighted']} w={nt['weighted']}; position stats identical={same}")
            for x in ch:
                print(f"   P={x['P']:.5f} win_u {x['win_u'][0]:.2f}->{x['win_u'][1]:.2f}  win_w {x['win_w'][0]:.2f}->{x['win_w'][1]:.2f}"
                      f"  famwin_u {x['famwin_u'][0]:.2f}->{x['famwin_u'][1]:.2f}  famwin_w {x['famwin_w'][0]:.2f}->{x['famwin_w'][1]:.2f}")
            for x in che:
                print(f"   ext P={x['P']:.2f} frac_above {x['frac_above_best_primary'][0]:.2f}->{x['frac_above_best_primary'][1]:.2f}")
    print(f"{n_changed} of {len(rows)} asteroids with win fractions changed by more than 0.02")
    not_same = [n for n, r in rows.items() if not all(x["same_tooth"][0] == x["same_tooth"][1]
                                                       and x["sigma_all"][0] == x["sigma_all"][1] for x in r["primary"])]
    print("position statistics (same_tooth, sigma_all) identical to legacy for all asteroids" if not not_same
          else f"position statistics differ for {not_same}")
    # preliminary proposals of adjudicate_pre.py (read-only call) from the legacy and the corrected bootstrap
    import adjudicate_pre
    leg_prop = json.load(open(os.path.join(ROOT, "proposals_legacy.json"), encoding="utf-8"))
    print("preliminary proposals that change with the corrected windows:")
    for num in nums:
        p = adjudicate_pre.propose(num)
        o = leg_prop[str(num)]
        if abs(p["P"] - o["P"]) > 1e-6 * o["P"] or p["quality_auto"] != o["quality_auto"]:
            print(f"   {num}: P {o['P']:.5f} -> {p['P']:.5f}, quality {o['quality_auto']} -> {p['quality_auto']}")


if __name__ == "__main__":
    nums = sorted(int(d) for d in os.listdir(RES) if d.isdigit()
                  and os.path.exists(os.path.join(RES, d, f"{d}_rebootstrap_legacy.json")))
    if "--compare" in sys.argv:
        compare(nums)
    else:
        bad, n_pairs, ov = check_candidates(nums)
        print(f"candidate lists identical for {len(nums) - len(bad)} of {len(nums)} asteroids")
        for b in bad:
            print("  DIFFERENT:", b)
        print("own windows: disjoint, contiguous, contain the candidate, within +/-TOOTH_RES (all asserts passed)")
        print(f"legacy +/-20 windows overlapped for {sum(map(len, ov.values()))} of {n_pairs} candidate pairs "
              f"in {len(ov)} of {len(nums)} asteroids")
        for n, o in ov.items():
            print(f"  {n}: {o}")
