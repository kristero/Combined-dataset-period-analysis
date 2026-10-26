"""
Isolated trees for re-running the period pipeline (the main MPC_redo results stay untouched).
  repo_snap/   the analysis classes of Combined-dataset-period-analysis at git HEAD (51f596b), read with `git show`;
               the working copy of that repository has since been changed for another paper and is not used.
  zp/          zero-point-corrected run: 703 and G96 datasets split at 2018 July 1 (zp_split.py writes zp/data/)
  zp_base/     reproducibility check: the original (unsplit) data of a few asteroids with the same snapshot
Every tree is a ROOT for the copied scripts (they locate their inputs relative to their own file).
Usage: python zp_setup.py
"""
import os
import shutil
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ORIG = r"C:\Users\kn18001\Documents\Asteroids\Combined-dataset-period-analysis"
SNAP = os.path.join(ROOT, "repo_snap")
SCRIPTS = ["mpc_redo.py", "rebootstrap.py", "rebootstrap2.py", "quality_rule.py", "finalize.py", "amp_data.py",
           "amp_methods.py", "amp_geometry.py", "geometry_validate.py", "geometry_reanalysis.py",
           "geometry_zeropoint_effect.py", "ls_longperiod.py", "longperiod_tests.py"]
JSONS = ["published.json", "adopted.json", "overrides.json", "proposals.json", "review_results.json",
         "quality_final.json", "geometry_results.json", "figure_info.json"]
DATA = ["mpc_obscodes.json", "lcdb_amplitudes.json", "sbdb_phys_orbit.json", "dataset_precision.json",
        "lcdb_poles.json"]


def snapshot():
    os.makedirs(os.path.join(SNAP, "LS_data"), exist_ok=True)
    for f in ("Asteroid_pickle.py", "asteroid_ls.py", "phases_and_phi.txt"):
        txt = subprocess.run(["git", "-C", REPO_ORIG, "show", f"HEAD:{f}"], capture_output=True, check=True).stdout
        open(os.path.join(SNAP, f), "wb").write(txt)
    rev = subprocess.run(["git", "-C", REPO_ORIG, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    open(os.path.join(SNAP, "SNAPSHOT_REV.txt"), "w").write(rev + "\n")


def tree(name):
    T = os.path.join(ROOT, name)
    os.makedirs(os.path.join(T, "data"), exist_ok=True)
    for s in SCRIPTS:
        txt = open(os.path.join(ROOT, s), encoding="utf-8").read()
        if s == "mpc_redo.py":
            old = 'REPO = r"C:\\Users\\kn18001\\Documents\\Asteroids\\Combined-dataset-period-analysis"'
            assert txt.count(old) == 1
            txt = txt.replace(old, f'REPO = r"{SNAP}"')
        open(os.path.join(T, s), "w", encoding="utf-8", newline="\n").write(txt)
    for j in JSONS:
        if os.path.exists(os.path.join(ROOT, j)):
            shutil.copy(os.path.join(ROOT, j), os.path.join(T, j))
    for d in DATA:
        if os.path.exists(os.path.join(ROOT, "data", d)):
            shutil.copy(os.path.join(ROOT, "data", d), os.path.join(T, "data", d))
    os.makedirs(os.path.join(T, "validation"), exist_ok=True)
    shutil.copy(os.path.join(ROOT, "validation", "sample.json"), os.path.join(T, "validation", "sample.json"))
    return T


def base_data(nums):
    T = os.path.join(ROOT, "zp_base")
    for sub in ("workbooks", "obs_tables"):
        os.makedirs(os.path.join(T, "data", sub), exist_ok=True)
    for n in nums:
        shutil.copy(os.path.join(ROOT, "data", "workbooks", f"{n}-mpc.xlsx"), os.path.join(T, "data", "workbooks"))
        for suf in ("_info.json", "_obs.csv"):
            shutil.copy(os.path.join(ROOT, "data", "obs_tables", f"{n}{suf}"), os.path.join(T, "data", "obs_tables"))


if __name__ == "__main__":
    snapshot()
    tree("zp")
    tree("zp_base")
    base_data([2607, 4303])
    print("snapshot", open(os.path.join(SNAP, "SNAPSHOT_REV.txt")).read().strip(), "; trees zp/ and zp_base/ ready")
