"""
Copy the analysis pipeline of the MPC rotation-period paper into the Combined-dataset-period-analysis repository
(folder mpc_rotation_periods/) and commit/push it. The layout mirrors this folder, because every script locates its
inputs relative to its own file. Copied: top-level *.py and *.sh, summary *.json, hpc/ job scripts, data/*.json,
repo_snap/ (the analysis classes used for the paper). Not copied: raw MPC downloads, workbooks, result arrays,
DAMIT runs, the LCDB copy (third-party data), scratch and audit folders.
Usage: python sync_to_git.py "commit message" [--no-push]
"""
import glob
import os
import shutil
import subprocess
import sys

SRC = os.path.dirname(os.path.abspath(__file__))
REPO = r"C:\Users\kn18001\Documents\Asteroids\Combined-dataset-period-analysis"
DST = os.path.join(REPO, "mpc_rotation_periods")
PATTERNS = ["*.py", "*.sh", "*.json", "hpc/*.sbatch", "hpc/*.sh", "hpc/*.lst", "data/*.json",
            "repo_snap/Asteroid_pickle.py", "repo_snap/asteroid_ls.pyc", "repo_snap/phases_and_phi.txt",
            "repo_snap/SNAPSHOT_REV.txt"]
def copy():
    n = 0
    os.makedirs(DST, exist_ok=True)
    shutil.copy2(os.path.join(SRC, "README_pipeline.md"), os.path.join(DST, "README.md"))
    for pat in PATTERNS:
        for f in glob.glob(os.path.join(SRC, pat)):
            rel = os.path.relpath(f, SRC)
            out = os.path.join(DST, rel)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            shutil.copy2(f, out)
            n += 1
    os.makedirs(os.path.join(DST, "repo_snap", "LS_data"), exist_ok=True)
    keep = os.path.join(DST, "repo_snap", "LS_data", ".gitkeep")
    open(keep, "a").close()
    return n


def git(*args):
    p = subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True)
    if p.returncode:
        raise SystemExit(f"git {' '.join(args)} failed:\n{p.stdout}\n{p.stderr}")
    return p.stdout


if __name__ == "__main__":
    msg = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "Update MPC rotation-period pipeline"
    print("copied", copy(), "files into", DST)
    git("add", "--", "mpc_rotation_periods")
    if not git("diff", "--cached", "--name-only", "--", "mpc_rotation_periods").strip():
        print("nothing to commit")
        raise SystemExit(0)
    git("commit", "-m", msg, "--", "mpc_rotation_periods")
    print(git("log", "--oneline", "-1").strip())
    if "--no-push" not in sys.argv:
        git("push", "origin", "HEAD")
        print("pushed")
