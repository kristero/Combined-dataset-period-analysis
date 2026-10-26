"""
Write the SLURM job scripts for the LU HPC (run locally, then push with hpc_sync.sh):
  hpc/main_<k>.lst + hpc/main.sbatch  array job: per node a balanced chunk of runs: poles -> loao -> ellipsoid -> eval
  hpc/summary.sbatch                  damit_eval.py --summary (after the array job)
  hpc/longls.sbatch                   ls_longperiod.py for the asteroids with P > 24 h
"""
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(ROOT, "damit_runs")
HPC = os.path.join(ROOT, "hpc")
PY = "/software/anaconda3/anaconda3-2023/bin/python3"
N_NODES = 7
os.makedirs(HPC, exist_ok=True)

runs = []
for d in sorted(os.listdir(R)):
    if os.path.isfile(os.path.join(R, d, "prep.json")):
        info = json.load(open(os.path.join(R, d, "prep.json")))
        done = os.path.exists(os.path.join(R, d, "poles", "poles.json"))
        cost = info["n"] * (1.0 + (0 if done else 1.5))
        runs.append((cost, info["num"], d))
runs.sort(reverse=True)
chunks = [[] for _ in range(N_NODES)]
load = [0.0] * N_NODES
for cost, num, d in runs:
    k = load.index(min(load))
    chunks[k].append(num)
    load[k] += cost
for k, c in enumerate(chunks):
    open(os.path.join(HPC, f"main_{k}.lst"), "w", newline="\n").write(",".join(c) + "\n")

common = f"""#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
cd $HOME/mpc_redo
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ulimit -s unlimited
"""
open(os.path.join(HPC, "main.sbatch"), "w", newline="\n").write(f"""#!/bin/bash
#SBATCH --job-name=damit_main
#SBATCH --partition=weakold
#SBATCH --time=10:00:00
#SBATCH --array=0-{N_NODES - 1}
#SBATCH --output=hpc/main_%a.out
{common}NUMS=$(cat hpc/main_${{SLURM_ARRAY_TASK_ID}}.lst)
echo "start $(date) on $(hostname): $NUMS"
{PY} -u damit_run.py poles --workers 16 --nums $NUMS
{PY} -u damit_run.py loao --workers 16 --nums $NUMS
{PY} -u damit_ellipsoid.py --workers 16 --nums $NUMS
{PY} -u damit_eval.py --workers 16 --nums $NUMS --no-summary
echo "end $(date)"
""")
open(os.path.join(HPC, "summary.sbatch"), "w", newline="\n").write(f"""#!/bin/bash
#SBATCH --job-name=damit_summary
#SBATCH --partition=weakold
#SBATCH --time=1:00:00
#SBATCH --output=hpc/summary.out
{common}{PY} -u damit_eval.py --summary
""")
open(os.path.join(HPC, "longls.sbatch"), "w", newline="\n").write(f"""#!/bin/bash
#SBATCH --job-name=longls
#SBATCH --partition=power
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --output=hpc/longls.out
cd $HOME/mpc_redo
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
{PY} -u ls_longperiod.py --workers 19
""")
print("chunks:", [len(c) for c in chunks], "loads:", [round(l) for l in load])
