#!/bin/bash
#SBATCH --job-name=ts_damit
#SBATCH --partition=power
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --output=hpc/testswap_damit.out
# Convex inversion of the six test asteroids added in the revision (same stages as zp_main.sbatch) and the
# long-period Lomb-Scargle and DAMIT window scans (ls_longperiod / damit_longscan skip finished asteroids).
cd $HOME/mpc_redo
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
ulimit -s unlimited
PY=$HOME/micromamba/envs/py312/bin/python
N=249,486,870,1600,1700,7750
echo "start $(date) on $(hostname)"
$PY -u damit_run.py scan --workers 32 --only paper --nums $N
$PY -u damit_run.py poles --workers 32 --only paper --nums $N
$PY -u damit_mirrors.py --only paper --workers 32 --nums $N
$PY -u damit_run.py loao --workers 32 --only paper --nums $N
$PY -u damit_ellipsoid.py --workers 32 --only paper --nums $N
$PY -u damit_eval.py --workers 32 --nums $N --no-summary
$PY -u ls_longperiod.py --workers 6 --nums 249,486,870
$PY -u damit_longscan.py --workers 32
$PY -u damit_longscan.py --summary
echo "end $(date)"
