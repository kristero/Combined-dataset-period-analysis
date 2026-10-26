#!/bin/bash
#SBATCH --job-name=zp_damit
#SBATCH --partition=weakold
#SBATCH --time=3-00:00:00
#SBATCH --array=0-6
#SBATCH --output=hpc/zp_main_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
# DAMIT re-run after the 703/G96 zero-point split (zp_split.py) for the paper asteroids whose datasets changed
cd $HOME/mpc_redo
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ulimit -s unlimited
PY=/software/anaconda3/anaconda3-2023/bin/python3
NUMS=$(cat hpc/zp_main_${SLURM_ARRAY_TASK_ID}.lst)
echo "start $(date) on $(hostname): $NUMS"
$PY -u damit_run.py scan --workers 16 --nums $NUMS
$PY -u damit_run.py poles --workers 16 --nums $NUMS
$PY -u damit_mirrors.py --workers 16 --nums $NUMS
$PY -u damit_run.py loao --workers 16 --nums $NUMS
$PY -u damit_ellipsoid.py --workers 16 --nums $NUMS
$PY -u damit_eval.py --workers 16 --nums $NUMS --no-summary
echo "end $(date)"
