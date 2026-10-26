#!/bin/bash
#SBATCH --job-name=zp_long
#SBATCH --partition=power
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --mem=192G
#SBATCH --output=hpc/zp_long.out
# Long-period analyses after the 703/G96 zero-point split: fine LS, reliability tests, DAMIT window scans and
# 120-3000 h scans (the window and range scans skip asteroids whose data did not change: their 'done' files remain)
cd $HOME/mpc_redo
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ulimit -s unlimited
PY=/software/anaconda3/anaconda3-2023/bin/python3
echo "start $(date) on $(hostname)"
$PY -u ls_longperiod.py --workers 19
$PY -u longperiod_tests.py --workers 9
$PY -u damit_longscan.py --workers 112
$PY -u damit_longscan.py --summary
$PY -u damit_longrange.py --workers 112
$PY -u damit_longrange.py --summary
echo "end $(date)"
