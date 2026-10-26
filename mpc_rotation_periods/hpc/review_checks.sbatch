#!/bin/bash
#SBATCH --job-name=revchk
#SBATCH --partition=power
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=hpc/review_checks.out
cd $HOME/mpc_redo
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ulimit -s unlimited
PY=/software/anaconda3/anaconda3-2023/bin/python3
echo "start $(date) on $(hostname)"
$PY -u review_checks.py
echo "end $(date)"
