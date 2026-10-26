#!/bin/bash
#SBATCH --job-name=pavpre
#SBATCH --partition=power
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=hpc/pav_preproc.out
# Sensitivity of the separate-dataset periods to the pre-processing (pav_preproc_test.py)
cd $HOME/mpc_redo
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/software/anaconda3/anaconda3-2023/bin/python3
echo "start $(date) on $(hostname)"
$PY -u pav_preproc_test.py 16
echo "end $(date)"
