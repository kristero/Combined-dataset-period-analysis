#!/bin/bash
#SBATCH --job-name=damit_summary
#SBATCH --partition=weakold
#SBATCH --time=1:00:00
#SBATCH --output=hpc/summary.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
cd $HOME/mpc_redo
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ulimit -s unlimited
/software/anaconda3/anaconda3-2023/bin/python3 -u damit_eval.py --summary
