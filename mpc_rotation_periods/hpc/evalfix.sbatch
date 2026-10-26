#!/bin/bash
#SBATCH --job-name=evalfix
#SBATCH --partition=weakold
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --output=hpc/evalfix.out
cd $HOME/mpc_redo
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/software/anaconda3/anaconda3-2023/bin/python3 -u damit_eval.py --workers 16 --missing --no-summary
