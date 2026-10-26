#!/bin/bash
#SBATCH --job-name=longscan
#SBATCH --partition=power
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G
#SBATCH --output=hpc/longscan.out
cd $HOME/mpc_redo
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ulimit -s unlimited
/software/anaconda3/anaconda3-2023/bin/python3 -u damit_longscan.py --workers 64
