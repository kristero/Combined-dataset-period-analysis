#!/bin/bash
#SBATCH --job-name=longrange
#SBATCH --partition=power
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=128G
#SBATCH --output=hpc/longrange.out
cd $HOME/mpc_redo
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ulimit -s unlimited
/software/anaconda3/anaconda3-2023/bin/python3 -u damit_longrange.py --workers 96
