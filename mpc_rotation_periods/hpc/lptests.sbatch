#!/bin/bash
#SBATCH --job-name=lptests
#SBATCH --partition=power
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --output=hpc/lptests.out
cd $HOME/mpc_redo
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/software/anaconda3/anaconda3-2023/bin/python3 -u longperiod_tests.py --workers 9
