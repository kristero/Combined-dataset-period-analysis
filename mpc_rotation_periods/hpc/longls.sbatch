#!/bin/bash
#SBATCH --job-name=longls
#SBATCH --partition=power
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --output=hpc/longls.out
cd $HOME/mpc_redo
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/software/anaconda3/anaconda3-2023/bin/python3 -u ls_longperiod.py --workers 19
