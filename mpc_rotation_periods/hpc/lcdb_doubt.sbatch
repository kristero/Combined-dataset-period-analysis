#!/bin/bash
#SBATCH --job-name=lcdb_doubt
#SBATCH --partition=power
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=hpc/lcdb_doubt.out
# Convex models of (2564) and (2993) at the blind and the LCDB summary period (lcdb_doubt_check.py).
cd $HOME/mpc_redo
export OMP_NUM_THREADS=1 MPLBACKEND=Agg MINK_TIMEOUT=300
PY=$HOME/micromamba/envs/py312/bin/python
echo "start $(date) on $(hostname)"
$PY -u lcdb_doubt_check.py --workers 4
echo "end $(date)"
