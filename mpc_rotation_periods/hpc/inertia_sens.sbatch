#!/bin/bash
#SBATCH --job-name=inert_sens
#SBATCH --partition=power
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --output=hpc/inertia_sens.out
# Check 2, sensitivity: convex models of the validation asteroids at the strongest wrong candidate of the blind run.
cd $HOME/mpc_redo
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg MINK_TIMEOUT=300
PY=$HOME/micromamba/envs/py312/bin/python
echo "start $(date) on $(hostname)"
$PY -u inertia_sensitivity.py --workers 30
echo "end $(date)"
