#!/bin/bash
#SBATCH --job-name=fine_chk
#SBATCH --partition=power
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --output=hpc/fine_checks.out
# Finer pole grids for the models that fail the principal-axis check, and finer periodogram, period scan and pole
# grid for the asteroids flagged by the synodic-sidereal check (fine_checks.py).
cd $HOME/mpc_redo
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg MINK_TIMEOUT=300
PY=$HOME/micromamba/envs/py312/bin/python
echo "start $(date) on $(hostname)"
$PY -u fine_checks.py inertia synsid --workers 30
echo "end $(date)"
