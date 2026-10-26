#!/bin/bash
#SBATCH --job-name=dshapes2
#SBATCH --partition=power
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=hpc/damit_shapes2.out
# Rank test, closest-pole distances and ellipsoid comparison of the DAMIT shape test (damit_shape_rank.py)
cd $HOME/mpc_redo
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export DAMIT_BIN=$HOME/mpc_redo/damit/build_linux
ulimit -s unlimited
PY=/software/anaconda3/anaconda3-2023/bin/python3
echo "start $(date) on $(hostname)"
$PY -u damit_shape_rank.py
echo "end $(date)"
