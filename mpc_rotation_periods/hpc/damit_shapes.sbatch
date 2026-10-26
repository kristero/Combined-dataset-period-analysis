#!/bin/bash
#SBATCH --job-name=dshapes
#SBATCH --partition=power
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=hpc/damit_shapes.out
# Shape test of the convex inversion against the DAMIT models of the validation asteroids (damit_shape_compare.py)
cd $HOME/mpc_redo
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export DAMIT_BIN=$HOME/mpc_redo/damit/build_linux
ulimit -s unlimited
PY=/software/anaconda3/anaconda3-2023/bin/python3
echo "start $(date) on $(hostname)"
$PY -u damit_shape_compare.py 16
echo "end $(date)"
