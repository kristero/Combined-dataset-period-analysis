#!/bin/bash
#SBATCH --job-name=ds_table
#SBATCH --partition=power
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=hpc/ds_table.out
# Per-dataset values behind the dataset-power figure (dataset_power_table.py)
cd $HOME/mpc_redo
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
PY=/software/anaconda3/anaconda3-2023/bin/python3
echo "start $(date) on $(hostname)"
$PY -u dataset_power_table.py 16
echo "end $(date)"
