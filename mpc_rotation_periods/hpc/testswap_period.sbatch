#!/bin/bash
#SBATCH --job-name=ts_period
#SBATCH --partition=power
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --output=hpc/testswap_period.out
# Period pipeline (zp tree: repo_snap classes, 703/G96 split data) for the six test asteroids added in the revision,
# with (1706) rerun as a reproducibility check of the HPC Python 3.12 environment against the stored local result.
# Run blind: published.json holds no period for the six.
cd $HOME/mpc_redo/zp
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
PY=$HOME/micromamba/envs/py312/bin/python
NUMS="1706 249 486 870 1600 1700 7750"
echo "start $(date) on $(hostname)"
$PY -u mpc_redo.py $NUMS --workers 7
$PY -u rebootstrap.py $NUMS --workers 7
echo "end $(date)"
