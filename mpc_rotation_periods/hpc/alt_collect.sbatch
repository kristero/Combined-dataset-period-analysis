#!/bin/bash
#SBATCH --job-name=alt_collect
#SBATCH --partition=power
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=hpc/alt_collect.out
# Rebuild review_alt_period.json from the finished runs (the parallel runs of 3716 and 4239 overwrote each other's entry).
cd $HOME/mpc_redo
export OMP_NUM_THREADS=1 MPLBACKEND=Agg MINK_TIMEOUT=300
PY=$HOME/micromamba/envs/py312/bin/python
rm -f review_alt_period.json
$PY -u review_alt_period.py 3716 13.4054 8.5939 --collect
$PY -u review_alt_period.py 4239 11.7753 15.6125 --collect
echo "end $(date)"
