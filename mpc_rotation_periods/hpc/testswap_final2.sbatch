#!/bin/bash
#SBATCH --job-name=ts_final2
#SBATCH --partition=power
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --output=hpc/testswap_final2.out
# Revised test sample, long-period steps: fine Lomb-Scargle for (7750) (P > 24 h, missing from testswap_damit) and for
# the aspect-corrected asteroids whose corrected data changed with the refit ((3216), (3777), (4048)); DAMIT window
# scans of the new windows and summary; reliability tests of the long periods with (870) added; convex model of
# (870) at its competing period 122.16 h with the principal-axis check (review_870_alt.py).
cd $HOME/mpc_redo
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg MINK_TIMEOUT=300
ulimit -s unlimited
PY=$HOME/micromamba/envs/py312/bin/python
echo "start $(date) on $(hostname)"
for s in "review_870_alt.py" "ls_longperiod.py --workers 4 --nums 7750,3216,3777,4048" "damit_longscan.py --workers 32" \
         "damit_longscan.py --summary" "longperiod_tests.py --workers 10"; do
  echo "=== $s $(date)"
  $PY -u $s || echo "!!! $s failed with code $?"
done
echo "end $(date)"
