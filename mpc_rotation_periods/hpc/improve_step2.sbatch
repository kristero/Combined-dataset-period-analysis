#!/bin/bash
#SBATCH --job-name=imp_step2
#SBATCH --partition=power
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --output=hpc/improve_step2.out
# Check 2 (principal-axis check as a period diagnostic): specificity on the validation models at their LCDB periods,
# and convex models at the competing periods of the paper asteroids whose models fail the check ((3716), (4239);
# (870) in review_870_alt.py).
cd $HOME/mpc_redo
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg MINK_TIMEOUT=300
ulimit -s unlimited
PY=$HOME/micromamba/envs/py312/bin/python
echo "start $(date) on $(hostname)"
$PY -u inertia_validation.py --workers 16 &
$PY -u review_alt_period.py 3716 13.4054 8.5939 &
$PY -u review_alt_period.py 4239 11.7753 15.6125 &
wait
echo "end $(date)"
