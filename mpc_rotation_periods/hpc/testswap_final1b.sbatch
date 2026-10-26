#!/bin/bash
#SBATCH --job-name=ts_final1b
#SBATCH --partition=power
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --output=hpc/testswap_final1b.out
# Rest of testswap_final1 after the principal-axis result for the new models: (870) rejected (damit_rejected.json).
# Table of the convex models, review checks (refined solutions of the six only; minkowski attempts limited to 300 s),
# shape figure, (3216) inspections.
cd $HOME/mpc_redo
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg MINK_TIMEOUT=300
ulimit -s unlimited
PY=$HOME/micromamba/envs/py312/bin/python
echo "start $(date) on $(hostname)"
for s in "damit_paper_table.py" "testswap_checks.py" "damit_shapes_page.py --workers 8" "inspect_3216_short.py" \
         "inspect_3216_more.py --workers 8"; do
  echo "=== $s $(date)"
  $PY -u $s || echo "!!! $s failed with code $?"
done
echo "end $(date)"
