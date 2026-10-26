#!/bin/bash
#SBATCH --job-name=ts_final1
#SBATCH --partition=power
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --output=hpc/testswap_final1.out
# Revised test sample, final steps that do not depend on the long-period scans:
#  - aspect-model validation with the LCDB poles / dense amplitudes regenerated for the new sample (the fits are
#    deterministic and do not use these files; only the pole and epoch comparisons change), amplitude-phase terms
#  - amplitude evaluation (full run: per-asteroid LCDB fields of the new sample) and report
#  - DAMIT evaluation of the 58 paper runs (E1 of the refitted aspect model, epochs of the new dense LCDB amplitudes)
#    and the summary without the retired runs; table of the convex models; review checks (principal axis, harmonic
#    test, phase function, DAMIT membership); shape figure
#  - (3216) inspections on the re-corrected data
cd $HOME/mpc_redo
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
ulimit -s unlimited
PY=$HOME/micromamba/envs/py312/bin/python
mkdir -p review manuscript_revised
NUMS=$($PY -c "import json; print(','.join(sorted(json.load(open('adopted.json')), key=int)))")
echo "start $(date) on $(hostname)"
for s in "geometry_validate.py" "geometry_terms.py" "amp_eval.py" "amp_report2.py" \
         "damit_eval.py --workers 32 --nums $NUMS --no-summary" "damit_eval.py --summary" "damit_paper_table.py" \
         "testswap_checks.py" "damit_shapes_page.py --workers 8" "inspect_3216_short.py" "inspect_3216_more.py --workers 8"; do
  echo "=== $s $(date)"
  $PY -u $s || echo "!!! $s failed with code $?"
done
echo "end $(date)"
