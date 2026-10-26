#!/bin/bash
#SBATCH --job-name=ts_post2
#SBATCH --partition=power
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --output=hpc/testswap_post2.out
# Second part of the post-processing for the revised test sample (the steps of zp_post.sbatch that testswap_post.sbatch
# did not run, and the amplitude steps that failed there): figures and figure_info.json (amplitudes of the tables),
# example figures, injection test (the (2150) template now folded at its period from adopted_retired.json),
# forward-model correction of the amplitudes, amplitude summary and report.
cd $HOME/mpc_redo
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
PY=$HOME/micromamba/envs/py312/bin/python
mkdir -p review manuscript_revised
echo "start $(date) on $(hostname)"
for s in "make_figures.py" "search_single_dataset_example.py" "search_tess_example.py" "alias_example.py" \
         "amp_injection2.py" "amp_debias.py" "amp_eval.py --eval-only" "amp_report2.py"; do
  echo "=== $s $(date)"
  $PY -u $s || echo "!!! $s failed with code $?"
done
echo "end $(date)"
