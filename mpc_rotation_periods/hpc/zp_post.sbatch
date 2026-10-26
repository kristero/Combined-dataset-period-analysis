#!/bin/bash
#SBATCH --job-name=zp_post
#SBATCH --partition=power
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --output=hpc/zp_post.out
# After zp_geo: figures, amplitudes and the derived statistics quoted in the manuscript (each step logged; a failing
# step does not stop the others)
cd $HOME/mpc_redo
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
PY=/software/anaconda3/anaconda3-2023/bin/python3
mkdir -p review manuscript_revised
echo "start $(date) on $(hostname)"
for s in diag_quantization.py check_no_quantized.py make_figures.py search_single_dataset_example.py \
         search_tess_example.py alias_example.py geometry_terms.py amp_eval.py amp_injection2.py amp_report2.py \
         binary_1717.py; do
  echo "=== $s $(date)"
  $PY -u $s || echo "!!! $s failed with code $?"
done
echo "end $(date)"
