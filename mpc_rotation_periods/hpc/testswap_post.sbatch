#!/bin/bash
#SBATCH --job-name=ts_post
#SBATCH --partition=power
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --output=hpc/testswap_post.out
# After testswap_geo: amplitudes, injections, quantization checks, per-dataset table (Figure 7) and the derived
# statistics for the new sample (same steps as zp_post.sbatch; each step logged, a failing step does not stop the others)
cd $HOME/mpc_redo
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
PY=$HOME/micromamba/envs/py312/bin/python
mkdir -p review manuscript_revised
echo "start $(date) on $(hostname)"
for s in "diag_quantization.py" "check_no_quantized.py" "geometry_terms.py" "amp_eval.py" "amp_injection2.py" \
         "amp_report2.py" "dataset_power_table.py 32"; do
  echo "=== $s $(date)"
  $PY -u $s || echo "!!! $s failed with code $?"
done
echo "end $(date)"
