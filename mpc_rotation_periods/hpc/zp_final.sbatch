#!/bin/bash
#SBATCH --job-name=zp_final
#SBATCH --partition=power
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --output=hpc/zp_final.out
# After the DAMIT and long-period jobs: DAMIT table, shape-model page, (3216) inspections
cd $HOME/mpc_redo
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
ulimit -s unlimited
PY=/software/anaconda3/anaconda3-2023/bin/python3
echo "start $(date) on $(hostname)"
for s in "damit_mirrors.py --only paper --workers 16" "damit_paper_table.py" "damit_shapes_page.py --workers 8" "inspect_3216_short.py" "inspect_3216_more.py --workers 8"; do
  echo "=== $s $(date)"
  $PY -u $s || echo "!!! $s failed with code $?"
done
echo "end $(date)"
