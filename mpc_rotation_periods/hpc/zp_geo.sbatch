#!/bin/bash
#SBATCH --job-name=zp_geo
#SBATCH --partition=power
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --output=hpc/zp_geo.out
# After the 703/G96 zero-point split: aspect model (paper + validation), period analysis of the aspect-corrected data,
# refined positions of the adopted periods, quality rule, adopted.json; DAMIT inputs of the changed paper asteroids
set -e
cd $HOME/mpc_redo
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/software/anaconda3/anaconda3-2023/bin/python3
echo "start $(date) on $(hostname)"
$PY -u geometry_validate.py
$PY -u geometry_reanalysis.py all
$PY -u refine_periods.py
$PY -u finalize.py > hpc/zp_finalize1.log
$PY -u quality_rule.py
$PY -u finalize.py
$PY -u damit_prep.py --only paper --nums $(cat hpc/zp_changed.lst)
echo "end $(date)"
