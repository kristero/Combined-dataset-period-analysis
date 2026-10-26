#!/bin/bash
#SBATCH --job-name=ts_geo
#SBATCH --partition=power
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --output=hpc/testswap_geo.out
# Six test asteroids added in the revision (five retired, retired.json): bootstrap and subset tests of the reviewed
# candidates, quality rule, aspect model for the paper and validation samples, corrected-data analysis, refined
# periods, final grades, DAMIT inputs of the six. Same sequence as zp_geo.sbatch, in the Python 3.12 environment.
set -e
cd $HOME/mpc_redo
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
PY=$HOME/micromamba/envs/py312/bin/python
echo "start $(date) on $(hostname)"
$PY -u rebootstrap2.py 249 486 870 1600 1700 7750 --workers 6
$PY -u quality_rule.py
$PY -u finalize.py > hpc/testswap_finalize0.log
$PY -u geometry_validate.py
$PY -u geometry_reanalysis.py all
$PY -u refine_periods.py
$PY -u finalize.py > hpc/testswap_finalize1.log
$PY -u quality_rule.py
$PY -u finalize.py > hpc/testswap_finalize2.log
$PY -u damit_prep.py --only paper --nums 249,486,870,1600,1700,7750
echo "end $(date)"
