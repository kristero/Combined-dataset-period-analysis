#!/bin/bash
#SBATCH --job-name=vb_blind
#SBATCH --partition=power
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --output=hpc/vb_blind.out
# Blind run of the automatic period pipeline on the 120 validation asteroids (tree vb/, vb_setup.py): period search,
# bootstrap of the candidates, algorithmic proposal, adopted = proposal (no review), bootstrap and subset tests of the
# final candidates, quality rule. Test of the checks suggested for slow rotators (vb_evaluate.py).
cd $HOME/mpc_redo/vb
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
PY=$HOME/micromamba/envs/py312/bin/python
echo "start $(date) on $(hostname)"
$PY -u mpc_redo.py --workers 30
$PY -u rebootstrap.py --workers 30
$PY -u adjudicate_pre.py > adjudicate_pre.log
$PY -u ../vb_autofinal.py vb
$PY -u rebootstrap2.py --workers 30
$PY -u quality_rule.py > quality_rule.log
echo "end $(date)"
