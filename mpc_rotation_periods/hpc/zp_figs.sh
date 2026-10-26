#!/bin/bash
# Light-curve figures and the shape-model page on the LU HPC (run on the login node through hpc_zp_figs in WSL)
set -e
cd $HOME/mpc_redo
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
PY=/software/anaconda3/anaconda3-2023/bin/python3
mkdir -p review manuscript_revised
$PY -u make_figures.py
$PY -u damit_shapes_page.py --workers 4
echo FIGS_DONE
