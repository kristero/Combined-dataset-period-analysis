#!/bin/bash
# Mirror the code and the data the DAMIT and long-period analyses need to the LU HPC (~/mpc_redo), or pull results back.
# Run in WSL (myrootfs) as kn18001:  bash hpc_sync.sh push | pull
set -e
SRC=/mnt/c/Users/kn18001/Documents/Asteroids/MPC_redo
DST=luhpc:mpc_redo
SSH="ssh -q -o BatchMode=yes"
cd "$SRC"
if [ "$1" = "push" ]; then
  rsync -a -e "$SSH" --prune-empty-dirs \
    --include='*/' \
    --include='/*.py' --include='/*.sh' --include='/*.json' --include='/hpc/***' \
    --include='/results/*/*_arrays.npz' --include='/results_geo/*/*_arrays.npz' \
    --include='/validation/sample.json' --include='/validation/results/*_arrays.npz' --include='/validation/results/*_info.json' \
    --include='/validation/data/obs_tables/***' --include='/validation/data/horizons/***' \
    --include='/data/obs_tables/***' --include='/data/horizons/***' --include='/data/*.json' \
    --include='/damit/version_0.2.1/***' --include='/damit/build/*.c' --include='/damit/build/*.h' \
    --include='/damit/build/Makefile' --include='/damit/build/*.f' \
    --include='/damit_runs/*/lcs' --include='/damit_runs/*/prep.json' --include='/damit_runs/*/prep.npz' \
    --include='/damit_runs/*/scan/***' --include='/damit_runs/*/poles/***' --include='/damit_runs/prep_summary.json' \
    --exclude='*' ./ "$DST/"
elif [ "$1" = "pull" ]; then
  rsync -a -e "$SSH" --include='*/' \
    --include='/damit_runs/*/poles/***' --include='/damit_runs/*/loao/***' --include='/damit_runs/*/ellipsoid.json' \
    --include='/damit_runs/*/eval.json' --include='/damit_runs/*/longscan/***' --include='/damit_runs/log_*' \
    --include='/hpc/*.out' --include='/damit_comparison.json' --include='/longperiod_*.json' --include='/review/longperiod_*' \
    --exclude='*' "$DST/" /tmp/mpc_pull/
  echo "pulled to /tmp/mpc_pull (copy into the Windows folder as root: bash hpc_sync.sh install)"
elif [ "$1" = "install" ]; then
  cp -r /tmp/mpc_pull/. "$SRC/"
fi
