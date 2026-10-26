#!/bin/bash
# Re-analysis after the 703/G96 zero-point split on the LU HPC (~/mpc_redo).
# Run in WSL (myrootfs) as kn18001:  bash hpc_zp.sh clean|push|submit|status|pull|install
set -e
SRC=/mnt/c/Users/kn18001/Documents/Asteroids/MPC_redo
DST=luhpc:mpc_redo
SSH="ssh -q -o BatchMode=yes"
CHANGED=$(tr ',' ' ' < "$SRC/hpc/zp_changed.lst")
case "$1" in
clean)   # move the remote outputs that the split makes stale out of the way (kept, not deleted)
  $SSH luhpc "cd ~/mpc_redo && mkdir -p unsplit_hpc/damit_runs && \
    for n in $CHANGED; do if [ -d damit_runs/paper_\$n ] && [ ! -e unsplit_hpc/damit_runs/paper_\$n ]; then mv damit_runs/paper_\$n unsplit_hpc/damit_runs/; fi; done; \
    for d in results results_geo data/obs_tables; do if [ -d \$d ] && [ ! -e unsplit_hpc/\$d ]; then mkdir -p unsplit_hpc/\$(dirname \$d); mv \$d unsplit_hpc/\$d; fi; done; \
    ls unsplit_hpc unsplit_hpc/damit_runs | head -50"
  ;;
push)
  cd "$SRC"
  rsync -a -e "$SSH" --prune-empty-dirs \
    --include='*/' \
    --include='/*.py' --include='/*.sh' --include='/*.json' --include='/hpc/***' \
    --include='/results/*/*_arrays.npz' --include='/results/*/*_result.json' \
    --include='/results/*/*_rebootstrap.json' --include='/results/*/*_rebootstrap2.json' \
    --include='/validation/sample.json' --include='/validation/results/*_arrays.npz' --include='/validation/results/*_info.json' \
    --include='/validation/data/obs_tables/***' --include='/validation/data/horizons/***' \
    --include='/data/obs_tables/***' --include='/data/horizons/***' --include='/data/*.json' \
    --include='/shapes_page_cache/***' \
    --exclude='*' ./ "$DST/"
  $SSH luhpc "cd ~/mpc_redo && mkdir -p review manuscript_revised && ls results | wc -l && ls data/obs_tables | wc -l"
  ;;
submit)
  $SSH luhpc 'cd ~/mpc_redo && \
    G=$(sbatch --parsable hpc/zp_geo.sbatch) && \
    D=$(sbatch --parsable --dependency=afterok:$G hpc/zp_main.sbatch) && \
    L=$(sbatch --parsable --dependency=afterok:$G hpc/zp_long.sbatch) && \
    F=$(sbatch --parsable --dependency=afterok:$G hpc/zp_post.sbatch) && \
    Z=$(sbatch --parsable --dependency=afterany:$D:$L:$F hpc/zp_final.sbatch) && \
    echo "geo=$G main=$D long=$L post=$F final=$Z" | tee hpc/zp_jobs.txt'
  ;;
status)
  $SSH luhpc 'cd ~/mpc_redo && cat hpc/zp_jobs.txt; squeue -u nagainis -o "%.10i %.12j %.8T %.10M %.6D %R"; \
    for f in hpc/zp_*.out; do echo "--- $f"; tail -n 4 $f; done 2>/dev/null'
  ;;
pull)
  mkdir -p /tmp/mpc_zp_pull
  rsync -a -e "$SSH" --include='*/' \
    --include='/results_geo/***' --include='/adopted.json' --include='/quality_final.json' --include='/period_refine.json' \
    --include='/geometry_*.json' --include='/figure_info.json' --include='/no_quantized_check.json' \
    --include='/single_dataset_candidates.json' --include='/tess_example_candidates.json' --include='/amp_eval_*.json' \
    --include='/amp_injection2*.json' --include='/amp_report2.json' --include='/binary_1717*.json' \
    --include='/inspect_3216_*.json' --include='/longperiod_*.json' --include='/damit_paper_table.json' \
    --include='/damit_shapes_page_info.json' --include='/data/dataset_precision.json' \
    --include='/manuscript_revised/fig_*' --include='/manuscript_revised/table_damit_rows.tex' \
    --include='/review/***' --include='/shapes_page_cache/***' --include='/hpc/*.out' --include='/hpc/*.log' \
    --include='/hpc/zp_jobs.txt' \
    --include='/damit_runs/paper_*/prep.json' --include='/damit_runs/paper_*/prep.npz' --include='/damit_runs/paper_*/lcs' \
    --include='/damit_runs/paper_*/eval.json' --include='/damit_runs/paper_*/ellipsoid.json' \
    --include='/damit_runs/paper_*/scan/***' --include='/damit_runs/paper_*/poles/***' --include='/damit_runs/paper_*/loao/***' \
    --include='/damit_runs/paper_*/longscan/***' --include='/damit_runs/paper_*/longrange/***' \
    --exclude='*' "$DST/" /tmp/mpc_zp_pull/
  du -sh /tmp/mpc_zp_pull; echo "pulled to /tmp/mpc_zp_pull (install as root: bash hpc_zp.sh install)"
  ;;
install)
  cp -r /tmp/mpc_zp_pull/. "$SRC/"
  ;;
esac
