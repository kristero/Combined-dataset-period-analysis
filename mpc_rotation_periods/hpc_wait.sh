#!/bin/bash
# Wait (up to $2 minutes) until the given SLURM jobs have left the queue on the LU HPC, then pull the results.
# Usage (WSL myrootfs, user kn18001): bash hpc_wait.sh "111483,111491" 110
JOBS=$1
MAXMIN=${2:-110}
SSH="ssh -q -o BatchMode=yes"
t0=$(date +%s)
while true; do
  left=$($SSH luhpc "squeue -h -j $JOBS -o '%i %T' 2>/dev/null" | wc -l)
  el=$(( ($(date +%s) - t0) / 60 ))
  if [ "$left" -eq 0 ] || [ "$el" -ge "$MAXMIN" ]; then break; fi
  sleep 300
done
cd /mnt/c/Users/kn18001/Documents/Asteroids/MPC_redo && bash hpc_sync.sh pull
$SSH luhpc "cd ~/mpc_redo; squeue -u \$USER; echo poles \$(ls damit_runs/*/poles/poles.json | wc -l) loao \$(ls damit_runs/*/loao/loao.json 2>/dev/null | wc -l) ell \$(ls damit_runs/*/ellipsoid.json 2>/dev/null | wc -l) eval \$(ls damit_runs/*/eval.json 2>/dev/null | wc -l); tail -n 3 hpc/longls.out hpc/longscan.out hpc/summary.out 2>/dev/null"
echo "jobs left: $left after $el min"
