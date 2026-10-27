mkdir -p ~/pull/sens
rsync -a -e "ssh -q -o BatchMode=yes" luhpc:mpc_redo/inertia_sensitivity.json luhpc:mpc_redo/hpc/inertia_sens.out ~/pull/sens/ 2>&1 | grep -v "micromamba/bin/micromamba: No such"
tail -2 ~/pull/sens/inertia_sens.out
