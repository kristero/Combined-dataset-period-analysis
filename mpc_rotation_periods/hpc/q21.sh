SSH="ssh -q -o BatchMode=yes"
mkdir -p ~/pull/step2
rsync -a -e "$SSH" luhpc:mpc_redo/inertia_validation.json luhpc:mpc_redo/review_alt_period.json luhpc:mpc_redo/hpc/improve_step2.out ~/pull/step2/ 2>&1 | grep -v "micromamba/bin/micromamba: No such"
ls -la ~/pull/step2
