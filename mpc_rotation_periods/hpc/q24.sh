cd /mnt/c/Users/kn18001/Documents/Asteroids/MPC_redo
SSH="ssh -q -o BatchMode=yes"
rsync -a -e "$SSH" review_alt_period.py inertia_sensitivity.py luhpc:mpc_redo/ 2>&1 | grep -v "micromamba/bin/micromamba: No such"
rsync -a -e "$SSH" hpc/inertia_sens.sbatch luhpc:mpc_redo/hpc/ 2>&1 | grep -v "micromamba/bin/micromamba: No such"
ssh -q -o BatchMode=yes luhpc 'cd ~/mpc_redo && ls validation/sample.json damit_eval.py && ls -d damit_runs/val_* | wc -l && sed -i "s/\r$//" hpc/inertia_sens.sbatch review_alt_period.py inertia_sensitivity.py && J=$(sbatch --parsable --dependency=afterok:136439 hpc/inertia_sens.sbatch) && echo "inert_sens=$J"; squeue -u $USER -o "%.10i %.12j %.8T %.10M %.20R"; grep -c "subsets" hpc/vb_blind.out' 2>&1 | grep -v "micromamba/bin/micromamba: No such"
