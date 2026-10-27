cd /mnt/c/Users/kn18001/Documents/Asteroids/MPC_redo
SSH="ssh -q -o BatchMode=yes"
rsync -a -e "$SSH" review_alt_period.py luhpc:mpc_redo/ 2>&1 | grep -v "micromamba/bin/micromamba: No such"
rsync -a -e "$SSH" hpc/alt_collect.sbatch luhpc:mpc_redo/hpc/ 2>&1 | grep -v "micromamba/bin/micromamba: No such"
ssh -q -o BatchMode=yes luhpc 'cd ~/mpc_redo && sed -i "s/\r$//" hpc/alt_collect.sbatch review_alt_period.py && J=$(sbatch --parsable hpc/alt_collect.sbatch) && echo "alt_collect=$J"; squeue -u $USER -o "%.10i %.12j %.8T %.10M"; echo; grep -c "P_top" hpc/vb_blind.out; tail -3 hpc/vb_blind.out' 2>&1 | grep -v "micromamba/bin/micromamba: No such"
