cd /mnt/c/Users/kn18001/Documents/Asteroids/MPC_redo
SSH="ssh -q -o BatchMode=yes"
rsync -a -e "$SSH" fine_checks.py synsid_check.json luhpc:mpc_redo/ 2>&1 | grep -v "micromamba/bin/micromamba: No such"
rsync -a -e "$SSH" hpc/fine_checks.sbatch luhpc:mpc_redo/hpc/ 2>&1 | grep -v "micromamba/bin/micromamba: No such"
ssh -q -o BatchMode=yes luhpc 'cd ~/mpc_redo && sed -i "s/\r$//" hpc/fine_checks.sbatch fine_checks.py && ls vb/results/553/553_arrays.npz results/4618/4618_arrays.npz results_geo/4761/4761_arrays.npz damit_runs/paper_3716_alt8p5939/scan/out_periods damit_runs/val_822/scan/out_periods && $HOME/micromamba/envs/py312/bin/python -c "import sys; sys.path.insert(0,\".\"); import fine_checks as F; print(len(F.fibonacci_poles(60)), F.synsid_rows()[\"paper:4618\"])" && J=$(sbatch --parsable hpc/fine_checks.sbatch) && echo "fine_chk=$J"; squeue -u $USER -o "%.10i %.12j %.8T %.10M"' 2>&1 | grep -v "micromamba/bin/micromamba: No such"
