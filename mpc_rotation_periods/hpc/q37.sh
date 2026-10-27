cd /mnt/c/Users/kn18001/Documents/Asteroids/MPC_redo
rsync -a -e "ssh -q -o BatchMode=yes" lcdb_doubt_check.py luhpc:mpc_redo/ 2>&1 | grep -v "micromamba/bin/micromamba: No such"
rsync -a -e "ssh -q -o BatchMode=yes" hpc/lcdb_doubt.sbatch luhpc:mpc_redo/hpc/ 2>&1 | grep -v "micromamba/bin/micromamba: No such"
ssh -q -o BatchMode=yes luhpc 'cd ~/mpc_redo && sed -i "s/\r$//" hpc/lcdb_doubt.sbatch lcdb_doubt_check.py && python3 -c "import json; print(json.load(open(\"damit_runs/val_2564/prep.json\"))[\"P0\"], json.load(open(\"damit_runs/val_2993/prep.json\"))[\"P0\"])" && J=$(sbatch --parsable hpc/lcdb_doubt.sbatch) && echo "lcdb_doubt=$J"; squeue -u $USER -o "%.10i %.12j %.8T %.10M"' 2>&1 | grep -v "micromamba/bin/micromamba: No such"
