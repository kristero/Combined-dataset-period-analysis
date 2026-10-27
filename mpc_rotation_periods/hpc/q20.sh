cd /mnt/c/Users/kn18001/Documents/Asteroids/MPC_redo
ssh -q -o BatchMode=yes luhpc 'cd ~/mpc_redo && squeue -u $USER -o "%.10i %.12j %.8T %.10M %.6C" ; echo ---; tail -5 hpc/vb_blind.out 2>/dev/null; echo ---; tail -15 hpc/improve_step2.out 2>/dev/null; ls hpc/ | tail -20' 2>&1 | grep -v "micromamba/bin/micromamba: No such"
