cd /mnt/c/Users/kn18001/Documents/Asteroids/MPC_redo
rsync -a -e "ssh -q -o BatchMode=yes" review_alt_period.py inertia_sensitivity.py luhpc:mpc_redo/ 2>&1 | grep -v "micromamba/bin/micromamba: No such"
ssh -q -o BatchMode=yes luhpc 'cd ~/mpc_redo && sed -i "s/\r$//" review_alt_period.py inertia_sensitivity.py && grep -c wide review_alt_period.py && squeue -u $USER -o "%.10i %.12j %.8T %.10M"' 2>&1 | grep -v "micromamba/bin/micromamba: No such"
