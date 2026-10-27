NUMS="$(python3 -c "import json; d=json.load(open('/mnt/c/Users/kn18001/Documents/Asteroids/MPC_redo/vb_evaluate.json')); print(' '.join(x['num'] for x in d['validation'] if x['dominant'] and x['dominant']['share']>0.5))")"
echo $NUMS
mkdir -p ~/pull/vbarr
for n in $NUMS; do echo "vb/results/$n/${n}_arrays.npz"; done > ~/pull/vbarr/list.txt
rsync -a -e "ssh -q -o BatchMode=yes" --files-from=$HOME/pull/vbarr/list.txt luhpc:mpc_redo/ ~/pull/vbarr/ 2>&1 | grep -v "micromamba/bin/micromamba: No such"
find ~/pull/vbarr -name "*_arrays.npz" | wc -l
