SSH="ssh -q -o BatchMode=yes"
mkdir -p ~/pull/vb
rsync -a -e "$SSH" --include="*/" --include="*_rebootstrap2.json" --include="*_rebootstrap.json" --exclude="results/**" \
  --include="adopted.json" --include="proposals.json" --include="quality_final.json" --include="*.log" --exclude="data/**" --exclude="*" \
  luhpc:mpc_redo/vb/ ~/pull/vb/ 2>&1 | grep -v "micromamba/bin/micromamba: No such"
rsync -a -e "$SSH" --include="*/" --include="*_rebootstrap2.json" --exclude="*" luhpc:mpc_redo/vb/results/ ~/pull/vb/results/ 2>&1 | grep -v "micromamba/bin/micromamba: No such"
ls ~/pull/vb; ls ~/pull/vb/results | wc -l; find ~/pull/vb/results -name "*_rebootstrap2.json" | wc -l
tail -3 ~/pull/vb/quality_rule.log
ssh -q -o BatchMode=yes luhpc 'squeue -u $USER -o "%.10i %.12j %.8T %.10M"' 2>&1 | grep -v "micromamba/bin/micromamba: No such"
