set -e
SRC=/mnt/c/Users/kn18001/Documents/Asteroids/MPC_redo/overleaf_revision/revised
B=~/build_rev
cp $SRC/mpc_rotation_periods.tex $SRC/mpc_rotation_periods.bib $B/
cp $SRC/*.pdf $SRC/*.png $B/ 2>/dev/null || true
cd $B
pdflatex -interaction=nonstopmode mpc_rotation_periods.tex > build1.log || true
bibtex mpc_rotation_periods > bib.log || true
pdflatex -interaction=nonstopmode mpc_rotation_periods.tex > build2.log || true
pdflatex -interaction=nonstopmode mpc_rotation_periods.tex > build3.log || true
grep -E "^!|Undefined|undefined|Overfull \\\\hbox \([0-9]{2,}" mpc_rotation_periods.log | head -20
grep -E "Output written" mpc_rotation_periods.log
grep -ci warning bib.log || true
