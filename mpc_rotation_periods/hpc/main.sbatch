#!/bin/bash
#SBATCH --job-name=damit_main
#SBATCH --partition=weakold
#SBATCH --time=10:00:00
#SBATCH --array=0-6
#SBATCH --output=hpc/main_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
cd $HOME/mpc_redo
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ulimit -s unlimited
NUMS=$(cat hpc/main_${SLURM_ARRAY_TASK_ID}.lst)
echo "start $(date) on $(hostname): $NUMS"
/software/anaconda3/anaconda3-2023/bin/python3 -u damit_run.py poles --workers 16 --nums $NUMS
/software/anaconda3/anaconda3-2023/bin/python3 -u damit_run.py loao --workers 16 --nums $NUMS
/software/anaconda3/anaconda3-2023/bin/python3 -u damit_ellipsoid.py --workers 16 --nums $NUMS
/software/anaconda3/anaconda3-2023/bin/python3 -u damit_eval.py --workers 16 --nums $NUMS --no-summary
echo "end $(date)"
