#!/bin/bash -x
#SBATCH --job-name=JOBNAME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=JOBNAME_.%J.out
#SBATCH --error=S_hist_err.%J.out
#SBATCH --time=36:00:00
#SBATCH --mail-user=me@whatever
#SBATCH --mail-type=ALL
#SBATCH --mem=10000M

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export KMP_AFFINITY=balanced,granularity=fine,verbose

cd /hiskp4/rolon/python
srun python SCRIPTREPLACE.py 

cd -
