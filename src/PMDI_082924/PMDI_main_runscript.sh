#!/bin/bash
#SBATCH -J PMDI # A single job name for the array
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # All cores on one machine
#SBATCH -p sapphire,shared # Partition
#SBATCH --mem 16000 # Memory request
#SBATCH -t 3-00:00 # (D-HH:MM)
#SBATCH -o /n/holyscratch01/kou_lab/swu/PMDI_082924/outputs/%A_%a.out # Standard output
#SBATCH -e /n/holyscratch01/kou_lab/swu/PMDI_082924/errors/%A_%a.err # Standard error
#SBATCH --array=0-1727  # Size of the array
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=skylerwu@college.harvard.edu
#SBATCH --account=murphy_lab

conda run -n afterburner python3 PMDI_main.py ${SLURM_ARRAY_TASK_ID}