#!/bin/bash
#SBATCH -J PMDI # A single job name for the array
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # All cores on one machine
#SBATCH -p sapphire,shared # Partition
#SBATCH --mem 16000 # Memory request
#SBATCH -t 3-00:00 # (D-HH:MM)
#SBATCH -o /n/home11/skbwu/PMDI/outputs/%j.out # Standard output
#SBATCH -e /n/home11/skbwu/PMDI/errors/%j.err # Standard error
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=skylerwu@college.harvard.edu
#SBATCH --account=murphy_lab

module load cmake/3.25.2-fasrc01
module load gcc/12.2.0-fasrc01
conda run -n afterburner python3 pmdi_main.py $1