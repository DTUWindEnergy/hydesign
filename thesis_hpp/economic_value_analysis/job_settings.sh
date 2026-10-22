#!/bin/bash

#SBATCH --job-name=ppa_value
#SBATCH --output=ppa_value.out
#SBATCH --error=ppa_value.err

#SBATCH --partition=fatq,rome
#SBATCH --ntasks-per-core 1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --time=2-00:00:00

source /groups/INP/miniconda3/bin/activate
conda activate integrated-energy-grids

export PATH=/home/s242856/cplex_studio/cplex/bin/x86-64_linux:$PATH

python -u ${filename}
