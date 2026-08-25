#!/bin/bash

#SBATCH --job-name=soc_cycle_72h
#SBATCH --output=soc_cycle_72h_%j.out
#SBATCH --error=soc_cycle_72h_%j.err

#SBATCH --partition=fatq,rome
#SBATCH --ntasks-per-core=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00

set -euo pipefail

source /groups/INP/miniconda3/bin/activate
conda activate integrated-energy-grids

export PATH=/home/s242856/cplex_studio/cplex/bin/x86-64_linux:$PATH
export MPLBACKEND=Agg

python -u ${filename}
