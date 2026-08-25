#!/bin/bash

#SBATCH --job-name=thesis_hpp
#SBATCH --output=thesis_hpp_%j.out
#SBATCH --error=thesis_hpp_%j.err

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
