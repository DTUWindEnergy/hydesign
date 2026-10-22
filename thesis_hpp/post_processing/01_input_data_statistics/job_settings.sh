#!/bin/bash

#SBATCH --job-name=input_data_statistics
#SBATCH --output=input_data_statistics.out
#SBATCH --error=input_data_statistics.err

#SBATCH --partition=fatq,rome,workq
#SBATCH --ntasks-per-core 1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00

source /groups/INP/miniconda3/bin/activate
conda activate integrated-energy-grids

export MPLBACKEND=Agg

python -u "${filename}"
