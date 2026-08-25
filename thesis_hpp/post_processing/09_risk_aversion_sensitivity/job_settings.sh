#!/bin/bash

#SBATCH --job-name=risk_sensitivity
#SBATCH --output=risk_sensitivity_%j.out
#SBATCH --error=risk_sensitivity_%j.err
#SBATCH --partition=fatq,rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00

source /groups/INP/miniconda3/bin/activate
conda activate integrated-energy-grids

export PATH=/home/s242856/cplex_studio/cplex/bin/x86-64_linux:$PATH
export MPLBACKEND=Agg

python -u run_risk_aversion_sensitivity.py
