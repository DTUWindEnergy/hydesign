#!/bin/bash

#SBATCH --job-name=risk_merch_audit
#SBATCH --output=risk_merch_audit_%j.out
#SBATCH --error=risk_merch_audit_%j.err
#SBATCH --partition=fatq,rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00

source /groups/INP/miniconda3/bin/activate
conda activate integrated-energy-grids

export PATH=/home/s242856/cplex_studio/cplex/bin/x86-64_linux:$PATH
export MPLBACKEND=Agg

python -u run_merchant_operational_audit.py
