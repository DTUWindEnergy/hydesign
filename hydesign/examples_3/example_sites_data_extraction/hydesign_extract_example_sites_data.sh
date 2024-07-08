#!/bin/bash
#SBATCH --job-name=hydesign_simple
#SBATCH --output=output_hydesign_simple_%J.log
#SBATCH --error=output_hydesign_simple_%J.log

#SBATCH --partition=rome
# #SBATCH --partition=workq 
# #SBATCH --partition=windq 
# #SBATCH --partition=windfatq

#SBATCH --ntasks-per-core 1 
#SBATCH --ntasks-per-node 32 
#SBATCH --nodes=1
#SBATCH --exclusive 
#SBATCH --time=02:00:00

#NODE_ID=$(head -1 $SLURM_JOB_NODELIST)
NODE_ID=$(scontrol show hostnames $SLURM_JOB_NODELIST)
#date=$(date '+%Y%m%d')
NAME="${filename%.*}"

export LC_ALL=en_US.UTF-8

echo -----------------------------------------------------------------
echo Date: $(date)
echo hydesign is running example_run_hpp_sizing_single_site.py
echo Sophia job is running on node: ${NODE_ID}
echo Sophia job identifier: $SLURM_JOBID
echo -----------------------------------------------------------------

# Set environment
source ~/miniconda3/bin/activate
conda activate hydesign

# to extract data for a specific example site, requires an integer index 
# python hydesign_extract_example_sites_data.py --example 0 
    
# to extract data for all example sites
python hydesign_extract_example_sites_data.py

# Example usage:
# --------------
# sbatch hydesign_extract_example_sites_data.sh
