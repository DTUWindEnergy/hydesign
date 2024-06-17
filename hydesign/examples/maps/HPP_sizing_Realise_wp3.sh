#!/bin/bash
#SBATCH --job-name=hydesign
#SBATCH --output=output_hydesign_%J.log
#SBATCH --error=output_hydesign_%J.log

#SBATCH --partition=rome
# #SBATCH --partition=workq 
# #SBATCH --partition=windq 
# #SBATCH --partition=windfatq

#SBATCH --ntasks-per-core 1 
#SBATCH --ntasks-per-node 32 
#SBATCH --nodes=1
#SBATCH --exclusive 
#SBATCH --time=02:00:00

# job array:
#      run sites 0 to 546
#      maximum of 10 sites running in parallel
#SBATCH --array=0-220%25
# #SBATCH --array=551-1000%25
# #SBATCH --array=0-2201%25
# #SBATCH --array=0-1

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
source /home/jumu/miniconda3/bin/activate
conda activate hydesign
python HPP_sizing_Realise_wp3.py --ID_start 2000 --ID $SLURM_ARRAY_TASK_ID

# Example usage:
# --------------
# sbatch HPP_sizing_Realise_wp3.sh
