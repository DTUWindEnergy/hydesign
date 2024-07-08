#!/bin/bash
#SBATCH --job-name=hydesign_simple
#SBATCH --output=output_hydesign_simple_%J.log
#SBATCH --error=output_hydesign_simple_%J.log

# #SBATCH --partition=rome
# #SBATCH --partition=workq 
#SBATCH --partition=windq 
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

# Set environment before running this script or comment in below lines to do it in the process
# source /home/jumu/miniconda3/bin/activate
# conda activate hydesign
ml git
python check_setup.py
python ../EGO_surrogate_based_optimization_simple_hpp.py \
    --example 1 \
    --opt_var "NPV_over_CAPEX"\
    --rotor_diameter_m 100\
    --hub_height_m 120\
    --wt_rated_power_MW 2\
    --surface_tilt_deg 20\
    --surface_azimuth_deg 180\
    --DC_AC_ratio 1\
    --num_batteries 1\
    --n_procs  31\
    --n_doe 31\
    --n_clusters 16\
    --n_seed 0\
    --max_iter 10\
    --final_design_fn 'hydesign_simple_design_0.csv'
    
# Example usage:
# --------------
# sbatch hydesign_sizing_simple_hpp.sh
