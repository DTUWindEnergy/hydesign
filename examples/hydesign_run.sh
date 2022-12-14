#!/bin/bash
#SBATCH --job-name=hydesign
#SBATCH --output=output_hydesign_%J.log
#SBATCH --error=output_hydesign_%J.log

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

# Set environment
source /home/jumu/miniconda3/bin/activate
conda activate hydesign
# python /home/jumu/hydesign/hydesign/EGO_surrogate_based_optimization.py --case "India" --site "Indian_site_bad_solar_bad_wind" --opt_var "NPV_over_CAPEX"

# Examples of other runs
# ----------------------
python /home/jumu/hydesign/hydesign/EGO_surrogate_based_optimization.py --case "India" --site "Indian_site_bad_solar_bad_wind" --opt_var "LCOE [Euro/MWh]"

# python /home/jumu/hydesign/hydesign/EGO_surrogate_based_optimization.py --case "Europe" --site "Denmark_good_solar" --opt_var "NPV_over_CAPEX"
# python /home/jumu/hydesign/hydesign/EGO_surrogate_based_optimization.py --case "Europe" --site "Denmark_good_solar" --opt_var "LCOE [Euro/MWh]"


# Example usage:
# --------------
# sbatch hydesign_run.sh
