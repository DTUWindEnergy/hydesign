#!/bin/bash
set -euo pipefail

filename="${1:-run_indifference_validations.py}"
case "${filename}" in
    run_indifference_validations.py|run_indifference_validations_c_ppa_120.py)
        ;;
    *)
        echo "Usage: bash submit_job.sh [run_indifference_validations.py|run_indifference_validations_c_ppa_120.py]"
        exit 1
        ;;
esac

sbatch --export=ALL,filename="${filename}" job_settings.sh
