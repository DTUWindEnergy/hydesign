# Input data statistics

This folder performs two tasks:

1. It plots the four shaped delivery profiles defined in `inputs/sdp.py`.
2. It calculates summary statistics and creates figures for the wind and
   electricity-price input data.

The scripts do not solve the optimization model.

## Run

From the project root, create the delivery-profile figure with:

```bash
python -u post_processing/01_input_data_statistics/plot_delivery_profiles.py
```

The input-data statistics require the prepared files in
`inputs/Input_files_created/`. On Sophia, run them with:

```bash
cd post_processing/01_input_data_statistics
bash submit_job.sh
```

The figures, CSV files and LaTeX tables are saved under
`post_processing/graphs/`.
