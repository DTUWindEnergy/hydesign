# Hourly PPA supply curves

This folder creates the hourly PPA supply curves used in the thesis. The
curves show the optimal contracted PPA capacity for the four investigated PPA
prices and three PPA deviation penalties.

The scripts use the completed 20 MW/60 MWh, hourly-settlement results for all
four shaped delivery profiles. They read the shared results catalog and do not
solve the optimization model.

## Run

From the project root:

```bash
python -u post_processing/02_hourly_supply_curves/plot_supply_curves_2x2.py
python -u post_processing/02_hourly_supply_curves/plot_supply_curves_individual.py
```

The first command creates one combined 2x2 figure and exports the exact plotted
values as CSV and LaTeX. The second command creates one figure for each shaped
delivery profile. All files are written below `outputs/`.

Regenerating the figures requires the four SDP result folders described in the
top-level `README.md`. The completed figures and tables can still be inspected
without those raw optimization results.

