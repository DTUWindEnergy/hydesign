# Contracted PPA delivery outputs

This folder performs four post-processing tasks using the existing
optimization results:

1. It calculates the hourly contracted PPA delivery
   `P_PPA[t] = SDP[t] * S_PPA` for every case.
2. It plots the diurnal contracted-delivery profiles of the four SDPs.
3. It calculates and plots daily contracted PPA energy, including supply
   curves with daily energy on the horizontal axis.
4. It exports the corresponding numerical and LaTeX tables.

The scripts read the shared results catalog and the profile definitions in
`inputs/sdp.py`. They do not solve the optimization model.

## Cases included

- four shaped delivery profiles;
- PPA prices of 40, 60, 80 and 100 EUR/MWh;
- PPA deviation penalties of 120, 160 and 200 EUR/MWh;
- hourly settlement for all three battery configurations;
- daily, weekly and monthly settlement for the 20 MW/60 MWh battery.

## Commands

Run these commands from this folder:

- `python build_datasets.py`: rebuild only the CSV datasets.
- `python plot_diurnal_profiles.py`: rebuild only the diurnal figures.
- `python plot_daily_energy.py`: rebuild only the daily-energy figures.
- `python plot_daily_energy_supply_curves.py`: rebuild only the daily-energy supply curves.
- `python create_tables.py`: rebuild only the LaTeX tables.
- `python run_all.py`: rebuild every output.

Plot palette selection is in `config.py`.

## Output structure

- `data/contracted_ppa_delivery_cases.csv`: one row per optimization case.
- `data/contracted_ppa_delivery_hourly.csv`: 24 rows per case with profile factors and hourly contracted delivery.
- `data/delivery_profile_metrics.csv`: profile full-load hours and average-to-peak factors.
- `diurnal_profiles/`: one four-profile figure per settlement, battery, PPA price, and deviation penalty.
- `daily_energy/`: one three-panel figure per settlement and battery configuration.
- `daily_energy_supply_curves/`: one 2x2 supply-curve figure per settlement and battery configuration.
- `tables/`: individual appendix tables and one combined LaTeX file.

## Generated counts

- optimization cases: 288
- hourly dataset rows: 6912
- diurnal figure files (PDF and PNG): 144
- daily-energy figure files (PDF and PNG): 12
- LaTeX table files: 8
