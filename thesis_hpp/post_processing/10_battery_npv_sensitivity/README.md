# Battery NPV sensitivity

This folder performs two analyses.

## 1. NPV by battery configuration

This analysis calculates the NPV for the following three battery
configurations under the main daily SoC cycling assumption:

- no battery;
- 10 MW/30 MWh;
- 20 MW/60 MWh.

For each battery configuration, NPV is calculated for the 12 combinations of
PPA price and PPA deviation penalty:

- lambda_PPA = 40, 60, 80 and 100 EUR/MWh;
- c_PPA = 120, 160 and 200 EUR/MWh.

All cases use the Generic Industrial delivery profile, hourly PPA deviation
settlement and daily SoC cycling. This analysis uses existing optimization
results and does not solve the model again.

## 2. Sensitivity to the SoC cycling period

This analysis examines how the SoC cycling assumption affects battery NPV. It
compares daily SoC cycling with three-day SoC cycling for one benchmark case:

- Generic Industrial delivery profile;
- hourly PPA deviation settlement;
- lambda_PPA = 80 EUR/MWh;
- c_PPA = 160 EUR/MWh;
- 10 MW/30 MWh and 20 MW/60 MWh batteries.

Under three-day cycling, the battery SoC is fixed to E0 at the start of each
72-hour block instead of at the start of every day. The no-battery result is
also included as a reference, but the cycling assumption does not apply to it.

The hourly SoC trajectory remains continuous. Because the 8,736-hour horizon
contains 364 days, it consists of 121 complete 72-hour blocks and one final
24-hour block. The existing year-end condition returns the final block to E0.

The main optimization model is not edited. The daily cycling constraint is
deactivated only in the in-memory model used for the three-day sensitivity.
Both required three-day optimization runs are already complete.

## Files

- `config.py`: set the benchmark case, battery configurations, cycling horizon
  and solver settings.
- `run_soc_cycling_sensitivity.py`: solve the new 72-hour case and export
  compact results. This file deactivates the daily cycling constraint only on
  the newly built in-memory model. It does not edit `src/thesis_hpp/model.py`.
- `compare_results.py`: calculate NPV for all 12 hourly price/penalty cases
  under the three existing daily-cycled battery configurations, then compare
  the new 72-hour result with the three daily-cycling benchmark results.
- `plot_results.py`: create the full daily-cycling battery NPV figure and the
  benchmark NPV comparison between daily and three-day SoC cycling.
- `job_settings.sh`: Sophia resources for any missing three-day optimization
  run.
- `submit_job.sh`: submit missing three-day cases if they need to be recreated.

## Run order

The required three-day optimization results are already complete. To recreate
the tables and both figures, run from this folder on Sophia:

```bash
python -u compare_results.py
python -u plot_results.py
```

Use `bash submit_job.sh` only if one of the three-day case summaries is
missing. Completed summaries are detected and skipped automatically.

## Outputs

The optimization creates:

- `outputs/three_day_run/battery_p_10_e_30/three_day_case_summary.csv`
- `outputs/three_day_run/battery_p_10_e_30/profit_components.csv`
- `outputs/three_day_run/battery_p_10_e_30/soc_cycle_audit.csv`
- the corresponding files for `battery_p_20_e_60` on a fresh run; the already
  completed 20 MW/60 MWh result remains in `outputs/three_day_run`

The post-processing creates:

- `outputs/soc_cycling_npv_comparison.csv`
- `outputs/daily_cycling_battery_npv_results.csv`
- `outputs/battery_npv_by_soc_cycling_period.png`
- `outputs/daily_cycling_battery_npv_sensitivity.png`

The NPV calculation uses the assumptions from
`post_processing/08_npv_analysis/settings.py`. Battery degradation is added
back to expected profit before NPV is calculated because the complete battery
investment cost is included in CAPEX.
