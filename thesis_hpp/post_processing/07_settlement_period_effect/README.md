# Effect of the PPA deviation settlement period

This folder examines how changing the PPA deviation settlement period affects
the optimization results. It compares hourly, daily, weekly and monthly
settlement for the 20 MW/60 MWh battery. The figures use the profile selected
in `settings.py`, while the appendix tables include all four SDPs.

For all four PPA prices and all three PPA deviation penalties, it calculates
and plots:

1. optimal contracted PPA capacity;
2. optimal daily contracted PPA energy;
3. expected annual profit;
4. expected PPA shortfall as a percentage of contracted annual energy.

It also exports appendix tables for every SDP, PPA price, PPA deviation
penalty and settlement period. Each table reports expected annual profit,
optimal contracted PPA capacity and the expected shortfall rate.

The scripts read and audit existing optimization results. They do not solve
the optimization model.

## Scripts

- `plot_optimal_ppa_capacity.py` plots the optimal first-stage PPA capacity
  for all PPA prices and penalties.
- `plot_optimal_daily_energy.py` plots the equivalent optimal contracted
  daily PPA energy using the selected delivery profile.
- `plot_expected_profit.py` plots expected annual profit for the same cases.
- `plot_normalized_shortfall.py` plots expected shortfall as a percentage of
  contracted annual energy for the profile selected in `settings.py`.
- `export_appendix_tables.py` exports one LaTeX table per SDP and CSV files
  containing all settlement-period contracting and shortfall results.

The expected-profit and shortfall calculations are audited against the
reported mean-CVaR objective. Audited values are cached in
`outputs/settlement_period_analysis.csv`. Existing hourly decomposition data
are reused, so the first detailed run calculates only missing non-hourly
cases.

## Run

From the project root, run:

```bash
python -u post_processing/07_settlement_period_effect/plot_optimal_ppa_capacity.py
python -u post_processing/07_settlement_period_effect/plot_optimal_daily_energy.py
python -u post_processing/07_settlement_period_effect/plot_expected_profit.py
python -u post_processing/07_settlement_period_effect/plot_normalized_shortfall.py
python -u post_processing/07_settlement_period_effect/export_appendix_tables.py
```

Settings, including the benchmark case and color palette, are in
`settings.py`.
