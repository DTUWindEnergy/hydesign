# Battery sensitivity

This folder compares the following three battery configurations under hourly
PPA deviation settlement:

- no battery;
- 10 MW/30 MWh;
- 20 MW/60 MWh.

It performs four post-processing tasks:

1. It compares optimal PPA capacity or daily contracted PPA energy across the
   complete PPA price and penalty grid.
2. It compares expected annual profit across the same cases.
3. It compares the revenue and cost decomposition for a selected PPA price,
   SDP and all three PPA deviation penalties.
4. It exports appendix tables for all four SDPs, all prices, all penalties and
   all three battery configurations.

The scripts read existing optimization results. They do not solve or modify
the optimization model.

## Outputs

- `plot_battery_contracts.py` creates the centered 2+1 contracting figure:
  - one panel for each `c_ppa`;
  - PPA price on the horizontal axis;
  - grouped bars for no battery, 10 MW/30 MWh, and 20 MW/60 MWh.
- `plot_battery_expected_profit.py` uses the same centered 2+1 layout to
  compare expected annual profit across the three battery configurations.
- `plot_battery_revenue_decomposition.py` creates one revenue and cost
  decomposition per configured PPA penalty for one SDP and PPA price.
- `export_appendix_tables.py` exports all four SDPs, all prices, all penalties,
  and all three battery configurations to CSV and four compact appendix-ready
  contracting tables plus four expected-profit tables, one per SDP.

The settings are in `config.py`. Change `CONTRACT_FIGURE_METRIC` to
`"capacity"` if the main contracting figure should show `S_PPA` instead of
daily contracted energy.

## Run

From the project root, run:

```bash
python -u post_processing/06_battery_sensitivity/plot_battery_contracts.py
python -u post_processing/06_battery_sensitivity/plot_battery_expected_profit.py
python -u post_processing/06_battery_sensitivity/plot_battery_revenue_decomposition.py
python -u post_processing/06_battery_sensitivity/export_appendix_tables.py
```

The expected-profit script reads all 36 detailed Generic Industrial cases on
its first run. It caches the audited results in
`outputs/battery_expected_profit_cases.csv`.

The appendix exporter requests expected profit for all four SDPs. It reuses
the existing decomposition caches and calculates only missing cases, so its
first complete run can take noticeably longer than the contracting-table
export.

The decomposition script reads nine detailed `stage3_results.csv` files the
first time it runs (three penalties by three battery configurations). It saves
the audited values in
`outputs/battery_revenue_decomposition.csv`, so later plotting runs reuse the
cache.
