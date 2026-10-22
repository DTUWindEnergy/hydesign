# Expected annual profit

This folder plots expected annual profit for the four shaped delivery profiles
under the hourly PPA deviation settlement period and the 20 MW/60 MWh battery.
It compares:

- PPA prices of 40, 60, 80 and 100 EUR/MWh;
- PPA deviation penalties of 120, 160 and 200 EUR/MWh;
- the four shaped delivery profiles.

Expected profit is the probability-weighted scenario profit and excludes the
CVaR term. The script reads the audited decomposition produced by
`04_profit_decomposition`; it does not solve the optimization model or repeat
the detailed decomposition calculation.

The required input file is:

`post_processing/04_profit_decomposition/outputs/expected_profit_decomposition_split_imbalance.csv`

If this file is missing, run `04_profit_decomposition` first.

The figure uses a centered 2+1 layout:

- `c_ppa = 120` and `160 EUR/MWh` in the first row;
- `c_ppa = 200 EUR/MWh` in the second row;
- four shaped delivery profiles on each horizontal axis;
- one grouped bar for each PPA price.

## Run

From the project root, run:

```bash
python -u post_processing/05_expected_profit/plot_expected_profit.py
```

Settings, including the shared color palette, are in `config.py`.
