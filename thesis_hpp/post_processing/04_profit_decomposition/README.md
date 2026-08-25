# Expected annual profit decomposition

This folder performs three tasks using the existing optimization results:

1. It calculates the probability-weighted expected annual value of each
   revenue and cost component.
2. It plots the revenue decomposition by PPA price for each selected SDP and
   PPA deviation penalty.
3. It exports the complete numerical decomposition as CSV and LaTeX appendix
   tables.

The decomposition contains PPA revenue, day-ahead spot revenue, imbalance
settlement, imbalance penalty, mFRR-up revenue, mFRR-down revenue, PPA
shortfall cost and battery degradation cost. The scripts reconstruct and
audit expected profit, but they do not solve the optimization model.

## Cases included by default

- Hourly PPA deviation settlement
- 20 MW/60 MWh battery
- `c_ppa = 120, 160, 200 EUR/MWh`
- `lambda_PPA = 40, 60, 80, 100 EUR/MWh`
- Four shaped delivery profiles

Each PPA deviation penalty and delivery profile produces one full-width,
grouped bar figure. The horizontal axis contains the eight expected
annual revenue and cost components, with one bar for each PPA price. The
representative `c_ppa = 160 EUR/MWh` figure for the selected delivery profile
can be used in the main results section; the remaining figures can be placed
in the appendix.

The compressed 2x2 and previously generated stacked figures remain available
but are disabled by default. Their switches are in `config.py`.

## Run

From the project root, run:

```bash
python -u post_processing/04_profit_decomposition/plot_profit_decomposition.py
```

The calculation may take several minutes because it reads the detailed
stage-three output for every selected case. Subsequent runs reuse the audited
decomposition CSV, so visual changes do not require rereading those files.

## Appendix tables

After the audited decomposition CSV has been created, export the numerical
appendix tables with:

```bash
python -u post_processing/04_profit_decomposition/export_appendix_tables.py
```

This command does not reread the detailed optimization results. It uses
`outputs/expected_profit_decomposition_split_imbalance.csv`, verifies that the
eight signed components sum to expected profit, and writes:

- one combined CSV with all selected cases;
- one LaTeX `longtable` for each PPA deviation penalty; and
- one combined LaTeX file containing all tables.

The table files are saved under `outputs/appendix_tables/`. They report
revenues as positive values, costs as negative values, and monetary components
in kEUR/year. The LaTeX document must load `booktabs`, `longtable`, and
`pdflscape`.

## Other settlement periods

Change `SETTLEMENT_PERIODS` in `config.py`, for example:

```python
SETTLEMENT_PERIODS = ("weekly",)
```

The same calculation works for daily, weekly, and monthly settlement because
the optimized shortfall is read from each case's
`settlement_period_results.csv`.

## Outputs

- `outputs/expected_profit_decomposition_split_imbalance.csv`: expected
  components, expected profit, CVaR, reported objective, reconstructed
  objective, and audit difference for every selected case. Imbalance
  settlement and the imbalance penalty are stored separately.
- `outputs/grouped_components_individual/settlement_<period>/`
  `battery_p_<MW>_e_<MWh>/c_ppa_<value>/`: full-width PDF and PNG figures.

Bar labels use compact EUR values such as `65k` and `1.12M`. Very small values
below the configurable label threshold remain in the CSV but are not labelled
in the figure.
