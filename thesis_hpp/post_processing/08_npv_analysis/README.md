# NPV analysis

This folder performs four financial post-processing tasks:

1. It calculates 30-year project NPV and IRR for all four SDPs, four PPA
   deviation settlement periods, four PPA prices and three PPA deviation
   penalties. This gives 192 cases for the 20 MW/60 MWh battery.
2. It creates NPV figures for the Generic Industrial profile, including one
   figure for each settlement period.
3. It exports NPV and IRR appendix tables for all four SDPs.
4. It creates a heatmap comparing all 12 price and penalty combinations across
   the four PPA deviation settlement periods, together with a comparison of
   the number of cases with positive NPV.

The calculation reads audited expected-profit results from
`07_settlement_period_effect`. It does not solve the optimization model.

## Run

From the project root, first rebuild the audited settlement-period dataset so
that it contains all four profiles:

```bash
python -u post_processing/07_settlement_period_effect/settlement_analysis.py
```

Then run the complete NPV workflow:

```bash
python -u post_processing/08_npv_analysis/run_all.py
```

The calculation uses the probability-weighted expected scenario profit, not the
mean-CVaR objective. Battery degradation is added back because the full battery
CAPEX is included in the investment cost. Wind fixed and variable O&M and
battery fixed O&M are then subtracted to obtain annual project cash flow.
The results CSV reports both the common break-even cash profit excluding the
battery degradation charge and the case-specific break-even value expressed
in the optimization model's reported-profit convention, which includes that
charge. It also reports the project IRR and whether it exceeds the 5% real
discount rate.

The base assumptions are in `settings.py`:

- 30-year project lifetime;
- 5% real discount rate;
- constant 2045 annual cash flow;
- complete wind and battery CAPEX paid at year zero;
- no battery replacement and no salvage value;
- wind variable O&M applied to probability-weighted gross available wind;
- battery variable O&M equal to zero.

## Outputs

- `outputs/npv_results.csv`: calculation and assumptions for all 192 cases.
- `outputs/appendix_tables/`: CSV plus one combined LaTeX NPV/IRR table for
  each of the four profiles.
- `outputs/positive_npv_case_counts.csv`: positive-NPV count by settlement period.
- `outputs/npv_by_settlement_period/`: one 2+1 NPV figure per period.
- `outputs/npv_12_bars_by_settlement_period/`: one equally spaced 12-bar NPV
  figure per period, with four PPA prices and three PPA penalties.
- `outputs/npv_heatmap_by_settlement_period/`: one heatmap comparing the four
  settlement periods.
- `outputs/positive_case_count/`: compact comparison of feasible-case counts.
