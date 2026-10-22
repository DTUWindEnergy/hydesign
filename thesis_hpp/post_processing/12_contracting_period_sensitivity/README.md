# Contracting-period sensitivity

This folder compares how the benchmark result changes when the HPP can select
one PPA capacity for the full year, four seasonal capacities, or 12 monthly
capacities.

The investigated case is:

- Generic Industrial shaped delivery profile
- PPA price: 80 EUR/MWh
- PPA deviation penalty: 160 EUR/MWh
- Battery: 20 MW/60 MWh
- Hourly PPA deviation settlement
- Daily battery SoC cycling

All capacity decisions remain first-stage decisions and are optimized jointly in
one annual stochastic model. The seasonal and monthly cases are not solved as
independent subproblems. This keeps the annual mean-CVaR objective and all 80
annual scenarios intact.

The seasonal definition is:

- Winter: January, February and December
- Spring: March, April and May
- Summer: June, July and August
- Autumn: September, October and November

## Run

From this folder, submit:

```bash
bash submit_job.sh
```

The job reuses the completed annual benchmark, solves only the seasonal and
monthly cases, and then creates the comparison figure. Existing completed cases
are skipped. Set `REUSE_EXISTING_ANNUAL_RESULT = False` only if you deliberately
want to solve the annual case again.

## Outputs

`outputs/contracting_period_summary.csv` compares expected profit, CVaR,
objective value, annual contracted energy and expected shortfall rate. It also
reports each seasonal and monthly result as a change from the annual case.

`outputs/contracting_period_capacities.csv` contains every optimized annual,
seasonal or monthly PPA capacity.

`outputs/contracting_period_sensitivity.png` compares the capacity schedules
and economic results.

Each individual solve is also stored below `outputs/results/`.
