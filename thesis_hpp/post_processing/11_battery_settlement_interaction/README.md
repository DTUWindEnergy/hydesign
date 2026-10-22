# Battery and settlement-period interaction

This folder performs one targeted test: it examines whether the effect of the
battery changes when the PPA deviation settlement period becomes longer.

The PPA case is fixed at:

- Generic Industrial profile
- `lambda_PPA = 80 EUR/MWh`
- `c_ppa = 160 EUR/MWh`
- daily battery SoC cycling

The runner solves the No Battery and 10 MW/30 MWh configurations under daily,
weekly and monthly settlement. These are six new optimization runs. The
existing 20 MW/60 MWh results are not rerun.

## Run

From this folder on Sophia:

```bash
bash submit_job.sh
```

Do not run this job at the same time as another script that calls
`prepare_inputs`, because they share `inputs/Input_files_created`.

Completed cases are skipped when the job is submitted again. To intentionally
rerun one case, delete only that case's `summary.json`.

## Outputs

Each case is saved under:

```text
outputs/results/settlement_<period>/battery_p_<power>_e_<energy>/
```

Each case contains:

- `summary.json` and `summary.csv`: expected profit, objective value, optimal
  PPA capacity, daily contracted energy, shortfall, battery throughput and
  solver information.
- `profit_components.csv`: the expected annual revenue and cost components.

All six cases are collected in:

```text
outputs/battery_settlement_interaction_summary.csv
```

This compact workflow does not create the very large hourly stage-result CSVs.
