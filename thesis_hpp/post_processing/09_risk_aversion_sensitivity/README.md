# Risk-aversion sensitivity

This folder examines how the risk-aversion weight `beta` affects one benchmark
optimization case. It performs three tasks:

1. It solves the benchmark case for `beta = 0, 0.1, 0.2, 0.5, 0.8, 1.0`.
2. It compares optimal PPA capacity, expected profit, CVaR and the mean-CVaR
   objective across the six beta values.
3. It reconstructs the 80 annual scenario profits for the existing
   `beta = 0.2` result and identifies the scenarios in the lower 10% CVaR tail.

## Benchmark case

- Generic Industrial profile
- hourly PPA deviation settlement
- `lambda_PPA = 80 EUR/MWh`
- `c_ppa = 160 EUR/MWh`
- `20 MW / 60 MWh` battery
- CVaR confidence level `alpha = 0.9`
- `beta = 0, 0.1, 0.2, 0.5, 0.8, 1.0`

The completed results through `beta=0.8` are reused. The solver therefore runs
only the missing `beta=1.0` benchmark case. The workflow saves compact JSON and CSV summaries only;
it does not create the large stage-result CSV files.

## Run the optimization

From this folder on Sophia:

```bash
bash submit_job.sh
```

Do not run this job at the same time as another script that calls
`prepare_inputs`, because both use the shared `inputs/Input_files_created`
directory.

The combined results are written to:

```text
outputs/risk_aversion_summary.csv
```

Each solve is also saved immediately under:

```text
outputs/results/beta_*/summary.json
```

If the job stops after one or two cases, submit it again. Completed beta values
are skipped and the workflow continues from the first missing result. Delete a
beta's `summary.json` only when you intentionally want to rerun that case.

## Create the figure

After the Slurm job finishes:

```bash
python -u plot_risk_aversion_sensitivity.py
```

## Inspect the existing beta = 0.2 CVaR tail

The detailed baseline result files are sufficient to reconstruct its 80 annual
scenario profits. This is post-processing only and does not solve the model:

```bash
python -u audit_baseline_tail.py
```

It exports all 80 scenario profits, the strict-tail and VaR-boundary scenarios,
and a weather-year profit summary below `outputs/baseline_tail_audit/`. Boundary
ties are assigned the fractional probability needed to complete the 10% tail.

The PNG and PDF figures are written to `outputs/figures/`.

## Additional comparison cases

The high-contract, high-penalty and fixed-merchant beta sweeps are kept
separately in `additional_cases/`. See its README for the cases and run order.
