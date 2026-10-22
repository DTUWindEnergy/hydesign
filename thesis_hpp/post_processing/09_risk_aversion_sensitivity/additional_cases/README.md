# Additional risk-aversion cases

This folder extends the benchmark risk-aversion analysis with three additional
cases. Every case is examined for `beta = 0, 0.1, 0.2, 0.5, 0.8, 1.0`. The
final value maximizes CVaR alone and is used as an extreme risk-aversion stress
test:

1. High-contract PPA: lambda_PPA=100, c_ppa=120, with the PPA capacity
   optimized.
2. Merchant operation: S_PPA=0, so the HPP participates only in the spot,
   imbalance and mFRR markets.
3. High-penalty PPA: lambda_PPA=80, c_ppa=200, with the PPA capacity
   optimized.

All three cases use the Generic Industrial profile, hourly PPA deviation
settlement and the 20 MW/60 MWh battery.

This folder performs two tasks:

1. `run_additional_cases.py` solves only missing case/beta combinations and
   saves compact summaries and annual scenario profits.
2. `plot_additional_cases.py` creates the two main risk-aversion figures using
   `beta = 0, 0.1, 0.2, 0.5, 0.8`. The first is a 2x2 expected-profit/CVaR
   comparison for merchant operation, the low-price case, the benchmark case
   and the high-contract case. The second shows the percentage change in
   optimal PPA capacity for the low-price, benchmark, high-contract and
   high-penalty PPA cases. The `beta=1` stress test is not shown in these main
   figures.

It also contains a separate merchant operational audit. This audit re-solves
only the merchant case for `beta=0` and `beta=0.8` and checks whether the
day-ahead and real-time operating decisions change even though the expected
profit changes very little.

## Files

- settings.py: set the beta values and define the cases examined.
- run_additional_cases.py: run the missing optimization cases and collect
  their compact summaries.
- plot_additional_cases.py: create the selected expected-profit/CVaR and PPA
  capacity comparison figures.
- job_settings.sh: Slurm resources for the sequential optimization job.
- submit_job.sh: submit the job.
- run_merchant_operational_audit.py: re-solve merchant operation for beta=0
  and beta=0.8 and export compact operating results.
- job_settings_merchant_audit.sh: Slurm resources for the merchant audit.
- submit_merchant_audit.sh: submit only the merchant audit.

## Reused results

The script reuses all completed results through `beta=0.8` and skips every
case/beta summary already present in `outputs/results/`. After this update it
therefore solves only the missing `beta=1.0` high-contract, high-penalty and
merchant cases.

## Run

From this folder on Sophia:

~~~bash
bash submit_job.sh
~~~

Each completed solve is immediately saved under:

~~~text
outputs/results/<case_id>/beta_*/summary.json
~~~

New solves also receive an 80-row scenario_profits.csv. If the Slurm job is
interrupted, submit it again; completed summaries are skipped.

When all cases are complete, the combined table is written to:

~~~text
outputs/additional_risk_aversion_summary.csv
~~~

Then create the two comparison figures:

~~~bash
python -u plot_additional_cases.py
~~~

The figures are saved under `outputs/figures/` as PNG and PDF files:

~~~text
risk_aversion_expected_profit_cvar_all_cases
risk_aversion_ppa_capacity_change_all_cases
~~~

The low-price `beta=0.2` point is reused from the main hourly results, so no
additional solve is needed for it. The merchant case is not included in the
capacity-change figure because its PPA capacity is fixed at zero. CVaR is the
probability-weighted mean profit of the lowest 10% of the 80 annual scenarios,
corresponding to eight equally probable scenarios here.

Do not run this workflow at the same time as another script that calls
prepare_inputs, because they share inputs/Input_files_created.

## Merchant operational audit

Run this separately after other optimization jobs that prepare inputs have
finished:

~~~bash
bash submit_merchant_audit.sh
~~~

The audit does not export the large stage-two and stage-three result files. It
exports the following compact files under
`outputs/merchant_operational_audit/`:

1. One annual expected operating summary for each beta.
2. An 80-row table of annual operating totals by scenario for each beta.
3. An 8,736-row expected hourly operating profile for each beta.
4. Direct beta=0 versus beta=0.8 comparison tables at all three levels.

The operating variables include day-ahead energy, real-time grid delivery,
battery charging and discharging, mFRR-up and mFRR-down activation, positive
and negative imbalance, curtailment and battery state of charge. Completed
beta audits are skipped if all three of their output files already exist.

## Low-price PPA test

Laura's thesis found the strongest response to risk aversion at low PPA
prices. The separate low-price test examines the lowest-price case in this
thesis that still contracts a non-negligible PPA volume:

- `lambda_PPA = 40 EUR/MWh`
- `c_ppa = 120 EUR/MWh`
- `beta = 0, 0.1, 0.5, 0.8`

Run only this test with:

~~~bash
bash submit_low_price.sh
~~~

This submits four optimization solves. It does not run the other additional
cases or any `beta=1` case. Results are written to:

~~~text
outputs/low_price_case/
~~~

Each beta directory contains a compact summary. A fresh solve also writes the
80 annual scenario profits. In the submitted archive, detailed low-price
scenario files are retained below `outputs/results/low_price/` only where they
were exported successfully; empty duplicate placeholders are not included.
The four summaries are combined in `low_price_beta_sensitivity.csv`.
Completed beta values are skipped if the job is submitted again.
