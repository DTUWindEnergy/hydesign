# Risk-Adjusted PPA Economic-Value Analysis

This folder calculates the risk-adjusted PPA indifference price for fixed PPA
capacities. It reuses the finalized optimization model and writes only below
`economic_value_analysis/results/`; the ordinary optimization results are not
modified.

The benchmark configuration is Generic Industrial Yearly, hourly PPA deviation
settlement and a `20 MW / 60 MWh` battery. Four optimal capacities from the
completed ordinary runs are examined, corresponding to reference PPA prices of
40, 60, 80 and 100 EUR/MWh. The direct optimization validation uses
`c_ppa=160 EUR/MWh`, while the final price-range figures can include several
PPA deviation penalties.

## What to run now

Open `economic_value_analysis/plot_settings.py` and set the PPA deviation
penalties that you want to examine:

```python
C_PPA_VALUES = (120.0, 160.0, 200.0)
```

This file also contains the selected delivery profile, settlement period and
battery configuration. In most cases, only `C_PPA_VALUES` needs to be changed.

Then run this single command from the `thesis_hpp` project root:

```bash
python -u economic_value_analysis/plot_ppa_price_range.py
```

This command reads the completed optimization, merchant, validation and NPV
results. It does not solve the optimization model. It creates one separate
price-range figure for every selected `c_ppa` value:

```text
economic_value_analysis/results/figures/ppa_price_range_curve_c_ppa_120.png
economic_value_analysis/results/figures/ppa_price_range_curve_c_ppa_160.png
economic_value_analysis/results/figures/ppa_price_range_curve_c_ppa_200.png
```

Each figure contains the reference-price curve, the risk-adjusted PPA
indifference curve and the `NPV = 0` curve. The expected-profit indifference
curve is not plotted. Values of the `NPV = 0` curve that are above the displayed
price range are written at the top of the graph without increasing the y-axis.

## Mathematical idea

For reference case \(k\), let \(S_k^{PPA,*}\) be the optimal PPA capacity
selected in the ordinary run at reference price \(\lambda_k^{PPA}\). Its
contracted annual energy is

$$
E_k^{PPA}
=
S_k^{PPA,*}\sum_{t\in T}d_t^{PPA}\Delta t.
$$

Let \(J^{merchant}\) denote the optimized mean-CVaR objective when
\(S^{PPA}=0\). Let \(J^{obligation}(S_k^{PPA,*})\) denote the optimized
objective when \(S^{PPA}=S_k^{PPA,*}\) is fixed and the PPA price is set to
zero. Because contracted energy is deterministic after the capacity is fixed,
PPA revenue adds the same constant to every scenario:

$$
J(S_k^{PPA,*},\lambda^{PPA})
=
J^{obligation}(S_k^{PPA,*})
+
\lambda^{PPA}E_k^{PPA}.
$$

The risk-adjusted indifference price is the price at which accepting the fixed
PPA obligation gives the same objective value as merchant operation:

$$
J(S_k^{PPA,*},\lambda_k^{PPA,ind})=J^{merchant}.
$$

It can be calculated directly as

$$
\lambda_k^{PPA,ind}
=
\frac{J^{merchant}-J^{obligation}(S_k^{PPA,*})}{E_k^{PPA}}.
$$

The completed ordinary reference run satisfies

$$
J_k
=
J^{obligation}(S_k^{PPA,*})
+
\lambda_k^{PPA}E_k^{PPA}.
$$

Substitution gives the equivalent shortcut used for the four-point curve:

$$
\lambda_k^{PPA,ind}
=
\lambda_k^{PPA}
+
\frac{J^{merchant}-J_k}{E_k^{PPA}}.
$$

The original single-point analysis verifies that the direct and shortcut
formulas give the same result. The four-point validation then fixes both `S_k`
and `lambda_k_ind` and solves the model again. A successful validation satisfies

$$
J(S_k^{PPA,*},\lambda_k^{PPA,ind})-J^{merchant}\approx0.
$$

Here `J` is the same risk-adjusted objective used by the optimization model:

$$
J=(1-\beta)\mathbb{E}[\Phi]+\beta\,\mathrm{CVaR}_{\alpha}.
$$

Expected profit and CVaR are reported separately as diagnostics. Their equality
with the merchant case is not required; only their weighted combination must
match `J_merchant`.

## Files

- `plot_settings.py`: set the `c_ppa` values that you want to examine in the
  figures. You can also change the delivery profile, settlement period, battery
  case and axis limits here.
- `plot_ppa_price_range.py`: run this file to create the three price curves:
  the ordinary optimal curve, the PPA indifference-price curve and the
  `NPV = 0` curve. It reads existing results and does not solve the model.
- `analysis_config.json`: contains the benchmark case and the four reference
  PPA prices, optimal capacities and objective values used by the validation.
- `run_merchant.py`: solves the case with `S_PPA=0` and saves the merchant
  objective, expected profit and CVaR.
- `calculate_indifference_curve.py`: calculates the four indifference prices
  from the shortcut equation. It does not solve the model.
- `run_indifference_validations.py`: fixes each of the four capacities at its
  calculated indifference price and solves all four cases. Each result is saved
  immediately.
- `collect_indifference_validations.py`: recreates the combined validation CSV
  from the completed individual results. Use it if the validation job stopped
  before completing all four cases.
- `common.py`: contains functions shared by the merchant and fixed-capacity
  runs.
- `run_fixed_obligation.py`: solves the original single test case with
  `S_PPA=5.9722824943 MW` and `lambda_PPA=0`.
- `calculate_indifference_price.py`: calculates the indifference price for the
  original single test case using both equations.
- `run_indifference_validation.py`: validates only the original single test
  case at approximately 60.8136 EUR/MWh.
- `run_below_indifference_validation.py`: fixes the capacity from the
  80 EUR/MWh reference run at `S_PPA=5.9722824943 MW`, lowers the PPA price
  to 55 EUR/MWh and reoptimizes all second- and third-stage decisions. It
  checks that the resulting objective is below merchant operation and matches
  the analytical value implied by the indifference curve.
- `settlement_period_comparison/`: calculates and plots the indifference-price
  curves for hourly, daily, weekly and monthly deviation settlement. It also
  contains one optional direct validation. Read its `README.md` for the exact
  commands.

## One-time calculation and validation workflow

The steps below have already been completed. Repeat them only if the merchant
case, model formulation or four `c_ppa=160` validation points change.

Run all commands from the `thesis_hpp` project root.

### 1. Solve the merchant case

```bash
sbatch --export=ALL,filename='economic_value_analysis/run_merchant.py' job_settings.sh
```

Wait for this job to finish. It overwrites only
`economic_value_analysis/results/merchant/summary.json` and adds the expected
profit and CVaR components required by the new validation summaries.

### 2. Calculate the four indifference prices

This is a lightweight calculation and does not require Slurm:

```bash
python -u economic_value_analysis/calculate_indifference_curve.py
```

It creates:

```text
economic_value_analysis/results/indifference_curve_summary.csv
economic_value_analysis/results/indifference_curve_summary.json
```

### 3. Validate all four points

One job solves all four points sequentially:

```bash
sbatch --export=ALL,filename='economic_value_analysis/run_indifference_validations.py' job_settings.sh
```

Each completed point is saved immediately below:

```text
economic_value_analysis/results/indifference_validations/reference_lambda_*/summary.json
```

After all four solves finish, the script automatically creates:

```text
economic_value_analysis/results/indifference_validations_summary.csv
```

The combined CSV has four rows. It reports the fixed capacity, calculated
indifference price, objective difference from merchant operation, expected
profit difference, CVaR difference and expected-profit indifference price.

### Validate the four `c_ppa=120` points

The `c_ppa=120` validation is stored separately and does not overwrite the
completed `c_ppa=160` results. From the `economic_value_analysis` folder, run:

```bash
bash submit_job.sh run_indifference_validations_c_ppa_120.py
```

The script calculates the four indifference prices and then solves the four
fixed-capacity validation cases. Outputs are written below:

```text
results/c_ppa_120/indifference_curve_summary.csv
results/c_ppa_120/indifference_validations/reference_lambda_*/summary.json
results/c_ppa_120/indifference_validations_summary.csv
```

If the job stops after only some cases, rebuild a partial combined CSV with:

```bash
python -u economic_value_analysis/collect_indifference_validations.py
```

### 4. Create the figures

Follow the instructions under **What to run now** at the top of this file.

### Optional validation below the merchant-equivalent curve

This single additional solve confirms the interpretation of the shaded price
range. It fixes `S_PPA=5.9722824943 MW`, sets `lambda_PPA=55 EUR/MWh`, and
leaves every second- and third-stage decision free to reoptimize. The relevant
merchant-equivalent price is approximately 60.8136 EUR/MWh.

From the `economic_value_analysis` folder, run:

```bash
bash submit_below_indifference.sh
```

The result is written separately and does not overwrite the four existing
indifference validations:

```text
results/below_indifference_validation/
reference_lambda_80_test_lambda_55/summary.json
```

The script reports the observed objective, the analytical prediction, both
differences from merchant operation, and the numerical prediction error. It
fails explicitly if the optimized objective is not below the merchant case or
if the analytical and solved objectives differ by more than 1 EUR.
