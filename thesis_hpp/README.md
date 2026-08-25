# Thesis HPP optimization model

**Author:** Angelos Konstantinos Sfakianoudis  
**Student ID:** s242856  
**Institution:** DTU Wind and Energy Systems

This repository contains the optimization and post-processing code used in
the thesis. The model determines the PPA contracted capacity and the operation
of a wind-battery hybrid power plant participating in the day-ahead and mFRR
markets.

The complete optimization was run on the DTU Sophia cluster. The full model is
large and requires CPLEX. The post-processing scripts are lighter, but some of
them require the completed optimization results described below.

## Main folders

- `src/thesis_hpp/`: data loader and main Pyomo model.
- `inputs/`: input-preparation scripts and the CorRES wind profiles.
- `results/`: raw optimization outputs. This folder is not included in the
  GitHub upload because the detailed files are very large.
- `post_processing/`: scripts, processed CSV files, tables and thesis figures.
- `economic_value_analysis/`: merchant-equivalent and project break-even PPA
  price analysis.
- `pay_as_delivered_sensitivity/`: additional pay-as-delivered PPA test.
- `leave_one_out/`: additional scenario sensitivity scripts.

Each main post-processing folder contains its own README with the purpose,
required inputs and commands.

## Main Python files

- `main.py`: selects the investigated cases, builds and solves the optimization
  model, and saves the results.

- `inputs/prepare_inputs_from_btc_2000.py`: processes the BTC and CorRES data
  and creates the CSV files required by the optimization model.

- `inputs/sdp.py`: defines the four shaped PPA delivery profiles.

- `src/thesis_hpp/data_loader.py`: loads the prepared CSV files into the data
  structures used by Pyomo.

- `src/thesis_hpp/model.py`: contains the objective function, decision
  variables and constraints of the three-stage stochastic optimization model.

## Software requirements

The Python packages are listed in `environment.yml`. CPLEX must be installed
and licensed separately because it is not included in the Conda environment.

## Input data

The input-preparation script combines two data sources:

1. CorRES wind profiles stored below `inputs/CORRES/`.
2. Balancing Tool Chain outputs stored on Sophia at:

```text
/groups/INP/to_angelos/from_angelos/outputs
```

The BTC directory contains one folder per weather year, for example
`Model_results_1990/annual_hourly/`, with the day-ahead, balancing and
imbalance-price CSV files used by the model.

The BTC location can be changed without editing the code:

```bash
export BTC_OUTPUTS_DIR=/path/to/btc/outputs
```

`main.py` has a `PREPARE_INPUTS` setting. With `PREPARE_INPUTS = True`, the BTC
and CorRES data are processed before the optimization starts. With
`PREPARE_INPUTS = False`, the model reuses the CSV files already stored in
`inputs/Input_files_created/`.

## How to run `main.py` on Sophia

1. The `environment.yml` file is included in the `thesis_hpp` folder. From
   this folder, create the Conda environment once before submitting any job.
   Skip this step if `integrated-energy-grids` already exists:

   ```bash
   source /groups/INP/miniconda3/bin/activate
   conda env create -f environment.yml
   ```

2. `job_settings.sh` already activates the Conda environment. Replace the
   CPLEX installation path in this line with your own:

   ```bash
   export PATH=/path/to/your/cplex/bin/x86-64_linux:$PATH
   ```

   This is the path to the CPLEX executable, not the licence file. CPLEX must
   also have a valid licence.

   To use another solver, add its installation directory to `PATH` and change
   `SolverFactory("cplex")` in `main.py` to the corresponding solver name.

3. Select the cases near the top of `main.py` using
   `DELIVERY_PROFILES_TO_RUN`, `LAMBDA_PPA_VALUES`, `C_PPA_VALUES`,
   `SETTLEMENT_PERIOD`, `BATTERY_CASES` and `PREPARE_INPUTS`.

4. From the `thesis_hpp` project folder, submit the job:

   ```bash
   bash submit_job.sh
   ```

   `submit_job.sh` submits `job_settings.sh`, which activates the environment,
   adds CPLEX to `PATH` and runs `main.py`.

5. Each case is saved below:

   ```text
   results/settlement_<period>/<delivery_profile>/c_ppa_<value>/battery_p_<value>_e_<value>/lambda_ppa_<value>/
   ```

   If the solver does not return an optimal solution, the run stops without
   saving that case as a completed result.

## Running the four shaped delivery profiles

The same optimization model was used for all four shaped delivery profiles.
The model was run separately for each profile so that their result files did
not get mixed together. A separate copy of the project folder was used for
each profile:

| Delivery profile | Project folder |
| --- | --- |
| Generic Industrial Yearly | `thesis_hpp` |
| Generic Private Yearly | `thesis_hpp_Private` |
| Inverse Solar Yearly | `thesis_hpp_Inverse_Solar` |
| Yearly Continuous Baseload | `thesis_hpp_Baseload` |

In each folder, `DELIVERY_PROFILES_TO_RUN` in `main.py` contains only the
profile shown in the table. The remaining model code is the same.

To recreate figures that compare the four profiles, the folders `thesis_hpp`,
`thesis_hpp_Private`, `thesis_hpp_Inverse_Solar` and `thesis_hpp_Baseload`
must be in the same parent directory.

The raw result folders are not included in the GitHub package because they are
very large. The final processed CSV files, tables and figures are included, so
the reported results can still be inspected without the raw result folders.

## Investigated optimization cases

For the 20 MW/60 MWh battery, all four SDPs were solved for four PPA prices,
three deviation penalties and four deviation settlement periods. This gives
192 cases. The no-battery and 10 MW/30 MWh configurations were additionally
solved under hourly settlement, giving 96 more cases and 288 main optimization
cases in total.

The sensitivity analyses in `post_processing/`, `economic_value_analysis/`,
`pay_as_delivered_sensitivity/` and `leave_one_out/` contain additional runs
outside these 288 main cases.

## Post-processing

The processed figures, CSV files and LaTeX tables used in the thesis are
included. Each numbered folder contains its own `README.md` with the required
instructions.

## Files excluded from the GitHub upload

The following generated files are intentionally excluded:

- root `results/`, especially the large `stage3_results.csv` files;
- `inputs/Input_files_created/`, because these CSV files are generated by the
  input-preparation script;
- `pay_as_delivered_sensitivity/results/`;
- Python cache files;
- Slurm `.out` and `.err` files, except any example logs supplied separately;
- local Codex metadata.

The compact post-processing outputs are retained because they contain the
numerical values and figures reported in the thesis.
