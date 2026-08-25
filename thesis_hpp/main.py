from pathlib import Path
import sys
import time
import pyomo.environ as pyo
import pandas as pd
import json
import re

from src.thesis_hpp.data_loader import load_all_data
from src.thesis_hpp.model import build_model

DELIVERY_PROFILES_TO_RUN = [
    "Generic Industrial Yearly",
    #"Generic Private Yearly",
    #"Inverse Solar Yearly",
    #"Peak Yearly",
    #"Yearly Continuous Baseload",
]

LAMBDA_PPA_VALUES = [40, 60, 80, 100]
C_PPA_VALUES = [120, 160, 200]
WEATHER_YEARS = [
    1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999,
    2000, 2002, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2013,
]
XI_WIND_FILES = [
    "P_forecast1.csv",
    "P_forecast2.csv",
    "P_forecast3.csv",
    "P_forecast4.csv",
]
STAGE2_WIND_METHOD = "mean"
# Set to False to reuse the CSV files already stored in
# inputs/Input_files_created without reading the BTC outputs again.
PREPARE_INPUTS = True
# Choose one of: "hourly", "daily", "weekly", "monthly", or "yearly".
SETTLEMENT_PERIOD = "daily"
BATTERY_CASES = [
    {"P_b_max": 20.0, "E_max": 60.0, "E0": 30.0},
    #{"P_b_max": 10.0, "E_max": 30.0, "E0": 15.0},
    #{"P_b_max": 0.0, "E_max": 0.0, "E0": 0.0},
]



def slugify(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def battery_case_slug(battery_case):
    return f"battery_p_{battery_case['P_b_max']:g}_e_{battery_case['E_max']:g}"


def value_slug(prefix, value):
    return f"{prefix}_{value:g}"


def main():
    total_start = time.perf_counter()
    base_path = Path(__file__).resolve().parent
    inputs_path = base_path / "inputs"
    generated_inputs_path = inputs_path / "Input_files_created"
    results_root = base_path / "results"
    settlement_results_root = (
        results_root / f"settlement_{slugify(SETTLEMENT_PERIOD)}"
    )
    run_summary_rows = []

    # Make inputs folder importable
    sys.path.insert(0, str(inputs_path))

    # Run input preparation first
    from prepare_inputs_from_btc_2000 import prepare_inputs

    solver = pyo.SolverFactory("cplex")
    #solver = pyo.SolverFactory("appsi_highs")
    #solver = pyo.SolverFactory("gurobi")

    for delivery_profile in DELIVERY_PROFILES_TO_RUN:
        if PREPARE_INPUTS:
            prepare_inputs(
                delivery_profile_name=delivery_profile,
                weather_years=WEATHER_YEARS,
                xi_wind_files=XI_WIND_FILES,
                stage2_wind_method=STAGE2_WIND_METHOD,
            )

        for c_ppa in C_PPA_VALUES:
            for battery_case in BATTERY_CASES:
                for lambda_ppa in LAMBDA_PPA_VALUES:
                    run_start = time.perf_counter()
                    data = load_all_data(generated_inputs_path)
                    data["config"]["lambda_PPA"] = lambda_ppa
                    data["config"]["c_ppa"] = c_ppa
                    data["config"]["P_b_max"] = battery_case["P_b_max"]
                    data["config"]["E_max"] = battery_case["E_max"]
                    data["config"]["E0"] = battery_case["E0"]
                    data["config"]["settlement_period"] = SETTLEMENT_PERIOD


                    model = build_model(data)
                    results = solver.solve(model, tee=True)
                    run_elapsed_seconds = time.perf_counter() - run_start

                    if not pyo.check_optimal_termination(results):
                        raise RuntimeError(
                            "Solver did not return an optimal solution for "
                            f"profile={delivery_profile}, lambda_PPA={lambda_ppa}, "
                            f"c_ppa={c_ppa}, battery={battery_case}, "
                            f"settlement={SETTLEMENT_PERIOD}. "
                            f"Status={results.solver.status}, "
                            f"termination={results.solver.termination_condition}."
                        )

                    print("\n=== RUN ===")
                    print("Delivery profile:", delivery_profile)
                    print("lambda_PPA:", lambda_ppa)
                    print("c_ppa:", c_ppa)
                    print("P_b_max:", battery_case["P_b_max"])
                    print("E_max:", battery_case["E_max"])
                    print("Settlement period:", SETTLEMENT_PERIOD)
                    print("Solver status:", results.solver.status)
                    print("Termination:", results.solver.termination_condition)
                    print("Objective:", pyo.value(model.OBJ))
                    print("S_PPA:", pyo.value(model.S_PPA))
                    print(f"Run elapsed seconds: {run_elapsed_seconds:.2f}")
                    print(f"Run elapsed minutes: {run_elapsed_seconds / 60:.2f}")

                    results_dir = (
                        settlement_results_root
                        / slugify(delivery_profile)
                        / value_slug("c_ppa", c_ppa)
                        / battery_case_slug(battery_case)
                        / value_slug("lambda_ppa", lambda_ppa)
                    )

                    save_results(
                        model,
                        results_dir,
                        delivery_profile=delivery_profile,
                        lambda_ppa=lambda_ppa,
                        c_ppa=c_ppa,
                        battery_case=battery_case,
                        weather_years=WEATHER_YEARS,
                        xi_wind_files=XI_WIND_FILES,
                        stage2_wind_method=STAGE2_WIND_METHOD,
                        settlement_period=SETTLEMENT_PERIOD,
                        run_elapsed_seconds=run_elapsed_seconds,
                    )

                    run_summary_rows.append(
                        {
                            "delivery_profile": delivery_profile,
                            "lambda_PPA": lambda_ppa,
                            "c_ppa": c_ppa,
                            "P_b_max": battery_case["P_b_max"],
                            "E_max": battery_case["E_max"],
                            "settlement_period": SETTLEMENT_PERIOD,
                            "objective_value": pyo.value(model.OBJ),
                            "S_PPA": pyo.value(model.S_PPA),
                            "solver_status": str(results.solver.status),
                            "termination_condition": str(results.solver.termination_condition),
                            "elapsed_seconds": run_elapsed_seconds,
                        }
                    )

    total_elapsed_seconds = time.perf_counter() - total_start
    results_root.mkdir(parents=True, exist_ok=True)
    settlement_results_root.mkdir(parents=True, exist_ok=True)
    run_summary = pd.DataFrame(run_summary_rows)
    run_summary.to_csv(settlement_results_root / "run_summary.csv", index=False)
    # Keep the latest run at the old location for existing plotting scripts.
    run_summary.to_csv(results_root / "run_summary.csv", index=False)

    print("\n=== TOTAL JOB TIME ===")
    print(f"Total elapsed seconds: {total_elapsed_seconds:.2f}")
    print(f"Total elapsed minutes: {total_elapsed_seconds / 60:.2f}")
    print(f"Total elapsed hours: {total_elapsed_seconds / 3600:.2f}")



def save_results(
    model,
    results_dir: Path,
    delivery_profile,
    lambda_ppa,
    c_ppa,
    battery_case,
    weather_years,
    xi_wind_files,
    stage2_wind_method,
    settlement_period,
    run_elapsed_seconds,
):
    results_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame([{
        "delivery_profile": delivery_profile,
        "lambda_PPA": lambda_ppa,
        "c_ppa": c_ppa,
        "P_b_max": battery_case["P_b_max"],
        "E_max": battery_case["E_max"],
        "settlement_period": settlement_period,
        "objective_value": pyo.value(model.OBJ),
        "S_PPA": pyo.value(model.S_PPA),
        "elapsed_seconds": run_elapsed_seconds,
        "elapsed_minutes": run_elapsed_seconds / 60,
    }])
    summary.to_csv(results_dir / "summary.csv", index=False)

    metadata = {
        "delivery_profile": delivery_profile,
        "lambda_PPA": lambda_ppa,
        "c_ppa": c_ppa,
        "battery_case": battery_case,
        "weather_years": weather_years,
        "omega_definition": "weather year",
        "xi_definition": "wind profile scenario",
        "xi_wind_files": xi_wind_files,
        "stage2_wind_method": stage2_wind_method,
        "settlement_period": settlement_period,
        "shortfall_column_definition": (
            "Hourly positive PPA deviation for diagnostics only; the optimized "
            "settlement shortfall is stored in settlement_period_results.csv."
        ),
        "elapsed_seconds": run_elapsed_seconds,
    }

    with open(results_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    rows_12 = []
    for t in model.T:
        row = {
            "t": t,
            "P_PPA": pyo.value(model.P_PPA[t]),
        }
        for w in model.Omega:
            row[f"P_DA_omega_{w}"] = pyo.value(model.P_DA[t, w])
        rows_12.append(row)

    pd.DataFrame(rows_12).to_csv(results_dir / "stage1_2_results.csv", index=False)

    hour_to_settlement_period = {
        t: sp for sp in model.SP for t in model.T_in_SP[sp]
    }

    rows_3 = []
    for w in model.Omega:
        for x in model.Xi:
            for t in model.T:
                p_ppa = pyo.value(model.P_PPA[t])
                p_ppa_rt = pyo.value(model.P_PPA_rt[t, w, x])
                ppa_deviation = p_ppa - p_ppa_rt
                rows_3.append({
                    "t": t,
                    "omega": w,
                    "xi": x,
                    "settlement_period": hour_to_settlement_period[t],
                    "P_rt_grid": pyo.value(model.P_rt_grid[t, w, x]),
                    "P_mfrr_up": pyo.value(model.a_up[t, w, x]),
                    "P_mfrr_down": pyo.value(model.a_down[t, w, x]),
                    "P_discharge": pyo.value(model.P_dis_rt[t, w, x]),
                    "P_charge": pyo.value(model.P_cha_rt[t, w, x]),
                    "P_PPA_rt": p_ppa_rt,
                    "ppa_deviation": ppa_deviation,
                    "shortfall": max(ppa_deviation, 0.0),
                    "imbalance": pyo.value(model.delta[t, w, x]),
                    "curtailment": pyo.value(model.curt[t, w, x]),
                    "E_rt": pyo.value(model.E_rt[t, w, x]),
                    "wind_real": pyo.value(model.wind_real[t, w, x]),
                    "lambda_mfrr_down": pyo.value(model.lambda_mfrr_down[t, w, x]),
                    "y_down": pyo.value(model.y_down[t, w, x]),
                    "A_down_max": pyo.value(model.A_down_max[t, w]),
                    "lambda_mfrr_up": pyo.value(model.lambda_mfrr_up[t, w, x]),
                    "y_up": pyo.value(model.y_up[t, w, x]),
                    "A_up_max": pyo.value(model.A_up_max[t, w]),
                    "lambda_imb": pyo.value(model.lambda_imb[t, w, x]),
                    "lambda_spot": pyo.value(model.lambda_spot[t, w]),

                })

    pd.DataFrame(rows_3).to_csv(results_dir / "stage3_results.csv", index=False)

    delta_t = pyo.value(model.delta_t)
    settlement_rows = []
    for w in model.Omega:
        for x in model.Xi:
            for sp in model.SP:
                hours = list(model.T_in_SP[sp])
                contracted_energy = sum(
                    pyo.value(model.P_PPA[t]) * delta_t for t in hours
                )
                delivered_energy = sum(
                    pyo.value(model.P_PPA_rt[t, w, x]) * delta_t for t in hours
                )
                settlement_rows.append({
                    "settlement_period": sp,
                    "start_t": hours[0],
                    "end_t": hours[-1],
                    "number_of_hours": len(hours),
                    "omega": w,
                    "xi": x,
                    "contracted_energy": contracted_energy,
                    "delivered_energy": delivered_energy,
                    "net_underdelivery": contracted_energy - delivered_energy,
                    "settlement_shortfall": pyo.value(model.S_short[sp, w, x]),
                })

    pd.DataFrame(settlement_rows).to_csv(
        results_dir / "settlement_period_results.csv", index=False
    )


if __name__ == "__main__":
    main()
