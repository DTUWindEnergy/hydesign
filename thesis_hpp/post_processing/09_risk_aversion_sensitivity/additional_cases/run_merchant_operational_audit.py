"""Re-solve merchant beta=0 and beta=0.8 and export compact operations."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import pandas as pd
import pyomo.environ as pyo


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
INPUTS_ROOT = PROJECT_ROOT / "inputs"
GENERATED_INPUTS = INPUTS_ROOT / "Input_files_created"
OUTPUT_ROOT = HERE / "outputs" / "merchant_operational_audit"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(INPUTS_ROOT))

from prepare_inputs_from_btc_2000 import prepare_inputs  # noqa: E402
from settings import (  # noqa: E402
    ALPHA,
    CASES,
    DELIVERY_PROFILE,
    E0,
    E_MAX,
    P_B_MAX,
    SETTLEMENT_PERIOD,
    SOLVER_NAME,
    SOLVER_THREADS,
    STAGE2_WIND_METHOD,
    WEATHER_YEARS,
    XI_WIND_FILES,
)
from run_additional_cases import (  # noqa: E402
    beta_slug,
    expected_profit,
    lower_tail_cvar,
    scenario_rows,
)
from src.thesis_hpp.data_loader import load_all_data  # noqa: E402
from src.thesis_hpp.model import build_model  # noqa: E402


AUDIT_BETAS = (0.0, 0.8)

ENERGY_VARIABLES = {
    "real_time_grid_energy_mwh": "P_rt_grid",
    "battery_discharge_energy_mwh": "P_dis_rt",
    "battery_charge_energy_mwh": "P_cha_rt",
    "mfrr_up_energy_mwh": "a_up",
    "mfrr_down_energy_mwh": "a_down",
    "positive_imbalance_energy_mwh": "delta_pos",
    "negative_imbalance_energy_mwh": "delta_neg",
    "curtailment_energy_mwh": "curt",
    "down_regulation_battery_charge_mwh": "P_cha_down",
    "down_regulation_curtailment_mwh": "curt_down",
}


def merchant_case() -> dict:
    return next(case for case in CASES if case["case_id"] == "merchant")


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def audit_paths(beta: float) -> dict[str, Path]:
    root = OUTPUT_ROOT / beta_slug(beta)
    return {
        "root": root,
        "summary": root / "annual_expected_summary.json",
        "scenarios": root / "scenario_operational_totals.csv",
        "hourly": root / "expected_hourly_operation.csv",
    }


def audit_is_complete(beta: float) -> bool:
    paths = audit_paths(beta)
    return all(paths[key].exists() for key in ("summary", "scenarios", "hourly"))


def solve_and_export(beta: float) -> dict:
    case = merchant_case()
    data = load_all_data(GENERATED_INPUTS)
    cfg = data["config"]
    cfg["lambda_PPA"] = float(case["lambda_ppa"])
    cfg["c_ppa"] = float(case["c_ppa"])
    cfg["P_b_max"] = P_B_MAX
    cfg["E_max"] = E_MAX
    cfg["E0"] = E0
    cfg["settlement_period"] = SETTLEMENT_PERIOD
    cfg["alpha"] = ALPHA
    cfg["beta"] = beta

    model = build_model(data)
    model.S_PPA.fix(0.0)

    solver = pyo.SolverFactory(SOLVER_NAME)
    if SOLVER_NAME == "cplex":
        solver.options["threads"] = SOLVER_THREADS
    started = time.perf_counter()
    results = solver.solve(model, tee=True)
    elapsed = time.perf_counter() - started
    if not pyo.check_optimal_termination(results):
        raise RuntimeError(
            f"Merchant audit failed for beta={beta:g}: "
            f"status={results.solver.status}, "
            f"termination={results.solver.termination_condition}"
        )

    profit_rows = scenario_rows(model, case, beta)
    profit_lookup = {
        (row["omega"], row["xi"]): row for row in profit_rows
    }
    delta_t = float(pyo.value(model.delta_t))
    time_points = list(model.T)
    n_time_points = len(time_points)
    hourly = {
        int(t): {
            "beta": beta,
            "t": int(t),
            "expected_day_ahead_power_mw": 0.0,
            "expected_real_time_grid_power_mw": 0.0,
            "expected_battery_discharge_power_mw": 0.0,
            "expected_battery_charge_power_mw": 0.0,
            "expected_mfrr_up_power_mw": 0.0,
            "expected_mfrr_down_power_mw": 0.0,
            "expected_signed_imbalance_power_mw": 0.0,
            "expected_curtailment_power_mw": 0.0,
            "expected_soc_mwh": 0.0,
        }
        for t in time_points
    }

    scenario_operations = []
    for w in model.Omega:
        pi_w = float(pyo.value(model.pi_omega[w]))
        day_ahead_energy = sum(
            float(pyo.value(model.P_DA[t, w])) * delta_t for t in time_points
        )
        for x in model.Xi:
            probability = pi_w * float(pyo.value(model.pi_xi_given_omega[w, x]))
            totals = {name: 0.0 for name in ENERGY_VARIABLES}
            signed_imbalance = 0.0
            soc_sum = 0.0
            soc_min = float("inf")
            soc_max = float("-inf")

            for t in time_points:
                values = {
                    name: float(pyo.value(getattr(model, variable)[t, w, x]))
                    for name, variable in ENERGY_VARIABLES.items()
                }
                delta = float(pyo.value(model.delta[t, w, x]))
                soc = float(pyo.value(model.E_rt[t, w, x]))
                for name, value in values.items():
                    totals[name] += value * delta_t
                signed_imbalance += delta * delta_t
                soc_sum += soc
                soc_min = min(soc_min, soc)
                soc_max = max(soc_max, soc)

                h = hourly[int(t)]
                h["expected_day_ahead_power_mw"] += (
                    probability * float(pyo.value(model.P_DA[t, w]))
                )
                h["expected_real_time_grid_power_mw"] += (
                    probability * values["real_time_grid_energy_mwh"]
                )
                h["expected_battery_discharge_power_mw"] += (
                    probability * values["battery_discharge_energy_mwh"]
                )
                h["expected_battery_charge_power_mw"] += (
                    probability * values["battery_charge_energy_mwh"]
                )
                h["expected_mfrr_up_power_mw"] += (
                    probability * values["mfrr_up_energy_mwh"]
                )
                h["expected_mfrr_down_power_mw"] += (
                    probability * values["mfrr_down_energy_mwh"]
                )
                h["expected_signed_imbalance_power_mw"] += probability * delta
                h["expected_curtailment_power_mw"] += (
                    probability * values["curtailment_energy_mwh"]
                )
                h["expected_soc_mwh"] += probability * soc

            profit = profit_lookup[(int(w), int(x))]
            scenario_operations.append(
                {
                    "beta": beta,
                    "omega": int(w),
                    "weather_year": profit["weather_year"],
                    "xi": int(x),
                    "wind_file": profit["wind_file"],
                    "scenario_probability": probability,
                    "scenario_profit_eur": profit["scenario_profit"],
                    "profit_rank": profit["profit_rank"],
                    "tail_status": profit["tail_status"],
                    "day_ahead_energy_mwh": day_ahead_energy,
                    **totals,
                    "signed_imbalance_energy_mwh": signed_imbalance,
                    "mean_soc_mwh": soc_sum / n_time_points,
                    "minimum_soc_mwh": soc_min,
                    "maximum_soc_mwh": soc_max,
                }
            )

    scenario_df = pd.DataFrame(scenario_operations).sort_values(["omega", "xi"])
    hourly_df = pd.DataFrame(hourly.values()).sort_values("t")
    expected = expected_profit(profit_rows)
    cvar = lower_tail_cvar(profit_rows, ALPHA)
    objective = float(pyo.value(model.OBJ))
    summary = {
        "case_id": "merchant",
        "beta": beta,
        "alpha": ALPHA,
        "delivery_profile": DELIVERY_PROFILE,
        "settlement_period": SETTLEMENT_PERIOD,
        "P_b_max_mw": P_B_MAX,
        "E_max_mwh": E_MAX,
        "S_PPA_mw": float(pyo.value(model.S_PPA)),
        "expected_profit_eur": expected,
        "cvar_profit_eur": cvar,
        "objective_value_eur": objective,
        "elapsed_seconds": elapsed,
        "solver_status": str(results.solver.status),
        "termination_condition": str(results.solver.termination_condition),
        "expected_day_ahead_energy_mwh": float(
            (scenario_df["scenario_probability"] * scenario_df["day_ahead_energy_mwh"]).sum()
        ),
        "expected_signed_imbalance_energy_mwh": float(
            (
                scenario_df["scenario_probability"]
                * scenario_df["signed_imbalance_energy_mwh"]
            ).sum()
        ),
        "expected_mean_soc_mwh": float(
            (scenario_df["scenario_probability"] * scenario_df["mean_soc_mwh"]).sum()
        ),
    }
    for name in ENERGY_VARIABLES:
        summary[f"expected_{name}"] = float(
            (scenario_df["scenario_probability"] * scenario_df[name]).sum()
        )

    paths = audit_paths(beta)
    paths["root"].mkdir(parents=True, exist_ok=True)
    scenario_df.to_csv(paths["scenarios"], index=False, float_format="%.12g")
    hourly_df.to_csv(paths["hourly"], index=False, float_format="%.12g")
    write_json(paths["summary"], summary)
    return summary


def write_comparisons() -> None:
    summaries = {}
    for beta in AUDIT_BETAS:
        with audit_paths(beta)["summary"].open("r", encoding="utf-8") as handle:
            summaries[beta] = json.load(handle)

    excluded = {
        "beta",
        "alpha",
        "elapsed_seconds",
        "P_b_max_mw",
        "E_max_mwh",
        "S_PPA_mw",
    }
    rows = []
    for metric, baseline in summaries[0.0].items():
        if metric in excluded or not isinstance(baseline, (int, float)):
            continue
        comparison = summaries[0.8][metric]
        change = comparison - baseline
        rows.append(
            {
                "metric": metric,
                "beta_0": baseline,
                "beta_0p8": comparison,
                "absolute_change": change,
                "percentage_change_from_beta_0": (
                    100.0 * change / baseline if baseline != 0.0 else None
                ),
            }
        )
    pd.DataFrame(rows).to_csv(
        OUTPUT_ROOT / "annual_expected_comparison_beta_0_vs_0p8.csv",
        index=False,
        float_format="%.12g",
    )

    scenario_frames = []
    hourly_frames = []
    for beta in AUDIT_BETAS:
        scenario = pd.read_csv(audit_paths(beta)["scenarios"])
        hourly = pd.read_csv(audit_paths(beta)["hourly"])
        scenario_frames.append(scenario)
        hourly_frames.append(hourly)

    scenario_keys = ["omega", "weather_year", "xi", "wind_file"]
    scenario_merged = scenario_frames[0].merge(
        scenario_frames[1],
        on=scenario_keys,
        suffixes=("_beta_0", "_beta_0p8"),
        validate="one_to_one",
    )
    for column in scenario_frames[0].select_dtypes(include="number").columns:
        if column in scenario_keys or column in {"beta", "scenario_probability", "profit_rank"}:
            continue
        scenario_merged[f"{column}_change"] = (
            scenario_merged[f"{column}_beta_0p8"]
            - scenario_merged[f"{column}_beta_0"]
        )
    scenario_merged.to_csv(
        OUTPUT_ROOT / "scenario_comparison_beta_0_vs_0p8.csv",
        index=False,
        float_format="%.12g",
    )

    hourly_merged = hourly_frames[0].merge(
        hourly_frames[1],
        on="t",
        suffixes=("_beta_0", "_beta_0p8"),
        validate="one_to_one",
    )
    for column in hourly_frames[0].columns:
        if column in {"beta", "t"}:
            continue
        hourly_merged[f"{column}_change"] = (
            hourly_merged[f"{column}_beta_0p8"]
            - hourly_merged[f"{column}_beta_0"]
        )
    hourly_merged.to_csv(
        OUTPUT_ROOT / "expected_hourly_comparison_beta_0_vs_0p8.csv",
        index=False,
        float_format="%.12g",
    )


def main() -> None:
    prepare_inputs(
        delivery_profile_name=DELIVERY_PROFILE,
        weather_years=list(WEATHER_YEARS),
        xi_wind_files=list(XI_WIND_FILES),
        stage2_wind_method=STAGE2_WIND_METHOD,
    )
    for beta in AUDIT_BETAS:
        if audit_is_complete(beta):
            print(f"Skipping completed merchant operational audit for beta={beta:g}.")
            continue
        print(f"\n=== MERCHANT OPERATIONAL AUDIT: BETA={beta:g} ===")
        summary = solve_and_export(beta)
        print(f"Expected profit: {summary['expected_profit_eur']:.6f} EUR")
        print(f"CVaR: {summary['cvar_profit_eur']:.6f} EUR")
        print(f"Output: {audit_paths(beta)['root']}")

    write_comparisons()
    print(f"\nMerchant beta=0 versus beta=0.8 audit: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
