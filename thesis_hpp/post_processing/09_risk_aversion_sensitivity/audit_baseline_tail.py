"""Reconstruct and inspect the CVaR tail of the existing beta=0.2 run."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    ALPHA,
    BASELINE_BETA,
    C_PPA,
    E_MAX,
    LAMBDA_PPA,
    P_B_MAX,
    SETTLEMENT_PERIOD,
)


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
INPUTS_DIR = PROJECT_ROOT / "inputs" / "Input_files_created"
OUTPUT_DIR = HERE / "outputs" / "baseline_tail_audit"
CSV_CHUNK_SIZE = 200_000

COMPONENTS = (
    "ppa_revenue",
    "spot_revenue",
    "imbalance_settlement",
    "imbalance_penalty",
    "mfrr_up",
    "mfrr_down",
    "ppa_shortfall_cost",
    "battery_degradation",
)


def number_slug(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def case_directory() -> Path:
    return (
        PROJECT_ROOT
        / "results"
        / f"settlement_{SETTLEMENT_PERIOD}"
        / "generic_industrial_yearly"
        / f"c_ppa_{number_slug(C_PPA)}"
        / f"battery_p_{number_slug(P_B_MAX)}_e_{number_slug(E_MAX)}"
        / f"lambda_ppa_{number_slug(LAMBDA_PPA)}"
    )


def load_probabilities() -> pd.DataFrame:
    omega = pd.read_csv(
        INPUTS_DIR / "omega_probabilities.csv",
        usecols=["omega", "weather_year", "prob"],
    ).rename(columns={"prob": "pi_omega"})
    xi = pd.read_csv(
        INPUTS_DIR / "xi_probabilities.csv",
        usecols=["omega", "xi", "wind_file", "prob"],
    ).rename(columns={"prob": "pi_xi_given_omega"})
    probabilities = xi.merge(omega, on="omega", validate="many_to_one")
    probabilities["scenario_probability"] = (
        probabilities["pi_omega"] * probabilities["pi_xi_given_omega"]
    )
    if not math.isclose(
        probabilities["scenario_probability"].sum(),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise ValueError("The 80 scenario probabilities do not sum to one.")
    return probabilities


def load_stage_two(case_dir: Path) -> pd.DataFrame:
    stage_two = pd.read_csv(case_dir / "stage1_2_results.csv")
    da_columns = [
        column for column in stage_two if column.startswith("P_DA_omega_")
    ]
    stage_two = stage_two.melt(
        id_vars=["t", "P_PPA"],
        value_vars=da_columns,
        var_name="omega_column",
        value_name="P_DA",
    )
    stage_two["omega"] = (
        stage_two["omega_column"].str.rsplit("_", n=1).str[-1].astype(int)
    )
    return stage_two[["t", "omega", "P_PPA", "P_DA"]]


def lower_tail_cvar(profits: np.ndarray, probabilities: np.ndarray) -> float:
    losses = -profits
    loss_cvar = min(
        eta
        + np.sum(probabilities * np.maximum(losses - eta, 0.0))
        / (1.0 - ALPHA)
        for eta in np.unique(losses)
    )
    return -float(loss_cvar)


def reconstruct_scenario_profits() -> pd.DataFrame:
    case_dir = case_directory()
    required = (
        case_dir / "stage1_2_results.csv",
        case_dir / "stage3_results.csv",
        case_dir / "settlement_period_results.csv",
        case_dir / "summary.csv",
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing result files:\n" + "\n".join(map(str, missing)))

    with (INPUTS_DIR / "config.json").open("r", encoding="utf-8") as handle:
        parameters = json.load(handle)

    probabilities = load_probabilities()
    stage_two = load_stage_two(case_dir)
    scenario_index = pd.MultiIndex.from_frame(probabilities[["omega", "xi"]])
    hourly_components = [
        component for component in COMPONENTS if component != "ppa_shortfall_cost"
    ]
    scenario_totals = pd.DataFrame(
        0.0,
        index=scenario_index,
        columns=hourly_components,
    )

    stage_three_columns = [
        "t",
        "omega",
        "xi",
        "P_mfrr_up",
        "P_mfrr_down",
        "P_discharge",
        "P_charge",
        "imbalance",
        "lambda_mfrr_down",
        "lambda_mfrr_up",
        "lambda_imb",
        "lambda_spot",
    ]
    for chunk in pd.read_csv(
        case_dir / "stage3_results.csv",
        usecols=stage_three_columns,
        chunksize=CSV_CHUNK_SIZE,
    ):
        chunk = chunk.merge(
            stage_two,
            on=["t", "omega"],
            how="left",
            validate="many_to_one",
        )
        chunk["ppa_revenue"] = LAMBDA_PPA * chunk["P_PPA"]
        chunk["spot_revenue"] = chunk["lambda_spot"] * chunk["P_DA"]
        chunk["imbalance_settlement"] = chunk["lambda_imb"] * chunk["imbalance"]
        chunk["imbalance_penalty"] = -float(parameters["c_imb"]) * chunk[
            "imbalance"
        ].abs()
        chunk["mfrr_up"] = chunk["lambda_mfrr_up"] * chunk["P_mfrr_up"]
        chunk["mfrr_down"] = -chunk["lambda_mfrr_down"] * chunk["P_mfrr_down"]
        chunk["battery_degradation"] = -float(
            parameters["degradation_cost_per_mw"]
        ) * (chunk["P_discharge"] + chunk["P_charge"])
        grouped = chunk.groupby(["omega", "xi"], sort=False)[
            scenario_totals.columns
        ].sum()
        scenario_totals.loc[grouped.index, grouped.columns] += grouped

    shortfall = pd.read_csv(
        case_dir / "settlement_period_results.csv",
        usecols=["omega", "xi", "settlement_shortfall"],
    )
    shortfall = shortfall.groupby(["omega", "xi"])[
        "settlement_shortfall"
    ].sum()
    scenario_totals["ppa_shortfall_cost"] = (
        -C_PPA * shortfall.reindex(scenario_totals.index, fill_value=0.0)
    )

    scenarios = scenario_totals.reset_index().merge(
        probabilities,
        on=["omega", "xi"],
        validate="one_to_one",
    )
    scenarios["scenario_profit"] = scenarios[list(COMPONENTS)].sum(axis=1)
    scenarios = scenarios.sort_values("scenario_profit").reset_index(drop=True)
    scenarios["profit_rank"] = scenarios.index + 1

    tail_probability = 1.0 - ALPHA
    equal_probability = scenarios["scenario_probability"].nunique() == 1
    tail_count = tail_probability / scenarios["scenario_probability"].iloc[0]
    if not equal_probability or not math.isclose(tail_count, round(tail_count)):
        raise ValueError(
            "This compact tail flag expects equal probabilities and an integer tail size."
        )
    tail_count = int(round(tail_count))
    var_profit = float(scenarios.iloc[tail_count - 1]["scenario_profit"])
    tolerance = max(1e-7 * abs(var_profit), 1e-6)
    strict_tail = scenarios["scenario_profit"] < var_profit - tolerance
    boundary = np.isclose(
        scenarios["scenario_profit"],
        var_profit,
        rtol=0.0,
        atol=tolerance,
    )
    scenarios["tail_status"] = np.select(
        [strict_tail, boundary],
        ["strict_tail", "var_boundary"],
        default="outside_tail",
    )
    strict_probability = float(
        scenarios.loc[strict_tail, "scenario_probability"].sum()
    )
    remaining_probability = tail_probability - strict_probability
    boundary_count = int(boundary.sum())
    scenarios["cvar_tail_probability"] = 0.0
    scenarios.loc[strict_tail, "cvar_tail_probability"] = scenarios.loc[
        strict_tail, "scenario_probability"
    ]
    scenarios.loc[boundary, "cvar_tail_probability"] = (
        remaining_probability / boundary_count
    )

    expected_profit = float(
        (scenarios["scenario_profit"] * scenarios["scenario_probability"]).sum()
    )
    cvar_profit = lower_tail_cvar(
        scenarios["scenario_profit"].to_numpy(),
        scenarios["scenario_probability"].to_numpy(),
    )
    objective = (1.0 - BASELINE_BETA) * expected_profit + BASELINE_BETA * cvar_profit
    reported_objective = float(pd.read_csv(case_dir / "summary.csv")["objective_value"].iloc[0])
    difference = objective - reported_objective
    if abs(difference) > 5.0:
        raise ValueError(
            "The reconstructed objective does not match the solved model: "
            f"difference={difference:.6f} EUR."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scenarios.to_csv(
        OUTPUT_DIR / "scenario_profit_audit_beta_0p2.csv",
        index=False,
        float_format="%.12g",
    )
    scenarios.loc[scenarios["tail_status"] != "outside_tail"].to_csv(
        OUTPUT_DIR / "cvar_tail_scenarios_beta_0p2.csv",
        index=False,
        float_format="%.12g",
    )

    weather_summary = scenarios.groupby("weather_year", as_index=False).agg(
        mean_scenario_profit=("scenario_profit", "mean"),
        minimum_scenario_profit=("scenario_profit", "min"),
        maximum_scenario_profit=("scenario_profit", "max"),
        strict_tail_scenarios=(
            "tail_status",
            lambda values: int((values == "strict_tail").sum()),
        ),
        var_boundary_scenarios=(
            "tail_status",
            lambda values: int((values == "var_boundary").sum()),
        ),
        cvar_tail_probability=("cvar_tail_probability", "sum"),
    )
    weather_summary.sort_values("mean_scenario_profit").to_csv(
        OUTPUT_DIR / "weather_year_profit_summary_beta_0p2.csv",
        index=False,
        float_format="%.12g",
    )

    print(f"Expected profit: {expected_profit:.6f} EUR")
    print(f"CVaR (mean of the worst 8 scenarios): {cvar_profit:.6f} EUR")
    print(f"Reconstructed objective: {objective:.6f} EUR")
    print(f"Reported objective: {reported_objective:.6f} EUR")
    print(f"Difference: {difference:.6f} EUR")
    print("\nStrict CVaR-tail and VaR-boundary scenarios:")
    print(
        scenarios.loc[
            scenarios["tail_status"] != "outside_tail",
            [
                "profit_rank",
                "weather_year",
                "xi",
                "wind_file",
                "scenario_profit",
                "tail_status",
                "cvar_tail_probability",
            ],
        ].to_string(index=False)
    )
    print(f"\nFiles written to: {OUTPUT_DIR}")
    return scenarios


if __name__ == "__main__":
    reconstruct_scenario_profits()
