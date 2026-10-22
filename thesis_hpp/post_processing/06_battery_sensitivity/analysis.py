"""Shared data preparation and profit-component calculations."""

from __future__ import annotations

from pathlib import Path
import json
import math
import os
import sys

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
POST_PROCESSING_ROOT = HERE.parent
THESIS_ROOT = POST_PROCESSING_ROOT.parent
CATALOG_PATH = POST_PROCESSING_ROOT / "common" / "results_catalog.csv"

sys.path.insert(0, str(POST_PROCESSING_ROOT))
sys.path.insert(0, str(THESIS_ROOT / "inputs"))

from common.results_catalog import (  # noqa: E402
    PROJECT_FOLDERS,
    build_results_catalog,
    project_roots,
)
from sdp import get_delivery_profile  # noqa: E402
from config import (  # noqa: E402
    BATTERY_CASES,
    C_PPA_VALUES,
    CSV_CHUNK_SIZE,
    LAMBDA_PPA_VALUES,
    OBJECTIVE_AUDIT_ABS_TOL_EUR,
    OBJECTIVE_AUDIT_REL_TOL,
    SETTLEMENT_PERIOD,
)


PROFILE_ORDER = list(PROJECT_FOLDERS)
COMPONENTS = (
    ("ppa_revenue", "PPA revenue"),
    ("spot_revenue", "Day-ahead spot revenue"),
    ("imbalance_settlement", "Imbalance settlement"),
    ("imbalance_penalty", "Imbalance penalty"),
    ("mfrr_up", "mFRR-up revenue"),
    ("mfrr_down", "mFRR-down revenue"),
    ("ppa_shortfall_cost", "PPA shortfall cost"),
    ("battery_degradation", "Battery degradation cost"),
)
COMPONENT_KEYS = [key for key, _ in COMPONENTS]
COMPONENT_LABELS = {key: label for key, label in COMPONENTS}


def number_slug(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def profile_slug(profile: str) -> str:
    return "_".join(str(profile).lower().replace("-", " ").split())


def battery_label(power: float, energy: float) -> str:
    if math.isclose(float(power), 0.0) and math.isclose(float(energy), 0.0):
        return "No battery"
    return f"{float(power):g} MW/{float(energy):g} MWh"


def load_catalog() -> pd.DataFrame:
    refresh = os.environ.get("REFRESH_RESULTS_CATALOG", "0") == "1"
    if refresh or not CATALOG_PATH.exists():
        catalog = build_results_catalog(save=True)
    else:
        catalog = pd.read_csv(CATALOG_PATH)
        print(f"Loaded cached results catalog: {CATALOG_PATH}")

    numeric_columns = [
        "lambda_PPA",
        "c_ppa",
        "P_b_max",
        "E_max",
        "objective_value",
        "S_PPA",
    ]
    for column in numeric_columns:
        catalog[column] = pd.to_numeric(catalog[column], errors="raise")
    return catalog


def battery_mask(catalog: pd.DataFrame) -> pd.Series:
    selected = pd.Series(False, index=catalog.index)
    for power, energy in BATTERY_CASES:
        selected |= np.isclose(catalog["P_b_max"], power) & np.isclose(
            catalog["E_max"], energy
        )
    return selected


def select_contract_cases(
    catalog: pd.DataFrame,
    profile: str | None,
) -> pd.DataFrame:
    profile_values = PROFILE_ORDER if profile is None else [profile]
    if not set(profile_values).issubset(PROFILE_ORDER):
        raise ValueError(
            f"Unknown profile '{profile}'. Choose one of: {', '.join(PROFILE_ORDER)}."
        )

    selected = catalog.loc[
        catalog["settlement_period"].str.lower().eq(SETTLEMENT_PERIOD.lower())
        & battery_mask(catalog)
        & catalog["c_ppa"].isin(C_PPA_VALUES)
        & catalog["lambda_PPA"].isin(LAMBDA_PPA_VALUES)
        & catalog["sdp_label"].isin(profile_values)
    ].copy()

    key_columns = [
        "sdp_label",
        "settlement_period",
        "P_b_max",
        "E_max",
        "c_ppa",
        "lambda_PPA",
    ]
    duplicates = selected.duplicated(key_columns, keep=False)
    if duplicates.any():
        raise ValueError(
            "Duplicate battery-sensitivity cases found:\n"
            + selected.loc[duplicates, key_columns + ["case_dir"]].to_string(
                index=False
            )
        )

    expected = (
        len(profile_values)
        * len(BATTERY_CASES)
        * len(C_PPA_VALUES)
        * len(LAMBDA_PPA_VALUES)
    )
    if len(selected) != expected:
        raise ValueError(
            f"Expected {expected} battery-sensitivity cases, found "
            f"{len(selected)}.\n"
            + selected[key_columns].to_string(index=False)
        )

    profile_flh = {}
    for label in profile_values:
        names = selected.loc[
            selected["sdp_label"] == label, "delivery_profile"
        ].drop_duplicates()
        if len(names) != 1:
            raise ValueError(
                f"Expected one delivery-profile definition for {label}, "
                f"found {len(names)}."
            )
        shape = get_delivery_profile(str(names.iloc[0])).hourly_shape
        profile_flh[label] = float(sum(shape))

    selected["profile_full_load_hours"] = selected["sdp_label"].map(profile_flh)
    selected["contracted_energy_mwh_per_day"] = (
        selected["S_PPA"] * selected["profile_full_load_hours"]
    )
    selected["battery_label"] = [
        battery_label(power, energy)
        for power, energy in zip(selected["P_b_max"], selected["E_max"])
    ]
    return selected.sort_values(key_columns).reset_index(drop=True)


def local_case_directory(case: pd.Series) -> Path:
    case_dir = str(case["case_dir"]).replace("\\", "/")
    marker = "/results/"
    if marker not in case_dir:
        raise ValueError(f"Cannot interpret catalog case path: {case_dir}")
    relative_path = case_dir.split(marker, maxsplit=1)[1]
    project_root = project_roots()[str(case["sdp_label"])]
    return project_root / "results" / Path(*relative_path.split("/"))


def load_project_parameters(project_root: Path) -> dict[str, float]:
    config_path = project_root / "inputs" / "Input_files_created" / "config.json"
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    required = ["c_imb", "degradation_cost_per_mw", "alpha", "beta"]
    missing = [key for key in required if key not in config]
    if missing:
        raise KeyError(f"Missing {missing} from {config_path}")
    return {key: float(config[key]) for key in required}


def load_scenario_probabilities(project_root: Path) -> pd.DataFrame:
    inputs_dir = project_root / "inputs" / "Input_files_created"
    omega = pd.read_csv(
        inputs_dir / "omega_probabilities.csv",
        usecols=["omega", "prob"],
    ).rename(columns={"prob": "pi_omega"})
    xi = pd.read_csv(
        inputs_dir / "xi_probabilities.csv",
        usecols=["omega", "xi", "prob"],
    ).rename(columns={"prob": "pi_xi_given_omega"})

    probabilities = xi.merge(omega, on="omega", validate="many_to_one")
    probabilities["scenario_probability"] = (
        probabilities["pi_omega"] * probabilities["pi_xi_given_omega"]
    )
    if not math.isclose(
        float(omega["pi_omega"].sum()), 1.0, rel_tol=0.0, abs_tol=1e-10
    ):
        raise ValueError(f"Omega probabilities do not sum to one in {inputs_dir}.")
    xi_sums = xi.groupby("omega")["pi_xi_given_omega"].sum()
    if not np.allclose(xi_sums.to_numpy(), 1.0, atol=1e-10, rtol=0.0):
        raise ValueError(
            f"Conditional xi probabilities do not sum to one in {inputs_dir}."
        )
    return probabilities[["omega", "xi", "scenario_probability"]]


def load_stage_two(case_dir: Path) -> pd.DataFrame:
    stage_two = pd.read_csv(case_dir / "stage1_2_results.csv")
    da_columns = [
        column for column in stage_two.columns if column.startswith("P_DA_omega_")
    ]
    if not da_columns:
        raise ValueError(f"No day-ahead columns found in {case_dir}.")

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


def lower_tail_cvar(
    scenario_profit: np.ndarray,
    probability: np.ndarray,
    alpha: float,
) -> float:
    losses = -np.asarray(scenario_profit, dtype=float)
    weights = np.asarray(probability, dtype=float)
    candidates = np.unique(losses)
    loss_cvar = min(
        eta + np.sum(weights * np.maximum(losses - eta, 0.0)) / (1.0 - alpha)
        for eta in candidates
    )
    return -float(loss_cvar)


def calculate_profit_components(case: pd.Series) -> dict[str, float | str]:
    case_dir = local_case_directory(case)
    required_files = [
        case_dir / "stage1_2_results.csv",
        case_dir / "stage3_results.csv",
        case_dir / "settlement_period_results.csv",
    ]
    missing = [path for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing detailed result files:\n" + "\n".join(map(str, missing))
        )

    project_root = project_roots()[str(case["sdp_label"])]
    parameters = load_project_parameters(project_root)
    probabilities = load_scenario_probabilities(project_root)
    stage_two = load_stage_two(case_dir)

    scenario_index = pd.MultiIndex.from_frame(probabilities[["omega", "xi"]])
    scenario_totals = pd.DataFrame(
        0.0,
        index=scenario_index,
        columns=[
            "ppa_revenue",
            "spot_revenue",
            "mfrr_up",
            "mfrr_down",
            "imbalance_settlement",
            "imbalance_penalty",
            "battery_degradation",
        ],
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
        if chunk[["P_PPA", "P_DA"]].isna().any().any():
            raise ValueError(f"Stage-two merge failed for {case_dir}.")

        chunk["ppa_revenue"] = float(case["lambda_PPA"]) * chunk["P_PPA"]
        chunk["spot_revenue"] = chunk["lambda_spot"] * chunk["P_DA"]
        chunk["mfrr_up"] = chunk["lambda_mfrr_up"] * chunk["P_mfrr_up"]
        chunk["mfrr_down"] = -chunk["lambda_mfrr_down"] * chunk["P_mfrr_down"]
        chunk["imbalance_settlement"] = chunk["lambda_imb"] * chunk["imbalance"]
        chunk["imbalance_penalty"] = (
            -parameters["c_imb"] * chunk["imbalance"].abs()
        )
        chunk["battery_degradation"] = -parameters[
            "degradation_cost_per_mw"
        ] * (chunk["P_discharge"] + chunk["P_charge"])

        grouped = chunk.groupby(["omega", "xi"], sort=False)[
            scenario_totals.columns
        ].sum()
        scenario_totals.loc[grouped.index, grouped.columns] += grouped

    shortfall = pd.read_csv(
        case_dir / "settlement_period_results.csv",
        usecols=["omega", "xi", "settlement_shortfall"],
    )
    shortfall_by_scenario = shortfall.groupby(["omega", "xi"])[
        "settlement_shortfall"
    ].sum()
    scenario_totals["ppa_shortfall_cost"] = (
        -float(case["c_ppa"])
        * shortfall_by_scenario.reindex(scenario_totals.index, fill_value=0.0)
    )

    scenario_totals = scenario_totals.reset_index().merge(
        probabilities,
        on=["omega", "xi"],
        validate="one_to_one",
    )
    scenario_totals["scenario_profit"] = scenario_totals[COMPONENT_KEYS].sum(
        axis=1
    )
    expected_components = {
        key: float(
            (scenario_totals[key] * scenario_totals["scenario_probability"]).sum()
        )
        for key in COMPONENT_KEYS
    }
    expected_profit = float(sum(expected_components.values()))
    cvar_profit = lower_tail_cvar(
        scenario_totals["scenario_profit"].to_numpy(),
        scenario_totals["scenario_probability"].to_numpy(),
        parameters["alpha"],
    )
    reconstructed_objective = (
        (1.0 - parameters["beta"]) * expected_profit
        + parameters["beta"] * cvar_profit
    )
    reported_objective = float(case["objective_value"])
    objective_difference = reconstructed_objective - reported_objective
    tolerance = max(
        OBJECTIVE_AUDIT_ABS_TOL_EUR,
        OBJECTIVE_AUDIT_REL_TOL * abs(reported_objective),
    )
    if abs(objective_difference) > tolerance:
        raise ValueError(
            "Profit decomposition does not reproduce the reported objective.\n"
            f"Case: {case_dir}\n"
            f"Reconstructed: {reconstructed_objective:.6f}\n"
            f"Reported: {reported_objective:.6f}\n"
            f"Difference: {objective_difference:.6f}"
        )

    return {
        "settlement_period": str(case["settlement_period"]),
        "P_b_max": float(case["P_b_max"]),
        "E_max": float(case["E_max"]),
        "battery_label": battery_label(case["P_b_max"], case["E_max"]),
        "c_ppa": float(case["c_ppa"]),
        "lambda_PPA": float(case["lambda_PPA"]),
        "sdp_label": str(case["sdp_label"]),
        "S_PPA": float(case["S_PPA"]),
        **expected_components,
        "expected_profit": expected_profit,
        "cvar_profit": cvar_profit,
        "reconstructed_objective": reconstructed_objective,
        "reported_objective": reported_objective,
        "objective_audit_difference": objective_difference,
        "case_dir": str(case_dir),
    }
