"""Compare daily SoC cycling across battery sizes with the 72-hour case."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
POST_PROCESSING_ROOT = HERE.parent
THESIS_ROOT = POST_PROCESSING_ROOT.parent
OUTPUT_ROOT = HERE / "outputs"
THREE_DAY_ROOT = OUTPUT_ROOT / "three_day_run"
DAILY_RESULTS = (
    POST_PROCESSING_ROOT
    / "06_battery_sensitivity"
    / "outputs"
    / "battery_expected_profit_cases.csv"
)
NPV_SETTINGS_PATH = POST_PROCESSING_ROOT / "08_npv_analysis" / "settings.py"
COMPARISON_PATH = OUTPUT_ROOT / "soc_cycling_npv_comparison.csv"
DAILY_NPV_PATH = OUTPUT_ROOT / "daily_cycling_battery_npv_results.csv"

sys.path.insert(0, str(HERE))

from config import (  # noqa: E402
    C_PPA,
    DAILY_CYCLING_BATTERY_CASES,
    LAMBDA_PPA,
    SDP_LABEL,
    SETTLEMENT_PERIOD,
    SOC_CYCLE_HOURS,
    THREE_DAY_BATTERY_CASES,
)


def load_npv_settings():
    spec = spec_from_file_location("soc_npv_settings", NPV_SETTINGS_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load NPV settings from {NPV_SETTINGS_PATH}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expected_wind_energy_mwh() -> float:
    inputs = THESIS_ROOT / "inputs" / "Input_files_created"
    wind = pd.read_csv(
        inputs / "wind_realization.csv",
        usecols=["omega", "xi", "wind"],
    )
    omega = pd.read_csv(
        inputs / "omega_probabilities.csv",
        usecols=["omega", "prob"],
    ).rename(columns={"prob": "pi_omega"})
    xi = pd.read_csv(
        inputs / "xi_probabilities.csv",
        usecols=["omega", "xi", "prob"],
    ).rename(columns={"prob": "pi_xi_given_omega"})
    weighted = wind.merge(xi, on=["omega", "xi"], validate="many_to_one")
    weighted = weighted.merge(omega, on="omega", validate="many_to_one")
    return float(
        (
            weighted["wind"]
            * weighted["pi_omega"]
            * weighted["pi_xi_given_omega"]
        ).sum()
    )


def select_all_daily_cases() -> pd.DataFrame:
    if not DAILY_RESULTS.exists():
        raise FileNotFoundError(
            f"Missing {DAILY_RESULTS}. Run the expected-profit calculation in "
            "post_processing/06_battery_sensitivity first."
        )
    data = pd.read_csv(DAILY_RESULTS)
    selected = data.loc[
        data["sdp_label"].eq(SDP_LABEL)
        & data["settlement_period"].str.lower().eq(SETTLEMENT_PERIOD)
    ].copy()

    keep = pd.Series(False, index=selected.index)
    for power, energy in DAILY_CYCLING_BATTERY_CASES:
        keep |= np.isclose(selected["P_b_max"], power) & np.isclose(
            selected["E_max"], energy
        )
    selected = selected.loc[keep].copy()
    expected = len(DAILY_CYCLING_BATTERY_CASES) * 4 * 3
    if len(selected) != expected:
        raise ValueError(
            f"Expected {expected} daily-cycling "
            f"battery cases, found {len(selected)}."
        )
    selected["soc_cycle_hours"] = 24
    selected["soc_cycle_label"] = selected["battery_label"] + ", daily"
    return selected


def select_benchmark_daily_cases(data: pd.DataFrame) -> pd.DataFrame:
    selected = data.loc[
        np.isclose(data["lambda_PPA"], LAMBDA_PPA)
        & np.isclose(data["c_ppa"], C_PPA)
    ].copy()
    if len(selected) != len(DAILY_CYCLING_BATTERY_CASES):
        raise ValueError(
            "The daily-cycling benchmark does not contain all three batteries."
        )
    return selected


def three_day_summary_path(power: float, energy: float) -> Path:
    case_path = (
        THREE_DAY_ROOT
        / f"battery_p_{power:g}_e_{energy:g}"
        / "three_day_case_summary.csv"
    )
    if case_path.exists():
        return case_path
    legacy_path = THREE_DAY_ROOT / "three_day_case_summary.csv"
    if power == 20.0 and energy == 60.0 and legacy_path.exists():
        return legacy_path
    raise FileNotFoundError(
        f"Missing three-day result for {power:g} MW/{energy:g} MWh. "
        "Run run_soc_cycling_sensitivity.py on Sophia first."
    )


def load_three_day_cases() -> pd.DataFrame:
    frames = []
    for power, energy, _initial_energy in THREE_DAY_BATTERY_CASES:
        path = three_day_summary_path(power, energy)
        frame = pd.read_csv(path)
        if len(frame) != 1:
            raise ValueError(f"Expected one row in {path}.")
        row = frame.iloc[0]
        if not (
            np.isclose(row["P_b_max"], power)
            and np.isclose(row["E_max"], energy)
            and np.isclose(row["soc_cycle_hours"], SOC_CYCLE_HOURS)
        ):
            raise ValueError(f"Three-day result metadata does not match: {path}")
        frame["battery_label"] = f"{power:g} MW/{energy:g} MWh"
        frame["soc_cycle_label"] = f"{power:g} MW/{energy:g} MWh, 3-day"
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def add_npv_columns(data: pd.DataFrame) -> pd.DataFrame:
    settings = load_npv_settings()
    wind_energy = expected_wind_energy_mwh()
    annuity_factor = (
        1.0
        - (1.0 + settings.DISCOUNT_RATE)
        ** (-settings.PROJECT_LIFETIME_YEARS)
    ) / settings.DISCOUNT_RATE

    wind_capex = settings.WIND_CAPACITY_MW * settings.WIND_CAPEX_EUR_PER_MW
    wind_fixed_om = (
        settings.WIND_CAPACITY_MW * settings.WIND_FIXED_OM_EUR_PER_MW_YEAR
    )
    wind_variable_om = wind_energy * settings.WIND_VARIABLE_OM_EUR_PER_MWH
    battery_fixed_om_rate = 0.5 * (
        settings.BATTERY_FIXED_OM_2040_KEUR2020_PER_MW_YEAR
        + settings.BATTERY_FIXED_OM_2050_KEUR2020_PER_MW_YEAR
    ) * 1_000.0 * settings.EUR2020_TO_EUR2016

    data = data.copy()
    data["battery_capex_eur"] = (
        (
            settings.BATTERY_ENERGY_CAPEX_EUR_PER_MWH
            + settings.BATTERY_PROJECT_COST_EUR_PER_MWH
        )
        * data["E_max"]
        + settings.BATTERY_POWER_CAPEX_EUR_PER_MW * data["P_b_max"]
    )
    data["battery_fixed_om_eur_per_year"] = (
        battery_fixed_om_rate * data["P_b_max"]
    )
    data["battery_degradation_addback_eur_per_year"] = -data[
        "battery_degradation"
    ]
    data["annual_cash_profit_excluding_degradation_eur"] = (
        data["expected_profit"]
        + data["battery_degradation_addback_eur_per_year"]
    )
    data["annual_cash_flow_eur"] = (
        data["annual_cash_profit_excluding_degradation_eur"]
        - wind_fixed_om
        - wind_variable_om
        - data["battery_fixed_om_eur_per_year"]
    )
    data["wind_capex_eur"] = wind_capex
    data["total_capex_eur"] = wind_capex + data["battery_capex_eur"]
    data["discount_rate"] = settings.DISCOUNT_RATE
    data["project_lifetime_years"] = settings.PROJECT_LIFETIME_YEARS
    data["npv_eur"] = (
        -data["total_capex_eur"]
        + data["annual_cash_flow_eur"] * annuity_factor
        + settings.SALVAGE_VALUE_EUR
        / (1.0 + settings.DISCOUNT_RATE) ** settings.PROJECT_LIFETIME_YEARS
    )
    data["npv_meur"] = data["npv_eur"] / 1e6
    return data


def build_comparison() -> pd.DataFrame:
    all_daily = add_npv_columns(select_all_daily_cases())
    all_daily = all_daily.sort_values(
        ["c_ppa", "lambda_PPA", "P_b_max", "E_max"]
    ).reset_index(drop=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_daily.to_csv(DAILY_NPV_PATH, index=False, float_format="%.15g")
    print(f"Saved daily-cycling battery NPV results: {DAILY_NPV_PATH}")

    daily = select_benchmark_daily_cases(all_daily)
    three_day = load_three_day_cases()
    common_columns = sorted(set(daily.columns).intersection(three_day.columns))
    required = {
        "sdp_label",
        "settlement_period",
        "P_b_max",
        "E_max",
        "c_ppa",
        "lambda_PPA",
        "S_PPA",
        "battery_label",
        "soc_cycle_hours",
        "soc_cycle_label",
        "expected_profit",
        "battery_degradation",
        "ppa_shortfall_cost",
        "mfrr_up",
    }
    missing = sorted(required.difference(common_columns))
    if missing:
        raise ValueError(f"Comparison inputs are missing columns: {missing}")

    no_battery_three_day = daily.loc[
        np.isclose(daily["P_b_max"], 0.0)
        & np.isclose(daily["E_max"], 0.0),
        common_columns,
    ].copy()
    if len(no_battery_three_day) != 1:
        raise ValueError("Expected one no-battery benchmark result.")
    no_battery_three_day["soc_cycle_hours"] = SOC_CYCLE_HOURS
    no_battery_three_day["soc_cycle_label"] = "No battery, 3-day"

    data = pd.concat(
        [
            daily[common_columns],
            no_battery_three_day,
            three_day[common_columns],
        ],
        ignore_index=True,
    )
    # Daily cases already contain NPV columns, but the common-column selection
    # intentionally removes them before recalculating all five rows together.
    data = add_npv_columns(data)
    order = {
        "No battery, daily": 0,
        "No battery, 3-day": 1,
        "10 MW/30 MWh, daily": 2,
        "10 MW/30 MWh, 3-day": 3,
        "20 MW/60 MWh, daily": 4,
        "20 MW/60 MWh, 3-day": 5,
    }
    data["_order"] = data["soc_cycle_label"].map(order)
    if data["_order"].isna().any():
        raise ValueError("Unexpected comparison label was generated.")
    data = data.sort_values("_order").drop(columns="_order")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    data.to_csv(COMPARISON_PATH, index=False, float_format="%.15g")
    print(f"Saved comparison: {COMPARISON_PATH}")
    print(
        data[
            [
                "soc_cycle_label",
                "S_PPA",
                "expected_profit",
                "annual_cash_flow_eur",
                "npv_meur",
            ]
        ].to_string(index=False)
    )
    return data


def main() -> None:
    build_comparison()


if __name__ == "__main__":
    main()
