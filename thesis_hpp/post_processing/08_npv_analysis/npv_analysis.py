"""Calculate project NPV from the audited expected-profit results."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
POST_PROCESSING_ROOT = HERE.parent
THESIS_ROOT = POST_PROCESSING_ROOT.parent
SOURCE_DATA_PATH = (
    POST_PROCESSING_ROOT
    / "07_settlement_period_effect"
    / "outputs"
    / "settlement_period_analysis.csv"
)
INPUT_ROOT = THESIS_ROOT / "inputs" / "Input_files_created"
OUTPUT_ROOT = HERE / "outputs"
NPV_RESULTS_PATH = OUTPUT_ROOT / "npv_results.csv"
NPV_COUNTS_PATH = OUTPUT_ROOT / "positive_npv_case_counts.csv"

sys.path.insert(0, str(HERE))

from settings import (  # noqa: E402
    BATTERY_ENERGY_CAPEX_EUR_PER_MWH,
    BATTERY_FIXED_OM_2040_KEUR2020_PER_MW_YEAR,
    BATTERY_FIXED_OM_2050_KEUR2020_PER_MW_YEAR,
    BATTERY_POWER_CAPEX_EUR_PER_MW,
    BATTERY_PROJECT_COST_EUR_PER_MWH,
    BATTERY_VARIABLE_OM_EUR_PER_MWH,
    C_PPA_VALUES,
    DISCOUNT_RATE,
    E_MAX,
    EUR2020_TO_EUR2016,
    INCLUDE_BATTERY_REPLACEMENT,
    LAMBDA_PPA_VALUES,
    P_B_MAX,
    PROFILE,
    PROFILES,
    PROJECT_LIFETIME_YEARS,
    SALVAGE_VALUE_EUR,
    SETTLEMENT_PERIODS,
    WIND_CAPACITY_MW,
    WIND_CAPEX_EUR_PER_MW,
    WIND_FIXED_OM_EUR_PER_MW_YEAR,
    WIND_VARIABLE_OM_EUR_PER_MWH,
)


KEY_COLUMNS = [
    "sdp_label",
    "settlement_period",
    "P_b_max",
    "E_max",
    "c_ppa",
    "lambda_PPA",
]


def calculate_project_irr(
    initial_capex: float,
    annual_cash_flow: float,
    lifetime_years: int,
    terminal_value: float = 0.0,
) -> float:
    """Return the annual IRR for a constant annual cash-flow series."""
    if initial_capex <= 0.0 or lifetime_years <= 0:
        raise ValueError("IRR requires positive CAPEX and project lifetime.")
    if annual_cash_flow <= 0.0 and terminal_value <= 0.0:
        return float("nan")

    def project_npv(rate: float) -> float:
        if np.isclose(rate, 0.0, atol=1e-14):
            return (
                -initial_capex
                + annual_cash_flow * lifetime_years
                + terminal_value
            )
        discount_factor = 1.0 + rate
        annuity_factor = (
            1.0 - discount_factor ** (-lifetime_years)
        ) / rate
        return (
            -initial_capex
            + annual_cash_flow * annuity_factor
            + terminal_value * discount_factor ** (-lifetime_years)
        )

    lower = -0.9999
    upper = 1.0
    lower_value = project_npv(lower)
    upper_value = project_npv(upper)
    while lower_value * upper_value > 0.0 and upper < 100.0:
        upper = 2.0 * upper + 1.0
        upper_value = project_npv(upper)
    if lower_value * upper_value > 0.0:
        return float("nan")

    for _ in range(200):
        midpoint = 0.5 * (lower + upper)
        midpoint_value = project_npv(midpoint)
        if abs(midpoint_value) < 1e-6:
            return midpoint
        if lower_value * midpoint_value <= 0.0:
            upper = midpoint
        else:
            lower = midpoint
            lower_value = midpoint_value
    return 0.5 * (lower + upper)


def _load_source_cases() -> pd.DataFrame:
    if not SOURCE_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing {SOURCE_DATA_PATH}. Run "
            "post_processing/07_settlement_period_effect/"
            "settlement_analysis.py first."
        )

    data = pd.read_csv(SOURCE_DATA_PATH)
    required = set(KEY_COLUMNS + ["expected_profit", "battery_degradation"])
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(
            f"{SOURCE_DATA_PATH} is missing columns: {', '.join(missing)}"
        )

    data["settlement_period"] = (
        data["settlement_period"].astype(str).str.lower()
    )
    selected = data.loc[
        data["sdp_label"].isin(PROFILES)
        & data["settlement_period"].isin(SETTLEMENT_PERIODS)
        & np.isclose(data["P_b_max"], P_B_MAX)
        & np.isclose(data["E_max"], E_MAX)
        & data["c_ppa"].isin(C_PPA_VALUES)
        & data["lambda_PPA"].isin(LAMBDA_PPA_VALUES)
    ].copy()

    expected = (
        len(PROFILES)
        * len(SETTLEMENT_PERIODS)
        * len(C_PPA_VALUES)
        * len(LAMBDA_PPA_VALUES)
    )
    if len(selected) != expected:
        counts = selected.groupby("settlement_period").size()
        raise ValueError(
            f"Expected {expected} NPV cases, found {len(selected)}.\n"
            f"Counts by settlement period:\n{counts.to_string()}"
        )
    if selected.duplicated(KEY_COLUMNS).any():
        raise ValueError("Duplicate NPV cases were found in the source data.")

    return selected


def expected_annual_wind_energy_mwh() -> float:
    """Return probability-weighted gross available wind energy."""
    wind = pd.read_csv(
        INPUT_ROOT / "wind_realization.csv",
        usecols=["t", "omega", "xi", "wind"],
    )
    omega = pd.read_csv(
        INPUT_ROOT / "omega_probabilities.csv",
        usecols=["omega", "prob"],
    ).rename(columns={"prob": "pi_omega"})
    xi = pd.read_csv(
        INPUT_ROOT / "xi_probabilities.csv",
        usecols=["omega", "xi", "prob"],
    ).rename(columns={"prob": "pi_xi_given_omega"})

    if not np.isclose(omega["pi_omega"].sum(), 1.0):
        raise ValueError("Omega probabilities do not sum to one.")
    xi_sums = xi.groupby("omega")["pi_xi_given_omega"].sum()
    if not np.allclose(xi_sums.to_numpy(), 1.0):
        raise ValueError("Conditional xi probabilities do not sum to one.")

    weighted = wind.merge(xi, on=["omega", "xi"], validate="many_to_one")
    weighted = weighted.merge(omega, on="omega", validate="many_to_one")
    return float(
        (
            weighted["wind"]
            * weighted["pi_omega"]
            * weighted["pi_xi_given_omega"]
        ).sum()
    )


def calculate_npv_results() -> pd.DataFrame:
    data = _load_source_cases()

    wind_energy = expected_annual_wind_energy_mwh()
    wind_capex = WIND_CAPACITY_MW * WIND_CAPEX_EUR_PER_MW
    wind_fixed_om = WIND_CAPACITY_MW * WIND_FIXED_OM_EUR_PER_MW_YEAR
    wind_variable_om = wind_energy * WIND_VARIABLE_OM_EUR_PER_MWH

    battery_capex = (
        (
            BATTERY_ENERGY_CAPEX_EUR_PER_MWH
            + BATTERY_PROJECT_COST_EUR_PER_MWH
        )
        * E_MAX
        + BATTERY_POWER_CAPEX_EUR_PER_MW * P_B_MAX
    )
    battery_fixed_om_2045 = 0.5 * (
        BATTERY_FIXED_OM_2040_KEUR2020_PER_MW_YEAR
        + BATTERY_FIXED_OM_2050_KEUR2020_PER_MW_YEAR
    )
    battery_fixed_om = (
        battery_fixed_om_2045
        * 1_000.0
        * EUR2020_TO_EUR2016
        * P_B_MAX
    )
    battery_variable_om = BATTERY_VARIABLE_OM_EUR_PER_MWH * 0.0

    if INCLUDE_BATTERY_REPLACEMENT:
        raise NotImplementedError(
            "Battery replacement is intentionally excluded from this analysis."
        )

    annuity_factor = (
        1.0 - (1.0 + DISCOUNT_RATE) ** (-PROJECT_LIFETIME_YEARS)
    ) / DISCOUNT_RATE
    total_capex = wind_capex + battery_capex
    discounted_salvage = SALVAGE_VALUE_EUR / (
        1.0 + DISCOUNT_RATE
    ) ** PROJECT_LIFETIME_YEARS
    required_annual_cash_flow = (
        total_capex - discounted_salvage
    ) / annuity_factor

    # The decomposition stores degradation as a negative cost. It is added
    # back because the complete battery CAPEX is included in the NPV.
    if (data["battery_degradation"] > 1e-6).any():
        raise ValueError(
            "Expected battery_degradation to use the negative-cost convention."
        )
    data["battery_degradation_addback_eur_per_year"] = -data[
        "battery_degradation"
    ]
    data["wind_capex_eur"] = wind_capex
    data["battery_capex_eur"] = battery_capex
    data["total_capex_eur"] = total_capex
    data["expected_gross_wind_energy_mwh_per_year"] = wind_energy
    data["wind_fixed_om_eur_per_year"] = wind_fixed_om
    data["wind_variable_om_eur_per_year"] = wind_variable_om
    data["battery_fixed_om_eur_per_year"] = battery_fixed_om
    data["battery_variable_om_eur_per_year"] = battery_variable_om
    data["annual_cash_flow_eur"] = (
        data["expected_profit"]
        + data["battery_degradation_addback_eur_per_year"]
        - wind_fixed_om
        - wind_variable_om
        - battery_fixed_om
        - battery_variable_om
    )
    data["discount_rate"] = DISCOUNT_RATE
    data["project_lifetime_years"] = PROJECT_LIFETIME_YEARS
    data["annuity_factor"] = annuity_factor
    data["present_value_annual_cash_flows_eur"] = (
        data["annual_cash_flow_eur"] * annuity_factor
    )
    data["salvage_value_eur"] = SALVAGE_VALUE_EUR
    data["npv_eur"] = (
        -total_capex
        + data["present_value_annual_cash_flows_eur"]
        + discounted_salvage
    )
    data["npv_meur"] = data["npv_eur"] / 1e6
    data["project_irr_decimal"] = data["annual_cash_flow_eur"].map(
        lambda annual_cash_flow: calculate_project_irr(
            initial_capex=total_capex,
            annual_cash_flow=float(annual_cash_flow),
            lifetime_years=PROJECT_LIFETIME_YEARS,
            terminal_value=SALVAGE_VALUE_EUR,
        )
    )
    data["project_irr_pct"] = 100.0 * data["project_irr_decimal"]
    data["irr_exceeds_discount_rate"] = (
        data["project_irr_decimal"] > DISCOUNT_RATE
    )
    data["required_annual_cash_flow_eur"] = required_annual_cash_flow
    data[
        "break_even_cash_profit_excluding_battery_degradation_eur_per_year"
    ] = (
        required_annual_cash_flow
        + wind_fixed_om
        + wind_variable_om
        + battery_fixed_om
        + battery_variable_om
    )
    data[
        "break_even_reported_profit_including_battery_degradation_eur_per_year"
    ] = (
        required_annual_cash_flow
        - data["battery_degradation_addback_eur_per_year"]
        + wind_fixed_om
        + wind_variable_om
        + battery_fixed_om
        + battery_variable_om
    )
    data["positive_npv"] = data["npv_eur"] > 0.0

    period_order = {
        period: index for index, period in enumerate(SETTLEMENT_PERIODS)
    }
    profile_order = {profile: index for index, profile in enumerate(PROFILES)}
    data["_npv_profile_order"] = data["sdp_label"].map(profile_order)
    data["_npv_period_order"] = data["settlement_period"].map(period_order)
    data = data.sort_values(
        [
            "_npv_profile_order",
            "_npv_period_order",
            "c_ppa",
            "lambda_PPA",
        ]
    ).drop(columns=["_npv_profile_order", "_npv_period_order"])

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    data.to_csv(NPV_RESULTS_PATH, index=False, float_format="%.15g")

    counts = (
        data.groupby(["sdp_label", "settlement_period"], as_index=False)
        .agg(
            positive_npv_cases=("positive_npv", "sum"),
            total_cases=("positive_npv", "size"),
            minimum_npv_meur=("npv_meur", "min"),
            maximum_npv_meur=("npv_meur", "max"),
        )
    )
    counts["_profile_order"] = counts["sdp_label"].map(profile_order)
    counts["_order"] = counts["settlement_period"].map(period_order)
    counts = counts.sort_values(["_profile_order", "_order"]).drop(
        columns=["_profile_order", "_order"]
    )
    counts.to_csv(NPV_COUNTS_PATH, index=False, float_format="%.15g")

    print(f"Expected gross wind energy: {wind_energy:,.2f} MWh/year")
    print(f"Total initial CAPEX: {total_capex / 1e6:,.3f} MEUR2016")
    print(
        "Required net annual cash flow: "
        f"{required_annual_cash_flow / 1e6:,.3f} MEUR2016/year"
    )
    print(f"Saved NPV results: {NPV_RESULTS_PATH}")
    print(f"Saved positive-case counts: {NPV_COUNTS_PATH}")
    print("\nPositive-NPV cases by settlement period:")
    print(counts.to_string(index=False))
    return data


def main() -> None:
    calculate_npv_results()


if __name__ == "__main__":
    main()
