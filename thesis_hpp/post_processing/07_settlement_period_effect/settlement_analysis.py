"""Select settlement-period cases and audit expected profit and shortfall."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
POST_PROCESSING_ROOT = HERE.parent
OUTPUT_ROOT = HERE / "outputs"
DATA_PATH = OUTPUT_ROOT / "settlement_period_analysis.csv"
EXISTING_HOURLY_DATA = (
    POST_PROCESSING_ROOT
    / "04_profit_decomposition"
    / "outputs"
    / "expected_profit_decomposition_split_imbalance.csv"
)
BATTERY_ANALYSIS_ROOT = POST_PROCESSING_ROOT / "06_battery_sensitivity"

sys.path.insert(0, str(BATTERY_ANALYSIS_ROOT))

from analysis import (  # noqa: E402
    calculate_profit_components,
    load_catalog,
    local_case_directory,
)
from sdp import get_delivery_profile  # noqa: E402

from settings import (  # noqa: E402
    CASE_WORKERS,
    C_PPA_VALUES,
    E_MAX,
    LAMBDA_PPA_VALUES,
    P_B_MAX,
    PROFILE,
    PROFILES,
    REBUILD_ANALYSIS_DATA,
    SETTLEMENT_PERIODS,
)


KEY_COLUMNS = [
    "sdp_label",
    "settlement_period",
    "P_b_max",
    "E_max",
    "c_ppa",
    "lambda_PPA",
]


def select_settlement_cases() -> pd.DataFrame:
    catalog = load_catalog()
    cases = catalog.loc[
        catalog["sdp_label"].isin(PROFILES)
        & catalog["settlement_period"].str.lower().isin(SETTLEMENT_PERIODS)
        & np.isclose(catalog["P_b_max"], P_B_MAX)
        & np.isclose(catalog["E_max"], E_MAX)
        & catalog["c_ppa"].isin(C_PPA_VALUES)
        & catalog["lambda_PPA"].isin(LAMBDA_PPA_VALUES)
    ].copy()
    cases["settlement_period"] = cases["settlement_period"].str.lower()

    duplicates = cases.duplicated(KEY_COLUMNS, keep=False)
    if duplicates.any():
        raise ValueError(
            "Duplicate settlement-period cases found:\n"
            + cases.loc[duplicates, KEY_COLUMNS + ["case_dir"]].to_string(
                index=False
            )
        )

    expected = (
        len(PROFILES)
        * len(SETTLEMENT_PERIODS)
        * len(C_PPA_VALUES)
        * len(LAMBDA_PPA_VALUES)
    )
    if len(cases) != expected:
        counts = cases.groupby("settlement_period").size()
        raise ValueError(
            f"Expected {expected} settlement-period cases, found {len(cases)}.\n"
            f"Counts by period:\n{counts.to_string()}"
        )

    period_order = {
        period: index for index, period in enumerate(SETTLEMENT_PERIODS)
    }
    profile_order = {profile: index for index, profile in enumerate(PROFILES)}
    cases["_period_order"] = cases["settlement_period"].map(period_order)
    cases["_profile_order"] = cases["sdp_label"].map(profile_order)
    return cases.sort_values(
        ["_profile_order", "c_ppa", "lambda_PPA", "_period_order"]
    ).reset_index(drop=True)


def load_contract_data() -> pd.DataFrame:
    columns = KEY_COLUMNS + [
        "S_PPA",
        "delivery_profile",
        "case_dir",
        "_period_order",
        "_profile_order",
    ]
    data = select_settlement_cases()[columns].copy()
    profile_full_load_hours = {
        name: float(sum(get_delivery_profile(str(name)).hourly_shape))
        for name in data["delivery_profile"].drop_duplicates()
    }
    data["contracted_energy_mwh_per_day"] = (
        data["S_PPA"] * data["delivery_profile"].map(profile_full_load_hours)
    )
    return data


def _load_cached_rows() -> pd.DataFrame:
    if REBUILD_ANALYSIS_DATA:
        return pd.DataFrame(columns=KEY_COLUMNS)

    frames = []
    for path in (DATA_PATH, EXISTING_HOURLY_DATA):
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        required = set(
            KEY_COLUMNS
            + [
                "expected_profit",
                "ppa_shortfall_cost",
                "objective_audit_difference",
            ]
        )
        if required.issubset(frame.columns):
            frame["settlement_period"] = (
                frame["settlement_period"].astype(str).str.lower()
            )
            frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=KEY_COLUMNS)

    cached = pd.concat(frames, ignore_index=True, sort=False)
    return cached.drop_duplicates(KEY_COLUMNS, keep="first")


def _annual_contracted_energy(case: pd.Series) -> float:
    case_dir = local_case_directory(case)
    stage_two = pd.read_csv(
        case_dir / "stage1_2_results.csv",
        usecols=["P_PPA"],
    )
    return float(stage_two["P_PPA"].sum())


def build_audited_settlement_data() -> pd.DataFrame:
    cases = select_settlement_cases()
    cached = _load_cached_rows()

    available = cases[KEY_COLUMNS].merge(
        cached,
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )
    missing_keys = cases[KEY_COLUMNS].merge(
        available[KEY_COLUMNS],
        on=KEY_COLUMNS,
        how="left",
        indicator=True,
    )
    missing_keys = missing_keys.loc[
        missing_keys["_merge"].eq("left_only"), KEY_COLUMNS
    ]
    missing_cases = cases.merge(
        missing_keys,
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )
    print(
        f"Reusing {len(available)} audited cases; "
        f"{len(missing_cases)} cases remain."
    )

    rows: list[dict[str, float | str] | None] = [None] * len(missing_cases)

    def run_case(index: int) -> tuple[int, dict[str, float | str]]:
        return index, calculate_profit_components(missing_cases.iloc[index])

    if len(missing_cases):
        with ThreadPoolExecutor(max_workers=CASE_WORKERS) as executor:
            futures = {
                executor.submit(run_case, index): index
                for index in range(len(missing_cases))
            }
            completed = 0
            for future in as_completed(futures):
                index, result = future.result()
                rows[index] = result
                completed += 1
                case = missing_cases.iloc[index]
                print(
                    f"[{completed}/{len(missing_cases)} completed] "
                    f"{case['sdp_label']}, "
                    f"{case['settlement_period']}, "
                    f"c_ppa={case['c_ppa']:g}, "
                    f"lambda_PPA={case['lambda_PPA']:g}"
                )

    if any(row is None for row in rows):
        raise RuntimeError("One or more settlement-period cases did not finish.")

    calculated = pd.DataFrame(rows)
    combined = pd.concat([available, calculated], ignore_index=True, sort=False)
    combined = combined.drop_duplicates(KEY_COLUMNS, keep="first")

    value_columns = [
        column
        for column in combined.columns
        if column
        not in KEY_COLUMNS
        + [
            "S_PPA",
            "delivery_profile",
            "case_dir",
            "_period_order",
            "_profile_order",
        ]
    ]
    requested = cases[
        KEY_COLUMNS
        + [
            "S_PPA",
            "delivery_profile",
            "case_dir",
            "_period_order",
            "_profile_order",
        ]
    ].merge(
        combined[KEY_COLUMNS + value_columns],
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    if requested[["expected_profit", "ppa_shortfall_cost"]].isna().any().any():
        raise RuntimeError("Audited settlement-period data remain incomplete.")

    requested["annual_contracted_energy_mwh"] = [
        _annual_contracted_energy(row)
        for _, row in requested.iterrows()
    ]
    requested["expected_shortfall_mwh"] = (
        -requested["ppa_shortfall_cost"] / requested["c_ppa"]
    )
    denominator = requested["annual_contracted_energy_mwh"]
    requested["expected_shortfall_rate_pct"] = np.where(
        denominator > 1e-9,
        100.0 * requested["expected_shortfall_mwh"] / denominator,
        np.nan,
    )

    requested = requested.sort_values(
        ["_profile_order", "c_ppa", "lambda_PPA", "_period_order"]
    ).reset_index(drop=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    requested.to_csv(DATA_PATH, index=False, float_format="%.15g")
    print(f"Saved audited settlement-period data: {DATA_PATH}")
    return requested


def main() -> None:
    build_audited_settlement_data()


if __name__ == "__main__":
    main()
