"""Export settlement-period profit, contracting and shortfall tables."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "outputs" / "appendix_tables"

from settings import (  # noqa: E402
    C_PPA_VALUES,
    LAMBDA_PPA_VALUES,
    PROFILES,
    SETTLEMENT_PERIODS,
)
from settlement_analysis import build_audited_settlement_data  # noqa: E402


TABLE_METRICS = (
    "expected_profit",
    "S_PPA",
    "expected_shortfall_rate_pct",
)
PERIOD_LABELS = {
    "hourly": "Hourly",
    "daily": "Daily",
    "weekly": "Weekly",
    "monthly": "Monthly",
}


def _profile_slug(profile: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", profile.lower()).strip("_")


def _validate_data(data: pd.DataFrame) -> None:
    expected_rows = (
        len(PROFILES)
        * len(SETTLEMENT_PERIODS)
        * len(LAMBDA_PPA_VALUES)
        * len(C_PPA_VALUES)
    )
    if len(data) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} settlement cases, found {len(data)}."
        )

    key_columns = [
        "sdp_label",
        "settlement_period",
        "lambda_PPA",
        "c_ppa",
    ]
    duplicates = data.duplicated(key_columns, keep=False)
    if duplicates.any():
        raise ValueError(
            "Duplicate settlement cases found:\n"
            + data.loc[duplicates, key_columns].to_string(index=False)
        )

    if data[list(TABLE_METRICS[:2])].isna().any().any():
        raise ValueError("Expected profit or optimal PPA capacity is missing.")

    positive_contract = data["annual_contracted_energy_mwh"] > 1e-9
    calculated_rate = (
        100.0
        * data.loc[positive_contract, "expected_shortfall_mwh"]
        / data.loc[positive_contract, "annual_contracted_energy_mwh"]
    )
    reported_rate = data.loc[
        positive_contract, "expected_shortfall_rate_pct"
    ]
    if not np.allclose(calculated_rate, reported_rate, rtol=1e-9, atol=1e-9):
        raise ValueError(
            "The reported shortfall rate is inconsistent with annual energy."
        )


def build_export_data() -> pd.DataFrame:
    data = build_audited_settlement_data().copy()
    _validate_data(data)

    selected_columns = [
        "sdp_label",
        "settlement_period",
        "lambda_PPA",
        "c_ppa",
        "S_PPA",
        "expected_profit",
        "expected_shortfall_rate_pct",
    ]
    data = data[selected_columns].copy()

    profile_order = {profile: index for index, profile in enumerate(PROFILES)}
    period_order = {
        period: index for index, period in enumerate(SETTLEMENT_PERIODS)
    }
    data["_profile_order"] = data["sdp_label"].map(profile_order)
    data["_period_order"] = data["settlement_period"].map(period_order)
    return data.sort_values(
        ["_profile_order", "lambda_PPA", "c_ppa", "_period_order"]
    ).drop(columns=["_profile_order", "_period_order"])


def build_wide_table(data: pd.DataFrame) -> pd.DataFrame:
    index_columns = ["sdp_label", "lambda_PPA", "c_ppa"]
    frames = []
    for metric in TABLE_METRICS:
        frame = data.pivot(
            index=index_columns,
            columns="settlement_period",
            values=metric,
        ).reindex(columns=SETTLEMENT_PERIODS)
        frame.columns = [f"{period}__{metric}" for period in frame.columns]
        frames.append(frame)

    wide = pd.concat(frames, axis=1).reset_index()
    ordered_columns = list(index_columns)
    for period in SETTLEMENT_PERIODS:
        ordered_columns.extend(
            f"{period}__{metric}" for metric in TABLE_METRICS
        )
    return wide[ordered_columns]


def _format_cell(row: pd.Series, period: str) -> str:
    expected_profit = float(row[f"{period}__expected_profit"]) / 1e6
    capacity = float(row[f"{period}__S_PPA"])
    rate = row[f"{period}__expected_shortfall_rate_pct"]
    rate_text = "--" if pd.isna(rate) else f"{float(rate):.1f}"
    return f"{expected_profit:.2f} / {capacity:.2f} / {rate_text}"


def _latex_profile_table(wide: pd.DataFrame, profile: str) -> str:
    table = wide.loc[wide["sdp_label"].eq(profile)].copy()
    expected_rows = len(LAMBDA_PPA_VALUES) * len(C_PPA_VALUES)
    if len(table) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows for {profile}, found {len(table)}."
        )

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.5pt}",
        (
            r"\caption{Expected annual profit, optimal contracted PPA "
            r"capacity and expected shortfall rate for the "
            + profile
            + r" profile under different PPA deviation settlement periods. "
            r"Each settlement-period column reports expected profit "
            r"[MEUR/year] / $S^{PPA,*}$ [MW] / shortfall rate [\%]. "
            r"Expected profit excludes the CVaR valuation term.}"
        ),
        r"\label{tab:settlement_period_" + _profile_slug(profile) + r"}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}rrrrrr@{}}",
        r"\toprule",
        (
            r"$\lambda^{PPA}$ & $c^{PPA}$ & "
            + " & ".join(PERIOD_LABELS[period] for period in SETTLEMENT_PERIODS)
            + r" \\"
        ),
        (
            r"\multicolumn{2}{c}{[EUR/MWh]} & "
            r"\multicolumn{4}{c}{Profit [MEUR/year] / "
            r"$S^{PPA,*}$ [MW] / shortfall [\%]} \\"
        ),
        r"\midrule",
    ]

    for price_index, lambda_ppa in enumerate(LAMBDA_PPA_VALUES):
        for c_ppa in C_PPA_VALUES:
            match = table.loc[
                np.isclose(table["lambda_PPA"], lambda_ppa)
                & np.isclose(table["c_ppa"], c_ppa)
            ]
            if len(match) != 1:
                raise ValueError(
                    f"Expected one row for {profile}, "
                    f"lambda_PPA={lambda_ppa:g}, c_ppa={c_ppa:g}; "
                    f"found {len(match)}."
                )
            row = match.iloc[0]
            cells = [_format_cell(row, period) for period in SETTLEMENT_PERIODS]
            lines.append(
                f"{lambda_ppa:g} & {c_ppa:g} & "
                + " & ".join(cells)
                + r" \\"
            )
        if price_index < len(LAMBDA_PPA_VALUES) - 1:
            lines.append(r"\addlinespace")

    lines.extend([r"\bottomrule", r"\end{tabular*}", r"\end{table}", ""])
    return "\n".join(lines)


def export_appendix_tables(data: pd.DataFrame | None = None) -> list[Path]:
    if data is None:
        data = build_export_data()
    else:
        data = data.copy()
        _validate_data(data)

    wide = build_wide_table(data)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    long_csv = OUTPUT_DIR / "settlement_period_all_sdps_long.csv"
    wide_csv = OUTPUT_DIR / "settlement_period_all_sdps_wide.csv"
    data.to_csv(long_csv, index=False, float_format="%.15g")
    wide.to_csv(wide_csv, index=False, float_format="%.15g")

    table_paths = []
    combined_tables = []
    for profile in PROFILES:
        table_text = _latex_profile_table(wide, profile)
        combined_tables.append(table_text)
        table_path = OUTPUT_DIR / (
            f"settlement_period_{_profile_slug(profile)}.tex"
        )
        table_path.write_text(table_text, encoding="utf-8")
        table_paths.append(table_path)

    combined_path = OUTPUT_DIR / "settlement_period_all_sdps_tables.tex"
    combined_path.write_text("\n".join(combined_tables), encoding="utf-8")
    return [long_csv, wide_csv, *table_paths, combined_path]


def main() -> None:
    outputs = export_appendix_tables()
    print("Created:")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
