"""Export compact NPV appendix tables for all delivery profiles."""

from __future__ import annotations

from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "outputs" / "appendix_tables"

sys.path.insert(0, str(HERE))

from npv_analysis import calculate_npv_results  # noqa: E402
from settings import (  # noqa: E402
    C_PPA_VALUES,
    DISCOUNT_RATE,
    LAMBDA_PPA_VALUES,
    PROFILES,
    SETTLEMENT_PERIODS,
)


def _profile_slug(profile: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", profile.lower()).strip("_")


def build_wide_table(data: pd.DataFrame) -> pd.DataFrame:
    index_columns = ["sdp_label", "settlement_period", "lambda_PPA"]
    npv_wide = (
        data.pivot(
            index=index_columns,
            columns="c_ppa",
            values="npv_meur",
        )
        .reindex(columns=C_PPA_VALUES)
        .reset_index()
    )
    npv_wide.columns.name = None
    npv_wide = npv_wide.rename(
        columns={value: f"npv_c_ppa_{value:g}_meur2016" for value in C_PPA_VALUES}
    )
    irr_wide = (
        data.pivot(
            index=index_columns,
            columns="c_ppa",
            values="project_irr_pct",
        )
        .reindex(columns=C_PPA_VALUES)
        .reset_index()
    )
    irr_wide.columns.name = None
    irr_wide = irr_wide.rename(
        columns={value: f"irr_c_ppa_{value:g}_pct" for value in C_PPA_VALUES}
    )
    wide = npv_wide.merge(
        irr_wide,
        on=index_columns,
        how="inner",
        validate="one_to_one",
    )

    profile_order = {profile: index for index, profile in enumerate(PROFILES)}
    period_order = {
        period: index for index, period in enumerate(SETTLEMENT_PERIODS)
    }
    wide["_profile_order"] = wide["sdp_label"].map(profile_order)
    wide["_period_order"] = wide["settlement_period"].map(period_order)
    return wide.sort_values(
        ["_profile_order", "_period_order", "lambda_PPA"]
    ).drop(columns=["_profile_order", "_period_order"])


def _latex_table(profile_data: pd.DataFrame, profile: str) -> str:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        (
            r"\caption{Net present value and internal rate of return for the "
            + profile
            + r" delivery profile, using a real discount rate of "
            + f"{100.0 * DISCOUNT_RATE:g}"
            + r"\%. Each settlement-period column reports NPV "
            + r"[MEUR$_{2016}$] / IRR [\%].}"
        ),
        r"\label{tab:npv_irr_" + _profile_slug(profile) + r"}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}rrrrrr@{}}",
        r"\toprule",
        (
            r"$\lambda^{PPA}$ & $c^{PPA}$ "
            r"& Hourly & Daily & Weekly & Monthly \\"
        ),
        (
            r"\multicolumn{2}{c}{[EUR/MWh]} "
            r"& \multicolumn{4}{c}{NPV [MEUR$_{2016}$] / IRR [\%]} \\"
        ),
        r"\midrule",
    ]

    for price_index, lambda_ppa in enumerate(LAMBDA_PPA_VALUES):
        for c_ppa in C_PPA_VALUES:
            cells = []
            for period in SETTLEMENT_PERIODS:
                match = profile_data.loc[
                    profile_data["settlement_period"].eq(period)
                    & np.isclose(profile_data["lambda_PPA"], lambda_ppa)
                ]
                if len(match) != 1:
                    raise ValueError(
                        f"Expected one appendix row for {profile}, {period}, "
                        f"lambda_PPA={lambda_ppa:g}; found {len(match)}."
                    )
                row = match.iloc[0]
                npv = float(row[f"npv_c_ppa_{c_ppa:g}_meur2016"])
                irr = float(row[f"irr_c_ppa_{c_ppa:g}_pct"])
                cells.append(f"{npv:.2f} / {irr:.2f}")
            lines.append(
                f"{lambda_ppa:g} & {c_ppa:g} & "
                + " & ".join(cells)
                + r" \\"
            )
        if price_index < len(LAMBDA_PPA_VALUES) - 1:
            lines.append(r"\addlinespace")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular*}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def export_appendix_tables(data=None) -> list[Path]:
    if data is None:
        data = calculate_npv_results()
    wide = build_wide_table(data)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "npv_all_sdps_appendix_table.csv"
    wide.to_csv(csv_path, index=False, float_format="%.2f")

    tables = []
    for profile in PROFILES:
        profile_data = wide.loc[wide["sdp_label"].eq(profile)]
        if len(profile_data) != len(SETTLEMENT_PERIODS) * len(LAMBDA_PPA_VALUES):
            raise ValueError(
                f"Expected 16 appendix rows for {profile}, found {len(profile_data)}."
            )
        tables.append(_latex_table(profile_data, profile))

    tex_path = OUTPUT_DIR / "npv_irr_all_sdps_appendix_tables.tex"
    tex_path.write_text("\n\n".join(tables) + "\n", encoding="utf-8")
    return [csv_path, tex_path]


def main() -> None:
    outputs = export_appendix_tables()
    print("Created:")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
