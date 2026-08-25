"""Compare expected revenue and cost components across battery cases."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
POST_PROCESSING_ROOT = HERE.parent
OUTPUT_ROOT = HERE / "outputs"
DATA_PATH = OUTPUT_ROOT / "battery_revenue_decomposition.csv"

sys.path.insert(0, str(POST_PROCESSING_ROOT))

from common.plotting_style import get_palette  # noqa: E402
from analysis import (  # noqa: E402
    COMPONENT_KEYS,
    COMPONENT_LABELS,
    battery_label,
    calculate_profit_components,
    load_catalog,
    profile_slug,
    select_contract_cases,
)
from config import (  # noqa: E402
    BATTERY_CASES,
    DECOMPOSITION_C_PPA_VALUES,
    DECOMPOSITION_LAMBDA_PPA,
    PALETTE_NAME,
    PROFILE,
    REBUILD_DECOMPOSITION_DATA,
    SETTLEMENT_PERIOD,
)


def compact_eur_label(value: float) -> str:
    absolute = abs(value)
    sign = "-" if value < 0 else ""
    if absolute >= 1_000_000:
        return f"{sign}{absolute / 1_000_000:.2f}M"
    if absolute >= 1_000:
        return f"{sign}{absolute / 1_000:.0f}k"
    return f"{value:.0f}"


def cache_matches(data: pd.DataFrame) -> bool:
    required = {
        "settlement_period",
        "P_b_max",
        "E_max",
        "c_ppa",
        "lambda_PPA",
        "sdp_label",
        *COMPONENT_KEYS,
    }
    expected_rows = len(BATTERY_CASES) * len(DECOMPOSITION_C_PPA_VALUES)
    if not required.issubset(data.columns) or len(data) != expected_rows:
        return False
    keys = {
        (
            float(row.c_ppa),
            float(row.P_b_max),
            float(row.E_max),
        )
        for row in data.itertuples(index=False)
    }
    expected_keys = {
        (float(c_ppa), float(power), float(energy))
        for c_ppa in DECOMPOSITION_C_PPA_VALUES
        for power, energy in BATTERY_CASES
    }
    return (
        keys == expected_keys
        and data["settlement_period"].str.lower().eq(
            SETTLEMENT_PERIOD.lower()
        ).all()
        and data["sdp_label"].eq(PROFILE).all()
        and np.isclose(data["lambda_PPA"], DECOMPOSITION_LAMBDA_PPA).all()
    )


def build_decomposition() -> pd.DataFrame:
    if DATA_PATH.exists() and not REBUILD_DECOMPOSITION_DATA:
        cached = pd.read_csv(DATA_PATH)
        if cache_matches(cached):
            print(f"Loaded audited decomposition data: {DATA_PATH}")
            return cached
        print("Cached decomposition does not match the selected cases; rebuilding.")

    catalog = load_catalog()
    cases = select_contract_cases(catalog, PROFILE)
    cases = cases.loc[
        cases["c_ppa"].isin(DECOMPOSITION_C_PPA_VALUES)
        & np.isclose(cases["lambda_PPA"], DECOMPOSITION_LAMBDA_PPA)
    ].copy()
    expected_rows = len(BATTERY_CASES) * len(DECOMPOSITION_C_PPA_VALUES)
    if len(cases) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} decomposition cases, "
            f"found {len(cases)}."
        )

    rows = []
    for case_index, (_, case) in enumerate(cases.iterrows(), start=1):
        print(
            f"[{case_index}/{len(cases)}] "
            f"{battery_label(case['P_b_max'], case['E_max'])}"
        )
        rows.append(calculate_profit_components(case))

    data = pd.DataFrame(rows).sort_values(
        ["c_ppa", "P_b_max", "E_max"]
    )
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(DATA_PATH, index=False, float_format="%.15g")
    print(f"Saved audited decomposition data: {DATA_PATH}")
    return data


def plot_decomposition(
    data: pd.DataFrame,
    c_ppa: float,
) -> list[Path]:
    palette = get_palette(PALETTE_NAME)
    if len(palette) < len(BATTERY_CASES):
        raise ValueError(
            f"Palette '{PALETTE_NAME}' does not contain enough colors."
        )

    selected = data.loc[np.isclose(data["c_ppa"], c_ppa)]
    ordered_rows = []
    for power, energy in BATTERY_CASES:
        match = selected.loc[
            np.isclose(selected["P_b_max"], power)
            & np.isclose(selected["E_max"], energy)
        ]
        if len(match) != 1:
            raise ValueError(
                f"Expected one decomposition row for {power:g} MW/"
                f"{energy:g} MWh, found {len(match)}."
            )
        ordered_rows.append(match.iloc[0])
    ordered = pd.DataFrame(ordered_rows)

    values_meur = ordered[COMPONENT_KEYS].to_numpy(dtype=float) / 1e6
    y_min = min(float(np.nanmin(values_meur)), 0.0)
    y_max = max(float(np.nanmax(values_meur)), 0.1)
    y_span = y_max - y_min

    fig, ax = plt.subplots(figsize=(12.2, 6.9))
    x_positions = np.arange(len(COMPONENT_KEYS), dtype=float)
    group_width = 0.78
    bar_width = group_width / len(BATTERY_CASES)

    for battery_index, ((power, energy), (_, row)) in enumerate(
        zip(BATTERY_CASES, ordered.iterrows())
    ):
        offset = (
            battery_index - (len(BATTERY_CASES) - 1) / 2
        ) * bar_width
        values_eur = row[COMPONENT_KEYS].to_numpy(dtype=float)
        battery_values_meur = values_eur / 1e6
        bars = ax.bar(
            x_positions + offset,
            battery_values_meur,
            width=bar_width * 0.92,
            color=palette[battery_index],
            edgecolor="white",
            linewidth=0.4,
            label=battery_label(power, energy),
        )

        for bar, value_eur, value_meur in zip(
            bars, values_eur, battery_values_meur
        ):
            if abs(value_eur) < 1_000.0:
                continue
            vertical_offset = 0.012 * y_span
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value_meur
                + (vertical_offset if value_meur >= 0 else -vertical_offset),
                compact_eur_label(value_eur),
                ha="center",
                va="bottom" if value_meur >= 0 else "top",
                rotation=55,
                fontsize=8.0,
            )

    ax.set_title(
        "Expected annual revenues and costs by battery configuration\n"
        f"{PROFILE}, "
        rf"$\lambda^{{PPA}}={DECOMPOSITION_LAMBDA_PPA:g}$ EUR/MWh, "
        rf"$c^{{PPA}}={c_ppa:g}$ EUR/MWh, "
        f"{SETTLEMENT_PERIOD.capitalize()} PPA deviation settlement",
        fontsize=12,
    )
    ax.set_ylabel("Expected annual value [MEUR/year]", fontsize=11.5)
    ax.set_xticks(
        x_positions,
        [COMPONENT_LABELS[key] for key in COMPONENT_KEYS],
        rotation=27,
        ha="right",
    )
    ax.set_ylim(y_min - 0.13 * y_span, y_max + 0.17 * y_span)
    ax.axhline(0.0, color="black", linewidth=0.9)
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=True, fontsize=10.5)
    ax.tick_params(axis="both", labelsize=9)
    fig.tight_layout()

    output_dir = OUTPUT_ROOT / "revenue_decomposition"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / (
        f"battery_revenue_decomposition_{profile_slug(PROFILE)}_"
        f"lambda_{DECOMPOSITION_LAMBDA_PPA:g}_"
        f"c_ppa_{c_ppa:g}"
    )
    outputs = []
    for suffix in ("pdf", "png"):
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    return outputs


def main() -> None:
    data = build_decomposition()
    outputs = []
    for c_ppa in DECOMPOSITION_C_PPA_VALUES:
        outputs.extend(plot_decomposition(data, float(c_ppa)))
    maximum_difference = data["objective_audit_difference"].abs().max()
    print(
        "Maximum reconstructed-objective difference: "
        f"{maximum_difference:.6g} EUR"
    )
    print("Created:")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
