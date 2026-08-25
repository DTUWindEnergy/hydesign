"""Plot expected annual profit across battery configurations."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
POST_PROCESSING_ROOT = HERE.parent
OUTPUT_ROOT = HERE / "outputs"
DATA_PATH = OUTPUT_ROOT / "battery_expected_profit_cases.csv"
EXISTING_DECOMPOSITION_PATH = (
    POST_PROCESSING_ROOT
    / "04_profit_decomposition"
    / "outputs"
    / "expected_profit_decomposition_split_imbalance.csv"
)

sys.path.insert(0, str(POST_PROCESSING_ROOT))

from common.plotting_style import get_palette  # noqa: E402
from analysis import (  # noqa: E402
    battery_label,
    calculate_profit_components,
    load_catalog,
    profile_slug,
    select_contract_cases,
)
from config import (  # noqa: E402
    BATTERY_CASES,
    C_PPA_VALUES,
    EXPECTED_PROFIT_CASE_WORKERS,
    LAMBDA_PPA_VALUES,
    PALETTE_NAME,
    PROFILE,
    REBUILD_EXPECTED_PROFIT_DATA,
    SETTLEMENT_PERIOD,
)


KEY_COLUMNS = [
    "settlement_period",
    "P_b_max",
    "E_max",
    "c_ppa",
    "lambda_PPA",
    "sdp_label",
]


def compact_eur_label(value: float) -> str:
    absolute = abs(value)
    sign = "-" if value < 0 else ""
    if absolute >= 1_000_000:
        return f"{sign}{absolute / 1_000_000:.2f}M"
    if absolute >= 1_000:
        return f"{sign}{absolute / 1_000:.0f}k"
    return f"{value:.0f}"


def build_expected_profit_data(profile: str | None = PROFILE) -> pd.DataFrame:
    catalog = load_catalog()
    cases = select_contract_cases(catalog, profile)

    cached_frames = []
    if not REBUILD_EXPECTED_PROFIT_DATA:
        for path in (DATA_PATH, EXISTING_DECOMPOSITION_PATH):
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            required = set(KEY_COLUMNS + ["expected_profit"])
            if required.issubset(frame.columns):
                cached_frames.append(frame)

    if cached_frames:
        cached = pd.concat(cached_frames, ignore_index=True, sort=False)
        cached = cached.drop_duplicates(KEY_COLUMNS, keep="first")
        available = cases[KEY_COLUMNS].merge(
            cached,
            on=KEY_COLUMNS,
            how="inner",
            validate="one_to_one",
        )
    else:
        available = pd.DataFrame(columns=KEY_COLUMNS)

    missing_keys = cases[KEY_COLUMNS].merge(
        available[KEY_COLUMNS],
        on=KEY_COLUMNS,
        how="left",
        indicator=True,
    )
    missing_keys = missing_keys.loc[
        missing_keys["_merge"] == "left_only", KEY_COLUMNS
    ]
    missing_cases = cases.merge(
        missing_keys,
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )
    print(
        f"Reusing {len(available)} audited expected-profit cases; "
        f"{len(missing_cases)} cases remain."
    )

    rows: list[dict[str, float | str] | None] = [None] * len(missing_cases)

    def run_case(index: int) -> tuple[int, dict[str, float | str]]:
        return index, calculate_profit_components(missing_cases.iloc[index])

    if len(missing_cases):
        with ThreadPoolExecutor(
            max_workers=EXPECTED_PROFIT_CASE_WORKERS
        ) as executor:
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
                    f"c_ppa={case['c_ppa']:g}, "
                    f"lambda_PPA={case['lambda_PPA']:g}, "
                    f"{battery_label(case['P_b_max'], case['E_max'])}"
                )

    if any(row is None for row in rows):
        raise RuntimeError("One or more expected-profit cases did not finish.")

    calculated = pd.DataFrame(rows)
    requested = pd.concat([available, calculated], ignore_index=True, sort=False)
    requested = cases[KEY_COLUMNS].merge(
        requested,
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    if requested["expected_profit"].isna().any():
        raise RuntimeError("Expected-profit data remain incomplete.")

    cache_parts = []
    if DATA_PATH.exists() and not REBUILD_EXPECTED_PROFIT_DATA:
        cache_parts.append(pd.read_csv(DATA_PATH))
    cache_parts.append(requested)
    data_to_cache = pd.concat(cache_parts, ignore_index=True, sort=False)
    data_to_cache = data_to_cache.drop_duplicates(KEY_COLUMNS, keep="last")
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    data_to_cache.to_csv(DATA_PATH, index=False, float_format="%.15g")
    print(f"Saved audited expected-profit data: {DATA_PATH}")
    return requested


def plot_expected_profit(data: pd.DataFrame) -> list[Path]:
    c_ppa_values = sorted(data["c_ppa"].unique())
    ppa_prices = sorted(data["lambda_PPA"].unique())
    if len(c_ppa_values) != 3:
        raise ValueError("The centered 2+1 layout requires three c_ppa values.")

    palette = get_palette(PALETTE_NAME)
    if len(palette) < len(BATTERY_CASES):
        raise ValueError(
            f"Palette '{PALETTE_NAME}' does not contain enough colors."
        )

    values_meur = data["expected_profit"].to_numpy(dtype=float) / 1e6
    y_min = min(float(values_meur.min()), 0.0)
    y_max = max(float(values_meur.max()), 0.1)
    y_span = y_max - y_min

    fig = plt.figure(figsize=(12.8, 8.6))
    grid = fig.add_gridspec(2, 4, hspace=0.42, wspace=0.42)
    axes = [
        fig.add_subplot(grid[0, 0:2]),
        fig.add_subplot(grid[0, 2:4]),
        fig.add_subplot(grid[1, 1:3]),
    ]

    x_positions = np.arange(len(ppa_prices), dtype=float)
    group_width = 0.76
    bar_width = group_width / len(BATTERY_CASES)

    for panel_index, (ax, c_ppa) in enumerate(zip(axes, c_ppa_values)):
        panel = data.loc[np.isclose(data["c_ppa"], c_ppa)]
        for battery_index, (power, energy) in enumerate(BATTERY_CASES):
            battery_data = (
                panel.loc[
                    np.isclose(panel["P_b_max"], power)
                    & np.isclose(panel["E_max"], energy)
                ]
                .set_index("lambda_PPA")
                .reindex(ppa_prices)
            )
            if battery_data["expected_profit"].isna().any():
                raise ValueError(
                    f"Incomplete expected-profit data for c_ppa={c_ppa:g}, "
                    f"battery={power:g} MW/{energy:g} MWh."
                )

            offset = (
                battery_index - (len(BATTERY_CASES) - 1) / 2
            ) * bar_width
            values_eur = battery_data["expected_profit"].to_numpy(dtype=float)
            panel_values_meur = values_eur / 1e6
            bars = ax.bar(
                x_positions + offset,
                panel_values_meur,
                width=bar_width * 0.92,
                color=palette[battery_index],
                edgecolor="white",
                linewidth=0.4,
                label=battery_label(power, energy),
            )

            for bar, value_eur, value_meur in zip(
                bars, values_eur, panel_values_meur
            ):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value_meur + 0.018 * y_span,
                    compact_eur_label(value_eur),
                    ha="center",
                    va="bottom",
                    rotation=55,
                    fontsize=7.6,
                )

        panel_letter = chr(ord("a") + panel_index)
        ax.set_title(
            rf"({panel_letter}) $c^{{PPA}}={c_ppa:g}$ EUR/MWh",
            fontsize=10.5,
        )
        ax.set_xticks(x_positions, [f"{value:g}" for value in ppa_prices])
        ax.set_ylim(y_min - 0.05 * y_span, y_max + 0.17 * y_span)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.grid(axis="y", alpha=0.22)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=8.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=3,
        frameon=False,
        fontsize=10.5,
    )
    fig.suptitle(
        "Expected annual profit by battery configuration\n"
        f"{PROFILE}, {SETTLEMENT_PERIOD.capitalize()} PPA deviation settlement",
        y=0.995,
        fontsize=12,
    )
    fig.supxlabel(
        r"PPA price, $\lambda^{PPA}$ [EUR/MWh]",
        y=0.03,
        fontsize=11.5,
    )
    fig.supylabel(
        "Expected annual profit [MEUR/year]",
        x=0.025,
        fontsize=11.5,
    )
    fig.subplots_adjust(top=0.84, bottom=0.09, left=0.085, right=0.98)

    output_dir = OUTPUT_ROOT / "expected_profit"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / (
        f"battery_expected_profit_{profile_slug(PROFILE)}_"
        f"settlement_{SETTLEMENT_PERIOD.lower()}"
    )
    outputs = []
    for suffix in ("pdf", "png"):
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    return outputs


def main() -> None:
    data = build_expected_profit_data(PROFILE)
    outputs = plot_expected_profit(data)
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
