"""Plot the number of positive-NPV cases for each settlement period."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
POST_PROCESSING_ROOT = HERE.parent
OUTPUT_DIR = HERE / "outputs" / "positive_case_count"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(POST_PROCESSING_ROOT))

from common.plotting_style import get_palette  # noqa: E402
from npv_analysis import calculate_npv_results  # noqa: E402
from settings import (  # noqa: E402
    E_MAX,
    P_B_MAX,
    PALETTE_NAME,
    PROFILE,
    SETTLEMENT_PERIODS,
)


PERIOD_LABELS = {
    "hourly": "Hourly",
    "daily": "Daily",
    "weekly": "Weekly",
    "monthly": "Monthly",
}


def plot_counts(data=None) -> list[Path]:
    if data is None:
        data = calculate_npv_results()
    data = data.loc[data["sdp_label"].eq(PROFILE)].copy()
    counts = (
        data.groupby("settlement_period")["positive_npv"]
        .agg(["sum", "size"])
        .reindex(SETTLEMENT_PERIODS)
    )
    palette = get_palette(PALETTE_NAME)
    x = np.arange(len(SETTLEMENT_PERIODS))

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    bars = ax.bar(x, counts["sum"], width=0.58, color=palette[: len(x)])
    ax.bar_label(
        bars,
        labels=[f"{int(value)}/{int(total)}" for value, total in counts.to_numpy()],
        padding=4,
        fontsize=10,
    )
    ax.set_xticks(x, [PERIOD_LABELS[value] for value in SETTLEMENT_PERIODS])
    ax.set_ylabel("Cases with positive NPV")
    ax.set_xlabel("PPA deviation settlement period")
    ax.set_ylim(0, float(counts["size"].max()) + 1.5)
    ax.set_title(
        "Financially feasible PPA cases by deviation settlement period\n"
        f"{PROFILE}, {P_B_MAX:g} MW/{E_MAX:g} MWh battery"
    )
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)
    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT_DIR / "positive_npv_cases_by_settlement_period"
    outputs = []
    for suffix in ("pdf", "png"):
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    return outputs


def main() -> None:
    outputs = plot_counts()
    print("Created:")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
