"""Plot the contracting-period sensitivity after the optimization job finishes."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


HERE = Path(__file__).resolve().parent
OUTPUT_ROOT = HERE / "outputs"
SUMMARY_PATH = OUTPUT_ROOT / "contracting_period_summary.csv"
CAPACITY_PATH = OUTPUT_ROOT / "contracting_period_capacities.csv"

COLORS = {
    "annual": "#0072B2",
    "seasonal": "#B07C4F",
    "monthly": "#009E73",
}
LABELS = {
    "annual": "Annual capacity",
    "seasonal": "Seasonal capacities",
    "monthly": "Monthly capacities",
}
MONTHS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]
SEASON_BY_MONTH = [
    "Winter", "Winter", "Spring", "Spring", "Spring", "Summer",
    "Summer", "Summer", "Autumn", "Autumn", "Autumn", "Winter",
]


def monthly_schedule(capacities: pd.DataFrame, period: str) -> list[float]:
    frame = capacities.loc[capacities["contracting_period"] == period]
    values = dict(zip(frame["period_label"], frame["S_PPA_mw"]))
    if period == "annual":
        return [values["Annual"]] * 12
    if period == "seasonal":
        return [values[season] for season in SEASON_BY_MONTH]
    return [values[month] for month in (
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    )]


def main() -> None:
    if not SUMMARY_PATH.exists() or not CAPACITY_PATH.exists():
        raise FileNotFoundError(
            "Run run_contracting_period_sensitivity.py before plotting."
        )
    summary = pd.read_csv(SUMMARY_PATH)
    capacities = pd.read_csv(CAPACITY_PATH)
    order = ["annual", "seasonal", "monthly"]
    summary = summary.set_index("contracting_period").loc[order].reset_index()

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.2), constrained_layout=True)
    x = range(12)
    for period in order:
        axes[0].step(
            x,
            monthly_schedule(capacities, period),
            where="mid",
            color=COLORS[period],
            linewidth=2.1,
            marker="o",
            markersize=4.5,
            label=LABELS[period],
        )
    axes[0].set_xticks(list(x), MONTHS)
    axes[0].set_ylabel(r"Optimal $S^{PPA}$ [MW]")
    axes[0].set_title("Optimal PPA capacity schedule")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, ncol=3, loc="upper center")

    expected = summary["expected_profit_eur"] / 1e6
    objective = summary["objective_value_eur"] / 1e6
    positions = list(range(3))
    width = 0.34
    axes[1].bar(
        [value - width / 2 for value in positions],
        expected,
        width,
        color="#0072B2",
        label="Expected annual profit",
    )
    axes[1].bar(
        [value + width / 2 for value in positions],
        objective,
        width,
        color="#B07C4F",
        label="Mean-CVaR objective",
    )
    axes[1].set_xticks(positions, ["Annual", "Seasonal", "Monthly"])
    axes[1].set_ylabel("MEUR/year")
    axes[1].set_title("Economic effect of contracting-period flexibility")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, ncol=2, loc="upper center")

    for bars in axes[1].containers:
        axes[1].bar_label(bars, fmt="%.2f", padding=3, fontsize=9)

    output = OUTPUT_ROOT / "contracting_period_sensitivity.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()

