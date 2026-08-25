from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd


HERE = Path(__file__).resolve().parent
POST_PROCESSING_ROOT = HERE.parent
SUMMARY_PATH = HERE / "outputs" / "risk_aversion_summary.csv"
FIGURES_DIR = HERE / "outputs" / "figures"

sys.path.insert(0, str(POST_PROCESSING_ROOT))
from common.plotting_style import get_palette  # noqa: E402

from config import PALETTE_NAME  # noqa: E402


def annotate(ax, x_values, y_values, formatter, offset: int = 7) -> None:
    for x_value, y_value in zip(x_values, y_values):
        ax.annotate(
            formatter(y_value),
            (x_value, y_value),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
        )


def main() -> None:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Missing {SUMMARY_PATH}. Run run_risk_aversion_sensitivity.py first."
        )
    data = pd.read_csv(SUMMARY_PATH).sort_values("beta")
    expected_betas = {0.0, 0.1, 0.2, 0.5, 0.8, 1.0}
    if set(data["beta"].round(10)) != expected_betas:
        raise ValueError(
            f"Expected beta values {sorted(expected_betas)}, "
            f"found {sorted(data['beta'].tolist())}."
        )

    palette = get_palette(PALETTE_NAME)
    beta = data["beta"].to_numpy(dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))

    axes[0].plot(beta, data["S_PPA"], marker="o", color=palette[0], linewidth=2)
    annotate(axes[0], beta, data["S_PPA"], lambda value: f"{value:.2f}")
    axes[0].set_ylabel(r"Optimal PPA capacity, $S^{PPA,*}$ [MW]")
    axes[0].set_title("(a) Optimal PPA contracting")

    axes[1].plot(
        beta,
        data["expected_profit"] / 1e6,
        marker="o",
        color=palette[0],
        linewidth=2,
        label="Expected profit",
    )
    axes[1].plot(
        beta,
        data["cvar_profit"] / 1e6,
        marker="s",
        color=palette[1],
        linewidth=2,
        label="CVaR",
    )
    axes[1].set_ylabel("Annual value [MEUR/year]")
    axes[1].set_title("(b) Expected profit and CVaR")
    axes[1].legend(frameon=False, fontsize=9)

    axes[2].plot(
        beta,
        data["expected_shortfall_rate_pct"],
        marker="o",
        color=palette[3],
        linewidth=2,
    )
    annotate(
        axes[2],
        beta,
        data["expected_shortfall_rate_pct"],
        lambda value: f"{value:.1f}%",
    )
    axes[2].set_ylabel("Expected PPA shortfall rate [%]")
    axes[2].set_title("(c) Expected shortfall rate")

    for ax in axes:
        ax.set_xlabel(r"Risk-aversion weight, $\beta$")
        ax.set_xticks(beta)
        ax.grid(axis="y", alpha=0.22)
        ax.set_axisbelow(True)

    fig.suptitle(
        "Risk-aversion sensitivity\n"
        "Generic Industrial, hourly settlement, "
        r"$\lambda^{PPA}=80$, $c^{PPA}=160$ EUR/MWh, 20 MW/60 MWh battery",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        output = FIGURES_DIR / f"risk_aversion_sensitivity.{suffix}"
        fig.savefig(output, dpi=300, bbox_inches="tight")
        print(f"Created {output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
