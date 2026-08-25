"""Plot expected profit, CVaR and PPA-capacity sensitivity across all cases."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd


HERE = Path(__file__).resolve().parent
RISK_DIR = HERE.parent
POST_PROCESSING_ROOT = RISK_DIR.parent
ADDITIONAL_SUMMARY = HERE / "outputs" / "additional_risk_aversion_summary.csv"
LOW_PRICE_SUMMARY = HERE / "outputs" / "low_price_case" / "low_price_beta_sensitivity.csv"
BENCHMARK_SUMMARY = RISK_DIR / "outputs" / "risk_aversion_summary.csv"
MAIN_GRID_SUMMARY = (
    POST_PROCESSING_ROOT
    / "04_profit_decomposition"
    / "outputs"
    / "expected_profit_decomposition_split_imbalance.csv"
)
FIGURE_DIR = HERE / "outputs" / "figures"

sys.path.insert(0, str(RISK_DIR))
sys.path.insert(0, str(POST_PROCESSING_ROOT))
from common.plotting_style import get_palette  # noqa: E402
from config import PALETTE_NAME  # noqa: E402


PLOTTED_BETAS = (0.0, 0.1, 0.2, 0.5, 0.8)
PROFIT_CASES = ("merchant", "low_price", "benchmark", "high_contract")
PPA_CASES = ("low_price", "benchmark", "high_contract", "high_penalty")
ALL_REQUIRED_CASES = tuple(dict.fromkeys(PROFIT_CASES + PPA_CASES))
LABELS = {
    "low_price": r"Low price: $\lambda^{PPA}=40$, $c^{PPA}=120$",
    "benchmark": r"Benchmark: $\lambda^{PPA}=80$, $c^{PPA}=160$",
    "high_contract": r"High contract: $\lambda^{PPA}=100$, $c^{PPA}=120$",
    "high_penalty": r"High penalty: $\lambda^{PPA}=80$, $c^{PPA}=200$",
    "merchant": "Merchant operation",
}


def load_data() -> pd.DataFrame:
    if not ADDITIONAL_SUMMARY.exists():
        raise FileNotFoundError(
            f"Missing {ADDITIONAL_SUMMARY}. Run run_additional_cases.py first."
        )
    if not BENCHMARK_SUMMARY.exists():
        raise FileNotFoundError(f"Missing benchmark summary: {BENCHMARK_SUMMARY}")
    if not LOW_PRICE_SUMMARY.exists():
        raise FileNotFoundError(
            f"Missing {LOW_PRICE_SUMMARY}. Run run_low_price_case.py first."
        )
    if not MAIN_GRID_SUMMARY.exists():
        raise FileNotFoundError(f"Missing main-results summary: {MAIN_GRID_SUMMARY}")

    additional = pd.read_csv(ADDITIONAL_SUMMARY)
    benchmark = pd.read_csv(BENCHMARK_SUMMARY)
    benchmark["case_id"] = "benchmark"
    low_price = pd.read_csv(LOW_PRICE_SUMMARY)

    main_grid = pd.read_csv(MAIN_GRID_SUMMARY)
    low_price_beta_0p2 = main_grid.loc[
        main_grid["sdp_label"].eq("Generic Industrial")
        & main_grid["settlement_period"].str.lower().eq("hourly")
        & main_grid["lambda_PPA"].astype(float).eq(40.0)
        & main_grid["c_ppa"].astype(float).eq(120.0)
        & main_grid["P_b_max"].astype(float).eq(20.0)
        & main_grid["E_max"].astype(float).eq(60.0)
    ].copy()
    if len(low_price_beta_0p2) != 1:
        raise ValueError(
            "Expected one existing low-price beta=0.2 result, "
            f"found {len(low_price_beta_0p2)}."
        )
    low_price_beta_0p2["case_id"] = "low_price"
    low_price_beta_0p2["beta"] = 0.2

    columns = [
        "case_id",
        "beta",
        "S_PPA",
        "expected_profit",
        "cvar_profit",
    ]
    data = pd.concat(
        [
            benchmark[columns],
            additional[columns],
            low_price[columns],
            low_price_beta_0p2[columns],
        ],
        ignore_index=True,
    )
    data = data.loc[data["beta"].round(10).isin(PLOTTED_BETAS)].copy()

    expected_betas = set(PLOTTED_BETAS)
    for case_id in ALL_REQUIRED_CASES:
        case_betas = set(
            data.loc[data["case_id"].eq(case_id), "beta"].round(10).tolist()
        )
        if case_betas != expected_betas:
            raise ValueError(
                f"Expected beta values {sorted(expected_betas)} for {case_id}, "
                f"found {sorted(case_betas)}."
            )
        if data.loc[data["case_id"].eq(case_id), "beta"].duplicated().any():
            raise ValueError(f"Duplicate beta results found for {case_id}.")

    return data


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        output = FIGURE_DIR / f"{stem}.{suffix}"
        fig.savefig(output, dpi=300, bbox_inches="tight")
        print(f"Created: {output}")
    plt.close(fig)


def plot_expected_profit_and_cvar(data: pd.DataFrame, palette: list[str]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.2), sharex=False)

    panel_letters = ("a", "b", "c", "d")
    for ax, panel_letter, case_id in zip(axes.flat, panel_letters, PROFIT_CASES):
        case = data.loc[data["case_id"].eq(case_id)].sort_values("beta")
        beta = case["beta"].to_numpy(dtype=float)
        expected = case["expected_profit"].to_numpy(dtype=float) / 1e6
        cvar = case["cvar_profit"].to_numpy(dtype=float) / 1e6

        positions = list(range(len(beta)))
        bar_width = 0.36
        expected_bars = ax.bar(
            [position - bar_width / 2 for position in positions],
            expected,
            width=bar_width,
            color=palette[0],
            label="Expected profit",
        )
        cvar_bars = ax.bar(
            [position + bar_width / 2 for position in positions],
            cvar,
            width=bar_width,
            color=palette[3],
            label="CVaR: mean of worst eight scenarios",
        )
        ax.bar_label(expected_bars, fmt="%.2f", padding=2, fontsize=8)
        ax.bar_label(cvar_bars, fmt="%.2f", padding=2, fontsize=8)
        ax.set_title(f"({panel_letter}) {LABELS[case_id]}", fontsize=10.5)
        ax.set_xticks(positions, [f"{value:g}" for value in beta])
        ax.tick_params(axis="x", labelbottom=True)
        ax.set_ylim(0, max(expected.max(), cvar.max()) * 1.13)
        ax.grid(axis="y", alpha=0.22)
        ax.set_axisbelow(True)

    for ax in axes[:, 0]:
        ax.set_ylabel("Annual profit [MEUR/year]")
    for ax in axes[1, :]:
        ax.set_xlabel(r"Risk-aversion weight, $\beta$")

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.925),
    )
    fig.suptitle(
        "Expected profit and CVaR by risk-aversion weight\n"
        "Generic Industrial, hourly settlement, 20 MW/60 MWh battery",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    save_figure(fig, "risk_aversion_expected_profit_cvar_all_cases")


def plot_capacity_change(data: pd.DataFrame, palette: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    colors = (palette[0], palette[1], palette[3], palette[4])

    for case_id, color in zip(PPA_CASES, colors):
        case = data.loc[data["case_id"].eq(case_id)].sort_values("beta").copy()
        baseline = float(case.loc[case["beta"].eq(0.0), "S_PPA"].iloc[0])
        case["capacity_change_pct"] = 100.0 * (case["S_PPA"] / baseline - 1.0)
        ax.plot(
            case["beta"],
            case["capacity_change_pct"],
            marker="o",
            linewidth=2,
            color=color,
            label=LABELS[case_id],
        )

    ax.axhline(0.0, color="#333333", linewidth=0.9)
    ax.set_xlabel(r"Risk-aversion weight, $\beta$")
    ax.set_ylabel(r"Change in optimal PPA capacity from $\beta=0$ [\%]")
    ax.set_xticks(PLOTTED_BETAS)
    ax.set_title(
        "Relative response of optimal PPA capacity to risk aversion\n"
        "Generic Industrial, hourly settlement, 20 MW/60 MWh battery",
        fontsize=12,
    )
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    save_figure(fig, "risk_aversion_ppa_capacity_change_all_cases")


def main() -> None:
    data = load_data()
    palette = get_palette(PALETTE_NAME)
    plot_expected_profit_and_cvar(data, palette)
    plot_capacity_change(data, palette)


if __name__ == "__main__":
    main()
