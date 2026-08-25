"""Create a combined 2x2 hourly PPA supply-curve figure for all SDPs."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt


POST_PROCESSING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POST_PROCESSING_ROOT))

from common.plotting_style import get_palette
from common.results_catalog import PROJECT_FOLDERS, hourly_supply_data


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "combined"
COLOR_PALETTE = "colorblind2"
COLORS = get_palette(COLOR_PALETTE)

LINE_STYLES = {
    120.0: {"color": COLORS[0], "marker": "o"},
    160.0: {"color": COLORS[1], "marker": "s"},
    200.0: {"color": COLORS[2], "marker": "^"},
}


def axis_limits(data):
    x_max = max(1.0, float(data["S_PPA"].max()))
    y_min = float(data["lambda_PPA"].min())
    y_max = float(data["lambda_PPA"].max())
    return (0.0, x_max * 1.03), (y_min - 5.0, y_max + 5.0)


def export_exact_values(data):
    """Export the values plotted in the 2x2 figure as CSV and LaTeX."""
    profile_order = list(PROJECT_FOLDERS)
    penalty_order = sorted(LINE_STYLES)

    table_data = data[["sdp_label", "lambda_PPA", "c_ppa", "S_PPA"]].copy()
    table_data["_profile_order"] = table_data["sdp_label"].map(
        {label: position for position, label in enumerate(profile_order)}
    )
    table_data = table_data.sort_values(
        ["_profile_order", "lambda_PPA", "c_ppa"]
    ).drop(columns="_profile_order")

    csv_path = OUTPUT_DIR / "hourly_supply_curves_exact_values.csv"
    table_data.to_csv(csv_path, index=False, float_format="%.15g")

    pivot = table_data.pivot(
        index=["sdp_label", "lambda_PPA"],
        columns="c_ppa",
        values="S_PPA",
    )

    latex_lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        (
            r"\caption{Optimal contracted PPA capacity for the hourly PPA "
            r"deviation settlement cases with a 20 MW/60 MWh (3-hour) battery. "
            r"The PPA deviation penalties are given in EUR/MWh and capacities "
            r"are reported in MW.}"
        ),
        r"\label{tab:hourly_supply_curve_exact_values}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrr@{}}",
        r"\toprule",
        (
            r"Delivery profile & $\lambda^{PPA}$ & "
            r"$c^{PPA}=120$ & $c^{PPA}=160$ & $c^{PPA}=200$ \\"
        ),
        r" & [EUR/MWh] & [MW] & [MW] & [MW] \\",
        r"\midrule",
    ]

    for profile_index, profile in enumerate(profile_order):
        profile_rows = pivot.loc[profile]
        for price_index, (ppa_price, row) in enumerate(profile_rows.iterrows()):
            profile_cell = profile if price_index == 0 else ""
            capacities = " & ".join(
                f"{float(row[penalty]):.6f}" for penalty in penalty_order
            )
            latex_lines.append(
                f"{profile_cell} & {float(ppa_price):g} & {capacities} \\\\"
            )
        if profile_index < len(profile_order) - 1:
            latex_lines.append(r"\addlinespace")

    latex_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular*}",
            r"\end{table}",
        ]
    )
    latex_path = OUTPUT_DIR / "hourly_supply_curves_exact_values.tex"
    latex_path.write_text("\n".join(latex_lines) + "\n", encoding="utf-8")

    return csv_path, latex_path


def main():
    data = hourly_supply_data(p_b_max=20.0, e_max=60.0)
    x_limits, y_limits = axis_limits(data)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table_outputs = export_exact_values(data)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10.5, 7.5),
        sharex=True,
        sharey=True,
    )

    handles = None
    labels = None
    for panel, (ax, sdp_label) in enumerate(zip(axes.flat, PROJECT_FOLDERS)):
        profile_data = data.loc[data["sdp_label"] == sdp_label]
        for c_ppa, style in LINE_STYLES.items():
            curve = profile_data.loc[profile_data["c_ppa"] == c_ppa].sort_values(
                "lambda_PPA"
            )
            ax.plot(
                curve["S_PPA"],
                curve["lambda_PPA"],
                linewidth=1.8,
                markersize=5.5,
                label=rf"$c^{{PPA}}={c_ppa:g}$ EUR/MWh",
                **style,
            )

        panel_letter = chr(ord("a") + panel)
        ax.set_title(f"({panel_letter}) {sdp_label}")
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        ax.grid(True, alpha=0.25)
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()

    fig.supxlabel(
        r"Optimal contracted PPA capacity, $S^{PPA,*}$ [MW]",
        y=0.03,
    )
    fig.supylabel(r"PPA price, $\lambda^{PPA}$ [EUR/MWh]", x=0.02)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0.04, 0.07, 1.0, 0.91))

    stem = OUTPUT_DIR / "hourly_supply_curves_all_sdps_2x2"
    outputs = []
    for suffix in ("pdf", "png"):
        output_path = stem.with_suffix(f".{suffix}")
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        outputs.append(output_path)
    plt.close(fig)

    print("Created combined supply-curve files:")
    for output in outputs + list(table_outputs):
        print(output)


if __name__ == "__main__":
    main()
