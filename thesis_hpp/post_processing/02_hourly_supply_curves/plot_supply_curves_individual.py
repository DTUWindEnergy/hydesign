"""Create one hourly PPA supply-curve figure for each delivery profile."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt


POST_PROCESSING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POST_PROCESSING_ROOT))

from common.plotting_style import get_palette
from common.results_catalog import PROJECT_FOLDERS, hourly_supply_data


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "individual"
COLOR_PALETTE = "colorblind2"
COLORS = get_palette(COLOR_PALETTE)

LINE_STYLES = {
    120.0: {"color": COLORS[0], "marker": "o"},
    160.0: {"color": COLORS[1], "marker": "s"},
    200.0: {"color": COLORS[2], "marker": "^"},
}


def slugify(text: str) -> str:
    return "_".join(text.lower().split())


def axis_limits(data):
    x_max = max(1.0, float(data["S_PPA"].max()))
    y_min = float(data["lambda_PPA"].min())
    y_max = float(data["lambda_PPA"].max())
    return (0.0, x_max * 1.03), (y_min - 5.0, y_max + 5.0)


def plot_profile(ax, profile_data, title, x_limits, y_limits):
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

    ax.set_title(title)
    ax.set_xlabel(r"Optimal contracted PPA capacity, $S^{PPA,*}$ [MW]")
    ax.set_ylabel(r"PPA price, $\lambda^{PPA}$ [EUR/MWh]")
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)


def main():
    data = hourly_supply_data(p_b_max=20.0, e_max=60.0)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    outputs = []
    for sdp_label in PROJECT_FOLDERS:
        profile_data = data.loc[data["sdp_label"] == sdp_label]
        x_limits, y_limits = axis_limits(profile_data)
        fig, ax = plt.subplots(figsize=(6.4, 4.5), constrained_layout=True)
        plot_profile(ax, profile_data, sdp_label, x_limits, y_limits)

        stem = OUTPUT_DIR / f"hourly_supply_curve_{slugify(sdp_label)}"
        for suffix in ("pdf", "png"):
            output_path = stem.with_suffix(f".{suffix}")
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            outputs.append(output_path)
        plt.close(fig)

    print(f"Created {len(outputs)} individual supply-curve files:")
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
