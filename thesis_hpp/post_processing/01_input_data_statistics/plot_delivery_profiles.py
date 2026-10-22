"""Plot the daily shapes of the delivery profiles defined in inputs/sdp.py."""

from pathlib import Path
import argparse
import re
import sys

import matplotlib.pyplot as plt
import pandas as pd
from cycler import cycler


BASE_PATH = Path(__file__).resolve().parents[2]
INPUTS_PATH = BASE_PATH / "inputs"
GRAPHS_PATH = BASE_PATH / "post_processing" / "graphs"

sys.path.insert(0, str(INPUTS_PATH))

from sdp import DELIVERY_PROFILES, get_delivery_profile


DEFAULT_PROFILES = [
    "Generic Industrial Yearly",
    "Generic Private Yearly",
    "Inverse Solar Yearly",
   #"Peak Yearly",
    "Yearly Continuous Baseload",
]

COLOR_PALETTE = "academic"

PALETTES = {
    "colorblind": ["#0072B2", "#009E73", "#999999", "#D55E00", "#CC79A7"],
    "academic": ["#1F4E79", "#5B8C5A", "#7F7F7F", "#9A6A2F", "#6B5B95"],
    "muted": ["#4C78A8", "#72B7B2", "#A0A0A0", "#F58518", "#B279A2"],
}


def slugify(text):
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def apply_color_palette():
    plt.rcParams["axes.prop_cycle"] = cycler(color=PALETTES[COLOR_PALETTE])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot 24-hour delivery profiles from inputs/sdp.py."
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=DEFAULT_PROFILES,
        help=(
            "Delivery profile names to plot. Use quotes for names with spaces, "
            'for example: --profiles "Generic Industrial Yearly" "Generic Private Yearly"'
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available fixed-shape delivery profiles and exit.",
    )
    return parser.parse_args()


def fixed_profile_names():
    return [
        name
        for name, profile in DELIVERY_PROFILES.items()
        if profile.hourly_shape is not None
    ]


def build_profile_dataframe(profile_names):
    out = pd.DataFrame({"hour": range(24)})
    for profile_name in profile_names:
        profile = get_delivery_profile(profile_name)
        out[profile.name] = [float(value) for value in profile.hourly_shape]
    return out


def close_daily_cycle(df):
    first_hour = df.iloc[[0]].copy()
    first_hour["hour"] = 24
    return pd.concat([df, first_hour], ignore_index=True)


def plot_profiles(df, profile_names):
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_df = close_daily_cycle(df)

    for profile_name in profile_names:
        ax.plot(
            plot_df["hour"],
            plot_df[profile_name],
            linewidth=2,
            label=profile_name,
        )

    ax.set_xlabel("Hour of the day")
    ax.set_ylabel("Normalized delivery profile")
    ax.set_xticks(range(0, 25, 2))
    ax.set_xlim(0, 24)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    filename_profiles = "_".join(slugify(name) for name in profile_names)
    output_path = GRAPHS_PATH / f"delivery_profiles_{filename_profiles}.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def main():
    apply_color_palette()
    args = parse_args()

    if args.list:
        print("Available fixed-shape delivery profiles:")
        for name in fixed_profile_names():
            print(f"- {name}")
        return

    GRAPHS_PATH.mkdir(parents=True, exist_ok=True)

    df = build_profile_dataframe(args.profiles)
    plot_path = plot_profiles(df, args.profiles)

    filename_profiles = "_".join(slugify(name) for name in args.profiles)
    csv_path = GRAPHS_PATH / f"delivery_profiles_{filename_profiles}.csv"
    df.to_csv(csv_path, index=False)

    print("Created delivery profile plot:")
    print(plot_path)
    print("Created delivery profile CSV:")
    print(csv_path)


if __name__ == "__main__":
    main()
