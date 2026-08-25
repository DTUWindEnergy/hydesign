"""Create compact input-data statistics plots for the thesis methodology section."""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from cycler import cycler


BASE_PATH = Path(__file__).resolve().parents[2]
INPUTS_PATH = BASE_PATH / "inputs" / "Input_files_created"
GRAPHS_PATH = BASE_PATH / "post_processing" / "graphs" / "input_data_statistics"

COLOR_PALETTE = "colorblind2"

PALETTES = {
    "colorblind": ["#0072B2", "#009E73", "#999999", "#D55E00", "#CC79A7"],
    "colorblind2": ["#0072B2", "#009E73", "#9A6A2F", "#D55E00", "#CC79A7"],
    "academic": ["#1F4E79", "#5B8C5A", "#7F7F7F", "#9A6A2F", "#6B5B95"],
    "muted": ["#4C78A8", "#72B7B2", "#A0A0A0", "#F58518", "#B279A2"],
}


def clean_columns(df):
    return df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]


def apply_style():
    plt.rcParams["axes.prop_cycle"] = cycler(color=PALETTES[COLOR_PALETTE])
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 11
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 9


def load_metadata():
    path = INPUTS_PATH / "input_scenario_metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_config():
    path = INPUTS_PATH / "config.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def omega_year_map():
    years = load_metadata().get("weather_years", [])
    return {i + 1: year for i, year in enumerate(years)}


def load_price_data():
    spot = clean_columns(pd.read_csv(INPUTS_PATH / "spot_prices.csv"))
    up = clean_columns(pd.read_csv(INPUTS_PATH / "mfrr_up_prices.csv"))
    down = clean_columns(pd.read_csv(INPUTS_PATH / "mfrr_down_prices.csv"))
    imb = clean_columns(pd.read_csv(INPUTS_PATH / "imbalance_prices.csv"))

    # mFRR price files repeat the same hourly price for each xi. Drop xi so
    # each weather-year hour has one price observation.
    up = up.drop_duplicates(["t", "omega"])[["t", "omega", "price"]]
    down = down.drop_duplicates(["t", "omega"])[["t", "omega", "price"]]
    imb = imb.drop_duplicates(["t", "omega"])[["t", "omega", "price"]]

    prices = spot.rename(columns={"price": "spot"})
    prices = prices.merge(up.rename(columns={"price": "up"}), on=["t", "omega"], how="inner")
    prices = prices.merge(down.rename(columns={"price": "down"}), on=["t", "omega"], how="inner")
    prices = prices.merge(imb.rename(columns={"price": "imbalance"}), on=["t", "omega"], how="inner")
    prices["hour"] = (prices["t"].astype(int) - 1) % 24
    prices["year"] = prices["omega"].map(omega_year_map()).fillna(prices["omega"])
    prices["up_above_spot"] = prices["up"] > prices["spot"]
    prices["down_below_spot"] = prices["down"] < prices["spot"]
    prices["down_non_positive"] = prices["down"] <= 0.0

    prices["system_y_up"] = 0
    prices["system_y_down"] = 0
    only_up = prices["up_above_spot"] & ~prices["down_below_spot"]
    only_down = prices["down_below_spot"] & ~prices["up_above_spot"]
    both = prices["up_above_spot"] & prices["down_below_spot"]
    prices.loc[only_up, "system_y_up"] = 1
    prices.loc[only_down, "system_y_down"] = 1

    # If both price signals occur within the hour, keep only the dominant
    # system direction already encoded in the prepared imbalance price.
    up_distance = (prices["imbalance"] - prices["up"]).abs()
    down_distance = (prices["imbalance"] - prices["down"]).abs()
    dominant_up = both & (up_distance <= down_distance)
    dominant_down = both & (down_distance < up_distance)
    prices.loc[dominant_up, "system_y_up"] = 1
    prices.loc[dominant_down, "system_y_down"] = 1
    return prices


def load_regulation_state_data():
    up = clean_columns(pd.read_csv(INPUTS_PATH / "mfrr_up_prices.csv"))
    down = clean_columns(pd.read_csv(INPUTS_PATH / "mfrr_down_prices.csv"))

    # y_up/y_down are identical across xi for a given weather-year hour in the
    # current input generation. Drop xi to count each hour only once.
    up_state = up.drop_duplicates(["t", "omega"])[["t", "omega", "y_up"]]
    down_state = down.drop_duplicates(["t", "omega"])[["t", "omega", "y_down"]]
    states = up_state.merge(down_state, on=["t", "omega"], how="inner")
    states["y_up"] = states["y_up"].astype(int)
    states["y_down"] = states["y_down"].astype(int)
    states["neither"] = ((states["y_up"] == 0) & (states["y_down"] == 0)).astype(int)
    return states


def load_wind_data():
    wind_path = INPUTS_PATH / "wind_realization.csv"
    if not wind_path.exists():
        raise FileNotFoundError(f"Missing {wind_path}")
    wind = clean_columns(pd.read_csv(wind_path))
    p_nom = float(load_config().get("P_nom", 1.0))
    wind["capacity_factor"] = wind["wind"] / p_nom if p_nom else wind["wind"]
    wind["hour"] = (wind["t"].astype(int) - 1) % 24
    wind["month"] = wind["t"].apply(synthetic_month_from_hour)
    wind["year"] = wind["omega"].map(omega_year_map()).fillna(wind["omega"])
    return wind


def close_daily_cycle(df):
    first_hour = df[df["hour"] == 0].copy()
    if first_hour.empty:
        return df
    first_hour["hour"] = 24
    return pd.concat([df, first_hour], ignore_index=True).sort_values("hour")


def boxplot_with_colors(ax, data, labels):
    box = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        showfliers=True,
        whis=1.5,
        widths=0.55,
        flierprops={
            "marker": "o",
            "markersize": 2.5,
            "markerfacecolor": "none",
            "markeredgecolor": "0.35",
            "alpha": 0.45,
        },
    )
    colors = PALETTES[COLOR_PALETTE]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    for median in box["medians"]:
        median.set_color("black")
        median.set_linewidth(1.4)
    ax.grid(True, axis="y", alpha=0.3)


def save_price_boxplot_all_years(prices):
    data = [
        prices["spot"],
        prices.loc[prices["up_above_spot"], "up"],
        prices.loc[prices["down_below_spot"], "down"],
    ]
    labels = [
        r"$\lambda^{spot}$",
        r"$\lambda^{up}$",
        r"$\lambda^{down}$",
    ]

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    boxplot_with_colors(ax, data, labels)
    ax.set_ylabel("Price [EUR/MWh]")
    ax.set_title("Input price distributions across all weather years")
    ax.tick_params(axis="x", labelrotation=10)
    fig.tight_layout()

    output_path = GRAPHS_PATH / "price_boxplot_all_weather_years.png"
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def save_price_boxplot_by_year(prices):
    year_labels = sorted(prices["year"].unique())
    series_specs = [
        ("spot", r"$\lambda^{spot}$", lambda df: df["spot"]),
        ("up", r"$\lambda^{up}$", lambda df: df.loc[df["up_above_spot"], "up"]),
        ("down", r"$\lambda^{down}$", lambda df: df.loc[df["down_below_spot"], "down"]),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(12.0, 9.0), sharex=True)
    colors = PALETTES[COLOR_PALETTE]
    for ax, (_, title, selector), color in zip(axes, series_specs, colors):
        data = []
        for year in year_labels:
            values = selector(prices[prices["year"] == year]).dropna()
            data.append(values if not values.empty else pd.Series([float("nan")]))
        box = ax.boxplot(
            data,
            patch_artist=True,
            showfliers=True,
            whis=1.5,
            widths=0.55,
            flierprops={
                "marker": "o",
                "markersize": 2.0,
                "markerfacecolor": "none",
                "markeredgecolor": "0.35",
                "alpha": 0.4,
            },
        )
        for patch in box["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.72)
        for median in box["medians"]:
            median.set_color("black")
            median.set_linewidth(1.2)
        ax.set_ylabel("EUR/MWh")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)

    axes[-1].set_xticks(range(1, len(year_labels) + 1))
    axes[-1].set_xticklabels([str(year) for year in year_labels], rotation=45, ha="right")
    axes[-1].set_xlabel("Weather year")
    fig.suptitle("Input price distributions by weather year", y=0.995)
    fig.tight_layout()

    output_path = GRAPHS_PATH / "price_boxplot_by_weather_year.png"
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def save_diurnal_price_plot(prices):
    spot = prices.groupby("hour", as_index=False)["spot"].mean()
    up = (
        prices[prices["system_y_up"] == 1]
        .groupby("hour", as_index=False)["up"]
        .mean()
    )
    down = (
        prices[prices["system_y_down"] == 1]
        .groupby("hour", as_index=False)["down"]
        .mean()
    )

    diurnal = spot.merge(up, on="hour", how="left").merge(down, on="hour", how="left")
    diurnal = close_daily_cycle(diurnal)

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.plot(diurnal["hour"], diurnal["spot"], linewidth=2.2, label=r"$\lambda^{spot}$")
    ax.plot(
        diurnal["hour"],
        diurnal["up"],
        linewidth=2.2,
        label=r"$\lambda^{up}$",
    )
    ax.plot(
        diurnal["hour"],
        diurnal["down"],
        linewidth=2.2,
        label=r"$\lambda^{down}$",
    )

    ax.set_xlabel("Hour of the day")
    ax.set_ylabel("Average price [EUR/MWh]")
    ax.set_xticks(range(0, 25, 2))
    ax.set_xlim(0, 24)
    ax.grid(True, alpha=0.3)
    ax.set_title("Average diurnal input price profiles across all weather years")
    ax.legend(loc="best", frameon=True, framealpha=0.92)
    fig.tight_layout()

    output_path = GRAPHS_PATH / "diurnal_price_profiles_all_weather_years.png"
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def save_diurnal_price_distribution_plot(prices):
    series = [
        (
            prices,
            "spot",
            r"$\lambda^{spot}$ (all hours)",
            PALETTES[COLOR_PALETTE][0],
        ),
        (
            prices[prices["system_y_up"] == 1],
            "up",
            r"$\lambda^{up}$ (dominant-up hours)",
            PALETTES[COLOR_PALETTE][1],
        ),
        (
            prices[prices["system_y_down"] == 1],
            "down",
            r"$\lambda^{down}$ (dominant-down hours)",
            PALETTES[COLOR_PALETTE][2],
        ),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(8.2, 8.0), sharex=True)
    for ax, (data, column, title, color) in zip(axes, series):
        hourly = (
            data.groupby("hour")[column]
            .agg(
                mean="mean",
                p25=lambda values: values.quantile(0.25),
                p75=lambda values: values.quantile(0.75),
            )
            .reset_index()
        )
        hourly = close_daily_cycle(hourly)

        hour = hourly["hour"].to_numpy()
        mean = hourly["mean"].to_numpy()
        p25 = hourly["p25"].to_numpy()
        p75 = hourly["p75"].to_numpy()
        ax.fill_between(hour, p25, p75, color=color, alpha=0.22, label="P25-P75")
        ax.plot(hour, mean, color=color, linewidth=2.2, label="Mean")
        ax.set_ylabel("Price [EUR/MWh]")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", frameon=True, framealpha=0.92)

    axes[-1].set_xlabel("Hour of the day")
    axes[-1].set_xticks(range(0, 25, 2))
    axes[-1].set_xlim(0, 24)
    fig.suptitle(
        "Diurnal input-price distributions across all weather years",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    output_path = GRAPHS_PATH / "diurnal_price_profiles_with_iqr.png"
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def save_regulation_frequency_plot(states):
    total_hours = len(states)
    rows = [
        ("Up-regulation", 100 * states["y_up"].mean()),
        ("Down-regulation", 100 * states["y_down"].mean()),
        ("Neither", 100 * states["neither"].mean()),
    ]
    labels = [row[0] for row in rows]
    values = [row[1] for row in rows]

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    bars = ax.bar(labels, values, color=PALETTES[COLOR_PALETTE][: len(labels)], alpha=0.82)
    ax.set_ylabel("Share of weather-year hours [%]")
    ax.set_ylim(0, max(values) * 1.2 if values else 1)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title("Frequency of model regulation states")
    ax.tick_params(axis="x", labelrotation=12)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.text(
        0.99,
        0.96,
        f"n = {total_hours:,} hourly observations",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )
    fig.tight_layout()

    output_path = GRAPHS_PATH / "regulation_price_signal_frequency.png"
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def synthetic_month_from_hour(t):
    month_days = [30, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 30]
    hour_of_year = int(t) - 1
    day_of_year = hour_of_year // 24
    cumulative = 0
    for month, days in enumerate(month_days, start=1):
        cumulative += days
        if day_of_year < cumulative:
            return month
    return 12


def summary_row(series_name, values):
    values = pd.Series(values).dropna()
    return {
        "series": series_name,
        "count": int(values.count()),
        "mean": values.mean(),
        "median": values.median(),
        "p05": values.quantile(0.05),
        "p25": values.quantile(0.25),
        "p75": values.quantile(0.75),
        "p95": values.quantile(0.95),
        "min": values.min(),
        "max": values.max(),
    }


def save_table(df, output_stem, float_format="%.3f"):
    csv_path = GRAPHS_PATH / f"{output_stem}.csv"
    tex_path = GRAPHS_PATH / f"{output_stem}.tex"
    df.to_csv(csv_path, index=False)
    df.to_latex(tex_path, index=False, float_format=float_format)
    return csv_path, tex_path


def save_monthly_wind_boxplot(wind):

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    data = [wind.loc[wind["month"] == month, "capacity_factor"] for month in range(1, 13)]

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    box = ax.boxplot(
        data,
        labels=month_labels,
        patch_artist=True,
        showfliers=False,
        whis=(5, 95),
        widths=0.55,
    )
    for patch in box["boxes"]:
        patch.set_facecolor(PALETTES[COLOR_PALETTE][0])
        patch.set_alpha(0.72)
    for median in box["medians"]:
        median.set_color("black")
        median.set_linewidth(1.2)
    ax.set_ylabel("Wind capacity factor [-]")
    ax.set_xlabel("Synthetic month")
    ax.set_title("Monthly distribution of CORRES wind realizations (P5-P95 whiskers)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    output_path = GRAPHS_PATH / "monthly_wind_capacity_factor_boxplot.png"
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def save_wind_distribution_plot(wind):
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.hist(
        wind["capacity_factor"].dropna(),
        bins=50,
        color=PALETTES[COLOR_PALETTE][0],
        alpha=0.78,
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set_xlabel("Wind capacity factor [-]")
    ax.set_ylabel("Number of scenario-hours")
    ax.set_title("Distribution of CORRES wind realizations across all weather years")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    output_path = GRAPHS_PATH / "wind_capacity_factor_distribution.png"
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def save_monthly_wind_profile_table(wind):
    month_labels = {
        1: "Jan",
        2: "Feb",
        3: "Mar",
        4: "Apr",
        5: "May",
        6: "Jun",
        7: "Jul",
        8: "Aug",
        9: "Sep",
        10: "Oct",
        11: "Nov",
        12: "Dec",
    }
    rows = []
    for month, group in wind.sort_values("month").groupby("month"):
        rows.append(
            {
                "month": month_labels[int(month)],
                "mean_cf": group["capacity_factor"].mean(),
                "median_cf": group["capacity_factor"].median(),
                "p05_cf": group["capacity_factor"].quantile(0.05),
                "p95_cf": group["capacity_factor"].quantile(0.95),
            }
        )
    return save_table(pd.DataFrame(rows), "monthly_wind_capacity_factor_statistics")


def save_price_summary_table(prices):
    rows = [
        summary_row("lambda_spot", prices["spot"]),
        summary_row("lambda_up_dominant_direction", prices.loc[prices["system_y_up"] == 1, "up"]),
        summary_row("lambda_down_dominant_direction", prices.loc[prices["system_y_down"] == 1, "down"]),
    ]
    return save_table(pd.DataFrame(rows), "price_summary_statistics")


def save_wind_summary_table(wind):
    rows = [
        summary_row("wind_capacity_factor_all_realizations", wind["capacity_factor"]),
        summary_row("wind_power_mw_all_realizations", wind["wind"]),
    ]
    return save_table(pd.DataFrame(rows), "wind_summary_statistics")


def save_regulation_frequency_table(prices, states):
    total_hours = len(states)
    rows = [
        {
            "signal": "model_up_regulation",
            "hours": int(states["y_up"].sum()),
            "share_percent": 100 * states["y_up"].mean(),
        },
        {
            "signal": "model_down_regulation",
            "hours": int(states["y_down"].sum()),
            "share_percent": 100 * states["y_down"].mean(),
        },
        {
            "signal": "model_neither_regulation",
            "hours": int(states["neither"].sum()),
            "share_percent": 100 * states["neither"].mean(),
        },
        {
            "signal": "raw_lambda_up_above_spot",
            "hours": int(prices["up_above_spot"].sum()),
            "share_percent": 100 * prices["up_above_spot"].mean(),
        },
        {
            "signal": "raw_lambda_down_below_spot",
            "hours": int(prices["down_below_spot"].sum()),
            "share_percent": 100 * prices["down_below_spot"].mean(),
        },
        {
            "signal": "raw_lambda_down_below_spot_and_non_positive",
            "hours": int((prices["down_below_spot"] & prices["down_non_positive"]).sum()),
            "share_percent": 100 * (prices["down_below_spot"] & prices["down_non_positive"]).mean(),
        },
    ]
    table = pd.DataFrame(rows)
    table["total_hours"] = total_hours
    return save_table(table, "regulation_signal_frequency_statistics")


def save_weather_year_summary(prices, wind):
    spot_summary = (
        prices.groupby("year")["spot"]
        .agg(
            mean_spot="mean",
            median_spot="median",
            p05_spot=lambda x: x.quantile(0.05),
            p95_spot=lambda x: x.quantile(0.95),
        )
        .reset_index()
    )
    up_summary = (
        prices[prices["system_y_up"] == 1]
        .groupby("year")["up"]
        .agg(
            mean_up="mean",
            median_up="median",
            p05_up=lambda x: x.quantile(0.05),
            p95_up=lambda x: x.quantile(0.95),
        )
        .reset_index()
    )
    down_summary = (
        prices[prices["system_y_down"] == 1]
        .groupby("year")["down"]
        .agg(
            mean_down="mean",
            median_down="median",
            p05_down=lambda x: x.quantile(0.05),
            p95_down=lambda x: x.quantile(0.95),
        )
        .reset_index()
    )
    wind_summary = (
        wind.groupby("year")["capacity_factor"]
        .agg(
            mean_wind_cf="mean",
            median_wind_cf="median",
            p05_wind_cf=lambda x: x.quantile(0.05),
            p95_wind_cf=lambda x: x.quantile(0.95),
        )
        .reset_index()
    )
    table = (
        spot_summary
        .merge(up_summary, on="year", how="left")
        .merge(down_summary, on="year", how="left")
        .merge(wind_summary, on="year", how="inner")
    )
    table = table.sort_values("year")
    return table, save_table(table, "appendix_weather_year_summary_statistics")


def save_weather_year_summary_plot(weather_year_summary):
    years = weather_year_summary["year"].astype(str)
    x = range(len(weather_year_summary))

    fig, axes = plt.subplots(2, 1, figsize=(11.0, 7.0), sharex=True)
    axes[0].plot(
        x,
        weather_year_summary["mean_spot"],
        marker="o",
        linewidth=2,
        color=PALETTES[COLOR_PALETTE][0],
        label="Mean",
    )
    axes[0].fill_between(
        list(x),
        weather_year_summary["p05_spot"],
        weather_year_summary["p95_spot"],
        color=PALETTES[COLOR_PALETTE][0],
        alpha=0.18,
        label="P5-P95",
    )
    axes[0].set_ylabel("Spot price [EUR/MWh]")
    axes[0].set_title("Spot-price variation by weather year")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(
        x,
        weather_year_summary["mean_wind_cf"],
        marker="o",
        linewidth=2,
        color=PALETTES[COLOR_PALETTE][1],
        label="Mean",
    )
    axes[1].fill_between(
        list(x),
        weather_year_summary["p05_wind_cf"],
        weather_year_summary["p95_wind_cf"],
        color=PALETTES[COLOR_PALETTE][1],
        alpha=0.18,
        label="P5-P95",
    )
    axes[1].set_ylabel("Wind capacity factor [-]")
    axes[1].set_title("Wind-realization variation by weather year")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(years, rotation=45, ha="right")
    axes[1].set_xlabel("Weather year")
    fig.tight_layout()

    output_path = GRAPHS_PATH / "appendix_weather_year_price_wind_summary.png"
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def main():
    apply_style()
    GRAPHS_PATH.mkdir(parents=True, exist_ok=True)

    prices = load_price_data()
    states = load_regulation_state_data()
    wind = load_wind_data()
    weather_year_summary, weather_year_outputs = save_weather_year_summary(prices, wind)

    outputs = [
        save_price_summary_table(prices),
        save_regulation_frequency_table(prices, states),
        save_wind_summary_table(wind),
        save_monthly_wind_profile_table(wind),
        save_diurnal_price_plot(prices),
        save_diurnal_price_distribution_plot(prices),
        save_regulation_frequency_plot(states),
        save_wind_distribution_plot(wind),
        save_monthly_wind_boxplot(wind),
        weather_year_outputs,
        save_weather_year_summary_plot(weather_year_summary),
    ]
    flattened_outputs = []
    for output in outputs:
        if output is None:
            continue
        if isinstance(output, tuple):
            flattened_outputs.extend(output)
        else:
            flattened_outputs.append(output)

    print(f"Created {len(flattened_outputs)} input-data statistic output(s):")
    for output in flattened_outputs:
        print(output)


if __name__ == "__main__":
    main()
