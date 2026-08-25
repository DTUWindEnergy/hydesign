from pathlib import Path
import json
import os

import numpy as np
import pandas as pd

from sdp import make_ppa_profile


BTC_OUTPUTS_DIR = Path(
    os.environ.get(
        "BTC_OUTPUTS_DIR",
        "/groups/INP/to_angelos/from_angelos/outputs",
    )
)

EXPECTED_HOURS = 8_736
EXPECTED_FIRST_MONTH_DAY_HOUR = (1, 2, 0)


def _validate_hourly_data(df, time_col, source_name, expected_year=None):
    df = df.sort_values(time_col).reset_index(drop=True)

    if len(df) != EXPECTED_HOURS:
        raise ValueError(
            f"{source_name} must contain exactly {EXPECTED_HOURS} hourly rows; "
            f"found {len(df)}."
        )

    if df[time_col].duplicated().any():
        duplicates = df.loc[df[time_col].duplicated(keep=False), time_col].head(5)
        raise ValueError(
            f"Duplicate timestamps in {source_name}. "
            f"Examples: {duplicates.astype(str).tolist()}"
        )

    first_time = df.at[0, time_col]
    first_month_day_hour = (first_time.month, first_time.day, first_time.hour)
    if first_month_day_hour != EXPECTED_FIRST_MONTH_DAY_HOUR:
        raise ValueError(
            f"{source_name} must start at 2 January 00:00; found {first_time}."
        )
    if expected_year is not None and first_time.year != expected_year:
        raise ValueError(
            f"{source_name} should contain weather year {expected_year}; "
            f"its first timestamp is {first_time}."
        )

    expected_times = pd.date_range(
        start=first_time,
        periods=EXPECTED_HOURS,
        freq="h",
    )
    actual_times = pd.DatetimeIndex(df[time_col])
    if not actual_times.equals(expected_times):
        mismatch = np.flatnonzero(actual_times != expected_times)[0]
        raise ValueError(
            f"Missing or irregular hourly timestamps in {source_name}. "
            f"At row {mismatch}, expected {expected_times[mismatch]} but found "
            f"{actual_times[mismatch]}."
        )

    df["hour_index"] = np.arange(EXPECTED_HOURS)
    return df



def _read_price_data(annual_hourly, area, bm_price_filename):
    price_da_file = annual_hourly / "PriceDA_ts_hourly.csv"
    bm_price_file = annual_hourly / bm_price_filename
    imb_hourly_file = annual_hourly / "Imb_hourly.csv"

    for fp in [price_da_file, bm_price_file, imb_hourly_file]:
        if not fp.exists():
            raise FileNotFoundError(f"Missing file: {fp}")

    da = pd.read_csv(price_da_file, low_memory=False)
    bm = pd.read_csv(bm_price_file, low_memory=False)
    imb = pd.read_csv(imb_hourly_file, low_memory=False)

    da = da.loc[:, ~da.columns.astype(str).str.startswith("Unnamed:")]
    bm = bm.loc[:, ~bm.columns.astype(str).str.startswith("Unnamed:")]
    imb = imb.loc[:, ~imb.columns.astype(str).str.startswith("Unnamed:")]

    if area not in da.columns:
        raise KeyError(f"Column '{area}' not found in {price_da_file.name}")
    if f"{area}_up" not in bm.columns or f"{area}_down" not in bm.columns:
        raise KeyError(f"Columns '{area}_up'/'{area}_down' not found in {bm_price_file.name}")
    if f"{area}_imb" not in imb.columns:
        raise KeyError(f"Column '{area}_imb' not found in {imb_hourly_file.name}")

    da_time_col = "time" if "time" in da.columns else "time_step"
    bm_time_col = "time_step" if "time_step" in bm.columns else "time"
    imb_time_col = "time" if "time" in imb.columns else "time_step"

    da[da_time_col] = pd.to_datetime(da[da_time_col])
    bm[bm_time_col] = pd.to_datetime(bm[bm_time_col])
    imb[imb_time_col] = pd.to_datetime(imb[imb_time_col])

    da = da[[da_time_col, area]].rename(columns={da_time_col: "time", area: "spot"})
    bm = bm[[bm_time_col, f"{area}_up", f"{area}_down"]].rename(
        columns={bm_time_col: "time", f"{area}_up": "up", f"{area}_down": "down"}
    )
    imb = imb[[imb_time_col, f"{area}_imb"]].rename(
        columns={imb_time_col: "time", f"{area}_imb": "imb"}
    )

    price_df = da.merge(bm, on="time", how="inner", validate="one_to_one")
    price_df = price_df.merge(
        imb,
        on="time",
        how="inner",
        validate="one_to_one",
    )
    return _validate_hourly_data(
        price_df,
        time_col="time",
        source_name=str(annual_hourly),
    )


def _read_wind_profile(wind_file, wind_col, output_col, weather_year):
    if not wind_file.exists():
        raise FileNotFoundError(f"Missing file: {wind_file}")

    wind = pd.read_csv(wind_file, low_memory=False)
    wind = wind.loc[:, ~wind.columns.astype(str).str.startswith("Unnamed:")]

    if "time" not in wind.columns or wind_col not in wind.columns:
        raise KeyError(f"Expected columns 'time' and '{wind_col}' in {wind_file.name}")

    wind["time"] = pd.to_datetime(wind["time"])
    wind = wind[["time", wind_col]].rename(columns={wind_col: output_col})
    wind = _validate_hourly_data(
        wind,
        time_col="time",
        source_name=str(wind_file),
        expected_year=weather_year,
    )
    return wind[["hour_index", output_col]]


def _build_year_dataframe(
    inputs_dir,
    weather_year,
    omega,
    area,
    wind_col,
    xi_wind_files,
    bm_price_filename,
    stage2_wind_method,
):
    annual_hourly = BTC_OUTPUTS_DIR / f"Model_results_{weather_year}" / "annual_hourly"
    price_df = _read_price_data(annual_hourly, area, bm_price_filename)

    corres_dir = inputs_dir / "CORRES" / f"corres_{weather_year}"
    df_year = price_df.copy()
    wind_xi_cols = []

    for xi, xi_wind_file in enumerate(xi_wind_files, start=1):
        wind_col_name = f"wind_xi_{xi}"
        wind_xi_cols.append(wind_col_name)
        wind_profile = _read_wind_profile(
            corres_dir / xi_wind_file,
            wind_col,
            wind_col_name,
            weather_year,
        )
        df_year = df_year.merge(
            wind_profile,
            on="hour_index",
            how="inner",
            validate="one_to_one",
        )

    if len(df_year) != EXPECTED_HOURS:
        raise ValueError(
            f"Merged data for weather year {weather_year} contains "
            f"{len(df_year)} rows instead of {EXPECTED_HOURS}."
        )

    if stage2_wind_method == "mean":
        df_year["wind_stage2"] = df_year[wind_xi_cols].mean(axis=1)
    elif stage2_wind_method == "min":
        df_year["wind_stage2"] = df_year[wind_xi_cols].min(axis=1)
    else:
        raise ValueError("stage2_wind_method must be either 'mean' or 'min'")

    df_year["omega"] = omega
    df_year["weather_year"] = weather_year
    return df_year.sort_values("hour_index").reset_index(drop=True)


def _capacity_factor_to_mw(value, p_nom): # because CORRES gices us capacity factors but we want MW in the model
    return float(np.clip(value, 0, 1) * p_nom)


PRICE_TOLERANCE = 0.0
DOWN_REGULATION_BID_PRICE = 0.0

def _classify_regulation(row, price_tolerance=PRICE_TOLERANCE):
    spot = float(row["spot"])
    up = float(row["up"])
    down = float(row["down"])
    imb_val = float(row["imb"]) if "imb" in row else 0.0

    up_active = up > spot + price_tolerance
    down_active = down < spot - price_tolerance

    if up_active and not down_active:
        return 1, 0, up
    if down_active and not up_active:
        return 0, 1, down
    if not up_active and not down_active:
        return 0, 0, spot

    # Both price signals indicate regulation. Use the dominant imbalance direction.
    if imb_val > 0:
        return 1, 0, up
    return 0, 1, down


def prepare_inputs(
    delivery_profile_name,
    weather_years,
    xi_wind_files,
    stage2_wind_method="mean",
):
    area = "DK1"
    wind_col = "Belwind_Pilot"
    bm_price_filename = "BM_price_hourly.csv"
    inputs_dir = Path(__file__).resolve().parent
    output_dir = inputs_dir / "Input_files_created"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not weather_years:
        raise ValueError("weather_years must contain at least one year.")
    if not xi_wind_files:
        raise ValueError("xi_wind_files must contain at least one wind profile file.")

    Omega = list(range(1, len(weather_years) + 1))
    Xi = list(range(1, len(xi_wind_files) + 1))

    year_dfs = []
    for omega, weather_year in zip(Omega, weather_years):
        year_dfs.append(
            _build_year_dataframe(
                inputs_dir=inputs_dir,
                weather_year=weather_year,
                omega=omega,
                area=area,
                wind_col=wind_col,
                xi_wind_files=xi_wind_files,
                bm_price_filename=bm_price_filename,
                stage2_wind_method=stage2_wind_method,
            )
        )

    prepared_years = []
    for df_year in year_dfs:
        df_year = df_year.copy()
        df_year["t"] = np.arange(1, EXPECTED_HOURS + 1)
        prepared_years.append(df_year)

    df = pd.concat(prepared_years, ignore_index=True)
    T_values = list(range(1, EXPECTED_HOURS + 1))
    n_days = EXPECTED_HOURS // 24

    print(f"Using {n_days} full days = {EXPECTED_HOURS} hours per weather year")
    print(f"Weather years: {weather_years}")
    print(f"Xi wind files: {xi_wind_files}")
    print(f"Stage-2 wind method: {stage2_wind_method}")

    config = {
        "solver": "cplex",
        #"solver": "appsi_highs",
        #"solver": "gurobi",
        # main.py overwrites lambda_PPA for each sensitivity run.
        "lambda_PPA": 60.0,
        "c_imb": 0.134,
        "c_ppa": 160.0,
        "P_nom": 37.8,
        "P_b_max": 20.0,
        "P_grid": 52.0,
        "eta_dis": 0.985,
        "eta_cha": 0.975,
        "E_max": 60.0,
        "SoC_min": 0.05,
        "SoC_max": 0.95,
        "E0": 30.0,
        "alpha": 0.90,
        "beta": 0.20,
        "degradation_cost_per_mw": 5.85,
        "first_hour_each_day": [1 + 24 * d for d in range(n_days)],
        
    }

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    ppa_profile = make_ppa_profile(T_values, delivery_profile_name)
    ppa_profile.to_csv(output_dir / "ppa_profile.csv", index=False)

    pd.DataFrame(
        {
            "omega": Omega,
            "weather_year": weather_years,
            "prob": [1 / len(Omega)] * len(Omega),
        }
    ).to_csv(output_dir / "omega_probabilities.csv", index=False)

    xi_rows = []
    for w in Omega:
        for x, xi_wind_file in zip(Xi, xi_wind_files):
            xi_rows.append(
                {
                    "omega": w,
                    "xi": x,
                    "wind_file": xi_wind_file,
                    "prob": 1 / len(Xi),
                }
            )
    pd.DataFrame(xi_rows).to_csv(output_dir / "xi_probabilities.csv", index=False)

    spot_rows = []
    up_rows, down_rows, imb_rows = [], [], []
    activation_rows = []
    wf_rows, wr_rows = [], []

    for _, row in df.iterrows():
        t = int(row["t"])
        w = int(row["omega"])

        spot_rows.append({"t": t, "omega": w, "price": float(row["spot"])})
        activation_limit = abs(float(row["imb"]))
        activation_rows.append(
            {
                "t": t,
                "omega": w,
                "A_up_max": activation_limit,
                "A_down_max": activation_limit,
            }
        )

        system_y_up, system_y_down, imb_price = _classify_regulation(row)
        up = float(row["up"])
        down = float(row["down"])

        # The system state determines the imbalance price, but this HPP offers
        # downward regulation only when its clearing price is non-positive.
        y_up = system_y_up
        y_down = int(
            system_y_down == 1
            and down <= DOWN_REGULATION_BID_PRICE
        )

        for x in Xi:
            up_rows.append(
                {
                    "t": t,
                    "omega": w,
                    "xi": x,
                    "price": up,
                    "y_up": y_up,
                    "y_down": y_down,
                }
            )
            down_rows.append(
                {
                    "t": t,
                    "omega": w,
                    "xi": x,
                    "price": down,
                    "y_up": y_up,
                    "y_down": y_down,
                }
            )
            imb_rows.append({"t": t, "omega": w, "xi": x, "price": imb_price})

            wr_rows.append(
                {
                    "t": t,
                    "omega": w,
                    "xi": x,
                    "wind": _capacity_factor_to_mw(row[f"wind_xi_{x}"], config["P_nom"]),
                }
            )

        wf_rows.append(
            {
                "t": t,
                "omega": w,
                "wind": _capacity_factor_to_mw(row["wind_stage2"], config["P_nom"]),
            }
        )

    pd.DataFrame(spot_rows).to_csv(output_dir / "spot_prices.csv", index=False)
    pd.DataFrame(up_rows).to_csv(output_dir / "mfrr_up_prices.csv", index=False)
    pd.DataFrame(down_rows).to_csv(output_dir / "mfrr_down_prices.csv", index=False)
    pd.DataFrame(imb_rows).to_csv(output_dir / "imbalance_prices.csv", index=False)
    pd.DataFrame(activation_rows).to_csv(output_dir / "activation_limits.csv", index=False)
    pd.DataFrame(wf_rows).to_csv(output_dir / "wind_forecast.csv", index=False)
    pd.DataFrame(wr_rows).to_csv(output_dir / "wind_realization.csv", index=False)

    scenario_metadata = {
        "delivery_profile": delivery_profile_name,
        "weather_years": weather_years,
        "omega_definition": "weather year",
        "xi_definition": "wind profile scenario",
        "xi_wind_files": xi_wind_files,
        "stage2_wind_method": stage2_wind_method,
        "stage2_wind_units": "MW",
        "wind_realization_units": "MW",
        "source_wind_units": "capacity factor",
        "hours_per_weather_year": EXPECTED_HOURS,
        "time_alignment": "BTC and CorRES matched by sequential hourly position",
        "activation_limit_source": f"absolute value of {area}_imb from Imb_hourly.csv",
        "price_tolerance": PRICE_TOLERANCE,
        "down_regulation_bid_price": DOWN_REGULATION_BID_PRICE,
        "down_regulation_eligibility": (
            "system down-regulation and downward clearing price <= 0 EUR/MWh"
        ),
    }
    with open(output_dir / "input_scenario_metadata.json", "w", encoding="utf-8") as f:
        json.dump(scenario_metadata, f, indent=2)

    print(f"Prepared inputs in: {output_dir}")
    print(f"Delivery profile: {delivery_profile_name}")
    print(f"Hours used per omega: {EXPECTED_HOURS}")
    print(f"Omega scenarios: {len(Omega)}")
    print(f"Xi scenarios per omega: {len(Xi)}")


if __name__ == "__main__":
    prepare_inputs(
        delivery_profile_name="Monthly Continuous Baseload",
        weather_years=[2000],
        xi_wind_files=["P_forecast1.csv"],
        stage2_wind_method="mean",
    )
