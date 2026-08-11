# -*- coding: utf-8 -*-
"""
Created on 24/01/2023

@author: jumu
"""
import pickle

import numpy as np
import pandas as pd
import pytest

from hydesign.examples import examples_filepath
from hydesign.tests.test_files import tfp
from hydesign.weather.weather import (
    ABL,
    interpolate_WD,
    interpolate_WS_loglog,
    isoprob_transfrom,
)


# ------------------------------------------------------------------------------------------------
def run_interp_ws():
    hh = 100
    weather = pd.read_csv(
        examples_filepath + "Europe/GWA2/input_ts_Denmark_good_solar.csv", index_col=0
    )
    interp_ws_out = interpolate_WS_loglog(weather, hh)
    df_out = pd.DataFrame()
    df_out["WS"] = interp_ws_out.WS.values
    df_out["dWS_dz"] = interp_ws_out.dWS_dz.values
    return df_out


def load_interp_ws():
    output_df = pd.read_csv(
        tfp + "weather_output_interp_ws.csv", index_col=0, parse_dates=False
    )
    return output_df


def test_interp_ws():
    interp_ws_out = run_interp_ws()
    interp_ws_out_data = load_interp_ws()
    for var in ["WS", "dWS_dz"]:
        np.testing.assert_allclose(
            interp_ws_out[var].values, interp_ws_out_data[var].values
        )


def test_interp_wind_direction():
    weather = pd.DataFrame(
        {
            "WD_10": [350.0, 0.0],
            "WD_100": [10.0, 90.0],
        }
    )

    np.testing.assert_allclose(interpolate_WD(weather, hh=55), [0.0, 45.0])


def test_abl_returns_wind_speed_and_direction(tmp_path):
    weather = pd.DataFrame(
        {
            "WS_10": [2.0, 4.0],
            "WS_100": [20.0, 40.0],
            "WD_10": [350.0, 0.0],
            "WD_100": [10.0, 90.0],
        },
        index=pd.date_range("2030-01-01", periods=2, freq="h"),
    )
    weather_file = tmp_path / "weather.csv"
    weather.to_csv(weather_file)

    model = ABL(weather_file, N_time=2, interpolate_wd=True)
    wind_speed, wind_direction = model.compute(hh=55)

    assert wind_speed.shape == (2,)
    assert wind_direction.shape == (2,)
    assert model.compute_partials().shape == (2,)


def test_isoprobabilistic_transform():
    transformed = isoprob_transfrom([0, 1, 2], [10, 20, 30])

    np.testing.assert_allclose(transformed, [10, 20, 30])


# ------------------------------------------------------------------------------------------------
def update_interp_ws():
    df = run_interp_ws()
    df.to_csv(tfp + "weather_output_interp_ws.csv")


# ------------------------------------------------------------------------------------------------
# update_interp_ws()
