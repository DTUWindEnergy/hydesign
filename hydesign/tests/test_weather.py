# -*- coding: utf-8 -*-
"""
Created on 24/01/2023

@author: jumu
"""
import numpy as np
import pandas as pd
import pytest
import pickle

from hydesign.tests.test_files import tfp
from hydesign.examples import examples_filepath
from hydesign.weather.weather import interpolate_WS_loglog

# ------------------------------------------------------------------------------------------------
def run_interp_ws():
    hh = 100
    weather = pd.read_csv(examples_filepath+'Europe/GWA2/input_ts_Denmark_good_solar.csv',index_col=0)
    interp_ws_out = interpolate_WS_loglog(weather,hh)
    df_out = pd.DataFrame()
    df_out['WS'] = interp_ws_out.WS.values
    df_out['dWS_dz'] = interp_ws_out.dWS_dz.values
    return df_out

def load_interp_ws():
    output_df = pd.read_csv(
        tfp+'weather_output_interp_ws.csv',
        index_col=0,
        parse_dates=False)
    return output_df

def test_interp_ws():
    interp_ws_out = run_interp_ws()
    interp_ws_out_data = load_interp_ws()
    for var in ['WS', 'dWS_dz']:
        np.testing.assert_allclose(
            interp_ws_out[var].values, interp_ws_out_data[var].values)

# ------------------------------------------------------------------------------------------------
def update_interp_ws():
    df = run_interp_ws()
    df.to_csv(tfp+'weather_output_interp_ws.csv')  
    
# ------------------------------------------------------------------------------------------------
# update_interp_ws()
