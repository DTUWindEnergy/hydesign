# -*- coding: utf-8 -*-
"""
Created on 24/01/2023

@author: jumu
"""
import numpy as np
import pandas as pd
import os

import pytest
from pvlib.location import Location

from hydesign.examples import examples_filepath
from hydesign.tests.test_files import tfp
from hydesign.pv.pv import get_solar_time_series, get_linear_solar_degradation

# ------------------------------------------------------------------------------------------------
def run_solar_time_series(tracking):
    
    examples_sites = pd.read_csv(f'{examples_filepath}examples_sites.csv', index_col=0, sep=';')
    name = 'France_good_wind'
    ex_site = examples_sites.loc[examples_sites.name == name]

    longitude = ex_site['longitude'].values[0]
    latitude = ex_site['latitude'].values[0]
    altitude = ex_site['altitude'].values[0]
    
    pvloc = Location(
        latitude=latitude,
        longitude=longitude,
        altitude=altitude,
        name='Plant')
    
    input_ts_fn = examples_filepath+'Europe/GWA2/input_ts_France_good_wind.csv'
    weather = pd.read_csv(
        input_ts_fn, 
        index_col=0,
        parse_dates=True)
    
    weather['temp_air'] = weather['temp_air_1'] - 273.15  # Celcius
    weather['wind_speed'] = weather['WS_1']
    
    surface_tilt = 35
    surface_azimuth = 180
    solar_MW = 150
    land_use_per_solar_MW = 0.01226
    DC_AC_ratio = 1.5
    
    solar_t = get_solar_time_series(
            surface_tilt = surface_tilt, 
            surface_azimuth = surface_azimuth, 
            solar_MW = solar_MW, 
            land_use_per_solar_MW = land_use_per_solar_MW, 
            DC_AC_ratio = DC_AC_ratio, 
            tracking = tracking, 
            pvloc = pvloc, 
            weather = weather)
    
    return solar_t

def load_solar_time_series(tracking):
    output_df = pd.read_csv(
        tfp+'pv_generation_output.csv', 
        index_col=0, 
        parse_dates = False)
    return output_df[tracking]

@pytest.mark.parametrize('tracking', ['no', 'single_axis'])
def test_solar_time_series(tracking):
    pv_ts = run_solar_time_series(tracking).values
    pv_ts_data = np.squeeze(load_solar_time_series(tracking).values)
    np.testing.assert_allclose(pv_ts, pv_ts_data)
    #print(np.allclose(pv_ts, pv_ts_data))

# ------------------------------------------------------------------------------------------------
def run_solar_degradation():
    pv_deg_per_year = 0.05
    life_h = 25*365*24
    return get_linear_solar_degradation(pv_deg_per_year, life_h)

def load_solar_degradation():
    output_df = pd.read_csv(
        tfp+'pv_degradation_output.csv', 
        index_col=0, 
        parse_dates = False)
    return output_df.pv_deg.values

def test_solar_degradation():
    pv_deg_ts = run_solar_degradation()
    pv_deg_ts_data = load_solar_degradation()
    np.testing.assert_allclose(pv_deg_ts, pv_deg_ts_data)
    #print(np.allclose(pv_deg_ts, pv_deg_ts_data))


# ------------------------------------------------------------------------------------------------
def update_solar_tests():
    pv_ts = pd.DataFrame()
    for tracking in ['no', 'single_axis']:
        pv_ts[tracking] = run_solar_time_series(tracking)
    pv_ts.to_csv(tfp+'pv_generation_output.csv')  
    
# ------------------------------------------------------------------------------------------------
# update_solar_tests()