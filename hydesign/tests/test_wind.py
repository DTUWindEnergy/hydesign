# -*- coding: utf-8 -*-
"""
Created on 24/01/2023

@author: jumu
"""
import numpy as np
import pandas as pd
import pytest
import pickle


from hydesign.examples import examples_filepath
from hydesign.tests.test_files import tfp
from hydesign.look_up_tables import lut_filepath
from hydesign.wind.wind import get_WT_curves, get_wake_affected_pc, get_wind_ts

# ------------------------------------------------------------------------------------------------
def run_get_WT_curves():
    genWT_fn = lut_filepath+'genWT_v3.nc'
    specific_power = 320
    return get_WT_curves(genWT_fn, specific_power)

def load_get_WT_curves():
    output_df = pd.read_csv(
        tfp+'wind_get_WT_curves_output.csv',
        index_col=0, 
        # parse_dates = True
        )
    return output_df.ws.values, output_df.pc.values, output_df.ct.values

def test_get_WT_curves():
    WT_curves = run_get_WT_curves()
    WT_curves_data = load_get_WT_curves()
    for i in range(len(WT_curves_data)):
        np.testing.assert_allclose(WT_curves[i], WT_curves_data[i])
        #print(np.allclose(WT_curves[i], WT_curves_data[i]))
        
# ------------------------------------------------------------------------------------------------
def run_get_wake_affected_pc():
    genWake_fn = lut_filepath+'genWake_v3.nc'
    specific_power = 320
    Nwt = 101
    wind_MW_per_km2 = 7.5
    p_rated = 2 #MW
    
    genWT_fn = lut_filepath+'genWT_v3.nc'
    ws, pc, ct = get_WT_curves(genWT_fn, specific_power)
    
    return ws, get_wake_affected_pc(
        genWake_fn, 
        specific_power,
        Nwt,
        wind_MW_per_km2,
        ws,
        pc,
        p_rated
    )

def load_get_wake_affected_pc():
    output_df = pd.read_csv(
        tfp+'wind_get_wake_affected_pc_output.csv',
        index_col=0, 
        # parse_dates = True
        )
    return output_df.ws.values, output_df.pcw.values

def test_get_wake_affected_pc():
    wake_affected_pc = run_get_wake_affected_pc()
    wake_affected_pc_data = load_get_wake_affected_pc()
    np.testing.assert_allclose(wake_affected_pc, wake_affected_pc_data)
    #print(np.allclose(wake_affected_pc, wake_affected_pc_data))

# ------------------------------------------------------------------------------------------------
def run_get_wind_ts():
    
    examples_sites = pd.read_csv(f'{examples_filepath}examples_sites.csv', index_col=0, sep=';')
    name = 'France_good_wind'
    ex_site = examples_sites.loc[examples_sites.name == name]

    input_ts_fn = examples_filepath+'Europe/GWA2/input_ts_France_good_wind.csv'
    weather = pd.read_csv(
        input_ts_fn, 
        index_col=0,
        parse_dates=True)
    
    wst = weather.WS_150.values
    
    pcw_df = pd.read_csv(
        tfp+'wind_get_wake_affected_pc_output.csv',
        index_col=0, 
        parse_dates = False)
    ws = pcw_df['ws'].values
    pcw = pcw_df['pcw'].values
    
    wpp_efficiency = 0.95
    
    return get_wind_ts(
        ws,
        pcw,
        wst,
        wpp_efficiency
    )

def load_get_wind_ts():
    output_df = pd.read_csv(
        tfp+'wind_get_wind_ts_output.csv',
        index_col=0, 
        parse_dates = False)
    return output_df.wind_ts.values

def test_get_wind_ts():
    wind_ts = run_get_wind_ts()
    wind_ts_data = load_get_wind_ts()
    np.testing.assert_allclose(wind_ts, wind_ts_data)
    #print(np.allclose(wind_ts, wind_ts_data))
    

# ------------------------------------------------------------------------------------------------
def update_wind_ts_tests():
    df = pd.DataFrame()
    df['wind_ts'] = run_get_wind_ts()
    df.to_csv(tfp+'wind_get_wind_ts_output.csv')  
    
# ------------------------------------------------------------------------------------------------
# update_wind_ts_tests()