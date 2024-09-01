
import argparse

import numpy as np
import pandas as pd
import xarray as xr

from hydesign.assembly.hpp_assembly import hpp_model
from hydesign.Parallel_EGO import EfficientGlobalOptimizationDriver
from hydesign.examples import examples_filepath

def fillin_prices_missing_days(pr_df):
    
    for t in list(pd.date_range(
            '2012-01-01 00:00','2012-01-01 23:00', freq='1h')):
        prices.loc[t] =  prices.loc[pd.to_datetime(t+pd.Timedelta('1d'))]
    
    for t in list(pd.date_range(
            '2012-02-29 00:00','2012-02-29 23:00', freq='1h')
        ) + list(pd.date_range(
            '2012-12-31 00:00','2012-12-31 23:00', freq='1h')):
        pr_df.loc[t] =  pr_df.loc[pd.to_datetime(t-pd.Timedelta('1d'))]
    
    return pr_df

if __name__ == '__main__':

    parser=argparse.ArgumentParser()
    parser.add_argument('--ID', default=0, 
                        help='ID (index) to run HPP sizing, based on grid_points_sample.csv')    
    parser.add_argument('--ID_start', default=0)
    args=parser.parse_args()
    
    ID_start = int(args.ID_start)
    iloc = int(args.ID)+ID_start

    # parameters
    year = 2025
    realize_nr = 1


    #sites  = pd.read_csv('grid_points_sample.csv')
    sites  = pd.read_csv('grid_points_onshore.csv')

    sites['name'] = sites.index.values
    site_sel = sites.loc[iloc,:]
    
    region = site_sel.Region
    name = site_sel.name
    longitude = site_sel.Longitude
    latitude = site_sel.Latitude
    
    # To store the input_ts (waether and prices)
    #input_ts_fn_out = f'input_ts_{name}.csv'
    
    ds = xr.open_zarr(
        '/groups/INP/REALISE/parsed_simulations.zarr')

    prices = ds.sel(
        year=year,
        region=region, 
        realize_nr=realize_nr
        )['Price_electricity'].to_dataframe()['Price_electricity']
    
    ds.close()
    
    prices = prices.reindex(
        pd.date_range('2012-01-01 00:00','2012-12-31 23:00', freq='1h'),
        fill_value=np.NaN)  
    prices = fillin_prices_missing_days(prices)

    inputs = {
        'example': None,
        'name': name,
        'longitude': longitude,
        'latitude': latitude,
        'altitude': None,
        'input_ts_fn': None,
        'sim_pars_fn': 'hpp_pars.yml',
        'price_fn': prices,
        #
        'opt_var': "NPV_over_CAPEX",
        'num_batteries': 5,
        'n_procs': 32,
        'n_doe': 160,
        'n_clusters': 8, # total number of evals per iteration = n_clusters + 2*n_dims
        'n_seed': 1,
        'max_iter': 15,
        'final_design_fn': f'final_design_{name}.csv',
        'npred': 3e4,
        'tol': 1e-6,
        'min_conv_iter': 3,
        'work_dir': './',
        #
        'hpp_model': hpp_model,
    'variables' : {
        'clearance [m]':
            {'var_type':'fixed',
             'value':10,
             },
         'sp [W/m2]':
            {'var_type':'fixed',
             'value':200,
             },
        'p_rated [MW]':
            {'var_type':'fixed',
             'value':1,
             },
        'Nwt':
            {'var_type':'fixed',
             'value':0,
             },
        'wind_MW_per_km2 [MW/km2]':
            {'var_type':'fixed',
             'value':5,
             },
        'solar_MW [MW]':
            {'var_type':'design',
             'limits':[0, 400],
             'types':'int'
             },
        'surface_tilt [deg]':
            {'var_type':'design',
             'limits':[0, 50],
             'types':'resolution',
             'resolution':0.5,
             },
        'surface_azimuth [deg]':
            {'var_type':'design',
             'limits':[90, 270],
             'types':'resolution',
             'resolution':0.5,
             },
        'DC_AC_ratio':
            {'var_type':'design',
             'limits':[0.8, 2.0],
             #'types':'float'
             'types':'resolution',
             'resolution':0.2,
             },
        'b_P [MW]':
            {'var_type':'design',
             'limits':[0, 100],
             'types':'int'
             },
        'b_E_h [h]':
            {'var_type':'design',
             'limits':[1, 4],
             'types':'int'
             },
        'cost_of_battery_P_fluct_in_peak_price_ratio':
            {'var_type':'design',
             'limits':[0, 20],
             #'types':'float'
             'types':'resolution',
             'resolution':0.5,
             },
    }  }
    EGOD = EfficientGlobalOptimizationDriver(**inputs)
    EGOD.run()
    result = EGOD.result
    
    
