# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 11:16:36 2024

@author: mikf
"""

import time
import numpy as np
import pandas as pd
import os
import xarray as xr

from hydesign.assembly.hpp_assembly import hpp_model
from hydesign.examples import examples_filepath
from hydesign.utils import sample_mean

class ReliabilityModel(hpp_model):
    def __init__(self, 
                 reliability_hpp_model,        
                 sim_pars_fn,
                 reliability_data_set_path=os.path.join(examples_filepath, 'reliability'),
                 battery_ds_fn='reliability_data_set_BESS.nc',
                 transformer_ds_fn='reliability_data_set_transformer.nc',
                 inverter_ds_fn='reliability_data_set_inverter.nc',
                 PV_ds_fn='reliability_data_set_PV.nc',
                 WT_ds_fn='reliability_data_set_WT.nc',
                 n_reliability_seed=5,
                 aggregation_method=sample_mean,
                 **kwargs
                 ):
        self.hpp = reliability_hpp_model
        self.sim_pars_fn = sim_pars_fn
        self.reliability_data_set_path=reliability_data_set_path
        self.battery_ds_fn = battery_ds_fn
        self.transformer_ds_fn = transformer_ds_fn
        self.inverter_ds_fn = inverter_ds_fn
        self.PV_ds_fn = PV_ds_fn
        self.WT_ds_fn = WT_ds_fn
        self.n_seed = n_reliability_seed
        self.aggregation_method=aggregation_method
        self.kwargs = kwargs
        hpp_model.__init__(self,
                           sim_pars_fn=sim_pars_fn,
                           **kwargs
                           )
        self.list_out_vars += ['Inverter size nominal [kW]',
                               'Panel size nominal [W]',
                               'Number of inverters [-]',
                               'Inverter size real [kW]',
                               'Number of panels per inverter [-]',
                               'Panel size real [W]',]
        self.list_vars += ['inverter_size [kW]', 'panel_size [W]']
        
    def evaluate(self,
                 # Wind plant design
                 clearance, sp, wt_rated_power_MW, Nwt, wind_MW_per_km2,
                 # PV plant design
                 solar_MW,  surface_tilt_deg, surface_azimuth_deg, DC_AC_ratio,
                 # Energy storage & EMS price constrains
                 b_P, b_E_h, cost_of_batt_degr,
                 # Reliability inputs
                 inverter_size, panel_size
                 ):
        outs =[]
        n_inverters = np.max((int(solar_MW * 10 ** 3 / inverter_size), 1)) #  ensure that there is at least 1 inverter
        inverter_size_real = solar_MW * 10 ** 3 / n_inverters
        n_panels_per_inverter = np.max((int(inverter_size_real * 10 ** 3 / panel_size), 1)) #  ensure that there is at least 1 inverter
        panel_size_real = inverter_size_real * 10 ** 3 / n_panels_per_inverter
        Nwt = int(Nwt)
        for i in range(self.n_seed):
            with xr.open_dataset(os.path.join(self.reliability_data_set_path,self.battery_ds_fn), engine='h5netcdf') as ds_batt:
                reliability_ts_battery = reliability_dataset_to_timeseries(ds_batt.sel(seed=i)).squeeze()
            with xr.open_dataset(os.path.join(self.reliability_data_set_path, self.transformer_ds_fn), engine='h5netcdf') as ds_tran:
                reliability_ts_trans = reliability_dataset_to_timeseries(ds_tran.sel(seed=i)).squeeze()
            with xr.open_dataset(os.path.join(self.reliability_data_set_path, self.WT_ds_fn), engine='h5netcdf') as ds_wind:
                reliability_ts_wind = reliability_dataset_to_timeseries(ds_wind.sel(seed=i))[:,:Nwt].mean((1)).squeeze()
            with xr.open_dataset(os.path.join(self.reliability_data_set_path, self.PV_ds_fn), engine='h5netcdf') as ds_pv, xr.open_dataset(os.path.join(self.reliability_data_set_path, self.inverter_ds_fn), engine='h5netcdf') as ds_inv:
                reliability_ts_pv = np.hstack([reliability_dataset_to_timeseries(ds_pv.sel(seed=i).sel(batch_no=batch)) for batch in ds_pv.batch_no.values])[:,:n_panels_per_inverter].mean((1)).squeeze()
                reliability_ts_inverter = reliability_dataset_to_timeseries(ds_inv.sel(seed=i))[:,:n_inverters].mean((1)).squeeze()
                reliability_ts_pv = reliability_ts_pv * reliability_ts_inverter
            x = [# Wind plant design
                 clearance, sp, wt_rated_power_MW, Nwt, wind_MW_per_km2,
                 # PV plant design
                 solar_MW,  surface_tilt_deg, surface_azimuth_deg, DC_AC_ratio,
                 # Energy storage & EMS price constrains
                 b_P, b_E_h, cost_of_batt_degr]
    
            out = self.hpp(sim_pars_fn = self.sim_pars_fn,
                           reliability_ts_battery=reliability_ts_battery,
                           reliability_ts_trans=reliability_ts_trans,
                           reliability_ts_wind=reliability_ts_wind,
                           reliability_ts_pv=reliability_ts_pv,
                           **self.kwargs
                           ).evaluate(*x)
            out = np.hstack([out, [inverter_size,
                                   panel_size,
                                   n_inverters,
                                   inverter_size_real,
                                   n_panels_per_inverter,
                                   panel_size_real,]])
            outs.append(out)
        return self.aggregation_method(outs)



def reliability_dataset_to_timeseries(ds):
    ts_start = str(ds.ts_start.values)
    ts_end = str(ds.ts_end.values)
    ts_freq = str(ds.ts_freq.values)
    N_components = int(ds.N_components )
    N_sample_needed = int(ds.N_sample_needed)
    i_FT=ds.TTF_indices.values
    i_BO=ds.TTR_indices.values
    
    ts_indices = pd.date_range(ts_start, ts_end, freq=ts_freq)
    N_ts = len(ts_indices)   
    i_OFF = np.ones(shape=(N_ts,N_components))                              # Evaluate the downtime and record it in i_OFF
    for i_t in range(N_components):
        for i_f in range(N_sample_needed):
            i_OFF[i_FT[i_f,i_t]:i_BO[i_f,i_t],i_t]=0                        # Set the availability as 0 while the plant is down, for other timeperiods set the availability to 1
    return i_OFF.astype(bool)


if __name__ == '__main__':
    name = 'France_good_wind'
    examples_sites = pd.read_csv(f'{examples_filepath}examples_sites.csv', index_col=0, sep=';')
    ex_site = examples_sites.loc[examples_sites.name == name]

    longitude = ex_site['longitude'].values[0]
    latitude = ex_site['latitude'].values[0]
    altitude = ex_site['altitude'].values[0]

    sim_pars_fn = examples_filepath+ex_site['sim_pars_fn'].values[0]
    input_ts_fn = examples_filepath+ex_site['input_ts_fn'].values[0]

    reliability_ts_battery=pd.read_csv(os.path.join(examples_filepath, 'reliability', 'battery_seed_0.csv'), index_col=0, sep=',').values.squeeze()
    reliability_ts_trans=pd.read_csv(os.path.join(examples_filepath, 'reliability', 'transformer_seed_0.csv'), index_col=0, sep=',').values.squeeze()
    reliability_ts_wind=pd.read_csv(os.path.join(examples_filepath, 'reliability', 'wt_seed_0_50_turbines.csv'), index_col=0, sep=',').values.squeeze()
    reliability_ts_pv=pd.read_csv(os.path.join(examples_filepath, 'reliability', 'pv_seed_0_100_inverters_of_1MW_500W_panels.csv'), index_col=0, sep=',').values.squeeze()

    n_reliability_seed = 2
    
    wt_rated_power_MW = 4
    surface_tilt_deg = 35
    surface_azimuth_deg = 180
    DC_AC_ratio = 1.5
    Nwt = 50
    wind_MW_per_km2 = 7
    solar_MW = 100
    b_P = 20
    b_E_h  = 3
    cost_of_batt_degr = 5
    clearance = 20
    sp = 350
    D = np.sqrt(4 * wt_rated_power_MW * 10 ** 6 / np.pi / sp)
    hh = clearance + D / 2
    
    inverter_size = 1000 #  [kW]
    panel_size = 500 # [W]
    x = [# Wind plant design
        clearance, sp, wt_rated_power_MW, Nwt, wind_MW_per_km2,
        # PV plant design
        solar_MW,  surface_tilt_deg, surface_azimuth_deg, DC_AC_ratio,
        # Energy storage & EMS price constrains
        b_P, b_E_h, cost_of_batt_degr,
        # Reliability inputs
        inverter_size, panel_size,
        ]

    # x=[41.99999999999999, 350.0, 6.0, 200.0, 7.0, 200.0, 25.0, 154.5050654737706, 1.0, 50.0, 6.0, 10.0, 1165.3385616121257, 183.41182263434803]
    RM = ReliabilityModel(hpp_model,
                          latitude=latitude,
                          longitude=longitude,
                          altitude=altitude,
                          num_batteries = 10,
                          sim_pars_fn = sim_pars_fn,
                          input_ts_fn = input_ts_fn,
                          n_reliability_seed=n_reliability_seed,
                          )
    
    start = time.time()
    outs = RM.evaluate(*x)
    end = time.time()
    RM.print_design(x, outs)
    print('exec. time [min]:', (end - start)/60 )
    

