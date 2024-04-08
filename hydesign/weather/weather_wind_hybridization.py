# %%
# import glob
# import os
# import time

# basic libraries
import numpy as np
# from numpy import newaxis as na
import pandas as pd
import xarray as xr
# import yaml
# import scipy
# import importlib
import openmdao.api as om

# from sklearn.neighbors import NearestNeighbors
# from statsmodels.distributions.empirical_distribution import ECDF, monotone_fn_inverter

from hydesign.weather.weather import interpolate_WS_loglog

# if not importlib.util.find_spec("finitediff"):
#     from hydesign.utils import get_weights
# else:
#     from finitediff import get_weights


class ABL_WD(om.ExplicitComponent):
    """Atmospheric boundary layer WS and WD interpolation
    
    Parameters
    ----------
    hh : Turbine's hub height

    Returns
    -------
    wst : wind speed time series at the hub height

    """


    def __init__(self, weather_fn, N_time):
        super().__init__()
        self.weather_fn = weather_fn
        self.N_time = N_time

    def setup(self):
        self.add_input('hh',
                       desc="Turbine's hub height",
                       units='m')
        self.add_output('wst',
                        desc="ws time series at the hub height",
                        units='m/s',
                        shape=[self.N_time])
        self.add_output('wdt',
                        desc="wd time series at the hub height",
                        units='deg',
                        shape=[self.N_time])


    def compute(self, inputs, outputs):

        hh = inputs['hh']
        weather = pd.read_csv(self.weather_fn, index_col=0, parse_dates=True)
        ds_interpolated = interpolate_WS_loglog(weather, hh=hh)
        ds_interpolated_aux = interpolate_WD_linear(weather, hh=hh)
        ds_interpolated['WD'] = ds_interpolated_aux['WD']
        
        self.ds_interpolated = ds_interpolated

        outputs['wst'] = ds_interpolated.WS.values.flatten()
        outputs['wdt'] = ds_interpolated.WD.values.flatten()

# -----------------------------------------------------------------------
# Auxiliar functions for weather handling
# -----------------------------------------------------------------------

cosd = lambda x: np.cos(np.radians(x))
sind = lambda x: np.sin(np.radians(x))

def interpolate_WD_linear(weather, hh):
    """
    Auxiliar functions for WD interpolation.

    Parameters
    ----------
    weather: pd.DataFrame
        WD time-series table for a location at multiple heights.
        The columns must be named WS_hh (for example  WD_10, WD_50, ...).
    
    hh: float
        Elevation (of a wind turbine) to interpolate WD

    Returns
    --------
    ds_interpolated: xr.Dataset
        Dataset that contains the interpolated time-series: WD and dWD_dz

    """
    wd_vars = [var for var in weather.columns if 'WD_' in var]
    heights = np.array([float(var.split('_')[-1]) for var in wd_vars])


    ds_all = xr.Dataset(
        data_vars = {
            'WD': (['time','height'], weather[wd_vars].values),
            },
        coords = {
            'time': weather.index.values,
            'height': heights,
            }  
        )

    ds_all['WD_x'] = cosd(ds_all.WD)
    ds_all['WD_y'] = sind(ds_all.WD)
    
    ds_interpolated = xr.Dataset()
    ds_interpolated['WD_x'] = ds_all.WD_x.interp(height=hh, method='linear')
    ds_interpolated['WD_y'] = ds_all.WD_y.interp(height=hh, method='linear')
    
    ds_interpolated['WD'] = np.mod( np.degrees(np.arctan2(ds_interpolated['WD_y'],ds_interpolated['WD_x']) ) + 360, 360)
    
    return ds_interpolated

# -----------------------------------------------------------------
# Auxiliar functions for extractions on a ERA5 database
# -----------------------------------------------------------------

# pvlib imports
# from pvlib import pvsystem, tools, irradiance, atmosphere
# from pvlib.location import Location        

def apply_interpolation_f(
    wrf_ds,
    weights_ds,
    vars_xy_logz=["WSPD"],
    vars_xyz=["WDIR", "RHO"],
    vars_xy=["UST", "RMOL", "TAIR", "DIF_AVG", "DNI_AVG"],
    vars_nearest_xy=[],
    vars_nearest_xyz=[],
    var_x_grid='west_east',
    var_y_grid='south_north',
    var_z_grid='height',
    varWD='WDIR',
):
    """
    Function that applies interpolation to a wrf simulation.

    Parameters
    ----------
    wrf_ds: xarray.Dataset
        Weather timeseries
    weights_ds: xarray.Dataset
        Weights for locs interpolation for several methods::

            <xarray.Dataset>
            Dimensions:        (ix: 4, iy: 4, iz: 5, loc: 14962)
            Coordinates:
            * loc            (loc) int64
            Dimensions without coordinates: ix, iy, iz
            Data variables:
                weights_x      (loc, ix) float64
                ind_x          (loc, ix) int64
                weights_y      (loc, iy) float64
                ind_y          (loc, iy) int64
                weights_z      (loc, iz) float64
                weights_log_z  (loc, iz) float64
                ind_z          (loc, iz) int64
                ind_x_1        (loc)     int64
                ind_y_1        (loc)     int64
                ind_z_1        (loc)     int64

    vars_xy_logz: list
        List of variables to be interpolated in horizontal (x,y) using finite
        differences and power law piecewise interpolation in z.
    vars_xyz: list
        List of variables to be interpolated in horizontal (x,y) using finite
        differences and linear piecewise interpolation in z.
    vars_xy: list
        List of variables to be interpolated in horizontal (x,y) using finite
        differences
    vars_nearest_xy: list
        List of variables to be approximated to the nearest horizontal point (x,y)
    vars_nearest_xyz: list
        List of variables to be approximated to the nearest point (x,y,z)
    var_x_grid: string, default:'west_east'
        Name of the variable in the weather data used as x in the interpolation
    var_y_grid: string, default: 'south_north'
        Name of the variable in the weather data used as y in the interpolation
    var_z_grid: string, default:'height'
        Name of the variable in the weather data used as z in the interpolation
    varWD: string, default:'wd'
        Name of the wind direction variable for ensuring it is in [0,360]

    Returns
    --------
    interp: xarray.Dataset
        Dataset including meso-variables timeseries, interpolated at each locs.
        The arrays have two dimensions: ('Time', 'locs').

    """

    interp = xr.Dataset()

    # power law profile in z
    for var in vars_xy_logz:
        if var not in ['WSPD', 'WS', 'ws', 'wspd']:
            interp[var] = (wrf_ds.get(var).isel({
                var_x_grid: weights_ds.ind_x,
                var_y_grid: weights_ds.ind_y,
                var_z_grid: weights_ds.ind_z,
            })
                * weights_ds.weights_x
                * weights_ds.weights_y
                * weights_ds.weights_log_z).sum(['ix', 'iy', 'iz'])
        else:
            interp[var] = np.exp((np.log(wrf_ds.get(var) + 1e-12).isel({
                var_x_grid: weights_ds.ind_x,
                var_y_grid: weights_ds.ind_y,
                var_z_grid: weights_ds.ind_z,
            })
                * weights_ds.weights_x
                * weights_ds.weights_y
                * weights_ds.weights_log_z).sum(['ix', 'iy', 'iz']))

    # linear profile in z
    for var in vars_xyz:
        if var != varWD:
            interp[var] = (wrf_ds.get(var).isel({
                var_x_grid: weights_ds.ind_x,
                var_y_grid: weights_ds.ind_y,
                var_z_grid: weights_ds.ind_z,
            })
                * weights_ds.weights_x
                * weights_ds.weights_y
                * weights_ds.weights_z).sum(['ix', 'iy', 'iz'])
        else:
            # Apply interpolation in WD using vector linear interpolation 
            interp[varWD+'_x'] = (cosd(wrf_ds.get(var).isel({
                var_x_grid: weights_ds.ind_x,
                var_y_grid: weights_ds.ind_y,
                var_z_grid: weights_ds.ind_z,
            }))
                * weights_ds.weights_x
                * weights_ds.weights_y
                * weights_ds.weights_z).sum(['ix', 'iy', 'iz'])

            interp[varWD+'_y'] = (sind(wrf_ds.get(var).isel({
                var_x_grid: weights_ds.ind_x,
                var_y_grid: weights_ds.ind_y,
                var_z_grid: weights_ds.ind_z,
            }))
                * weights_ds.weights_x
                * weights_ds.weights_y
                * weights_ds.weights_z).sum(['ix', 'iy', 'iz'])

    # only horizontal interpolation
    for var in vars_xy:
        interp[var] = (wrf_ds.get(var).isel({
            var_x_grid: weights_ds.ind_x,
            var_y_grid: weights_ds.ind_y,
        })
            * weights_ds.weights_x
            * weights_ds.weights_y).sum(['ix', 'iy'])

    # nearest horizontal point approximation
    for var in vars_nearest_xy:
        interp[var] = wrf_ds.get(var).isel({
            var_x_grid: weights_ds.ind_x_1,
            var_y_grid: weights_ds.ind_y_1,
        })

    # nearest point approximation
    for var in vars_nearest_xyz:
        interp[var] = wrf_ds.get(var).isel({
            var_x_grid: weights_ds.ind_x_1,
            var_y_grid: weights_ds.ind_y_1,
            var_z_grid: weights_ds.ind_z_1,
        })

    if varWD in vars_xy_logz + vars_xyz + \
            vars_xy + vars_nearest_xy + vars_nearest_xyz:
        interp[varWD] = np.mod( np.degrees(np.arctan2(interp[varWD+'_y'],interp[varWD+'_x'])) +360, 360)

    return interp

