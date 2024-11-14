# %%
import glob
import os
import time

# basic libraries
import numpy as np
from numpy import newaxis as na
import pandas as pd
import xarray as xr
import yaml
import scipy
import importlib
import openmdao.api as om

from sklearn.neighbors import NearestNeighbors
from statsmodels.distributions.empirical_distribution import ECDF, monotone_fn_inverter

if not importlib.util.find_spec("finitediff"):
    from hydesign.utils import get_weights
else:
    from finitediff import get_weights



class ABL(om.ExplicitComponent):
    """Atmospheric boundary layer WS interpolation and gradient
    
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

    def setup_partials(self):
        self.declare_partials('*', '*')

    def precompute(self, hh):

        weather = pd.read_csv(self.weather_fn, index_col=0, parse_dates=True)
        ds_interpolated = interpolate_WS_loglog(weather, hh=hh)
        return ds_interpolated

    def compute(self, inputs, outputs):

        ds_interpolated = self.precompute(inputs['hh'])
        self.ds_interpolated = ds_interpolated

        outputs['wst'] = np.nan_to_num(ds_interpolated.WS.values.flatten())

    def compute_partials(self, inputs, partials):

        ds_interpolated = self.ds_interpolated

        partials['wst', 'hh'] = ds_interpolated.dWS_dz.values.flatten()


# -----------------------------------------------------------------------
# Auxiliar functions for weather handling
# -----------------------------------------------------------------------

def interpolate_WS_loglog(weather, hh):
    """
    Auxiliar functions for WS, shear and WS gradient interpolation.

    Parameters
    ----------
    weather: pd.DataFrame
        WS time-series table for a location at multiple heights.
        The columns must be named WS_hh (for example  WS_1, WS_10, WS_50).
    
    hh: float
        Elevation (of a wind turbine) to interpolate WS, shear and dWS_dz

    Returns
    --------
    ds_interpolated: xr.Dataset
        Dataset that contains the interpolated time-series: WS, shear and dWS_dz

    """
    ws_vars = [var for var in weather.columns if 'WS_' in var]
    heights = np.array([float(var.split('_')[-1]) for var in ws_vars])
    weather[ws_vars] = weather[ws_vars].clip(lower=1e-6)  # to avoid log from throwing error if wind speed is zero

    ds_all = xr.Dataset(
        data_vars = {
            'log_WS': (['time','log_height'], np.log(weather[ws_vars].values)),
            'height': (['log_height'], heights),
            },
        coords = {
            'time': weather.index.values,
            'log_height': np.log(heights),
            }  
        )

    ds_all['shear'] = ds_all.differentiate("log_height").log_WS

    ds_interpolated = xr.Dataset()
    ds_interpolated['shear'] = ds_all.shear.interp(
        log_height=np.log(hh), method='nearest')
    ds_interpolated['height'] = hh
    ds_interpolated['WS'] = np.exp(ds_all.interp(log_height=np.log(hh)).log_WS)
    ds_interpolated['dWS_dz'] = (
        ds_interpolated['shear']/ds_interpolated['height'])*ds_interpolated['WS']
    return ds_interpolated


# -----------------------------------------------------------------
# Auxiliar functions for extractions on a ERA5 database
# -----------------------------------------------------------------


# pvlib imports
from pvlib import pvsystem, tools, irradiance, atmosphere
from pvlib.location import Location        

def extract_weather_for_HPP(
    longitude, latitude, altitude,
    era5_zarr = '/groups/reanalyses/era5/app/era5.zarr',
    #ratio_gwa_era5 = '/groups/INP/era5/ratio_gwa3_era5.nc',
    ratio_gwa_era5 = '/groups/INP/era5/ratio_gwa2_era5.nc',
    era5_ghi_zarr = '/groups/INP/era5/ghi.zarr',
    year_start = '1990',
    year_end = '1990',
    ):
    """
    Extracting weather data using the era5 datasets, by specifying the location coordinates.

    Parameters
    ----------
    longitude: location longitude
    latitude: location latitude
    altitude: location altitude
    year_start: first year of the lifetime
    year_end: last year of the last time

    Returns
    --------
    weather: weather time series including wind speed, wind direction, temperature, ghi, dni, dhi

    """
    
    # Extract ERA5
    ds = xr.open_zarr(era5_zarr,consolidated=False)
    ds = ds.sel(time=slice(year_start, year_end))
    
    lon = longitude
    lat = latitude
    heights = np.array([1, 50, 100, 150, 200])
    px=np.array([lon] * len(heights))
    py=np.array([lat] * len(heights))

    # weights for interpolation
    weights_ds = get_interpolation_weights(
        px=px,
        py=py,
        pz=heights,
        all_x=ds['longitude'].values,
        all_y=ds['latitude'].values,
        all_z=ds['height'].values,
        n_stencil=2,
        locs_ID=range(len(heights))
    )

    # Apply interpolation
    ds_interp = apply_interpolation_f(
        wrf_ds=ds,
        weights_ds=weights_ds,
        vars_xy_logz=['WS'],
        vars_xyz=['WD'],
        vars_xy=['T2'],
        vars_nearest_xy=[],
        vars_nearest_xyz=[],
        var_x_grid='longitude',
        var_y_grid='latitude',
        var_z_grid='height',
        varWD='WD')

    # Extract GWA scaling
    ratio_gwa_era5_ds = xr.open_dataset(ratio_gwa_era5, engine='h5netcdf')
    var_ratio = list(ratio_gwa_era5_ds.data_vars)[0]

    ratio_gwa_era5_ds_A = ratio_gwa_era5_ds.isel(height=0)
    ratio_gwa_era5_ds_A['height'] = 1e-6

    ratio_gwa_era5_ds_B = ratio_gwa_era5_ds.isel(height=-1)
    ratio_gwa_era5_ds_B['height'] = 500
    
    ratio_gwa_era5_ds = xr.concat(
        [ratio_gwa_era5_ds_A,
         ratio_gwa_era5_ds,
         ratio_gwa_era5_ds_B],
        dim = 'height')
    
    # weights for interpolation
    weights_ds_era5 = get_interpolation_weights(
        px=px,
        py=py,
        pz=heights,
        all_x=ratio_gwa_era5_ds['longitude'].values,
        all_y=ratio_gwa_era5_ds['latitude'].values,
        all_z=ratio_gwa_era5_ds['height'].values,
        n_stencil=2,
        locs_ID=range(len(heights))
    )

    # Apply interpolation
    ws_scale = apply_interpolation_f(
        wrf_ds=ratio_gwa_era5_ds,
        weights_ds=weights_ds_era5,
        vars_xy_logz=[],
        vars_xyz=[var_ratio],
        vars_xy=[],
        vars_nearest_xy=[],
        vars_nearest_xyz=[],
        var_x_grid='longitude',
        var_y_grid='latitude',
        var_z_grid='height',
        varWD='WD')

    # apply scaling
    ds_interp['WS'] = ds_interp['WS']*ws_scale[var_ratio]

    # swap coordinate to height
    ds_interp['height'] = xr.DataArray(
                    dims=['locs_ID'],
                    data=heights
                )
    ds_interp = ds_interp.swap_dims({"locs_ID": "height"})

    # Extract GHI
    ds_ghi = xr.open_zarr(era5_ghi_zarr,consolidated=False)
    ds_ghi['height'] = [1]
    ds_ghi = ds_ghi.sel(time=slice(year_start, year_end))
    
    px = np.array([[np.mod(lon, 360), lat]])

    # Apply interpolation for GHI
    ds_ghi_site = apply_interpolation_IDW(
        ds_ghi, px, var='ghi')
    ds_ghi_site = ds_ghi_site.sel(
        locs_ID=0
    ).drop('locs_ID')

    # build weather dataset
    weather = pd.DataFrame( index = ds_interp.time.values )
    for var in ['WS', 'WD']:
        for h in ds_interp.height.values:
            hh = f'{int(h)}'
            weather[f'{var}_{hh}'] = ds_interp.sel(height=h)[var].values
            
    weather['temp_air_1'] = ds_interp.sel(
                    height=1.).T2.drop(['height', 'locs_ID']).values
    
    weather['ghi'] = ds_ghi_site.ghi.values
    
    pvloc = Location(
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            name='Plant')

    times = weather.index
    pressure = atmosphere.alt2pres(altitude)
    solpos = pvloc.get_solarposition(times, pressure)

    disc_out = irradiance.disc(
        ghi=weather['ghi'],
        solar_zenith=solpos.zenith,
        datetime_or_doy=times,
        pressure=pressure,
        min_cos_zenith=0.065,
        max_zenith=87,
        max_airmass=12)

    weather['dni'] = disc_out['dni']
    weather['dhi'] = weather.ghi - \
        weather.dni * tools.cosd(solpos.zenith)

    return weather

def isoprob_transfrom(y_input, y_desired): 
    """
    Method to perform a 1D isoprobabilistic trasnformation in order to
    force the input sample to be have the desired distribution.
    Correlations are kept in rank-sense.
    """
    ecdf_desired = ECDF(y_desired)
    ecdf_input = ECDF(y_input)
    ecdf_input_y = ecdf_input(y_input)
    
    y = scipy.interpolate.interp1d(
        ecdf_desired.y, ecdf_desired.x)(ecdf_input_y)
    
    return y

def select_years(
    df,
    seed=0,
    weeks_per_season_per_year=1,
):
    """
    Method to select a number of weeks per season per year from a time series df
    To each variable a isoprobabilistic trasnformation is applied in order to
    force the sample distribution to be the same as the long term distribution.
    Correlations are kept in rank-sense.
    """
    df_inter = df.copy()
    columns = df_inter.columns
    df_inter['year'] = df_inter.index.year
    df_inter['week'] = df_inter.index.isocalendar().week
    month_to_season = {
        1: 'DJF',
        2: 'DJF',
        3: 'MAM',
        4: 'MAM',
        5: 'MAM',
        6: 'JJA',
        7: 'JJA',
        8: 'JJA',
        9: 'SON',
        10: 'SON',
        11: 'SON',
        12: 'DJF'}
    df_inter['season'] = df_inter.index.month.map(month_to_season)

    # to ensure only full weeks are sampled
    week_sel_table = df_inter.groupby(
        ['year', 'week', 'season']).count().iloc[:, :1].reset_index()
    week_sel_table.columns = ['year', 'week', 'season', 'tmsp']
    week_sel_table = week_sel_table.loc[week_sel_table.tmsp == 168, :]

    df_inter_sel = []
    years = np.random.RandomState(
        seed=seed).permutation(
        df_inter.index.year.unique())
    for iy, year in enumerate(years):
        i_week = 0
        for season in df_inter.season.unique():
            df_inter_yr_ss = df_inter.loc[
                (df_inter.year == year) & (df_inter.season == season), :].copy()
            df_inter_yr_ss['i_year'] = iy
            seed += 1
            week = np.random.RandomState(seed).permutation(
                week_sel_table.loc[
                    (week_sel_table.year == year) & (week_sel_table.season == season),
                    'week'].unique())[:weeks_per_season_per_year]
            aux = df_inter_yr_ss.loc[df_inter_yr_ss.week.isin(week), :].copy()
            for week in aux.week.unique():
                aux['i_week'] = i_week
                i_week += 1
            df_inter_sel += [aux]

    df_inter_sel = pd.concat(df_inter_sel)

    df_inter_sel['i_life'] = df_inter_sel.i_year + \
        df_inter_sel.i_week / (4 * weeks_per_season_per_year)

    df_out = df_inter_sel
    for var in columns:
        y = isoprob_transfrom(y_input=df_inter_sel[var].values, y_desired=df_inter[var].values)
        df_out[var] = y

    return df_out

def get_interpolation_weights(
    px, py, pz, all_x, all_y, all_z, n_stencil=4, locs_ID=[],
):
    """
    Function that creates the 3D interpolation weights using finite
    differences for multiple interpolations points (px,py,pz), given a grid of
    observed points [all_x, all_y, all_z].

    This function computes the weights for interpolation for different order
    in the horizontal dimensions (x,y), while it computes the weights for both
    linear interpolation and for piecewise logarithmic profile in z.

    Parameters
    ----------
    px: numpy.array
        Interpolation (prediction) points in x
    py: numpy.array
        Interpolation (prediction) points in y
    pz: numpy.array
        Interpolation (prediction) points in z
    all_x: numpy.array
        Observed points in x
    all_y: numpy.array
        Observed points in y
    all_z: numpy.array
        Observed points in z
    n_stencil: int, optional, default=4
        Number of points used in the horizontal interpolation
    locs_ID: list
        Names or ID to identify the locations
    """
    # Number of prediction points and observed points
    Np = len(px)
    Nx = len(all_x)
    Ny = len(all_y)
    Nz = len(all_z)

    if (Np != len(py)) or (Np != len(pz)):
        raise Exception("The len of px, py and pz should be the same")

    # get stencils for interpolations
    n_st_x = n_stencil
    n_st_y = n_stencil
    n_st_z = 2  # In z, interpolation is always based on two points
    if n_stencil > Nx:
        n_st_x = Nx
    if n_stencil > Ny:
        n_st_y = Ny
    if 2 > Nz:
        n_st_z = Nz
    nnx = NearestNeighbors(n_neighbors=n_st_x).fit(all_x[:, na])
    nny = NearestNeighbors(n_neighbors=n_st_y).fit(all_y[:, na])
    nnz = NearestNeighbors(n_neighbors=n_st_z).fit(all_z[:, na])

    # Find the indexes of the observed points to be used for interpolation
    ind_x = np.sort(nnx.kneighbors(px[:, na], return_distance=False), axis=1)
    ind_y = np.sort(nny.kneighbors(py[:, na], return_distance=False), axis=1)
    ind_z = np.sort(nnz.kneighbors(pz[:, na], return_distance=False), axis=1)

    # Find the index for nearest point selection
    nnx_1 = NearestNeighbors(n_neighbors=1).fit(all_x[:, na])
    nny_1 = NearestNeighbors(n_neighbors=1).fit(all_y[:, na])
    nnz_1 = NearestNeighbors(n_neighbors=1).fit(all_z[:, na])
    ind_x_1 = nnx_1.kneighbors(px[:, na], return_distance=False)
    ind_y_1 = nny_1.kneighbors(py[:, na], return_distance=False)
    ind_z_1 = nnz_1.kneighbors(pz[:, na], return_distance=False)

    # Allocate weight matrices
    # Horizontal interpolation weights have the size of the stencil
    weights_x = np.zeros([Np, n_st_x])
    weights_y = np.zeros([Np, n_st_y])
    # Vertical extrapolation weights are always the same size: all available
    # heights
    weights_log_z = np.zeros([Np, Nz])
    weights_z = np.zeros([Np, Nz])
    for i in range(Np):
        weights_x[i, :] = get_weights(
            grid=all_x[ind_x[i, :]],
            xtgt=px[i],
            maxorder=0)[:, 0]

        weights_y[i, :] = get_weights(
            grid=all_y[ind_y[i, :]],
            xtgt=py[i],
            maxorder=0)[:, 0]

        weights_log_z[i, ind_z[i, :]] = get_weights(
            grid=np.log(all_z[ind_z[i, :]]),
            xtgt=np.log(pz[i]),
            maxorder=0)[:, 0]

        weights_z[i, ind_z[i, :]] = get_weights(
            grid=all_z[ind_z[i, :]],
            xtgt=pz[i],
            maxorder=0)[:, 0]

    if len(locs_ID) == 0:
        locs_ID = np.arange(Np, dtype=int)

    # Build dataset
    weights_ds = xr.Dataset(
        data_vars={
            'weights_x': (
                ['locs_ID', 'ix'],
                weights_x,
                {'description': 'Interpolation weights based on finite differences'}),
            'ind_x': (
                ['locs_ID', 'ix'],
                ind_x,
                {'description': 'Indices of WRF grid to use in the interpolation'}),
            'weights_y': (
                ['locs_ID', 'iy'],
                weights_y,
                {'description': 'Interpolation weights based on finite differences'}),
            'ind_y': (
                ['locs_ID', 'iy'],
                ind_y,
                {'description': 'Indices of WRF grid to use in the interpolation'}),
            'weights_z': (
                ['locs_ID', 'iz'],
                weights_z,
                {'description': 'Interpolation weights based on finite differences'}),
            'weights_log_z': (
                ['locs_ID', 'iz'],
                weights_log_z,
                {'description': 'Logaritmic interpolation weights based on finite differences'}),
            'ind_z': (
                ['locs_ID', 'iz'],
                np.repeat(np.arange(len(all_z))[na, :], Np, axis=0),
                {'description': 'Indices of WRF grid to use in the interpolation'}),
            'ind_x_1': (
                ['locs_ID'],
                ind_x_1.flatten(),
                {'description': 'Indices of WRF grid to use in nearest point selection'}),
            'ind_y_1': (
                ['locs_ID'],
                ind_y_1.flatten(),
                {'description': 'Indices of WRF grid to use in nearest point selection'}),
            'ind_z_1': (
                ['locs_ID'],
                ind_z_1.flatten(),
                {'description': 'Indices of WRF grid to use in nearest point selection'}), },
        coords={'locs_ID': locs_ID})

    return weights_ds

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
        interp[var] = (wrf_ds.get(var).isel({
            var_x_grid: weights_ds.ind_x,
            var_y_grid: weights_ds.ind_y,
            var_z_grid: weights_ds.ind_z,
        })
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
        interp[varWD] = np.mod(interp[varWD], 360)

    return interp

def apply_interpolation_IDW(ds_dssr, px, var='ghi', n_neighbors=4, IDW_p=2):
    """
    Function that interpolates as dataset using inverse distance weighting (IDW).
    Usefull for unstructured datasets or with nan's on several locations.

    Parameters
    ----------
    ds_dssr: Dataset
        Dataset with latitude and longitude as coordinates

    px: array
        points to interpolate in [lon,lat]

    var: str
        Variable to interpolate available in ds_dssr

    n_neighbors: int
        Number of neares neighbor points to use in the interpolation

    IDW_p: float
        IDW exponent coefficient to compute the weitghs = 1/d**IDW_p
    """

    ds_dssr['ilat'] = xr.DataArray(
        data=np.arange(len(ds_dssr.latitude)),
        dims={'latitude': ds_dssr.latitude})
    ds_dssr['ilon'] = xr.DataArray(
        data=np.arange(len(ds_dssr.longitude)),
        dims={'longitude': ds_dssr.longitude})

    locs_with_values = ds_dssr[['longitude', 'latitude', 'ilon', 'ilat', var]].isel(
        time=[0, 12, 24, -25, -13, -1]).sum('time').to_dataframe().reset_index()
    locs_with_values.columns = ['longitude', 'latitude', 'ilon', 'ilat', var]
    locs_with_values = locs_with_values.loc[locs_with_values[var] > 0, [
        'longitude', 'latitude', 'ilon', 'ilat', var]]

    nx = NearestNeighbors(n_neighbors=n_neighbors).fit(
        locs_with_values.loc[:, ['longitude', 'latitude']].values)
    dx, ix = nx.kneighbors(px, return_distance=True)

    # add a small distance to avoid dividing by 0
    weights = 1 / (dx + 1e-12)**IDW_p
    nnp = ix.shape[0]
    nnx = ix.shape[1]

    xr_weights = xr.Dataset(
        data_vars={
            'weights': (
                ['locs_ID', 'ix'],
                weights,
                {'description': 'Interpolation weights, IDW'}),
        },
        coords={'locs_ID': np.arange(nnp)})

    # Select the nearest points in ds_dssr
    xr_sel = xr.concat([
        xr.concat(
            [ds_dssr[var].isel(
                longitude=locs_with_values.ilon.values[ixi],
                latitude=locs_with_values.ilat.values[ixi])
                for ixi in ix[ip, :]], dim='ix')
        for ip in range(nnp)], dim='locs_ID')

    interp_IDW = xr.Dataset()
    interp_IDW[var] = (xr_weights.weights * xr_sel).sum('ix') / \
        xr_weights.weights.sum('ix')

    return interp_IDW


def project_locations(
        locs,
        region_domain_fn,
        ds,
        domain,
        var_lon='longitude',
        var_lat='latitude',
):
    """
    Function that uses the wrf projection properties for
    converting the locations (lat,lon) to regular grid coordinates used in wrf.

    Parameters
    ----------
    locs: Dataframe
        Table with locations including Latitude and Longitude

    region_domain_fn: str
        Filename of region_domain_fn excel file. To describe wich domain to use per
        region (or country).

    ds: Dataset
        Meso-scale xarray dataset including Lat, Lon and wrf projection

    domains: str
        Domain name
    """
    locs = locs.copy()

    if domain is None:
        locs_in_domain = locs
        wrf_proj = read_projections_zarr(ds, domain='')
    else:
        regdom = pd.read_excel(region_domain_fn)

        if "WRF_Domain" not in locs.columns:
            # Add a column to identify on which domain each locs belongs
            if 'Country' in locs.columns:
                locs["WRF_Domain"] = [
                    regdom.loc[regdom.Country == country, "WRF_Domain"].tolist()[0]
                    for country in locs.Country.values]
            else:
                locs["WRF_Domain"] = domain

        # extract available domains from the WRF files
        wrf_proj = read_projections_zarr(ds, domain,
                                         var_lon=var_lon, var_lat=var_lat)

        locs_in_domain = locs.loc[locs["WRF_Domain"] == domain, :]

    # Get wrf projection coordinates of the locations
    ds_x_y = wrf.ll_to_xy_proj(
        latitude=locs_in_domain.Latitude.values,
        longitude=locs_in_domain.Longitude.values,
        as_int=False,
        **wrf_proj,
    )
    locs_x = ds_x_y.values[0, :]
    locs_y = ds_x_y.values[1, :]
    locs_z = locs_in_domain.Hub_height.values

    locs_in_domain.loc[:, 'x'] = locs_x
    locs_in_domain.loc[:, 'y'] = locs_y
    return locs_in_domain


